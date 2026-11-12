"""Auditable exact per-example DP-SGD via torch.func, logical Poisson batches.

Microbatches only bound memory: clip each example jointly, sum ALL microbatches,
add noise ONCE, divide by public reference expected batch size. Empty batches
still receive a Gaussian step. Momentum/AMP/BatchNorm are deliberately absent.
"""
import secrets
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

def generator(device):
    gen = torch.Generator(device=device)
    gen.manual_seed(secrets.randbits(63))
    return gen

def sample_gradients(module,x,y):
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    def loss(p,b,xi,yi):
        logits = functional_call(module,(p,b),(xi.unsqueeze(0),))
        # y=-1 represents an absent record in a fixed public capacity slot.
        return F.cross_entropy(logits,yi.clamp_min(0).unsqueeze(0),reduction='sum')*(yi>=0)
    return vmap(grad(loss),in_dims=(None,None,0,0))(params,buffers,x,y)

def clipped_sum(module,x,y,clip,microbatch):
    sums = {n:torch.zeros_like(p) for n,p in module.named_parameters()}
    for start in range(0,len(y),microbatch):
        grads = sample_gradients(module,x[start:start+microbatch],y[start:start+microbatch])
        norms = torch.stack([g.flatten(1).square().sum(1) for g in grads.values()]).sum(0).sqrt()
        scales = (clip/norms.clamp_min(1e-12)).clamp(max=1)
        for n,g in grads.items():
            shape = (len(scales),)+(1,)*(g.ndim-1)
            sums[n].add_((g*scales.reshape(shape)).sum(0).detach())
    return sums

def step(module,x,y,*,q,sigma,clip,lr,microbatch,rng,private=True,anchor=None,lam=0):
    expected_batch = q*len(y)  # fixed public slot/reference capacity, not sampled size
    ids = (torch.rand(len(y),device=y.device,generator=rng)<q).nonzero().flatten()
    if private:
        sums = clipped_sum(module,x[ids],y[ids],clip,microbatch)
    else:
        sums = {n:torch.zeros_like(p) for n,p in module.named_parameters()}
        if len(ids):
            loss = F.cross_entropy(module(x[ids]),y[ids],reduction='sum')
            gradients = torch.autograd.grad(loss,tuple(module.parameters()))
            sums = {n:g.detach() for (n,_),g in zip(module.named_parameters(),gradients)}
    with torch.no_grad():
        for name,p in module.named_parameters():
            g = sums[name]
            if private:
                g = g + torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=rng)*(sigma*clip)
            g = g/expected_batch
            # Conditional on DP history, this is data-independent. Outside clipping.
            if anchor is not None and lam:
                g = g + lam*(p-anchor[name])
            p.add_(g,alpha=-lr)
    return len(ids)

@torch.no_grad()
def accuracy(model,x,y,device,batch=512):
    model.eval(); correct=0
    for i in range(0,len(y),batch):
        correct += int((model(x[i:i+batch].to(device)).argmax(1).cpu()==y[i:i+batch].cpu()).sum())
    return correct/len(y)

def cpu_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
