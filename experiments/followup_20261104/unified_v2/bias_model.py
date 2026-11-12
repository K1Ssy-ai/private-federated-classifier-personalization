import torch
from torch import nn
from torch.nn import functional as F
class AllBias(nn.Module):
 def __init__(self,state):
  super().__init__()
  for i,key in enumerate(['body.0.weight','body.3.weight','body.8.weight','head.weight']):
   self.register_buffer(f'w{i}',state[key].detach().clone())
  for i,key in enumerate(['body.0.bias','body.3.bias','body.8.bias','head.bias']):
   setattr(self,f'b{i}',nn.Parameter(state[key].detach().clone()))
 def forward(self,x):
  x=F.avg_pool2d(torch.tanh(F.conv2d(x,self.w0,self.b0,padding=1)),2)
  x=F.avg_pool2d(torch.tanh(F.conv2d(x,self.w1,self.b1,padding=1)),2)
  x=F.adaptive_avg_pool2d(x,(4,4)).flatten(1)
  return F.linear(torch.tanh(F.linear(x,self.w2,self.b2)),self.w3,self.b3)

def verify():
 from torch.func import functional_call,grad
 for channels in [1,3]:
  m=nn.Module();m.body=nn.Sequential(nn.Conv2d(channels,16,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.Tanh());m.head=nn.Linear(64,10)
  b=AllBias(m.state_dict());x=torch.randn(4,channels,32 if channels==3 else 28,32 if channels==3 else 28);y=torch.tensor([0,1,2,3])
  assert sum(p.numel() for p in b.parameters())==122
  assert torch.allclose(b(x),m.head(m.body(x)),atol=1e-6)
  loss=F.cross_entropy(m.head(m.body(x)),y);gs=torch.autograd.grad(loss,[m.body[0].bias,m.body[3].bias,m.body[8].bias,m.head.bias])
  gg=torch.autograd.grad(F.cross_entropy(b(x),y),list(b.parameters()))
  assert all(torch.allclose(a,c,atol=1e-6) for a,c in zip(gs,gg))
 print('PASS: 122 bias parameters; grayscale/RGB forward and gradients match original shared model',flush=True)
if __name__=='__main__':verify()
