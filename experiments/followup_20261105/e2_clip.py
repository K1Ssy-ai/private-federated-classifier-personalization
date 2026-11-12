import argparse,json,copy,time,secrets,hashlib,os
import numpy as np,torch
from torch.nn import functional as F
from common import P,OLD,data,model,Offset
import engine
from engine import step
def recorded_clipped_sum(module,x,y,clip,microbatch):
 sums={n:torch.zeros_like(p) for n,p in module.named_parameters()};allnorm=[]
 for start in range(0,len(y),microbatch):
  grads=engine.sample_gradients(module,x[start:start+microbatch],y[start:start+microbatch]);norm=torch.stack([v.flatten(1).square().sum(1) for v in grads.values()]).sum(0).sqrt();allnorm.extend(norm.detach().tolist());scales=(clip/norm.clamp_min(1e-12)).clamp(max=1)
  for name,g in grads.items():sums[name].add_((g*scales.reshape((len(scales),)+(1,)*(g.ndim-1))).sum(0).detach())
 if allnorm:
  v=np.asarray(allnorm);training_steps.append({'sampled':len(v),'clipped':int((v>clip).sum()),'norm_quantiles':np.quantile(v,[0,.1,.25,.5,.75,.9,1]).tolist()})
 else:training_steps.append({'sampled':0,'clipped':0,'norm_quantiles':None})
 return sums
engine.clipped_sum=recorded_clipped_sum
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fashion','cifar'],required=True);p.add_argument('--alpha',type=float,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();d=a.dataset
root=(OLD if d=='fashion' else P/'dual')/'formal'/d/f'a{a.alpha:g}_e4_s{a.seed}';done=json.loads((root/'shared_done.json').read_text());split=json.loads((root/'partition.json').read_text());sel=json.loads((OLD/'pilot'/d/'selection_frozen.json').read_text())['selected'];x,y,xe,ye=data(d);g,ck=model(root,d)
target=P/'E2'/d/f'a{a.alpha:g}_s{a.seed}.pt';target.parent.mkdir(parents=True,exist_ok=True)
state=torch.load(target,weights_only=True,map_location='cpu') if target.exists() else {'metadata':{'dataset':d,'alpha':a.alpha,'epsilon':4,'seed':a.seed,'shared_sha256':done['sha256'],'clipping_thresholds':[.5,1.,2.,4.,8.],'lr':{m:sel[m]['lr'] for m in ['B1','B2']},'head_noise_multiplier':ck['budget']['sigma_head'],'diagnostic_notice':'gradient statistics are non-DP training-data diagnostics; all branches jointly require privacy composition'},'clients':[]}
assert state['metadata']['shared_sha256']==done['sha256']
for k in range(len(state['clients']),20):
 tr=split['train'][k];va=split['validation'][k];te=split['test'][k];ty=y[tr];vy=y[va];ey=ye[te]
 with torch.no_grad():
  ft=torch.cat([g.body(x[tr[j:j+256]]) for j in range(0,len(tr),256)]);vf=g.body(x[va]);ef=g.body(xe[te]);lg=g.head(ft);vl=g.head(vf);el=g.head(ef)
 row={}
 for scope in ['B1','B2']:
  internal_random_seed=secrets.randbits(63)
  for clip in [.5,1.,2.,4.,8.]:
   rng=torch.Generator(device='cpu');rng.manual_seed(internal_random_seed)
   module=Offset() if scope=='B1' else copy.deepcopy(g.head);tt=lg if scope=='B1' else ft;vv=vl if scope=='B1' else vf;ee=el if scope=='B1' else ef;curve=[];training_steps=[]
   for i in range(51):
    if i%10==0:
     with torch.no_grad():
      z=module(tt);res=z.softmax(1)-F.one_hot(ty,10);norm=res.norm(dim=1)
      if scope=='B2':norm=norm*torch.sqrt(ft.square().sum(1)+1)
      zv=module(vv);qt=torch.quantile(norm,torch.tensor([0.,.1,.25,.5,.75,.9,1.]));curve.append({'step':i,'validation_accuracy':float((zv.argmax(1)==vy).float().mean()),'validation_cross_entropy':float(F.cross_entropy(zv,vy)),'training_clipping_fraction':float((norm>clip).float().mean()),'training_gradient_norm_quantiles':qt.tolist()})
    if i<50:step(module,tt,ty,q=.08,sigma=ck['budget']['sigma_head'],clip=clip,lr=sel[scope]['lr'],microbatch=16,rng=rng)
   with torch.no_grad():
    z=module(ee);test={'accuracy':float((z.argmax(1)==ey).float().mean()),'cross_entropy':float(F.cross_entropy(z,ey))}
   row[f'{scope}_C{clip:g}']={'curve':curve,'training_steps':training_steps,'test':test,'parameters':{n:v.detach().clone() for n,v in module.named_parameters()}}
 state['clients'].append(row);tmp=target.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,target);print('E2',d,a.alpha,a.seed,'client',k+1,flush=True)
print('E2 COMPLETE',d,a.alpha,a.seed,flush=True)
