import sys,json,time,tarfile,pickle,copy,argparse,importlib.util,hashlib,os,secrets,zipfile
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
P=Path(__file__).resolve().parent;sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'kmnist_extension_v1'));sys.path.insert(0,str(P))
from engine import step,generator
from privacy import epsilon
from bias_model import AllBias
torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fashion','kmnist','cifar'],required=True);p.add_argument('--alpha',type=float,required=True);p.add_argument('--epsilon',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
root=P/'formal'/a.dataset/f'a{a.alpha:g}_e{a.epsilon}_s{a.seed}';done=json.loads((root/'shared_done.json').read_text());cfg=done['cfg'];budget=done['budget'];split=json.loads((root/'partition.json').read_text());selected=json.loads((P/'pilot'/a.dataset/'selection_frozen.json').read_text())['selected']
assert hashlib.sha256((root/'shared.pt').read_bytes()).hexdigest()==done['sha256']
if a.dataset=='cifar':
 with tarfile.open(str(Path(__file__).resolve().parents[2] / "data" / "cifar-10-python.tar.gz"),'r:gz') as t:
  ds=[pickle.load(t.extractfile(f'cifar-10-batches-py/data_batch_{i}'),encoding='bytes') for i in range(1,6)];te=pickle.load(t.extractfile('cifar-10-batches-py/test_batch'),encoding='bytes')
 x=torch.from_numpy(np.concatenate([v[b'data'] for v in ds]).reshape(-1,3,32,32).copy()).float()/127.5-1;y=torch.tensor(sum([v[b'labels'] for v in ds],[]));xe=torch.from_numpy(te[b'data'].reshape(-1,3,32,32).copy()).float()/127.5-1;ye=torch.tensor(te[b'labels']);del ds,te
else:
 path=Path(str(Path(__file__).resolve().parents[2] / 'kmnist_extension_v1/data.py')) if a.dataset=='kmnist' else Path(str(Path(__file__).resolve().parents[2] / 'data.py'))
 sp=importlib.util.spec_from_file_location('load_data',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);x,y=m.load(path.parent/'data',True);xe,ye=m.load(path.parent/'data',False)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Conv2d(3 if a.dataset=='cifar' else 1,16,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.Tanh());self.head=nn.Linear(64,10)
 def forward(self,z):return self.head(self.body(z))
class Offset(nn.Module):
 def __init__(self):super().__init__();self.offset=nn.Parameter(torch.zeros(10))
 def forward(self,z):return z+self.offset
g=Net();g.load_state_dict(torch.load(root/'shared.pt',map_location='cpu',weights_only=True)['model']);g.eval();rng=generator(torch.device('cpu'));target=root/'branches.pt'
signature={'shared_sha256':done['sha256'],'selection_sha256':hashlib.sha256((P/'pilot'/a.dataset/'selection_frozen.json').read_bytes()).hexdigest()}
state=torch.load(target,map_location='cpu',weights_only=True) if target.exists() else {'signature':signature,'rows':[]}
assert state['signature']==signature
for k in range(len(state['rows']),20):
 tr=split['train'][k];tx=x[tr];ty=y[tr]
 with torch.no_grad():ft=torch.cat([g.body(tx[i:i+256]) for i in range(0,len(tx),256)]);lg=g.head(ft)
 saved={}
 for scope in ['B1','B2','B3']:
  params=selected[scope];module=Offset() if scope=='B1' else copy.deepcopy(g.head) if scope=='B2' else AllBias(g.state_dict())
  tt=lg if scope=='B1' else ft if scope=='B2' else tx;begin=time.perf_counter()
  for _ in range(50):step(module,tt,ty,q=.08,sigma=budget['sigma_head'],clip=params['clip'],lr=params['lr'],microbatch=16,rng=rng)
  saved[scope]={'parameters':{n:v.detach().clone() for n,v in module.named_parameters()},'optimization_seconds':time.perf_counter()-begin}
 state['rows'].append(saved);tmp=target.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,target)
 print('FORMAL ADAPT',a.dataset,a.alpha,a.epsilon,a.seed,'client',k+1,flush=True)
sigma_lo,sigma_hi=.001,10.
for _ in range(55):
 mid=(sigma_lo+sigma_hi)/2
 if epsilon([(.08,budget['sigma_body'],300),(1.,mid,1)],cfg['delta'])>a.epsilon:sigma_lo=mid
 else:sigma_hi=mid
hf=root/'histogram.json'
if hf.exists():hist=json.loads(hf.read_text());assert hist['shared_sha256']==done['sha256'];d=np.array(hist['offsets'])
else:
 prior=None;origin='new Gaussian draw';condition=f'a{a.alpha:g}_e{a.epsilon}_s{a.seed}'
 if prior is None:
  counts=np.array([np.bincount(y[ix].numpy(),minlength=10) for ix in split['train']]);r=np.random.default_rng(secrets.randbits(128));histp=np.maximum(counts+r.normal(0,sigma_hi,counts.shape),0)+1;histp/=histp.sum(1,keepdims=True);prior=np.log(histp)-np.log(.1);prior-=prior.mean(1,keepdims=True)
 d=prior;hf.write_text(json.dumps({'shared_sha256':done['sha256'],'sigma':sigma_hi,'offsets':d.tolist(),'origin':origin}),encoding='utf-8')
names=['B0','B1','B2','B3','B4_1','B4_05','B5'];metrics={n:{s:[] for s in ['validation','test']} for n in names};conf={n:{s:np.zeros((10,10),dtype=int) for s in ['validation','test']} for n in names}
with torch.no_grad():
 for k in range(20):
  b3=AllBias(g.state_dict());b3.load_state_dict({**b3.state_dict(),**state['rows'][k]['B3']['parameters']});params=state['rows'][k]
  for sp in ['validation','test']:
   src,labels=(x,y) if sp=='validation' else (xe,ye);ix=split[sp][k];xx=src[ix];yy=labels[ix];ft=g.body(xx);z=g.head(ft);dh=torch.tensor(d[k],dtype=z.dtype);ph=params['B2']['parameters'];out={'B0':z,'B1':z+params['B1']['parameters']['offset'],'B2':F.linear(ft,ph['weight'],ph['bias']),'B3':b3(xx),'B4_1':z+dh,'B4_05':z+.5*dh}
   out['B5']=torch.zeros_like(z);out['B5'][:,int(d[k].argmax())]=1
   for name,logits in out.items():
    pred=logits.argmax(1);metrics[name][sp].append({'accuracy':float((pred==yy).float().mean()),'cross_entropy':float(F.cross_entropy(logits,yy)) if name!='B5' else None});conf[name][sp]+=np.bincount((yy*10+pred).numpy(),minlength=100).reshape(10,10)
summ={name:{sp:{'mean':float(np.mean([r['accuracy'] for r in arr])),'p10':float(np.quantile([r['accuracy'] for r in arr],.1)),'per_client':arr,'confusion':conf[name][sp].tolist()} for sp,arr in m.items()} for name,m in metrics.items()}
result={'cfg':cfg,'budget':budget,'histogram_sigma':sigma_hi,'histogram_epsilon':epsilon([(.08,budget['sigma_body'],300),(1.,sigma_hi,1)],cfg['delta']),'signature':signature,'pilot_selection':selected,'metrics':summ,'privacy_scope':'each branch separately; validation/test and multiple branch releases are outside single-branch guarantee','history':'historic test datasets used before protocol; exploratory follow-up'}
(root/'formal_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('FORMAL COMPLETE',a.dataset,a.alpha,a.epsilon,a.seed,flush=True)
