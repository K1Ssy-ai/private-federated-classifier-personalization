import sys,json,time,tarfile,pickle,copy,argparse,importlib.util,os,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'kmnist_extension_v1'));sys.path.insert(0,str(BASE))
from engine import step,generator
from bias_model import AllBias
torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fashion','kmnist','cifar'],required=True);p.add_argument('--seed',type=int,choices=[9000,9001],required=True);args=p.parse_args()
root=BASE/'pilot'/args.dataset/f's{args.seed}';done=json.loads((root/'shared_done.json').read_text());cfg=done['cfg'];budget=done['budget'];split=json.loads((root/'partition.json').read_text())
assert hashlib.sha256((root/'shared.pt').read_bytes()).hexdigest()==done['sha256']
if args.dataset=='cifar':
 with tarfile.open(str(Path(__file__).resolve().parents[2] / "data" / "cifar-10-python.tar.gz"),'r:gz') as t:ds=[pickle.load(t.extractfile(f'cifar-10-batches-py/data_batch_{i}'),encoding='bytes') for i in range(1,6)]
 x=torch.from_numpy(np.concatenate([v[b'data'] for v in ds]).reshape(-1,3,32,32).copy()).float()/127.5-1;y=torch.tensor(sum([v[b'labels'] for v in ds],[]));del ds
else:
 path=Path(str(Path(__file__).resolve().parents[2] / 'kmnist_extension_v1/data.py')) if args.dataset=='kmnist' else Path(str(Path(__file__).resolve().parents[2] / 'data.py'))
 sp=importlib.util.spec_from_file_location('load_data',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);x,y=m.load(path.parent/'data',True)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Conv2d(3 if args.dataset=='cifar' else 1,16,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.Tanh());self.head=nn.Linear(64,10)
 def forward(self,z):return self.head(self.body(z))
class Offset(nn.Module):
 def __init__(self):super().__init__();self.offset=nn.Parameter(torch.zeros(10))
 def forward(self,z):return z+self.offset
g=Net();g.load_state_dict(torch.load(root/'shared.pt',map_location='cpu',weights_only=True)['model']);g.eval()
candidates=[(scope,lr,clip) for scope in ['B1','B2','B3'] for lr in [.03,.1] for clip in [.5,1.,2.]]
out=root/'adapt';out.mkdir(exist_ok=True);rng=generator(torch.device('cpu'));started=time.time()
for k in range(20):
 tr=split['train'][k];va=split['validation'][k];tx=x[tr];ty=y[tr];vx=x[va];vy=y[va]
 with torch.no_grad():ft=torch.cat([g.body(tx[i:i+256]) for i in range(0,len(tx),256)]);vf=g.body(vx);lg=g.head(ft);vl=g.head(vf)
 for scope,lr,clip in candidates:
  target=out/f'{scope}_lr{lr:g}_clip{clip:g}.pt'
  state=torch.load(target,map_location='cpu',weights_only=True) if target.exists() else {'scope':scope,'lr':lr,'clip':clip,'body_sha256':done['sha256'],'rows':[]}
  assert state['body_sha256']==done['sha256'] and state['scope']==scope and state['lr']==lr and state['clip']==clip
  if len(state['rows'])>k:continue
  assert len(state['rows'])==k
  module=Offset() if scope=='B1' else copy.deepcopy(g.head) if scope=='B2' else AllBias(g.state_dict())
  tt=lg if scope=='B1' else ft if scope=='B2' else tx;vv=vl if scope=='B1' else vf if scope=='B2' else vx
  for _ in range(50):step(module,tt,ty,q=.08,sigma=budget['sigma_head'],clip=clip,lr=lr,microbatch=16,rng=rng)
  with torch.no_grad():acc=float((module(vv).argmax(1)==vy).float().mean())
  state['rows'].append({'client':k,'validation_accuracy':acc,'parameters':{n:v.detach().clone() for n,v in module.named_parameters()}})
  tmp=target.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,target)
 print(args.dataset,args.seed,'pilot client',k+1,'/20 seconds',round(time.time()-started,1),flush=True)
summary=[]
for scope,lr,clip in candidates:
 st=torch.load(out/f'{scope}_lr{lr:g}_clip{clip:g}.pt',map_location='cpu',weights_only=True);assert len(st['rows'])==20
 summary.append({'scope':scope,'lr':lr,'clip':clip,'validation_mean':float(np.mean([a['validation_accuracy'] for a in st['rows']])), 'client_accuracies':[a['validation_accuracy'] for a in st['rows']]})
(root/'pilot_validation.json').write_text(json.dumps({'dataset':args.dataset,'seed':args.seed,'shared_sha256':done['sha256'],'results':summary,'elapsed_seconds':time.time()-started},indent=2),encoding='utf-8')
print('PILOT VALIDATION COMPLETE',args.dataset,args.seed,flush=True)
