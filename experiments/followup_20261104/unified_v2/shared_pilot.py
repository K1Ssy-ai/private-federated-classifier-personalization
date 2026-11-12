import sys,json,time,tarfile,pickle,hashlib,copy,argparse,importlib.util,os
from pathlib import Path
import numpy as np,torch
from torch import nn
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'kmnist_extension_v1'))
from data import partition
from engine import step,generator
from privacy import plan
torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fashion','kmnist','cifar'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',choices=['pilot','formal'],default='pilot');p.add_argument('--alpha',type=float,default=.5);p.add_argument('--epsilon',type=float,default=4.);args=p.parse_args()
root=(BASE/'pilot'/args.dataset/f's{args.seed}') if args.stage=='pilot' else (BASE/'formal'/args.dataset/f'a{args.alpha:g}_e{args.epsilon:g}_s{args.seed}');root.mkdir(parents=True,exist_ok=True)
size=(2250,250) if args.dataset=='cifar' else (2700,300)
cfg={'dataset':args.dataset,'clients':20,'alpha':args.alpha,'epsilon':args.epsilon,'seed':args.seed,'rounds':30,'local_steps':10,'head_steps':50,'q':.08,'delta':1e-5,'body_fraction':.8,'clip':1.,'body_lr':.1,'head_lr':.1,'microbatch':16,'train_per_client':size[0],'validation_per_client':size[1],'test_per_client':500}
cfgpath=root/'config.json'
if cfgpath.exists():assert json.loads(cfgpath.read_text())==cfg,'Cannot resume with different settings'
else:cfgpath.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
if args.dataset=='cifar':
 ar=Path(str(Path(__file__).resolve().parents[2] / "data" / "cifar-10-python.tar.gz"))
 with ar.open('rb') as f:assert hashlib.file_digest(f,'md5').hexdigest()=='c58f30108f718f92721af3b95e74349a'
 with tarfile.open(ar,'r:gz') as t:
  ds=[pickle.load(t.extractfile(f'cifar-10-batches-py/data_batch_{i}'),encoding='bytes') for i in range(1,6)];te=pickle.load(t.extractfile('cifar-10-batches-py/test_batch'),encoding='bytes')
 x=torch.from_numpy(np.concatenate([v[b'data'] for v in ds]).reshape(-1,3,32,32).copy()).float()/127.5-1;y=torch.tensor(sum([v[b'labels'] for v in ds],[]));yt=torch.tensor(te[b'labels']);del ds,te
else:
 path=Path(str(Path(__file__).resolve().parents[2] / 'kmnist_extension_v1/data.py')) if args.dataset=='kmnist' else Path(str(Path(__file__).resolve().parents[2] / 'data.py'))
 sp=importlib.util.spec_from_file_location('dataset_loader',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 dr=path.parent/'data';m.download(dr);x,y=m.load(dr,True);_,yt=m.load(dr,False)
split=partition(y,yt,cfg);fp=root/'partition.json'
if fp.exists():assert json.loads(fp.read_text())==split
else:fp.write_text(json.dumps(split))
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Conv2d(3 if args.dataset=='cifar' else 1,16,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.Tanh());self.head=nn.Linear(64,10)
 def forward(self,z):return self.head(self.body(z))
budget=plan(cfg,'dual');torch.manual_seed(args.seed);g=Net();rng=generator(torch.device('cpu'));ck=root/'shared.pt';first=0;start=time.time()
if ck.exists():
 state=torch.load(ck,map_location='cpu',weights_only=True);g.load_state_dict(state['model']);first=state['round']
for rnd in range(first,30):
 total=None
 for ids in split['train']:
  model=copy.deepcopy(g);xx=x[ids];yy=y[ids]
  for _ in range(10):step(model,xx,yy,q=.08,sigma=budget['sigma_body'],clip=1.,lr=.1*(.5 if rnd>=21 else 1),microbatch=16,rng=rng)
  st=model.state_dict()
  if total is None:total={k:v.detach().clone()/20 for k,v in st.items()}
  else:
   for k,v in st.items():total[k]+=v.detach()/20
 g.load_state_dict(total);tmp=root/'shared.tmp';torch.save({'round':rnd+1,'model':g.state_dict(),'budget':budget},tmp);os.replace(tmp,ck)
 print(args.dataset,args.seed,'round',rnd+1,'elapsed',round(time.time()-start,1),flush=True)
(root/'shared_done.json').write_text(json.dumps({'cfg':cfg,'budget':budget,'sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start},indent=2),encoding='utf-8')
