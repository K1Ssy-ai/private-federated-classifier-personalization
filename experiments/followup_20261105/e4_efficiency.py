import sys,json,time,tarfile,pickle,copy,argparse,importlib.util,threading,os
from pathlib import Path
import numpy as np,torch,ctypes
from torch import nn
P=Path(__file__).resolve().parent;sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'kmnist_extension_v1'));sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'followup_20261104/unified_v2'))
from engine import step,generator
from bias_model import AllBias
torch.set_num_threads(4)
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fashion','kmnist','cifar'],required=True);p.add_argument('--method',choices=['B1','B2','B3','B4'],required=True);p.add_argument('--client',type=int,choices=[0,7,14],required=True);a=p.parse_args()
seed={'fashion':2030,'kmnist':2041,'cifar':2040}[a.dataset];root=Path(str(Path(__file__).resolve().parents[1] / 'followup_20261104/unified_v2'))/'formal'/a.dataset/f'a0.5_e4_s{seed}';split=json.loads((root/'partition.json').read_text());saved=torch.load(root/'shared.pt',map_location='cpu',weights_only=True);selected=json.loads((Path(str(Path(__file__).resolve().parents[1] / 'followup_20261104/unified_v2'))/'pilot'/a.dataset/'selection_frozen.json').read_text())['selected'];hist=json.loads((root/'histogram.json').read_text())
if a.dataset=='cifar':
 with tarfile.open(str(Path(__file__).resolve().parents[1] / "data" / "cifar-10-python.tar.gz"),'r:gz') as t:ds=[pickle.load(t.extractfile(f'cifar-10-batches-py/data_batch_{i}'),encoding='bytes') for i in range(1,6)]
 x=torch.from_numpy(np.concatenate([v[b'data'] for v in ds]).reshape(-1,3,32,32).copy()).float()/127.5-1;y=torch.tensor(sum([v[b'labels'] for v in ds],[]));del ds
else:
 path=Path(str(Path(__file__).resolve().parents[1] / 'kmnist_extension_v1/data.py')) if a.dataset=='kmnist' else Path(str(Path(__file__).resolve().parents[1] / 'data.py'));sp=importlib.util.spec_from_file_location('load_data',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);x,y=m.load(path.parent/'data',True)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.body=nn.Sequential(nn.Conv2d(3 if a.dataset=='cifar' else 1,16,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.Tanh());self.head=nn.Linear(64,10)
 def forward(self,z):return self.head(self.body(z))
class Offset(nn.Module):
 def __init__(self):super().__init__();self.offset=nn.Parameter(torch.zeros(10))
 def forward(self,z):return z+self.offset
class MemoryCounters(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
def rss():
 c=MemoryCounters();c.cb=ctypes.sizeof(c);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p;h=ctypes.windll.kernel32.GetCurrentProcess()
 fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(MemoryCounters),ctypes.c_ulong];fn.restype=ctypes.c_int
 assert fn(h,ctypes.byref(c),c.cb)
 return c.WorkingSetSize
g=Net();g.load_state_dict(saved['model']);g.eval();ids=split['train'][a.client];tx=x[ids];ty=y[ids];rng=generator(torch.device('cpu'));measure=[]
for rep in range(6):
 peak=[rss()];run=[True]
 def monitor():
  while run[0]:peak[0]=max(peak[0],rss());time.sleep(.005)
 thread=threading.Thread(target=monitor,daemon=True);thread.start();start=time.perf_counter();module=None
 if a.method in ['B1','B2']:
  with torch.no_grad():features=torch.cat([g.body(tx[j:j+256]) for j in range(0,len(tx),256)]);logits=g.head(features)
  module=Offset() if a.method=='B1' else copy.deepcopy(g.head);train=logits if a.method=='B1' else features
 elif a.method=='B3':module=AllBias(g.state_dict());train=tx
 else:train=ty
 prep=time.perf_counter()-start;update_start=time.perf_counter()
 if a.method=='B4':
  counts=np.bincount(ty.numpy(),minlength=10);r=np.random.default_rng();q=np.maximum(counts+r.normal(0,hist['sigma'],10),0)+1;q=q/q.sum();_ =np.log(q)-np.log(.1)
 else:
  params=selected[a.method]
  for _ in range(50):step(module,train,ty,q=.08,sigma=saved['budget']['sigma_head'],clip=params['clip'],lr=params['lr'],microbatch=16,rng=rng)
 update=time.perf_counter()-update_start;total=time.perf_counter()-start;run[0]=False;thread.join()
 if rep:measure.append({'preparation_seconds':prep,'update_seconds':update,'total_seconds':total,'rss_peak_bytes':peak[0]})
 print(a.dataset,a.method,'rep',rep,'seconds',round(total,3),flush=True)
out={'dataset':a.dataset,'method':a.method,'client':a.client,'seed':seed,'alpha':.5,'epsilon':4,'warmups':1,'measured_repetitions':5,'threads':4,'device':'CPU','param_count':{'B1':10,'B2':650,'B3':122,'B4':0}[a.method],'rss_measure':'process RSS including loaded dataset and model, sampled every 5ms','cache_cost_included_in_total':True,'measurements':measure}
target=P/'E4'/a.dataset;target.mkdir(parents=True,exist_ok=True);(target/f'{a.method}_client{a.client}.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
