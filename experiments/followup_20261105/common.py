import sys, tarfile, pickle, importlib.util
from pathlib import Path
import numpy as np, torch
from torch import nn
P=Path(__file__).resolve().parent
OLD=P.parent/'followup_20261104'/'unified_v2'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'kmnist_extension_v1'))
torch.set_num_threads(4)
class Net(nn.Module):
 def __init__(self,dataset):
  super().__init__();self.body=nn.Sequential(nn.Conv2d(3 if dataset=='cifar' else 1,16,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.Tanh(),nn.AvgPool2d(2),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(512,64),nn.Tanh());self.head=nn.Linear(64,10)
 def forward(self,z):return self.head(self.body(z))
class Offset(nn.Module):
 def __init__(self):super().__init__();self.offset=nn.Parameter(torch.zeros(10))
 def forward(self,z):return z+self.offset
def data(dataset):
 if dataset=='cifar':
  with tarfile.open(str(Path(__file__).resolve().parents[1] / "data" / "cifar-10-python.tar.gz"),'r:gz') as t:
   ds=[pickle.load(t.extractfile(f'cifar-10-batches-py/data_batch_{i}'),encoding='bytes') for i in range(1,6)];te=pickle.load(t.extractfile('cifar-10-batches-py/test_batch'),encoding='bytes')
  x=torch.from_numpy(np.concatenate([v[b'data'] for v in ds]).reshape(-1,3,32,32).copy()).float()/127.5-1;y=torch.tensor(sum([v[b'labels'] for v in ds],[]));xe=torch.from_numpy(te[b'data'].reshape(-1,3,32,32).copy()).float()/127.5-1;ye=torch.tensor(te[b'labels']);return x,y,xe,ye
 path=Path(str(Path(__file__).resolve().parents[1] / 'kmnist_extension_v1/data.py')) if dataset=='kmnist' else Path(str(Path(__file__).resolve().parents[1] / 'data.py'))
 sp=importlib.util.spec_from_file_location('dataset_loader',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);x,y=m.load(path.parent/'data',True);xe,ye=m.load(path.parent/'data',False);return x,y,xe,ye
def model(root,dataset):
 m=Net(dataset);s=torch.load(root/'shared.pt',map_location='cpu',weights_only=True);m.load_state_dict(s['model']);m.eval();return m,s
