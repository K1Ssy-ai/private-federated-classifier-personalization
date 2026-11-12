import sys,json,argparse,secrets,hashlib,itertools
import numpy as np, torch
import torch.nn.functional as F
from common import P,OLD,data,model
from privacy import epsilon
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fashion','kmnist','cifar'],required=True);a=p.parse_args();d=a.dataset
out=P/'E1'/d;out.mkdir(parents=True,exist_ok=True);x,y,xe,ye=data(d);lambdas=[.25,.5,.75,1.]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
pilot=[]
for seed in [9000,9001]:
 root=OLD/'pilot'/d/f's{seed}';net,ck=model(root,d);split=json.loads((root/'partition.json').read_text());budget=ck['budget'];lo,hi=.001,10.
 for _ in range(55):
  mid=(lo+hi)/2
  if epsilon([(.08,budget['sigma_body'],300),(1.,mid,1)],1e-5)>4:lo=mid
  else:hi=mid
 rng=np.random.default_rng(secrets.randbits(128));counts=np.array([np.bincount(y[ix].numpy(),minlength=10) for ix in split['train']]);noisy=np.maximum(counts+rng.normal(0,hi,counts.shape),0)+1;probs=noisy/noisy.sum(1,keepdims=True);offset=np.log(probs)-np.log(.1);offset-=offset.mean(1,keepdims=True)
 rows={str(l):[] for l in lambdas}
 with torch.no_grad():
  for k in range(20):
   ix=split['validation'][k];z=net(x[ix]);yy=y[ix];delta=torch.tensor(offset[k],dtype=z.dtype)
   for l in lambdas:rows[str(l)].append(float(((z+l*delta).argmax(1)==yy).float().mean()))
 record={'seed':seed,'shared_sha256':digest(root/'shared.pt'),'histogram_sigma':hi,'noised_offsets':offset.tolist(),'validation_client_accuracy':rows,'means':{k:float(np.mean(v)) for k,v in rows.items()}}
 (out/f'pilot_s{seed}.json').write_text(json.dumps(record,indent=2),encoding='utf-8');pilot.append(record)
avgs={str(l):float(np.mean([v['means'][str(l)] for v in pilot])) for l in lambdas};best=max(lambdas,key=lambda l:(avgs[str(l)],-l))
selection={'dataset':d,'pilot_seeds':[9000,9001],'alpha':.5,'epsilon':4,'selection':'highest mean pilot client validation accuracy; ties choose smaller lambda','pilot_means':avgs,'selected_lambda':best,'privacy_note':'pilot selection uses validation labels; not included in the formal branch DP accounting'}
(out/'selection_frozen.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
seeds={'fashion':range(2030,2035),'kmnist':range(2041,2046),'cifar':range(2040,2045)}[d]
for alpha,eps,seed in itertools.product([.1,.5,1.],[4,8],seeds):
 root=OLD/'formal'/d/f'a{alpha:g}_e{eps}_s{seed}';target=out/f'a{alpha:g}_e{eps}_s{seed}.json'
 if target.exists():continue
 net,ck=model(root,d);split=json.loads((root/'partition.json').read_text());h=json.loads((root/'histogram.json').read_text());offset=np.array(h['offsets']);result={str(l):{s:[] for s in ['validation','test']} for l in lambdas}
 with torch.no_grad():
  for k in range(20):
   for sp in ['validation','test']:
    xx,yy=(x,y) if sp=='validation' else (xe,ye);ix=split[sp][k];z=net(xx[ix]);labels=yy[ix];delta=torch.tensor(offset[k],dtype=z.dtype)
    for l in lambdas:
     zz=z+l*delta;result[str(l)][sp].append({'accuracy':float((zz.argmax(1)==labels).float().mean()),'cross_entropy':float(F.cross_entropy(zz,labels))})
 rec={'dataset':d,'alpha':alpha,'epsilon':eps,'seed':seed,'selected_lambda':best,'shared_sha256':digest(root/'shared.pt'),'histogram_sha256':digest(root/'histogram.json'),'metrics':result}
 target.write_text(json.dumps(rec,indent=2),encoding='utf-8');print('E1',d,alpha,eps,seed,flush=True)
print('E1 COMPLETE',d,flush=True)
