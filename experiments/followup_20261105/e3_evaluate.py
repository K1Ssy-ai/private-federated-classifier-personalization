import json,argparse,hashlib
import numpy as np,torch
import torch.nn.functional as F
from common import P,OLD,data,model
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fashion','cifar'],required=True);p.add_argument('--alpha',type=float,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();d=a.dataset
root=P/'shared'/'formal'/d/f'a{a.alpha:g}_e4_s{a.seed}';dual=(OLD if d=='fashion' else P/'dual')/'formal'/d/f'a{a.alpha:g}_e4_s{a.seed}'
split=json.loads((root/'partition.json').read_text());ref=json.loads((dual/'partition.json').read_text())
# Historic JSON serialization may differ at machine precision in saved floating
# Dirichlet priors. Pairing requires exact record indices, not byte-identical
# decimal representations of those ancillary metadata values.
assert all(split[k]==ref[k] for k in ['train','validation','test']),'record partition mismatch: comparisons cannot be paired'
prior_max_abs_diff=float(np.max(np.abs(np.asarray(split['priors'])-np.asarray(ref['priors']))))
assert prior_max_abs_diff < 1e-12 and split['definition']==ref['definition'],'partition metadata differs materially'
x,y,xe,ye=data(d);net,ck=model(root,d);rows={sp:[] for sp in ['validation','test']};conf={sp:np.zeros((10,10),dtype=int) for sp in rows}
with torch.no_grad():
 for k in range(20):
  for sp in rows:
   xx,yy=(x,y) if sp=='validation' else (xe,ye);ix=split[sp][k];z=net(xx[ix]);lab=yy[ix];pred=z.argmax(1);rows[sp].append({'accuracy':float((pred==lab).float().mean()),'cross_entropy':float(F.cross_entropy(z,lab))});conf[sp]+=np.bincount((lab*10+pred).numpy(),minlength=100).reshape(10,10)
out={'dataset':d,'alpha':a.alpha,'epsilon':4,'seed':a.seed,'stage_a_budget':ck['budget'],'shared_sha256':hashlib.sha256((root/'shared.pt').read_bytes()).hexdigest(),'paired_two_stage_shared_sha256':hashlib.sha256((dual/'shared.pt').read_bytes()).hexdigest(),'partition_match':True,'record_indices_exact_match':True,'prior_max_abs_diff':prior_max_abs_diff,'metrics':{sp:{'clients':arr,'mean_accuracy':float(np.mean([z['accuracy'] for z in arr])),'p10_accuracy':float(np.quantile([z['accuracy'] for z in arr],.1)),'mean_cross_entropy':float(np.mean([z['cross_entropy'] for z in arr])),'confusion':conf[sp].tolist()} for sp,arr in rows.items()}}
target=P/'E3'/d/f'a{a.alpha:g}_s{a.seed}.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2),encoding='utf-8');print('E3 EVALUATION COMPLETE',d,a.alpha,a.seed,flush=True)
