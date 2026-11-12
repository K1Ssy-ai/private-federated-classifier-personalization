import argparse,json,hashlib
import numpy as np,torch
import torch.nn.functional as F
from common import P,data,model
p=argparse.ArgumentParser();p.add_argument('--alpha',type=float,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
root=P/'dual'/'formal'/'cifar'/f'a{a.alpha:g}_e4_s{a.seed}';split=json.loads((root/'partition.json').read_text());h=json.loads((root/'histogram.json').read_text());strength=json.loads((P/'E1'/'cifar'/'selection_frozen.json').read_text())['selected_lambda'];x,y,xe,ye=data('cifar');net,ck=model(root,'cifar');rows={sp:[] for sp in ['validation','test']}
with torch.no_grad():
 for k in range(20):
  for sp in rows:
   xx,yy=(x,y) if sp=='validation' else (xe,ye);ix=split[sp][k];z=net(xx[ix]);lab=yy[ix];z=z+strength*torch.tensor(h['offsets'][k],dtype=z.dtype);rows[sp].append({'accuracy':float((z.argmax(1)==lab).float().mean()),'cross_entropy':float(F.cross_entropy(z,lab))})
out={'dataset':'cifar','alpha':a.alpha,'epsilon':4,'seed':a.seed,'selected_lambda':strength,'shared_sha256':hashlib.sha256((root/'shared.pt').read_bytes()).hexdigest(),'histogram_sha256':hashlib.sha256((root/'histogram.json').read_bytes()).hexdigest(),'metrics':rows}
target=P/'E3'/'cifar'/f'a{a.alpha:g}_s{a.seed}_hist.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2),encoding='utf-8');print('E3 HIST COMPLETE',a.alpha,a.seed,flush=True)
