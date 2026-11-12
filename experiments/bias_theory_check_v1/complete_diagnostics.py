import sys,json,os
from pathlib import Path
import torch
B=Path(str(Path(__file__).resolve().parents[1]));O=B/'bias_theory_check_v1';sys.path.insert(0,str(O));import run_experiment as run
torch.set_num_threads(4);torch.set_default_dtype(torch.float64)
part=run.load(run.R/'partition.json');zall=torch.load(run.P/'cache/9000_train_full.pt',weights_only=True)['logits'].double();_,y=run.data.load(B/'kmnist_extension_v1/data',True);sigma=run.load(O/'settings.json')['sigma_B']
for p in (O/'trajectories').glob('*/*/*.json'):
 v=run.load(p);k=v['client'];C=v['C'];z=zall[part['train'][k]];labels=y[part['train'][k]];d=torch.zeros(10)
 for step in v['steps']:
  loss,g,c,mu,beta,m2,clip=run.stats(z,labels,d,C,sigma)
  assert abs(float(loss)-step['loss'])<1e-12 and abs(float(m2)-step['m2'])<1e-12
  step.update({'d':d.tolist(),'mu_C':mu.tolist(),'beta_C':beta.tolist(),'full_gradient':g.mean(0).tolist()});d-=.03*torch.tensor(step['G'])
 assert torch.allclose(d,torch.tensor(v['d_final']),atol=1e-12);run.save(p,v)
print('saved per-step vectors for all trajectories')
