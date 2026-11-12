import os,sys,json,time,hashlib,math,secrets,platform,traceback
from pathlib import Path
import numpy as np, torch, opacus
from opacus.accountants.analysis import rdp
ROOT=Path(str(Path(__file__).resolve().parents[1])); OUT=ROOT/'prior_residual_pilot_v1'; OLD=ROOT/'followup_20261104/unified_v2'
sys.path.insert(0,str(ROOT/'followup_20261105')); sys.path.insert(0,str(ROOT/'kmnist_extension_v1'))
import common, privacy, data
torch.set_num_threads(4); torch.set_grad_enabled(True)
PILOT=[9000,9001]; FORMAL=[2041,2042,2043,2044,2045]; METHODS=['G','M0','M1','M2','H-init','M3','M4','NP']
def root(seed): return OLD/('pilot/kmnist/s'+str(seed) if seed in PILOT else 'formal/kmnist/a0.5_e4_s'+str(seed))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); os.replace(t,p)
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def t_save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');torch.save(obj,t);os.replace(t,p)
def log(s): print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
def scores(z,y,d):
 a=z.double()+d.double(); return {'acc':float((a.argmax(1)==y).double().mean()),'ce':float(torch.nn.functional.cross_entropy(a,y))}
def centered(x): return x-x.mean(dim=-1,keepdim=True)
def calibrate(a):
 A=[(.08,a,300)]
 def solve(prefix,q,steps):
  lo,hi=.05,1.
  while privacy.epsilon(prefix+[(q,hi,steps)],1e-5)>4: hi*=2
  for _ in range(36):
   mid=(lo+hi)/2
   if privacy.epsilon(prefix+[(q,mid,steps)],1e-5)>4:lo=mid
   else:hi=mid
  return hi
 b=solve(A,.08,50);h=solve(A,1,1);hi=2*h;r=solve(A+[(1,hi,1)],.08,50)
 events={'M0/M1':A+[(.08,b,50)],'M2':A+[(1,h,1)],'M3/M4':A+[(1,hi,1),(.08,r,50)]}; result={}
 for k,e in events.items():
  total=sum(rdp.compute_rdp(q=q,noise_multiplier=s,steps=n,orders=privacy.ORDERS) for q,s,n in e);ep,order=rdp.get_privacy_spent(orders=privacy.ORDERS,rdp=total,delta=1e-5);assert ep<=4+1e-6;result[k]={'events':e,'epsilon':float(ep),'best_order':float(order)}
 return {'sigma_A':a,'sigma_B':b,'sigma_H_full':h,'sigma_H_init':hi,'sigma_R':r,'branches':result,'delta':1e-5,'orders':privacy.ORDERS}
def checks():
 torch.manual_seed(394); z=torch.randn(12,10,dtype=torch.float64); y=torch.randint(10,(12,));d=torch.randn(10,dtype=torch.float64,requires_grad=True)
 aut=torch.stack([torch.autograd.grad(torch.nn.functional.cross_entropy(z[i:i+1]+d,y[i:i+1]),d)[0] for i in range(12)]);ana=torch.softmax(z+d,1)-torch.nn.functional.one_hot(y,10); assert torch.allclose(aut,ana,atol=1e-12)
 h=centered(torch.randn(10,dtype=torch.float64));r=centered(torch.randn(10,dtype=torch.float64));g=centered(torch.randn(10,dtype=torch.float64));lr=.1;gamma=.1
 r2=centered(r-lr*(g+gamma*r)); d2=centered(h+r-lr*(g+gamma*(centered(h+r)-h)));assert torch.allclose(torch.softmax(z+h+r2,1),torch.softmax(z+d2,1),atol=1e-12)
 assert torch.norm((h+centered(r-lr*gamma*r))-h)<torch.norm(r);assert torch.norm(centered(r-lr*gamma*r))<torch.norm(r)
 return {'gradient_max_abs_error':float((aut-ana).abs().max()),'coordinate_equivalence':True,'regularization_direction':True}
def np_opt(z,y):
 z=z.double();u=torch.zeros(10,dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([u],lr=1,max_iter=500,tolerance_grad=1e-6,tolerance_change=1e-10,line_search_fn='strong_wolfe'); calls=0
 def closure():
  nonlocal calls
  calls+=1;opt.zero_grad();d=centered(u);loss=torch.nn.functional.cross_entropy(z+d,y)+.5e-6*d.square().sum();loss.backward();return loss
 opt.step(closure)
 def getgrad():
  opt.zero_grad();d=centered(u);loss=torch.nn.functional.cross_entropy(z+d,y)+.5e-6*d.square().sum();loss.backward();return float(u.grad.abs().max())
 grad=getgrad();first=int(opt.state[u]['n_iter'])
 if grad>1e-6:
  opt=torch.optim.LBFGS([u],lr=1,max_iter=1500,tolerance_grad=1e-6,tolerance_change=1e-12,line_search_fn='strong_wolfe');opt.step(closure);grad=getgrad();iters=first+int(opt.state[u]['n_iter'])
 else: iters=first
 return centered(u.detach()),{'gradient_inf':grad,'iterations':iters,'function_calls':calls,'converged':grad<=1e-6,'train_ce':scores(z,y,centered(u.detach()))['ce']}
def histogram(y,sigma,gen):
 n=torch.bincount(y,minlength=10).double();n=(n+torch.randn(10,generator=gen,dtype=torch.float64)*sigma).clamp_min(0)+1;v=torch.log(n/n.sum())-math.log(.1);return .25*centered(v)
def train(z,y,h,budget,configs,gen):
 # Candidates share Poisson masks and standard Gaussian innovations.
 count=len(configs);r=torch.zeros((count,10),dtype=torch.float64); H=torch.stack([h if c['method'] in ('M3','M4') else torch.zeros(10,dtype=torch.float64) for c in configs]);lrs=torch.tensor([c['lr'] for c in configs],dtype=torch.float64)[:,None];gam=torch.tensor([c['gamma'] for c in configs],dtype=torch.float64)[:,None];sig=torch.tensor([budget['sigma_R'] if c['method'] in ('M3','M4') else budget['sigma_B'] for c in configs],dtype=torch.float64)[:,None];z=z.double()
 for step in range(50):
  ids=torch.rand(2700,generator=gen)<.08;z1=z[ids];y1=y[ids];sums=torch.zeros_like(r)
  for st in range(0,len(y1),16):
   zz=z1[st:st+16];yy=y1[st:st+16];g=torch.softmax(zz[None,:,:]+(H+r)[:,None,:],dim=2)-torch.nn.functional.one_hot(yy,10)[None,:,:];norm=g.norm(dim=2,keepdim=True);g=g*torch.clamp(2/norm,min=0,max=1);sums+=g.sum(dim=1)
  noise=torch.randn(10,generator=gen,dtype=torch.float64)[None,:];r=centered(r-lrs*((sums+noise*sig*2)/216+gam*r))
 return H+r,r
def configs():
 return [{'method':m,'lr':lr,'gamma':g} for m in ['M0','M1','M3','M4'] for lr in [.03,.1] for g in ([0.] if m in ['M0','M3'] else [.01,.1,1.])]
def cache(seed,split,x):
 p=OUT/'cache'/f'{seed}_{split}.pt'
 if p.exists():return torch.load(p,weights_only=True)
 model,_=common.model(root(seed),'kmnist');before=sha(root(seed)/'shared.pt');tm=time.perf_counter();parts=[]
 with torch.no_grad():
  for batch in x.split(512):parts.append(model(batch))
 assert sha(root(seed)/'shared.pt')==before
 obj={'logits':torch.cat(parts),'seconds':time.perf_counter()-tm,'source_sha256':before};t_save(p,obj);log(f'cache {seed} {split} {obj["seconds"]:.1f}s');return obj

def main():
 OUT.mkdir(exist_ok=True); allseeds=PILOT+FORMAL
 for s in allseeds:
  for name in ['shared.pt','partition.json','config.json','shared_done.json']:
   p=root(s)/name
   if not p.exists():raise FileNotFoundError(p)
 inputs={};a_values=[]
 for s in allseeds:
  rr=root(s);part=load(rr/'partition.json');cfg=load(rr/'config.json');sh=torch.load(rr/'shared.pt',map_location='cpu',weights_only=True);a_values.append(sh['budget']['sigma_body']);assert cfg['alpha']==.5 and cfg['epsilon']==4 and cfg['seed']==s
  for key,n,limit in [('train',2700,60000),('validation',300,60000),('test',500,10000)]:
   arrays=part[key];assert len(arrays)==20 and all(len(a)==n for a in arrays);flat=sum(arrays,[]);assert len(set(flat))==len(flat) and min(flat)>=0 and max(flat)<limit
  assert not set(sum(part['train'],[])) & set(sum(part['validation'],[]));assert sha(rr/'shared.pt')==load(rr/'shared_done.json')['sha256']
  inputs[str(s)]={f:sha(rr/f) for f in ['shared.pt','partition.json','config.json','shared_done.json']}
 assert len(set(a_values))==1
 save(OUT/'input_manifest.json',{'sources':inputs,'imports':{'common':common.__file__,'data':data.__file__,'privacy':privacy.__file__},'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'opacus':opacus.__version__},'source_files_sha256':{str(p):sha(p) for p in [Path(common.__file__),Path(data.__file__),Path(privacy.__file__),ROOT/'kmnist_extension_v1/engine.py',Path(__file__)]}})
 check=checks();bud=calibrate(a_values[0]);assert abs(bud['sigma_B']-sh['budget']['sigma_head'])<1e-8;save(OUT/'budget_verification.json',bud);save(OUT/'implementation_checks.json',check)
 x,y,xe,ye=common.data('kmnist'); selection_path=OUT/'selection_frozen.json';cs=configs()
 # Training and validation caches and NP on all seven precede pilot selection.
 for s in allseeds:
  rr=root(s);part=load(rr/'partition.json');tr=cache(s,'train_full',x)['logits'];va=tr # same full raw-training tensor, no extra forward
  for c in range(20):
   p=OUT/'np'/str(s)/f'{c:02}.json'
   if p.exists():continue
   t=time.perf_counter();it=torch.tensor(part['train'][c]);iv=torch.tensor(part['validation'][c]);d,diag=np_opt(tr[it],y[it]);save(p,{'seed':s,'client':c,'d':d.tolist(),'diagnostic':diag,'validation':scores(va[iv],y[iv],d),'seconds':time.perf_counter()-t});log(f'NP {s}/{c} converged={diag["converged"]}')
 # Entropy seeds persist privately in the local state, ensuring resumability.
 statepath=OUT/'local_rng_state.json'
 if not statepath.exists(): save(statepath,{str(s):[secrets.randbits(63) for _ in range(20)] for s in allseeds})
 seeds=load(statepath)
 def runclient(s,c,formal,chosen=None):
  p=OUT/('formal' if formal else 'pilot')/str(s)/f'{c:02}.json'
  if p.exists():return
  part=load(root(s)/'partition.json');it=torch.tensor(part['train'][c]);iv=torch.tensor(part['validation'][c]);trainlog=cache(s,'train_full',x)['logits'];z=trainlog[it];zv=trainlog[iv];yt=y[it];yv=y[iv];gen=torch.Generator().manual_seed(seeds[str(s)][c]);t=time.perf_counter();h=histogram(yt,bud['sigma_H_init'],gen);hfull=histogram(yt,bud['sigma_H_full'],gen);histtime=time.perf_counter()-t
  candidates=chosen if formal else cs;t=time.perf_counter();d,r=train(z,yt,h,bud,candidates,gen);train_seconds=time.perf_counter()-t;npdata=load(OUT/'np'/str(s)/f'{c:02}.json');vectors={'G':torch.zeros(10,dtype=torch.float64),'M2':hfull,'H-init':h,'NP':torch.tensor(npdata['d'],dtype=torch.float64)};res={}
  for j,conf in enumerate(candidates):
   key=conf['method'] if formal else f'{conf["method"]}_lr{conf["lr"]}_g{conf["gamma"]}';vectors[key]=d[j]
   res[key]={'config':conf,'r':r[j].tolist(),'r_norm':float(r[j].norm()),'seconds':train_seconds/len(candidates)}
  for key,v in vectors.items():
   v=v.detach();res.setdefault(key,{});res[key].update({'d':v.tolist(),'validation':scores(zv,yv,v)});
   if formal:
    te=cache(s,'test_full',xe)['logits'];idx=torch.tensor(part['test'][c]);res[key]['test']=scores(te[idx],ye[idx],v)
  res['M2']['seconds']=histtime/2;res['H-init']['seconds']=histtime/2;res['G']['seconds']=0.;res['NP']['seconds']=npdata['seconds']
  save(p,{'seed':s,'client':c,'status':'complete','counts':{'train':2700,'validation':300,'test':500},'h_init':h.tolist(),'h_full':hfull.tolist(),'results':res,'shared_sha256':sha(root(s)/'shared.pt'),'partition_sha256':sha(root(s)/'partition.json'),'budget_file':'budget_verification.json','training_wall_seconds_all_candidates':train_seconds});log(f'{"formal" if formal else "pilot"} {s}/{c}')
 for s in PILOT:
  for c in range(20):runclient(s,c,False)
 if not selection_path.exists():
  entries=[];selected=[]
  for conf in cs:
   key=f'{conf["method"]}_lr{conf["lr"]}_g{conf["gamma"]}';vals=[load(OUT/'pilot'/str(s)/f'{c:02}.json')['results'][key]['validation'] for s in PILOT for c in range(20)];entries.append({'config':conf,'accuracy':sum(v['acc'] for v in vals)/40,'ce':sum(v['ce'] for v in vals)/40})
  for method in ['M0','M1','M3','M4']:
   arr=[e for e in entries if e['config']['method']==method];best=max(e['accuracy'] for e in arr);ties=[e for e in arr if best-e['accuracy']<1e-8];winner=min(ties,key=lambda e:(e['ce'],e['config']['lr'],-e['config']['gamma']));selected.append(winner['config'])
  save(selection_path,{'rule':'mean of 20-client validation accuracy, equal weight for two pilots; ties <1e-8: lower mean CE, lower lr, larger gamma','candidates':entries,'selected':selected,'source_sha256':{str(s):inputs[str(s)] for s in PILOT},'created_at':time.strftime('%Y-%m-%d %H:%M:%S')});log('selection frozen')
 chosen=load(selection_path)['selected']
 for s in FORMAL:
  for c in range(20):runclient(s,c,True,chosen)
 save(OUT/'progress.json',{'pilot_completed':40,'np_completed':140,'formal_completed':100,'status':'complete'})
 log('all completed')
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();raise
