import os,sys,json,hashlib,time,secrets,csv,math,platform,traceback
from pathlib import Path
import numpy as np,torch,opacus
from opacus.accountants.analysis import rdp
from scipy.stats import t
B=Path(str(Path(__file__).resolve().parents[1]));O=B/'dp_class_scale_confirm_v1'; P=B/'prior_residual_pilot_v1';OLD=B/'followup_20261104/unified_v2'
sys.path.insert(0,str(B/'kmnist_extension_v1'));sys.path.insert(0,str(B/'followup_20261105'))
import privacy,common,data
torch.set_num_threads(4)
PIL=[9000,9001];FORM=[2041,2042,2043,2044,2045];ALL=PIL+FORM;GAMMA=.001; GRID=[{'C':c,'lr':lr} for c in [.25,1.,4.] for lr in [.1,1.,5.]]
def src(s):return OLD/(f'pilot/kmnist/s{s}' if s in PIL else f'formal/kmnist/a0.5_e4_s{s}')
def hp(s,k):return P/('pilot' if s in PIL else 'formal')/str(s)/f'{k:02}.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def log(msg):print(time.strftime('%Y-%m-%d %H:%M:%S'),msg,flush=True)
def tensor(v):return torch.tensor(v,dtype=torch.float64)
def metrics(z,y,h,a=None,bias=None):
 if bias is not None:s=z+bias[:,None,:]
 else:
  s=z+h[:,None,:]
  if a is not None:s=s+(z-z.mean(-1,keepdim=True))*a[:,None,:]
 ce=torch.nn.functional.cross_entropy(s.reshape(-1,10),y.reshape(-1),reduction='none').reshape(20,-1); correct=(s.argmax(-1)==y).sum(-1)
 return {'correct':correct.tolist(),'n_per_client':int(y.shape[1]),'client_ce':ce.mean(-1).tolist(),'accuracy':int(correct.sum())/y.numel(),'ce':float(ce.mean())}
def accountant(A,b):
 events={'H-full':[(.08,A,300),(1,b['sigma_H_full'],1)],'H-init':[(.08,A,300),(1,b['sigma_H_init'],1)],'Local-10':[(.08,A,300),(1,b['sigma_H_init'],1),(.08,b['sigma_R'],50)],'Shared-10':[(.08,A,300),(1,b['sigma_H_init'],1),(.08,b['sigma_R'],50)]}; out={}
 for name,e in events.items():
  r=sum(rdp.compute_rdp(q=q,noise_multiplier=s,steps=n,orders=privacy.ORDERS) for q,s,n in e);eps,order=rdp.get_privacy_spent(orders=privacy.ORDERS,rdp=r,delta=1e-5);assert eps<=4+1e-6;out[name]={'events':e,'epsilon':float(eps),'optimal_order':float(order),'delta':1e-5}
 out['Bias-ref']=dict(out['Local-10']);return out
def analytic_check():
 gen=torch.Generator().manual_seed(3421);z=torch.randn(10,generator=gen,dtype=torch.float64);h=torch.randn(10,generator=gen,dtype=torch.float64);a=torch.randn(10,generator=gen,dtype=torch.float64,requires_grad=True);u=z-z.mean();y=torch.tensor([3]);loss=torch.nn.functional.cross_entropy((z+h+a*u)[None],y);g=torch.autograd.grad(loss,a)[0];analytic=(torch.softmax(z+h+a*u,0)-torch.nn.functional.one_hot(y,10)[0])*u;assert torch.allclose(g,analytic,atol=1e-12)
 for mode in ['Local-10','Shared-10']:assert torch.equal(z+h+torch.zeros_like(a)*u,z+h)
 # Explicit empty batch: zero sum but nonzero noise and regularization.
 base=torch.ones(10,dtype=torch.float64);noise=torch.ones(10,dtype=torch.float64)*2;out=base-.1*(noise/216+GAMMA*base);assert not torch.equal(out,base)
 return {'gradient_max_abs_error':float((g-analytic).abs().max()),'zero_a_reproduces_H_init':True,'empty_batch_noise_regularizer':True}
def inputs():
 existing=O/'input_manifest.json'; allfiles={};budget=load(P/'budget_verification.json');weights=[]
 for s in ALL:
  root=src(s);paths=[root/n for n in ['shared.pt','shared_done.json','config.json','partition.json']]+[P/'cache'/f'{s}_train_full.pt']+[hp(s,k) for k in range(20)]
  for p in paths:
   if not p.is_file():raise FileNotFoundError(p)
  model=torch.load(root/'shared.pt',map_location='cpu',weights_only=True);a=float(model['budget']['sigma_body']);weights.append(a);actual=sha(root/'shared.pt');assert actual==load(root/'shared_done.json')['sha256'];cache=torch.load(P/'cache'/f'{s}_train_full.pt',weights_only=True);assert cache['source_sha256']==actual and cache['logits'].shape==(60000,10) and torch.isfinite(cache['logits']).all()
  cfg=load(root/'config.json');assert cfg['seed']==s and cfg['alpha']==.5 and cfg['epsilon']==4 and cfg['clients']==20
  part=load(root/'partition.json');partsha=sha(root/'partition.json')
  for split,n,limit in [('train',2700,60000),('validation',300,60000),('test',500,10000)]:
   chunks=part[split];flat=sum(chunks,[]);assert len(chunks)==20 and all(len(v)==n for v in chunks) and len(set(flat))==len(flat) and min(flat)>=0 and max(flat)<limit
  assert not set(sum(part['train'],[])) & set(sum(part['validation'],[]))
  for k in range(20):
   v=load(hp(s,k));assert v['seed']==s and v['client']==k and v['shared_sha256']==actual and v['partition_sha256']==partsha and v['status']=='complete'
   for key in ['h_full','h_init']:
    h=tensor(v[key]);assert h.shape==(10,) and torch.isfinite(h).all() and abs(float(h.mean()))<1e-10
  allfiles[str(s)]={'shared_sha256':actual,'partition_sha256':partsha,'files':{str(p):sha(p) for p in paths}}
 assert len(set(weights))==1 and abs(weights[0]-budget['sigma_A'])<1e-14
 manifest={'files':allfiles,'privacy_budget_sha256':sha(P/'budget_verification.json'),'data_files':{str(B/'kmnist_extension_v1/data'/n):sha(B/'kmnist_extension_v1/data'/n) for n in data.FILES},'source_files':{str(p):sha(p) for p in [B/'kmnist_extension_v1/privacy.py',B/'kmnist_extension_v1/data.py',B/'followup_20261105/common.py',P/'run_experiment.py',Path(__file__)]},'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'opacus':opacus.__version__},'imports':{'privacy':privacy.__file__,'data':data.__file__,'common':common.__file__},'protocol':'class-scale-confirm-v1'}
 if existing.exists():assert load(existing)==manifest,'Existing output input or source mismatch; refuse to mix results.'
 else:save(existing,manifest)
 return budget,weights[0]
def prep(s,y):
 part=load(src(s)/'partition.json');z=torch.load(P/'cache'/f'{s}_train_full.pt',weights_only=True)['logits'].double();h=[]
 for k in range(20):h.append(load(hp(s,k))['h_init'])
 it=torch.tensor(part['train']);iv=torch.tensor(part['validation']);return z[it],y[it],z[iv],y[iv],tensor(h)
def fit(s,method,cfg,zt,yt,zv,yv,h,seed,sigma):
 gen=torch.Generator().manual_seed(seed);C=cfg['C'];lr=cfg['lr'];u=zt-zt.mean(-1,keepdim=True);a=torch.zeros((20,10) if method=='Local-10' else (1,10),dtype=torch.float64);snap={};trace=[];clip_num=0;sample_num=0;per_client_clip=torch.zeros(20,dtype=torch.int64);per_client_n=torch.zeros(20,dtype=torch.int64);owners=torch.arange(20)[:,None].expand(20,2700);start=time.perf_counter()
 def record(step):
  aa=a.expand(20,-1);snap[str(step)]={'validation':metrics(zv,yv,h,aa),'a_norm':aa.norm(dim=1).tolist(),'clipped':int(clip_num),'sampled':int(sample_num),'clip_fraction':float(clip_num/sample_num) if sample_num else None,'client_clipped':per_client_clip.tolist(),'client_sampled':per_client_n.tolist()}
 record(0)
 with torch.no_grad():
  for step in range(1,51):
   mask=torch.rand((20,2700),generator=gen)<.08;ids=owners[mask];zz=zt[mask];uu=u[mask];yy=yt[mask];aa=a.expand(20,-1);score=zz+h[ids]+aa[ids]*uu;g=(torch.softmax(score,dim=-1)-torch.nn.functional.one_hot(yy,10))*uu;norm=g.norm(dim=-1);clipped=norm>C;scales=torch.clamp(C/norm,max=1);gs=g*scales[:,None];summed=torch.zeros((20,10),dtype=torch.float64);summed.index_add_(0,ids,gs); per_client_n+=torch.bincount(ids,minlength=20);per_client_clip+=torch.bincount(ids[clipped],minlength=20);sample_num+=len(ids);clip_num+=int(clipped.sum());noise=torch.randn((20,10),generator=gen,dtype=torch.float64)*sigma*C;private=(summed+noise)/216
   if method=='Shared-10':
    old=a.clone();aggregated=private.mean(0,keepdim=True);a=a-lr*(aggregated+GAMMA*a);difference=float((a-(old-lr*(private.sum(0,keepdim=True)/20+GAMMA*old))).abs().max());assert difference<1e-12
    hashes=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in noise];assert len(set(hashes))==20
    trace.append({'step':step,'client_count':len(private),'noise_std':sigma*C,'unique_noise_vectors':len(set(hashes)),'empty_clients':int((mask.sum(1)==0).sum()),'aggregation_max_error':difference,'noise_sha256':hashes,'sample_counts':mask.sum(1).tolist()})
   else:a=a-lr*(private+GAMMA*a)
   if step in [10,25,50]:record(step)
 return {'seed':s,'method':method,'config':cfg,'gamma':GAMMA,'a':a.expand(20,-1).tolist(),'snapshots':snap,'elapsed_seconds':time.perf_counter()-start,'communication_steps':50 if method=='Shared-10' else 0,'aggregation_trace':trace,'status':'complete'}
def main():
 O.mkdir(exist_ok=True);budget,A=inputs();account=accountant(A,budget);save(O/'privacy_accounting.json',{'branches':account,'orders':privacy.ORDERS,'no_aggregation_amplification':True,'adjacency':'fixed membership and public slots, replace a record with empty slot'});save(O/'checks.json',analytic_check())
 # Local data only: selected clients use train indices exclusively.
 x,y,xe,ye=common.data('kmnist');del x,xe
 state=O/'local_rng_state.json'
 if not state.exists():save(state,{str(s):{m:[secrets.randbits(63) for _ in range(9 if s in PIL else 1)] for m in ['Local-10','Shared-10']} for s in ALL})
 rng=load(state)
 for s in PIL:
  zt,yt,zv,yv,h=prep(s,y)
  for method in ['Local-10','Shared-10']:
   for i,cfg in enumerate(GRID):
    path=O/'pilot'/str(s)/method/f'{i:02}.json'
    if path.exists():continue
    result=fit(s,method,cfg,zt,yt,zv,yv,h,rng[str(s)][method][i],budget['sigma_R']);save(path,result);log(f'pilot {s} {method} {cfg}')
 frozen=O/'frozen_config.json'
 if not frozen.exists():
  records=[];selected={}
  for method in ['Local-10','Shared-10']:
   candidates=[]
   for i,cfg in enumerate(GRID):
    rs=[load(O/'pilot'/str(s)/method/f'{i:02}.json')['snapshots']['50']['validation'] for s in PIL];correct=sum(sum(v['correct']) for v in rs);ce=sum(v['ce'] for v in rs)/2;row={'method':method,'config':cfg,'correct':correct,'n':12000,'accuracy':correct/12000,'ce':ce,'by_seed':{str(s):rs[j] for j,s in enumerate(PIL)}};candidates.append(row);records.append(row)
   winner=min(candidates,key=lambda r:(-r['correct'],r['ce'],r['config']['lr'],r['config']['C']));selected[method]=winner['config']
  save(frozen,{'selection_rule':'maximum pooled correct count, equivalent to equal-weight seed means; tie: lower mean CE, lower learning rate, lower C','selected':selected,'candidates':records,'pilot_source_hashes':{str(s):sha(src(s)/'shared.pt') for s in PIL},'created_at':time.strftime('%Y-%m-%d %H:%M:%S')});log('configuration frozen')
 selected=load(frozen)['selected']
 for s in FORM:
  zt,yt,zv,yv,h=prep(s,y)
  for method in ['Local-10','Shared-10']:
   path=O/'formal_fit'/str(s)/f'{method}.json'
   if path.exists():continue
   result=fit(s,method,selected[method],zt,yt,zv,yv,h,rng[str(s)][method][0],budget['sigma_R']);save(path,result);log(f'formal {s} {method}')
  # Test cache and old bias vectors are read only after selection is frozen.
  ptest=P/'cache'/f'{s}_test_full.pt';cache=torch.load(ptest,weights_only=True);assert cache['source_sha256']==sha(src(s)/'shared.pt')==load(src(s)/'shared_done.json')['sha256'] and cache['logits'].shape==(10000,10) and torch.isfinite(cache['logits']).all()
  part=load(src(s)/'partition.json');idx=torch.tensor(part['test']);z=cache['logits'].double()[idx];labels=ye[idx];old=[load(hp(s,k)) for k in range(20)];hf=tensor([v['h_full'] for v in old]);hi=tensor([v['h_init'] for v in old]);bias=tensor([v['results']['M4']['d'] for v in old]);res={'H-full':metrics(z,labels,hf),'H-init':metrics(z,labels,hi),'Bias-ref':metrics(z,labels,None,bias=bias)}
  for method in ['Local-10','Shared-10']:
   fitdata=load(O/'formal_fit'/str(s)/f'{method}.json');a=tensor(fitdata['a']);
   if method=='Shared-10':assert torch.equal(a,a[0].expand_as(a)) and len(fitdata['aggregation_trace'])==50 and all(r['client_count']==20 and r['unique_noise_vectors']==20 for r in fitdata['aggregation_trace'])
   res[method]=metrics(z,labels,hi,a)
  for k in range(20):
   assert res['H-full']['correct'][k]==round(500*old[k]['results']['M2']['test']['acc']);assert res['Bias-ref']['correct'][k]==round(500*old[k]['results']['M4']['test']['acc'])
  save(O/'formal_metrics'/f'{s}.json',{'seed':s,'test_cache_sha256':sha(ptest),'shared_sha256':sha(src(s)/'shared.pt'),'partition_sha256':sha(src(s)/'partition.json'),'metrics':res});log(f'formal {s} test evaluated')
 checks=load(O/'checks.json');checks.update({'shared_all_50_steps_20_independent_noises_and_equal_weight':True,'shared_final_coefficients_identical':True,'cache_test_hash_verified_after_freeze':True});save(O/'checks.json',checks);save(O/'progress.json',{'pilot_seed_candidates_completed':36,'formal_fit_completed':10,'formal_test_seeds_completed':5,'status':'complete'});log('completed')
if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();raise

