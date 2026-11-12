import sys,os,json,time,hashlib,secrets,csv,math,traceback
from pathlib import Path
import numpy as np,torch,opacus
from scipy.special import ndtr
B=Path(str(Path(__file__).resolve().parents[1]));O=B/'bias_theory_check_v1';P=B/'prior_residual_pilot_v1';R=B/'followup_20261104/unified_v2/pilot/kmnist/s9000'
sys.path.insert(0,str(B/'kmnist_extension_v1'));sys.path.insert(0,str(B/'followup_20261105'));import common,data,privacy
torch.set_num_threads(4);torch.set_default_dtype(torch.float64)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,d):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(t,p)
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def writecsv(name,rows):
 with (O/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def stats(z,y,d,C,sigma):
 scores=z+d;lp=torch.log_softmax(scores,dim=1);loss=-lp[torch.arange(len(y)),y].mean();g=lp.exp()-torch.nn.functional.one_hot(y,10);norm=g.norm(dim=1);c=g*torch.clamp(C/norm.clamp_min(1e-30),max=1)[:,None];mu=c.mean(0);beta=g.mean(0)-mu;m2=mu.square().sum()+(.92/(.08*2700**2))*c.square().sum()+10*(sigma*C/216)**2
 return loss,g,c,mu,beta,m2,(norm>C).double().mean()
def main():
 O.mkdir(exist_ok=True); refs=[R/'shared.pt',R/'partition.json',R/'shared_done.json',R/'config.json',P/'cache/9000_train_full.pt']+[P/'np/9000'/f'{k:02}.json' for k in [0,7,14]]+[B/'kmnist_extension_v1'/x for x in ['data.py','privacy.py','engine.py']]+[B/'followup_20261105/common.py',Path(__file__)]+[B/'kmnist_extension_v1/data'/name for name in ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz']]
 for p in refs:
  if not p.is_file():raise FileNotFoundError(p)
 manifest={str(p):sha(p) for p in refs}; mp=O/'source_hashes.json'
 if mp.exists():assert load(mp)==manifest,'Input or script hash mismatch.'
 else:save(mp,manifest)
 sh=torch.load(R/'shared.pt',map_location='cpu',weights_only=True);sigmaA=sh['budget']['sigma_body'];sigma=sh['budget']['sigma_head'];eps=privacy.epsilon([(.08,sigmaA,300),(.08,sigma,50)],1e-5);assert eps<=4+1e-6;chk=sha(R/'shared.pt');assert chk==load(R/'shared_done.json')['sha256'];cache=torch.load(P/'cache/9000_train_full.pt',weights_only=True);assert cache['source_sha256']==chk and cache['logits'].shape==(60000,10);part=load(R/'partition.json');config=load(R/'config.json');assert config['seed']==9000 and config['epsilon']==4 and config['alpha']==.5
 for split,n,lim in [('train',2700,60000),('validation',300,60000),('test',500,10000)]:
  a=part[split];f=sum(a,[]);assert len(a)==20 and all(len(c)==n for c in a) and len(set(f))==len(f) and min(f)>=0 and max(f)<lim
 assert not set(sum(part['train'],[]))&set(sum(part['validation'],[]))
 model,_=common.model(R,'kmnist');model.double();model.requires_grad_(False);model.eval()
 def whash():return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
 before=whash();x,y=data.load(B/'kmnist_extension_v1/data',True);zall=cache['logits'].double(); ids=sum([part['validation'][k][:30] for k in [0,7,14]],[])
 with torch.no_grad():feat=model.body(x[ids].double());zcalc=model.head(feat)
 maxerr=float((zcalc-zall[ids]).abs().max());assert maxerr<1e-5 and before==whash();
 zz=zall[part['train'][0]][:8];yy=y[part['train'][0]][:8];d=torch.randn(10,requires_grad=True);a=torch.autograd.grad(torch.nn.functional.cross_entropy(zz+d,yy),d)[0];g=(torch.softmax(zz+d,1)-torch.nn.functional.one_hot(yy,10)).mean(0);graderr=float((a-g).abs().max());assert graderr<1e-12
 settings={'seed':9000,'clients':[0,7,14],'dtype':'float64','threads':4,'n0':2700,'q':.08,'expected_batch':216,'sigma_A':sigmaA,'sigma_B':sigma,'joint_epsilon':eps,'delta':1e-5,'eta':.03,'A_monte_carlo':20000,'stress_kappa':.05,'stress_trigger':'nominal total reversals <0.1% over 90 images times 20000 draws for either method','B_mc_draws':2000,'trajectories_initial':40,'trajectory_max':200,'S':[10,25,50],'head_vs_cache_max_error':maxerr,'autograd_max_error':graderr,'frozen_weight_sha256':before,'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'opacus':opacus.__version__}}
 if (O/'settings.json').exists():assert load(O/'settings.json')==settings
 else:save(O/'settings.json',settings)
 state=O/'local_rng_state.json'
 if not state.exists():save(state,{'A':{f'{level}_{i}':secrets.randbits(63) for level in ['nominal','stress'] for i in range(90)},'B_moment':{f'{k}_{C}':secrets.randbits(63) for k in [0,7,14] for C in [1,2]},'B_trajectory':{f'{k}_{C}_{i}':secrets.randbits(63) for k in [0,7,14] for C in [1,2] for i in range(200)}})
 rngs=load(state);kappa=.03*sigma*2/216
 def a_compute(level,kap):
  for i in range(90):
   p=O/'A'/level/f'{i:03}.json'
   if p.exists():continue
   gen=torch.Generator().manual_seed(rngs['A'][f'{level}_{i}']);v=feat[i];z=zall[ids[i]];ci=int(z.argmax());j=int(torch.argsort(z,descending=True)[1]);m=float(z[ci]-z[j]);scale=math.sqrt(float(v.square().sum())+1);values={}
   for method in ['bias','full']:
    each=[];pair=[]
    for st in range(20):
     if method=='bias':out=torch.randn((1000,10),generator=gen)*kap
     else:
      w=torch.randn((1000,10,64),generator=gen)*kap;b=torch.randn((1000,10),generator=gen)*kap;out=torch.einsum('bch,h->bc',w,v)+b
     each.append(out[:,ci]);pair.append(out[:,ci]-out[:,j])
    each=torch.cat(each);pair=torch.cat(pair);s=1 if method=='bias' else scale;tp=float(ndtr(-m/(math.sqrt(2)*kap*s)));count=int((pair < -m).sum());pobs=count/20000
    values[method]={'single_var':float(each.var(unbiased=True)),'single_theory':(kap*s)**2,'pair_var':float(pair.var(unbiased=True)),'pair_theory':2*(kap*s)**2,'reversals':count,'p_observed':pobs,'p_theory':tp,'mc_se_empirical':math.sqrt(pobs*(1-pobs)/20000),'mc_se_theory':math.sqrt(tp*(1-tp)/20000),'expected_events':20000*tp}
   save(p,{'i':i,'client':[0,7,14][i//30],'validation_index':ids[i],'v_norm':float(v.norm()),'c':ci,'j':j,'margin':m,'kappa':kap,'values':values});
   if (i+1)%30==0:log(f'A {level}: {i+1}/90')
 a_compute('nominal',kappa)
 nominal=[load(p) for p in sorted((O/'A/nominal').glob('*.json'))];stress=any(sum(r['values'][method]['reversals'] for r in nominal)/(90*20000)<.001 for method in ['bias','full'])
 if stress:a_compute('stress',.05)
 arows=[]
 for level in (['nominal','stress'] if stress else ['nominal']):
  for p in sorted((O/'A'/level).glob('*.json')):
   r=load(p)
   for method in ['bias','full']:arows.append({'level':level,'client':r['client'],'validation_index':r['validation_index'],'feature_norm':r['v_norm'],'c':r['c'],'j':r['j'],'margin':r['margin'],'kappa':r['kappa'],'method':method,**r['values'][method]})
 writecsv('A_image_noise.csv',arows)
 for k in [0,7,14]:
  z=zall[part['train'][k]];yy=y[part['train'][k]];dref=torch.tensor(load(P/'np/9000'/f'{k:02}.json')['d']);zv=zall[part['validation'][k]];yv=y[part['validation'][k]]
  for C in [1,2]:
   path=O/'B_moment'/f'{k}_{C}.json'
   if not path.exists():
    d=torch.zeros(10);loss,g,c,mu,beta,m2,clipped=stats(z,yy,d,C,sigma);gen=torch.Generator().manual_seed(rngs['B_moment'][f'{k}_{C}']);samples=[]
    for _ in range(20):
     masks=(torch.rand((100,2700),generator=gen)<.08).double();samples.append((masks@c+torch.randn((100,10),generator=gen)*sigma*C)/216)
    vals=torch.cat(samples);mean=vals.mean(0);vals2=vals.square().sum(1);covdiag=.92/(.08*2700**2)*c.square().sum(0)+(sigma*C/216)**2;se=(covdiag/2000).sqrt()
    save(path,{'client':k,'C':C,'mean_theory':mu.tolist(),'mean_empirical':mean.tolist(),'mean_error_l2':float((mean-mu).norm()),'mean_error_coordinate':(mean-mu).tolist(),'mean_theory_se_coordinate':se.tolist(),'mean_error_z_max':float(((mean-mu).abs()/se).max()),'mean_error_rms_theory':float(se.square().sum().sqrt()),'m2_theory':float(m2),'m2_empirical':float(vals2.mean()),'m2_empirical_se':float(vals2.std(unbiased=True)/math.sqrt(2000)),'m2_relative_error':float(vals2.mean()/m2-1),'initial_clip_fraction':float(clipped),'beta_norm':float(beta.norm())})
   def runtraj(i):
    p=O/'trajectories'/str(k)/str(C)/f'{i:03}.json'
    if p.exists():return
    gen=torch.Generator().manual_seed(rngs['B_trajectory'][f'{k}_{C}_{i}']);d=torch.zeros(10);dist0=float(dref.square().sum());running=[];out=[]
    for step in range(50):
     loss,g,c,mu,beta,m2,clip=stats(z,yy,d,C,sigma);dot=float(beta@(d-dref));ref_loss=float(torch.nn.functional.cross_entropy(z+dref,yy));d2=float((d-dref).square().sum());mask=torch.rand(2700,generator=gen)<.08;G=(c[mask].sum(0)+torch.randn(10,generator=gen)*sigma*C)/216;running.append({'t':step,'loss':float(loss),'gap':float(loss)-ref_loss,'m2':float(m2),'beta_dot':dot,'beta_norm':float(beta.norm()),'clip_fraction':float(clip),'distance_sq':d2,'G':G.tolist(),'G_sq':float(G.square().sum()),'sampled':int(mask.sum())});d=d-.03*G
     if step+1 in [10,25,50]:
      S=step+1;lhs=np.mean([r['gap'] for r in running]);middle=.03/2*np.mean([r['m2'] for r in running]);bias=np.mean([r['beta_dot'] for r in running]);initial=dist0/(2*.03*S);terminal=float((d-dref).square().sum())/(2*.03*S);out.append({'S':S,'lhs':lhs,'initial':initial,'m2_term':middle,'bias_term':bias,'terminal':terminal,'bound':initial+middle+bias,'tight_bound':initial+middle+bias-terminal,'slack':initial+middle+bias-lhs,'tight_slack':initial+middle+bias-terminal-lhs})
    vs=zv+d;val={'accuracy':float((vs.argmax(1)==yv).double().mean()),'ce':float(torch.nn.functional.cross_entropy(vs,yv))};save(p,{'client':k,'C':C,'trajectory':i,'steps':running,'summary':out,'val':val,'d_final':d.tolist()})
   for i in range(40):runtraj(i)
   # Additional trajectories only when the mean tight slack is dominated by MC error.
   records=[load(p) for p in sorted((O/'trajectories'/str(k)/str(C)).glob('*.json'))]
   unstable=False
   for index in range(3):
    slack=np.array([v['summary'][index]['tight_slack'] for v in records]);se=slack.std(ddof=1)/math.sqrt(len(slack));unstable |= abs(slack.mean())<=2*se
   if unstable:
    for i in range(40,200):runtraj(i)
   log(f'B client={k} C={C} trajectories={len(list((O/"trajectories"/str(k)/str(C)).glob("*.json")))}')
 assert before==whash();save(O/'progress.json',{'A_rows':len(arows),'stress_executed':stress,'B_moments':6,'B_trajectories':len(list((O/'trajectories').glob('*/*/*.json'))),'complete':True});log('complete')
if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();raise

