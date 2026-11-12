import json,time,hashlib
from pathlib import Path
p=Path(__file__).parent
for dataset in ['fashion','kmnist','cifar']:
 files=[p/'pilot'/dataset/f's{s}'/'pilot_validation.json' for s in [9000,9001]]
 while not all(f.exists() for f in files):time.sleep(10)
 data=[json.loads(f.read_text()) for f in files];candidates=[]
 for scope in ['B1','B2','B3']:
  for lr in [.03,.1]:
   for clip in [.5,1.,2.]:
    scores=[next(z for z in run['results'] if (z['scope'],z['lr'],z['clip'])==(scope,lr,clip))['validation_mean'] for run in data]
    candidates.append({'scope':scope,'lr':lr,'clip':clip,'scores':scores,'mean':sum(scores)/2})
 selected={scope:sorted([z for z in candidates if z['scope']==scope],key=lambda z:(-z['mean'],z['lr'],z['clip']))[0] for scope in ['B1','B2','B3']}
 result={'dataset':dataset,'pilot_seeds':[9000,9001],'pilot_alpha':.5,'pilot_epsilon':4,'candidates':candidates,'selected':selected,'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'rule':'highest mean validation accuracy across the two pilot seeds; ties select smaller lr then clip','selection_data':'pilot public validation only, before this protocol formal evaluation'}
 target=p/'pilot'/dataset/'selection_frozen.json'
 if target.exists():assert json.loads(target.read_text())==result
 else:target.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print('FROZEN',dataset,selected,flush=True)
