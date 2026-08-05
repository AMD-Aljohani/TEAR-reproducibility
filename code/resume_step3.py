#!/usr/bin/env python3
import json, math, platform, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_step3 import *

base=Path(__file__).resolve().parent.parent
out=base/'results'; fig=base/'figures'; out.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
seed0=20260803
# retain already-completed correct-specification outputs
correct=pd.read_csv(out/'correct_spec_raw.csv')
plugin=pd.read_csv(out/'plugin_invariance_summary.csv')
# update stress wall rate after calibrated stress regime q=5,gamma=5
true_rates=json.load(open(out/'true_wall_rates.json'))
c,se=approximate_true_wall_rate(SCENARIOS['stress'],seed0+13,n=800000)
true_rates['stress']={'c':c,'mcse':se}
json.dump(true_rates,open(out/'true_wall_rates.json','w'),indent=2)

plans=[('baseline',1000,300,1),('baseline',10000,250,5),('stress',10000,300,3),('stress',100000,150,10)]
ed=[]
for j,(name,n,reps,fit_every) in enumerate(plans):
    print('endpoint',name,n,reps,flush=True)
    ed.append(run_endpoint_only(SCENARIOS[name],n,reps,seed0+2000+j,true_rates[name]['c'],fit_every=fit_every))
endpoint=pd.concat(ed,ignore_index=True);endpoint.to_csv(out/'endpoint_raw.csv',index=False)
summarize_endpoint(endpoint).to_csv(out/'endpoint_summary.csv',index=False)

idx=pd.concat([run_initial_index_experiment(n,300,seed0+3000+i,SCENARIOS['baseline']) for i,n in enumerate([300,1000])],ignore_index=True)
idx.to_csv(out/'initial_index_raw.csv',index=False)
idx.groupby('n').agg(reps=('rep','count'),exclude_mean=('exclude_X0_gap','mean'),exclude_zero_frequency=('exclude_X0_gap',lambda x:float(np.mean(x==0))),include_mean=('include_X0_gap','mean'),include_zero_frequency=('include_X0_gap',lambda x:float(np.mean(x==0)))).reset_index().to_csv(out/'initial_index_summary.csv',index=False)

print('pseudo true',flush=True)
pseudo=pseudo_true_beta_ar(seed0+4000,n=250000);json.dump(pseudo,open(out/'misspec_pseudo_true.json','w'),indent=2)
md=[]
for i,n in enumerate([500,2000]):
    print('misspec',n,flush=True);md.append(run_misspec(n,120,seed0+4100+i,pseudo))
miss=pd.concat(md,ignore_index=True);miss.to_csv(out/'misspec_raw.csv',index=False);summarize_misspec(miss,pseudo).to_csv(out/'misspec_summary.csv',index=False)

generic=run_generic_rate_experiments(120000,seed0+5000);generic.to_csv(out/'generic_rate_summary.csv',index=False)
opt=run_optimizer_tolerance(SCENARIOS['baseline'],1000,80,seed0+6000);opt.to_csv(out/'optimizer_tolerance_raw.csv',index=False)
pd.DataFrame([{"n":1000,"reps":len(opt),"mean_abs_sqrt_n_delta_beta1":float(np.mean(np.abs(opt.sqrt_n_delta_beta1))),"q95_abs_sqrt_n_delta_beta1":float(np.quantile(np.abs(opt.sqrt_n_delta_beta1),.95)),"mean_abs_sqrt_n_delta_gamma":float(np.mean(np.abs(opt.sqrt_n_delta_gamma))),"q95_abs_sqrt_n_delta_gamma":float(np.quantile(np.abs(opt.sqrt_n_delta_gamma),.95)),"loose_success_rate":float(opt.loose_success.mean()),"tight_success_rate":float(opt.tight_success.mean()),"median_loose_grad":float(opt.loose_grad.median()),"median_tight_grad":float(opt.tight_grad.median())}]).to_csv(out/'optimizer_tolerance_summary.csv',index=False)
make_figures(endpoint,plugin,generic,miss,fig)
meta={"mode":"expanded_locked","seed_base":seed0,"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"endpoint_convention":"max X_1,...,X_n; likelihood X_1->X_2,...,X_{n-1}->X_n","scenarios":{k:asdict(v) for k,v in SCENARIOS.items()},"counts":{"correct_spec_reps_per_cell":160,"endpoint_plans":plans,"misspec_reps_per_cell":120,"generic_reps":120000,"optimizer_reps":80}}
json.dump(meta,open(out/'run_metadata.json','w'),indent=2)
print('done',flush=True)
