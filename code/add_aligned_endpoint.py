#!/usr/bin/env python3
import sys,json
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_step3 import Scenario,approximate_true_wall_rate,run_endpoint_only,summarize_endpoint
base=Path(__file__).resolve().parent.parent;out=base/'results'
s=Scenario('endpoint_baseline',1.0,2.5,2.5)
c,se=approximate_true_wall_rate(s,20268001,n=800000)
d=run_endpoint_only(s,10000,300,20268002,c,fit_every=2)
d.to_csv(out/'endpoint_aligned_baseline_raw.csv',index=False)
summarize_endpoint(d).to_csv(out/'endpoint_aligned_baseline_summary.csv',index=False)
json.dump({'c':c,'mcse':se,'scenario':{'K':1.0,'q':2.5,'gamma':2.5,'beta1':5.0}},open(out/'endpoint_aligned_baseline_rate.json','w'),indent=2)
print(c,se);print(summarize_endpoint(d).to_string(index=False))
