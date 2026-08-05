#!/usr/bin/env python3
import json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_step3 import pseudo_true_nonlinear,run_smooth_misspec,summarize_misspec
base=Path(__file__).resolve().parent.parent;out=base/'results'
p=pseudo_true_nonlinear(20266803,n=300000)
json.dump(p,open(out/'smooth_misspec_pseudo_true.json','w'),indent=2)
d=pd.concat([run_smooth_misspec(n,200,20266900+i,p) for i,n in enumerate([500,2000])],ignore_index=True)
d.to_csv(out/'smooth_misspec_raw.csv',index=False)
summarize_misspec(d,p).to_csv(out/'smooth_misspec_summary.csv',index=False)
print(p);print(summarize_misspec(d,p).to_string(index=False))
