#!/usr/bin/env python3
"""Reproduce the GWA-T-12 known-wall application.

The raw Bitbrains fastStorage archive is not redistributed. Supply --archive,
or allow the script to download the official archive. The program parses each
VM file, extracts the longest exact five-minute run, takes hourly snapshots
(one observation every 12 records), applies the manuscript screen, fits TEAR
and comparison models, and writes per-series and aggregate outputs.

The script also compares aggregate and focal-series values against the
locked CSV summaries shipped with Online Resource 1. A mismatch is reported
rather than hidden.
"""
from __future__ import annotations
import argparse, io, math, re, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, expit, logit
from scipy.stats import chi2

from run_step3 import fit_tear, mean_trunc_exp

OFFICIAL_URL = "https://atlarge-research.com/gwa-traces/gwa_t_12_fastStorage.zip"
COLS = ["timestamp_ms","cpu_cores","cpu_capacity_mhz","cpu_usage_mhz",
        "cpu_usage_pct","memory_kb","memory_used_kb","disk_read_kbps",
        "disk_write_kbps","net_rx_kbps","net_tx_kbps"]


def download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {OFFICIAL_URL} -> {path}")
    urllib.request.urlretrieve(OFFICIAL_URL, path)


def read_vm_bytes(data: bytes) -> pd.DataFrame:
    text=data.decode('utf-8',errors='replace')
    df=pd.read_csv(io.StringIO(text),sep=r';\s*|\t+',engine='python',header=None,comment='#')
    if df.shape[1] < 5:
        raise ValueError(f"Unexpected VM file format: {df.shape[1]} columns")
    df=df.iloc[:,:min(df.shape[1],len(COLS))]
    df.columns=COLS[:df.shape[1]]
    for c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['timestamp_ms','cpu_usage_pct']).sort_values('timestamp_ms')
    return df


def longest_five_minute_run(df: pd.DataFrame, tolerance_ms: float=1000.0) -> pd.DataFrame:
    if len(df)<2: return df
    ts=df.timestamp_ms.to_numpy(float)
    # The public GWA-T-12 files label this field "Timestamp [ms]", but the
    # current archive stores Unix seconds.  Accept both representations so a
    # genuine five-minute run is not split into single-observation fragments.
    expected_gap=300000.0 if np.nanmedian(np.abs(ts))>1e11 else 300.0
    tolerance=tolerance_ms if expected_gap==300000.0 else tolerance_ms/1000.0
    gap=np.abs(np.diff(ts)-expected_gap)>tolerance
    cuts=np.r_[0,np.where(gap)[0]+1,len(df)]
    lengths=np.diff(cuts); j=int(np.argmax(lengths))
    return df.iloc[cuts[j]:cuts[j+1]].copy()


def hourly_snapshots(run: pd.DataFrame) -> np.ndarray:
    # One fixed-cadence snapshot per hour preserves the conditional sequence
    # without averaging away the upper tail.
    x=run.cpu_usage_pct.to_numpy(float)[::12]
    if np.nanmax(x)>1.5: x=x/100.0
    return x[np.isfinite(x)]


def lag1(x: np.ndarray) -> float:
    if len(x)<3 or np.std(x[:-1])==0 or np.std(x[1:])==0: return float('nan')
    return float(np.corrcoef(x[:-1],x[1:])[0,1])


def passes_screen(x: np.ndarray) -> bool:
    if len(x)<250 or np.any((x<0)|(x>1)): return False
    if np.mean(x==0)>0.05: return False
    if not (0.02<=np.mean(x)<=0.48): return False
    if np.std(x,ddof=1)<0.02: return False
    h=len(x)//2
    if abs(np.mean(x[:h])-np.mean(x[-h:]))>0.10: return False
    return True


def cdf_trunc_exp(y: np.ndarray, rate: np.ndarray, K: float=1.0) -> np.ndarray:
    return -np.expm1(-rate*y)/(-np.expm1(-rate*K))


def independent_fit(x: np.ndarray, K: float=1.0, delta: float=1e-6):
    xp,y=x[:-1],x[1:]
    def nll(q):
        r=float(q)
        ll=np.log(r)-r*y-np.log(-np.expm1(-r*K))
        return -float(np.sum(ll))
    res=minimize_scalar(nll,bounds=(delta,100),method='bounded')
    q=float(res.x)
    return q,float(-res.fun)


def tear_logdensity(xp,y,q,gamma,K=1.0):
    r=q+gamma*(K-xp)
    return np.log(r)-r*y-np.log(-np.expm1(-r*K))


def beta_ar_fit(x: np.ndarray):
    eps=1e-6; xp=np.clip(x[:-1],eps,1-eps); y=np.clip(x[1:],eps,1-eps)
    lx=logit(xp)
    def obj(p):
        a,b,lp=p; phi=math.exp(lp); mu=expit(a+b*lx)
        aa=np.clip(mu*phi,1e-6,None); bb=np.clip((1-mu)*phi,1e-6,None)
        ll=(aa-1)*np.log(y)+(bb-1)*np.log1p(-y)-betaln(aa,bb)
        return -float(np.sum(ll))
    start=np.array([logit(np.clip(np.mean(y),eps,1-eps)),0.2,math.log(20)])
    res=minimize(obj,start,method='L-BFGS-B',bounds=[(-20,20),(-10,10),(-5,10)])
    return res.x, bool(res.success)


def beta_ar_predict(xp: np.ndarray,p):
    eps=1e-6; a,b,lp=p; phi=math.exp(lp); mu=expit(a+b*logit(np.clip(xp,eps,1-eps)))
    aa=np.clip(mu*phi,1e-6,None); bb=np.clip((1-mu)*phi,1e-6,None)
    return mu,aa,bb


def nystrom_stationary(q,gamma,K=1.0,nodes=200):
    z,w=np.polynomial.legendre.leggauss(nodes)
    x=(z+1)*K/2; weights=w*K/2
    r=q+gamma*(K-x)
    # matrix maps density at x_j to density at y_i
    y=x[:,None]; rr=r[None,:]
    P=rr*np.exp(-rr*y)/(-np.expm1(-rr*K))
    A=P*weights[None,:]
    vals,vecs=np.linalg.eig(A)
    j=int(np.argmin(np.abs(vals-1)))
    pi=np.real(vecs[:,j]); pi=np.abs(pi); pi/=np.sum(pi*weights)
    mean=float(np.sum(x*pi*weights)); sd=float(np.sqrt(np.sum((x-mean)**2*pi*weights)))
    wall=float(np.sum((r/np.expm1(r*K))*pi*weights))
    return wall,mean,sd


def fit_series(vm_id,x,delta=1e-6):
    n=len(x); split=max(4,int(math.floor(0.70*n)))
    full=fit_tear(np.r_[x[0],x],1.0,delta=delta)  # fit_tear expects X0,...,Xn
    train=x[:split]; test=x[split:]
    tr=fit_tear(np.r_[train[0],train],1.0,delta=delta)
    q,g=tr['q'],tr['gamma']
    xp=np.r_[train[-1],test[:-1]]; y=test
    ll_tear=tear_logdensity(xp,y,q,g)
    pred_tear=mean_trunc_exp(q+g*(1-xp),1.0)
    qi,_=independent_fit(train)
    ll_ind=tear_logdensity(xp,y,qi,0.0)
    pred_ind=np.full_like(y,mean_trunc_exp(np.array([qi]),1.0)[0])
    bp,bs=beta_ar_fit(train)
    mu,aa,bb=beta_ar_predict(xp,bp)
    yy=np.clip(y,1e-6,1-1e-6)
    ll_beta=(aa-1)*np.log(yy)+(bb-1)*np.log1p(-yy)-np.array([betaln(a,b) for a,b in zip(aa,bb)])
    rfull=full['q']+full['gamma']*(1-x[:-1])
    pit=cdf_trunc_exp(x[1:],rfull,1.0)
    pits=np.sort(pit); u=(np.arange(1,len(pits)+1)-0.5)/len(pits)
    cdf_dist=float(np.max(np.abs(pits-u)))
    q0,ll0=independent_fit(x)
    ll1=-full['fun']; lr=max(0,2*(ll1-ll0)); p_mix=0.5*chi2.sf(lr,1)
    return {
      'vm_id':vm_id,'n':n,'mean':np.mean(x),'sd':np.std(x,ddof=1),'maximum':np.max(x),
      'zero_fraction':np.mean(x==0),'lag1':lag1(x),'half_mean_diff':abs(np.mean(x[:n//2])-np.mean(x[-n//2:])),
      'q_hat':full['q'],'gamma_hat':full['gamma'],'beta1_hat':full['beta1'],
      'q_face':full['q']<=delta*(1+1e-5),'gamma_face':full['gamma']<=1e-8,
      'lr_gamma0':lr,'mixture_p':p_mix,'pit_mean':np.mean(pit),'pit_var':np.var(pit,ddof=1),
      'pit_lag1':lag1(pit),'cdf_distance':cdf_dist,
      'tear_log_density':np.mean(ll_tear),'ind_log_density':np.mean(ll_ind),'beta_log_density':np.mean(ll_beta),
      'tear_mae':np.mean(np.abs(y-pred_tear)),'ind_mae':np.mean(np.abs(y-pred_ind)),'beta_mae':np.mean(np.abs(y-mu)),
      'tear_rmse':np.sqrt(np.mean((y-pred_tear)**2)),'ind_rmse':np.sqrt(np.mean((y-pred_ind)**2)),'beta_rmse':np.sqrt(np.mean((y-mu)**2)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,default=Path('gwa_t_12_fastStorage.zip'))
    ap.add_argument('--out',type=Path,default=Path('../results/application_reproduced'))
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--delta',type=float,default=1e-6)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if not args.archive.exists():
        if not args.download: raise SystemExit('Raw archive missing; pass --download or --archive PATH')
        download_archive(args.archive)
    series={}
    with zipfile.ZipFile(args.archive) as zf:
        for name in zf.namelist():
            if name.endswith('/') or Path(name).name.startswith('.'): continue
            try:
                d=read_vm_bytes(zf.read(name)); run=longest_five_minute_run(d); x=hourly_snapshots(run)
            except Exception as e:
                print(f'SKIP {name}: {e}'); continue
            if passes_screen(x):
                m=re.search(r'(\d+)',Path(name).stem); vm_id=int(m.group(1)) if m else Path(name).stem
                series[vm_id]=x
    print(f'Screened {len(series)} series')
    rows=[]
    for i,(vm,x) in enumerate(sorted(series.items(),key=lambda kv:str(kv[0]))):
        try: rows.append(fit_series(vm,x,args.delta))
        except Exception as e: print(f'FIT FAIL {vm}: {e}')
    per=pd.DataFrame(rows); per.to_csv(args.out/'application_per_series.csv',index=False)
    if per.empty: raise SystemExit('No successful series fits')
    pop=pd.DataFrame([
      ['screened_series',len(per)],
      ['boundary_aware_persistence_p_lt_0_05',int((per.mixture_p<.05).sum())],
      ['positive_heldout_mean_log_density_gain',int((per.tear_log_density>per.ind_log_density).sum())],
      ['median_heldout_mean_log_density_gain',float(np.median(per.tear_log_density-per.ind_log_density))],
      ['full_sample_wall_rate_constraint_active',int(per.q_face.sum())],
      ['cdf_distance_below_1_36_over_sqrt_n_minus_1',int((per.cdf_distance<1.36/np.sqrt(per.n-1)).sum())],
    ],columns=['diagnostic','value'])
    pop.to_csv(args.out/'application_population_summary.csv',index=False)
    eligible=per[(per.lag1>=0)&(per.lag1<=.25)&(per.maximum>=.75)].sort_values(['n','vm_id'],ascending=[False,True])
    comparison_rows=[]
    if len(eligible):
        vm=eligible.iloc[0]; x=series[vm.vm_id]
        wall,mean,sd=nystrom_stationary(vm.q_hat,vm.gamma_hat)
        vmout=vm.to_dict(); vmout.update({'stationary_wall_density':wall,'stationary_mean':mean,'stationary_sd':sd})
        pd.DataFrame([vmout]).to_csv(args.out/'application_focal_series.csv',index=False)
        pred=pd.DataFrame([
          ['Independent truncated exponential',vm.ind_log_density,vm.ind_mae,vm.ind_rmse],
          ['TEAR',vm.tear_log_density,vm.tear_mae,vm.tear_rmse],
          ['Beta autoregression',vm.beta_log_density,vm.beta_mae,vm.beta_rmse],
        ],columns=['model','mean_log_density','MAE','RMSE'])
        pred.to_csv(args.out/'application_predictive_summary.csv',index=False)
        quantities={
          'n':vm.n,'mean':vm['mean'],'standard_deviation':vm.sd,'maximum':vm.maximum,
          'exact_zeros':int(round(vm.zero_fraction*vm.n)),'lag1_correlation':vm.lag1,
          'first_second_half_mean_difference':vm.half_mean_diff,'beta1_hat':vm.beta1_hat,
          'gamma_hat':vm.gamma_hat,'rmin_hat':vm.q_hat,'lr_gamma_zero':vm.lr_gamma0,
          'boundary_mixture_p':vm.mixture_p,'stationary_wall_density':wall,
          'stationary_mean':mean,'stationary_sd':sd,'pit_lag1':vm.pit_lag1,
          'pit_mean':vm.pit_mean,'pit_variance':vm.pit_var,'pit_cdf_distance':vm.cdf_distance,
        }
        pd.DataFrame(list(quantities.items()),columns=['quantity','value']).to_csv(
            args.out/'application_focal_summary.csv',index=False)

    # Machine-readable comparison with locked manuscript summaries. Differences
    # are reported, never silently overwritten.
    locked_root=Path(__file__).resolve().parents[1]/'results'
    generated={
      'application_population_summary.csv':args.out/'application_population_summary.csv',
      'application_predictive_summary.csv':args.out/'application_predictive_summary.csv',
      'application_focal_summary.csv':args.out/'application_focal_summary.csv',
    }
    for name,gpath in generated.items():
        lpath=locked_root/name
        if not (gpath.exists() and lpath.exists()):
            comparison_rows.append([name,'file_present',float(gpath.exists()),float(lpath.exists()),float('nan')])
            continue
        gd=pd.read_csv(gpath); ld=pd.read_csv(lpath)
        if name.startswith('application_population'):
            gm=dict(zip(gd.diagnostic,gd.value)); lm=dict(zip(ld.diagnostic,ld.value))
            for key in sorted(set(gm)|set(lm)):
                gv=float(gm.get(key,float('nan'))); lv=float(lm.get(key,float('nan')))
                comparison_rows.append([name,key,gv,lv,gv-lv])
        elif name.startswith('application_focal'):
            gm=dict(zip(gd.quantity,gd.value)); lm=dict(zip(ld.quantity,ld.value))
            for key in sorted(set(gm)|set(lm)):
                gv=float(gm.get(key,float('nan'))); lv=float(lm.get(key,float('nan')))
                comparison_rows.append([name,key,gv,lv,gv-lv])
        else:
            gm=gd.set_index('model'); lm=ld.set_index('model')
            for model in sorted(set(gm.index)|set(lm.index)):
                for col in ['mean_log_density','MAE','RMSE']:
                    gv=float(gm.loc[model,col]) if model in gm.index else float('nan')
                    lv=float(lm.loc[model,col]) if model in lm.index else float('nan')
                    comparison_rows.append([name,f'{model}:{col}',gv,lv,gv-lv])
    pd.DataFrame(comparison_rows,columns=['file','quantity','generated','locked','difference']).to_csv(
        args.out/'locked_summary_comparison.csv',index=False)
    print(pop.to_string(index=False))
    print(f'Completed. Inspect {args.out / "locked_summary_comparison.csv"} before submission.')

if __name__=='__main__': main()
