#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_step3 import (trunc_exp_draw, nll_grad_qg, fit_tear, simulate_tear,
                       mean_trunc_exp, var_trunc_exp, wall_density_conditional,
                       estimate_wall_rate)
from build_v5_endpoint_bounds import phi, g, build_bounds_table, build_wall_brackets
from reproduce_application import longest_five_minute_run


def main():
    rng=np.random.default_rng(123)
    passed=0
    # 1. inverse CDF support and empirical mean
    r=3.2; K=1.0
    vals=np.array([trunc_exp_draw(r,K,float(u)) for u in rng.random(200000)])
    assert np.all((vals>=0)&(vals<=K))
    assert abs(vals.mean()-mean_trunc_exp(np.array([r]),K)[0])<0.002
    passed+=1
    # 2. variance formula
    assert abs(vals.var()-var_trunc_exp(np.array([r]),K)[0])<0.002
    passed+=1
    # 3. objective gradient
    x=simulate_tear(500,1.0,4.0,1.0,rng)
    xp,y=x[1:-1],x[2:]
    p=np.array([1.2,3.7])
    _,grad=nll_grad_qg(p,xp,y,1.0)
    ng=approx_derivative(lambda z: np.array([nll_grad_qg(z,xp,y,1.0)[0]]),p,method='3-point').ravel()
    assert np.max(np.abs(grad-ng))<1e-4, (grad,ng)
    passed+=1
    # 4. parameter recovery smoke test
    fit=fit_tear(x,1.0)
    assert fit['success'] and fit['q']>0 and fit['gamma']>=0
    passed+=1
    # 5. wall density identity finite and positive
    rates=np.array([0.1,1,10.0]); wd=wall_density_conditional(rates,1.0)
    assert np.all(np.isfinite(wd)&(wd>0))
    passed+=1
    # 6. post-initial endpoint convention prevents an arbitrary X0 wall atom
    xwall=simulate_tear(300,1.0,4.0,1.0,rng,x0=1.0)
    assert np.max(xwall)==1.0 and np.max(xwall[1:])<1.0
    passed+=1
    # 7. fitted wall-density plug-in is finite
    Khat=float(np.max(x[1:])); fit_k=fit_tear(x,Khat)
    c=estimate_wall_rate(x,fit_k,Khat)
    assert np.isfinite(c) and c>0
    passed+=1
    # 8. boundary flag is not equivalent to optimiser failure
    xb=simulate_tear(300,1.5,0.75,1.0,rng)
    fb=fit_tear(xb,1.0,delta=0.02)
    assert fb['success'] and fb['q']>=0.02 and fb['gamma']>=0
    passed+=1
    # 9. exact upper-strip probability decreases in the rate and dominates its linear minorant
    grid=np.linspace(0.1,12.0,100)
    vals_phi=np.array([phi(float(rr),0.2,1.0) for rr in grid])
    assert np.all(np.diff(vals_phi)<0)
    assert all(phi(float(rr),0.2,1.0) >= 0.2*g(float(rr),1.0) for rr in grid)
    passed+=1
    # 10. locked simulations satisfy the exact tail sandwich and wall-density bracket
    bt=build_bounds_table(); wb=build_wall_brackets()
    assert np.all(bt['lower_bound'] <= bt['empirical_survival'] + 1e-15)
    assert np.all(bt['empirical_survival'] <= bt['upper_bound'] + 1e-15)
    assert np.all(wb['g_beta1'] <= wb['pi_K'] + 1e-12)
    assert np.all(wb['pi_K'] <= wb['g_q'] + 1e-12)
    passed+=1
    # 11. the public archive's mislabeled Unix-second timestamps and genuine
    # millisecond timestamps both preserve a complete five-minute run
    import pandas as pd
    for origin,step in ((1.37e9,300.0),(1.37e12,300000.0)):
        frame=pd.DataFrame({'timestamp_ms':origin+np.arange(8)*step})
        assert len(longest_five_minute_run(frame))==8
    passed+=1
    print(f'{passed}/11 tests passed')

if __name__=='__main__': main()
