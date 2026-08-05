#!/usr/bin/env python3
"""Expanded TEAR simulation campaign for Step 3.

The script is self-contained and produces machine-readable CSV/JSON outputs and
publication-ready figures. It uses the endpoint convention
    Khat_n = max(X_1, ..., X_n)
and the trimmed conditional likelihood based on X_t -> X_{t+1}, t=1,...,n-1.

Run:
    python run_step3.py --mode medium --out ../results --fig ../figures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist
from scipy.stats import kstest
import matplotlib.pyplot as plt

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Scenario:
    name: str
    K: float
    q: float
    gamma: float
    x0_mode: str = "stationary"

    @property
    def beta1(self) -> float:
        return self.q + self.gamma * self.K


SCENARIOS = {
    "low_persistence": Scenario("low_persistence", K=1.0, q=1.5, gamma=0.75),
    "baseline": Scenario("baseline", K=1.0, q=1.0, gamma=4.0),
    "near_active": Scenario("near_active", K=1.0, q=0.15, gamma=4.0),
    "stress": Scenario("stress", K=1.0, q=5.0, gamma=5.0),
}


def trunc_exp_draw(rate: float, K: float, u: float) -> float:
    # F^{-1}(u) = -log(1-u(1-exp(-rK)))/r, evaluated stably.
    a = -math.expm1(-rate * K)
    return -math.log1p(-u * a) / rate


def simulate_tear(
    n: int,
    q: float,
    gamma: float,
    K: float,
    rng: np.random.Generator,
    burn: int = 800,
    x0: float | None = None,
) -> Array:
    """Return X_0,...,X_n. Burn-in is used when x0 is None."""
    if n < 2:
        raise ValueError("n must be at least 2")
    total = n + 1 + (burn if x0 is None else 0)
    x = np.empty(total, dtype=float)
    x[0] = K / 2 if x0 is None else float(x0)
    u = rng.random(total - 1)
    for t in range(total - 1):
        rate = q + gamma * (K - x[t])
        if rate <= 0:
            raise FloatingPointError("non-positive TEAR rate")
        x[t + 1] = trunc_exp_draw(rate, K, float(u[t]))
    return x[-(n + 1):]


def mean_trunc_exp(rate: Array, K: float) -> Array:
    z = rate * K
    return 1.0 / rate - K / np.expm1(z)


def var_trunc_exp(rate: Array, K: float) -> Array:
    z = rate * K
    ez = np.exp(z)
    denom = np.expm1(z)
    v = 1.0 / (rate * rate) - (K * K) * ez / (denom * denom)
    return np.maximum(v, 1e-14)


def wall_density_conditional(rate: Array, K: float) -> Array:
    return rate / np.expm1(rate * K)


def nll_grad_qg(params: Array, xp: Array, y: Array, K: float) -> tuple[float, Array]:
    q, gamma = float(params[0]), float(params[1])
    d = K - xp
    rate = q + gamma * d
    if q <= 0 or gamma < 0 or np.any(rate <= 0) or np.max(y) > K + 1e-12:
        return 1e100, np.array([0.0, 0.0])
    z = rate * K
    # log(1-exp(-z)) = log(-expm1(-z))
    ll = np.log(rate) - rate * y - np.log(-np.expm1(-z))
    score_r = mean_trunc_exp(rate, K) - y
    grad_ll = np.array([np.sum(score_r), np.dot(score_r, d)])
    return -float(np.sum(ll)), -grad_ll


def observed_info_qg(params: Array, xp: Array, K: float) -> Array:
    q, gamma = float(params[0]), float(params[1])
    d = K - xp
    rate = q + gamma * d
    v = var_trunc_exp(rate, K)
    return np.array([
        [np.sum(v), np.dot(v, d)],
        [np.dot(v, d), np.dot(v, d * d)],
    ])


def score_rows_qg(params: Array, xp: Array, y: Array, K: float) -> Array:
    q, gamma = float(params[0]), float(params[1])
    d = K - xp
    rate = q + gamma * d
    sr = mean_trunc_exp(rate, K) - y
    return np.column_stack([sr, sr * d])


def fit_tear(
    x: Array,
    K_fit: float,
    delta: float = 0.02,
    gtol: float = 1e-9,
    maxiter: int = 500,
) -> dict:
    """Fit on trimmed transitions X_1->X_2,...,X_{n-1}->X_n."""
    if len(x) < 4:
        raise ValueError("trajectory too short")
    xp, y = x[1:-1], x[2:]
    if np.max(y) > K_fit + 1e-10 or np.max(xp) > K_fit + 1e-10:
        raise ValueError("K_fit below an observation used in likelihood")
    # Stable, data-adaptive starts in q,gamma coordinates.
    ybar = float(np.clip(np.mean(y), 1e-4, max(K_fit - 1e-4, 1e-4)))
    r0 = max(delta * 1.2, min(30.0, 1.0 / ybar))
    starts = [
        np.array([max(delta * 1.2, min(r0, 10.0)), 0.2]),
        np.array([1.0, 2.0]),
        np.array([0.2, 5.0]),
    ]
    best = None
    bounds = [(delta, 100.0), (0.0, 200.0)]
    for st in starts:
        res = minimize(
            lambda p: nll_grad_qg(p, xp, y, K_fit),
            st,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": gtol, "maxiter": maxiter, "maxls": 50},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    qhat, ghat = map(float, best.x)
    bhat = qhat + ghat * K_fit
    info = observed_info_qg(best.x, xp, K_fit)
    try:
        cov_qg = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov_qg = np.full((2, 2), np.nan)
    J = np.array([[1.0, K_fit], [0.0, 1.0]])
    cov_bg = J @ cov_qg @ J.T
    grad_norm = float(np.linalg.norm(best.jac, ord=np.inf))
    return {
        "success": bool(best.success),
        "message": str(best.message),
        "q": qhat,
        "gamma": ghat,
        "beta1": bhat,
        "fun": float(best.fun),
        "nit": int(best.nit),
        "grad_norm": grad_norm,
        "cov_bg": cov_bg,
        "cov_qg": cov_qg,
        "xp": xp,
        "y": y,
    }


def hac_cov_beta_gamma(fit: dict, K_fit: float, bandwidth: int | None = None) -> Array:
    p = np.array([fit["q"], fit["gamma"]])
    xp, y = fit["xp"], fit["y"]
    scores = score_rows_qg(p, xp, y, K_fit)
    scores = scores - np.mean(scores, axis=0, keepdims=True)
    m = len(scores)
    if bandwidth is None:
        bandwidth = max(1, int(math.floor(4.0 * (m / 100.0) ** (2.0 / 9.0))))
    S = scores.T @ scores
    for lag in range(1, min(bandwidth, m - 1) + 1):
        w = 1.0 - lag / (bandwidth + 1.0)
        G = scores[lag:].T @ scores[:-lag]
        S += w * (G + G.T)
    H = observed_info_qg(p, xp, K_fit)
    try:
        Hinv = np.linalg.inv(H)
        cov_qg = Hinv @ S @ Hinv
    except np.linalg.LinAlgError:
        return np.full((2, 2), np.nan)
    J = np.array([[1.0, K_fit], [0.0, 1.0]])
    return J @ cov_qg @ J.T


def estimate_wall_rate(x: Array, fit: dict, K_fit: float) -> float:
    xp = x[1:-1]
    rate = fit["q"] + fit["gamma"] * (K_fit - xp)
    return float(np.mean(wall_density_conditional(rate, K_fit)))


def approximate_true_wall_rate(s: Scenario, seed: int, n: int = 800_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = simulate_tear(n=n, q=s.q, gamma=s.gamma, K=s.K, rng=rng, burn=3000)
    xp = x[1:-1]
    vals = wall_density_conditional(s.q + s.gamma * (s.K - xp), s.K)
    # batch-means Monte Carlo standard error
    b = 200
    k = len(vals) // b
    bm = vals[: b * k].reshape(b, k).mean(axis=1)
    return float(np.mean(vals)), float(np.std(bm, ddof=1) / math.sqrt(b))


def coverage(est: float, se: float, truth: float, z: float = 1.959963984540054) -> bool:
    return bool(np.isfinite(se) and est - z * se <= truth <= est + z * se)


def run_correct_spec(
    scenario: Scenario,
    n: int,
    reps: int,
    seed: int,
    delta: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for rep in range(reps):
        x = simulate_tear(n, scenario.q, scenario.gamma, scenario.K, rng)
        Khat = float(np.max(x[1:]))
        fk = fit_tear(x, scenario.K, delta=delta)
        fe = fit_tear(x, Khat, delta=delta)
        c_hat = estimate_wall_rate(x, fe, Khat)
        for label, fit, Kuse in [("known", fk, scenario.K), ("estimated", fe, Khat)]:
            cov = fit["cov_bg"]
            se_b = math.sqrt(cov[0, 0]) if np.isfinite(cov[0, 0]) and cov[0, 0] >= 0 else math.nan
            se_g = math.sqrt(cov[1, 1]) if np.isfinite(cov[1, 1]) and cov[1, 1] >= 0 else math.nan
            rows.append({
                "scenario": scenario.name,
                "n": n,
                "rep": rep,
                "wall": label,
                "Khat": Khat,
                "Kuse": Kuse,
                "c_hat": c_hat,
                "beta1_hat": fit["beta1"],
                "gamma_hat": fit["gamma"],
                "q_hat": fit["q"],
                "se_beta1": se_b,
                "se_gamma": se_g,
                "cover_beta1": coverage(fit["beta1"], se_b, scenario.beta1),
                "cover_gamma": coverage(fit["gamma"], se_g, scenario.gamma),
                "success": fit["success"],
                "grad_norm": fit["grad_norm"],
                "nit": fit["nit"],
                "endpoint_gap_scaled": n * (scenario.K - Khat),
                "endpoint_upper95": Khat + (-math.log(0.05)) / (n * c_hat),
                "endpoint_median_corrected": Khat + math.log(2.0) / (n * c_hat),
            })
    return pd.DataFrame(rows)


def summarize_correct(df: pd.DataFrame, truths: dict[str, Scenario]) -> pd.DataFrame:
    out = []
    for (sc, n, wall), g in df.groupby(["scenario", "n", "wall"]):
        s = truths[sc]
        out.append({
            "scenario": sc,
            "n": int(n),
            "wall": wall,
            "reps": int(len(g)),
            "success_rate": float(g.success.mean()),
            "beta1_bias": float((g.beta1_hat - s.beta1).mean()),
            "beta1_rmse": float(np.sqrt(np.mean((g.beta1_hat - s.beta1) ** 2))),
            "beta1_coverage": float(g.cover_beta1.mean()),
            "gamma_bias": float((g.gamma_hat - s.gamma).mean()),
            "gamma_rmse": float(np.sqrt(np.mean((g.gamma_hat - s.gamma) ** 2))),
            "gamma_coverage": float(g.cover_gamma.mean()),
            "median_grad_norm": float(g.grad_norm.median()),
        })
    return pd.DataFrame(out).sort_values(["scenario", "n", "wall"])


def summarize_plugin(df: pd.DataFrame) -> pd.DataFrame:
    k = df[df.wall == "known"].set_index(["scenario", "n", "rep"])
    e = df[df.wall == "estimated"].set_index(["scenario", "n", "rep"])
    j = k[["beta1_hat", "gamma_hat"]].join(
        e[["beta1_hat", "gamma_hat"]], lsuffix="_known", rsuffix="_estimated"
    ).reset_index()
    j["sqrt_n_delta_beta1"] = np.sqrt(j.n) * (j.beta1_hat_estimated - j.beta1_hat_known)
    j["sqrt_n_delta_gamma"] = np.sqrt(j.n) * (j.gamma_hat_estimated - j.gamma_hat_known)
    return j.groupby(["scenario", "n"]).agg(
        reps=("rep", "count"),
        mean_abs_sqrt_n_delta_beta1=("sqrt_n_delta_beta1", lambda x: float(np.mean(np.abs(x)))),
        q95_abs_sqrt_n_delta_beta1=("sqrt_n_delta_beta1", lambda x: float(np.quantile(np.abs(x), 0.95))),
        mean_abs_sqrt_n_delta_gamma=("sqrt_n_delta_gamma", lambda x: float(np.mean(np.abs(x)))),
        q95_abs_sqrt_n_delta_gamma=("sqrt_n_delta_gamma", lambda x: float(np.quantile(np.abs(x), 0.95))),
    ).reset_index()


def run_endpoint_only(
    scenario: Scenario,
    n: int,
    reps: int,
    seed: int,
    true_c: float,
    fit_every: int = 1,
    delta: float = 0.02,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        x = simulate_tear(n, scenario.q, scenario.gamma, scenario.K, rng)
        Khat = float(np.max(x[1:]))
        c_oracle = float(np.mean(wall_density_conditional(
            scenario.q + scenario.gamma * (scenario.K - x[1:-1]), scenario.K
        )))
        c_feasible = math.nan
        if rep % fit_every == 0:
            fe = fit_tear(x, Khat, delta=delta)
            c_feasible = estimate_wall_rate(x, fe, Khat)
        rows.append({
            "scenario": scenario.name,
            "n": n,
            "rep": rep,
            "Khat": Khat,
            "Z": n * (scenario.K - Khat),
            "scaled_exp": true_c * n * (scenario.K - Khat),
            "c_oracle": c_oracle,
            "c_feasible": c_feasible,
            "cover_oracle95": Khat + (-math.log(0.05))/(n*c_oracle) >= scenario.K,
            "median_above_oracle": Khat + math.log(2)/(n*c_oracle) >= scenario.K,
            "cover_feasible95": (
                Khat + (-math.log(0.05))/(n*c_feasible) >= scenario.K
                if np.isfinite(c_feasible) else np.nan
            ),
            "median_above_feasible": (
                Khat + math.log(2)/(n*c_feasible) >= scenario.K
                if np.isfinite(c_feasible) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def summarize_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    out=[]
    for (sc,n),g in df.groupby(["scenario","n"]):
        ks = kstest(g.scaled_exp, "expon")
        gf = g.dropna(subset=["c_feasible"])
        out.append({
            "scenario":sc,"n":int(n),"reps":len(g),
            "mean_scaled_exp":float(g.scaled_exp.mean()),
            "median_scaled_exp":float(g.scaled_exp.median()),
            "ks_D_vs_Exp1":float(ks.statistic),"ks_p":float(ks.pvalue),
            "oracle_95_coverage":float(g.cover_oracle95.mean()),
            "oracle_median_frequency":float(g.median_above_oracle.mean()),
            "feasible_reps":len(gf),
            "feasible_95_coverage":float(gf.cover_feasible95.mean()) if len(gf) else math.nan,
            "feasible_median_frequency":float(gf.median_above_feasible.mean()) if len(gf) else math.nan,
            "mean_c_oracle":float(g.c_oracle.mean()),
            "mean_c_feasible":float(gf.c_feasible.mean()) if len(gf) else math.nan,
        })
    return pd.DataFrame(out).sort_values(["scenario","n"])


def run_initial_index_experiment(n: int, reps: int, seed: int, scenario: Scenario) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    rows=[]
    for rep in range(reps):
        x=simulate_tear(n,scenario.q,scenario.gamma,scenario.K,rng,burn=0,x0=scenario.K)
        mex=float(np.max(x[1:]))
        minc=float(np.max(x))
        rows.append({"n":n,"rep":rep,"exclude_X0_gap":n*(scenario.K-mex),
                     "include_X0_gap":n*(scenario.K-minc)})
    return pd.DataFrame(rows)


def simulate_beta_ar(n:int,rng:np.random.Generator,a:float=0.0,b:float=0.7,phi:float=18.0,burn:int=1000)->Array:
    total=n+1+burn
    x=np.empty(total)
    x[0]=0.5
    for t in range(total-1):
        eta=a+b*logit(np.clip(x[t],1e-8,1-1e-8))
        mu=float(expit(eta))
        aa=max(mu*phi,1e-5); bb=max((1-mu)*phi,1e-5)
        x[t+1]=rng.beta(aa,bb)
    return x[-(n+1):]


def pseudo_true_beta_ar(seed:int, n:int=350_000)->dict:
    rng=np.random.default_rng(seed)
    x=simulate_beta_ar(n,rng)
    fit=fit_tear(x,1.0,delta=0.02)
    return {"beta1":fit["beta1"],"gamma":fit["gamma"],"q":fit["q"]}


def run_misspec(n:int,reps:int,seed:int,pseudo:dict)->pd.DataFrame:
    rng=np.random.default_rng(seed)
    rows=[]
    for rep in range(reps):
        x=simulate_beta_ar(n,rng)
        Khat=float(np.max(x[1:]))
        fk=fit_tear(x,1.0,delta=0.02)
        fe=fit_tear(x,Khat,delta=0.02)
        cm=fk["cov_bg"]; ch=hac_cov_beta_gamma(fk,1.0)
        se_mb=np.sqrt(np.maximum(np.diag(cm),0))
        se_hac=np.sqrt(np.maximum(np.diag(ch),0))
        rows.append({
            "n":n,"rep":rep,"Khat":Khat,
            "beta1_known":fk["beta1"],"gamma_known":fk["gamma"],
            "beta1_estimated":fe["beta1"],"gamma_estimated":fe["gamma"],
            "sqrt_n_delta_beta1":math.sqrt(n)*(fe["beta1"]-fk["beta1"]),
            "sqrt_n_delta_gamma":math.sqrt(n)*(fe["gamma"]-fk["gamma"]),
            "cover_beta1_model":coverage(fk["beta1"],se_mb[0],pseudo["beta1"]),
            "cover_gamma_model":coverage(fk["gamma"],se_mb[1],pseudo["gamma"]),
            "cover_beta1_hac":coverage(fk["beta1"],se_hac[0],pseudo["beta1"]),
            "cover_gamma_hac":coverage(fk["gamma"],se_hac[1],pseudo["gamma"]),
            "se_beta1_model":se_mb[0],"se_gamma_model":se_mb[1],
            "se_beta1_hac":se_hac[0],"se_gamma_hac":se_hac[1],
        })
    return pd.DataFrame(rows)


def summarize_misspec(df:pd.DataFrame,pseudo:dict)->pd.DataFrame:
    out=[]
    for n,g in df.groupby("n"):
        out.append({
            "n":int(n),"reps":len(g),
            "beta1_bias":float((g.beta1_known-pseudo["beta1"]).mean()),
            "gamma_bias":float((g.gamma_known-pseudo["gamma"]).mean()),
            "beta1_model_coverage":float(g.cover_beta1_model.mean()),
            "gamma_model_coverage":float(g.cover_gamma_model.mean()),
            "beta1_hac_coverage":float(g.cover_beta1_hac.mean()),
            "gamma_hac_coverage":float(g.cover_gamma_hac.mean()),
            "mean_abs_sqrt_n_plugin_beta1":float(np.mean(np.abs(g.sqrt_n_delta_beta1))),
            "mean_abs_sqrt_n_plugin_gamma":float(np.mean(np.abs(g.sqrt_n_delta_gamma))),
            "median_endpoint_gap":float(np.median(1-g.Khat)),
        })
    return pd.DataFrame(out)


def run_optimizer_tolerance(s:Scenario,n:int,reps:int,seed:int)->pd.DataFrame:
    rng=np.random.default_rng(seed); rows=[]
    for rep in range(reps):
        x=simulate_tear(n,s.q,s.gamma,s.K,rng)
        Khat=float(np.max(x[1:]))
        loose=fit_tear(x,Khat,gtol=1e-4,maxiter=100)
        tight=fit_tear(x,Khat,gtol=1e-11,maxiter=1000)
        rows.append({"rep":rep,"n":n,
                     "sqrt_n_delta_beta1":math.sqrt(n)*(loose["beta1"]-tight["beta1"]),
                     "sqrt_n_delta_gamma":math.sqrt(n)*(loose["gamma"]-tight["gamma"]),
                     "loose_grad":loose["grad_norm"],"tight_grad":tight["grad_norm"],
                     "loose_success":loose["success"],"tight_success":tight["success"]})
    return pd.DataFrame(rows)


def make_figures(endpoint:pd.DataFrame,plugin:pd.DataFrame,misspec:pd.DataFrame,figdir:Path)->None:
    figdir.mkdir(parents=True,exist_ok=True)
    # 1: Endpoint exponential QQ
    for sc in endpoint.scenario.unique():
        g=endpoint[endpoint.scenario==sc]
        ns=sorted(g.n.unique())
        plt.figure(figsize=(6.4,4.5))
        for n in ns:
            vals=np.sort(g[g.n==n].scaled_exp.to_numpy())
            p=(np.arange(1,len(vals)+1)-0.5)/len(vals)
            theo=-np.log1p(-p)
            plt.plot(theo,vals,marker='.',linestyle='none',label=f"n={n}")
        mx=max(4,float(np.quantile(g.scaled_exp,0.99)))
        plt.plot([0,mx],[0,mx],linestyle='--')
        plt.xlabel("Exponential(1) quantile")
        plt.ylabel("Empirical c n(K-Khat) quantile")
        plt.title(f"Endpoint limit: {sc}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir/f"endpoint_qq_{sc}.pdf")
        plt.savefig(figdir/f"endpoint_qq_{sc}.png",dpi=180)
        plt.close()
    # 2: plug-in scaled difference
    plt.figure(figsize=(6.6,4.5))
    for sc in plugin.scenario.unique():
        g=plugin[plugin.scenario==sc].sort_values('n')
        plt.plot(g.n,g.mean_abs_sqrt_n_delta_gamma,marker='o',label=sc)
    plt.xscale('log')
    plt.xlabel("n")
    plt.ylabel("Mean absolute sqrt(n) plug-in difference (gamma)")
    plt.title("Known-wall versus estimated-wall dynamic estimates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir/"plugin_invariance_gamma.pdf")
    plt.savefig(figdir/"plugin_invariance_gamma.png",dpi=180)
    plt.close()
    # 3: misspecification coverage
    sm=summarize_misspec(misspec, {"beta1":0,"gamma":0}) if False else None
    plt.figure(figsize=(6.4,4.5))
    cov=misspec.groupby('n').agg(model_beta=('cover_beta1_model','mean'),hac_beta=('cover_beta1_hac','mean'),
                                 model_gamma=('cover_gamma_model','mean'),hac_gamma=('cover_gamma_hac','mean')).reset_index()
    for col,label in [('model_beta','model beta1'),('hac_beta','HAC beta1'),('model_gamma','model gamma'),('hac_gamma','HAC gamma')]:
        plt.plot(cov.n,cov[col],marker='o',label=label)
    plt.axhline(0.95,linestyle='--')
    plt.xscale('log')
    plt.ylim(0,1.02)
    plt.xlabel('n'); plt.ylabel('Empirical 95% coverage')
    plt.title('TEAR inference under beta-AR misspecification')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir/'misspec_coverage.pdf')
    plt.savefig(figdir/'misspec_coverage.png',dpi=180)
    plt.close()


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['quick','medium','full'],default='medium')
    ap.add_argument('--out',type=Path,default=Path('../results'))
    ap.add_argument('--fig',type=Path,default=Path('../figures'))
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); args.fig.mkdir(parents=True,exist_ok=True)
    if args.mode=='quick':
        reps_main=50; reps_endpoint=120; reps_miss=60; reps_opt=40
    elif args.mode=='medium':
        reps_main=160; reps_endpoint=400; reps_miss=160; reps_opt=100
    else:
        reps_main=500; reps_endpoint=1500; reps_miss=500; reps_opt=300
    seed0=20260803
    true_rates={}
    for i,name in enumerate(['low_persistence','baseline','near_active','stress']):
        c,se=approximate_true_wall_rate(SCENARIOS[name],seed0+10+i,n=800_000 if args.mode!='quick' else 200_000)
        true_rates[name]={"c":c,"mcse":se}
    with open(args.out/'true_wall_rates.json','w') as f: json.dump(true_rates,f,indent=2)

    dfs=[]
    for si,name in enumerate(['low_persistence','baseline','near_active']):
        for ni,n in enumerate([300,1000,3000] if args.mode!='quick' else [300,1000]):
            dfs.append(run_correct_spec(SCENARIOS[name],n,reps_main,seed0+1000+100*si+ni,delta=0.02))
    correct=pd.concat(dfs,ignore_index=True)
    correct.to_csv(args.out/'correct_spec_raw.csv',index=False)
    scsum=summarize_correct(correct,SCENARIOS)
    scsum.to_csv(args.out/'correct_spec_summary.csv',index=False)
    plugin=summarize_plugin(correct)
    plugin.to_csv(args.out/'plugin_invariance_summary.csv',index=False)

    edfs=[]
    endpoint_plan=[('baseline',1000,reps_endpoint,1),('baseline',10000,reps_endpoint,2),
                   ('stress',10000,reps_endpoint,2),('stress',100000,max(120,reps_endpoint//2),5)]
    if args.mode=='quick': endpoint_plan=[('baseline',1000,reps_endpoint,2),('stress',10000,reps_endpoint,3)]
    for j,(name,n,reps,fit_every) in enumerate(endpoint_plan):
        edfs.append(run_endpoint_only(SCENARIOS[name],n,reps,seed0+2000+j,
                                      true_rates[name]['c'],fit_every=fit_every))
    endpoint=pd.concat(edfs,ignore_index=True)
    endpoint.to_csv(args.out/'endpoint_raw.csv',index=False)
    epsum=summarize_endpoint(endpoint)
    epsum.to_csv(args.out/'endpoint_summary.csv',index=False)

    idx=pd.concat([run_initial_index_experiment(n,reps_endpoint,seed0+3000+i,SCENARIOS['baseline'])
                   for i,n in enumerate([300,1000])],ignore_index=True)
    idx.to_csv(args.out/'initial_index_raw.csv',index=False)
    idxsum=idx.groupby('n').agg(reps=('rep','count'),
        exclude_mean=('exclude_X0_gap','mean'),exclude_zero_frequency=('exclude_X0_gap',lambda x:float(np.mean(x==0))),
        include_mean=('include_X0_gap','mean'),include_zero_frequency=('include_X0_gap',lambda x:float(np.mean(x==0)))).reset_index()
    idxsum.to_csv(args.out/'initial_index_summary.csv',index=False)

    pseudo=pseudo_true_beta_ar(seed0+4000,n=120_000 if args.mode=='quick' else 350_000)
    with open(args.out/'misspec_pseudo_true.json','w') as f: json.dump(pseudo,f,indent=2)
    mdfs=[run_misspec(n,reps_miss,seed0+4100+i,pseudo) for i,n in enumerate([500,2000])]
    miss=pd.concat(mdfs,ignore_index=True)
    miss.to_csv(args.out/'misspec_raw.csv',index=False)
    msum=summarize_misspec(miss,pseudo)
    msum.to_csv(args.out/'misspec_summary.csv',index=False)


    opt=run_optimizer_tolerance(SCENARIOS['baseline'],1000,reps_opt,seed0+6000)
    opt.to_csv(args.out/'optimizer_tolerance_raw.csv',index=False)
    optsum=pd.DataFrame([{
        "n":1000,"reps":len(opt),
        "mean_abs_sqrt_n_delta_beta1":float(np.mean(np.abs(opt.sqrt_n_delta_beta1))),
        "q95_abs_sqrt_n_delta_beta1":float(np.quantile(np.abs(opt.sqrt_n_delta_beta1),.95)),
        "mean_abs_sqrt_n_delta_gamma":float(np.mean(np.abs(opt.sqrt_n_delta_gamma))),
        "q95_abs_sqrt_n_delta_gamma":float(np.quantile(np.abs(opt.sqrt_n_delta_gamma),.95)),
        "loose_success_rate":float(opt.loose_success.mean()),"tight_success_rate":float(opt.tight_success.mean()),
        "median_loose_grad":float(opt.loose_grad.median()),"median_tight_grad":float(opt.tight_grad.median()),
    }])
    optsum.to_csv(args.out/'optimizer_tolerance_summary.csv',index=False)

    make_figures(endpoint,plugin,miss,args.fig)
    metadata={
        "mode":args.mode,"seed_base":seed0,"python":sys.version,"platform":platform.platform(),
        "numpy":np.__version__,"pandas":pd.__version__,
        "endpoint_convention":"max X_1,...,X_n; likelihood X_1->X_2,...,X_{n-1}->X_n",
        "scenarios":{k:asdict(v) for k,v in SCENARIOS.items()},
        "counts":{"reps_main":reps_main,"reps_endpoint":reps_endpoint,"reps_misspec":reps_miss,"reps_optimizer":reps_opt},
    }
    with open(args.out/'run_metadata.json','w') as f:json.dump(metadata,f,indent=2)
    print(json.dumps({"status":"ok","mode":args.mode,"out":str(args.out),"fig":str(args.fig)},indent=2))

if __name__=='__main__':
    main()

# --- Smooth conditional misspecification helpers (used by add_smooth_misspec.py) ---
def simulate_nonlinear_rate_tear(n:int,rng:np.random.Generator,q:float=1.0,gamma:float=2.0,eta:float=3.0,K:float=1.0,burn:int=1000)->Array:
    total=n+1+burn
    x=np.empty(total);x[0]=K/2;u=rng.random(total-1)
    for t in range(total-1):
        d=K-x[t]
        rate=q+gamma*d+eta*d*d
        x[t+1]=trunc_exp_draw(rate,K,float(u[t]))
    return x[-(n+1):]


def pseudo_true_nonlinear(seed:int,n:int=300000)->dict:
    rng=np.random.default_rng(seed)
    x=simulate_nonlinear_rate_tear(n,rng)
    f=fit_tear(x,1.0,delta=0.02)
    return {"beta1":f["beta1"],"gamma":f["gamma"],"q":f["q"]}


def run_smooth_misspec(n:int,reps:int,seed:int,pseudo:dict)->pd.DataFrame:
    rng=np.random.default_rng(seed);rows=[]
    for rep in range(reps):
        x=simulate_nonlinear_rate_tear(n,rng)
        Khat=float(np.max(x[1:]))
        fk=fit_tear(x,1.0,delta=0.02);fe=fit_tear(x,Khat,delta=0.02)
        cm=fk["cov_bg"];ch=hac_cov_beta_gamma(fk,1.0)
        se_m=np.sqrt(np.maximum(np.diag(cm),0));se_h=np.sqrt(np.maximum(np.diag(ch),0))
        rows.append({"n":n,"rep":rep,"Khat":Khat,
                     "beta1_known":fk["beta1"],"gamma_known":fk["gamma"],
                     "beta1_estimated":fe["beta1"],"gamma_estimated":fe["gamma"],
                     "sqrt_n_delta_beta1":math.sqrt(n)*(fe["beta1"]-fk["beta1"]),
                     "sqrt_n_delta_gamma":math.sqrt(n)*(fe["gamma"]-fk["gamma"]),
                     "cover_beta1_model":coverage(fk["beta1"],se_m[0],pseudo["beta1"]),
                     "cover_gamma_model":coverage(fk["gamma"],se_m[1],pseudo["gamma"]),
                     "cover_beta1_hac":coverage(fk["beta1"],se_h[0],pseudo["beta1"]),
                     "cover_gamma_hac":coverage(fk["gamma"],se_h[1],pseudo["gamma"]),
                     "se_beta1_model":se_m[0],"se_gamma_model":se_m[1],
                     "se_beta1_hac":se_h[0],"se_gamma_hac":se_h[1]})
    return pd.DataFrame(rows)
