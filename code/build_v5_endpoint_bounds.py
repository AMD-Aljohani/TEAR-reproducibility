#!/usr/bin/env python3
"""Build the V5 exact endpoint-bound and exact-coverage audit tables.

This script uses the locked endpoint raw outputs. It does not rerun the
Monte Carlo campaign.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

PARAMS = {
    "low_persistence": {"label": "Low persistence", "K": 1.0, "q": 1.5, "gamma": 0.75},
    "baseline": {"label": "Baseline", "K": 1.0, "q": 1.0, "gamma": 4.0},
    "near_active": {"label": "Boundary stress", "K": 1.0, "q": 0.15, "gamma": 4.0},
    "stress": {"label": "Small wall density", "K": 1.0, "q": 5.0, "gamma": 5.0},
    "endpoint_baseline": {"label": "Endpoint illustration", "K": 1.0, "q": 2.5, "gamma": 2.5},
}


def phi(rate: float, s: float, K: float) -> float:
    """Exact conditional probability of the upper strip (K-s,K]."""
    if not (0.0 <= s <= K):
        raise ValueError("s must lie in [0,K]")
    if abs(rate) < 1e-10:
        return s / K
    return math.expm1(rate * s) / math.expm1(rate * K)


def g(rate: float, K: float) -> float:
    """Conditional density at the wall."""
    if abs(rate) < 1e-10:
        return 1.0 / K
    return rate / math.expm1(rate * K)


def clopper_pearson(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - level
    lower = 0.0 if successes == 0 else float(beta_dist.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta_dist.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def load_wall_rates() -> dict[str, float]:
    raw = json.loads((RES / "true_wall_rates.json").read_text())
    rates = {name: float(entry["c"]) for name, entry in raw.items()}
    aligned = json.loads((RES / "endpoint_aligned_baseline_rate.json").read_text())
    rates["endpoint_baseline"] = float(aligned["c"])
    return rates


def build_bounds_table() -> pd.DataFrame:
    rates = load_wall_rates()
    raw = pd.concat(
        [
            pd.read_csv(RES / "endpoint_raw.csv"),
            pd.read_csv(RES / "endpoint_aligned_baseline_raw.csv"),
        ],
        ignore_index=True,
    )
    rows: list[dict[str, float | int | str]] = []
    for (scenario, n), frame in raw.groupby(["scenario", "n"], sort=False):
        p = PARAMS[str(scenario)]
        K, q, gamma = p["K"], p["q"], p["gamma"]
        beta1 = q + gamma * K
        c = rates[str(scenario)]
        s = min(K, 1.0 / (int(n) * c))
        lower = (1.0 - phi(q, s, K)) * (1.0 - phi(q + gamma * s, s, K)) ** (int(n) - 1)
        upper = (1.0 - phi(beta1, s, K)) ** int(n)
        gaps = K - frame["Khat"].to_numpy(float)
        empirical = float(np.mean(gaps >= s))
        rows.append(
            {
                "scenario": scenario,
                "design": p["label"],
                "n": int(n),
                "s_equals_1_over_nc": s,
                "lower_bound": lower,
                "empirical_survival": empirical,
                "upper_bound": upper,
                "reps": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def build_wall_brackets() -> pd.DataFrame:
    rates = load_wall_rates()
    rows = []
    for scenario in ["low_persistence", "baseline", "near_active", "stress", "endpoint_baseline"]:
        p = PARAMS[scenario]
        K, q, gamma = p["K"], p["q"], p["gamma"]
        beta1 = q + gamma * K
        rows.append(
            {
                "scenario": scenario,
                "design": p["label"],
                "q": q,
                "gamma": gamma,
                "beta1": beta1,
                "g_beta1": g(beta1, K),
                "pi_K": rates[scenario],
                "g_q": g(q, K),
            }
        )
    return pd.DataFrame(rows)


def build_coverage_exact() -> pd.DataFrame:
    base = pd.concat(
        [
            pd.read_csv(RES / "endpoint_summary.csv"),
            pd.read_csv(RES / "endpoint_aligned_baseline_summary.csv"),
        ],
        ignore_index=True,
    )
    rows = []
    for _, row in base.iterrows():
        scenario = str(row["scenario"])
        label = PARAMS[scenario]["label"]
        n_oracle = int(row["reps"])
        x_oracle = int(round(float(row["oracle_95_coverage"]) * n_oracle))
        n_feasible = int(row["feasible_reps"])
        x_feasible = int(round(float(row["feasible_95_coverage"]) * n_feasible))
        ol, ou = clopper_pearson(x_oracle, n_oracle)
        fl, fu = clopper_pearson(x_feasible, n_feasible)
        rows.append(
            {
                "scenario": scenario,
                "design": label,
                "n": int(row["n"]),
                "oracle_successes": x_oracle,
                "oracle_trials": n_oracle,
                "oracle_coverage": x_oracle / n_oracle,
                "oracle_cp95_lower": ol,
                "oracle_cp95_upper": ou,
                "feasible_successes": x_feasible,
                "feasible_trials": n_feasible,
                "feasible_coverage": x_feasible / n_feasible,
                "feasible_cp95_lower": fl,
                "feasible_cp95_upper": fu,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    bounds = build_bounds_table()
    brackets = build_wall_brackets()
    coverage = build_coverage_exact()
    bounds.to_csv(RES / "endpoint_exact_tail_bounds.csv", index=False)
    brackets.to_csv(RES / "endpoint_wall_density_brackets.csv", index=False)
    coverage.to_csv(RES / "endpoint_coverage_exact_binomial.csv", index=False)
    print("Wrote V5 exact endpoint-bound and exact-coverage tables.")


if __name__ == "__main__":
    main()
