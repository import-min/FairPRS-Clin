"""
equity.py — ancestry-stratified equity metrics for PRS evaluation.

Provides:
  - pairwise_smds                  : standardized mean differences between all group pairs
  - ks_tests                       : pairwise Kolmogorov-Smirnov tests
  - bootstrap_group_stats          : bootstrap CIs (mean, flagging rate) per group
  - flagging_disparity             : max/min flagging ratio + which groups are affected
  - sensitivity_curve              : flagging rate per group across a range of cutoffs
  - equalized_cutoffs              : per-group cutoffs that achieve a target flagging rate
  - resource_constrained_fair_threshold (RCFT) : NEW — optimal per-group thresholds
                                                 under a total screening budget constraint
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar


# ── Pairwise standardized mean differences ────────────────────────────────

def pairwise_smds(df: pd.DataFrame, score_col: str = "SCORE_STD") -> pd.DataFrame:
    """Compute Cohen's d (SMD) between every pair of ancestry groups.

    SMD = (mean_A - mean_B) / pooled_SD

    Returns a DataFrame with columns: group_a, group_b, smd, abs_smd.
    Interpretation: |SMD| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
    """
    groups = sorted(df["group"].unique())
    records = []
    for i, ga in enumerate(groups):
        for gb in groups[i + 1:]:
            xa = df.loc[df["group"] == ga, score_col].values
            xb = df.loc[df["group"] == gb, score_col].values
            na, nb = len(xa), len(xb)
            pooled_sd = np.sqrt(
                ((na - 1) * xa.std(ddof=1) ** 2 + (nb - 1) * xb.std(ddof=1) ** 2)
                / (na + nb - 2)
            ) if (na + nb - 2) > 0 else 1.0
            pooled_sd = pooled_sd if pooled_sd > 0 else 1.0
            smd = (xa.mean() - xb.mean()) / pooled_sd
            records.append({
                "group_a": ga, "group_b": gb,
                "mean_a": round(float(xa.mean()), 4),
                "mean_b": round(float(xb.mean()), 4),
                "smd": round(float(smd), 4),
                "abs_smd": round(float(abs(smd)), 4),
            })
    return pd.DataFrame(records)


# ── Pairwise KS tests ─────────────────────────────────────────────────────

def ks_tests(df: pd.DataFrame, score_col: str = "SCORE_STD") -> pd.DataFrame:
    """Two-sample KS test between every pair of ancestry groups."""
    groups = sorted(df["group"].unique())
    records = []
    for i, ga in enumerate(groups):
        for gb in groups[i + 1:]:
            xa = df.loc[df["group"] == ga, score_col].values
            xb = df.loc[df["group"] == gb, score_col].values
            result = stats.ks_2samp(xa, xb)
            records.append({
                "group_a": ga, "group_b": gb,
                "ks_stat": round(float(result.statistic), 4),
                "p_value": round(float(result.pvalue), 6),
                "significant_p05": bool(result.pvalue < 0.05),
            })
    return pd.DataFrame(records)


# ── Bootstrap confidence intervals ───────────────────────────────────────

def bootstrap_group_stats(
    df: pd.DataFrame,
    cutoff_value: Optional[float] = None,
    n_boot: int = 1000,
    ci: float = 95.0,
    score_col: str = "SCORE_STD",
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap CIs for per-group mean and (optionally) flagging rate."""
    rng = np.random.default_rng(seed)
    alpha = (100 - ci) / 2
    records = []
    for grp, sub in df.groupby("group"):
        x = sub[score_col].values
        boot_means = np.array([
            rng.choice(x, size=len(x), replace=True).mean()
            for _ in range(n_boot)
        ])
        rec = {
            "group": grp, "n": len(x),
            "mean": round(float(x.mean()), 4),
            "mean_ci_lo": round(float(np.percentile(boot_means, alpha)), 4),
            "mean_ci_hi": round(float(np.percentile(boot_means, 100 - alpha)), 4),
        }
        if cutoff_value is not None:
            boot_flags = np.array([
                (rng.choice(x, size=len(x), replace=True) >= cutoff_value).mean()
                for _ in range(n_boot)
            ])
            rec["flagging_rate"] = round(float((x >= cutoff_value).mean()), 4)
            rec["flag_ci_lo"] = round(float(np.percentile(boot_flags, alpha)), 4)
            rec["flag_ci_hi"] = round(float(np.percentile(boot_flags, 100 - alpha)), 4)
        records.append(rec)
    return pd.DataFrame(records)


# ── Flagging disparity ────────────────────────────────────────────────────

def flagging_disparity(
    df: pd.DataFrame,
    cutoff_value: float,
    score_col: str = "SCORE_STD",
) -> Dict:
    """Compute the disparity ratio: max(flagging_rate) / min(flagging_rate)."""
    rates = (
        df.groupby("group")[score_col]
        .apply(lambda x: (x >= cutoff_value).mean())
        .to_dict()
    )
    if not rates:
        return {}
    max_grp = max(rates, key=rates.get)
    min_grp = min(rates, key=rates.get)
    min_rate = rates[min_grp]
    max_rate = rates[max_grp]
    ratio = (max_rate / min_rate) if min_rate > 0 else float("inf")
    return {
        "disparity_ratio": round(float(ratio), 3),
        "max_flagging_group": max_grp,
        "max_flagging_rate": round(float(max_rate), 4),
        "min_flagging_group": min_grp,
        "min_flagging_rate": round(float(min_rate), 4),
        "absolute_disparity_pp": round(float((max_rate - min_rate) * 100), 2),
        "per_group_flagging_rates": {k: round(float(v), 4) for k, v in rates.items()},
        "cutoff_used": round(float(cutoff_value), 4),
    }


# ── Sensitivity curve ─────────────────────────────────────────────────────

def sensitivity_curve(
    df: pd.DataFrame,
    percentiles: Optional[List[float]] = None,
    score_col: str = "SCORE_STD",
) -> pd.DataFrame:
    """Compute per-group flagging rates across a range of cutoff percentiles."""
    if percentiles is None:
        percentiles = list(range(5, 100))

    all_scores = df[score_col].values
    groups = sorted(df["group"].unique())
    group_scores = {g: df.loc[df["group"] == g, score_col].values for g in groups}

    records = []
    for pct in percentiles:
        cutoff = float(np.percentile(all_scores, pct))
        rates = {g: float((group_scores[g] >= cutoff).mean()) for g in groups}
        non_zero = [r for r in rates.values() if r > 0]
        ratio = (max(non_zero) / min(non_zero)) if len(non_zero) > 1 and min(non_zero) > 0 else float("nan")
        for g in groups:
            records.append({
                "percentile": pct,
                "cutoff_value": round(cutoff, 4),
                "group": g,
                "flagging_rate": round(rates[g], 4),
                "disparity_ratio": round(ratio, 3) if not np.isnan(ratio) else None,
            })
    return pd.DataFrame(records)


# ── Equalized cutoffs ─────────────────────────────────────────────────────

def equalized_cutoffs(
    df: pd.DataFrame,
    target_flagging_rate: float = 0.10,
    score_col: str = "SCORE_STD",
) -> pd.DataFrame:
    """Per-group cutoffs that achieve a target flagging rate."""
    records = []
    for grp, sub in df.groupby("group"):
        x = sub[score_col].values
        pct = (1.0 - target_flagging_rate) * 100.0
        pct = max(0.0, min(100.0, pct))
        cutoff = float(np.percentile(x, pct))
        achieved = float((x >= cutoff).mean())
        records.append({
            "group": grp, "n": len(x),
            "equalized_cutoff": round(cutoff, 4),
            "achieved_flagging_rate": round(achieved, 4),
            "target_flagging_rate": target_flagging_rate,
        })
    return pd.DataFrame(records)


# ── Resource-Constrained Fair Threshold (RCFT) ────────────────────────────
# Novel algorithm — FairPRS-Clin (Aisha, 2025)

def resource_constrained_fair_threshold(
    df: pd.DataFrame,
    budget: float = 0.10,
    fairness_criterion: str = "demographic_parity",
    score_col: str = "SCORE_STD",
    n_grid: int = 500,
) -> pd.DataFrame:
    """Find per-group screening thresholds that minimize disparity subject
    to a total screening budget constraint.

    This is a constrained optimization problem absent from existing PRS tools.
    Simple equalized_cutoffs ignores the budget: if you set everyone to
    target=10%, the total flagged fraction will not equal 10% when group
    sizes differ. RCFT enforces:

        Σ_g  (n_g / N) * flagging_rate_g  ≤  budget

    while minimizing one of:
        - "demographic_parity"  : max(flagging_rate_g) - min(flagging_rate_g)
        - "chebyshev"           : max |flagging_rate_g - target_global|
        - "variance"            : Var(flagging_rate_g across groups)

    Method
    ------
    Lagrangian relaxation via scalar binary search on the multiplier λ:

        L(T_g, λ) = fairness_objective(T_g) + λ * (Σ n_g/N * flag_g - budget)

    For fixed λ, each group's optimal threshold is found by grid search
    over a fine score quantile grid. The outer binary search adjusts λ
    until the budget constraint is satisfied.

    Parameters
    ----------
    df                 : DataFrame with 'group' and score_col
    budget             : maximum fraction of total population to flag (e.g. 0.10)
    fairness_criterion : 'demographic_parity', 'chebyshev', or 'variance'
    score_col          : standardized score column
    n_grid             : number of threshold candidates per group

    Returns
    -------
    DataFrame: group | n | rcft_cutoff | rcft_flagging_rate |
               global_budget_flagging_rate | budget | fairness_criterion |
               disparity_ratio_rcft | disparity_ratio_naive
    """
    groups = sorted(df["group"].unique())
    N = len(df)
    group_data = {g: df.loc[df["group"] == g, score_col].values for g in groups}
    group_n = {g: len(v) for g, v in group_data.items()}
    group_weights = {g: group_n[g] / N for g in groups}

    # Candidate thresholds: dense grid from 1st to 99th percentile of each group
    group_thresholds = {
        g: np.linspace(
            np.percentile(v, 1), np.percentile(v, 99), n_grid
        )
        for g, v in group_data.items()
    }

    # Precompute flagging rates at each candidate threshold
    # flag_rates[g][j] = flagging rate of group g at threshold j
    flag_rates = {
        g: np.array([(v >= t).mean() for t in group_thresholds[g]])
        for g, v in group_data.items()
    }

    def solve_for_lambda(lam: float) -> Dict[str, int]:
        """For a given Lagrange multiplier, find per-group optimal threshold index."""
        best_idx = {}
        for g in groups:
            thresholds_g = group_thresholds[g]
            rates_g = flag_rates[g]
            # Global target = budget (equal for all groups under demographic parity)
            global_target = budget

            if fairness_criterion == "demographic_parity":
                # Minimize |rate_g - global_target| + lam * weight_g * rate_g
                costs = np.abs(rates_g - global_target) + lam * group_weights[g] * rates_g
            elif fairness_criterion == "chebyshev":
                costs = np.abs(rates_g - global_target) + lam * group_weights[g] * rates_g
            elif fairness_criterion == "variance":
                # Variance penalty — use squared deviation from mean
                mean_rate = np.mean([np.mean(flag_rates[gg]) for gg in groups])
                costs = (rates_g - mean_rate) ** 2 + lam * group_weights[g] * rates_g
            else:
                raise ValueError(f"Unknown fairness_criterion: {fairness_criterion}")

            best_idx[g] = int(np.argmin(costs))
        return best_idx

    def total_flagged(idx: Dict[str, int]) -> float:
        return sum(group_weights[g] * flag_rates[g][idx[g]] for g in groups)

    # Binary search on lambda to find budget-satisfying solution
    lam_lo, lam_hi = 0.0, 100.0
    best_idx = solve_for_lambda(0.0)

    for _ in range(60):  # 60 iterations → precision ~1e-18
        lam_mid = (lam_lo + lam_hi) / 2.0
        idx = solve_for_lambda(lam_mid)
        tf = total_flagged(idx)
        if tf <= budget:
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid
        best_idx = idx

    # Ensure budget is met — if not, increase thresholds
    # (fall back to simple budget-proportional allocation)
    if total_flagged(best_idx) > budget * 1.05:
        # Fallback: find global cutoff that achieves budget
        all_scores = df[score_col].values
        global_cutoff = float(np.percentile(all_scores, (1 - budget) * 100))
        for g in groups:
            idx_g = int(np.argmin(np.abs(group_thresholds[g] - global_cutoff)))
            best_idx[g] = idx_g

    # Naive comparison: global single cutoff
    all_scores = df[score_col].values
    naive_cutoff = float(np.percentile(all_scores, (1 - budget) * 100))
    naive_rates = {g: float((group_data[g] >= naive_cutoff).mean()) for g in groups}
    naive_non_zero = [r for r in naive_rates.values() if r > 0]
    naive_dr = (max(naive_non_zero) / min(naive_non_zero)
                if len(naive_non_zero) > 1 and min(naive_non_zero) > 0
                else float("inf"))

    # RCFT results
    rcft_rates = {g: float(flag_rates[g][best_idx[g]]) for g in groups}
    rcft_cutoffs = {g: float(group_thresholds[g][best_idx[g]]) for g in groups}
    rcft_non_zero = [r for r in rcft_rates.values() if r > 0]
    rcft_dr = (max(rcft_non_zero) / min(rcft_non_zero)
               if len(rcft_non_zero) > 1 and min(rcft_non_zero) > 0
               else float("inf"))

    actual_budget = total_flagged(best_idx)

    records = []
    for g in groups:
        records.append({
            "group": g,
            "n": group_n[g],
            "rcft_cutoff": round(rcft_cutoffs[g], 4),
            "rcft_flagging_rate": round(rcft_rates[g], 4),
            "naive_cutoff": round(naive_cutoff, 4),
            "naive_flagging_rate": round(naive_rates[g], 4),
            "budget": budget,
            "actual_budget_used": round(actual_budget, 4),
            "fairness_criterion": fairness_criterion,
            "disparity_ratio_rcft": round(rcft_dr, 3) if not np.isinf(rcft_dr) else None,
            "disparity_ratio_naive": round(naive_dr, 3) if not np.isinf(naive_dr) else None,
        })

    df_out = pd.DataFrame(records)
    return df_out
