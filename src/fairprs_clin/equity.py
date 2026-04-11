"""
equity.py — ancestry-stratified equity metrics for PRS evaluation.

Provides:
  - pairwise_smds          : standardized mean differences between all group pairs
  - ks_tests               : pairwise Kolmogorov-Smirnov tests
  - bootstrap_group_stats  : bootstrap CIs (mean, flagging rate) per group
  - flagging_disparity     : max/min flagging ratio + which groups are affected
  - sensitivity_curve      : flagging rate per group across a range of cutoffs
  - equalized_cutoffs      : per-group cutoffs that achieve a target flagging rate
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


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
            # pooled SD (Cohen's d)
            pooled_sd = np.sqrt(
                ((na - 1) * xa.std(ddof=1) ** 2 + (nb - 1) * xb.std(ddof=1) ** 2)
                / (na + nb - 2)
            ) if (na + nb - 2) > 0 else 1.0
            pooled_sd = pooled_sd if pooled_sd > 0 else 1.0
            smd = (xa.mean() - xb.mean()) / pooled_sd
            records.append({
                "group_a": ga,
                "group_b": gb,
                "mean_a": round(float(xa.mean()), 4),
                "mean_b": round(float(xb.mean()), 4),
                "smd": round(float(smd), 4),
                "abs_smd": round(float(abs(smd)), 4),
            })
    return pd.DataFrame(records)


# ── Pairwise KS tests ─────────────────────────────────────────────────────

def ks_tests(df: pd.DataFrame, score_col: str = "SCORE_STD") -> pd.DataFrame:
    """Two-sample KS test between every pair of ancestry groups.

    Returns a DataFrame with columns: group_a, group_b, ks_stat, p_value, significant (p<0.05).
    """
    groups = sorted(df["group"].unique())
    records = []
    for i, ga in enumerate(groups):
        for gb in groups[i + 1:]:
            xa = df.loc[df["group"] == ga, score_col].values
            xb = df.loc[df["group"] == gb, score_col].values
            result = stats.ks_2samp(xa, xb)
            records.append({
                "group_a": ga,
                "group_b": gb,
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
    """Bootstrap CIs for per-group mean and (optionally) flagging rate.

    Parameters
    ----------
    df           : DataFrame with 'group' and score_col columns
    cutoff_value : if provided, also bootstraps flagging rate CI
    n_boot       : number of bootstrap resamples
    ci           : confidence level (default 95)
    seed         : random seed for reproducibility

    Returns DataFrame with: group, mean, mean_ci_lo, mean_ci_hi,
                            [flagging_rate, flag_ci_lo, flag_ci_hi]
    """
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
            "group": grp,
            "n": len(x),
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
    """Compute the disparity ratio: max(flagging_rate) / min(flagging_rate).

    This single number summarises how unequal the screening burden is
    across ancestry groups under a given cutoff.

    Returns a dict with:
        disparity_ratio   : max / min flagging rate
        max_group         : group with highest flagging rate
        min_group         : group with lowest flagging rate
        flagging_rates    : per-group rates
        absolute_disparity: max - min flagging rate (percentage points)
    """
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
    """Compute per-group flagging rates across a range of cutoff percentiles.

    Parameters
    ----------
    df          : DataFrame with 'group' and score_col columns
    percentiles : list of percentiles (0-100) to evaluate; defaults to 5..99 step 1

    Returns a long-format DataFrame:
        percentile | cutoff_value | group | flagging_rate | disparity_ratio
    """
    if percentiles is None:
        percentiles = list(range(50, 100))  # 50th–99th percentile

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


# ── Equalized cutoffs (recalibration) ────────────────────────────────────

def equalized_cutoffs(
    df: pd.DataFrame,
    target_flagging_rate: float = 0.10,
    score_col: str = "SCORE_STD",
) -> pd.DataFrame:
    """For each group, find the score threshold that achieves a target flagging rate.

    This is a recalibration step: instead of one global cutoff, compute
    group-specific cutoffs such that each group has approximately the same
    proportion flagged. Useful for showing what 'equalized screening' would require.

    Parameters
    ----------
    target_flagging_rate : desired proportion flagged per group (e.g. 0.10 = top 10%)

    Returns DataFrame: group | equalized_cutoff | achieved_flagging_rate | n
    """
    records = []
    for grp, sub in df.groupby("group"):
        x = sub[score_col].values
        # the cutoff that gives target_flagging_rate is the (1-target)th percentile
        pct = (1.0 - target_flagging_rate) * 100.0
        pct = max(0.0, min(100.0, pct))
        cutoff = float(np.percentile(x, pct))
        achieved = float((x >= cutoff).mean())
        records.append({
            "group": grp,
            "n": len(x),
            "equalized_cutoff": round(cutoff, 4),
            "achieved_flagging_rate": round(achieved, 4),
            "target_flagging_rate": target_flagging_rate,
        })
    return pd.DataFrame(records)
