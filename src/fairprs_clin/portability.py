"""
portability.py — Ancestry Portability Score (APS) for PRS.

The Ancestry Portability Score is a single scalar in [0, 1] that
quantifies how evenly a PRS performs across ancestry groups.
APS = 1 indicates perfect portability; APS = 0 indicates complete failure
to generalize beyond the discovery population.

Two modes
---------
With outcomes (binary case/control):
    APS combines:
    - APS_disc  : 1 - (AUC_max - AUC_min) / AUC_max
                  Penalizes large AUC gaps between ancestry groups.
    - APS_cal   : 1 - mean(|calibration_slope_g - 1|) / normalization
                  Penalizes deviation of calibration slopes from 1.0.
    - APS = harmonic_mean(APS_disc, APS_cal)

Without outcomes (scores + groups only):
    APS combines:
    - APS_dist  : 1 - Wasserstein_max / SD_global
                  Wasserstein-1 distance between each group and the global
                  distribution, normalized by global SD.
    - APS_flag  : 1 - (flag_max - flag_min) / flag_max  [= 1/disparity_ratio]
                  Direct penalty for unequal flagging burden.
    - APS = harmonic_mean(APS_dist, APS_flag)

Bootstrap confidence intervals are computed for all APS values.

Reference
---------
Novel metric introduced in FairPRS-Clin (Aisha, 2025).
Conceptually related to the portability index of Privé et al. (2022)
but generalized to combine discrimination, calibration, and distributional
evidence into one standardized scalar.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


# ── Internal helpers ──────────────────────────────────────────────────────

def _harmonic_mean(values: List[float]) -> float:
    """Harmonic mean. Returns 0 if any component is 0; NaN if all NaN or empty."""
    vals = [v for v in values if not np.isnan(v)]
    if not vals:
        return float("nan")
    if any(v == 0.0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def _clamp(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# ── Distributional APS (no outcomes required) ─────────────────────────────

def aps_distributional(
    df: pd.DataFrame,
    score_col: str = "SCORE_STD",
    cutoff_value: Optional[float] = None,
) -> Dict:
    """Compute APS from score distributions alone (no outcomes needed).

    Parameters
    ----------
    df           : DataFrame with 'group' and score_col columns
    cutoff_value : if provided, includes flagging-disparity component

    Returns
    -------
    dict with keys:
        aps               : overall APS (harmonic mean of components)
        aps_distributional: 1 - max_wasserstein_distance / global_SD
        aps_flagging      : 1 / disparity_ratio (None if no cutoff)
        components        : dict of all sub-scores
        interpretation    : human-readable label
    """
    all_scores = df[score_col].values
    global_sd = float(all_scores.std(ddof=0))
    if global_sd == 0:
        global_sd = 1.0

    groups = sorted(df["group"].unique())

    # Wasserstein distance between each group and the global distribution
    wd_vals = []
    for grp in groups:
        x_g = df.loc[df["group"] == grp, score_col].values
        wd = float(wasserstein_distance(x_g, all_scores))
        wd_vals.append(wd)

    max_wd = max(wd_vals) if wd_vals else 0.0
    aps_dist = _clamp(1.0 - max_wd / global_sd)

    components: Dict = {
        "aps_distributional": round(aps_dist, 4),
        "max_wasserstein_distance": round(max_wd, 4),
        "global_sd": round(global_sd, 4),
        "per_group_wasserstein": {
            g: round(w, 4) for g, w in zip(groups, wd_vals)
        },
    }

    aps_components = [aps_dist]

    # Flagging disparity component (optional)
    aps_flag = None
    if cutoff_value is not None:
        rates = {
            g: float((df.loc[df["group"] == g, score_col] >= cutoff_value).mean())
            for g in groups
        }
        non_zero = [r for r in rates.values() if r > 0]
        if len(non_zero) >= 2 and max(non_zero) > 0:
            disparity_ratio = max(non_zero) / min(non_zero)
            aps_flag = _clamp(1.0 / disparity_ratio)
        else:
            aps_flag = 1.0  # no disparity if nobody is flagged
        components["aps_flagging"] = round(aps_flag, 4)
        components["per_group_flagging_rates"] = {
            g: round(r, 4) for g, r in rates.items()
        }
        aps_components.append(aps_flag)

    aps = _harmonic_mean(aps_components)
    components["aps"] = round(aps, 4)
    components["interpretation"] = _interpret_aps(aps)

    return components


# ── Clinical APS (requires outcomes) ─────────────────────────────────────

def aps_clinical(
    disc_df: pd.DataFrame,
    cal_df: pd.DataFrame,
) -> Dict:
    """Compute APS from per-group AUC and calibration statistics.

    Parameters
    ----------
    disc_df : output of calibration.discrimination_stats()
              must have columns: group, auc
    cal_df  : output of calibration.calibration_stats()
              must have columns: group, calibration_slope

    Returns
    -------
    dict with APS components and overall score
    """
    valid_disc = disc_df.dropna(subset=["auc"])
    valid_cal = cal_df.dropna(subset=["calibration_slope"])

    components: Dict = {}
    aps_components = []

    # Discrimination component
    if len(valid_disc) >= 2:
        aucs = valid_disc["auc"].values
        auc_max = float(aucs.max())
        auc_min = float(aucs.min())
        if auc_max > 0:
            aps_disc = _clamp(1.0 - (auc_max - auc_min) / auc_max)
        else:
            aps_disc = float("nan")
        components["aps_discrimination"] = round(aps_disc, 4) if not np.isnan(aps_disc) else float("nan")
        components["auc_max"] = round(auc_max, 4)
        components["auc_min"] = round(auc_min, 4)
        components["auc_gap"] = round(auc_max - auc_min, 4)
        if not np.isnan(aps_disc):
            aps_components.append(aps_disc)
    else:
        components["aps_discrimination"] = float("nan")
        components["note_discrimination"] = "fewer than 2 groups with sufficient sample size"

    # Calibration component
    if len(valid_cal) >= 1:
        slopes = valid_cal["calibration_slope"].values
        # Mean absolute deviation from 1.0 (perfect calibration)
        mad_from_1 = float(np.mean(np.abs(slopes - 1.0)))
        # Normalize: MAD of 1.0 means the average slope is 0 or 2 (very poor)
        aps_cal = _clamp(1.0 - mad_from_1)
        components["aps_calibration"] = round(aps_cal, 4)
        components["mean_abs_slope_deviation"] = round(mad_from_1, 4)
        components["calibration_slopes"] = {
            row["group"]: round(row["calibration_slope"], 4)
            for _, row in valid_cal.iterrows()
        }
        aps_components.append(aps_cal)
    else:
        components["aps_calibration"] = float("nan")
        components["note_calibration"] = "insufficient data for calibration"

    aps = _harmonic_mean(aps_components)
    components["aps"] = round(aps, 4) if not np.isnan(aps) else float("nan")
    components["interpretation"] = _interpret_aps(aps)
    components["mode"] = "clinical"

    return components


# ── Bootstrap CI for APS ──────────────────────────────────────────────────

def bootstrap_aps(
    df: pd.DataFrame,
    score_col: str = "SCORE_STD",
    cutoff_value: Optional[float] = None,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
    outcome_col: Optional[str] = None,
) -> Dict:
    """Bootstrap confidence interval for APS.

    Resamples individuals (stratified by group to preserve group sizes)
    and recomputes APS each time.

    Parameters
    ----------
    df           : DataFrame with 'group', score_col, and optionally outcome_col
    cutoff_value : used for flagging component
    n_boot       : number of bootstrap resamples
    ci           : confidence level
    outcome_col  : if provided, computes clinical APS using outcomes

    Returns
    -------
    dict: aps_point | aps_ci_lo | aps_ci_hi | n_boot_valid
    """
    rng = np.random.default_rng(seed)
    alpha = (100 - ci) / 2

    boot_aps = []
    for _ in range(n_boot):
        # Stratified resample: resample within each group
        parts = []
        for grp, sub in df.groupby("group"):
            idx = rng.integers(0, len(sub), size=len(sub))
            parts.append(sub.iloc[idx])
        df_boot = pd.concat(parts, ignore_index=True)

        try:
            if outcome_col is not None and outcome_col in df_boot.columns:
                # Clinical APS — requires outcomes in the DataFrame
                from .calibration import discrimination_stats, calibration_stats
                disc = discrimination_stats(df_boot, n_boot=50, score_col=score_col,
                                            outcome_col=outcome_col)
                cal = calibration_stats(df_boot, score_col=score_col,
                                        outcome_col=outcome_col)
                result = aps_clinical(disc, cal)
            else:
                result = aps_distributional(df_boot, score_col=score_col,
                                            cutoff_value=cutoff_value)
            val = result.get("aps")
            if val is not None and not np.isnan(val):
                boot_aps.append(float(val))
        except Exception:
            continue

    if len(boot_aps) < 10:
        return {
            "aps_point": float("nan"),
            "aps_ci_lo": float("nan"),
            "aps_ci_hi": float("nan"),
            "n_boot_valid": len(boot_aps),
        }

    # Point estimate from the full sample
    if outcome_col is not None and outcome_col in df.columns:
        try:
            from .calibration import discrimination_stats, calibration_stats
            disc = discrimination_stats(df, n_boot=200, score_col=score_col,
                                        outcome_col=outcome_col)
            cal = calibration_stats(df, score_col=score_col, outcome_col=outcome_col)
            point_result = aps_clinical(disc, cal)
        except Exception:
            point_result = aps_distributional(df, score_col=score_col,
                                              cutoff_value=cutoff_value)
    else:
        point_result = aps_distributional(df, score_col=score_col,
                                          cutoff_value=cutoff_value)

    return {
        "aps_point": round(float(point_result.get("aps", float("nan"))), 4),
        "aps_ci_lo": round(float(np.percentile(boot_aps, alpha)), 4),
        "aps_ci_hi": round(float(np.percentile(boot_aps, 100 - alpha)), 4),
        "n_boot_valid": len(boot_aps),
        "interpretation": _interpret_aps(point_result.get("aps", float("nan"))),
        "components": {k: v for k, v in point_result.items()
                       if k not in ("aps", "interpretation")},
    }


# ── Interpretation ────────────────────────────────────────────────────────

def _interpret_aps(aps: float) -> str:
    if np.isnan(aps):
        return "insufficient data"
    if aps >= 0.90:
        return "high portability — minimal ancestry-related performance gap"
    if aps >= 0.75:
        return "moderate portability — meaningful gap exists; group-specific thresholds recommended"
    if aps >= 0.50:
        return "low portability — substantial gap; strong caution warranted for multi-ancestry use"
    return "very low portability — score may not generalize beyond discovery ancestry"


# ── APS comparison across multiple PRS ───────────────────────────────────

def compare_scores_aps(
    score_dfs: Dict[str, pd.DataFrame],
    score_col: str = "SCORE_STD",
    cutoff_value: Optional[float] = None,
) -> pd.DataFrame:
    """Compare APS across multiple PRS applied to the same cohort.

    Parameters
    ----------
    score_dfs : dict mapping score_name → DataFrame (with group + score_col)
    cutoff_value : used for flagging component

    Returns
    -------
    DataFrame: score_name | aps | aps_distributional | aps_flagging | interpretation
    """
    records = []
    for name, df in score_dfs.items():
        result = aps_distributional(df, score_col=score_col, cutoff_value=cutoff_value)
        records.append({
            "score_name": name,
            "aps": result.get("aps"),
            "aps_distributional": result.get("aps_distributional"),
            "aps_flagging": result.get("aps_flagging"),
            "max_wasserstein": result.get("max_wasserstein_distance"),
            "interpretation": result.get("interpretation"),
        })
    return pd.DataFrame(records).sort_values("aps", ascending=False)
