"""
calibration.py — clinical validity metrics for PRS evaluation.

This module computes performance and calibration statistics when
phenotype/outcome data is available alongside PRS scores.

Provides
--------
load_outcomes          : load a case/control or continuous outcome file
discrimination_stats   : per-group AUC with bootstrap CIs (binary outcomes)
calibration_stats      : calibration slope, intercept, Brier score (binary)
plot_roc_by_group      : ROC curves per ancestry group
plot_calibration_by_group : calibration plots (observed vs predicted) per group

These metrics address ClinGen PRS-RS item PRS-RS:Performance, which
cannot be computed from scores and ancestry labels alone.

Input format
------------
Outcomes file: TSV or CSV with at minimum:
    IID        : sample identifier (must match scores file)
    outcome    : binary (0/1) or continuous phenotype value

Optional additional columns are ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # sigmoid


# ── Outcome loading ───────────────────────────────────────────────────────

def load_outcomes(path: Path, outcome_col: Optional[str] = None) -> pd.DataFrame:
    """Load an outcomes file (TSV or CSV) and return a tidy DataFrame.

    Parameters
    ----------
    path        : path to the outcomes file
    outcome_col : column name for the outcome; if None, tries 'outcome',
                  'phenotype', 'case', 'status', 'y'

    Returns
    -------
    DataFrame with columns: IID, outcome (numeric)
    """
    if path.suffix.lower() in [".tsv", ".tab"]:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    cols_lower = {c.lower(): c for c in df.columns}
    iid_col = cols_lower.get("iid") or cols_lower.get("sample") or cols_lower.get("id")
    if iid_col is None:
        raise ValueError(f"Outcomes file must contain an IID column. Found: {list(df.columns)}")

    if outcome_col is None:
        for candidate in ["outcome", "phenotype", "case", "status", "y", "pheno"]:
            if candidate in cols_lower:
                outcome_col = cols_lower[candidate]
                break
    if outcome_col is None:
        raise ValueError(
            f"Cannot find outcome column. Specify --outcome-column. Found: {list(df.columns)}"
        )

    out = df[[iid_col, outcome_col]].copy()
    out.columns = ["IID", "outcome"]
    out["IID"] = out["IID"].astype(str)
    out["outcome"] = pd.to_numeric(out["outcome"], errors="coerce")
    out = out.dropna(subset=["outcome"])
    return out


# ── Discrimination (AUC) ──────────────────────────────────────────────────

def _auc_ci_bootstrap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (auc, ci_lo, ci_hi) via bootstrap."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    alpha = (100 - ci) / 2

    auc = float(roc_auc_score(y_true, y_score))
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(float(roc_auc_score(yt, ys)))

    if len(boot_aucs) < 10:
        return auc, float("nan"), float("nan")

    return auc, float(np.percentile(boot_aucs, alpha)), float(np.percentile(boot_aucs, 100 - alpha))


def discrimination_stats(
    df: pd.DataFrame,
    n_boot: int = 1000,
    score_col: str = "SCORE_STD",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    """Per-group AUC (C-statistic) with bootstrap 95% CIs.

    Requires binary outcome (0/1).

    Returns DataFrame: group | n | n_cases | n_controls | auc | auc_ci_lo | auc_ci_hi
    """
    from sklearn.metrics import roc_auc_score

    records = []
    for grp, sub in df.groupby("group"):
        y = sub[outcome_col].values
        s = sub[score_col].values
        n_cases = int((y == 1).sum())
        n_controls = int((y == 0).sum())

        if n_cases < 5 or n_controls < 5:
            records.append({
                "group": grp, "n": len(y),
                "n_cases": n_cases, "n_controls": n_controls,
                "auc": float("nan"), "auc_ci_lo": float("nan"), "auc_ci_hi": float("nan"),
                "note": "insufficient cases or controls (<5)",
            })
            continue

        auc, lo, hi = _auc_ci_bootstrap(y, s, n_boot=n_boot)
        records.append({
            "group": grp, "n": len(y),
            "n_cases": n_cases, "n_controls": n_controls,
            "auc": round(auc, 4),
            "auc_ci_lo": round(lo, 4) if not np.isnan(lo) else float("nan"),
            "auc_ci_hi": round(hi, 4) if not np.isnan(hi) else float("nan"),
            "note": "",
        })
    return pd.DataFrame(records)


# ── Calibration ───────────────────────────────────────────────────────────

def calibration_stats(
    df: pd.DataFrame,
    score_col: str = "SCORE_STD",
    outcome_col: str = "outcome",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-group calibration statistics for a binary outcome.

    Fits a logistic regression of outcome ~ score per group.
    Reports:
        calibration_slope     : slope of logistic regression (1.0 = perfect)
        calibration_intercept : intercept (0.0 = perfect)
        brier_score           : mean squared error of predicted probabilities
        observed_rate         : observed case proportion
        mean_predicted        : mean predicted probability from logistic model

    A well-calibrated PRS has slope ≈ 1.0 and intercept ≈ 0.0.
    Slope < 1 indicates overconfident predictions (too spread out).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss

    records = []
    for grp, sub in df.groupby("group"):
        y = sub[outcome_col].values.astype(float)
        s = sub[score_col].values.reshape(-1, 1)
        n_cases = int((y == 1).sum())
        n_controls = int((y == 0).sum())

        if n_cases < 5 or n_controls < 5:
            records.append({
                "group": grp, "n": len(y), "n_cases": n_cases,
                "calibration_slope": float("nan"),
                "calibration_intercept": float("nan"),
                "brier_score": float("nan"),
                "observed_rate": round(float(y.mean()), 4),
                "mean_predicted": float("nan"),
                "note": "insufficient cases or controls (<5)",
            })
            continue

        try:
            lr = LogisticRegression(solver="lbfgs", max_iter=500)
            lr.fit(s, y)
            slope = float(lr.coef_[0][0])
            intercept = float(lr.intercept_[0])
            p_pred = lr.predict_proba(s)[:, 1]
            brier = float(brier_score_loss(y, p_pred))
            records.append({
                "group": grp, "n": len(y), "n_cases": n_cases,
                "calibration_slope": round(slope, 4),
                "calibration_intercept": round(intercept, 4),
                "brier_score": round(brier, 4),
                "observed_rate": round(float(y.mean()), 4),
                "mean_predicted": round(float(p_pred.mean()), 4),
                "note": "",
            })
        except Exception as e:
            records.append({
                "group": grp, "n": len(y), "n_cases": n_cases,
                "calibration_slope": float("nan"),
                "calibration_intercept": float("nan"),
                "brier_score": float("nan"),
                "observed_rate": round(float(y.mean()), 4),
                "mean_predicted": float("nan"),
                "note": str(e),
            })
    return pd.DataFrame(records)


# ── Plots ─────────────────────────────────────────────────────────────────

_GROUP_COLORS = {
    "AFR": "#1D7874", "AMR": "#F5A623", "EAS": "#065A82",
    "EUR": "#8B4CA8", "SAS": "#D94F3D",
}


def _group_color(grp: str, idx: int = 0) -> str:
    import matplotlib.pyplot as plt
    return _GROUP_COLORS.get(grp, plt.cm.tab10.colors[idx % 10])


def plot_roc_by_group(
    df: pd.DataFrame,
    out_path: Path,
    score_col: str = "SCORE_STD",
    outcome_col: str = "outcome",
    title: str = "ROC curves by ancestry group",
) -> None:
    """Per-group ROC curves with AUC in legend."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    from pathlib import Path as _P

    out_path = _P(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = sorted(df["group"].unique())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, ls="--")

    for i, grp in enumerate(groups):
        sub = df[df["group"] == grp]
        y = sub[outcome_col].values
        s = sub[score_col].values
        if len(np.unique(y)) < 2 or (y == 1).sum() < 5:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, color=_group_color(grp, i), lw=2,
                label=f"{grp}  (AUC = {auc:.3f})")

    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_by_group(
    df: pd.DataFrame,
    out_path: Path,
    score_col: str = "SCORE_STD",
    outcome_col: str = "outcome",
    n_bins: int = 10,
    title: str = "Calibration by ancestry group",
) -> None:
    """Calibration plot: observed vs mean predicted probability per decile."""
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import calibration_curve
    from pathlib import Path as _P

    out_path = _P(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = sorted(df["group"].unique())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=1, ls="--", label="perfect")

    for i, grp in enumerate(groups):
        sub = df[df["group"] == grp]
        y = sub[outcome_col].values
        s = sub[score_col].values.reshape(-1, 1)
        if (y == 1).sum() < 5 or (y == 0).sum() < 5:
            continue
        try:
            lr = LogisticRegression(solver="lbfgs", max_iter=500)
            lr.fit(s, y)
            p_pred = lr.predict_proba(s)[:, 1]
            frac_pos, mean_pred = calibration_curve(y, p_pred, n_bins=n_bins, strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", color=_group_color(grp, i),
                    lw=1.5, ms=4, label=grp)
        except Exception:
            continue

    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Observed fraction of cases", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_auc_comparison(
    disc_df: pd.DataFrame,
    out_path: Path,
    title: str = "AUC by ancestry group with 95% bootstrap CI",
) -> None:
    """Bar chart of AUC per group with error bars."""
    import matplotlib.pyplot as plt
    from pathlib import Path as _P

    out_path = _P(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid = disc_df.dropna(subset=["auc"])
    if valid.empty:
        return

    groups = valid["group"].tolist()
    aucs = valid["auc"].values
    lo = aucs - valid["auc_ci_lo"].values
    hi = valid["auc_ci_hi"].values - aucs
    colors = [_GROUP_COLORS.get(g, "#888888") for g in groups]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(groups, aucs, yerr=[lo, hi], color=colors,
           capsize=5, edgecolor="white", error_kw={"elinewidth": 1.5}, alpha=0.85)
    ax.axhline(0.5, color="#aaaaaa", lw=1, ls="--", label="random (AUC=0.5)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC (C-statistic)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
