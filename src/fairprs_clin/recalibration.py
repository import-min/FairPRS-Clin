"""
recalibration.py — Bayesian Group Recalibration (BGR) for PRS.

Standard per-group logistic regression fails when ancestry-specific
sample sizes are small (n < 100 cases), which is the typical setting
for PRS validation in non-European populations.

BGR addresses this by estimating group-specific recalibration parameters
(intercept α_g, slope β_g) with L2 regularization toward the global
model — equivalent to a Gaussian prior centered on the population-level
fit. When group sample sizes are large, BGR converges to per-group MLE.
When group sample sizes are small, parameters shrink toward the global
estimate, preventing overfitting.

Mathematical formulation
------------------------
For group g with n_g labeled samples:

    outcome_i ~ Bernoulli(sigmoid(α_g + β_g * score_i))

    Prior:  α_g ~ N(α_global, λ_α⁻¹)
            β_g ~ N(β_global, λ_β⁻¹)

MAP estimation is equivalent to:

    minimize -log_likelihood(α_g, β_g) + λ_α/2 * (α_g - α_global)²
                                        + λ_β/2 * (β_g - β_global)²

The regularization strength λ is chosen automatically by:
    λ = 1 / (1 + n_g / n_threshold)

where n_threshold controls the crossover between heavy shrinkage (small
groups) and near-MLE (large groups). Default n_threshold = 100.

Outputs
-------
- Per-group recalibrated scores: α_g + β_g * score_raw
- Recalibration parameters table: group | alpha | beta | shrinkage_weight | n
- Recalibrated AUC and calibration statistics
- Plot: raw vs recalibrated calibration curves

References
----------
Novel method introduced in FairPRS-Clin (Aisha, 2025).
Related to: Platt scaling, temperature scaling, and hierarchical Bayes
approaches in the fairness-aware ML literature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid


# ── Core MAP estimation ───────────────────────────────────────────────────

def _logistic_nll(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    lambda_reg: float,
) -> float:
    """Negative log-likelihood with L2 regularization toward prior_mean."""
    alpha, beta = params
    logits = alpha + beta * X
    # Numerically stable log-sigmoid
    log_p = np.where(logits >= 0,
                     -np.log1p(np.exp(-logits)),
                     logits - np.log1p(np.exp(logits)))
    log_1mp = np.where(logits >= 0,
                       -logits - np.log1p(np.exp(-logits)),
                       -np.log1p(np.exp(logits)))
    nll = -np.sum(y * log_p + (1 - y) * log_1mp)
    # L2 regularization toward prior
    reg = 0.5 * lambda_reg * np.sum((params - prior_mean) ** 2)
    return nll + reg


def _logistic_gradient(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    prior_mean: np.ndarray,
    lambda_reg: float,
) -> np.ndarray:
    """Gradient of the regularized NLL."""
    alpha, beta = params
    p = expit(alpha + beta * X)
    residuals = p - y
    grad_alpha = np.sum(residuals) + lambda_reg * (alpha - prior_mean[0])
    grad_beta = np.sum(residuals * X) + lambda_reg * (beta - prior_mean[1])
    return np.array([grad_alpha, grad_beta])


def _fit_group_map(
    scores: np.ndarray,
    outcomes: np.ndarray,
    global_alpha: float,
    global_beta: float,
    lambda_reg: float,
) -> Tuple[float, float, bool]:
    """Fit MAP estimate of (alpha_g, beta_g) for one group.

    Returns (alpha_g, beta_g, converged)
    """
    prior_mean = np.array([global_alpha, global_beta])
    x0 = prior_mean.copy()

    result = minimize(
        _logistic_nll,
        x0,
        args=(scores, outcomes, prior_mean, lambda_reg),
        jac=_logistic_gradient,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9},
    )
    return float(result.x[0]), float(result.x[1]), bool(result.success)


def _shrinkage_lambda(n_g: int, n_threshold: int = 100) -> float:
    """Compute regularization strength.

    λ decreases as n_g increases — shrinks heavily for small groups,
    approaches MLE for large groups.
    """
    return float(n_threshold / max(n_g, 1))


# ── Global model ──────────────────────────────────────────────────────────

def _fit_global_model(
    scores: np.ndarray,
    outcomes: np.ndarray,
) -> Tuple[float, float]:
    """Fit a global logistic regression (no regularization).

    Returns (global_alpha, global_beta).
    """
    result = minimize(
        _logistic_nll,
        np.array([0.0, 1.0]),
        args=(scores, outcomes, np.array([0.0, 1.0]), 0.0),
        jac=_logistic_gradient,
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    return float(result.x[0]), float(result.x[1])


# ── Main BGR function ─────────────────────────────────────────────────────

def bayesian_group_recalibration(
    df: pd.DataFrame,
    score_col: str = "SCORE_STD",
    outcome_col: str = "outcome",
    n_threshold: int = 100,
    min_cases: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit Bayesian Group Recalibration model.

    For each ancestry group, estimates group-specific recalibration
    parameters (α_g, β_g) with shrinkage toward the global model.
    Groups with fewer than min_cases use the global model directly.

    Parameters
    ----------
    df          : DataFrame with 'group', score_col, outcome_col columns
    score_col   : column name for the PRS (standardized)
    outcome_col : column name for binary outcome (0/1)
    n_threshold : controls shrinkage crossover point (default 100)
                  at n_g = n_threshold, λ = 1.0 (moderate regularization)
    min_cases   : groups with fewer cases use global model entirely

    Returns
    -------
    df_recal  : input DataFrame with added column 'SCORE_BGR'
                containing the recalibrated score (logit scale)
    params_df : DataFrame with per-group recalibration parameters:
                group | n | n_cases | alpha | beta | lambda_reg |
                shrinkage_weight | converged | recal_method
    """
    all_scores = df[score_col].values
    all_outcomes = df[outcome_col].values.astype(float)

    n_cases_total = int((all_outcomes == 1).sum())
    n_controls_total = int((all_outcomes == 0).sum())

    if n_cases_total < 5 or n_controls_total < 5:
        raise ValueError(
            f"Insufficient data for global model: "
            f"{n_cases_total} cases, {n_controls_total} controls."
        )

    # Fit global model
    global_alpha, global_beta = _fit_global_model(all_scores, all_outcomes)

    # Per-group MAP estimation
    records = []
    df_out = df.copy()
    df_out["SCORE_BGR"] = float("nan")

    for grp, sub in df.groupby("group"):
        idx = sub.index
        s = sub[score_col].values
        y = sub[outcome_col].values.astype(float)
        n_g = len(s)
        n_cases_g = int((y == 1).sum())

        if n_cases_g < min_cases or (len(y) - n_cases_g) < min_cases:
            # Not enough data — use global model
            alpha_g, beta_g = global_alpha, global_beta
            lam = float("inf")
            shrinkage = 1.0
            converged = True
            method = "global_fallback"
        else:
            lam = _shrinkage_lambda(n_g, n_threshold)
            alpha_g, beta_g, converged = _fit_group_map(
                s, y, global_alpha, global_beta, lam
            )
            # Shrinkage weight: 0 = full MLE, 1 = full prior
            shrinkage = lam / (1.0 + lam)
            method = "BGR_MAP"

        # Recalibrated score = logit of predicted probability
        # = α_g + β_g * raw_score (on logit scale, comparable across groups)
        df_out.loc[idx, "SCORE_BGR"] = alpha_g + beta_g * s

        records.append({
            "group": grp,
            "n": n_g,
            "n_cases": n_cases_g,
            "alpha_g": round(alpha_g, 5),
            "beta_g": round(beta_g, 5),
            "global_alpha": round(global_alpha, 5),
            "global_beta": round(global_beta, 5),
            "lambda_reg": round(lam, 4) if not np.isinf(lam) else None,
            "shrinkage_weight": round(shrinkage, 4),
            "converged": converged,
            "recal_method": method,
        })

    params_df = pd.DataFrame(records)
    return df_out, params_df


# ── Evaluate recalibration improvement ───────────────────────────────────

def evaluate_recalibration(
    df_original: pd.DataFrame,
    df_recal: pd.DataFrame,
    score_col_orig: str = "SCORE_STD",
    score_col_recal: str = "SCORE_BGR",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    """Compare AUC and calibration before and after BGR.

    Returns DataFrame: group | n | auc_before | auc_after | auc_delta |
                       slope_before | slope_after | brier_before | brier_after
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.linear_model import LogisticRegression

    records = []
    groups = sorted(df_original["group"].unique())

    for grp in groups:
        orig_sub = df_original[df_original["group"] == grp]
        recal_sub = df_recal[df_recal["group"] == grp]

        y = orig_sub[outcome_col].values.astype(float)
        s_orig = orig_sub[score_col_orig].values
        s_recal = recal_sub[score_col_recal].values

        n_cases = int((y == 1).sum())
        if n_cases < 5 or (len(y) - n_cases) < 5:
            records.append({
                "group": grp, "n": len(y), "n_cases": n_cases,
                "auc_before": float("nan"), "auc_after": float("nan"),
                "auc_delta": float("nan"),
                "slope_before": float("nan"), "slope_after": float("nan"),
                "brier_before": float("nan"), "brier_after": float("nan"),
                "note": "insufficient cases",
            })
            continue

        try:
            auc_before = float(roc_auc_score(y, s_orig))
            auc_after = float(roc_auc_score(y, s_recal))

            # Calibration slope before
            lr_before = LogisticRegression(solver="lbfgs", max_iter=500)
            lr_before.fit(s_orig.reshape(-1, 1), y)
            slope_before = float(lr_before.coef_[0][0])
            p_before = lr_before.predict_proba(s_orig.reshape(-1, 1))[:, 1]

            # After recalibration, convert BGR logit score to probability
            p_after = expit(s_recal)
            # Calibration slope in logit space ≈ 1.0 by construction for BGR
            # Compute empirical Brier score
            brier_before = float(brier_score_loss(y, p_before))
            brier_after = float(brier_score_loss(y, np.clip(p_after, 1e-6, 1 - 1e-6)))

            # Slope of recalibrated: fit LR on logit scores
            lr_after = LogisticRegression(solver="lbfgs", max_iter=500)
            lr_after.fit(s_recal.reshape(-1, 1), y)
            slope_after = float(lr_after.coef_[0][0])

            records.append({
                "group": grp, "n": len(y), "n_cases": n_cases,
                "auc_before": round(auc_before, 4),
                "auc_after": round(auc_after, 4),
                "auc_delta": round(auc_after - auc_before, 4),
                "slope_before": round(slope_before, 4),
                "slope_after": round(slope_after, 4),
                "brier_before": round(brier_before, 4),
                "brier_after": round(brier_after, 4),
                "note": "",
            })
        except Exception as e:
            records.append({
                "group": grp, "n": len(y), "n_cases": n_cases,
                "auc_before": float("nan"), "auc_after": float("nan"),
                "auc_delta": float("nan"),
                "slope_before": float("nan"), "slope_after": float("nan"),
                "brier_before": float("nan"), "brier_after": float("nan"),
                "note": str(e),
            })

    return pd.DataFrame(records)


# ── Plots ─────────────────────────────────────────────────────────────────

_GROUP_COLORS = {
    "AFR": "#1D7874", "AMR": "#F5A623", "EAS": "#065A82",
    "EUR": "#8B4CA8", "SAS": "#D94F3D",
}


def plot_recalibration_comparison(
    df_recal: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_path: Path,
    title: str = "Bayesian Group Recalibration: AUC and Brier score",
) -> None:
    """Side-by-side bar chart comparing AUC and Brier score before/after BGR."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid = eval_df.dropna(subset=["auc_before"])
    if valid.empty:
        return

    groups = valid["group"].tolist()
    x = np.arange(len(groups))
    w = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # AUC comparison
    ax = axes[0]
    ax.bar(x - w/2, valid["auc_before"], w, label="Before BGR",
           color="#888888", alpha=0.8, edgecolor="white")
    ax.bar(x + w/2, valid["auc_after"], w, label="After BGR",
           color=[_GROUP_COLORS.get(g, "#065A82") for g in groups],
           alpha=0.85, edgecolor="white")
    ax.axhline(0.5, color="#cccccc", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title("AUC before/after BGR", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Brier score comparison (lower = better)
    ax = axes[1]
    ax.bar(x - w/2, valid["brier_before"], w, label="Before BGR",
           color="#888888", alpha=0.8, edgecolor="white")
    ax.bar(x + w/2, valid["brier_after"], w, label="After BGR",
           color=[_GROUP_COLORS.get(g, "#065A82") for g in groups],
           alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Brier score (lower = better)")
    ax.set_title("Brier score before/after BGR", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bgr_parameters(
    params_df: pd.DataFrame,
    out_path: Path,
    title: str = "BGR recalibration parameters by ancestry group",
) -> None:
    """Plot per-group alpha and beta with shrinkage weight visualization."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid = params_df[params_df["recal_method"] == "BGR_MAP"].copy()
    if valid.empty:
        return

    groups = valid["group"].tolist()
    x = np.arange(len(groups))
    colors = [_GROUP_COLORS.get(g, "#888888") for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Intercept (alpha)
    ax = axes[0]
    ax.bar(x, valid["alpha_g"], color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(valid["global_alpha"].iloc[0], color="#C0392B", lw=1.5,
               ls="--", label="global α")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Recalibration intercept (α)")
    ax.set_title("Intercept by group", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Slope (beta)
    ax = axes[1]
    ax.bar(x, valid["beta_g"], color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(valid["global_beta"].iloc[0], color="#C0392B", lw=1.5,
               ls="--", label="global β")
    ax.axhline(1.0, color="#aaaaaa", lw=1, ls=":", label="slope=1 (ideal)")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Recalibration slope (β)")
    ax.set_title("Slope by group", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Shrinkage weight
    ax = axes[2]
    ax.bar(x, valid["shrinkage_weight"], color=colors, alpha=0.85,
           edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Shrinkage weight\n(0 = MLE, 1 = prior)")
    ax.set_title("Shrinkage by group", fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
