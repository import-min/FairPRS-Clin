from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .io import load_groups, load_scores
from .utils import ensure_dir, write_json
from .equity import (
    pairwise_smds, ks_tests, bootstrap_group_stats,
    flagging_disparity, sensitivity_curve, equalized_cutoffs,
    resource_constrained_fair_threshold,
)
from .plots import (
    plot_distributions_kde, plot_sensitivity_curve, plot_disparity_curve,
    plot_smd_heatmap, plot_bootstrap_ci_bars, plot_equalized_cutoffs,
)
from .portability import aps_distributional, bootstrap_aps
from .recalibration import (
    bayesian_group_recalibration, evaluate_recalibration,
    plot_recalibration_comparison, plot_bgr_parameters,
)


def standardize_scores(df: pd.DataFrame, how: str) -> pd.DataFrame:
    out = df.copy()
    if how == "none":
        out["SCORE_STD"] = out["SCORE"].astype(float)
        return out
    if how == "global":
        mu = out["SCORE"].mean()
        sd = out["SCORE"].std(ddof=0)
        sd = sd if sd > 0 else 1.0
        out["SCORE_STD"] = (out["SCORE"] - mu) / sd
        return out
    if how == "within_group":
        out["SCORE_STD"] = out.groupby("group")["SCORE"].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0)
        )
        return out
    raise ValueError("standardize must be one of: global, within_group, none")


def compute_cutoff(
    df: pd.DataFrame,
    cutoff: Optional[float],
    cutoff_percentile: Optional[float],
) -> Tuple[Optional[float], str]:
    if cutoff is not None and cutoff_percentile is not None:
        raise ValueError("Provide either cutoff or cutoff_percentile, not both.")
    if cutoff_percentile is not None:
        if not (0.0 < cutoff_percentile < 100.0):
            raise ValueError("cutoff_percentile must be between 0 and 100.")
        c = float(np.percentile(df["SCORE_STD"].values, cutoff_percentile))
        return c, f"percentile:{cutoff_percentile}"
    if cutoff is not None:
        return float(cutoff), f"absolute:{cutoff}"
    return None, "none"


def summarize_by_group(df: pd.DataFrame, cutoff_value: Optional[float]) -> pd.DataFrame:
    g = df.groupby("group", dropna=False)
    summ = g["SCORE_STD"].agg(
        n="count", mean="mean",
        sd=lambda x: x.std(ddof=0), median="median",
        q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
    ).reset_index()
    summ["iqr"] = summ["q75"] - summ["q25"]
    if cutoff_value is not None:
        flagged = g.apply(lambda d: (d["SCORE_STD"] >= cutoff_value).mean()).reset_index(name="flagged_prop")
        flagged_n = g.apply(lambda d: int((d["SCORE_STD"] >= cutoff_value).sum())).reset_index(name="flagged_n")
        summ = summ.merge(flagged, on="group").merge(flagged_n, on="group")
    else:
        summ["flagged_prop"] = np.nan
        summ["flagged_n"] = np.nan
    return summ


def evaluate_scores(
    scores_path: Path,
    groups_path: Path,
    out_dir: Path,
    cutoff: Optional[str],
    cutoff_percentile: Optional[float],
    standardize: str,
    score_column: Optional[str] = None,
    n_boot: int = 1000,
    equalize_target: float = 0.10,
    outcomes_path: Optional[Path] = None,
    outcome_column: Optional[str] = None,
    run_bgr: bool = True,
    bgr_n_threshold: int = 100,
    rcft_budget: float = 0.10,
    rcft_criterion: str = "demographic_parity",
) -> Dict:
    """Run full ancestry-stratified PRS evaluation.

    Parameters
    ----------
    scores_path     : .sscore or CSV with IID + score column
    groups_path     : TSV/CSV with IID + group (ancestry) column
    out_dir         : output directory
    cutoff          : absolute score cutoff
    cutoff_percentile : percentile cutoff 0-100
    standardize     : 'global', 'within_group', or 'none'
    score_column    : score column name (auto-detected if None)
    n_boot          : bootstrap resamples for CIs
    equalize_target : target flagging rate for equalized cutoffs
    outcomes_path   : optional binary outcome file for clinical validity
    outcome_column  : outcome column name (auto-detected if None)
    run_bgr         : run Bayesian Group Recalibration (requires outcomes)
    bgr_n_threshold : BGR shrinkage crossover point
    rcft_budget     : total screening budget for RCFT (e.g. 0.10 = top 10%)
    rcft_criterion  : fairness objective for RCFT
    """
    out_dir = ensure_dir(out_dir)
    tables_dir = ensure_dir(out_dir / "tables")
    figs_dir = ensure_dir(out_dir / "figures")
    ensure_dir(out_dir / "reports")
    logs_dir = ensure_dir(out_dir / "logs")

    # ── Load ──────────────────────────────────────────────────────────────
    scores = load_scores(scores_path, score_column=score_column)
    groups = load_groups(groups_path)
    df = scores.merge(groups, on="IID", how="inner")
    if df.empty:
        raise ValueError("No IID overlap between scores and groups.")

    df = standardize_scores(df, how=standardize)
    c_val, c_kind = compute_cutoff(
        df, float(cutoff) if cutoff is not None else None, cutoff_percentile
    )

    # ── Basic summary ─────────────────────────────────────────────────────
    summ = summarize_by_group(df, cutoff_value=c_val)
    summ.to_csv(tables_dir / "summary_by_group.tsv", sep="\t", index=False)
    df.to_csv(tables_dir / "scores_with_groups.tsv", sep="\t", index=False)

    # ── Standard equity metrics ───────────────────────────────────────────
    smd_df = pairwise_smds(df)
    smd_df.to_csv(tables_dir / "pairwise_smds.tsv", sep="\t", index=False)

    ks_df = ks_tests(df)
    ks_df.to_csv(tables_dir / "ks_tests.tsv", sep="\t", index=False)

    boot_df = bootstrap_group_stats(df, cutoff_value=c_val, n_boot=n_boot)
    boot_df.to_csv(tables_dir / "bootstrap_stats.tsv", sep="\t", index=False)

    disparity_meta: Dict = {}
    eq_df = pd.DataFrame()
    if c_val is not None:
        disparity_meta = flagging_disparity(df, cutoff_value=c_val)
        eq_df = equalized_cutoffs(df, target_flagging_rate=equalize_target)
        eq_df.to_csv(tables_dir / "equalized_cutoffs.tsv", sep="\t", index=False)

    sens_df = sensitivity_curve(df, percentiles=list(range(5, 100)))
    sens_df.to_csv(tables_dir / "sensitivity_curve.tsv", sep="\t", index=False)

    # ── Novel Algorithm 3: RCFT ───────────────────────────────────────────
    rcft_df = resource_constrained_fair_threshold(
        df, budget=rcft_budget,
        fairness_criterion=rcft_criterion,
    )
    rcft_df.to_csv(tables_dir / "rcft_thresholds.tsv", sep="\t", index=False)

    # ── Novel Algorithm 1: APS ────────────────────────────────────────────
    aps_result = bootstrap_aps(df, score_col="SCORE_STD",
                               cutoff_value=c_val, n_boot=n_boot)
    write_json(logs_dir / "aps_result.json", aps_result)

    # ── Figures ───────────────────────────────────────────────────────────
    plot_distributions_kde(df, figs_dir / "score_distributions_kde.png", cutoff_value=c_val)
    plot_sensitivity_curve(sens_df, figs_dir / "sensitivity_curve.png",
                           highlight_percentile=cutoff_percentile)
    plot_disparity_curve(sens_df, figs_dir / "disparity_curve.png",
                         highlight_percentile=cutoff_percentile)
    if len(smd_df) > 0:
        plot_smd_heatmap(smd_df, figs_dir / "smd_heatmap.png")
    if "mean_ci_lo" in boot_df.columns:
        plot_bootstrap_ci_bars(boot_df, figs_dir / "bootstrap_means.png")
    if c_val is not None and not eq_df.empty:
        plot_equalized_cutoffs(eq_df, c_val, figs_dir / "equalized_cutoffs.png")
    _plot_rcft(rcft_df, figs_dir / "rcft_thresholds.png")
    _plot_aps(aps_result, figs_dir / "aps_gauge.png")

    # ── Clinical validity + BGR (when outcomes provided) ──────────────────
    calibration_meta: Dict = {}
    bgr_meta: Dict = {}

    if outcomes_path is not None:
        from .calibration import (
            load_outcomes, discrimination_stats, calibration_stats,
            plot_roc_by_group, plot_calibration_by_group, plot_auc_comparison,
        )
        from .portability import aps_clinical

        outcomes = load_outcomes(outcomes_path, outcome_col=outcome_column)
        df_clin = df.merge(outcomes, on="IID", how="inner")

        if df_clin.empty:
            calibration_meta["warning"] = "No IID overlap between scores and outcomes."
        else:
            n_cases = int((df_clin["outcome"] == 1).sum())
            n_controls = int((df_clin["outcome"] == 0).sum())

            disc_df = discrimination_stats(df_clin, n_boot=n_boot)
            disc_df.to_csv(tables_dir / "discrimination_auc.tsv", sep="\t", index=False)

            cal_df = calibration_stats(df_clin)
            cal_df.to_csv(tables_dir / "calibration_stats.tsv", sep="\t", index=False)

            plot_roc_by_group(df_clin, figs_dir / "roc_by_group.png")
            plot_calibration_by_group(df_clin, figs_dir / "calibration_by_group.png")
            plot_auc_comparison(disc_df, figs_dir / "auc_comparison.png")

            # Clinical APS
            aps_clin = aps_clinical(disc_df, cal_df)
            write_json(logs_dir / "aps_clinical.json", aps_clin)

            calibration_meta = {
                "n_with_outcomes": int(df_clin.shape[0]),
                "n_cases": n_cases, "n_controls": n_controls,
                "outcomes_file": str(outcomes_path),
                "per_group_auc": {
                    row["group"]: row["auc"]
                    for _, row in disc_df.iterrows() if not np.isnan(row["auc"])
                },
                "aps_clinical": aps_clin,
                "artifacts": {
                    "discrimination_auc": str(tables_dir / "discrimination_auc.tsv"),
                    "calibration_stats": str(tables_dir / "calibration_stats.tsv"),
                    "roc_plot": str(figs_dir / "roc_by_group.png"),
                    "calibration_plot": str(figs_dir / "calibration_by_group.png"),
                    "auc_comparison_plot": str(figs_dir / "auc_comparison.png"),
                },
            }

            # Novel Algorithm 2: BGR
            if run_bgr:
                try:
                    df_recal, params_df = bayesian_group_recalibration(
                        df_clin, n_threshold=bgr_n_threshold
                    )
                    params_df.to_csv(tables_dir / "bgr_parameters.tsv", sep="\t", index=False)
                    df_recal.to_csv(tables_dir / "scores_bgr.tsv", sep="\t", index=False)

                    eval_df = evaluate_recalibration(df_clin, df_recal)
                    eval_df.to_csv(tables_dir / "bgr_evaluation.tsv", sep="\t", index=False)

                    plot_recalibration_comparison(df_recal, eval_df,
                                                  figs_dir / "bgr_comparison.png")
                    plot_bgr_parameters(params_df, figs_dir / "bgr_parameters.png")

                    bgr_meta = {
                        "n_threshold": bgr_n_threshold,
                        "n_groups_bgr": int((params_df["recal_method"] == "BGR_MAP").sum()),
                        "n_groups_global_fallback": int((params_df["recal_method"] == "global_fallback").sum()),
                        "artifacts": {
                            "bgr_parameters": str(tables_dir / "bgr_parameters.tsv"),
                            "bgr_scores": str(tables_dir / "scores_bgr.tsv"),
                            "bgr_evaluation": str(tables_dir / "bgr_evaluation.tsv"),
                            "bgr_comparison_plot": str(figs_dir / "bgr_comparison.png"),
                            "bgr_parameters_plot": str(figs_dir / "bgr_parameters.png"),
                        },
                    }
                except Exception as e:
                    bgr_meta["error"] = str(e)

    # ── Metadata ──────────────────────────────────────────────────────────
    meta = {
        "inputs": {"scores": str(scores_path), "groups": str(groups_path)},
        "n_samples_scored": int(scores.shape[0]),
        "n_samples_with_groups": int(df.shape[0]),
        "standardize": standardize,
        "cutoff": {"value": c_val, "kind": c_kind},
        "equity": disparity_meta,
        "aps_distributional": aps_result,
        "clinical_validity": calibration_meta,
        "bgr": bgr_meta,
        "rcft": {
            "budget": rcft_budget,
            "criterion": rcft_criterion,
            "disparity_ratio_rcft": rcft_df["disparity_ratio_rcft"].iloc[0] if not rcft_df.empty else None,
            "disparity_ratio_naive": rcft_df["disparity_ratio_naive"].iloc[0] if not rcft_df.empty else None,
        },
        "artifacts": {
            "scores_with_groups": str(tables_dir / "scores_with_groups.tsv"),
            "summary_by_group": str(tables_dir / "summary_by_group.tsv"),
            "pairwise_smds": str(tables_dir / "pairwise_smds.tsv"),
            "ks_tests": str(tables_dir / "ks_tests.tsv"),
            "bootstrap_stats": str(tables_dir / "bootstrap_stats.tsv"),
            "sensitivity_curve": str(tables_dir / "sensitivity_curve.tsv"),
            "equalized_cutoffs": str(tables_dir / "equalized_cutoffs.tsv") if c_val else None,
            "rcft_thresholds": str(tables_dir / "rcft_thresholds.tsv"),
            "distribution_plot": str(figs_dir / "score_distributions_kde.png"),
            "sensitivity_plot": str(figs_dir / "sensitivity_curve.png"),
            "disparity_plot": str(figs_dir / "disparity_curve.png"),
            "smd_heatmap": str(figs_dir / "smd_heatmap.png"),
            "bootstrap_plot": str(figs_dir / "bootstrap_means.png"),
            "equalized_cutoffs_plot": str(figs_dir / "equalized_cutoffs.png") if c_val else None,
            "rcft_plot": str(figs_dir / "rcft_thresholds.png"),
            "aps_plot": str(figs_dir / "aps_gauge.png"),
        },
    }
    write_json(logs_dir / "evaluation_metadata.json", meta)
    return meta


# ── Helper plots for new algorithms ──────────────────────────────────────

def _plot_rcft(rcft_df: pd.DataFrame, out_path: Path) -> None:
    """Compare naive vs RCFT flagging rates per group."""
    import matplotlib.pyplot as plt

    _GROUP_COLORS = {
        "AFR": "#1D7874", "AMR": "#F5A623", "EAS": "#065A82",
        "EUR": "#8B4CA8", "SAS": "#D94F3D",
    }

    if rcft_df.empty:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = rcft_df["group"].tolist()
    x = np.arange(len(groups))
    w = 0.3
    colors = [_GROUP_COLORS.get(g, "#888888") for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.bar(x - w/2, rcft_df["naive_flagging_rate"] * 100, w,
           label="Global cutoff", color="#999999", alpha=0.8, edgecolor="white")
    ax.bar(x + w/2, rcft_df["rcft_flagging_rate"] * 100, w,
           label="RCFT", color=colors, alpha=0.85, edgecolor="white")
    budget = rcft_df["budget"].iloc[0]
    ax.axhline(budget * 100, color="#C0392B", lw=1.5, ls="--",
               label=f"Budget ({budget:.0%})")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("% flagged")
    ax.set_title("Flagging rates: global vs RCFT", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.bar(x - w/2, rcft_df["naive_cutoff"], w,
           label="Global cutoff", color="#999999", alpha=0.8, edgecolor="white")
    ax.bar(x + w/2, rcft_df["rcft_cutoff"], w,
           label="RCFT threshold", color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Score threshold (standardized PRS)")
    ax.set_title("Thresholds: global vs RCFT", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    dr_naive = rcft_df["disparity_ratio_naive"].iloc[0]
    dr_rcft = rcft_df["disparity_ratio_rcft"].iloc[0]
    dr_naive_str = f"{dr_naive:.2f}" if dr_naive is not None and not (isinstance(dr_naive, float) and np.isnan(dr_naive)) else "∞"
    dr_rcft_str = f"{dr_rcft:.2f}" if dr_rcft is not None and not (isinstance(dr_rcft, float) and np.isnan(dr_rcft)) else "∞"
    fig.suptitle(
        f"Resource-Constrained Fair Threshold  |  "
        f"Disparity ratio: {dr_naive_str} → {dr_rcft_str}",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_aps(aps_result: Dict, out_path: Path) -> None:
    """Horizontal bar showing APS with CI."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    aps = aps_result.get("aps_point", float("nan"))
    lo = aps_result.get("aps_ci_lo", float("nan"))
    hi = aps_result.get("aps_ci_hi", float("nan"))
    interp = aps_result.get("interpretation", "")

    if np.isnan(aps):
        return

    fig, ax = plt.subplots(figsize=(7, 2.5))

    # Background gradient bar
    for i, (xstart, color) in enumerate([(0, "#FDECEA"), (0.5, "#FFF8E1"), (0.75, "#E8F5E9"), (0.9, "#C8E6C9")]):
        xend = [0.5, 0.75, 0.9, 1.0][i]
        ax.barh(0, xend - xstart, left=xstart, height=0.5,
                color=color, edgecolor="none", zorder=1)

    # CI bar
    if not np.isnan(lo) and not np.isnan(hi):
        ax.barh(0, hi - lo, left=lo, height=0.22, color="#065A82",
                alpha=0.35, zorder=2, label=f"95% CI [{lo:.2f}, {hi:.2f}]")

    # Point estimate
    color = "#C0392B" if aps < 0.5 else "#E67E22" if aps < 0.75 else "#27AE60"
    ax.plot(aps, 0, "D", color=color, ms=12, zorder=3, label=f"APS = {aps:.3f}")

    # Zone labels
    for xpos, label in [(0.25, "Very low"), (0.62, "Low"), (0.82, "Moderate"), (0.95, "High")]:
        ax.text(xpos, -0.35, label, ha="center", va="top", fontsize=8, color="#555555")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Ancestry Portability Score (APS)", fontsize=10)
    ax.set_yticks([])
    ax.set_title(f"APS = {aps:.3f}  —  {interp}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
