from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .io import load_groups, load_scores
from .utils import ensure_dir, write_json
from .equity import (
    pairwise_smds,
    ks_tests,
    bootstrap_group_stats,
    flagging_disparity,
    sensitivity_curve,
    equalized_cutoffs,
)
from .plots import (
    plot_distributions_kde,
    plot_sensitivity_curve,
    plot_disparity_curve,
    plot_smd_heatmap,
    plot_bootstrap_ci_bars,
    plot_equalized_cutoffs,
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
        n="count",
        mean="mean",
        sd=lambda x: x.std(ddof=0),
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()
    summ["iqr"] = summ["q75"] - summ["q25"]
    if cutoff_value is not None:
        flagged = g.apply(
            lambda d: (d["SCORE_STD"] >= cutoff_value).mean()
        ).reset_index(name="flagged_prop")
        flagged_n = g.apply(
            lambda d: int((d["SCORE_STD"] >= cutoff_value).sum())
        ).reset_index(name="flagged_n")
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
) -> Dict:
    out_dir = ensure_dir(out_dir)
    tables_dir = ensure_dir(out_dir / "tables")
    figs_dir = ensure_dir(out_dir / "figures")
    ensure_dir(out_dir / "reports")
    logs_dir = ensure_dir(out_dir / "logs")

    # ── Load and merge ────────────────────────────────────────────────────
    scores = load_scores(scores_path, score_column=score_column)
    groups = load_groups(groups_path)
    df = scores.merge(groups, on="IID", how="inner")
    if df.empty:
        raise ValueError(
            "No samples overlapped between scores and groups. Check IID formatting."
        )

    df = standardize_scores(df, how=standardize)
    c_val, c_kind = compute_cutoff(
        df, float(cutoff) if cutoff is not None else None, cutoff_percentile
    )

    # ── Basic summary ─────────────────────────────────────────────────────
    summ = summarize_by_group(df, cutoff_value=c_val)
    summ.to_csv(tables_dir / "summary_by_group.tsv", sep="\t", index=False)
    df.to_csv(tables_dir / "scores_with_groups.tsv", sep="\t", index=False)

    # ── Equity metrics ────────────────────────────────────────────────────
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

    # sensitivity curve (50th–99th percentile)
    sens_df = sensitivity_curve(df)
    sens_df.to_csv(tables_dir / "sensitivity_curve.tsv", sep="\t", index=False)

    # ── Figures ───────────────────────────────────────────────────────────
    plot_distributions_kde(
        df, figs_dir / "score_distributions_kde.png", cutoff_value=c_val
    )
    plot_sensitivity_curve(
        sens_df, figs_dir / "sensitivity_curve.png",
        highlight_percentile=cutoff_percentile,
    )
    plot_disparity_curve(
        sens_df, figs_dir / "disparity_curve.png",
        highlight_percentile=cutoff_percentile,
    )
    if len(smd_df) > 0:
        plot_smd_heatmap(smd_df, figs_dir / "smd_heatmap.png")
    if "mean_ci_lo" in boot_df.columns:
        plot_bootstrap_ci_bars(boot_df, figs_dir / "bootstrap_means.png")
    if c_val is not None and not eq_df.empty:
        plot_equalized_cutoffs(eq_df, c_val, figs_dir / "equalized_cutoffs.png")

    # ── Metadata ──────────────────────────────────────────────────────────
    meta = {
        "inputs": {"scores": str(scores_path), "groups": str(groups_path)},
        "n_samples_scored": int(scores.shape[0]),
        "n_samples_with_groups": int(df.shape[0]),
        "standardize": standardize,
        "cutoff": {"value": c_val, "kind": c_kind},
        "equity": disparity_meta,
        "artifacts": {
            "scores_with_groups": str(tables_dir / "scores_with_groups.tsv"),
            "summary_by_group": str(tables_dir / "summary_by_group.tsv"),
            "pairwise_smds": str(tables_dir / "pairwise_smds.tsv"),
            "ks_tests": str(tables_dir / "ks_tests.tsv"),
            "bootstrap_stats": str(tables_dir / "bootstrap_stats.tsv"),
            "sensitivity_curve": str(tables_dir / "sensitivity_curve.tsv"),
            "equalized_cutoffs": str(tables_dir / "equalized_cutoffs.tsv") if c_val else None,
            "distribution_plot": str(figs_dir / "score_distributions_kde.png"),
            "sensitivity_plot": str(figs_dir / "sensitivity_curve.png"),
            "disparity_plot": str(figs_dir / "disparity_curve.png"),
            "smd_heatmap": str(figs_dir / "smd_heatmap.png"),
            "bootstrap_plot": str(figs_dir / "bootstrap_means.png"),
            "equalized_cutoffs_plot": str(figs_dir / "equalized_cutoffs.png") if c_val else None,
        },
    }
    write_json(logs_dir / "evaluation_metadata.json", meta)
    return meta
