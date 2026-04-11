"""
plots.py — publication-quality visualizations for FairPRS-Clin.

Functions:
  plot_distributions_kde    : per-group KDE + rug plot (replaces basic histogram)
  plot_sensitivity_curve    : flagging rate vs cutoff percentile per group
  plot_disparity_curve      : disparity ratio vs cutoff percentile
  plot_smd_heatmap          : pairwise SMD heatmap
  plot_bootstrap_ci_bars    : per-group mean ± CI bar chart
  plot_equalized_cutoffs    : grouped bar showing global vs equalized cutoffs
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .utils import ensure_dir

# Consistent color palette across all plots
_GROUP_COLORS = {
    "AFR": "#1D7874",
    "AMR": "#F5A623",
    "EAS": "#065A82",
    "EUR": "#8B4CA8",
    "SAS": "#D94F3D",
}
_FALLBACK_COLORS = plt.cm.tab10.colors  # type: ignore


def _group_color(grp: str, idx: int = 0) -> str:
    return _GROUP_COLORS.get(grp, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _save(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── KDE distribution plot ─────────────────────────────────────────────────

def plot_distributions_kde(
    df: pd.DataFrame,
    out_path: Path,
    score_col: str = "SCORE_STD",
    cutoff_value: Optional[float] = None,
    title: str = "PRS distributions by ancestry group",
) -> None:
    """Per-group KDE curves with optional cutoff line. Replaces the basic histogram."""
    from scipy.stats import gaussian_kde

    groups = sorted(df["group"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x_all = df[score_col].values
    x_range = np.linspace(x_all.min() - 0.5, x_all.max() + 0.5, 300)

    for i, grp in enumerate(groups):
        x = df.loc[df["group"] == grp, score_col].values
        if len(x) < 3:
            continue
        try:
            kde = gaussian_kde(x, bw_method="scott")
            y = kde(x_range)
        except Exception:
            continue
        color = _group_color(grp, i)
        ax.plot(x_range, y, color=color, lw=2.0, label=grp)
        ax.fill_between(x_range, y, alpha=0.12, color=color)

    if cutoff_value is not None:
        ax.axvline(cutoff_value, color="#C0392B", lw=1.8, ls="--",
                   label=f"cutoff = {cutoff_value:.2f}")

    ax.set_xlabel("Standardized PRS", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)


# ── Sensitivity curve ─────────────────────────────────────────────────────

def plot_sensitivity_curve(
    sensitivity_df: pd.DataFrame,
    out_path: Path,
    highlight_percentile: Optional[float] = None,
    title: str = "Flagging rate by ancestry group across cutoff percentiles",
) -> None:
    """Line plot: x = cutoff percentile, y = flagging rate, one line per group."""
    groups = sorted(sensitivity_df["group"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, grp in enumerate(groups):
        sub = sensitivity_df[sensitivity_df["group"] == grp].sort_values("percentile")
        ax.plot(sub["percentile"], sub["flagging_rate"] * 100,
                color=_group_color(grp, i), lw=2.0, label=grp)

    if highlight_percentile is not None:
        ax.axvline(highlight_percentile, color="#555555", lw=1.2, ls=":",
                   label=f"selected cutoff ({highlight_percentile:.0f}th pct)")

    ax.set_xlabel("Cutoff percentile (global)", fontsize=11)
    ax.set_ylabel("% individuals flagged", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)


# ── Disparity ratio curve ─────────────────────────────────────────────────

def plot_disparity_curve(
    sensitivity_df: pd.DataFrame,
    out_path: Path,
    highlight_percentile: Optional[float] = None,
    title: str = "Flagging disparity ratio across cutoff percentiles",
) -> None:
    """Single-line plot showing max/min flagging ratio vs cutoff percentile."""
    curve = (
        sensitivity_df.dropna(subset=["disparity_ratio"])
        .drop_duplicates(subset=["percentile"])
        .sort_values("percentile")
    )
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(curve["percentile"], curve["disparity_ratio"],
            color="#065A82", lw=2.2)
    ax.axhline(1.0, color="#aaaaaa", lw=1.0, ls="--")

    if highlight_percentile is not None:
        ax.axvline(highlight_percentile, color="#C0392B", lw=1.4, ls="--",
                   label=f"selected ({highlight_percentile:.0f}th pct)")
        ax.legend(fontsize=9)

    ax.set_xlabel("Cutoff percentile (global)", fontsize=11)
    ax.set_ylabel("Disparity ratio\n(max / min flagging rate)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)


# ── SMD heatmap ───────────────────────────────────────────────────────────

def plot_smd_heatmap(
    smd_df: pd.DataFrame,
    out_path: Path,
    title: str = "Pairwise standardized mean differences (Cohen's d)",
) -> None:
    """Symmetric heatmap of |SMD| between group pairs."""
    groups = sorted(set(smd_df["group_a"]) | set(smd_df["group_b"]))
    n = len(groups)
    idx = {g: i for i, g in enumerate(groups)}
    mat = np.zeros((n, n))
    for _, row in smd_df.iterrows():
        i, j = idx[row["group_a"]], idx[row["group_b"]]
        mat[i, j] = abs(row["smd"])
        mat[j, i] = abs(row["smd"])

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(mat, cmap="Blues", vmin=0)
    plt.colorbar(im, ax=ax, label="|SMD|")
    ax.set_xticks(range(n)); ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(groups)

    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if val > 0.5 else "black")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    _save(fig, out_path)


# ── Bootstrap CI bar chart ────────────────────────────────────────────────

def plot_bootstrap_ci_bars(
    boot_df: pd.DataFrame,
    out_path: Path,
    metric: str = "mean",
    ci_lo_col: str = "mean_ci_lo",
    ci_hi_col: str = "mean_ci_hi",
    ylabel: str = "Mean standardized PRS ± 95% CI",
    title: str = "Per-group mean PRS with bootstrap confidence intervals",
) -> None:
    """Bar chart with bootstrap error bars per group."""
    groups = boot_df["group"].tolist()
    vals = boot_df[metric].values
    lo = vals - boot_df[ci_lo_col].values
    hi = boot_df[ci_hi_col].values - vals
    colors = [_group_color(g, i) for i, g in enumerate(groups)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(groups, vals, yerr=[lo, hi], color=colors,
                  capsize=5, edgecolor="white", error_kw={"elinewidth": 1.5})
    ax.axhline(0, color="#888888", lw=0.8, ls="--")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)


# ── Equalized cutoffs comparison ──────────────────────────────────────────

def plot_equalized_cutoffs(
    eq_df: pd.DataFrame,
    global_cutoff: float,
    out_path: Path,
    title: str = "Global vs. equalized per-group cutoffs",
) -> None:
    """Grouped bar: global cutoff vs. equalized (per-group) cutoff per group."""
    groups = eq_df["group"].tolist()
    eq_vals = eq_df["equalized_cutoff"].values
    global_vals = np.full(len(groups), global_cutoff)

    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, global_vals, w, label="Global cutoff",
           color="#888888", alpha=0.75, edgecolor="white")
    ax.bar(x + w / 2, eq_vals, w, label="Equalized cutoff",
           color=[_group_color(g, i) for i, g in enumerate(groups)],
           alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Score threshold (standardized PRS)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)
