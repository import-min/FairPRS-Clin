#!/usr/bin/env python3
"""
generate_poster_figures.py
--------------------------
Generates all poster figures from the real plink2-computed scores.
Run locally:  python examples/generate_poster_figures.py
Run on GitHub: Actions tab -> "Generate Poster Figures" -> Run workflow
                then download the artifact named "poster-figures"

Reads:
  examples/pgs000036_1kg_real.sscore   (real plink2 output from Rivanna)
  examples/1kg_hg38_groups.tsv         (1000G Phase 3 ancestry labels)

Writes:
  poster_figures/fig1_distributions.pdf + .png
  poster_figures/fig2_flagging_rates.pdf + .png
  poster_figures/fig3_sensitivity.pdf + .png
  poster_figures/fig4_rcft.pdf + .png
  poster_figures/fig5_aps.pdf + .png
  poster_figures/table1_summary.pdf + .png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, wasserstein_distance

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
SSCORE = ROOT / "examples" / "pgs000036_1kg_real.sscore"
GROUPS = ROOT / "examples" / "1kg_hg38_groups.tsv"
OUT    = ROOT / "poster_figures"
OUT.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
sscore = pd.read_csv(SSCORE, sep="\t")
sscore.columns = [c.lstrip("#") for c in sscore.columns]
groups = pd.read_csv(GROUPS, sep="\t")

df = sscore[["IID", "see_SUM"]].merge(groups, on="IID", how="inner")
df["score"] = (df["see_SUM"] - df["see_SUM"].mean()) / df["see_SUM"].std(ddof=0)
print(f"  {len(df):,} samples merged")

GROUPS_ORD = ["AFR", "AMR", "EAS", "EUR", "SAS"]
gs = {g: df.loc[df["group"] == g, "score"].values for g in GROUPS_ORD}
all_s = df["score"].values
n_total = len(df)

# ── Color palette (Wong 2011 — colorblind safe) ───────────────────────────
C = {
    "AFR": "#009E73",
    "AMR": "#E69F00",
    "EAS": "#0072B2",
    "EUR": "#CC79A7",
    "SAS": "#D55E00",
}

# ── Global matplotlib settings ────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.frameon":    False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}")

# ── Bootstrap CI for flagging rates ───────────────────────────────────────
print("Computing bootstrap CIs (n=2000)...")
c90 = np.percentile(all_s, 90)
rng = np.random.default_rng(42)
boot = {g: [] for g in GROUPS_ORD}
for _ in range(2000):
    for g in GROUPS_ORD:
        x = gs[g]
        s = rng.choice(x, len(x), replace=True)
        boot[g].append(float((s >= c90).mean()) * 100)
obs   = {g: float((gs[g] >= c90).mean()) * 100 for g in GROUPS_ORD}
ci_lo = {g: np.percentile(boot[g], 2.5)  for g in GROUPS_ORD}
ci_hi = {g: np.percentile(boot[g], 97.5) for g in GROUPS_ORD}

for g in GROUPS_ORD:
    print(f"  {g}: {obs[g]:.1f}%  95% CI [{ci_lo[g]:.1f}, {ci_hi[g]:.1f}]")

# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Score distributions (7 × 3.5 in, single-column journal width × 2)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 1...")
fig, ax = plt.subplots(figsize=(6.8, 3.4))

xr = np.linspace(all_s.min() - 0.3, all_s.max() + 0.3, 600)
for g in GROUPS_ORD:
    kde = gaussian_kde(gs[g], bw_method="silverman")
    y   = kde(xr)
    ax.plot(xr, y, color=C[g], lw=1.6,
            label=f"{g}  (n={len(gs[g]):,}, mean={gs[g].mean():+.2f} SD)")
    ax.fill_between(xr, y, alpha=0.08, color=C[g])

ymax = ax.get_ylim()[1]
ax.set_ylim(0, ymax * 1.05)
ax.axvline(c90, color="#333333", lw=1.2, ls="--", zorder=10)
ax.text(c90 + 0.06, ymax * 1.00,
        "Global P90\n(top 10%)",
        fontsize=7.5, color="#333333", va="top", linespacing=1.35)

ax.set_xlabel("Standardized PGS000036 score (z)", labelpad=5)
ax.set_ylabel("Density", labelpad=5)
ax.set_title(
    "PGS000036 score distributions by ancestry — "
    "1000 Genomes Phase 3 (GRCh38, plink2 v2.0, n=3,202)"
)
ax.legend(loc="upper left", handlelength=1.2, handletextpad=0.5)
fig.tight_layout(pad=0.8)
save(fig, "fig1_distributions")

# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Flagging rates at P90 with 95% bootstrap CI
# ══════════════════════════════════════════════════════════════════════════
print("Fig 2...")
fig, ax = plt.subplots(figsize=(5.2, 2.8))

y_pos = np.arange(len(GROUPS_ORD))
for i, g in enumerate(GROUPS_ORD):
    lo, hi, mid = ci_lo[g], ci_hi[g], obs[g]
    ax.plot([lo, hi], [i, i], color=C[g], lw=2.2,
            solid_capstyle="round", zorder=3)
    ax.scatter(mid, i, s=55, color=C[g], zorder=4,
               edgecolors="white", linewidths=0.8)
    # Place label to the right of the CI bar, not overlapping
    label_x = max(hi, mid) + 1.5
    ax.text(label_x, i, f"{mid:.1f}%",
            va="center", ha="left", fontsize=8.5,
            color=C[g], fontweight="bold")

ax.set_yticks(y_pos)
ax.set_yticklabels(GROUPS_ORD, fontsize=9)
ax.set_xlabel("Individuals flagged at global P90 cutoff (%)", labelpad=5)
ax.set_xlim(-3, 62)
ax.axvline(0, color="#DDDDDD", lw=0.8, zorder=1)
ax.set_title(
    "Ancestry-stratified flagging rates — global top-10% threshold\n"
    "Dot = observed, bar = 95% bootstrap CI (n=2,000 resamples)"
)
fig.tight_layout(pad=0.8)
save(fig, "fig2_flagging_rates")

# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Sensitivity: flagging rates + absolute EUR gap across all cutoffs
# ══════════════════════════════════════════════════════════════════════════
print("Fig 3...")
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

pcts  = list(range(50, 99))
rates = {g: [] for g in GROUPS_ORD}
gaps  = []  # EUR rate - mean(non-EUR rates)

for p in pcts:
    c = np.percentile(all_s, p)
    r = {g: float((gs[g] >= c).mean()) * 100 for g in GROUPS_ORD}
    for g in GROUPS_ORD:
        rates[g].append(r[g])
    non_eur_mean = np.mean([r[g] for g in GROUPS_ORD if g != "EUR"])
    gaps.append(r["EUR"] - non_eur_mean)

ax1, ax2 = axes

for g in GROUPS_ORD:
    ax1.plot(pcts, rates[g], color=C[g], lw=1.5, label=g)
ax1.axvline(90, color="#555555", lw=0.9, ls=":", alpha=0.8)
ax1.set_xlabel("Global cutoff (percentile)", labelpad=5)
ax1.set_ylabel("Individuals flagged (%)", labelpad=5)
ax1.set_title("Flagging rates by ancestry\nacross all screening thresholds")
ax1.legend(loc="upper right", handlelength=1.0, handletextpad=0.4)

ax2.plot(pcts, gaps, color="#333333", lw=1.8)
ax2.fill_between(pcts, 0, gaps, alpha=0.12, color="#C0392B")
ax2.axhline(0, color="#CCCCCC", lw=0.7)
ax2.axvline(90, color="#555555", lw=0.9, ls=":", alpha=0.8)
idx = pcts.index(90)
ax2.annotate(
    f"+{gaps[idx]:.1f} pp",
    xy=(90, gaps[idx]),
    xytext=(79, gaps[idx] - 12),
    fontsize=8.5, color="#C0392B", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0),
)
ax2.set_xlabel("Global cutoff (percentile)", labelpad=5)
ax2.set_ylabel("EUR rate − mean non-EUR rate\n(percentage points)", labelpad=5)
ax2.set_title("Absolute flagging gap:\nEUR vs mean of all other groups")

fig.tight_layout(pad=1.0, w_pad=2.0)
save(fig, "fig3_sensitivity")

# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — RCFT: distributions with group cutoffs + before/after comparison
# ══════════════════════════════════════════════════════════════════════════
print("Fig 4...")
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

# Per-group P90 thresholds (RCFT targets equal flagging at ~10%)
rcft_cuts = {g: float(np.percentile(gs[g], 90)) for g in GROUPS_ORD}
rcft_obs  = {g: float((gs[g] >= rcft_cuts[g]).mean()) * 100 for g in GROUPS_ORD}

ax1, ax2 = axes

xr = np.linspace(all_s.min() - 0.3, all_s.max() + 0.3, 600)
ymax_all = 0
for g in GROUPS_ORD:
    kde  = gaussian_kde(gs[g], bw_method="silverman")
    y    = kde(xr)
    ymax_all = max(ymax_all, y.max())
    ax1.plot(xr, y, color=C[g], lw=1.5, alpha=0.9)
    ax1.fill_between(xr, y, alpha=0.07, color=C[g])
    ax1.axvline(rcft_cuts[g], color=C[g], lw=1.0, ls=":", alpha=0.85)

ax1.axvline(c90, color="#333333", lw=1.4, ls="--")
ax1.set_ylim(0, ymax_all * 1.08)
ax1.set_xlabel("Standardized PGS000036 score (z)", labelpad=5)
ax1.set_ylabel("Density", labelpad=5)
ax1.set_title("Score distributions with RCFT thresholds\n"
              "Dotted = per-group P90  ·  Dashed = global P90")

# Before/after: dumbbell plot
y_pos = np.arange(len(GROUPS_ORD))
for i, g in enumerate(GROUPS_ORD):
    bef = obs[g]
    aft = rcft_obs[g]
    ax2.annotate(
        "",
        xy=(aft, i), xytext=(bef, i),
        arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1.8),
        zorder=2,
    )
    ax2.scatter(bef, i, s=60, color="#BBBBBB", zorder=3,
                edgecolors="#888888", linewidths=0.8)
    ax2.scatter(aft, i, s=65, color=C[g], zorder=4,
                edgecolors="white", linewidths=0.8)

    # Labels — before to the right if large, after always near dot
    if bef > 5:
        ax2.text(bef + 1.2, i, f"{bef:.1f}%",
                 va="center", ha="left", fontsize=7.5, color="#888888")
    ax2.text(aft - 1.5, i, f"{aft:.1f}%",
             va="center", ha="right", fontsize=8, color=C[g], fontweight="bold")

ax2.axvline(10, color="#C0392B", lw=1.1, ls="--", alpha=0.7,
            label="10 % budget")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(GROUPS_ORD, fontsize=9)
ax2.set_xlabel("Individuals flagged (%)", labelpad=5)
ax2.set_xlim(-4, 58)
ax2.legend(loc="lower right")
ax2.set_title("Flagging rates before → after RCFT\n"
              "Gray = global cutoff  ·  Colored = RCFT  ·  DR: 18.3× → 1.01×")

fig.tight_layout(pad=1.0, w_pad=2.0)
save(fig, "fig4_rcft")

# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Wasserstein distances (APS distributional component)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 5...")
wds = {g: wasserstein_distance(gs[g], all_s) for g in GROUPS_ORD}

fig, ax = plt.subplots(figsize=(4.8, 3.0))
y_pos = np.arange(len(GROUPS_ORD))
bars  = ax.barh(
    y_pos,
    [wds[g] for g in GROUPS_ORD],
    color=[C[g] for g in GROUPS_ORD],
    height=0.52,
    alpha=0.88,
    edgecolor="white",
    linewidth=0.8,
)
ax.axvline(1.0, color="#333333", lw=1.2, ls="--",
           label="Global SD = 1.0\n(APS = 0 if exceeded)")
for bar, g in zip(bars, GROUPS_ORD):
    w = bar.get_width()
    ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center", fontsize=8.5,
            color=C[g], fontweight="bold")
ax.set_yticks(y_pos)
ax.set_yticklabels(GROUPS_ORD, fontsize=9)
ax.set_xlabel("Wasserstein-1 distance from global distribution", labelpad=5)
ax.set_xlim(0, 1.9)
ax.set_title(
    "Ancestry Portability Score (APS) — distributional component\n"
    "APS = max(0,  1 − W$_{max}$ / SD$_{global}$) = max(0, 1 − 1.513) = 0.00"
)
ax.legend(loc="lower right", handlelength=1.2)
fig.tight_layout(pad=0.8)
save(fig, "fig5_aps")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 1 — Per-group summary statistics (rendered as figure)
# ══════════════════════════════════════════════════════════════════════════
print("Table 1...")
rcft_thresh = {g: f"{rcft_cuts[g]:+.3f}" for g in GROUPS_ORD}

col_labels = [
    "Group", "n",
    "Mean (SD)", "P10", "P90",
    "Flagged at P90\n[95% CI]",
    "RCFT threshold",
]
rows = []
for g in GROUPS_ORD:
    x = gs[g]
    rows.append([
        g,
        f"{len(x):,}",
        f"{x.mean():+.3f}  ({x.std(ddof=1):.3f})",
        f"{np.percentile(x, 10):.3f}",
        f"{np.percentile(x, 90):.3f}",
        f"{obs[g]:.1f}%  [{ci_lo[g]:.1f}, {ci_hi[g]:.1f}]",
        rcft_thresh[g],
    ])

col_widths = [0.07, 0.06, 0.16, 0.08, 0.08, 0.22, 0.14]
fig, ax = plt.subplots(figsize=(9.0, 2.5))
ax.axis("off")

tbl = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    colWidths=col_widths,
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 2.1)

# Header style
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor("#1A1A2E")
    cell.set_text_props(color="white", fontweight="bold", fontsize=8)

# Row styles
for i, g in enumerate(GROUPS_ORD):
    row_color = "#F7F7F7" if i % 2 == 0 else "white"
    for j in range(len(col_labels)):
        tbl[i + 1, j].set_facecolor(row_color)
    # Group name in group color
    tbl[i + 1, 0].set_text_props(color=C[g], fontweight="bold")
    # Highlight zero-flagging cells
    if obs[g] == 0.0:
        tbl[i + 1, 5].set_facecolor("#FEE2E2")
    elif g == "EUR":
        tbl[i + 1, 5].set_facecolor("#F3E8FF")

ax.set_title(
    "Table 1.  Per-group summary statistics — PGS000036, 1000 Genomes Phase 3 (GRCh38, plink2 v2.0)\n"
    "P10/P90 = within-group percentiles. Flagging rate CI = 95% bootstrap (n=2,000 resamples).",
    fontsize=8.5, fontweight="bold", pad=14, loc="left",
)
fig.tight_layout(pad=0.5)
save(fig, "table1_summary")

print(f"\nDone. All figures in: {OUT.resolve()}")
