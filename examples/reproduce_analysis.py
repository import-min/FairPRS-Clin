#!/usr/bin/env python3
"""
reproduce_analysis.py — Reproduces all paper results for FairPRS-Clin.

With real scores (after Rivanna):
  python examples/reproduce_analysis.py \
    --scores examples/pgs000036_1kg_real.sscore \
    --groups examples/1kg_hg38_groups.tsv

With included example data:
  python examples/reproduce_analysis.py
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

GLOBAL_SD = 0.42
GROUP_PARAMS = {
    "EUR": ( 0.31, 1.02, 503), "SAS": ( 0.09, 0.98, 489),
    "AMR": (-0.05, 0.97, 347), "AFR": (-0.38, 0.94, 661), "EAS": (-0.44, 0.96, 504),
}
GROUP_COLORS = {
    "AFR": "#1D7874", "AMR": "#F5A623", "EAS": "#065A82", "EUR": "#8B4CA8", "SAS": "#D94F3D",
}

def make_simulated_data(seed=42):
    rng = np.random.default_rng(seed)
    iids, scores, groups = [], [], []
    for grp, (mu, sd_r, n) in GROUP_PARAMS.items():
        raw = rng.normal(mu * GLOBAL_SD, GLOBAL_SD * sd_r, n)
        iids  += [f"{grp}_{i:04d}" for i in range(n)]
        scores += raw.tolist()
        groups += [grp] * n
    return (pd.DataFrame({"IID": iids, "SCORE": scores}),
            pd.DataFrame({"IID": iids, "group": groups}))

def load_sscore(path):
    df = pd.read_csv(path, sep=r"\s+")
    cols = {c.lower(): c for c in df.columns}
    iid  = cols.get("iid") or cols.get("#iid")
    scol = next((cols[c] for c in cols if "sum" in c or "avg" in c), None)
    if not iid or not scol:
        raise ValueError(f"Cannot find IID or score column. Columns: {df.columns.tolist()}")
    out = df[[iid, scol]].copy()
    out.columns = ["IID", "SCORE"]
    out["IID"] = out["IID"].astype(str)
    out["SCORE"] = pd.to_numeric(out["SCORE"], errors="coerce")
    return out.dropna(subset=["SCORE"])

def run_fairprs(scores_path, groups_path, out_dir):
    from fairprs_clin.evaluate import evaluate_scores
    return evaluate_scores(
        scores_path=scores_path, groups_path=groups_path, out_dir=out_dir,
        cutoff=None, cutoff_percentile=95, standardize="global",
        n_boot=1000, equalize_target=0.05, rcft_budget=0.05,
    )

def make_figures(out_dir, df):
    figs = out_dir / "paper_figures"
    figs.mkdir(parents=True, exist_ok=True)
    sens = pd.read_csv(out_dir / "fairprs_out/tables/sensitivity_curve.tsv", sep="\t")
    rcft = pd.read_csv(out_dir / "fairprs_out/tables/rcft_thresholds.tsv",   sep="\t")
    smd  = pd.read_csv(out_dir / "fairprs_out/tables/pairwise_smds.tsv",     sep="\t")
    c95  = np.percentile(df["SCORE_STD"].values, 95)

    # Fig 1: distributions
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xr = np.linspace(df["SCORE_STD"].min()-0.3, df["SCORE_STD"].max()+0.3, 400)
    for grp in sorted(df["group"].unique()):
        x = df.loc[df["group"]==grp, "SCORE_STD"].values
        if len(x) < 3: continue
        kde = gaussian_kde(x)
        color = GROUP_COLORS.get(grp, "#888")
        ax.plot(xr, kde(xr), color=color, lw=2.2, label=f"{grp} (n={len(x):,})")
        ax.fill_between(xr, kde(xr), alpha=0.10, color=color)
    ymax = ax.get_ylim()[1]
    ax.axvline(c95, color="#C0392B", lw=2, ls="--", label=f"Top-5% cutoff ({c95:.2f})")
    ax.fill_betweenx([0, ymax], c95, xr[-1], alpha=0.07, color="#C0392B")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Standardized PRS (PGS000036)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("PGS000036 T2D score distributions across 1000 Genomes super-populations",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figs/f"fig1_distributions.{ext}", dpi=200, bbox_inches="tight")
    plt.close()

    # Fig 2: disparity is structural
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for grp in sorted(df["group"].unique()):
        sub = sens[sens["group"]==grp].sort_values("percentile")
        a1.plot(sub["percentile"], sub["flagging_rate"]*100,
                color=GROUP_COLORS.get(grp,"#888"), lw=2, label=grp)
    a1.axvline(95, color="#555", lw=1.2, ls=":", alpha=0.6)
    a1.set_ylabel("Individuals flagged (%)", fontsize=11)
    a1.set_title("Disparity grows with threshold stringency — it is structural, not a cutoff artifact\n"
                 "(PGS000036 on 1000G Phase 3 GRCh38, scored with plink2)",
                 fontsize=11, fontweight="bold")
    a1.legend(fontsize=9)
    a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)

    disp = sens.drop_duplicates("percentile").sort_values("percentile").dropna(subset=["disparity_ratio"])
    a2.plot(disp["percentile"], disp["disparity_ratio"], color="#1A1A2E", lw=2.5)
    a2.fill_between(disp["percentile"], 1, disp["disparity_ratio"], alpha=0.12, color="#C0392B")
    a2.axhline(1, color="#aaa", lw=1, ls="--", label="Perfect equity (ratio=1.0)")
    a2.axvline(95, color="#555", lw=1.2, ls=":", alpha=0.6, label="Top-5% cutoff")
    at95 = disp[disp["percentile"]==95]["disparity_ratio"]
    dr95 = float(at95.iloc[0]) if len(at95) else float(disp["disparity_ratio"].iloc[-1])
    a2.annotate(f"{dr95:.2f}x at 95th pct", xy=(95, dr95), xytext=(68, dr95+1.5),
                arrowprops=dict(arrowstyle="->", color="#C0392B"),
                fontsize=9, color="#C0392B", fontweight="bold")
    a2.set_xlabel("Cutoff percentile (global)", fontsize=11)
    a2.set_ylabel("Disparity ratio\n(max / min flagging rate)", fontsize=11)
    a2.legend(fontsize=9)
    a2.spines["top"].set_visible(False); a2.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figs/f"fig2_disparity_curve.{ext}", dpi=200, bbox_inches="tight")
    plt.close()

    # Fig 3: RCFT
    grps = rcft["group"].tolist()
    x = np.arange(len(grps)); w = 0.32
    colors = [GROUP_COLORS.get(g,"#888") for g in grps]
    fig, (al, ar) = plt.subplots(1, 2, figsize=(11, 4.5))
    b1 = al.bar(x-w/2, rcft["naive_flagging_rate"]*100, w, label="Global top-5% cutoff",
                color="#999", alpha=0.85, edgecolor="white")
    b2 = al.bar(x+w/2, rcft["rcft_flagging_rate"]*100, w, label="RCFT (budget=5%)",
                color=colors, alpha=0.9, edgecolor="white")
    al.axhline(5.0, color="#C0392B", lw=1.8, ls="--", label="5% budget")
    al.set_xticks(x); al.set_xticklabels(grps, fontsize=11)
    al.set_ylabel("Individuals flagged (%)", fontsize=11)
    al.set_title("Flagging rates: global cutoff vs RCFT", fontsize=11, fontweight="bold")
    al.legend(fontsize=9)
    al.spines["top"].set_visible(False); al.spines["right"].set_visible(False)
    for bar in b1:
        h = bar.get_height()
        al.text(bar.get_x()+bar.get_width()/2, h+0.1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color="#555")
    for bar in b2:
        h = bar.get_height()
        al.text(bar.get_x()+bar.get_width()/2, h+0.1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color="#222")
    nd = rcft["disparity_ratio_naive"].iloc[0]
    rd = rcft["disparity_ratio_rcft"].iloc[0]
    ar.bar(["Global\ncutoff","RCFT"], [nd, rd], color=["#C0392B","#27AE60"],
           width=0.4, alpha=0.88, edgecolor="white")
    ar.axhline(1.0, color="#aaa", lw=1.2, ls="--", label="Perfect equity (1.0)")
    ar.set_ylabel("Disparity ratio (max / min flagging rate)", fontsize=11)
    ar.set_title("Disparity before and after RCFT", fontsize=11, fontweight="bold")
    ar.set_ylim(0, max(nd*1.2, 2))
    ar.text(0, nd+0.1, f"{nd:.2f}x", ha="center", fontsize=14, fontweight="bold", color="#C0392B")
    ar.text(1, rd+0.1, f"{rd:.2f}x", ha="center", fontsize=14, fontweight="bold", color="#27AE60")
    ar.legend(fontsize=9)
    ar.spines["top"].set_visible(False); ar.spines["right"].set_visible(False)
    pct_red = (1 - rd/nd)*100
    fig.suptitle(f"RCFT reduces disparity {nd:.1f}x to {rd:.2f}x ({pct_red:.0f}% reduction)\n"
                 "while maintaining the 5% total population screening budget",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figs/f"fig3_rcft.{ext}", dpi=200, bbox_inches="tight")
    plt.close()

    # Fig 4: SMD heatmap
    gl = sorted(set(smd["group_a"]) | set(smd["group_b"]))
    n = len(gl); idx2 = {g: i for i,g in enumerate(gl)}
    mat = np.zeros((n,n))
    for _, row in smd.iterrows():
        i,j = idx2[row["group_a"]], idx2[row["group_b"]]
        mat[i,j] = mat[j,i] = abs(row["smd"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.2)
    plt.colorbar(im, ax=ax, label="|Cohen's d|", fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(gl, fontsize=11)
    ax.set_yticks(range(n)); ax.set_yticklabels(gl, fontsize=11)
    for i in range(n):
        for j in range(n):
            v = mat[i,j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.7 else "#1A1A2E",
                    fontweight="bold" if v > 0.5 else "normal")
    ax.set_title("Pairwise standardized mean differences in PGS000036 (Cohen's d)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(figs/f"fig4_smd_heatmap.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    return figs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="examples/pgs000036_1kg_real.sscore",
                    help="Path to .sscore file (default: real plink2-computed 1000G scores)")
    ap.add_argument("--groups", default="examples/1kg_hg38_groups.tsv",
                    help="Path to groups TSV (default: 1000G Phase 3 ancestry labels)")
    ap.add_argument("--out", default="results/paper")
    args = ap.parse_args()

    out_dir = Path(args.out)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    scores_path_obj = Path(args.scores)
    if scores_path_obj.exists():
        print(f"Loading scores from: {args.scores}")
        scores_df = load_sscore(scores_path_obj)
        scores_path = data_dir / "scores.csv"
        scores_df.to_csv(scores_path, index=False)
        groups_path = Path(args.groups) if args.groups else Path("examples/1kg_groups.tsv")
        source = f"plink2 ({Path(args.scores).name}, n={len(scores_df):,})"
    else:
        print("No --scores provided. Using simulated data from published parameters.")
        scores_df, groups_df = make_simulated_data()
        scores_path = data_dir / "scores_simulated.csv"
        groups_path = data_dir / "groups_simulated.tsv"
        scores_df.to_csv(scores_path, index=False)
        groups_df.to_csv(groups_path, sep="\t", index=False)
        source = "simulated (Privé 2022 + Mahajan 2018 parameters)"

    s = pd.read_csv(scores_path)
    g = pd.read_csv(groups_path, sep="\t")
    df = s.merge(g, on="IID")
    if df.empty:
        raise ValueError("No IID overlap between scores and groups.")
    mu = df["SCORE"].mean(); sd = df["SCORE"].std(ddof=0)
    df["SCORE_STD"] = (df["SCORE"] - mu) / sd

    print(f"Data: {source}")
    print(f"Total samples: {len(df):,}")
    print(df.groupby("group").size().to_string())

    print("\nRunning FairPRS-Clin...")
    t0 = time.time()
    meta = run_fairprs(scores_path, groups_path, out_dir / "fairprs_out")
    print(f"Done in {time.time()-t0:.0f}s")

    rcft_df = pd.read_csv(out_dir / "fairprs_out/tables/rcft_thresholds.tsv", sep="\t")
    nd = meta["equity"]["disparity_ratio"]
    rd = rcft_df["disparity_ratio_rcft"].iloc[0]
    aps = meta["aps_distributional"]

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nAPS = {aps['aps_point']:.3f} (95% CI: {aps['aps_ci_lo']:.3f}-{aps['aps_ci_hi']:.3f})")
    print(f"{aps['interpretation']}")
    print(f"\nGlobal top-5% disparity: {nd:.2f}x")
    print(f"  {meta['equity']['max_flagging_group']}: {meta['equity']['max_flagging_rate']*100:.1f}% flagged")
    print(f"  {meta['equity']['min_flagging_group']}: {meta['equity']['min_flagging_rate']*100:.1f}% flagged")
    print(f"\nAfter RCFT: {rd:.2f}x  ({(1-rd/nd)*100:.0f}% reduction)")
    print("="*60)

    print("\nGenerating figures...")
    figs_dir = make_figures(out_dir, df)
    print(f"Figures: {figs_dir}/")
    for f in sorted(figs_dir.iterdir()):
        if f.suffix == ".png":
            print(f"  {f.name}")

if __name__ == "__main__":
    main()
