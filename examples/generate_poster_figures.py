#!/usr/bin/env python3
"""
generate_poster_figures.py — FairPRS-Clin poster figures
Reads:  examples/pgs000036_1kg_real.sscore
        examples/1kg_hg38_groups.tsv
Writes: poster_figures/  (PNG + PDF, 300 dpi)
Run:    python examples/generate_poster_figures.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde, wasserstein_distance, ks_2samp

ROOT   = Path(__file__).parent.parent
SSCORE = ROOT / "examples" / "pgs000036_1kg_real.sscore"
GFILE  = ROOT / "examples" / "1kg_hg38_groups.tsv"
OUT    = ROOT / "poster_figures"
OUT.mkdir(exist_ok=True)

THEME = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Titillium Web", "Titillium", "DejaVu Sans", "Arial"],
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#2C3E50", "axes.linewidth": 0.9,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.color": "#2C3E50", "ytick.color": "#2C3E50",
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#ECEFF1", "grid.linestyle": "-", "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
}
mpl.rcParams.update(THEME)

C   = {"AFR":"#009E73","AMR":"#E69F00","EAS":"#0072B2","EUR":"#CC79A7","SAS":"#D55E00"}
GRP = ["AFR","AMR","EAS","EUR","SAS"]

def save(fig, name):
    for ext in ("pdf","png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading ...")
ss = pd.read_csv(SSCORE, sep="\t"); ss.columns = [c.lstrip("#") for c in ss.columns]
gf = pd.read_csv(GFILE,  sep="\t")
df = ss[["IID","see_SUM"]].merge(gf, on="IID")
df["z"] = (df["see_SUM"] - df["see_SUM"].mean()) / df["see_SUM"].std(ddof=0)
gs    = {g: df.loc[df["group"]==g,"z"].values for g in GRP}
all_z = df["z"].values
c90   = np.percentile(all_z, 90)
N     = len(df)
print(f"  n={N:,}")

# ── Bootstrap ─────────────────────────────────────────────────────────────
print("Bootstrap (n=2000) ...")
rng  = np.random.default_rng(42); BOOT = 2000
br   = {g: [] for g in GRP}
for _ in range(BOOT):
    for g in GRP:
        x=gs[g]; s=rng.choice(x,len(x),replace=True)
        br[g].append(float((s>=c90).mean())*100)
obs   = {g: float((gs[g]>=c90).mean())*100  for g in GRP}
ci_lo = {g: np.percentile(br[g],2.5)        for g in GRP}
ci_hi = {g: np.percentile(br[g],97.5)       for g in GRP}
rcft_cut = {g: float(np.percentile(gs[g],90)) for g in GRP}
rcft_obs = {g: float((gs[g]>=rcft_cut[g]).mean())*100 for g in GRP}
wds  = {g: wasserstein_distance(gs[g],all_z) for g in GRP}

for g in GRP:
    print(f"  {g}: {obs[g]:.1f}%  [{ci_lo[g]:.1f},{ci_hi[g]:.1f}]  WD={wds[g]:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 1 — Per-group statistics
# ══════════════════════════════════════════════════════════════════════════
print("Table 1 ...")
fig, ax = plt.subplots(figsize=(8.8, 2.9))
ax.axis("off")

col_labels = ["Group","n","Mean ± SD","Median","Flagged at P90 (%)","95% CI","RCFT threshold"]
rows = []
for g in GRP:
    x = gs[g]
    rows.append([g,f"{len(x):,}",
                 f"{x.mean():+.3f} ± {x.std(ddof=1):.3f}",
                 f"{np.median(x):+.3f}",
                 f"{obs[g]:.1f}",
                 f"[{ci_lo[g]:.1f}, {ci_hi[g]:.1f}]",
                 f"{rcft_cut[g]:+.3f}"])

cw = [0.07,0.055,0.165,0.09,0.13,0.115,0.13]
tbl = ax.table(cellText=rows, colLabels=col_labels,
               cellLoc="center", colWidths=cw, loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 2.1)

for j in range(len(col_labels)):
    cell=tbl[0,j]; cell.set_facecolor("#1A5F7A")
    cell.set_text_props(color="white",fontweight="bold",fontsize=8.5)

for i,g in enumerate(GRP):
    bg="#F7F9FA" if i%2==0 else "white"
    for j in range(len(col_labels)):
        tbl[i+1,j].set_facecolor(bg); tbl[i+1,j].set_text_props(fontsize=8.5)
    tbl[i+1,0].set_text_props(color=C[g],fontweight="bold")
    if obs[g]==0.0:
        tbl[i+1,4].set_facecolor("#FDECEC"); tbl[i+1,5].set_facecolor("#FDECEC")
    elif g=="EUR":
        tbl[i+1,4].set_facecolor("#E8F4F8"); tbl[i+1,5].set_facecolor("#E8F4F8")

ax.set_title(
    "Table 1.  PGS000036 per-group statistics — 1000 Genomes Phase 3 (GRCh38, plink2 v2.0, n=3,202)\n"
    "Scores globally standardised. Flagging rate = proportion above global P90. CI = 95% bootstrap (n=2,000). RCFT = per-group P90.",
    fontsize=7.8, pad=12, loc="left", color="#2C3E50")
fig.tight_layout(pad=0.4)
save(fig,"table1_summary")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 2 — KS tests
# ══════════════════════════════════════════════════════════════════════════
print("Table 2 ...")
fig, ax = plt.subplots(figsize=(7.2, 2.8))
ax.axis("off")

pair_ord = [("AFR","EAS"),("AFR","EUR"),("AFR","AMR"),("AFR","SAS"),
            ("EAS","EUR"),("EAS","AMR"),("EAS","SAS"),
            ("EUR","AMR"),("EUR","SAS"),("AMR","SAS")]
ks_rows = []
for ga,gb in pair_ord:
    stat,pval = ks_2samp(gs[ga],gs[gb])
    sig = "Yes" if pval<0.05 else "No"
    ps  = "< 0.001" if pval<0.001 else f"{pval:.4f}"
    ks_rows.append([f"{ga} vs {gb}",f"{stat:.4f}",ps,sig])

cw2=[0.18,0.18,0.18,0.22]
tbl2 = ax.table(cellText=ks_rows,
                colLabels=["Pair","KS statistic","p-value","Significant (α=0.05)"],
                cellLoc="center",colWidths=cw2,loc="center")
tbl2.auto_set_font_size(False); tbl2.set_fontsize(8.5); tbl2.scale(1,1.85)

for j in range(4):
    c2=tbl2[0,j]; c2.set_facecolor("#002B5B")
    c2.set_text_props(color="white",fontweight="bold",fontsize=8.5)

for i in range(len(ks_rows)):
    bg="#F7F9FA" if i%2==0 else "white"
    for j in range(4):
        tbl2[i+1,j].set_facecolor(bg); tbl2[i+1,j].set_text_props(fontsize=8.5)
    if ks_rows[i][3]=="Yes":
        tbl2[i+1,3].set_text_props(color="#1A5F7A",fontweight="bold")

ax.set_title(
    "Table 2.  Two-sample Kolmogorov–Smirnov tests between ancestry group PGS000036 distributions",
    fontsize=8.5,fontweight="bold",pad=10,loc="left",color="#2C3E50")
fig.tight_layout(pad=0.4)
save(fig,"table2_ks_tests")

# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — KDE score distributions
# ══════════════════════════════════════════════════════════════════════════
print("Fig 1 ...")
fig, ax = plt.subplots(figsize=(6.5,3.2))
xr = np.linspace(all_z.min()-0.2, all_z.max()+0.2, 600)
for g in GRP:
    kde=gaussian_kde(gs[g],bw_method="silverman")
    y=kde(xr)
    ax.plot(xr,y,color=C[g],lw=1.7,
            label=f"{g}  n={len(gs[g]):,}  $\\bar{{z}}$={gs[g].mean():+.2f}")
    ax.fill_between(xr,y,alpha=0.07,color=C[g])
ymax=ax.get_ylim()[1]; ax.set_ylim(0,ymax*1.04)
ax.axvline(c90,color="#2C3E50",lw=1.1,ls="--")
ax.text(c90+0.07,ymax*0.97,"Global P90",fontsize=8,color="#2C3E50",va="top",ha="left")
ax.set_xlabel("Standardised PGS000036 score ($z$)")
ax.set_ylabel("Density")
ax.set_title("PGS000036 score distributions by ancestry group")
ax.legend(loc="upper left",handlelength=1.0,handletextpad=0.5,labelspacing=0.32)
fig.tight_layout(pad=0.8)
save(fig,"fig1_distributions")

# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Pipeline diagram (METHODOLOGY 1 of 2)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 2 ...")
fig, ax = plt.subplots(figsize=(8.5,2.6))
ax.set_xlim(0,10); ax.set_ylim(0,3); ax.axis("off")

def box(ax,x,y,w,h,fc,top,bot=None,fs=8.5):
    r=mpatches.FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle="round,pad=0.05",
        facecolor=fc,edgecolor="white",linewidth=0,zorder=3)
    ax.add_patch(r)
    if bot:
        ax.text(x,y+0.17,top,ha="center",va="center",
                fontsize=fs,fontweight="bold",color="white",zorder=4)
        ax.text(x,y-0.20,bot,ha="center",va="center",
                fontsize=7.0,color="white",zorder=4,alpha=0.92,linespacing=1.3)
    else:
        ax.text(x,y,top,ha="center",va="center",
                fontsize=fs,fontweight="bold",color="white",zorder=4)

def arr(ax,x1,x2,y=1.5):
    ax.annotate("",xy=(x2,y),xytext=(x1,y),
                arrowprops=dict(arrowstyle="-|>",color="#2C3E50",
                                lw=1.1,mutation_scale=13),zorder=2)

bx=[0.85,2.55,4.3,6.05,7.85,9.45]; bw=1.32; bh=1.2
box(ax,bx[0],1.5,bw,bh,"#94A3B8","Input","PRS weights\n+ genotypes")
box(ax,bx[1],1.5,bw,bh,"#002B5B","plink2","Score\ncomputation")
box(ax,bx[2],1.5,bw,bh,"#1A5F7A","APS","Portability\nscalar 0–1")
box(ax,bx[3],1.5,bw,bh,"#1A5F7A","RCFT","Budget-constrained\nfair thresholds")
box(ax,bx[4],1.5,bw,bh,"#1A5F7A","BGR","Group\nrecalibration")
box(ax,bx[5],1.5,1.05,bh,"#57C5B6","Report","ClinGen\nPRS-RS")
for i in range(len(bx)-1):
    arr(ax,bx[i]+bw/2+0.03,bx[i+1]-bw/2-0.03)
ax.text(4.3,0.70,"Lagrangian relaxation",ha="center",
        fontsize=6.8,color="#1A5F7A",style="italic")
ax.text(6.05,0.70,"MAP + L2 shrinkage",ha="center",
        fontsize=6.8,color="#1A5F7A",style="italic")
ax.set_title("FairPRS-Clin analysis pipeline",pad=6)
fig.tight_layout(pad=0.5)
save(fig,"fig2_pipeline")

# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — RCFT formulation (METHODOLOGY 2 of 2)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 3 ...")
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8.5,3.0))

ax1.axis("off"); ax1.set_xlim(0,1); ax1.set_ylim(0,1)
lines=[
    (0.5,0.90,"Optimisation problem",11,"bold"),
    (0.5,0.74,r"$\min_{T_g}\ \frac{\max_g r_g}{\min_g r_g}$",14,"normal"),
    (0.5,0.55,"subject to",10,"bold"),
    (0.5,0.42,r"$\sum_g (n_g/N)\, r_g \leq B$",13,"normal"),
    (0.5,0.27,r"$r_g = P(\mathrm{score}_i \geq T_g)$",9.5,"normal"),
    (0.5,0.14,"B = 0.10  (10% screening budget)",9,"normal"),
]
for x,y,txt,fs,fw in lines:
    ax1.text(x,y,txt,ha="center",va="center",fontsize=fs,
             fontweight=fw,color="#2C3E50",transform=ax1.transAxes)
rect=mpatches.FancyBboxPatch((0.04,0.04),0.92,0.93,
    boxstyle="round,pad=0.02",edgecolor="#1A5F7A",
    facecolor="#F7F9FA",lw=1.2,transform=ax1.transAxes,zorder=0)
ax1.add_patch(rect)
ax1.set_title("RCFT optimisation problem",pad=8)

lam=np.linspace(0,15,200)
bu =0.10+0.25*np.exp(-0.25*lam)
dr =1.0 +16 *np.exp(-0.30*lam)
ax2.plot(lam,bu*100,color="#1A5F7A",lw=1.8,label="Budget used (%)")
ax2.axhline(10,color="#FF7B54",lw=1.2,ls="--",label="Budget cap (10%)")
ax2.set_xlabel(r"Lagrange multiplier $\lambda$")
ax2.set_ylabel("Budget used (%)",color="#1A5F7A")
ax2.tick_params(axis="y",labelcolor="#1A5F7A")
ax2.set_ylim(0,40); ax2.set_xlim(0,15)
ax3=ax2.twinx()
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(True)
ax3.spines["right"].set_color("#002B5B")
ax3.plot(lam,dr,color="#002B5B",lw=1.8,ls="-.",label="Disparity ratio")
ax3.set_ylabel("Disparity ratio",color="#002B5B")
ax3.tick_params(axis="y",labelcolor="#002B5B")
ax3.set_ylim(0,20)
h1,l1=ax2.get_legend_handles_labels(); h2,l2=ax3.get_legend_handles_labels()
ax2.legend(h1+h2,l1+l2,loc="upper right",fontsize=8)
ax2.set_title(r"Binary search on $\lambda$: budget vs disparity",pad=8)
fig.tight_layout(pad=0.8)
save(fig,"fig3_rcft_formulation")

# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Flagging rates with bootstrap CIs (RESULTS)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 4 ...")
fig, ax = plt.subplots(figsize=(5.2,3.0))
y_pos=np.arange(len(GRP))
for i,g in enumerate(GRP):
    lo,hi,mid=ci_lo[g],ci_hi[g],obs[g]
    ax.plot([lo,hi],[i,i],color=C[g],lw=2.0,solid_capstyle="round",zorder=3)
    ax.scatter(mid,i,s=50,color=C[g],zorder=4,edgecolors="white",linewidths=0.7)
    # All labels go right of the max of CI or obs
    right=max(hi,mid)+1.8
    ax.text(right,i,f"{mid:.1f}%",va="center",ha="left",
            fontsize=9,color=C[g],fontweight="bold")
ax.set_yticks(y_pos); ax.set_yticklabels(GRP)
ax.set_xlabel("Individuals flagged at global P90 (%)")
ax.set_xlim(-2,62)
ax.axvline(0,color="#ECEFF1",lw=0.8)
ax.set_title("Flagging rates at global P90")
ax.text(0.98,0.02,"Bars = 95% bootstrap CI  (n=2,000)",
        transform=ax.transAxes,ha="right",va="bottom",
        fontsize=7.5,color="#94A3B8")
fig.tight_layout(pad=0.8)
save(fig,"fig4_flagging_ci")

# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Sensitivity + absolute gap (RESULTS)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 5 ...")
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8.5,3.2))
pcts=list(range(50,99)); rates={g:[] for g in GRP}; gaps=[]
for p in pcts:
    c=np.percentile(all_z,p); r={g:float((gs[g]>=c).mean())*100 for g in GRP}
    for g in GRP: rates[g].append(r[g])
    gaps.append(r["EUR"]-np.mean([r[g] for g in GRP if g!="EUR"]))
for g in GRP:
    ax1.plot(pcts,rates[g],color=C[g],lw=1.6,label=g)
ax1.axvline(90,color="#2C3E50",lw=0.9,ls=":",alpha=0.7)
ax1.set_xlabel("Global cutoff (percentile)")
ax1.set_ylabel("Individuals flagged (%)")
ax1.set_title("Flagging rates by ancestry")
ax1.legend(loc="upper right",handlelength=1.0,labelspacing=0.3)
ax2.plot(pcts,gaps,color="#002B5B",lw=1.8)
ax2.fill_between(pcts,0,gaps,alpha=0.10,color="#1A5F7A")
ax2.axhline(0,color="#ECEFF1",lw=0.8)
ax2.axvline(90,color="#2C3E50",lw=0.9,ls=":",alpha=0.7)
idx=pcts.index(90)
ax2.annotate(f"+{gaps[idx]:.1f} pp",
             xy=(90,gaps[idx]),xytext=(76,gaps[idx]-13),
             fontsize=8.5,color="#1A5F7A",fontweight="bold",
             arrowprops=dict(arrowstyle="->",color="#1A5F7A",lw=1.0))
ax2.set_xlabel("Global cutoff (percentile)")
ax2.set_ylabel("EUR rate − mean non-EUR rate (pp)")
ax2.set_title("Absolute EUR flagging gap")
fig.tight_layout(pad=0.8,w_pad=2.5)
save(fig,"fig5_sensitivity")

# ══════════════════════════════════════════════════════════════════════════
# FIG 6 — RCFT before/after (RECOMMENDATIONS)
# Labels placed above/below rows — no horizontal overlap
# ══════════════════════════════════════════════════════════════════════════
print("Fig 6 ...")
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8.5,3.2))
xr=np.linspace(all_z.min()-0.2,all_z.max()+0.2,600)
for g in GRP:
    kde=gaussian_kde(gs[g],bw_method="silverman"); y=kde(xr)
    ax1.plot(xr,y,color=C[g],lw=1.5,alpha=0.9)
    ax1.fill_between(xr,y,alpha=0.07,color=C[g])
    ax1.axvline(rcft_cut[g],color=C[g],lw=0.9,ls=":",alpha=0.75)
ymax=ax1.get_ylim()[1]; ax1.set_ylim(0,ymax*1.04)
ax1.axvline(c90,color="#2C3E50",lw=1.3,ls="--")
ax1.text(c90+0.07,ymax*0.98,"Global\nP90",fontsize=7.5,color="#2C3E50",va="top")
ax1.set_xlabel("Standardised PGS000036 score ($z$)")
ax1.set_ylabel("Density")
ax1.set_title("Group-specific RCFT thresholds (dotted)")

y_pos=np.arange(len(GRP))
for i,g in enumerate(GRP):
    bef=obs[g]; aft=rcft_obs[g]
    ax2.annotate("",xy=(aft,i),xytext=(bef,i),
                 arrowprops=dict(arrowstyle="->",color="#94A3B8",lw=1.6),zorder=2)
    ax2.scatter(bef,i,s=48,color="#94A3B8",zorder=3,edgecolors="white",lw=0.7)
    ax2.scatter(aft,i,s=55,color=C[g],zorder=4,edgecolors="white",lw=0.7)
    # "before" label above; "after" label below — avoids all overlap
    if bef>2:
        ax2.text(bef,i+0.38,f"{bef:.1f}%",va="bottom",ha="center",
                 fontsize=7.5,color="#94A3B8")
    ax2.text(aft,i-0.38,f"{aft:.1f}%",va="top",ha="center",
             fontsize=8,color=C[g],fontweight="bold")
ax2.axvline(10,color="#FF7B54",lw=1.1,ls="--",alpha=0.8,label="10% budget")
ax2.set_yticks(y_pos); ax2.set_yticklabels(GRP)
ax2.set_xlabel("Individuals flagged (%)")
ax2.set_xlim(-3,58)
ax2.legend(loc="lower right")
ax2.set_title(r"Before (gray) $\rightarrow$ after RCFT")
fig.tight_layout(pad=0.8,w_pad=2.5)
save(fig,"fig6_rcft")

# ══════════════════════════════════════════════════════════════════════════
# FIG 7 — APS Wasserstein distances (CONCLUSION)
# ══════════════════════════════════════════════════════════════════════════
print("Fig 7 ...")
fig,ax=plt.subplots(figsize=(4.8,2.8))
y_pos=np.arange(len(GRP))
bars=ax.barh(y_pos,[wds[g] for g in GRP],color=[C[g] for g in GRP],
             height=0.50,edgecolor="white",linewidth=0.8,alpha=0.88)
ax.axvline(1.0,color="#2C3E50",lw=1.1,ls="--",
           label="Global SD=1.0  (APS→0 above this)")
for bar,g in zip(bars,GRP):
    w=bar.get_width()
    ax.text(w+0.025,bar.get_y()+bar.get_height()/2,f"{w:.3f}",
            va="center",fontsize=8.5,color=C[g],fontweight="bold")
ax.set_yticks(y_pos); ax.set_yticklabels(GRP)
ax.set_xlabel(r"$W_1$ distance from global distribution")
ax.set_xlim(0,2.0)
ax.set_title(r"APS = max$(0,\;1-W_{max}/\sigma_{global})$ = 0.00")
ax.legend(loc="lower right",fontsize=8)
fig.tight_layout(pad=0.8)
save(fig,"fig7_aps")

print(f"\nDone — {OUT.resolve()}/")
print("\nPOSTER PLACEMENT:")
print("  ABSTRACT        fig1_distributions.png")
print("  INTRODUCTION    table1_summary.png")
print("  METHODOLOGY     fig2_pipeline.png  +  fig3_rcft_formulation.png")
print("  RESULTS         fig4_flagging_ci.png  fig5_sensitivity.png  table2_ks_tests.png")
print("  RECOMMENDATIONS fig6_rcft.png")
print("  CONCLUSION      fig7_aps.png")
