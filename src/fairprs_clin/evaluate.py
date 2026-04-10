from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .io import load_groups, load_scores
from .utils import ensure_dir, write_json

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
        out["SCORE_STD"] = out.groupby("group")["SCORE"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0))
        return out
    raise ValueError("standardize must be one of: global, within_group, none")

def compute_cutoff(df: pd.DataFrame, cutoff: Optional[float], cutoff_percentile: Optional[float]) -> Tuple[Optional[float], str]:
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
        flagged = g.apply(lambda d: (d["SCORE_STD"] >= cutoff_value).mean()).reset_index(name="flagged_prop")
        flagged_n = g.apply(lambda d: int((d["SCORE_STD"] >= cutoff_value).sum())).reset_index(name="flagged_n")
        summ = summ.merge(flagged, on="group").merge(flagged_n, on="group")
    else:
        summ["flagged_prop"] = np.nan
        summ["flagged_n"] = np.nan
    return summ

def plot_distributions(df: pd.DataFrame, out_path: Path, bins: int = 40) -> None:
    ensure_dir(out_path.parent)
    groups = sorted(df["group"].unique().tolist())
    plt.figure()
    for grp in groups:
        x = df.loc[df["group"] == grp, "SCORE_STD"].values
        plt.hist(x, bins=bins, alpha=0.35, density=True, label=str(grp))
    plt.xlabel("Standardized PRS (SCORE_STD)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def evaluate_scores(
    scores_path: Path,
    groups_path: Path,
    out_dir: Path,
    cutoff: Optional[str],
    cutoff_percentile: Optional[float],
    standardize: str,
    score_column: Optional[str] = None,
) -> Dict:
    out_dir = ensure_dir(out_dir)
    tables_dir = ensure_dir(out_dir / "tables")
    figs_dir = ensure_dir(out_dir / "figures")
    reports_dir = ensure_dir(out_dir / "reports")
    logs_dir = ensure_dir(out_dir / "logs")

    scores = load_scores(scores_path, score_column=score_column)
    groups = load_groups(groups_path)
    df = scores.merge(groups, on="IID", how="inner")
    if df.empty:
        raise ValueError("No samples overlapped between scores and groups. Check IID formatting.")

    df = standardize_scores(df, how=standardize)

    c_val, c_kind = compute_cutoff(df, float(cutoff) if cutoff is not None else None, cutoff_percentile)

    summ = summarize_by_group(df, cutoff_value=c_val)
    summ.to_csv(tables_dir / "summary_by_group.tsv", sep="\t", index=False)

    df.to_csv(tables_dir / "scores_with_groups.tsv", sep="\t", index=False)

    plot_distributions(df, figs_dir / "score_distributions.png")

    meta = {
        "inputs": {"scores": str(scores_path), "groups": str(groups_path)},
        "n_samples_scored": int(scores.shape[0]),
        "n_samples_with_groups": int(df.shape[0]),
        "standardize": standardize,
        "cutoff": {"value": c_val, "kind": c_kind},
        "artifacts": {
            "scores_with_groups": str(tables_dir / "scores_with_groups.tsv"),
            "summary_by_group": str(tables_dir / "summary_by_group.tsv"),
            "distribution_plot": str(figs_dir / "score_distributions.png"),
        },
    }
    write_json(logs_dir / "evaluation_metadata.json", meta)
    return meta
