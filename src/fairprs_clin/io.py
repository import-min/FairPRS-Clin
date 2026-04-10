from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .utils import read_table

def load_groups(groups_path: Path) -> pd.DataFrame:
    df = read_table(groups_path)
    # flexible column naming
    cols = {c.lower(): c for c in df.columns}
    iid_col = cols.get("iid") or cols.get("sample") or cols.get("id")
    grp_col = cols.get("group") or cols.get("ancestry") or cols.get("super_population") or cols.get("population")
    if iid_col is None or grp_col is None:
        raise ValueError(f"Groups file must contain IID and group columns. Found columns: {list(df.columns)}")
    out = df[[iid_col, grp_col]].copy()
    out.columns = ["IID", "group"]
    out["IID"] = out["IID"].astype(str)
    out["group"] = out["group"].astype(str)
    return out

def load_scores(scores_path: Path, score_column: Optional[str] = None) -> pd.DataFrame:
    # PLINK2 .sscore is space/tab-delimited and includes IID and SCORE1_SUM or SCORE1_AVG
    if scores_path.suffix.lower() in [".sscore", ".txt", ".tsv", ".tab"]:
        # try whitespace first
        try:
            df = pd.read_csv(scores_path, sep=r"\s+")
        except Exception:
            df = read_table(scores_path)
    else:
        df = read_table(scores_path)

    cols_lower = {c.lower(): c for c in df.columns}
    iid_col = cols_lower.get("iid") or cols_lower.get("sample") or cols_lower.get("id")
    if iid_col is None:
        # sometimes FID/IID exist
        if "IID" in df.columns:
            iid_col = "IID"
        elif "iid" in df.columns:
            iid_col = "iid"
        else:
            raise ValueError(f"Could not find IID column in scores file. Columns: {list(df.columns)}")

    if score_column is None:
        # common plink2 names
        candidates = [c for c in df.columns if c.upper().startswith("SCORE") and ("SUM" in c.upper() or "AVG" in c.upper())]
        if len(candidates) == 0:
            # fallback: common "score"
            if "score" in cols_lower:
                candidates = [cols_lower["score"]]
        if len(candidates) == 0:
            raise ValueError(f"Could not infer score column. Provide --score-column. Columns: {list(df.columns)}")
        score_col = candidates[0]
    else:
        score_col = score_column
        if score_col not in df.columns:
            raise ValueError(f"score_column='{score_col}' not found. Columns: {list(df.columns)}")

    out = df[[iid_col, score_col]].copy()
    out.columns = ["IID", "SCORE"]
    out["IID"] = out["IID"].astype(str)
    out["SCORE"] = pd.to_numeric(out["SCORE"], errors="coerce")
    out = out.dropna(subset=["SCORE"])
    return out
