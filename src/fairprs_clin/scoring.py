from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .utils import call, ensure_dir, read_table, sha256_file, write_json

def normalize_weights_file(weights_path: Path, out_path: Path) -> Dict:
    """Normalize PRS weights into PLINK2 --score-compatible format.

    Output columns:
      - ID (variant identifier matching PLINK variant IDs)
      - A1 (effect allele)
      - BETA (weight)

    Supported input formats:
      1) Simple TSV/CSV with: variant_id, effect_allele, weight
      2) PGS Catalog scoring file: tries common column names

    Returns metadata about parsing and overlap (overlap computed later).
    """
    df = read_table(weights_path)

    cols = {c.lower(): c for c in df.columns}

    # attempt PGS scoring file columns
    var_col = cols.get("variant_id") or cols.get("rsid") or cols.get("rs_id") or cols.get("rsid") or cols.get("hm_rsid")
    ea_col = cols.get("effect_allele") or cols.get("allele") or cols.get("hm_effect_allele") or cols.get("effectallele")
    w_col = cols.get("weight") or cols.get("effect_weight") or cols.get("beta") or cols.get("or") or cols.get("hm_effect_weight")

    # sometimes PGS files store chr/pos separately; in that case we can construct a chr:pos key
    if var_col is None and ("chr_name" in cols and "chr_position" in cols):
        chr_c = cols["chr_name"]; pos_c = cols["chr_position"]
        df["variant_id"] = df[chr_c].astype(str) + ":" + df[pos_c].astype(str)
        var_col = "variant_id"

    if var_col is None or ea_col is None or w_col is None:
        raise ValueError(
            "Weights file must include variant_id/rsid, effect_allele, and weight/effect_weight/beta. "
            f"Found columns: {list(df.columns)}"
        )

    out = df[[var_col, ea_col, w_col]].copy()
    out.columns = ["ID", "A1", "BETA"]

    # clean
    out["ID"] = out["ID"].astype(str)
    out["A1"] = out["A1"].astype(str).str.upper()
    out["BETA"] = pd.to_numeric(out["BETA"], errors="coerce")
    out = out.dropna(subset=["BETA"])

    ensure_dir(out_path.parent)
    out.to_csv(out_path, sep="\t", index=False)

    meta = {
        "weights_input": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "normalized_weights": str(out_path),
        "n_weights_rows": int(out.shape[0]),
        "columns_used": {"variant": var_col, "effect_allele": ea_col, "weight": w_col},
        "notes": "Output is PLINK2 --score compatible: ID A1 BETA",
    }
    return meta

def run_plink2_score(
    plink2_path: str,
    dataset_mode: str,
    dataset_prefix: Path,
    normalized_weights: Path,
    out_prefix: Path,
    score_cols: Tuple[int, int, int] = (1, 2, 3),
    extra_args: Optional[list] = None,
) -> Path:
    """Run PLINK2 scoring.

    dataset_mode: 'pfile' or 'bfile'
    dataset_prefix: prefix path (without extension) for the PLINK dataset
    normalized_weights: file with ID A1 BETA
    out_prefix: output prefix; PLINK2 writes .sscore
    score_cols: 1-based col indexes for ID, A1, BETA in the score file
    """
    extra_args = extra_args or []
    cmd = [plink2_path]
    if dataset_mode == "pfile":
        cmd += ["--pfile", str(dataset_prefix)]
    elif dataset_mode == "bfile":
        cmd += ["--bfile", str(dataset_prefix)]
    else:
        raise ValueError("dataset_mode must be 'pfile' or 'bfile'")

    # header-read ensures we can pass col indexes reliably; we pass 1 2 3 for ID A1 BETA
    cmd += [
        "--score", str(normalized_weights), str(score_cols[0]), str(score_cols[1]), str(score_cols[2]), "header-read", "cols=+scoresums",
        "--out", str(out_prefix),
    ]
    cmd += extra_args
    call(cmd)

    sscore = Path(str(out_prefix) + ".sscore")
    if not sscore.exists():
        raise FileNotFoundError(f"Expected PLINK2 output not found: {sscore}")
    return sscore
