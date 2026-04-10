#!/usr/bin/env python3
"""Convert a 1000 Genomes panel file into IID->group TSV.

Typical 1000G sample panel includes columns like:
  sample  population  super_population  gender

Usage:
  python examples/1000g_panel_to_groups.py --panel integrated_call_samples_v3.20130502.ALL.panel --out groups.tsv --use super_population

"""
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="Path to 1000G .panel file")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--use", choices=["population", "super_population"], default="super_population")
    args = ap.parse_args()

    df = pd.read_csv(args.panel, sep="\t")
    sample_col = "sample" if "sample" in df.columns else df.columns[0]
    grp_col = args.use
    if grp_col not in df.columns:
        raise ValueError(f"Column '{grp_col}' not found. Columns: {list(df.columns)}")

    out = df[[sample_col, grp_col]].copy()
    out.columns = ["IID", "group"]
    out.to_csv(args.out, sep="\t", index=False)

if __name__ == "__main__":
    main()
