import argparse
from pathlib import Path

from .pipeline import run_from_config, evaluate_only
from .utils import ensure_dir

def main():
    p = argparse.ArgumentParser(prog="fairprs-clin", description="FairPRS-Clin: PRS scoring + ancestry-stratified evaluation + PRS-RS report")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline from YAML config (optionally calls plink2).")
    p_run.add_argument("--config", required=True, help="Path to YAML config.")
    p_run.add_argument("--out", default=None, help="Override output directory from config.")

    p_eval = sub.add_parser("evaluate", help="Evaluate from existing scores (PLINK2 .sscore or CSV).")
    p_eval.add_argument("--scores", required=True, help="Path to .sscore or CSV containing IID + score.")
    p_eval.add_argument("--groups", required=True, help="TSV/CSV with IID and group columns.")
    p_eval.add_argument("--out", required=True, help="Output directory.")
    p_eval.add_argument("--cutoff", default=None, help="Cutoff (float) on the SCORE field (after standardization selection).")
    p_eval.add_argument("--cutoff-percentile", type=float, default=None, help="Percentile cutoff (0-100). Mutually exclusive with --cutoff.")
    p_eval.add_argument("--standardize", choices=["global", "within_group", "none"], default="global", help="How to standardize scores for reporting.")
    p_eval.add_argument("--score-column", default=None, help="If CSV: which column holds the score (default guesses).")

    args = p.parse_args()

    if args.cmd == "run":
        run_from_config(Path(args.config), out_override=Path(args.out) if args.out else None)
    elif args.cmd == "evaluate":
        ensure_dir(Path(args.out))
        evaluate_only(
            scores_path=Path(args.scores),
            groups_path=Path(args.groups),
            out_dir=Path(args.out),
            cutoff=args.cutoff,
            cutoff_percentile=args.cutoff_percentile,
            standardize=args.standardize,
            score_column=args.score_column,
        )
