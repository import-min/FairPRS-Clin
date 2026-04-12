import argparse
from pathlib import Path

from .pipeline import run_from_config, evaluate_only
from .utils import ensure_dir


def main():
    p = argparse.ArgumentParser(
        prog="fairprs-clin",
        description="FairPRS-Clin: ancestry-stratified PRS evaluation + ClinGen PRS-RS reporting",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── run ───────────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Run full pipeline from YAML config (optionally calls plink2).",
    )
    p_run.add_argument("--config", required=True, help="Path to YAML config file.")
    p_run.add_argument("--out", default=None, help="Override output directory from config.")

    # ── evaluate ──────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate from existing scores (PLINK2 .sscore or CSV).",
    )
    p_eval.add_argument(
        "--scores", required=True,
        help="Path to .sscore or CSV containing IID + score column.",
    )
    p_eval.add_argument(
        "--groups", required=True,
        help="TSV/CSV with IID and ancestry group columns.",
    )
    p_eval.add_argument("--out", required=True, help="Output directory.")
    p_eval.add_argument(
        "--cutoff", default=None,
        help="Absolute score cutoff on standardized scores (mutually exclusive with --cutoff-percentile).",
    )
    p_eval.add_argument(
        "--cutoff-percentile", type=float, default=None,
        help="Percentile cutoff 0–100 (e.g. 95 = top 5%%). Mutually exclusive with --cutoff.",
    )
    p_eval.add_argument(
        "--standardize",
        choices=["global", "within_group", "none"],
        default="global",
        help="Score standardization method (default: global z-score).",
    )
    p_eval.add_argument(
        "--score-column", default=None,
        help="Column name for the score in a CSV file (auto-detected if not provided).",
    )
    p_eval.add_argument(
        "--n-boot", type=int, default=1000,
        help="Number of bootstrap resamples for confidence intervals (default: 1000).",
    )
    p_eval.add_argument(
        "--equalize-target", type=float, default=0.10,
        help="Target flagging rate for equalized per-group cutoffs (default: 0.10 = top 10%%).",
    )
    p_eval.add_argument(
        "--outcomes", default=None,
        help=(
            "Optional: TSV/CSV with IID + binary outcome column (0/1). "
            "When provided, computes per-group AUC, calibration slope, and Brier score."
        ),
    )
    p_eval.add_argument(
        "--outcome-column", default=None,
        help="Column name for the outcome in the outcomes file (auto-detected if not provided).",
    )

    args = p.parse_args()

    if args.cmd == "run":
        run_from_config(
            Path(args.config),
            out_override=Path(args.out) if args.out else None,
        )

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
            n_boot=args.n_boot,
            equalize_target=args.equalize_target,
            outcomes_path=Path(args.outcomes) if args.outcomes else None,
            outcome_column=args.outcome_column,
        )
