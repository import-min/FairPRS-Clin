from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .evaluate import evaluate_scores
from .report import write_report
from .scoring import normalize_weights_file, run_plink2_score
from .utils import ensure_dir, env_metadata, write_json

def run_from_config(config_path: Path, out_override: Optional[Path] = None) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = Path(out_override) if out_override else Path(cfg.get("out", "out/run"))
    out_dir = ensure_dir(out_dir)
    logs_dir = ensure_dir(out_dir / "logs")

    meta: Dict[str, Any] = {"config": str(config_path), "env": env_metadata(), "inputs": {}}

    # groups and weights
    groups_path = Path(cfg["groups"]["path"])
    meta["inputs"]["groups"] = str(groups_path)

    scores_mode = cfg.get("scores", {}).get("mode", "plink2")  # plink2 or existing
    standardize = cfg.get("evaluate", {}).get("standardize", "global")

    cutoff = cfg.get("evaluate", {}).get("cutoff", None)
    cutoff_percentile = cfg.get("evaluate", {}).get("cutoff_percentile", None)

    if scores_mode == "existing":
        scores_path = Path(cfg["scores"]["path"])
        meta["inputs"]["scores"] = str(scores_path)
        eval_meta = evaluate_scores(
            scores_path=scores_path,
            groups_path=groups_path,
            out_dir=out_dir,
            cutoff=cutoff,
            cutoff_percentile=cutoff_percentile,
            standardize=standardize,
            score_column=cfg.get("scores", {}).get("score_column", None),
        )
        meta.update(eval_meta)
        write_report(out_dir, meta, title=cfg.get("report", {}).get("title", "FairPRS-Clin Report"))
        write_json(logs_dir / "run_metadata.json", meta)
        return

    if scores_mode != "plink2":
        raise ValueError("scores.mode must be 'plink2' or 'existing'")

    # plink2 scoring path
    plink2_path = cfg["plink2"]["path"]
    dataset_mode = cfg["genotypes"]["mode"]  # pfile or bfile
    dataset_prefix = Path(cfg["genotypes"]["prefix"])
    meta["inputs"]["genotypes"] = {"mode": dataset_mode, "prefix": str(dataset_prefix)}
    meta["inputs"]["plink2"] = plink2_path

    weights_path = Path(cfg["weights"]["path"])
    meta["inputs"]["weights"] = str(weights_path)

    norm_weights = out_dir / "intermediate" / "weights_plink2.tsv"
    weights_meta = normalize_weights_file(weights_path, norm_weights)
    meta["weights"] = weights_meta

    out_prefix = out_dir / "intermediate" / "plink2_score"
    sscore_path = run_plink2_score(
        plink2_path=plink2_path,
        dataset_mode=dataset_mode,
        dataset_prefix=dataset_prefix,
        normalized_weights=norm_weights,
        out_prefix=out_prefix,
        extra_args=cfg.get("plink2", {}).get("extra_args", []),
    )
    meta["inputs"]["scores"] = str(sscore_path)

    eval_meta = evaluate_scores(
        scores_path=sscore_path,
        groups_path=groups_path,
        out_dir=out_dir,
        cutoff=cutoff,
        cutoff_percentile=cutoff_percentile,
        standardize=standardize,
        score_column=None,
    )
    meta.update(eval_meta)

    write_report(out_dir, meta, title=cfg.get("report", {}).get("title", "FairPRS-Clin Report"))
    write_json(logs_dir / "run_metadata.json", meta)

def evaluate_only(
    scores_path: Path,
    groups_path: Path,
    out_dir: Path,
    cutoff: Optional[str],
    cutoff_percentile: Optional[float],
    standardize: str,
    score_column: Optional[str] = None,
) -> None:
    meta = evaluate_scores(
        scores_path=scores_path,
        groups_path=groups_path,
        out_dir=out_dir,
        cutoff=cutoff,
        cutoff_percentile=cutoff_percentile,
        standardize=standardize,
        score_column=score_column,
    )
    write_report(out_dir, meta, title="FairPRS-Clin Report")
