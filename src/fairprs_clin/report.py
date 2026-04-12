from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .utils import ensure_dir, write_json

PRSRS_ITEMS = [
    {"id": "PRS-RS:ScoreDefinition",  "label": "Score definition and variant specification"},
    {"id": "PRS-RS:AncestryData",     "label": "Ancestry composition of evaluation data"},
    {"id": "PRS-RS:Distribution",     "label": "PRS distributional behavior across groups"},
    {"id": "PRS-RS:Thresholding",     "label": "Thresholding and intended-use sensitivity"},
    {"id": "PRS-RS:Performance",      "label": "Model performance and calibration evidence"},
    {"id": "PRS-RS:IntendedUse",      "label": "Intended use and limitations"},
]


def build_prsrs_mapping(meta: Dict) -> Dict:
    artifacts = meta.get("artifacts", {})
    clin = meta.get("clinical_validity", {})
    clin_arts = clin.get("artifacts", {})
    has_clin = bool(clin.get("n_with_outcomes"))

    mapping = {
        "PRS-RS:ScoreDefinition": {
            "provided_by_tool": True,
            "artifacts": [
                meta.get("weights", {}).get("normalized_weights"),
                meta.get("weights", {}).get("weights_input"),
            ],
            "notes": (
                "Variant list, effect alleles, and weights parsed from the provided "
                "score file. Harmonization logs recorded in plink2 mode."
            ),
        },
        "PRS-RS:AncestryData": {
            "provided_by_tool": True,
            "artifacts": [meta.get("inputs", {}).get("groups")],
            "notes": (
                "Ancestry group labels supplied by user. FairPRS-Clin does not infer "
                "ancestry. Labels are not equivalent to clinical race/ethnicity."
            ),
        },
        "PRS-RS:Distribution": {
            "provided_by_tool": True,
            "artifacts": [
                artifacts.get("distribution_plot"),
                artifacts.get("summary_by_group"),
                artifacts.get("pairwise_smds"),
                artifacts.get("ks_tests"),
                artifacts.get("bootstrap_stats"),
            ],
            "notes": (
                "Per-group KDE distributions, summary statistics, pairwise Cohen's d, "
                "KS tests, and bootstrap 95% CIs computed on the target dataset."
            ),
        },
        "PRS-RS:Thresholding": {
            "provided_by_tool": True,
            "artifacts": [
                artifacts.get("summary_by_group"),
                artifacts.get("sensitivity_plot"),
                artifacts.get("disparity_plot"),
                artifacts.get("equalized_cutoffs"),
            ],
            "notes": (
                "Flagging rates computed at user-specified cutoff with disparity ratio "
                "(max/min across groups). Sensitivity curve covers 5th–99th percentile. "
                "Equalized per-group cutoffs provided as an alternative to global threshold."
            ),
        },
        "PRS-RS:Performance": {
            "provided_by_tool": has_clin,
            "artifacts": [
                clin_arts.get("discrimination_auc"),
                clin_arts.get("calibration_stats"),
                clin_arts.get("roc_plot"),
                clin_arts.get("calibration_plot"),
            ] if has_clin else [],
            "notes": (
                "Per-group AUC (C-statistic) with bootstrap 95% CIs, calibration slope "
                "and intercept, and Brier score computed from provided outcome data."
            ) if has_clin else (
                "Not computed: no outcome file provided. Supply --outcomes to compute "
                "per-group AUC, calibration slope, and Brier score."
            ),
        },
        "PRS-RS:IntendedUse": {
            "provided_by_tool": True,
            "artifacts": [],
            "notes": (
                "Report documents distributional behavior, equity metrics, and "
                "(when outcomes provided) clinical validity statistics. "
                "Clinical use requires prospective validation in the intended population."
            ),
        },
    }

    for v in mapping.values():
        v["artifacts"] = [a for a in v.get("artifacts", []) if a]
    return mapping


def _read_tsv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return None


def write_report(out_dir: Path, meta: Dict, title: str = "FairPRS-Clin Report") -> None:
    reports = ensure_dir(out_dir / "reports")
    tables = out_dir / "tables"

    summary   = _read_tsv(tables / "summary_by_group.tsv")
    smd_df    = _read_tsv(tables / "pairwise_smds.tsv")
    ks_df     = _read_tsv(tables / "ks_tests.tsv")
    boot_df   = _read_tsv(tables / "bootstrap_stats.tsv")
    eq_df     = _read_tsv(tables / "equalized_cutoffs.tsv")
    disc_df   = _read_tsv(tables / "discrimination_auc.tsv")
    cal_df    = _read_tsv(tables / "calibration_stats.tsv")

    mapping = build_prsrs_mapping(meta)
    write_json(reports / "prsrs_mapping.json", mapping)

    lines = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [f"# {title}\n"]
    lines += ["## Run summary\n"]
    lines += [f"- Samples scored: **{meta.get('n_samples_scored', '?')}**"]
    lines += [f"- Samples with group labels: **{meta.get('n_samples_with_groups', '?')}**"]
    lines += [f"- Standardization: **{meta.get('standardize', '?')}**"]
    cutoff = meta.get("cutoff", {})
    lines += [f"- Cutoff: **{cutoff.get('kind', 'none')}** (value: {cutoff.get('value')})\n"]

    # ── Ancestry-stratified summary ───────────────────────────────────────
    lines += ["\n## Ancestry-stratified summary\n"]
    if summary is not None and not summary.empty:
        cols = [c for c in ["group", "n", "mean", "sd", "median", "iqr",
                            "flagged_prop", "flagged_n"] if c in summary.columns]
        lines += [summary[cols].to_markdown(index=False), ""]
    else:
        lines += ["_Summary table not found._\n"]

    # ── Equity metrics ────────────────────────────────────────────────────
    lines += ["\n## Equity metrics\n"]

    equity = meta.get("equity", {})
    if equity:
        dr = equity.get("disparity_ratio")
        mg = equity.get("max_flagging_group")
        ng = equity.get("min_flagging_group")
        pp = equity.get("absolute_disparity_pp")
        lines += [
            f"- **Flagging disparity ratio:** {dr} "
            f"({mg} flagged most, {ng} least; {pp} percentage point gap)"
        ]

    if smd_df is not None and not smd_df.empty:
        lines += ["\n### Pairwise standardized mean differences (Cohen's d)\n"]
        lines += [smd_df.to_markdown(index=False), ""]

    if ks_df is not None and not ks_df.empty:
        lines += ["\n### Pairwise KS tests\n"]
        lines += [ks_df.to_markdown(index=False), ""]

    if boot_df is not None and not boot_df.empty:
        lines += ["\n### Bootstrap 95% CIs (per group)\n"]
        cols = [c for c in ["group", "n", "mean", "mean_ci_lo", "mean_ci_hi",
                            "flagging_rate", "flag_ci_lo", "flag_ci_hi"]
                if c in boot_df.columns]
        lines += [boot_df[cols].to_markdown(index=False), ""]

    if eq_df is not None and not eq_df.empty:
        lines += ["\n### Equalized per-group cutoffs\n"]
        lines += [
            f"Target flagging rate: **{eq_df['target_flagging_rate'].iloc[0]:.0%}**\n"
        ]
        lines += [eq_df.to_markdown(index=False), ""]

    # ── Clinical validity ─────────────────────────────────────────────────
    clin = meta.get("clinical_validity", {})
    if clin and clin.get("n_with_outcomes"):
        lines += ["\n## Clinical validity\n"]
        lines += [
            f"- Samples with outcomes: **{clin.get('n_with_outcomes')}**",
            f"- Cases: **{clin.get('n_cases')}** | "
            f"Controls: **{clin.get('n_controls')}**\n",
        ]

        if disc_df is not None and not disc_df.empty:
            lines += ["\n### Discrimination (AUC per ancestry group)\n"]
            lines += [disc_df.to_markdown(index=False), ""]

        if cal_df is not None and not cal_df.empty:
            lines += ["\n### Calibration (logistic regression per group)\n"]
            lines += [
                "Calibration slope ≈ 1.0 and intercept ≈ 0.0 indicate well-calibrated scores.\n"
            ]
            lines += [cal_df.to_markdown(index=False), ""]
    else:
        lines += [
            "\n## Clinical validity\n",
            "No outcome file provided. To compute per-group AUC, calibration slope, "
            "and Brier score, run with `--outcomes path/to/outcomes.tsv`.\n",
        ]

    # ── PRS-RS alignment ──────────────────────────────────────────────────
    lines += ["\n## ClinGen PRS-RS alignment\n"]
    for item in PRSRS_ITEMS:
        mid = item["id"]
        m = mapping.get(mid, {})
        status = "✅ computed" if m.get("provided_by_tool") else "⚠️ requires external evidence"
        lines += [f"- **{item['label']}** ({mid}): {status}"]
        if m.get("notes"):
            lines += [f"  - {m['notes']}"]
        for a in m.get("artifacts", []):
            lines += [f"  - artifact: `{a}`"]
    lines += [""]

    (reports / "report.md").write_text("\n".join(lines), encoding="utf-8")
