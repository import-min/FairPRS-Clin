from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .utils import ensure_dir, write_json

PRSRS_ITEMS = [
    # A small, pragmatic subset meant to be helpful for a proceedings artifact.
    # You can extend this list to match the full PRS-RS checklist you want to include.
    {"id": "PRS-RS:ScoreDefinition", "label": "Score definition and variant specification"},
    {"id": "PRS-RS:AncestryData", "label": "Ancestry composition of evaluation data"},
    {"id": "PRS-RS:Distribution", "label": "PRS distributional behavior across groups"},
    {"id": "PRS-RS:Thresholding", "label": "Thresholding and intended-use sensitivity"},
    {"id": "PRS-RS:Performance", "label": "Model performance and calibration evidence"},
    {"id": "PRS-RS:IntendedUse", "label": "Intended use and limitations"},
]

def build_prsrs_mapping(meta: Dict) -> Dict:
    artifacts = meta.get("artifacts", {})
    mapping = {
        "PRS-RS:ScoreDefinition": {
            "provided_by_tool": True,
            "artifacts": [meta.get("weights", {}).get("normalized_weights"), meta.get("weights", {}).get("weights_input")],
            "notes": "Variant list + effect allele + weight are summarized from the provided score file; harmonization logs recorded if plink2 mode used."
        },
        "PRS-RS:AncestryData": {
            "provided_by_tool": True,
            "artifacts": [meta.get("inputs", {}).get("groups")],
            "notes": "Group labels file is treated as provenance; FairPRS-Clin does not infer ancestry."
        },
        "PRS-RS:Distribution": {
            "provided_by_tool": True,
            "artifacts": [artifacts.get("distribution_plot"), artifacts.get("summary_by_group")],
            "notes": "Distributions and summary statistics are computed on the target dataset."
        },
        "PRS-RS:Thresholding": {
            "provided_by_tool": True,
            "artifacts": [artifacts.get("summary_by_group")],
            "notes": "Flagging rates are computed for a user-specified cutoff; report includes cutoff definition."
        },
        "PRS-RS:Performance": {
            "provided_by_tool": False,
            "artifacts": [],
            "notes": "Not inferable without phenotype-linked validation/calibration data; must be sourced from PRS publication or external cohort evaluation."
        },
        "PRS-RS:IntendedUse": {
            "provided_by_tool": True,
            "artifacts": [],
            "notes": "Report includes an explicit non-clinical disclaimer and prompts for intended-use language."
        },
    }
    # remove Nones
    for k, v in mapping.items():
        v["artifacts"] = [a for a in v.get("artifacts", []) if a]
    return mapping

def write_report(out_dir: Path, meta: Dict, title: str = "FairPRS-Clin Report") -> None:
    reports = ensure_dir(out_dir / "reports")
    tables = out_dir / "tables"
    summary_path = tables / "summary_by_group.tsv"

    summary = None
    if summary_path.exists():
        summary = pd.read_csv(summary_path, sep="\t")

    mapping = build_prsrs_mapping(meta)
    write_json(reports / "prsrs_mapping.json", mapping)

    lines = []
    lines.append(f"# {title}\n")
    lines.append("## Run summary\n")
    lines.append(f"- Samples with group labels: **{meta.get('n_samples_with_groups','?')}**\n")
    lines.append(f"- Standardization: **{meta.get('standardize','?')}**\n")
    cutoff = meta.get("cutoff", {})
    lines.append(f"- Cutoff: **{cutoff.get('kind','?')}** (value: {cutoff.get('value')})\n")

    lines.append("\n## Key artifacts\n")
    arts = meta.get("artifacts", {})
    for k, v in arts.items():
        lines.append(f"- `{k}`: `{v}`\n")

    lines.append("\n## Ancestry-stratified summary\n")
    if summary is None or summary.empty:
        lines.append("_Summary table not found. Did the evaluation step complete?_\n")
    else:
        # render a small markdown table (limited columns)
        cols = [c for c in ["group","n","mean","sd","median","iqr","flagged_prop","flagged_n"] if c in summary.columns]
        show = summary[cols].copy()
        lines.append(show.to_markdown(index=False))
        lines.append("\n")

    lines.append("\n## ClinGen PRS-RS alignment (high-level)\n")
    for item in PRSRS_ITEMS:
        mid = item["id"]
        m = mapping.get(mid, {})
        status = "✅ computed" if m.get("provided_by_tool") else "⚠️ requires external evidence"
        lines.append(f"- **{item['label']}** ({mid}): {status}\n")
        note = m.get("notes")
        if note:
            lines.append(f"  - {note}\n")
        if m.get("artifacts"):
            for a in m["artifacts"]:
                lines.append(f"  - artifact: `{a}`\n")

    lines.append("\n## Interpretation guardrails\n")
    lines.append("- This report summarizes PRS behavior on the chosen dataset and grouping labels.\n")
    lines.append("- It does **not** establish clinical utility, calibration, or absolute risk without phenotype-linked evaluation.\n")
    lines.append("- Ancestry labels in reference panels are not equivalent to clinical race/ethnicity.\n")

    (reports / "report.md").write_text("\n".join(lines), encoding="utf-8")
