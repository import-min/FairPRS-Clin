FairPRS-Clin applies published polygenic risk scores to genotype data and evaluates how scores behave across ancestry groups. Outputs are mapped to the ClinGen PRS Reporting Standards (PRS-RS).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```
## Quick start

If you already have scores:

```bash
fairprs-clin evaluate \
  --scores examples/toy_scores.csv \
  --groups examples/toy_groups.tsv \
  --cutoff-percentile 80 \
  --out out/demo
```

If you want to compute scores from genotype data using plink2:

```bash
fairprs-clin run --config examples/config_plink2.yaml
```

The groups file is a two-column TSV with `IID` and `group`. The weight file can be a PGS Catalog scoring file or any TSV with `variant_id`, `effect_allele`, `weight`.

## What gets computed

Per-group summary stats and flagging rates, pairwise KS tests and Cohen's d between ancestry groups, bootstrap 95% CIs, a sensitivity curve showing how flagging rates shift across all possible cutoffs, equalized per-group cutoffs, and a PRS-RS–aligned report flagging what can and can't be computed without phenotype data.

## Outputs

Everything lands in your `--out` folder: tables in `tables/`, figures in `figures/`, the PRS-RS report in `reports/`, and a provenance log in `logs/`.

## Tests

```bash
pytest -q
```
