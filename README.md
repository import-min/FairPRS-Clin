# FairPRS-Clin

Polygenic risk scores are increasingly discussed for clinical use, but most published scores were built on European ancestry cohorts and behave unevenly when applied to diverse populations. FairPRS-Clin is a command-line tool that takes a published PRS and evaluates how it performs across ancestry groups, including whether the same screening threshold flags people at very different rates depending on ancestry, and what a fairer threshold would look like.

Beyond standard distribution plots and summary statistics, FairPRS-Clin introduces three novel methods: an Ancestry Portability Score (APS) that summarizes overall PRS portability as a single number with confidence intervals, Bayesian Group Recalibration (BGR) that estimates ancestry-specific recalibration parameters without overfitting when group sample sizes are small, and a Resource-Constrained Fair Threshold (RCFT) that finds per-group screening thresholds minimizing disparity under a fixed total screening budget. When outcome data is available, it also computes per-group AUC, calibration, and Brier score. All outputs are linked to the ClinGen PRS Reporting Standards checklist.

## Install

```bash
pip install -e .
```

## Run

```bash
fairprs-clin evaluate \
  --scores examples/toy_scores.csv \
  --groups examples/toy_groups.tsv \
  --outcomes examples/toy_outcomes.tsv \
  --cutoff-percentile 80 \
  --out out/demo
```

If you want to compute scores from raw genotype data instead, use `fairprs-clin run --config examples/config_plink2.yaml`. The groups file needs two columns: `IID` and `group`. The outcomes file needs `IID` and a binary `outcome` column (0/1). Everything else is auto-detected.

Results go in `out/demo/` — tables in `tables/`, figures in `figures/`, a PRS-RS report in `reports/`, and a provenance log in `logs/`.

```bash
pytest -q  # 39 tests
```
