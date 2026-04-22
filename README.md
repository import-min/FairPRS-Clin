# FairPRS-Clin

Polygenic risk scores are increasingly discussed for clinical use, but most published scores were built on European ancestry cohorts and behave unevenly when applied to diverse populations. Applying PGS000036, the most cited Type 2 diabetes PRS, to 1000 Genomes Phase 3 data with a standard global top-5% threshold flags 10.3% of European-ancestry individuals and 1.8% of African-ancestry individuals. That is a 5.7x gap that grows as the threshold gets stricter, reaching 8.5x at the 99th percentile. It exists because the score was trained on European GWAS data and risk allele frequencies do not transfer evenly across populations.

FairPRS-Clin addresses this with three methods. The Ancestry Portability Score summarizes how evenly a PRS performs across ancestry groups as a single number from 0 to 1, with bootstrap confidence intervals. The Resource Constrained Fair Threshold finds per group screening thresholds that equalize flagging rates while keeping the total fraction of people flagged fixed. This reduces the 5.7x disparity to 1.05x. Bayesian Group Recalibration estimates ancestry specific recalibration parameters with shrinkage toward the global model. This prevents overfitting when some groups only have a few dozen cases. All outputs are linked to the ClinGen PRS Reporting Standards checklist.

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
