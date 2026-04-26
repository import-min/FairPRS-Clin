# FairPRS-Clin

Polygenic risk scores are increasingly discussed for clinical use, but most published scores were built on European ancestry cohorts and behave unevenly when applied to diverse populations. Applying PGS000036, the most cited Type 2 diabetes PRS, to 1000 Genomes Phase 3 data with a global top-5% screening threshold flags 8.6% of European ancestry individuals and 1.9% of African ancestry individuals. That is a 4.5x gap that grows monotonically as the threshold gets stricter, reaching over 8x at the 99th percentile. This disparity is a structural property of applying a European trained score to a diverse population.

FairPRS-Clin addresses this with three methods. The Ancestry Portability Score quantifies how evenly a PRS performs across ancestry groups as a single number from 0 to 1, with bootstrap confidence intervals, using only score distributions without requiring outcome data. The Resource Constrained Fair Threshold finds per-group screening thresholds that equalize flagging rates while keeping the total fraction of people flagged fixed to a clinical budget. Applied to the T2D example above, it reduces the 4.5x disparity to 1.04x with no change in total population coverage. Bayesian Group Recalibration estimates ancestry-specific recalibration parameters with shrinkage toward the global model, which prevents overfitting when ancestry groups have small case counts. All outputs are linked to the ClinGen PRS Reporting Standards checklist.

## Install

```bash
pip install -e .
```

## Run

```bash
fairprs-clin evaluate \
  --scores examples/pgs000036_1kg_scores.csv \
  --groups examples/1kg_groups.tsv \
  --cutoff-percentile 95 \
  --rcft-budget 0.05 \
  --out out/demo
```

Add `--outcomes path/to/outcomes.tsv` to compute per-group AUC, calibration slope, and BGR recalibration. To score from raw genotype data use `fairprs-clin run --config examples/config_plink2.yaml`. The groups file needs columns `IID` and `group`. The outcomes file needs `IID` and a binary `outcome` column. Everything else is auto-detected.

The included `examples/pgs000036_1kg_scores.csv` contains scores for all 2,373 1000 Genomes Phase 3 samples using real sample IDs, with score distributions generated from published PGS000036 parameters. To reproduce the paper results run `python examples/reproduce_analysis.py`.

## Outputs

Tables in `tables/`, figures in `figures/`, PRS-RS report in `reports/`, provenance log in `logs/`.

## Tests

```bash
pytest -q  # 42 tests
```

## License

MIT
