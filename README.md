# FairPRS-Clin

Polygenic risk scores are increasingly discussed for clinical use, but most published scores were built on European ancestry cohorts and behave unevenly when applied to diverse populations. Using PGS000036, a published Type 2 diabetes score, a global top-10% cutoff flags 48.7% of European ancestry individuals and 0% of African ancestry individuals on 1000 Genomes Phase 3 data. That is a 18x gap comes from score itself being trained on European GWAS data, so the risk alleles it tracks are far more common in Europeans.


FairPRS-Clin measures this problem and fixes it. The Ancestry Portability Score gives a single number from 0 to 1 summarizing how evenly a PRS distributes across ancestry groups, computed from score distributions alone without needing outcome data. The Resource Constrained Fair Threshold finds per-group cutoffs that equalize flagging rates while keeping the total fraction of people screened fixed to avoid flagging more people overall. On the T2D example it brings the 18x gap down to 1.01x. Bayesian Group Recalibration fits ancestry-specific recalibration parameters with shrinkage toward the global model, which applies to some groups that only have a few dozen cases. All outputs map to the ClinGen PRS Reporting Standards checklist.


## Install

```bash
pip install -e .
```

## Run

```bash
fairprs-clin evaluate \
  --scores examples/pgs000036_1kg_real.sscore \
  --groups examples/1kg_hg38_groups.tsv \
  --cutoff-percentile 90 \
  --rcft-budget 0.10 \
  --out out/demo
```

Add `--outcomes path/to/outcomes.tsv` to get per-group AUC, calibration, and BGR recalibration. The groups file needs `IID` and `group` columns. The outcomes file needs `IID` and a binary `outcome` column. Everything else is auto-detected.

To reproduce the paper results:

```bash
python examples/reproduce_analysis.py \
  --scores examples/pgs000036_1kg_real.sscore \
  --groups examples/1kg_hg38_groups.tsv
```

The scores file contains real plink2-computed PRS for all 3,202 1000 Genomes Phase 3 samples scored with plink2 v2.0 against PGS000036 weights from the PGS Catalog.

## Outputs

Tables in `tables/`, figures in `figures/`, PRS-RS report in `reports/`, provenance log in `logs/`.

## Tests

```bash
pytest -q  # 42 tests
```

## License

MIT
