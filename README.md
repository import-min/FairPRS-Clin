# FairPRS-Clin

FairPRS-Clin is a lightweight, ClinGen PRS-RS–aligned **evaluation and reporting** pipeline for applying **published polygenic risk scores (PRS)** to **public genotype data** and summarizing **ancestry-stratified score behavior** (distributions, summary stats, and cutoff-based flagging).

This repo is designed to be:
- a reusable **research utility** (swap in any PRS + grouping file)
- a **teaching resource** (clear artifacts + PRS-RS mapping)

> ⚠️ FairPRS-Clin **does not** claim clinical validity. It summarizes *how a PRS behaves* on a target genotype dataset. It does not infer disease risk without phenotype-linked validation and calibration.

---

## What you provide (minimum)

You can run FairPRS-Clin in **two modes**:

### A) Recommended: compute PRS with `plink2` (fast, standard)
You provide:
1) A PLINK2 dataset prefix (`--pfile` or `--bfile`)  
2) A PRS weight file (PGS Catalog scoring file or a simple TSV)  
3) An ancestry/group labels file for samples  
4) `plink2` installed (or a path to it)

FairPRS-Clin will:
- harmonize the weight file
- call `plink2 --score ...` to compute per-sample PRS
- generate plots + tables + PRS-RS–aligned report

### B) If you already computed PRS elsewhere
You provide:
- an `.sscore` file (PLINK2 output) **or** a CSV with `IID` + `SCORE`  
- ancestry/group labels file  
FairPRS-Clin skips scoring and goes straight to evaluation/reporting.

---

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Prepare inputs

**Ancestry/group labels** (TSV):
```text
IID    group
HG00096    EUR
HG00097    EUR
...
```

**PRS weights** (supported formats):
- PGS Catalog scoring file (common columns like `chr_name`, `chr_position`, `effect_allele`, `effect_weight`)
- Simple TSV with columns:
  - `variant_id` (rsID or chr:pos:ref:alt)
  - `effect_allele`
  - `weight`

### 3) Run (plink2 scoring)
```bash
fairprs-clin run \
  --config examples/config_plink2.yaml
```

### 4) Run (from existing .sscore)
```bash
fairprs-clin evaluate \
  --scores path/to/output.sscore \
  --groups path/to/groups.tsv \
  --out out/demo
```

---

## Outputs

In `--out` you get:
- `tables/summary_by_group.tsv` : ancestry-stratified stats + cutoff flagging
- `figures/score_distributions.png` : per-group distributions
- `reports/report.md` : PRS-RS–aligned report (human-readable)
- `reports/prsrs_mapping.json` : PRS-RS item → computed artifact mapping
- `logs/run_metadata.json` : provenance (versions, hashes, parameters)

---

## Notes on 1000 Genomes
FairPRS-Clin includes helper scripts to convert the 1000G panel file into a `IID → super_population` grouping file.

See `examples/1000g_panel_to_groups.py`.

---

## License
MIT (see `LICENSE`).

---

## Citation
If you use this repository, please cite the accompanying manuscript or include a link to the GitHub release.
