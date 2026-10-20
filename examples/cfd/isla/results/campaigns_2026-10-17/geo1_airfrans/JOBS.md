# GEO1-AF jobs (LHR, `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`)

| job | what | QOS | outcome | log |
|---|---|---|---|---|
| 243376 | STEP C: a2s / geo1 AirfRANS checks, geo1 DrivAerML (failed: recipe_v11 dataset_paths `???`), geo1 on .venv-recipe | short | AF parts rc=0; DV rc=1 | `$T/kernel/geo1_af/sbatch_logs/isla.geo1af.check-243376.log`, `check_logs/` |
| 243405 | STEP C: geo1 DrivAerML with full STL on recipe_v11dv | short | rc=0 | `.../isla.geo1af.checkdv-243405.log`, `check_logs/geo1_drivaer_v2.log` |
| 243455_10 -> 243462_[10] | lane afu_a2s_isla_seed42 (row 10), chain | normal 03:59 | running (accepted) | `$T/runs/afu_a2s_isla_seed42/train.log` |
| 243456_11 -> 243461_[11] | lane afu_a2s_isla_seed43 (row 11) | normal | running (accepted) | `$T/runs/afu_a2s_isla_seed43/train.log` |
| 243457_12 -> 243463_[12] | lane afu_geo1_isla_seed42 (row 12) | normal | running (accepted) | `$T/runs/afu_geo1_isla_seed42/train.log` |
| 243458_13 -> 243468_[13] | lane afu_geo1_isla_seed43 (row 13) | normal | running (accepted) | `$T/runs/afu_geo1_isla_seed43/train.log` |
| 243551_0 | smoke of `kernel/geo1_ca2/score.sbatch` on epoch-3 checkpoint copies (rows_smoke.tsv) | short | 2 units scored, 1 refused as designed | `$T/kernel/geo1_ca2/sbatch_logs/`, `smoke_evals/` |

Submission record: `$T/kernel/geo1_af/SUBMITTED.txt`. Later links are submitted by each running link (`isla.afu.<slug>`,
singleton, `sbatch_logs/resubmit.log` in `kernel/airfrans_u`).
