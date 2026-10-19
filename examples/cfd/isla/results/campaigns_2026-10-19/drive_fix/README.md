# DRIVE-FIX score records (scored 2026-10-19)

Three baseline-input fairness reruns: GeoTransolver and Transolver at their reference configurations, retrained with the
drive or boundary-condition information ISLA reads, at O(1) scale. Released model code untouched; the input is chosen by
Hydra overrides only. Preregistration: `PREREG.md` (written before any lane ran); jobs, launchers and md5s: `JOBS.md`;
the coordinator's readout: `RESULTS.md`. All three files are copied unmodified from
`scratch/isla-drive-fix-2026-10-18/` of the main checkout.

| arm | runs (seeds 42, 43) | input change | first training link (LHR) | scored into |
|---|---|---|---|---|
| R1, AirfRANS AF-UNIFORM | `afu_ud_gt_seed42/43`, `afu_ud_t_seed42/43` | `U_inf_dir` plus `U_inf_mag_std` (speed standardized, mean 62.116, std 17.914 m/s) in place of `U_inf` | 271344, 271345, 271347, 271348 | `scores/evals_afu_ud/` |
| R2, DrivAerML Surface | `r80ud_dr_t_h512_l16_c100k_seed42/43` (Transolver 512 x 16, 100,000 cells) | `U_inf_dir` in place of `U_inf` | 271349, 271350 | `scores/evals_r80ud/` |
| R3, GINOT lug | `lugbc_gt_seed42/43` (GeoTransolver 256) | geometry context = surface triangles with normals and `bc_fixed`, `bc_hole`, `bc_loaded`, `bc_pressure` | 271351, 271352 | `scores/evals_lugbc/` |

- `scores/evals_*/<run>/<protocol>/<run>/metrics.jsonl`: the scorer output for each run and each protocol (native, area,
  posed), final checkpoints (epoch 500), float32, fetched from the LHR site's `$T/evals_{afu_ud,r80ud,lugbc}/` (scorers
  `kernel/{afu_ud,r80_ud,lug_bc}/score.sbatch`, listed in `JOBS.md`).
- `reduce_drivefix.py` reduces every record to its infer_summary row (all 24 have one) and writes
  `drivefix_scores_2026-10-19.json`. Run here, it reproduces that file byte for byte.
- Readers: `results/char/airfrans_a2_2026-10-18.py` (R1 arms become the AirfRANS baselines; the raw-drive arms are kept
  under `raw_drive_*`) and `results/char/a2r_protocol_artifacts_2026-10-17.py` (R3 becomes the lug's GeoTransolver arm;
  the arm without labels is kept under `unlabelled_geotransolver`). R2 does not enter a standing artifact; the book reads its
  ratio to the reference lanes (`results/char/drivaerml_surface_a2.json`, Transolver arm) in chapter 07.
