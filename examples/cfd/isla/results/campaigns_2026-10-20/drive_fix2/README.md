# DRIVE-FIX2 score records: the unit-direction baseline reruns on DrivAerML and SHIFT-SUV (scored 2026-10-19/20)

GeoTransolver and Transolver at their reference configurations, retrained reading `U_inf_dir`, the unit freestream
direction ISLA reads, in place of `U_inf`, the constant physical vector in m/s. Released model code untouched; the input is
chosen by Hydra overrides only (`cluster/score_rows.tsv`, column `extra`). Float32, final checkpoints (epoch 500), CHAR
protocols, seeds 42 and 43.

| run (seeds 42, 43) | task | replaces |
|---|---|---|
| `r80ud_dr_gt_h256` | DrivAerML Surface, GeoTransolver 256 | `r80_dr_gt_h256` |
| `clud_vol_t` | DrivAerML Volume, Transolver 512 x 16 (100,000 training cells) | `cl_r_vol_t` |
| `clud_suv_gt` | SHIFT-SUV, GeoTransolver 256 | `cl_r_suv_gt` |
| `clud_suv_t` | SHIFT-SUV, Transolver 512 x 16 (100,000 training cells) | `cl_r_suv_t` |
| `clud_vol_gt` | DrivAerML Volume, GeoTransolver-volume 256 | `cl_r_vol_gt` |

- Copied unmodified from `scratch/isla-drive-fix2-2026-10-19/` of the main checkout: `RESULTS.md` (the readout), `JOBS.md`
  (lanes, jobs, md5s), `scores/evals_ud2/<run>/<protocol>/<run>/metrics.jsonl` (the scorer output, fetched from the LHR site's
  `$T/evals_ud2/`), and `cluster/` (`partA/cluster/cl_ud/`: `lanes.sbatch`, `lanes.tsv`, `score.sbatch`, `score_rows.tsv`).
- `reduce_ud2.py` (a copy of `campaigns_2026-10-19/drive_fix/reduce_drivefix.py` with its output name) reduces every record to its
  `infer_summary` row and writes `ud2_scores_2026-10-20.json`.
- The Transolver reruns are scored at 200,000 cells on ISLA's scored set; their predecessors in the CHAR record were scored on
  their own 100,000-cell draw, so their area and posed figures compare with the new rows' only through the native mesh. On the
  DrivAerML Volume task this scorer's area rows equal its native rows exactly; the book prints no Volume area figure.
- The DrivAerML Surface Transolver unit-direction arm is DRIVE-FIX's control `r80ud_dr_t_h512_l16_c100k`, re-scored on ISLA's
  200,000-cell set in `campaigns_2026-10-19/drive_fix2/scoreset/tset200k_scores.json`.

Reader: `results/char/a2r_protocol_artifacts_2026-10-17.py` (FAIR_INPUT): each rerun becomes its standing's baseline arm in
`results/char/<dataset>_a2.json`, the replaced arm kept under `raw_drive_<arch>`.
