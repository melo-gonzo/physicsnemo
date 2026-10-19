# DRIVE-FIX preregistration: three baseline-input fairness reruns (2026-10-18)

Written 2026-10-18 16:40–17:00 EDT, before any DRIVE-FIX lane was submitted. Only verification jobs had run: LHR
271171 (sha256 of the DrivAerML copy, CPU) and LHR 271294 (the 20-step smokes). Source: the drive-input audit
`scratch/isla-drive-audit-2026-10-18/REPORT.md`, rows R1, R2 and R3 of its "Recommended reruns" table. Supervisor
decisions: R1, R2 and R3 run; R1b (ISLA's own raw speed input) and R4 (AeroZoo inlet channels) do not.

Rule (supervisor): the baselines get the same information as ISLA, sensibly scaled, with no augmentation advantage. The
released GeoTransolver and Transolver code stays untouched; the input is chosen in recipe config only.

Cluster: LHR GB300 for all three arms. `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`.
The launcher directories are `$T/kernel/{afu_ud,r80_ud,lug_bc}`, with local copies in `cluster/`. Jobs, md5s and the
scoring commands are in `JOBS.md`.

## Question

Once GeoTransolver and Transolver read the drive or boundary-condition information that ISLA reads, at O(1) scale,
where do they stand against ISLA on

- **R1**, AirfRANS (AF-UNIFORM)?
- **R2**, DrivAerML Surface (a control: there the drive is constant)?
- **R3**, the GINOT lug?

## Arms

Every arm copies its reference rows exactly: model, width, depth, lr, seed, cells, epochs, code snapshot and the extras.
The launcher's fixed arguments are also the reference launcher's: bfloat16, augment=false, compile=false, save_interval 2,
log every 50 steps, 4 GPUs, batch 1 per GPU. Each arm then changes only the input overrides listed below, plus the R1
dataset file. Seeds 42 and 43; 500 epochs.

### R1: AirfRANS baselines with the unit direction and the standardized speed (`$T/kernel/afu_ud`)

| row | run id | reference row (`$T/kernel/airfrans_u/lanes.tsv`, md5 030c8422) | model | lr | cells |
|---|---|---|---|---|---|
| 0, 1 | afu_ud_gt_seed42/43 | idx 2 / 3 `afu_r_gt_seed42/43` | geotransolver_volume_airfrans (width 256, 12 layers) | 1e-3 | 100,000 |
| 2, 3 | afu_ud_t_seed42/43 | idx 4 / 5 `afu_r_t_seed42/43` | transolver_volume with `model.n_hidden=512 model.n_layers=16` | 1e-3 | 100,000 |

- Dataset: `airfrans_uvolume_ud`, in the new recipe snapshot `$T/recipe_v8ud`. It is a copy of `recipe_v8`, the
  snapshot the reference lanes trained and scored on. `diff -rq` shows only the files below; the copy leaves out
  `outputs/` and the caches.
- `airfrans_uvolume_ud.yaml` is `airfrans_uvolume.yaml` plus one last transform:
  `SplitGlobalVector{field: U_inf_mag, names: [U_inf_mag_std], mean: 62.116, std: 17.914}`.
  - These statistics were checked against the data. They are the mean and the population std (ddof 0) of
    `global_data.U_inf_mag`, read from all 800 full_train case files: 62.1160 and 17.9140 m/s
    (`verify/af_stats.txt`; the ddof-1 std is 17.9252).
  - The standardized speed spans −1.72 to 1.76 on full_train and −1.69 to 1.74 on full_test.
- Twins for the scorer's area and posed protocols, `airfrans_uvolume_ud_area.yaml` and `airfrans_uvolume_ud_posed.yaml`,
  add the same step to `airfrans_uvolume_area.yaml` and `airfrans_uvolume_posed.yaml`. This is a deviation from "one
  file": the AF-UNIFORM scorer reads `<dataset>_area` and `<dataset>_posed`.
- SNAPSHOT_HEAD has a new last line recording the copy and the three files.
- Input overrides:
  - GeoTransolver: `forward_kwargs.global_embedding=[global_data.U_inf_dir,global_data.U_inf_mag_std] model.global_dim=4`
    replaces `global_data.U_inf`, which has 3 channels of 31–94 m/s.
  - Transolver: `forward_kwargs.fx.source=[global_data.U_inf_dir,global_data.U_inf_mag_std] model.functional_dim=4`.
- This is the information ISLA reads, which is `U_inf_dir` plus the speed. ISLA's own raw-speed input is unchanged
  (R1b, not run).

### R2: Transolver on a constant drive, DrivAerML Surface (`$T/kernel/r80_ud`)

| row | run id | reference row (`$T/kernel/char/r80_lanes.tsv`, md5 a9f5c6f9, the same on AGA and LHR) | extra | lr | cells | code |
|---|---|---|---|---|---|---|
| 0, 1 | r80ud_dr_t_h512_l16_c100k_seed42/43 | idx 10 / 21 `r80_dr_t_h512_l16_c100k_seed42/43` | `model.n_hidden=512 model.n_layers=16 forward_kwargs.fx.source=global_data.U_inf_dir` | 1e-3 | 100,000 | code_v18 |

- `U_inf_dir` replaces `U_inf`, which is (38.889, 0, 0) for every car. `functional_dim` stays 3.
- Cluster: the reference lanes trained on **AGA** (`$T/runs/r80_dr_t_h512_l16_c100k_seed42/train.log`, job 976568), on
  AGA's group copy `datasets/PhysicsNeMo-DrivaerML`. The arm runs on **LHR**, for three reasons:
  - LHR's group copy at the same path holds the same files. LHR 271171 checked all 40,173 files that the reader can
    touch (`run_*/domain_run_*.pdmsh/**` and `manifest.json`) against AGA's `SHA256SUMS`, which JHB's mirror was
    verified against: 0 mismatches.
  - `code_v18` and `recipe_v8` on LHR are identical to AGA's. All 1,529 `.py` files match. In `src/`, `conf/` and
    `datasets/`, the only difference is the six AF-UNIFORM files that LHR added and this arm does not read.
  - A new job starts in about 1 h on LHR, against about 10.5 h on AGA and about 19 h on JHB (`wsq capacity`, 16:25 EDT).
- JHB's copy is also identical to AGA's, but JHB lacks the code_v18 and recipe_v8 snapshots and the venv.
- The venv is LHR's `.venv-recipe`. It matches AGA's in torch 2.12.0+cu130, tensordict 0.12.4 and numpy 2.4.6; it differs
  in grpcio, markdown, werkzeug and the installed nvidia_physicsnemo dist-info, which PYTHONPATH shadows with code_v18.
- Launcher: a copy of `r80_lanes.sbatch` (md5 29e98972). It changes the job name and, by necessity, the `SELF` and
  `LANES` paths, which that launcher hard-codes. As in the reference, its STATUS and RUN markers and its chain-ahead
  logs go to `$T` and `$T/sbatch_logs`.

### R3: GeoTransolver on the lug with explicit boundary-condition labels (`$T/kernel/lug_bc`)

| row | run id | reference row (`$T/kernel/char/cl_lanes.tsv` idx 68 / 69, the same on AGA and LHR) | lr | cells | code |
|---|---|---|---|---|---|
| 0, 1 | lugbc_gt_seed42/43 | `cl_r_lug_gt_seed42/43` (geotransolver_lug, `model.n_hidden=256`) | 1e-3 | 20,000 | code_v20 |

- Extra: the reference's `train_split=train val_split=test +training.val_interval=10 model.n_hidden=256`, plus
  `++dataset_reader_overrides.drop_in_file_boundaries=false 'forward_kwargs.geometry=[boundaries.surface.cell_centroids,boundaries.surface.cell_data.normals,boundaries.surface.cell_data.bc_fixed,boundaries.surface.cell_data.bc_hole,boundaries.surface.cell_data.bc_loaded,boundaries.surface.cell_data.bc_pressure]' model.geometry_dim=10`.
- GeoTransolver's geometry context becomes the lug's surface triangles, each carrying its normal and the four labels
  ISLA reads: clamped, hole, loaded half, and pressure/75.3. It used to be the interior nodes. The local embedding
  (interior nodes, sdf, sdf normals) and the global `params_std` are unchanged.
- Cluster: LHR. The reference lanes trained on AGA (job 1017317). The lug data has the same manifest md5 on both
  clusters (d1e378f6), and ISLA's A2 comparator (`a2r_lug_isla_seed42/43`) trained and scored on LHR.
- Launcher: a copy of `cl_lanes.sbatch` (md5 0ad8638f). It changes the job name and the `SELF` and `LANES` paths.

### Released model source

Both snapshots carry the same GeoTransolver and Transolver files, md5-identical in code_v18 and code_v20 and on AGA and
LHR (listing in JOBS.md). R1 and R3 use code_v20, the reference lanes' own snapshot, and R2 uses code_v18. No model
file is touched; every change is a Hydra override or a new dataset yaml.

## Metric (each dataset's existing protocol, unchanged)

- **R1, AF-UNIFORM.** Per-case relative L2 of pressure, velocity and eddy viscosity, averaged over the 200 full_test
  airfoils. Final checkpoint, float32, at 100,000 cells, under the three protocols native, area (equal to native by
  construction on AirfRANS) and posed.
  - Seed means over 42 and 43, reduced as `afu_protocol_artifacts_2026-10-14.py` reduces (infer_summary, else the
    mean of the per-sample rows).
  - Scorer: `$T/kernel/afu_ud/score.sbatch`, a copy of `kernel/airfrans_u/score.sbatch` (md5 f629ffc5). It maps
    `afu_ud_*` to code_v20 plus recipe_v8ud and writes `$T/evals_afu_ud/`; the inference command is the reference's.
    The shared scorer is not edited.
  - Primary readout: the native velocity ratio baseline ÷ ISLA, against ISLA's AF-UNIFORM arm (0.01198, from the
    `afu_r_isla` lanes; old architecture). The comparison is the same as the one behind the current 3.05× and 5.59×.
- **R2, CHAR DrivAerML Surface.** Relative L2 of pressure and wall shear over the validation cars. Final checkpoint,
  float32, 100,000 cells (the CHAR fleet rule for `_c100k_` lanes), native / area / posed.
  - Scored on code_v20 + recipe_v8, the CHAR scorer's mapping for `r80_*` (AGA `kernel/char/protocol_eval.sbatch`).
  - Scorer: `$T/kernel/r80_ud/score.sbatch`, a copy of `kernel/a2r/score.sbatch` (md5 8e238045, the A2-RERUN copy of
    the CHAR-PROTOCOLS scorer). Rows are in `$T/kernel/r80_ud/score_rows.tsv` and output goes to `$T/evals_r80ud/`.
- **R3, CHAR lug.** Relative L2 of von Mises stress over the 600 test cases. Final checkpoint, float32, 200,000 cells
  (the whole part), native / area / posed.
  - Scored on code_v20 + recipe_v8 (the CHAR rule for `cl_*`).
  - Scorer: `$T/kernel/lug_bc/score.sbatch`, a copy of `kernel/a2r/score.sbatch`, the scorer of the a2r lug lanes.
    Rows are in a new table, `$T/kernel/lug_bc/score_rows.tsv`, and output goes to `$T/evals_lugbc/`.
  - Primary readout: GeoTransolver native ÷ ISLA A2 native (0.001726, `results/char/lug_a2.json`).

## Record (seed means; per seed 42 / 43)

| arm | dataset | native | posed |
|---|---|---|---|
| ISLA 512 × 512 slices, AF-UNIFORM | AirfRANS v / p / nut | 0.01198 / 0.02076 / 0.02424 | same |
| GeoTransolver 256, U_inf | AirfRANS v / p / nut | **0.03650** (0.03801 / 0.03499) / 0.01229 / 0.01943 | 1.094 / 2.508 / 4.381 |
| Transolver 512 × 16, U_inf | AirfRANS v / p / nut | **0.06694** (0.06517 / 0.06871) / 0.01353 / 0.02120 | 1.124 / 1.697 / 1.817 |
| Transolver 512 × 16 at 100k, U_inf | DrivAerML p / wss | **0.04368** (0.04384 / 0.04351) / **0.06961** (0.06994 / 0.06929) | 0.984 / 1.164 |
| ISLA A2 512 × 512 | lug von Mises | 0.001726 (0.001602 / 0.001850) | 0.00597 |
| GeoTransolver 256, no BC labels | lug von Mises | **0.002092** (0.002291 / 0.001893) | 0.939 |

Ratios baseline ÷ ISLA today: AirfRANS velocity 3.05 (GeoTransolver) and 5.59 (Transolver); lug 1.21 (GeoTransolver
÷ A2 ISLA).

## Predictions (written before any lane ran)

- **P1 (R1, the main one): the velocity error of both AirfRANS baselines falls substantially.**
  - Reasoning. Velocity scaled by |U| is dominated by the far field (cos α, sin α). A relative L2 of 0.037 is about 2° of
    direction error. Today the angle reaches the models only through Uy (−7 to 23) on a vector whose norm varies
    threefold, and the drive enters a projector that is already near-degenerate at these magnitudes. With a unit
    direction the angle is an O(1) input.
  - GeoTransolver: velocity ÷ ISLA falls from 3.05 to **1.0–2.5** (point 1.6), i.e. velocity 0.012–0.030.
  - Transolver: from 5.59 to **1.2–4.0** (point 2.2), i.e. velocity 0.014–0.048. Transolver gains more, because its raw
    drive enters every point's features.
  - Falsified if either baseline's velocity falls by less than 10 % (GeoTransolver above 0.0329, Transolver above 0.0602).
  - Confidence: moderate that both fall by at least 10 %, low on the size.
- **P1b (R1): pressure and eddy viscosity change little.** Native pressure and eddy viscosity for each baseline stay
  within 0.75–1.10× their current values. The baselines already lead ISLA there, at 0.59–0.87×, and should still lead.
- **P1c (R1): posed stays broken.** The baselines' posed errors stay at or above 0.5 on every field. The unit direction
  rotates with the pose, but the coordinates are not invariant.
- **P2 (R2): within 3 %.** Both seed-mean native errors stay within ±3 % of the reference: pressure 0.04237–0.04499 and
  wall shear 0.06752–0.07170. Expected pressure about 0.0435 (range 0.042–0.045), with any change more likely downward.
  Confidence about 65 %. A constant input only shifts the first-layer bias, and REF-80 Transolver already trains well
  on DrivAerML. A HiLift-like gain (0.091 → 0.032 at a constant 2,680) is not expected at a constant 38.9.
- **P3 (R3): a small gain.** GeoTransolver lug native von Mises **0.0017–0.0023** (point 0.0020); ÷ ISLA A2 0.001726 =
  **0.95–1.35** (point 1.15). Reasoning: the clamp and the loaded half sit at fixed places in the lug's canonical frame
  with augmentation off, so the labels are mostly inferable from coordinates already. The surface context gives
  GeoTransolver's projector the part's outline as well. Posed stays at or above 0.5.

## Decision rules

- **R2 (pre-registered, supervisor).** Let r_p and r_tau be the seed-mean native pressure and wall-shear errors divided
  by the reference's (0.04368 and 0.06961). Area and posed are reported but do not enter the rule.
  - If both r_p and r_tau lie in [0.97, 1.03], no further reruns follow, and the book text that claims the baselines
    read the unit direction (F3 of the audit) is corrected instead.
  - Otherwise the supervisor decides whether to run Transolver on DrivAerML Volume and SHIFT-SUV, and GeoTransolver on
    all three (about 175 + 365 GPU-h).
- **R1 and R3 have no adoption rule.** They correct the standing. Once scored, the DRIVE-FIX rows replace the
  physical-drive (R1) and no-label (R3) rows as the baselines' standing in the CHAR record, and the old rows are kept
  as the record of what was measured.

## Smokes (LHR 271294, before submission; logs `$T/kernel/drive_fix_verify/smoke_*.log`)

One smoke per arm model. Each ran on one GPU for 20 training steps, through `smoke_wrap.py`, which wraps the released
classes' `forward` at runtime to print the inputs. The train.py command was the arm launcher's, built from the new
lanes.tsv row. All four exited with rc 0. Every step's loss was finite and every output finite; there were no tracebacks.

| arm | wrapped class (file) | drive / geometry input seen by the model | range over the 20 steps | loss steps 1 → 20 |
|---|---|---|---|---|
| afu_ud GT | GeoTransolver (code_v20) | `global_embedding` (1, 1, 4) | dir x 0.969–1.000, y −0.073–0.246, z 0; speed_std −1.41–1.55 | 0.64 → 0.08 |
| afu_ud T | Transolver (code_v20) | `fx` (1, 100000, 4), one value per case broadcast | the same per-case values | 0.40 → 0.07 |
| r80ud T | Transolver (code_v18) | `fx` (1, 100000, 3) | (1, 0, 0) on every car | 0.14 → 0.07 |
| lugbc GT | GeoTransolver (code_v20) | `geometry` (1, 5.6k–6.7k surface cells, 10) | centroids, normals ±1, bc_fixed / bc_hole / bc_loaded 0–1, bc_pressure 0–1.30 | 0.31 → 0.20 |

## Acceptance of each lane at its first link

At least 100 training steps, finite loss, no fatal traceback, and the override visible in the logged config. Checked in
`train.log`'s Config block:

- R1: `global_embedding` or `source` lists `global_data.U_inf_mag_std`, with `global_dim: 4` / `functional_dim: 4`.
- R2: `source: global_data.U_inf_dir`.
- R3: `geometry` lists the six surface fields, `geometry_dim: 10` and `drop_in_file_boundaries: false`.
