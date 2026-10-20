# DRIVE-FIX2 jobs (LHR), 2026-10-19

`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`. Local copies of
every new cluster file: `partA/cluster/` (md5s in `partA/md5s.txt`). Submission log: `launch.log`. Part B (T-SCORESET) and
Part C (AF-PLANAR) have their own reports in `partB/REPORT.md` and `partC/REPORT.md`.

## Part A: the ten unit-drive reruns

Decision (supervisor, 2026-10-19, after R2 moved Transolver by about 3 %): rerun every remaining constant-drive baseline lane
with the unit freestream direction. The speed is constant on these datasets, so no magnitude channel is added.

Each lane is its reference row, byte for byte, plus one drive override (`forward_kwargs.*`, because the model YAMLs are
`@package _global_`). Same code snapshot as the reference row, recipe_v8, the reference launcher (copied), 500 epochs,
bfloat16, augment=false, compile=false, 4 GPUs, normal QoS, 03:59:00 singleton chains.

| run (`$T/runs/<run>`) | reference row | dataset / model | extra (reference extra + override) | train cells | code |
|---|---|---|---|---|---|
| r80ud_dr_gt_h256_seed42/43 | `char/r80_lanes.tsv` idx 4 / 15, `r80_dr_gt_h256_seed42/43` | drivaer_ml_surface / geotransolver_surface | `forward_kwargs.global_embedding=global_data.U_inf_dir` | 200,000 | code_v18 |
| clud_vol_gt_seed42/43 | `char/cl_lanes.tsv` idx 44 / 45, `cl_r_vol_gt_*` | drivaer_ml_volume_reference / geotransolver_volume | `forward_kwargs.global_embedding=global_data.U_inf_dir` | 200,000 | code_v20 |
| clud_vol_t_seed42/43 | idx 46 / 47, `cl_r_vol_t_*` | drivaer_ml_volume_reference / transolver_volume | `model.n_hidden=512 model.n_layers=16 forward_kwargs.fx.source=global_data.U_inf_dir` | 100,000 | code_v20 |
| clud_suv_gt_seed42/43 | idx 56 / 57, `cl_r_suv_gt_*` | shift_suv_surface / geotransolver_surface | `forward_kwargs.global_embedding=global_data.U_inf_dir` | 200,000 | code_v20 |
| clud_suv_t_seed42/43 | idx 58 / 59, `cl_r_suv_t_*` | shift_suv_surface / transolver_surface | `model.n_hidden=512 model.n_layers=16 forward_kwargs.fx.source=global_data.U_inf_dir` | 100,000 | code_v20 |

`global_dim` / `functional_dim` stay 3: `U_inf_dir` has the same three channels as `U_inf`.

### Launchers and md5s

- `kernel/r80_ud2/lanes.sbatch` f4b37dcc = `kernel/char/r80_lanes.sbatch` 29e98972 with `-J isla.r80ud2` and the SELF / LANES
  paths (diff: 3 lines). `lanes.tsv` 6017f4f4 (rows 0, 1).
- `kernel/cl_ud/lanes.sbatch` 8363520e = `kernel/char/cl_lanes.sbatch` 0ad8638f with `-J isla.clud` and the SELF / LANES
  paths (diff: 3 lines). `lanes.tsv` f50a22fc (rows 0-7). Reference tables: `char/cl_lanes.tsv` af60022c, `char/r80_lanes.tsv`
  a9f5c6f9 (unchanged).
- The lanes.tsv rows were generated with awk from the reference rows: only idx, group, run_id and the appended override differ.

### Verification before submission

- **Smokes** (LHR 283959, 283960, short QoS; `kernel/drive_fix2_verify/smoke.sbatch` 4f8cc059, a copy of DRIVE-FIX's with
  two units per GPU; `smoke_wrap.py` d2c9e387 = DRIVE-FIX's byte copy). 20 training steps of every one of the ten rows, plus
  the five reference rows (seed 42) beside them. All 15 rc 0, 0 non-finite outputs, no traceback.
  - Drive seen by the model: (1, 0, 0) on every step of every new lane (`global_embedding` (1, 1, 3) for GeoTransolver, `fx`
    (1, N, 3) for Transolver). The reference rows see (38.889, 0, 0) on DrivAerML and (30, 0, 0) on SHIFT-SUV.
  - Nothing else changes: for each new/reference pair at the same seed, every other input line the wrapper prints (shapes,
    per-channel min and max on calls 1-2 and over all 20 steps: geometry, local_embedding, local_positions, embedding) is
    identical. The resolved configs differ only in run_id and the drive key (`partA/smoke_compare.txt`).
  - The smoke's reference configs equal the trained reference lanes' own `resolved_config.yaml` on AGA, apart from
    log_every_n_steps and paths (`partA/refcfg_diff.txt`).
- **Data** (the reference lanes trained on AGA, these on LHR):
  - DrivAerML `run_*/domain_*.pdmsh`: checked in DRIVE-FIX (LHR 271171, 40,173 files, 0 mismatches against AGA's SHA256SUMS).
  - DrivAerML `run_*/*_single_solid.stl.pmsh` (the volume reader's extra boundary): LHR 283961, 5,324 files, 0 mismatches.
  - SHIFT-SUV (every file the reader touches: `domain.pdmsh/_tensordict/{boundaries/vehicle,global_data}/**` and the
    manifests; 37,832 files): sha256 lists on LHR (283961) and AGA (1330728) are byte-identical (md5 60173b0c).

### Training lanes (normal QoS, 03:59:00 singleton chains)

Wave 1 submitted 09:20:49 PDT (12:20 EDT); wave 2 at 09:34:53 PDT (12:34 EDT). Commands (from `launch.log`, as R2's):

```
cd $T && sbatch --parsable -J isla.r80ud2.<slug> --array=<i> --dependency=singleton kernel/r80_ud2/lanes.sbatch   # i = 0, 1
cd $T && sbatch --parsable -J isla.clud.<slug>   --array=<i> --dependency=singleton kernel/cl_ud/lanes.sbatch     # i = 0..7
```

| run | first link | chain-ahead link | job name (chain) | node |
|---|---|---|---|---|
| r80ud_dr_gt_h256_seed42 | 284059_[0] | 284083 | isla.r80ud2.dr-gt-h256-seed42 | nvl72d040-T03 |
| r80ud_dr_gt_h256_seed43 | 284188_[1] | 284202 | isla.r80ud2.dr-gt-h256-seed43 | nvl72d008-T12 |
| clud_vol_gt_seed42 | 284060_[0] | 284130 | isla.clud.vol-gt-seed42 | nvl72d024-T08 |
| clud_vol_gt_seed43 | 284189_[1] | 284201 | isla.clud.vol-gt-seed43 | nvl72d047-T16 |
| clud_vol_t_seed42 | 284061_[2] | 284129 | isla.clud.vol-t-seed42 | nvl72d013-T13 |
| clud_vol_t_seed43 | 284190_[3] | 284203 | isla.clud.vol-t-seed43 | nvl72d018-T10 |
| clud_suv_gt_seed42 | 284062_[4] | 284145 | isla.clud.suv-gt-seed42 | nvl72d052-T04 |
| clud_suv_gt_seed43 | 284191_[5] | 284204 | isla.clud.suv-gt-seed43 | nvl72d021-T05 |
| clud_suv_t_seed42 | 284063_[6] | 284144 | isla.clud.suv-t-seed42 | nvl72d052-T15 |
| clud_suv_t_seed43 | 284192_[7] | 284200 | isla.clud.suv-t-seed43 | nvl72d015-T10 |

Logs: `$T/runs/<run>/train.log` (every link appends); Slurm `$T/sbatch_logs/<job name>-<jobid>_<idx>.log`; markers
`$T/STATUS_{r80,cl}<jobid>_<idx>`, `$T/RUN_*`, `$T/BLOCKED_<run>` (the reference launchers' layout).

Monitor: `ssh lhr2 "squeue -u psharpe -h -o '%i %j %T %M %R' | grep -E 'isla\.(r80ud2|clud)\.'"`;
`ssh lhr2 'bash $T/kernel/drive_fix2_verify/accept_check.sh'`.

### Acceptance at the first link: all 10 ACCEPTED (2026-10-19 09:41 PDT = 12:41 EDT)

Readout `kernel/drive_fix2_verify/accept_check.sh` (full output `partA/accept_first_link.txt`). Every lane: ≥ 185 training
steps, 0 non-finite loss lines, 0 tracebacks, and the override both in train.log's Config block and in the saved
`runs/<run>/resolved_config.yaml` (`global_embedding: global_data.U_inf_dir`, `global_dim: 3`; or `fx.source:
global_data.U_inf_dir`, `functional_dim: 3`).

| run | steps at check | epoch-0 train / val loss | s/step | epochs/h | active h left |
|---|---|---|---|---|---|
| r80ud_dr_gt_h256_seed42 | 2,241 | (epoch 19: 0.00237 / 0.00238) | 0.39 | 73.5 | 6.5 |
| r80ud_dr_gt_h256_seed43 | 324 | (epoch 2: 0.0178 / 0.0165) | 0.38 | 71.3 | 7.0 |
| clud_vol_gt_seed42 | 638 | (epoch 4: 0.0219 / 0.0187) | 0.99 | 31.3 | 15.8 |
| clud_vol_gt_seed43 | 185 | 0.0736 / 0.0559 | 0.97 | 30.2 | 16.5 |
| clud_vol_t_seed42 | 1,870 | (epoch 16: 0.00447 / 0.00443) | 0.27 | 89.7 | 5.4 |
| clud_vol_t_seed43 | 428 | (epoch 2: 0.0364 / 0.0334) | 0.27 | 83.3 | 6.0 |
| clud_suv_gt_seed42 | 1,203 | (epoch 5: 0.00458 / 0.00436) | 0.39 | 38.7 | 12.8 |
| clud_suv_gt_seed43 | 350 | 0.0469 / 0.0215 | 0.39 | 38.9 | 12.8 |
| clud_suv_t_seed42 | 1,386 | (epoch 5: 0.00503 / 0.00465) | 0.26 | 44.4 | 11.1 |
| clud_suv_t_seed43 | 365 | 0.0651 / 0.0325 | 0.27 | 41.6 | 12.0 |

The epoch rates come from the first minutes of the first link and include its startup, so they are rough.

### Expected training finish (EDT), from active hours left, 3h59 links and LHR's queue between links (new jobs start at once today)

| lanes | links | expected finish |
|---|---|---|
| clud_vol_t (Transolver, Volume) | 2 | 2026-10-19 19:00-21:00 |
| r80ud_dr_gt (GeoTransolver, DrivAerML Surface) | 2 | 2026-10-19 19:30-22:00 |
| clud_suv_t (Transolver, SHIFT-SUV) | 3-4 | 2026-10-20 00:30-03:30 |
| clud_suv_gt (GeoTransolver, SHIFT-SUV) | 4 | 2026-10-20 01:30-04:30 |
| clud_vol_gt (GeoTransolver, Volume) | 5 | 2026-10-20 05:00-08:00 |

### Scoring (CHAR protocol: native / area / posed, final checkpoint, float32, code_v20 + recipe_v8, all at 200,000 cells)

Run once a lane's train.log has "Training completed"; a row whose lane is unfinished is a no-op (it writes a NO-OP line and no
`.done`), so the command can simply be repeated until every row has a `.done`. It retries a failed row on the next submission
and gives up after 3 attempts.

```
ssh lhr2 'cd /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/cl_ud && sbatch --array=0-7 score.sbatch'
# subsets, when only some lanes have finished (block b = rows 4b..4b+3 of score_rows.tsv):
#   r80ud_dr_gt rows 0-5 -> --array=0-1;  clud_vol_gt rows 6-11 -> 1-2;  clud_vol_t rows 12-17 -> 3-4
#   clud_suv_gt rows 18-23 -> 4-5;  clud_suv_t rows 24-29 -> 6-7
```

- Scorer `kernel/cl_ud/score.sbatch` 561a8ccf = `kernel/r80_ud/score.sbatch` d5d16eef (the R2 scorer, itself a copy of
  `kernel/a2r/score.sbatch` 8e238045) with job name `isla.ud2.score`, default root `kernel/cl_ud`, output tree `evals_ud2`.
  The inference command is unchanged.
- Rows `kernel/cl_ud/score_rows.tsv` 3faedd87: 30 rows (10 lanes x 3 protocols), the lane's dataset (or its `_area` /
  `_posed` twin), model and extra, code_v20, recipe_v8, `.venv-recipe`, **200,000 cells for every row**.
- Output `$T/evals_ud2/<run>/<protocol>/<run>/metrics.jsonl`. Reduce as `scratch/isla-drive-fix-2026-10-18/reduce_drivefix.py`
  (infer_summary row, else the mean of the per-sample rows; seed means): fetch with
  `scp -r lhr2:$T/evals_ud2/<run> <dir>/scores/evals_ud2/` and run it with `root` pointed at `<dir>/scores`.
- Cell counts follow the CHAR rule for each reference: GeoTransolver 256 is scored at 200,000 (its references
  `r80_dr_gt_h256`, `cl_r_vol_gt`, `cl_r_suv_gt` were). For Transolver the reference rows `cl_r_vol_t` and `cl_r_suv_t` were
  also scored at 200,000 on AGA (`cl_r_*` maps to 200,000; only `*_c100k_*` and AirfRANS map to 100,000), so the CHAR scoring
  of the new Transolver lanes **is already the 200,000-cell scoring of Part B**: no second scoring is needed. The scored
  set then equals ISLA A2's (Part B checks the identity on these two datasets).
- Comparators: the reference rows in `examples/cfd/isla/results/char/{drivaerml_surface,drivaerml_volume,shift_suv_surface}_a2.json`
  ("GeoTransolver 256", "Transolver 512 x 16 at 100k cells", the latter scored at 200,000 on Volume and SHIFT-SUV).
