# DRIVE-FIX jobs (LHR), 2026-10-18

`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`; preregistration:
`PREREG.md`. Local copies of every new cluster file are in `cluster/`; md5 listing in `verify/md5s.txt`.

## Training lanes (submitted 2026-10-18 13:42 PDT = 16:42 EDT, normal QoS, 03:59:00 singleton chains)

| arm | run id (`$T/runs/<run>`) | first link job | job name (chain) | launcher row |
|---|---|---|---|---|
| R1 | afu_ud_gt_seed42 | 271344_[0] | isla.afud.ud-gt-seed42 | `kernel/afu_ud/lanes.tsv` 0 |
| R1 | afu_ud_gt_seed43 | 271345_[1] | isla.afud.ud-gt-seed43 | 1 |
| R1 | afu_ud_t_seed42 | 271347_[2] | isla.afud.ud-t-seed42 | 2 |
| R1 | afu_ud_t_seed43 | 271348_[3] | isla.afud.ud-t-seed43 | 3 |
| R2 | r80ud_dr_t_h512_l16_c100k_seed42 | 271349_[0] | isla.r80ud.dr-t-h512-l16-c100k-seed42 | `kernel/r80_ud/lanes.tsv` 0 |
| R2 | r80ud_dr_t_h512_l16_c100k_seed43 | 271350_[1] | isla.r80ud.dr-t-h512-l16-c100k-seed43 | 1 |
| R3 | lugbc_gt_seed42 | 271351_[0] | isla.clbc.gt-seed42 | `kernel/lug_bc/lanes.tsv` 0 |
| R3 | lugbc_gt_seed43 | 271352_[1] | isla.clbc.gt-seed43 | 1 |

Submission commands (from `launch.log`):

```
bash $T/kernel/afu_ud/submit.sh lanes 0 3
cd $T && sbatch --parsable -J isla.r80ud.<slug> --array=<i> --dependency=singleton kernel/r80_ud/lanes.sbatch   # i = 0, 1
cd $T && sbatch --parsable -J isla.clbc.<slug>  --array=<i> --dependency=singleton kernel/lug_bc/lanes.sbatch   # i = 0, 1
```

The R2 and R3 submissions are `kernel/lane_launch.sh`'s `sbatch` call (the way the REF-80 and CHAR lanes were launched)
without its append to the shared fleet manifest `kernel/p200k_fleet/manifest.tsv`, which is not touched.

Logs: `$T/runs/<run>/train.log` (every link appends). Slurm logs: R1 `$T/kernel/afu_ud/sbatch_logs/`; R2 and R3
`$T/sbatch_logs/<job name>-<jobid>_<idx>.log` (submitted from `$T`, as the reference launchers do), markers
`$T/STATUS_{r80,cl}<jobid>_<idx>`, `$T/RUN_*`, `$T/BLOCKED_<run>` (the reference launchers' layout).

Monitor: `ssh lhr2 "squeue -u psharpe -h -o '%i %j %T %M %R' | grep -E 'isla\.(afud|r80ud|clbc)\.'"`;
`ssh lhr2 "grep -h 'Epoch \[' $T/runs/<run>/train.log | tail -1"`.

## Expected finish

Reference wall clock, active training: AirfRANS GeoTransolver about 6 h, Transolver about 8.5 h (afu_r lanes); DrivAerML
Transolver 512 x 16 about 9 h (REF-80); lug GeoTransolver 256 about 8 h. That is 2-3 links each. With LHR's queue between
links (start estimate about 1 h at submission), training should finish 2026-10-19 between about 08:00 and 20:00 EDT; the
per-lane estimate from the first link's epoch rate is in the acceptance table below.

## Acceptance (first link): all 8 ACCEPTED, 2026-10-18 13:53 PDT (16:53 EDT)

The readout is `kernel/drive_fix_verify/accept_check.sh` (local copy in `cluster/drive_fix_verify/`); rates come from
`rate.py`. Every first link started 13:45 to 13:48 PDT. At the check: 0 non-finite loss lines, 0 tracebacks, and the
override appears in the Config block that train.log prints.

| run | node | steps at check | epoch 0 train / val loss | override in the logged config | epochs/h | active h left |
|---|---|---|---|---|---|---|
| afu_ud_gt_seed42 | nvl72d008-T07 | 1,156 | 0.1098 / 0.0731 | `global_embedding: [global_data.U_inf_dir, global_data.U_inf_mag_std]`, `global_dim: 4` | 75 | 6.5 |
| afu_ud_gt_seed43 | nvl72d004-T15 | 1,182 | 0.1070 / 0.0626 | same | 75 | 6.5 |
| afu_ud_t_seed42 | nvl72d055-T06 | 909 | 0.1025 / 0.0796 | `fx.source: [global_data.U_inf_dir, global_data.U_inf_mag_std]`, `expand_like: embedding`, `functional_dim: 4` | 57 | 8.7 |
| afu_ud_t_seed43 | nvl72d043-T13 | 902 | 0.1081 / 0.0694 | same | 55 | 9.0 |
| r80ud_dr_t_h512_l16_c100k_seed42 | nvl72d026-T04 | 337 | 0.0672 / 0.0420 | `fx.source: global_data.U_inf_dir`, `functional_dim: 3` | 96 | 5.2 |
| r80ud_dr_t_h512_l16_c100k_seed43 | nvl72d059-T09 | 332 | 0.0723 / 0.0430 | same | 95 | 5.2 |
| lugbc_gt_seed42 | nvl72d047-T16 | 983 | 0.1023 / 0.0298 | `geometry: [boundaries.surface.cell_centroids, ...normals, ...bc_fixed, ...bc_hole, ...bc_loaded, ...bc_pressure]`, `geometry_dim: 10`, `drop_in_file_boundaries: false` | 48 | 10.5 |
| lugbc_gt_seed43 | nvl72d034-T01 | 1,040 | 0.1013 / 0.0235 | same | 48 | 10.3 |

The chain-ahead links were queued at the first link's start: 271386 to 271389 (afud) and 271423 to 271426 (r80ud, clbc).

Step times match the reference lanes', so the input change costs nothing measurable:

| arm | reference s/step | this arm s/step |
|---|---|---|
| AirfRANS GeoTransolver | 0.201 | 0.205 |
| AirfRANS Transolver | 0.262 | 0.262 |
| DrivAerML Transolver | 0.271 | 0.272 |
| lug GeoTransolver | 0.096 | 0.11 (+10 %: the surface context now has up to 8.6k cells and 10 channels) |

Early observation, not a result: on R2 the unit drive trains faster at the start. At epoch 3 the validation loss is
0.0142 / 0.0136 (seeds 42 / 43), against 0.0202 for the reference seed 42 (epoch 2: 0.0166 against 0.0322). The
HiLift unit-drive ladder showed the same "slow start" under the physical drive. The pre-registered rule reads only the
final checkpoint.

Expected training finish, from active hours left plus 3h59 links and the queue between links (EDT):

| lanes | expected finish |
|---|---|
| R2 | 2026-10-18 23:00 to 2026-10-19 02:00 |
| R1 GeoTransolver | 2026-10-19 00:30 to 03:00 |
| R1 Transolver | 2026-10-19 03:00 to 07:00 |
| R3 | 2026-10-19 05:00 to 09:00 |

Scoring takes about 1 h per batch after that.

## Scoring (run once a lane's train.log has "Training completed"; each scorer is a no-op without it, leaving no .done)

R1, AF-UNIFORM protocol (native / area / posed, 200 full_test airfoils, 100,000 cells, float32, code_v20 + recipe_v8ud):

```
ssh lhr2 'bash /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/afu_ud/score_when_done.sh'
# or per run: bash $T/kernel/afu_ud/score_submit.sh afu_ud_gt_seed42 afu_ud_gt_seed43 afu_ud_t_seed42 afu_ud_t_seed43
```

Output `$T/evals_afu_ud/<run>/<protocol>/<run>/metrics.jsonl`. Reduce as `results/char/afu_protocol_artifacts_2026-10-14.py`
(per-case relative L2 over the 200 cases, infer_summary else the per-sample rows; seed means), e.g. fetch with
`scp lhr2:$T/evals_afu_ud/<run>/<p>/<run>/metrics.jsonl <dir>/<run>/<p>/` and point `--src` at that tree.

R2, CHAR DrivAerML Surface (native / area / posed, 100,000 cells, float32, code_v20 + recipe_v8):

```
ssh lhr2 'cd /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/r80_ud && sbatch --array=0-1 score.sbatch'
```

Rows `kernel/r80_ud/score_rows.tsv` 0-5 (block 0 = rows 0-3, block 1 = rows 4-5); output
`$T/evals_r80ud/<run>/<protocol>/<run>/metrics.jsonl`. Decision rule: PREREG.md (native seed means of pressure_l2 and
wss_l2 against 0.04368 and 0.06961, band [0.97, 1.03]).

R3, CHAR lug (native / area / posed, 200,000 cells = the whole part, float32, code_v20 + recipe_v8):

```
ssh lhr2 'cd /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/lug_bc && sbatch --array=0-1 score.sbatch'
```

Rows `kernel/lug_bc/score_rows.tsv` 0-5; output `$T/evals_lugbc/<run>/<protocol>/<run>/metrics.jsonl`; readout
von_mises_l2, GeoTransolver native ÷ ISLA A2 native 0.001726.

The a2r-derived scorers (R2, R3) retry an unfinished row on the next submission and give up after 3 failed attempts; a
scoring job submitted before training finished only writes NO-OP lines.

## Verification jobs

| job | what | result |
|---|---|---|
| LHR 271171 (cpu-short) | sha256 of LHR `datasets/PhysicsNeMo-DrivaerML` (40,173 files: `run_*/domain_run_*.pdmsh/**` + manifest.json) against AGA's SHA256SUMS (JHB `agents/2026-10-16-dataset-mirror/PhysicsNeMo-DrivaerML.SHA256SUMS`) | 0 mismatches (`cluster/drive_fix_verify/sha_drivaerml-271171.log`) |
| login node | `af_stats.py`: U_inf_mag over the 800 full_train / 200 full_test case files | full_train mean 62.1160, std 17.9140 (ddof 0) |
| LHR 271294 (short) | 4 x 20-step smokes through `smoke_wrap.py` | all rc 0, finite; inputs as intended (PREREG.md, Smokes) |

## Snapshots and md5s

- `recipe_v8ud` = `recipe_v8` (SNAPSHOT_HEAD 343b284eb + the AF-UNIFORM line) copied 2026-10-18, plus
  `datasets/airfrans_uvolume_ud.yaml` 81557b30, `_ud_area.yaml` 36f5da31, `_ud_posed.yaml` 62923787 (sources
  airfrans_uvolume 1a121d80, _area 023a2e6e, _posed 8de029d7); SNAPSHOT_HEAD a4926a23 (new last line).
- Model source, the same in code_v18 and code_v20, on AGA and LHR (whole trees: 1,529 `.py` files each, identical across
  the clusters):
  - `physicsnemo/models/geotransolver/geotransolver.py` f59a2101, `context_projector.py` 348c00d0,
    `flare_plus_plus.py` f3960a77, `activation_checkpointing.py` 1de22e84, `__init__.py` c887ea32
  - `physicsnemo/models/transolver/transolver.py` 0096c59e, `__init__.py` 848e8b41
  - the smokes imported exactly these files (`SMOKE wrapper:` line of each smoke log).
- Launchers (new) and their sources:
  - `kernel/afu_ud/lanes.sbatch` 00965068 = `kernel/airfrans_u/lanes.sbatch` 517f9107 with `#SBATCH -J isla.afud`;
    `submit.sh` fc0333f6 (from 2eef9094: AFU_ROOT, RECIPE=recipe_v8ud, names isla.afud.*); `lanes.tsv` b2e3ad5e;
    `score.sbatch` b23f4507 (from f629ffc5: AFU dir, evals_afu_ud, job name, `afu_ud_*` -> code_v20 + recipe_v8ud);
    `score_submit.sh` 06656844; `score_when_done.sh` f4cb7e9b.
  - `kernel/r80_ud/lanes.sbatch` 907d4c67 = `kernel/char/r80_lanes.sbatch` 29e98972 with `-J isla.r80ud` and SELF / LANES
    paths; `lanes.tsv` a8e43a45; `score.sbatch` d5d16eef (from `kernel/a2r/score.sbatch` 8e238045: job name, default
    root kernel/r80_ud, evals_r80ud); `score_rows.tsv` 8e9bb96e.
  - `kernel/lug_bc/lanes.sbatch` 184bf79f = `kernel/char/cl_lanes.sbatch` 0ad8638f with `-J isla.clbc` and SELF / LANES
    paths; `lanes.tsv` 1be5509a; `score.sbatch` aded5fc1 (from 8e238045: job name, root kernel/lug_bc, evals_lugbc);
    `score_rows.tsv` 68410295.
