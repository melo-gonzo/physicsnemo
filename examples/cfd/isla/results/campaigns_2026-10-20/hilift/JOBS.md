# UDRV-HL jobs (JHB, 2026-10-18)

```bash
J=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-17-isla-a2r   # JHB task dir
K=$J/kernel/udrv_hl                                                                                   # this launcher
```

Prereg: `PREREG.md`. Local copies of every new cluster file: `cluster/udrv_hl/` (with `MD5SUMS`, `SNAPSHOT_HEAD`, smoke logs
in `smoke/logs/`). Verification artifacts: `verify/` (code md5 listings and diff, recipe yaml copies, resolved configs of
the REF-80 reference runs and of the smoke runs).

## Files (md5, all new; nothing existing was changed)

| file | md5 | origin |
|---|---|---|
| lanes.sbatch | 8192a90b0d5dd5aec2bfb50a28af3cbf | byte copy of kernel/a2r_hl |
| ckpt_preflight.py | e34bd6119ad1e0dc8a71170927dc8116 | byte copy of kernel/a2r_hl |
| site.env | d76a0b073455fbe1290b63dea71ac5b5 | byte copy of kernel/a2r_hl |
| submit.sh | 84de444c5a10d8c1b4f65fb169d20a32 | kernel/a2r_hl/submit.sh (3ba4dec1), `isla.a2r.` -> `isla.udrv.` (4 places) |
| lanes.tsv | f758a444e79cb4e8deda96b7fa093ee3 | new, 4 rows |
| score.sbatch | 8e238045248bbb2ef3407d4cb377812c | byte copy of kernel/a2r_hl |
| score_rows.tsv | 933ca5cb804135c5337c374b373350b6 | new, 18 rows (6 blocks) |
| smoke/probe_drive.py | 3a1bad7399aac41fcac7738c09e249ae | new |
| smoke/smoke.sbatch | f22e12fad28720511aafaa3a274f0911 | new |
| smoke/infer_smoke.sbatch | 3fabfdb24544b7a4044d5f04de8d6825 | new |
| SNAPSHOT_HEAD | bcec0549a4b693dd5bbbd1fba0fb9985 | one line per file |

## Lanes (submitted 2026-10-18 13:20 PDT by `cd $K && bash submit.sh lanes 0 3`; normal QoS, 03:59:00 singleton chains)

| row | run id | job name | first job | reference REF-80 row |
|---|---|---|---|---|
| 0 | udrv_hl_gt_seed42 | isla.udrv.hl-gt-seed42 | 1097254_0 | idx 32 r80_hl_gt_h256_seed42 |
| 1 | udrv_hl_gt_seed43 | isla.udrv.hl-gt-seed43 | 1097255_1 | idx 46 r80_hl_gt_h256_seed43 |
| 2 | udrv_hl_t_seed42 | isla.udrv.hl-t-seed42 | 1097256_2 | idx 38 r80_hl_t_h512_l16_c100k_seed42 |
| 3 | udrv_hl_t_seed43 | isla.udrv.hl-t-seed43 | 1097257_3 | idx 52 r80_hl_t_h512_l16_c100k_seed43 |

Runs land in `$J/runs/<run id>/` (train.log, resolved_config.yaml, checkpoints); link status in `$K/STATUS_a2r<job>_<row>`
(the TAG prefix `a2r` comes from the byte-copied lanes.sbatch), chain-ahead submissions in `$K/sbatch_logs/resubmit.log`.

Smoke jobs (done): 1097163 (probe + one-epoch training, short QoS, 8 min) and 1097266 (inference path, short QoS, 12 min).

## Expected finish

- Compute: about 500 epochs at roughly 2.5 min per epoch on 4 GB300 (the smoke ran 255 two-GPU steps plus validation in
  4.7 min per model; the A2 ISLA lanes on the same data ran about 24 epochs per hour), so about 20-21 h, 6 links.
- Queue: on 2026-10-18 the a2r_hl links waited 5 h (1084590_8, submitted 20:18 PDT Oct 17, started 01:18) and then over
  12 h (1088956_8, pending since 01:18 PDT Oct 18). The chain-ahead link queues while a link runs, so a link cycle is
  about max(4 h, queue wait). At 8-12 h waits: about 2-3 days after the first link starts, so around Oct 21-22 PDT. At
  4-5 h waits: about 1.5 days, around Oct 20.

## Acceptance (after the first link of each lane has run at least 15 minutes)

```bash
ssh jhb1 "for r in udrv_hl_gt_seed42 udrv_hl_gt_seed43 udrv_hl_t_seed42 udrv_hl_t_seed43; do f=$J/runs/\$r/train.log; \
  echo \$r steps=\$(grep -c '] Loss:' \$f) tb=\$(grep -c Traceback \$f) nonfinite=\$(grep -ciE 'Loss: (nan|inf)|non-finite' \$f) \
  drive=\$(grep -cE 'global_embedding: global_data.U_inf_dir|source: global_data.U_inf_dir' \$f); \
  grep -E 'Epoch [0-9]+ \[' \$f | tail -1 | cut -c1-140; done; ls $J/kernel/udrv_hl/BLOCKED_* 2>/dev/null"
```

- `logging.log_every_n_steps=50`, so one Loss line is 50 steps: accept at `steps >= 2` Loss lines plus the epoch counter
  (epoch >= 1 of 128 steps per epoch is at least 128 steps), `tb=0` (a traceback at a link's time limit is the SIGTERM, not a
  failure), `nonfinite=0`, `drive >= 1`.
- Smoke reference values: first-step loss 0.70 (GT) and 0.59 (T); Parameters 9,047,301 (GT) and 47,239,305 (T), equal to the
  REF-80 lanes'; peak memory 78.9 GB (GT) and 62.5-64.3 GB (T), equal to the REF-80 lanes' 78.85 GB.

## Scoring (after "Training completed" in each lane's train.log)

Rows (`$K/score_rows.tsv`; idx run_id protocol dataset model extra code recipe cells venv; block b = rows 4b..4b+3):

| block | rows | run | cells | extra |
|---|---|---|---|---|
| 0 | 0-2 native/area/posed | udrv_hl_gt_seed42 | 200000 | `+training.val_interval=25 forward_kwargs.global_embedding=global_data.U_inf_dir` |
| 1 | 4-6 | udrv_hl_gt_seed43 | 200000 | same |
| 2 | 8-10 | udrv_hl_t_seed42 | 100000 | `+training.val_interval=25 model.n_hidden=512 model.n_layers=16 forward_kwargs.fx.source=global_data.U_inf_dir` |
| 3 | 12-14 | udrv_hl_t_seed43 | 100000 | same |
| 4 | 16-18 | a2r_hl_isla_seed42 | 200000 | `+training.val_interval=25 model.hidden=512 model.n_slices=512 model.compile_blocks=true` |
| 5 | 20-22 | a2r_hl_isla_seed43 | 200000 | same |

All rows: dataset highlift_surface_medium (the scorer appends `_area` / `_posed`), code_a2r, recipe_a2r_hl, venv
`$J/.venv-recipe`. The scorer is a no-op for a row whose lane has not completed, so a block can be resubmitted safely.

```bash
# baselines (blocks 0-3) once the four lanes completed; ISLA (blocks 4-5) once a2r_hl_isla_seed42/43 completed
ssh jhb1 "cd $J/kernel/udrv_hl && sbatch --export=ALL,A2R_TASK=$J,A2R_ROOT=$J/kernel/udrv_hl,EVAL_ROOT=$J/evals_udrv --array=0-3 score.sbatch"
ssh jhb1 "cd $J/kernel/udrv_hl && sbatch --export=ALL,A2R_TASK=$J,A2R_ROOT=$J/kernel/udrv_hl,EVAL_ROOT=$J/evals_udrv --array=4-5 score.sbatch"
# check: 18 .done markers, 180 samples per protocol
ssh jhb1 "find $J/evals_udrv -name .done | wc -l; grep -h '\"infer_summary\"' $J/evals_udrv/*/*/*/metrics.jsonl | grep -o '\"num_samples\": [0-9]*' | sort | uniq -c"
# fetch and reduce (local)
cd /home/psharpe/gh/physicsnemo/scratch/isla-udrv-hl-2026-10-18 && rsync -a --exclude predictions jhb1:$J/evals_udrv/ scores/ && python udrv_table.py
```

- EVAL_ROOT `$J/evals_udrv` keeps these scores apart from the A2-RERUN tree `$J/evals_a2r`. If the A2-RERUN program has
  already scored a2r_hl_isla_seed42/43 with identical rows (same extra, 200000, recipe_a2r_hl, code_a2r), copy those into
  `scores/` instead of running blocks 4-5.
- `udrv_table.py` writes `udrv_hl_table.md` and `udrv_hl_scores.json`: the table of the three arms plus the CHAR record,
  and the P1, P2 (ratio, cases lower, band) and P3 readouts of PREREG.md.
- The scoring path was smoke-tested on the one-epoch smoke checkpoints (job 1097266, 180 samples each, 0 tracebacks):
  GT native pressure 0.339 with the override against 1.520 without it, so the rows' override reaches inference; GT posed
  0.912; T native 0.316.

## Resubmitted on batch_long (supervisor, 2026-10-18 20:26 EDT)

The four first links 1097254_0 to 1097257_3 never started (JHB `batch` estimate ~14.5 h per link start: 6,262 nodes ahead, our
account has RawShares 0). They were cancelled while PENDING (0 GPU-h) and resubmitted with `bash submit.sh lanes 0 3` after
`site.env` was changed to `LANE_TIME=1-06:00:00` and `LANE_SBATCH_EXTRA="--segment=1 -p batch_long"` (backup
`site.env.bak-2026-10-18-batch`; new md5 02506885, appended to MD5SUMS). `batch_long` shares the same 9,216 nodes and PriorityTier
as `batch`, so a link should start about as soon, and one 30 h link replaces about six 3h59 links and their queue waits. New
first links: 1100111_0 (hl-gt-seed42), 1100112_1 (hl-gt-seed43), 1100113_2 (hl-t-seed42), 1100114_3 (hl-t-seed43). Each link
exports the batch_long settings to its chain-ahead link. Rows, recipe, code and the scoring plan are unchanged.

## Moved to LHR (2026-10-19; HILIFT-LHR, handoff `scratch/isla-hilift-lhr-2026-10-19/HANDOFF.md`)

```bash
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0   # LHR campaign root
K=$T/kernel/udrv_hl                                                                                   # LHR launcher
```

- The HiLift subset (690 cases, 19,241 files, 14,304,669,734,189 bytes) was copied AGA -> LHR
  `datasets/PhysicsNeMo-HighLiftAeroML-medium`. `.COPY_DONE` was written 14:16 PDT. It was verified 17:28 EDT: dry-run
  reconcile empty (rc 0), counts and bytes equal, root md5s equal, 690 / 690 manifest cases present, 40 / 40 sample
  md5 OK.
- JHB first links 1100111_0..1100114_3 were still PENDING (never started). They were cancelled 17:28 EDT by job id and by
  name (CANCELLED, Elapsed 0).
- LHR launcher `$K`: byte copies of the JHB files, except `site.env` (LHR paths, the LHR `.venv-recipe`, normal QoS,
  03:59:00 links, `--segment=1`) and `score_rows.tsv` (venv column; cells, below). md5s are in `$K/MD5SUMS` and
  `$K/SNAPSHOT_HEAD`. Recipe: `$T/recipe_a2r_hl`. It equals LHR recipe_a2r except the HiLift root.
- Submitted 2026-10-19 17:47 EDT by `cd $K && bash submit.sh lanes 0 3`: **288792_0** (hl-gt-seed42), **288793_1**
  (hl-gt-seed43), **288795_2** (hl-t-seed42), **288796_3** (hl-t-seed43). Runs: `$T/runs/<run id>/`. Link status:
  `$K/STATUS_a2r<job>_<row>`.
- First links started 15:22-15:24 PDT. 288796_3 (hl-t-seed43) lost its node (NODE_FAIL 15:35 PDT, epoch 0, no
  checkpoint) and was requeued fresh under the same id at 15:36 PDT; its duplicate chain-ahead 290167 was cancelled
  (pending), and 289825 was kept. Chain-ahead links: 289797_0, 289824_1, 289827_2, 289825_3.
- Accepted 18:53 EDT, every lane at least 15 min into its link: steps 1951-3044 per link, tb 0, non-finite 0, the
  `U_inf_dir` override in both train.log and resolved_config.yaml (no physical `U_inf`), and Parameters and first-step
  loss equal to the smoke runs. Details: `scratch/isla-hilift-lhr-2026-10-19/RUN_2026-10-19.md`. LHR runs about 47
  (GT) to 55 (T) epochs/h, so completion is expected about 05:00-08:00 EDT 2026-10-20.
- The ISLA comparator `a2r_hl_isla_seed42/43` moved too. Its run dirs were copied JHB -> LHR (592 files, md5 identical)
  and its chain resumes from `checkpoint.0.285.pt`: `$T/kernel/a2r_hl`, **288498_8**, **288499_9** (submitted 17:34 EDT).

### Scoring plan change (T-SCORESET, 2026-10-19): every HiLift arm on the 200,000-cell set ISLA uses

The Transolver rows (8-10, 12-14) were changed from 100,000 to 200,000 cells. That makes every arm and protocol, ISLA,
GeoTransolver and Transolver alike, use the same sampling_resolution 200000. `$K/score_rows.tsv` is now md5 04fcc4be;
the 100,000-cell version is kept as `score_rows.tsv.bak-2026-10-19-c100k` (d57e2cbb). The Transolver lanes still train
at 100,000 cells (the REF-80 configuration); only the scoring set changes.

Identity of the sets: `$K/scoreset/hash_scoreset.py` replays `src/infer.py` up to its loop for each score row (same
Hydra overrides, `set_seed`, `build_dataloaders`, model instantiation, first-sample read, then `dataset[idx]` over the val
sampler). It writes a per-case sha256 of the geometry (`interior.points`, `interior.cell_centroids`, the vehicle
`points` / `cells`), of the targets (`interior.point_data.{pressure,tau_wall,velocity,density,temperature}`), and of
the whole sample. LHR job **288618** (short QoS, 13 min) ran it, with outputs in `$K/scoreset/out/`; local copy and
table: `scratch/isla-hilift-lhr-2026-10-19/scoreset/` (`scoreset_identity.md`). Result:

| protocol | arms | cases | sample order | geometry | targets | whole sample |
|---|---|---|---|---|---|---|
| native | ISLA, GeoTransolver, Transolver @ 200000 | 180 | equal | equal | equal | equal |
| area | same | 180 | equal | equal | equal | equal |
| posed | same | 180 | equal | equal | equal | equal |

- Negative control: Transolver native at 100,000 cells differs from the 200,000-cell set on 180/180 cases, in geometry
  and targets. The area and posed sets differ from native on 180/180, as designed.
- The subsample comes from a per-sample generator seeded by (training.seed 42 from base.yaml, epoch 0, index). The scorer
  passes no run seed, so the set does not depend on the model or the run.
- Training normalization: `norm_stats.pt` of a2r_hl_isla_seed42/43 equals the dataset-YAML stats. The udrv runs had not
  saved one yet. Before scoring, re-check them (command below); the scorer replaces the YAML stats with a run's own, so
  a mismatch would change that run's training-space targets.

Exact scoring commands on LHR (after "Training completed" in every lane's train.log; the scorer is a no-op for an
unfinished lane):

```bash
# 0. norm-stats check of all six runs against the dataset YAML (expect "matches_dataset_yaml": true for all six)
ssh lhr1 "cd $K/scoreset && sbatch normcheck.sbatch"     # then: cat $K/scoreset/sbatch_logs/normcheck_<job>.log
# 1. the four baselines (blocks 0-3: GT rows 0-6, T rows 8-14, all 200000 cells)
ssh lhr1 "cd $K && sbatch --export=ALL,A2R_TASK=$T,A2R_ROOT=$K,EVAL_ROOT=$T/evals_udrv --array=0-3 score.sbatch"
# 2. the ISLA comparator (blocks 4-5, rows 16-22, 200000 cells). Same rows as $T/kernel/a2r_hl/score_rows.tsv: if the
#    A2-RERUN scores ($T/evals_a2r/a2r_hl_isla_seed4{2,3}/) already exist, copy those instead.
ssh lhr1 "cd $K && sbatch --export=ALL,A2R_TASK=$T,A2R_ROOT=$K,EVAL_ROOT=$T/evals_udrv --array=4-5 score.sbatch"
# 3. check: 18 .done markers, 180 samples per protocol, and every row ran at 200000 cells
ssh lhr1 "find $T/evals_udrv -name .done | wc -l; grep -h '\"infer_summary\"' $T/evals_udrv/*/*/*/metrics.jsonl | grep -o '\"num_samples\": [0-9]*' | sort | uniq -c; \
  grep -ho 'cells=[0-9]*' $T/evals_udrv/*/*.log | sort | uniq -c"
# 4. fetch and reduce (local)
cd /home/psharpe/gh/physicsnemo/scratch/isla-udrv-hl-2026-10-18 && rsync -a --exclude predictions lhr1:$T/evals_udrv/ scores/ && python udrv_table.py
```

Optional secondary record, not run unless asked: Transolver at its training resolution, 100,000 cells. This is
like-for-like with the CHAR record's physical-drive Transolver (scored at 100k), which P1 compares against. It uses
the kept rows file and a separate output tree, so the primary 200k scores are untouched:
`ssh lhr1 "cd $K && sbatch --export=ALL,A2R_TASK=$T,A2R_ROOT=$K,EVAL_ROOT=$T/evals_udrv_c100k,SCORE_ROWS=$K/score_rows.tsv.bak-2026-10-19-c100k --array=2-3 score.sbatch"`.

## Readout (2026-10-20, readout lane): see RESULTS.md

- 299041 (blocks 0-3) checked: 12 .done, 180 samples per protocol, cells=200000, epoch 500. A2 ISLA scores copied from
  `$T/evals_a2r` (identical rows and scorer) into `scores/`.
- Physical-drive predecessors: checkpoints copied AGA -> LHR `$T/runs_from_aga/` (aga2dc rsync, md5 equal; checkpoints
  deleted after scoring). New launcher dir `$K/phys_s200k/` (local copies in `phys/`): `hash_phys.sbatch` (299154,
  set identity + norm stats), `leafdiff.sbatch` (299279), `protocol_eval_s200k.sbatch` (299159, the CHAR scorer copy at
  200k -> `$T/evals_udrv/r80_hl_*_s200k`), and the A2 scorer on the same checkpoints (299326 -> `$T/evals_udrv_phys_a2r`):
  `cd $K/phys_s200k && sbatch -q short --time=01:30:00 --array=0-2%2 --export=ALL,A2R_TASK=$K/phys_s200k/a2r_task,A2R_ROOT=$K,SCORE_ROWS=$K/phys_s200k/score_rows_phys_a2r.tsv,EVAL_ROOT=$T/evals_udrv_phys_a2r $K/score.sbatch`.
- New recipe snapshot `$T/recipe_v8_hl`: recipe_v8 with the HiLift root set to the LHR copy (one line).
- Reduce: `rsync -a --exclude predictions lhr2:$T/evals_udrv/ scores/ && python udrv_table.py`.
