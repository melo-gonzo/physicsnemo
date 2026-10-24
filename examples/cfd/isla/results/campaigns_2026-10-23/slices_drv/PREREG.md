# SLICES-DRV: ISLA 0.9.0 at 128 slices on DrivAerML Volume, the second-dataset no-harm check (pre-registration, 2026-10-24)

Lane SLICES-DRV of ISLA (research-lane, launched 2026-10-24 about 00:07 EDT by the supervisor 01a13c2c). Written 00:18 EDT
2026-10-24, **before any job of this lane: nothing has been submitted, and no S checkpoint, log or score exists.**

## 1. Question

SLICES-AF (`../isla-slices-2026-10-23/RESULTS.md`, md5 53b1cdc2) found ISLA 0.9.0 with `n_slices=128` no different from
512 on full AirfRANS (S ÷ B 0.998 / 0.997 / 0.969 on pressure / velocity / nu_t, every field "no change") at 0.42 of the
step and 0.78 of the memory. That is a cost result on one dataset. Before 128 slices can enter the scope freeze (about
10-27), it needs a no-harm check on a second dataset. **Does 128 slices harm ISLA 0.9.0 on DrivAerML Volume** (body-only
configuration, the standing `r090_vol_isla`)? DrivAerML Volume is the likeliest place for harm: CAP-AUDIT found that far-field
points own slices from block 4 on (1-31 slices, at 0.5-9 times their share), and that ISLA's nu_t excess sits in the far
field (`../isla-cap-audit-2026-10-20/RESULTS.md`, Part B and the probes).

## 2. What I have seen before registering (disclosure)

- **B's scores, all of them** (`../isla-ref090-2026-10-20/r090_final_tables.md`, the DrivAerML Volume block), epoch 500,
  48 cases, float32, native:

  | field | seed 42 | seed 43 | mean | s_B (sd of ln, n-1) | two-seed gap |
  |---|---|---|---|---|---|
  | pressure | 0.037954 | 0.038801 | 0.038377 | 0.0156 | 2.2 % |
  | velocity | 0.057326 | 0.058084 | 0.057705 | 0.0093 | 1.3 % |
  | nu_t | 0.084012 | 0.085298 | 0.084655 | 0.0107 | 1.5 % |

  posed ÷ native: 1.0027 / 1.0035 (p), 1.0040 / 1.0033 (v), 1.0001 / 0.9998 (nu_t). **area ÷ native 1.0000 on every field
  and seed** (REF-090: ISLA draws by area in the model, so area equals native by construction: degenerate).
- B's training logs (`runs/r090_vol_isla_seed4x/metrics.jsonl`): median trainer `step_time_s` 0.593 s on both seeds
  (54,000 steps, 108 per epoch), peak `mem_gb` 155.1, worst validation loss over running minimum after epoch 10: 1.12
  (s42, epoch 59) and 1.13 (s43, epoch 68). Links (sacct): s42 298084, 298441, 300747 (+ 303090, 9 s); s43 298085 (failed
  at 4 min on the NFS cache race), 298443, 298506, 300826 (+ 303171, 10 s): about 40.4 / 40.9 GPU-h. The 2026-10-20 smoke
  297254 (r090 row 8): median step 0.662 s, peak allocated 128.6 GiB.
- SLICES-AF in full (PREREG, RESULTS, JOBS, reading, near-wall table); REF-090's RESULTS and final tables; COMPILE-AF's
  RESULTS sections 1-2; CAP-AUDIT's RESULTS (Part B, 15.10 drv ladder: single case, seed 44, s128 1.03 / 0.99 / 1.04 per
  halving on velocity / nu_t / pressure, "plateau"; the far-field nu_t level beyond 1 L_ref unchanged on every rung, e_R
  0.029-0.042; the probes and the erratum on L_ref = 5 m) and `tables/resid_drv_partB.md`.
- Nothing of S exists.

## 3. Arms

`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0` (LHR, GB300).

- **B (reused, not retrained):** `r090_vol_isla_seed42` / `_seed43`, rows 8-9 of `$T/kernel/r090/lanes.tsv` (md5 81d34e1d):
  code `$T/code_r090`, recipe `$T/recipe_r090`, `model=isla_volume`, `dataset=drivaer_ml_volume_reference`, extra
  `model.hidden=512 model.n_slices=512 model.compile_blocks=true` (launcher: `compile=false`, so the per-block default-mode
  compile, mix-order reduction on), lr 3.75e-4, 500 epochs, 200,000 cells, bf16 (site.env LANE_PRECISION), seeds 42 / 43.
  Scored by kernel/r090/score.sbatch (a4a24f5e), score_rows 32-34 and 36-38 (2104fd1e), `$T/evals_r090/`.
- **S:** B with one change, `model.n_slices=128`, on the same code, recipe and compile path (supervisor's ruling, 00:14 EDT
  10-24: the like-for-like rule decides it; the compiled-path defect needs the eager geometry region, and 3-D DrivAerML
  Volume runs the fused kernel, which 128 slices keeps since MAX_SLICES is 1024; guarded by section 4). Launcher
  `$T/kernel/slicesdrv/`:
  - `lanes.sbatch`, `ckpt_preflight.py`: byte copies of kernel/r090's (cmp; md5 8eecec8f / e34bd611);
  - `submit.sh`: kernel/r090's (7f559159) with only the job-name prefix `isla.slicesdrv.` (and its squeue grep, header line);
  - `site.env`: kernel/r090's (4a6ed5fb) with `LANE_SBATCH_EXTRA="--segment=1 --exclude=$T/kernel/slicesdrv/exclude_nodes.txt"`
    and a header line; `exclude_nodes.txt` = kernel/lreaz's five bad nodes (nvl72d015-T12, nvl72d058-T08, nvl72d052-T01,
    nvl72d057-T13, nvl72d043-T18);
  - `lanes.tsv`: kernel/r090 rows 8-9 re-indexed 0-1, group `slicesdrv`, run ids `slicesdrv_vol_isla_s128_seed42` /
    `_seed43`, `model.n_slices=512` → `model.n_slices=128`; nothing else differs (diff recorded in JOBS.md). Job names
    `isla.slicesdrv.vol-isla-s128-seed4x`; normal QoS, 03:59:00 chained singleton links.
  - `$T/kernel/slicesdrvb512/`: byte copies of kernel/r090's lanes.sbatch, site.env, lanes.tsv, ckpt_preflight.py, used only
    by the smoke's B unit (so it runs B's files without writing into kernel/r090).

## 4. Smoke and certification (one short-QoS job, before any lane; `isla.slicesdrv.smoke`, 01:00:00, bad nodes excluded)

kernel/r090's smoke.sbatch and smoke_wrap_r090.py (md5 32a2991e), copied to `kernel/slicesdrv/smoke/` with its paths, job
name and time limit changed and a certification stage appended. In order, on one node:

1. **B unit** (`slicesdrvb512:8`) and **S unit** (`slicesdrv:0`), each 50 training steps through the lane launcher with
   NO_CHAIN, one after the other.
2. **Certification training:** S's lane command for 2 epochs (row `slicesdrv_cert_s128_seed42` of `smoke/lanes_cert.tsv`:
   S row 0 with that run id and `epochs=2`; the lane's save_interval is 2, so this writes the first checkpoint; ATTRIB's
   certification did the same), through the lane launcher with NO_CHAIN into `smoke/runs/`.
3. **Block-wise compiled-against-eager check:** COMPILE-AF's `cmpaf_diag.py` (byte copy, md5 9fc35d83) with `--blockwise
   --cases 20` on that checkpoint: the first 20 of the 48 val cases of the scorer's loader (real DrivAerML Volume cases,
   native, float32), compiled blocks from the model's own `ISLA._block_callables` against the eager model. Beside it, the
   same check on B's final checkpoint `r090_vol_isla_seed42` (COMPILE-AF's row), as the reference for code_r090 at 512.

**Passes if** (every item; otherwise no lane is submitted and I report):
- both smoke units rc 0, no Traceback, `nonfinite=[]`; the certification training reaches "Training completed";
- **geometry kernel (guard 1):** S's train.log in the smoke and in the certification shows `ISLA geo_kernel=auto resolved to
  'fused'` on all four ranks (cuda:0-3) and no `resolved to 'eager'` anywhere; also the diag's `geo_kernel=fused`. If it
  resolves to eager anywhere, I stop and tell the supervisor;
- **compiled graph (guard 2):** on every one of the 20 cases and every field, S's prediction deviation
  d = RMS(pred_compiled − pred_eager) / RMS(target) ≤ 1e-5. Block errors, metric differences and B's reference check are
  reported beside, not gated. (Counting mix-order reduction kernels is not attempted.)
- S's smoke resolved_config.yaml equals B's smoke resolved_config.yaml except `model.n_slices` and `run_id`, and equals
  `runs/r090_vol_isla_seed42/resolved_config.yaml` except `model.n_slices`, `run_id`, `output_dir`, `checkpoint_dir`.

Beside, the cost: S ÷ B wrapper median step and peak allocated memory per rank, same node, same settings.

## 5. Lanes, event guard, scoring, checks

- **Lanes:** `bash submit.sh lanes 0 1` from kernel/slicesdrv, only after section 4 passes. At the first link each lane's
  resolved_config.yaml must equal B's (`runs/r090_vol_isla_seed4x`) except `model.n_slices` and `run_id`; otherwise I stop.
  The supervisor's watcher may move final links to short QoS (recorded in JOBS.md from `watchers/chain_moves.log`). After
  "Training completed" I cancel only that lane's trailing pending links, by explicit id. A hung or failed link goes to the
  supervisor; I never resubmit.
- **Event guard (a stop, not a verdict):** an S lane's val_summary loss above 2x its running minimum after epoch 10 (B's
  worst 1.13), any non-finite loss, a BLOCKED marker, or a lane with no running or pending link.
- **Scoring exactly as B:** kernel/r090/score.sbatch unchanged (a4a24f5e), submitted from kernel/slicesdrv with
  `-J isla.slicesdrv.score`, the excluded nodes, `SCORE_ROWS=$T/kernel/slicesdrv/score_rows.tsv`,
  `EVAL_ROOT=$T/evals_slicesdrv`, `A2R_ROOT=$T/kernel/slicesdrv`. score_rows.tsv = kernel/r090 rows 32-34 → 0-2 and 36-38 →
  4-6 with the run ids and `n_slices` changed: native, area, posed; float32, `infer_split=val`, 200,000 points, final
  checkpoint, code_r090 (the compiled scorer, as B's).
- **Eager re-score (guard 3):** both S final checkpoints, native, with the same scorer and `score_rows_eager.tsv` (rows 0-1:
  the native rows with ` compile=false model.compile_blocks=false` appended to extra, as kernel/r090fix's eager rows),
  `EVAL_ROOT=$T/evals_slicesdrv_eager`. If any case's compiled and eager scores differ by more than 1e-3 relative on any
  field, the read uses eager scores on both arms (B re-scored eager the same way, all protocols), labelled "eager-scored".
- **Checks before reading** each S row: `"epoch": 500` in the config row; 48 `infer_step` rows with the same sample ids as
  B's; SNAPSHOT_USED code_r090; DATASET_USED as B's; the score log's extra holds `model.n_slices=128`. A failed check makes
  the row invalid and the read "incomplete as registered".

## 6. Rules (registered)

**R1, per field, native protocol, the 48 val cases** (SLICES-AF's rule, unchanged). Fields: pressure, velocity, nu_t, all
headline. CM = the scorer's case-mean relative L2 (`infer_summary`, else the mean of the 48 `infer_step` rows).
- `D = ln( mean_seeds CM(S) / mean_seeds CM(B) )`.
- `SE = sqrt((s_S^2 + s_B^2) / 2)`, s = the arm's two-seed sample sd (n - 1) of ln CM; s_B = 0.0156 / 0.0093 / 0.0107
  (B's own seeds: 2.2 / 1.3 / 1.5 % gaps).
- **Helps** if D < ln 0.90 and |D| > 2 SE. **Harms** if D > ln 1.10 and |D| > 2 SE. **Otherwise no change.**

On this dataset the noise term is small: 2 SE is 0.022 / 0.013 / 0.015 if s_S = 0 and 0.031 / 0.019 / 0.021 if s_S = s_B,
so the ln 1.10 bar (0.095) binds on every field unless S's own seed spread exceeds about 7 %. **Harms therefore means S ÷ B
above 1.10** on that field.

**Decision use (registered):** "no harm" (no change or helps) on all three fields makes 128 slices a candidate for the scope
freeze on two datasets (AirfRANS and DrivAerML Volume). A harm on any field keeps 512.

Beside, not verdicts:
- **A sub-bar shift flag:** for each field, whether S ÷ B > 1.03 with both S seeds above both B seeds (or < 0.97 with both
  below). The rule cannot resolve effects under 10 %, while B's seeds sit 1.3-2.2 % apart; this flag tells the supervisor
  of a seed-separated shift the label calls "no change". It changes no label.
- the same D, SE and label on the **posed** protocol (non-degenerate on DrivAerML Volume: the pose changes the drawn
  boundary tokens; the same four runs, so no independent number);
- the **area** protocol: checked to equal native for both arms, then labelled **degenerate**, no claim;
- seed-matched ratios S42 ÷ B42 and S43 ÷ B43; case medians and the fraction of the 48 cases where S beats B (seed means).

Slots: 2 arms x 2 seeds = 4 independent training runs. Three fields and two protocols are readings of those 4 runs.

**Amendment before submission (supervisor, 00:20 EDT 2026-10-24, at approval of md5 9d8c65e1; no job of this lane had
been submitted and nothing of S existed).** The decision use changes; the labels do not. A raised sub-bar flag (S ÷ B above
1.03 on a field with both S seeds above both B seeds) on any of the three fields **also keeps 512 as the reference**; 128
slices then goes to the scope freeze as a cost-against-accuracy trade, not as a free cost cut. A harm label keeps 512 as
registered. The flag in the other direction (below 0.97, seeds separated) is reported and changes nothing. Reason: on this
dataset the ln 1.10 bar binds, so "no change" would admit a resolved loss of up to 10 %, and a no-harm check meant to clear
a cost cut should not. So 128 slices is a scope-freeze candidate on two datasets only if every field reads no harm **and**
no field raises the upward flag.

**R2, descriptive (no verdict): the far field and the wall-distance decades.** One extra native inference per run (B and S,
both seeds) with `+save_predictions=true` (the scorer's command otherwise; `kernel/slicesdrv/farfield.sbatch`, as
SLICES-AF's nearwall.sbatch), then a CPU reduction (`farfield.py`, numpy only, cpu-short) with CAP-AUDIT's definitions
(`harness/residuals.py`): decades (-inf, 1e-4, 1e-3, 1e-2, 0.1, 1, inf) of the signed wall distance `sdf` from the datapipe,
in units of L_ref = 5 m (so 0.5 mm, 5 mm, 5 cm, 0.5 m, 5 m), count weights, e_R = sqrt(sum_R |p - t|^2 / sum_R |t|^2), share
of e^2. Tables, per arm and seed, pooled over the 48 cases' points and, beside, the case mean of per-case band e_R:
- **nu_t beyond 5 m** (band [1, inf) L_ref): e_R and share of the nu_t e^2; and nu_t in every decade;
- **velocity in every decade** (all three components);
- pressure in every decade (beside).

"CAP-AUDIT's corrected held-out instrument" is a single-case training harness (overfit.py, heldout_check.py) and cannot
score these full-data models. What carries over is its rule (15.8, LANE_RULES): each evaluation case uses its own draw
whole, never one draw's points against another draw's boundary. The scorer's dumps satisfy it by construction (each case's
points, sdf, predictions and targets come from one draw, in that draw's frame; geometry is not rescaled, as
`rescale_geometry` is false). Checks: the four runs see the same points per case (coordinate hash); the per-case whole-set
e from the dump reproduces the scorer's `<field>_l2` to 1e-4 relative on pressure and velocity, with the field mapped to
the scorer's space (the recipe's non-dimensionalization; nu_t through the dataset's standardization, mean 4.8e-4, std
9.4e-4). If nu_t does not reproduce, its bands are reported in physical units with the deviation, as SLICES-AF did.

**R3, descriptive: cost.** S ÷ B smoke median step and peak allocated memory (same job, same node), and beside it the median
trainer `train_step` time and peak `mem_gb` over the whole run, and the GPU-h per lane from sacct.

## 7. Predictions (mine)

| id | prediction | p |
|---|---|---|
| P1 | The smoke and certification pass on the first run | 0.85 |
| P2 | R1 pressure: no change (else harms 0.07, helps 0.03) | 0.90 |
| P3 | R1 velocity: no change (else harms 0.08, helps 0.02) | 0.90 |
| P4 | R1 nu_t: no change (else harms 0.10, helps 0.03) | 0.87 |
| P5 | "No harm" on all three fields | 0.82 |
| P6 | The sub-bar flag fires on at least one field (either direction) | 0.30 |
| P7 | S ÷ B native point estimates within [0.95, 1.05] on all three fields | 0.60 |
| P8 | R2: S's pooled nu_t e_R beyond 5 m within 10 % of B's on both seeds (seed-matched) | 0.55 |
| P9 | R3: S ÷ B smoke median step in [0.55, 0.80] | 0.60 |
| P10 | R3: S ÷ B smoke peak allocated memory in [0.80, 0.95] | 0.55 |
| P11 | Both S lanes finish with no event-guard stop | 0.88 |

Why: on AirfRANS, quartering the slices moved nothing beyond 3 %; on DrivAerML Volume's single case it plateaued on all three
fields, and the far-field nu_t level did not move with any rung (CAP-AUDIT 15.10), which points to a missing input (the
floor) rather than slice resolution. Far points still own slices at 512; with 128 they may lose some, which is the main
route to a harm, hence nu_t's larger harm probability. Cost: DrivAerML Volume runs the fused geometry kernel, which never
materializes the N x S tensors, so the saving is smaller than AirfRANS's eager-region 0.42; the slice projection and
de-slicing (about 2 N S C flops per block) still shrink fourfold against an MLP of about 8 N C^2.

## 8. Cost, cap and schedule

- S lanes: B took about 40.5 GPU-h per lane (54,000 steps at 0.593 s, about 8.9 h of steps plus 1.2 h of validation and
  restarts). At 0.55-0.80 of the step, **about 25-35 GPU-h per lane, 50-70 for two** (2-3 links each).
- Smoke and certification: one short-QoS job, at most 4 GPU-h (01:00:00 x 4). Scoring (compiled, 6 rows) and the eager
  re-score (2 rows): about 1-2 GPU-h. Far-field inference (4 runs, one short job): at most 2 GPU-h; cpu-short reduction.
- **Total about 60-80 GPU-h; cap 120 GPU-h** (the supervisor's). If failures or reruns project past 120, I ask first.
- Wall: S lanes about 7-9 h of training each. LHR is busy (start estimates near 06:00 EDT 10-24), so training ends about
  10-24 16:00 to 10-25 04:00 EDT, scored within a few hours of that. Harness deadline 10-26 06:07 EDT; HANDOFF_SLICESDRV.md
  at 03:07 EDT 10-26 if S is still unread.
- Registry row `^isla\.slicesdrv\.` (expected_end 2026-10-25) added at the first job (the smoke), removed after the last.
  Every job is named `isla.slicesdrv.*`, with the five bad nodes excluded.
