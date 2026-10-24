# SLICES-AF: ISLA 0.9.0 at 128 slices on full AirfRANS (pre-registration, 2026-10-23)

Lane SLICES-AF of ISLA (research-lane, launched 2026-10-23 about 12:00 EDT by the supervisor 01a13c2c). Written 12:10 EDT
2026-10-23, **before any job of this lane: nothing has been submitted, and no S checkpoint, log or score exists.**

## 1. Question

Does ISLA 0.9.0 with `n_slices=128`, in place of the reference 512, predict AirfRANS better on the full dataset
(full_train → full_test, 200 cases)? Pressure decides. Velocity and nu_t must not be harmed. What does it cost per step?

The lead: CAP-AUDIT's single-case ladder (`../isla-cap-audit-2026-10-20/RESULTS.md` 15.2 / 15.11, HANDOFF.md decision 1).
There s128 had a 1.27x lower pressure floor (gm 0.00157 against 0.00198) and a lower near-wall band (1e-3 to 0.1 chords:
e_R 0.0017-0.0018 against 0.0024-0.0031 on every other rung). Both are inside that case's run-to-run spread (1.43x over
seven reference runs), and single-case floors sit about 9x below ISLA's cross-case test error, so only a full-data pair
answers whether the lead matters.

## 2. What I have seen before registering (disclosure)

- **B's scores, all of them** (`../isla-ref090-2026-10-20/eag_af_isla.md`, `scores/evals_r090eag/`), epoch 500, 200 cases,
  float32:

  | field | seed 42 | seed 43 | mean | s_B (sd of ln, n-1) |
  |---|---|---|---|---|
  | pressure | 0.0131631 | 0.0117940 | 0.0124786 | 0.0777 |
  | velocity (in-plane) | 0.0064407 | 0.0062701 | 0.0063554 | 0.0190 |
  | nu_t | 0.0157382 | 0.0149675 | 0.0153529 | 0.0355 |

  posed ÷ native: pressure 1.0022 / 1.0025, velocity 1.0016 / 1.0016, nu_t 1.0000 / 1.0000. **area equals native
  exactly on every field and seed** (the airfoil is kept whole, so the area draw is the identity: degenerate).
  The same seed split on pressure holds on all three AirfRANS ISLA 0.9.0 training paths: s42 ÷ s43 = 1.116 (eager),
  1.133 (fix), 1.125 (bug).
- B's training logs: validation-loss maximum over running minimum 3.05 (s42, epoch 80) and 1.95 (s43, epoch 90); median
  train step 0.754 / 0.756 s over all steps (trainer's `train_step` rows); peak `mem_gb` 45.4; 7 links per lane,
  about 89 GPU-h per lane (sacct elapsed x 4 GPUs); the 2026-10-21 smoke 323028: wrapper median 0.766 s, peak allocated
  43.0 GiB.
- CAP-AUDIT's results and handoff in full (phase 1, 1b, 2, 15.9-15.11, drv ladder: s128 plateau on DrivAerML Volume,
  1.03 / 0.99 / 1.04 per halving on velocity / nu_t / pressure).
- The book at 960b83731 (chapters 12, 19, 20 and the ledger), REF-090's final tables and LRE-AZ's pre-registration.
- Nothing of S exists.

## 3. Arms

`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0` (LHR, GB300).

- **B (reused, not retrained):** `r090eag_af_isla_seed42` / `_seed43`, rows 0-1 of `$T/kernel/r090eagaf/lanes.tsv`
  (md5 81937645): code `$T/code_cmpaf_fix`, recipe `$T/recipe_r090`, `model=isla_volume_airfrans`,
  `dataset=airfrans_uvolume_2d`, extra `train_split=full_train val_split=full_test +training.val_interval=10
  model.hidden=512 model.n_slices=512 model.compile_blocks=true compile=false model.compile_blocks=false` (eager), lr
  3.75e-4, 500 epochs, 100,000 cells, bf16, seeds 42 / 43.
- **S:** B with one change, `model.n_slices=128`. Launcher `$T/kernel/slices/`:
  - `lanes.sbatch` and `ckpt_preflight.py`: byte copies of kernel/r090eagaf's (cmp, md5 8eecec8f / e34bd611);
  - `submit.sh`: r090eagaf's with only the job-name prefix `isla.slices.` (and its squeue grep and header line);
  - `site.env`: r090eagaf's (6a5aca06) with `LANE_SBATCH_EXTRA="--segment=1 --exclude=$T/kernel/slices/exclude_nodes.txt"`
    and a header line; `exclude_nodes.txt` is a copy of kernel/lreaz's (nvl72d015-T12, nvl72d058-T08, nvl72d052-T01,
    nvl72d057-T13), a file because a comma list would break the chain-ahead `--export`;
  - `lanes.tsv`: r090eagaf rows 0-1 with group `slices`, run ids `slices_af_isla_s128_seed42` / `_seed43`, and
    `model.n_slices=512` → `model.n_slices=128`. Nothing else differs (`diff` recorded in JOBS.md). Job names
    `isla.slices.af-isla-s128-seed4x`; normal QoS, 03:59:00 chained singleton links.
  - MAX_SLICES is 1024 and AirfRANS is 2-D, so both arms run the eager geometry region; the change touches no code path
    choice.

## 4. Smoke, training, scoring, checks

- **Smoke** (one short-QoS 4-GPU job, 00:30:00 limit, nodes excluded as above): kernel/r090eagaf's smoke.sbatch and
  smoke_wrap.py (md5 704f8188 / 32a2991e), copied to `kernel/slices/smoke/` with only its paths and job name changed
  (`isla.slices.smoke`). It runs two units on the same node, one after the other, each 50 training steps through the lane
  launcher with NO_CHAIN: **B** (`kernel/slicesb512`, byte copies of r090eagaf's lanes.sbatch, site.env, lanes.tsv and
  ckpt_preflight.py, so B's unit runs B's files without writing into r090eagaf) and **S** (`kernel/slices` row 0).
  Passes if:
  - both units rc 0, no traceback, `nonfinite=[]`;
  - S's smoke resolved_config.yaml equals B's smoke resolved_config.yaml except `model.n_slices` and `run_id`;
  - and equals `runs/r090eag_af_isla_seed42/resolved_config.yaml` except `model.n_slices`, `run_id` and the two smoke
    paths (`output_dir`, `checkpoint_dir`).

  Lanes are submitted only if it passes. Beside, the cost: S ÷ B wrapper median step and peak allocated memory per rank,
  same node, same settings.
- **Lanes:** `bash submit.sh lanes 0 1`. At the first link, the lane's own resolved_config.yaml must equal B's
  (`runs/r090eag_af_isla_seed4x`) except `model.n_slices` and `run_id`; otherwise I stop and report. Final links may be
  moved to short QoS by the supervisor's watcher (recorded in JOBS.md from `watchers/chain_moves.log`). I never cancel or
  resubmit a link; a hung or failed link goes to the supervisor.
- **Event guard:** if an S lane's val_summary loss passes 4x its running minimum after epoch 10 (B's worst: 3.05), any
  loss is non-finite, or a BLOCKED marker appears, I stop and report. This is a stop, not a verdict.
- **Scoring exactly as B:** kernel/r090eagaf/score.sbatch unchanged (md5 bd40af29), submitted from kernel/slices with
  `-J isla.slices.score`, the excluded nodes, and `SCORE_ROWS=$T/kernel/slices/score_rows.tsv`,
  `EVAL_ROOT=$T/evals_slices`. score_rows.tsv is r090eagaf's rows 0-2 and 4-6 with the run ids and `n_slices` changed:
  native, area, posed; float32, `infer_split=val` (= full_test), 100,000 points, final (epoch-500) checkpoint.
- **Checks before reading** each S row: `"epoch": 500` in the config row, 200 `infer_step` rows with the same sample ids
  as B's, SNAPSHOT_USED `code_cmpaf_fix`, DATASET_USED as B's, the score log's extra holds `model.n_slices=128` (the
  checkpoint would not load otherwise). A failed check makes the row invalid and the read incomplete as registered.

## 5. Rules (registered)

**R1, per field, native protocol, the 200 full_test cases.** Fields: pressure (headline), velocity (in-plane; never
velocity_z), nu_t. CM is the scorer's case-mean relative L2 (`infer_summary`, else the mean of the 200 `infer_step` rows).
- `D = ln( mean_seeds CM(S) / mean_seeds CM(B) )`.
- `SE = sqrt((s_S^2 + s_B^2) / 2)`, s = the arm's two-seed sample sd (n - 1) of ln CM.
- **Helps** if D < ln 0.90 and |D| > 2 SE. **Harms** if D > ln 1.10 and |D| > 2 SE. **Otherwise no change.**

This is LRE-AZ's rule unchanged. On AirfRANS the noise term binds on pressure: s_B = 0.0777, so 2 SE >= 0.110 (if s_S
were 0) and = 0.155 if s_S = s_B. **Helps on pressure therefore needs S ÷ B below about 0.90-0.86**, depending on S's own
spread (the single-case lead was 0.79). On velocity (2 SE >= 0.027) and nu_t (2 SE >= 0.050) the ln 1.10 / ln 0.90 bars
bind unless S's spread is much larger than B's. I keep the program's bar rather than a narrower one: the seed split is
the same on three training paths, which says it is a real seed effect of about 12 %, not an outlier of one pair.

Beside, not verdicts:
- the same D, SE and label on the **posed** protocol (non-degenerate: it rotates and translates the input; it is the same
  runs, so it adds no independent number);
- the **area** protocol: checked to equal native for both arms and then labelled **degenerate**, no claim;
- seed-matched ratios S42 ÷ B42 and S43 ÷ B43 (S's initialization differs from B's, since the routing weights change
  shape, so I do not assume the pairing cancels the seed effect);
- case medians and the fraction of the 200 cases where S beats B (seed means per case).

Slots: 2 arms x 2 seeds = 4 independent training runs. Three fields and two protocols are readings of those 4 runs.

**R2, descriptive, the near-wall band (no verdict).** For both arms and seeds: pressure e_R by wall-distance band, with
CAP-AUDIT's definitions (`harness/residuals.py`: decades (-inf, 1e-4, 1e-3, 1e-2, 0.1, 1, inf) of the stored signed
wall distance in chords, count weights, e_R = sqrt(sum_R |p - t|^2 / sum_R |t|^2), share of e^2), plus the combined
band [1e-3, 0.1). Pooled over all points of the 200 cases and, beside, the case mean of per-case band e_R. From one extra
native inference per run with `+save_predictions=true` (the scorer's command otherwise; `kernel/slices/nearwall.sbatch`)
and a CPU reduction (`nearwall.py`, an adapter of residuals.py to the saved `.pdmsh`). Checks: the four runs see the
same points per case (coordinate hash), and the per-case pressure e from the dump reproduces the scorer's
`pressure_l2` to 1e-4 relative.

**R3, descriptive, cost.** S ÷ B smoke median step and peak allocated memory (same job and node), and beside it the
median `train_step` time over the whole run and the GPU-h per lane from sacct.

## 6. Stage 2 (decision rule)

- **Stage 2 is proposed only if** R1 pressure reads **helps** and **no field reads harms** (native or posed). Then I
  propose a second dataset, chosen by noise and cost, to the supervisor, and launch nothing without approval. My
  candidate now, before results: **DrivAerML Volume** (the other interior task with a boundary layer; REF-090 two-seed
  spreads 1.3-2.2 %; about 46 GPU-h per lane at 512 slices, so an s128 pair is about 70-90 GPU-h against the existing
  `r090_vol_isla` pair). Caveat already known: CAP-AUDIT's single-case drv s128 rung plateaued. DrivAerML Surface is
  cheaper and quieter but has no near-wall interior band.
- **No change on pressure:** 128 slices is not adopted for accuracy. If S is cheaper and harms no field, I report the
  cost alone as an input for the scope freeze (a speed argument, Peter's third priority), with no stage 2.
  A cheaper S that reads no change counts as a cost result on one dataset only; adopting it at the scope freeze would
  also need a no-harm check on a second dataset, which the supervisor decides and which is not registered here.
  (Added at the supervisor's approval, 2026-10-23, before any job.)
- **Harms on any field:** no stage 2; the 512 reference stands.

## 7. Predictions (mine)

| id | prediction | p |
|---|---|---|
| P1 | The smoke passes on its first run | 0.90 |
| P2 | R1 pressure: helps / no change / harms | 0.20 / 0.70 / 0.10 |
| P3 | R1 velocity: no change (else harms 0.10, helps 0.05) | 0.85 |
| P4 | R1 nu_t: no change (else harms 0.12, helps 0.05) | 0.83 |
| P5 | S ÷ B pressure point estimate below 1 | 0.55 |
| P6 | R2: S's pooled [1e-3, 0.1) pressure e_R below B's on both seeds | 0.40 |
| P7 | R3: S ÷ B smoke median step in [0.60, 0.85] | 0.65 |
| P8 | R3: S ÷ B peak allocated memory below 0.80 | 0.60 |
| P9 | Both S lanes finish with no event-guard stop | 0.85 |
| P10 | Stage 2 is proposed | 0.17 |

Why: the single-case lead is within that case's spread, the floors it moved are 9x below the cross-case error that R1
reads, and quartering the slices did nothing on DrivAerML Volume's single case. A small gain in the near-wall band, if
real, is diluted by the generalization error. Quartering the slices shrinks every N x M routing and geometry tensor
fourfold; AirfRANS runs the eager (unfused) geometry region, so those tensors are a large share of the step.

## 8. Cost, cap and schedule

- S lanes: B used about 89 GPU-h per lane; at 0.65-0.8x the step, **about 58-71 GPU-h per lane, 116-143 for two**.
  (CAP-AUDIT's 150-180 assumed 110-120 per lane.)
- Smoke at most 2 GPU-h (00:30:00 x 4). Scoring about 1 GPU-h (B's two tasks took 5.4 min each). Near-wall inference
  (4 runs, one short job) at most 2 GPU-h, plus cpu-short for the reduction.
- **Total about 120-150 GPU-h, cap 200.** If failures or reruns project past 200, I ask first.
- Wall: about 14-17 h of training (100,000 steps at 0.5-0.6 s), 4-5 links, so 18-22 h with queue gaps. If the lanes
  start by about 14:00 EDT 10-23, S is scored about 10-24 08:00-14:00 EDT. Harness deadline 10-24 22:00 EDT;
  HANDOFF_SLICES.md at 20:00 EDT if S is still unread.
- Registry row `^isla\.slices\.` (expected_end 2026-10-25) is added at the first job (the smoke) and removed after the
  last.
