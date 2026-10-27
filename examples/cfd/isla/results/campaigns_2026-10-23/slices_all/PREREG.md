# SLICES-ALL: 128 slices on every ISLA form, the no-harm check for the scope freeze (pre-registration, 2026-10-24)

Lane SLICES-ALL of ISLA (research-lane, launched 2026-10-24 about 12:35 EDT by the supervisor 01a13c2c). Written 12:40-13:05
EDT 2026-10-24, **before any job of this lane: nothing has been submitted, and no checkpoint, log or score of an arm exists.**
The launcher `$T/kernel/slicesall` and two byte-copy control launchers are built (file operations on the LHR login node
only; `cluster/setup.sh`, output `cluster/setup.out`). `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`,
`K=$T/kernel/slicesall`.

## 1. Question

SLICES-AF (AirfRANS, `../isla-slices-2026-10-23/RESULTS.md`, md5 53b1cdc2) and SLICES-DRV (DrivAerML Volume,
`../isla-slices-drv-2026-10-24/RESULTS.md`, md5 b2fd11e2) found that 128 slices cost **interacting ISLA 0.9.0** nothing
against 512 (every field "no change", no flag), at 0.42 and 0.63 of the step. Since 2026-10-24 ISLA-DTT is ISLA's default
interior form, and the surface task uses the interacting form (option A). Neither form has been tested at 128 slices, and
no solid-mechanics dataset has. **Does 128 slices harm ISLA-DTT on AirfRANS, ISLA-DTT on the lug (solid mechanics), or the
interacting surface form on DrivAerML Surface?** This is the check priced in FREEZE-PLAN's FINAL_RERUN_PLAN.md section 8a
(md5 33026a22) and FREEZE_DECISIONS.md section 8 (md5 d6eb3327). With 128 slices the final rerun would cost about 2,560
GPU-h instead of about 3,380.

**What the surface arm tests.** On 0.9.0 code (code_r090) the DrivAerML Surface task reads the vehicle only: option A's
reading of every named boundary is not in 0.9.0. So the third arm tests the slice count of the interacting surface form
on the vehicle boundary, not option A as the rerun will run it. Likewise every arm runs 0.9.0 code, not the rerun's code
(as SLICES-DRV did).

## 2. What I have seen before registering (disclosure)

- The controls' scores in full: REF-090's `r090_final_tables.md`, `r090_final.json`, `lug_dtt_read.md`, `eag_af_dtt.md`
  and the DrivAerML Surface score trees (`../isla-ref090-2026-10-20/scores/evals_r090/r090_dr_isla_seed4x/`), quoted in
  section 3.
- The controls' training logs (`$T/runs/<control>/metrics.jsonl`): median trainer step, peak `mem_gb` and worst
  validation loss over the running minimum after epoch 10, quoted in section 3; the `geo_kernel` lines of their train.log.
- SLICES-AF and SLICES-DRV in full (PREREG, RESULTS, JOBS); FREEZE-PLAN section 8a and FREEZE_DECISIONS section 8;
  DTT-POSE-EXP's PREREG and JOBS (its controls); REF-090's RESULTS and JOBS; COMPILE-AF's diag row tables.
- Nothing of any arm exists.

## 3. Arms and controls

Every arm is its control's lane row with exactly two changes, `run_id` (and the group / index columns of the row table)
and `model.n_slices=512 -> 128`: the same code snapshot, recipe snapshot (`$T/recipe_r090`), launcher settings
(lanes.sbatch 8eecec8f, ckpt_preflight.py e34bd611, site.env = the control's apart from the excluded nodes), compile path,
epochs (500), points, learning rate (3.75e-4), precision (bf16 via LANE_PRECISION), seeds 42 and 43. The field-wise diff
of every row against its control row is in `cluster/setup.out` and shows only those fields. At the first link each lane's
`resolved_config.yaml` must equal its control's except `run_id` and `model.n_slices` (asserted; otherwise I stop that arm).

| arm | lanes (`$K/lanes.tsv`) | control | control's row | code | compile path | scored |
|---|---|---|---|---|---|---|
| **DR**: interacting surface form, DrivAerML Surface (vehicle only) | 0-1 `slicesall_dr_isla_s128_seed42/43` | `r090_dr_isla` (the standing) | `$T/kernel/r090/lanes.tsv` rows 6-7 (md5 81d34e1d) | code_r090 | per-block compile (`model.compile_blocks=true`, launcher `compile=false`) | kernel/r090/score.sbatch (a4a24f5e), compiled, native / area / posed, 48 val cases, 200,000 cells |
| **LUG**: ISLA-DTT, lug | 2-3 `slicesall_lug_dtt_s128_seed42/43` | `r090fix_lug_dtt` (the lug ISLA-DTT reference, labelled fix-trained) | `$T/kernel/r090fix/lanes.tsv` rows 0-1 (ce24cf80) | code_cmpaf_fix | per-block compile on the fixed path | kernel/r090fix/score.sbatch (3a5e4ca9), eager, native / area / posed, 600 test cases, 200,000 cells (trained at 20,000) |
| **AF**: ISLA-DTT, AirfRANS | 4-5, **held** (section 3a) | named: `r090fix_af_dtt`; standing: `r090eag_af_dtt` | r090fix rows 4-5 / r090eagaf rows 2-3 | code_cmpaf_fix | fix-trained compiled / eager | eager, native / posed, 200 full_test cases, 100,000 cells |

Launcher `$K`: lanes.sbatch and ckpt_preflight.py byte copies of kernel/r090's (= kernel/r090fix's, cmp); site.env =
kernel/r090's (4a6ed5fb, which equals kernel/r090fix's apart from its header) with
`LANE_SBATCH_EXTRA="--segment=1 --exclude=$K/exclude_nodes.txt"` (the five bad nodes nvl72d015-T12, nvl72d058-T08,
nvl72d052-T01, nvl72d057-T13, nvl72d043-T18); submit.sh = kernel/r090's (7f559159) with only the prefix `isla.slicesall.`.
Job names `isla.slicesall.<slug>` (lanes), `.smoke`, `.score`. Normal QoS, 03:59:00 chained singleton links; short QoS
only for final links, moved by the supervisor's watcher. `$T/kernel/slicesall_b090` and `slicesall_bfix`: byte copies of
kernel/r090's and kernel/r090fix's lanes.sbatch, site.env, lanes.tsv and ckpt_preflight.py, used only by the smokes'
control units.

**The controls (native case-mean relative L2, the scorer's `infer_summary`, float32, epoch 500).**

| control | field | seed 42 | seed 43 | s_c (sd of ln, n-1) | other protocols | trainer step / peak mem / worst val ÷ min |
|---|---|---|---|---|---|---|
| r090_dr_isla | pressure | 0.0411604 | 0.0411084 | 0.00089 | posed ÷ native 1.0000 / 1.0000; area 1.0016 / 1.0070 | 0.241 / 0.245 s; 66.8 GB; 1.10 / 1.10 |
| | wall shear (`wss_l2`) | 0.0661942 | 0.0662482 | 0.00058 | posed 1.0000 / 1.0001; area 1.0124 / 1.0157 | |
| r090fix_lug_dtt | von Mises | 0.00221933 | 0.00183431 | 0.1347 | posed ÷ native 1.528 / 1.977; area degenerate | 0.425 / 0.420 s; 10.9 GB; 1.43 / 2.00 |
| r090fix_af_dtt | pressure | 0.0127425 | 0.0125662 | 0.0099 | posed 1.0016 / 1.0023 | 0.566 / 0.575 s; 41.1 GB; 1.57 / 1.92 |
| | velocity (in-plane) | 0.00710411 | 0.00714819 | 0.0044 | posed 1.0011 / 1.0013 | |
| | nu_t | 0.0198947 | 0.0230940 | 0.1054 | posed 1.0003 / 1.0001 | |
| r090eag_af_dtt | pressure | 0.0124280 | 0.0123887 | 0.0022 | posed 1.0017 / 1.0022 | 0.931 / 0.937 s; 50.1 GB; 1.34 / 1.66 |
| | velocity (in-plane) | 0.00737341 | 0.00697252 | 0.0395 | posed 1.0010 / 1.0012 | |
| | nu_t | 0.0207112 | 0.0204842 | 0.0078 | posed 1.0001 / 1.0001 | |

### 3a. The AirfRANS control is held (brief: "if a control is not what I named, stop that arm and tell me")

The brief names `r090fix_af_dtt` and says DTT-POSE-EXP uses the same AirfRANS control. It does not: DTT-POSE-EXP (and
DTT-KNOBS stage 2) use `r090eag_af_dtt`, the eager-trained pair that REF-090 made AirfRANS's ISLA-DTT reference (REF-090
RESULTS section 4, "Secondary arms ... AirfRANS ISLA-DTT -> r090eag_af_dtt"; book chapters 12 and 14 cite it).
`r090fix_af_dtt` exists with full scores on the same code (code_cmpaf_fix), fix-trained on the compiled path; it is
ATTRIB-090's B, not the standing. REF-090 §14 read the two training paths of ISLA-DTT as different on pressure (eager ÷
fix 0.981 at 1.37 x 2 SE). So the AF arm is not launched until the supervisor chooses. Either choice is registered here,
with the same rule:
- **AF-eag** (my recommendation): lanes 4-5 = `$T/kernel/r090eagaf/lanes.tsv` rows 2-3 with the two changes, eager-trained
  (`compile=false model.compile_blocks=false` in extra, as the control). Its seeds agree (s_c 0.0022 / 0.0395 / 0.0078),
  so the ln 1.10 bar binds on pressure and nu_t and the flag can resolve a 3 % shift; it is the standing and the control
  of the two other ISLA-DTT experiments. Its smoke has no compiled region, so no block-wise check.
- **AF-fix** (as named): lanes 4-5 = r090fix rows 4-5 with the two changes, compiled on the fixed path, with the same
  block-wise check as LUG. Cheaper and on the rerun's compile path, but its nu_t seeds sit 15 % apart (s_c 0.105), so a
  nu_t harm label would need S ÷ B above about 1.16 and the up flag needs both S seeds above 0.0231.
The AF launch, its smoke and its row diff are recorded in JOBS.md with the supervisor's answer, before any AF job.

## 4. Smokes (one short-QoS job per arm, before that arm's lanes; `isla.slicesall.smoke`, 01:30:00, bad nodes excluded)

`$K/smoke.sbatch` = SLICES-DRV's smoke.sbatch (ccb413d3) with paths, job name and time limit changed and the arm made a
parameter (`cluster/smoke.sbatch`; diff in JOBS.md). In order, on one node:
1. **Control unit** (`slicesall_b090:6` for DR, `slicesall_bfix:0` for LUG, the AF control row for AF) and **arm unit**
   (`slicesall:0`, `slicesall:2`, `slicesall:4`): 50 training steps each through the lane launcher with NO_CHAIN and
   smoke_wrap_r090.py (md5 32a2991e, byte copy), one after the other.
2. **Certification training:** the arm's lane command for 2 epochs (`<smoke>/lanes_cert.tsv`, seed 42, epochs 2; the
   lane's save_interval is 2, so this writes the first checkpoint).
3. **Block-wise compiled-against-eager check** (compiled arms DR, LUG, and AF-fix): COMPILE-AF's `cmpaf_diag.py` (byte
   copy, 9fc35d83) `--blockwise --cases 20` on that checkpoint, the first 20 cases of the scorer's loader (native,
   float32, the scorer's points), the model's own compiled blocks against the eager model. Beside it, the same on the
   control's final checkpoint (seed 42).

**An arm's smoke passes if** (every item; otherwise that arm is not submitted and I report):
- both units rc 0, no Traceback, `nonfinite=[]`; the certification training reaches "Training completed";
- **resolved config:** the arm unit's resolved_config.yaml equals the control unit's except `run_id` and `model.n_slices`,
  and equals the control's own `runs/<control>_seed42/resolved_config.yaml` except `run_id`, `output_dir`,
  `checkpoint_dir` and `model.n_slices`; the certification config equals the arm unit's except `run_id` and `num_epochs`;
- **compile path:** the resolved config shows `compile: false` (launcher) and `model.compile_blocks` equal to the
  control's (true for DR, LUG, AF-fix; false for AF-eag); for compiled arms the diag runs the model's compiled
  `_block_callables`; Dynamo recompiles and graph breaks in the arm unit equal the control unit's;
- **geometry kernel:** every rank of the arm unit and the certification logs `ISLA geo_kernel=auto resolved to` the same
  kernel as the control's train.log ('fused' for DR, 'eager' for LUG and AF), with slices=128;
- **compiled graph (compiled arms):** on every one of the 20 cases and every field, d = RMS(pred_compiled -
  pred_eager) / RMS(target) <= 1e-5 (SLICES-DRV's bar). Block errors, metric differences and the control checkpoint's
  check are reported beside, not gated. (The program's float64 certification of compiled regions applies to new compiled
  code; these are the same compiled regions as the controls', certified by COMPILE-AF, at another slice count.)

Beside: arm ÷ control median step and peak allocated memory per rank (same node, same settings).

## 5. Lanes, event guard, scoring, checks

- **Lanes:** `bash submit.sh lanes <first> <last>` from `$K`, each arm right after its smoke passes, the LUG lanes
  first. Up to 6 lanes. A hung or failed link goes to the supervisor; I resubmit only a duplicate-free restart after a
  node failure as LANE_RULES say (keep the oldest pending link). After "Training completed" I cancel only that lane's
  trailing pending links, by explicit id.
- **Event guard (a stop, not a verdict):** an arm lane's validation loss above its running minimum by more than **2x**
  after epoch 10 for DR (control worst 1.10), **4x** for LUG and AF (controls' worst 2.00 and 1.92; SLICES-AF and
  DTT-POSE-EXP's guard); any non-finite loss; a BLOCKED marker; a lane with no running or pending link.
- **Scoring exactly as each control:**
  - DR: kernel/r090/score.sbatch unchanged, `-J isla.slicesall.score`, excluded nodes,
    `SCORE_ROWS=$K/score_rows_dr.tsv` (r090 rows 24-26, 28-30 with the run ids and n_slices changed),
    `EVAL_ROOT=$T/evals_slicesall`, `A2R_ROOT=$K`, array 0-1. **Eager re-score (guard):** `score_rows_dr_eager.tsv`
    (native rows + ` compile=false model.compile_blocks=false`) into `$T/evals_slicesall_eager`; if any case's compiled
    and eager scores differ by more than 1e-3 relative on any field, the DR read uses eager scores on both arms (the
    control re-scored eager the same way, all protocols), labelled "eager-scored" (SLICES-DRV's guard 3).
  - LUG: kernel/r090fix/score.sbatch unchanged, the same overrides, `SCORE_ROWS=$K/score_rows_lug.tsv` (r090fix rows
    0-2, 4-6 with the run ids and n_slices changed; eager, as the control's).
  - AF: the chosen control's scorer and rows with the same two changes (recorded before the AF launch).
- **Checks before reading** each arm row: `"epoch": 500` in its config row; 48 / 600 / 200 `infer_step` rows with the
  control's sample ids; SNAPSHOT_USED the control's code; DATASET_USED as the control's per protocol; the score log's
  extra holds `model.n_slices=128`. A failed check makes the row invalid and that arm "incomplete as registered".

## 6. Rules (registered)

**R1 per arm, per field, native protocol** (SLICES-DRV's section 6 rule, unchanged). Fields: DR pressure and wall shear
(`wss_l2`); LUG von Mises (`von_mises_l2`); AF pressure, in-plane velocity (`velocity_l2`; never velocity_z) and nu_t
(`nut_l2`). CM = the scorer's case-mean relative L2 (`infer_summary`, else the mean of the `infer_step` rows).
- `D = ln( mean_seeds CM(arm) / mean_seeds CM(control) )`;
- `SE = sqrt((s_arm^2 + s_c^2) / 2)`, s = the pair's own two-seed sample sd (n - 1) of ln CM; s_c from each control's
  own two seeds (section 3 table);
- **helps** if D < ln 0.90 and |D| > 2 SE; **harms** if D > ln 1.10 and |D| > 2 SE; **otherwise no change**;
- **sub-bar flag:** up if arm ÷ control > 1.03 with both arm seeds above both control seeds; down if < 0.97 with both
  below. The flag changes no label.

**Resolution, stated now.** DR: s_c 0.0009 / 0.0006, so the ln 1.10 bar binds unless the arm's seeds spread more than
about 13 %; "harms" means arm ÷ control above 1.10, and the up flag resolves a seed-separated shift above 3 %. LUG: s_c
0.1347, so 2 SE binds: a harm label needs arm ÷ control above 1.21 (if the arm's seeds agree exactly) to 1.31 (if they
spread like the control's); the control's seeds are 21 % apart, so the up flag needs both arm seeds above 0.002219. AF:
see 3a.

**The verdict (the supervisor's design, brief items 2-3).** 128 slices is a **candidate for every form** only if every arm
reads **no change** on every field and **no up flag** fires on any field (SLICES-DRV's §6 rule as amended at 00:20 EDT
10-24: an upward sub-bar flag keeps 512). **Otherwise 512 stays.** I read the brief literally: a "helps" label is not "no
change", so it too keeps 512 under this rule and is reported as a gain for the freeze to weigh; a down flag is reported
and, as in SLICES-DRV's amendment, changes nothing. A harm label keeps 512. An arm lost or invalid makes the verdict
"incomplete as registered", escalated; no verdict is improvised. The adoption is the supervisor's decision at the scope
freeze, reported to Peter; this registration states the rule, not the adoption.

**Beside, not verdicts:**
- the same D, SE and label on the **posed** protocol (it changes every arm's input: DrivAerML Surface posed rotates and
  translates the vehicle; the lug's `ginot_lug_volume_posed` applies an SO(3) pose and a translation; AirfRANS posed
  likewise; same runs, so no independent number). On the lug posed is not a no-harm reading of pose invariance: 0.9.0's
  ISLA-DTT forest is pose-dependent (DTT-POSE), and the control's posed ÷ native is 1.53 / 1.98;
- **area**: DR (non-degenerate: area ÷ native 1.0016-1.0157 for the control), D, SE and label beside; LUG and AF: checked
  to equal native, then labelled **degenerate**, no claim;
- **seed-matched ratios** arm42 ÷ control42 and arm43 ÷ control43 on every field (for the lug as DTT-POSE-EXP reports
  them). The rule stays unpaired: a different `n_slices` changes the parameter shapes, so the arms do not share an
  initialization with their controls;
- case medians of arm ÷ control and the share of cases where the arm beats the control (seed means);
- **cost:** arm ÷ control smoke median step and peak allocated memory (same job, same node), the whole-run trainer median
  `train_step` and peak `mem_gb`, and GPU-h per lane from sacct.

Slots: 3 arms x 2 arms (arm, control) x 2 seeds = 12 independent training runs, 6 of them new. Fields and protocols are
readings of those runs: DR 2 fields, LUG 1, AF 3.

## 7. Predictions (mine, before any result)

| id | prediction | p |
|---|---|---|
| P1 | DR and LUG smokes pass on the first run | 0.80 |
| P2 | DR: no change on both fields (else harms 0.05, helps 0.02) | 0.93 |
| P3 | DR: an up flag on at least one field | 0.20 |
| P4 | LUG: no change on von Mises (else harms 0.05, helps 0.03) | 0.92 |
| P5 | LUG: an up flag | 0.15 |
| P6 | AF (either control): no change on all three fields | 0.85 |
| P7 | AF: an up flag on at least one field | 0.20 |
| P8 | The verdict: candidate for every form (all no change, no up flag) | 0.50 |
| P9 | DR arm ÷ control native point estimates within [0.97, 1.03] on both fields | 0.70 |
| P10 | DR smoke median step ratio in [0.50, 0.75] | 0.60 |
| P11 | LUG smoke median step ratio in [0.80, 0.97] | 0.60 |
| P12 | AF smoke median step ratio in [0.40, 0.70] | 0.60 |
| P13 | Every lane finishes with no event-guard stop | 0.85 |
| P14 | LUG: posed ÷ native of the arm in [1.3, 2.2] on both seeds (the forest's pose defect stays) | 0.80 |

Why: on two interior datasets quartering the slices moved interacting ISLA by at most 3 %. DrivAerML Surface predicts on
the vehicle surface from vehicle tokens, so its slices partition the same surface it predicts on, with 200,000 cells per
case; the fused kernel never materializes the N x S tensors, so the saving resembles DrivAerML Volume's. ISLA-DTT's
slices only build the boundary's token state; the prediction points read it through the tree read and do not interact,
so the slice count acts on the boundary representation alone; its blocks are a smaller share of the step (lug 26 %,
PROFILE-2), hence the small lug saving. The lug's tiny memory (10.9 GB) and 21 % seed spread mean it can show only a
large harm. The joint verdict needs six fields and three flag tests to come out clean, hence P8's 0.50 despite each
arm's high no-harm probability.

## 8. Cost, cap and schedule

| item | basis | GPU-h |
|---|---|---|
| DR pair | control 16.4 GPU-h per lane (54,000 steps at 0.243 s), x 0.55-0.70 | 18-24 |
| LUG pair | control 300,000 steps at 0.42 s, about 140-160 GPU-h per lane (sacct 171 / 153 with losses), x 0.85-0.95 | 260-300 |
| AF pair, AF-eag | control 110-111 GPU-h per lane (100,000 steps at 0.93 s), x 0.50-0.60 | 110-135 |
| AF pair, AF-fix | control 66.5 GPU-h per lane (0.57 s), x 0.6-0.8 | 80-106 |
| smokes (3 short jobs) | | about 8 |
| scoring (compiled, eager guard, 3 arms) | | about 8 |
| **total** | | **about 395-475 (AF-eag) / 375-445 (AF-fix)** |

**Cap 450 GPU-h** (the supervisor's). AF-eag's upper end exceeds it; I re-price after the AF smoke and ask before any job
that would project the total past 450.

Wall (calendar about 1.15x elapsed plus an hour to score): DR lanes about 2.5 h of training each, read about 19:00-21:00
10-24; AF about 10-25 morning (AF-fix) to midday (AF-eag), from its launch; LUG about 33 h of training each, read about
10-26 06:00-14:00. Harness deadline about 22:00 EDT 10-26; HANDOFF_SLICESALL.md at 20:00 EDT 10-26 if anything is
unread. Registry row `^isla\.slicesall\.` (expected_end 2026-10-26) added before the first job, removed after the last.

## 9. Amendments

### A1. Amendment before submission of the AF arm (supervisor, reply sent about 00:13 EDT 10-25 by the supervisor's correction, received 00:14 EDT by `date`)

**What was known when written:** the DR arm's full read (RESULTS.md, reading_DR.md: no change, no flag, 1.006 / 1.005); the
DR and LUG smokes, including the LUG step ratio of 1.04; the LUG lanes' training curves to epoch 150 (event guard only, no
score). No AF job, checkpoint, log or score exists. Nothing about the LUG scores is known.

1. **The AF control is r090eag_af_dtt** (option AF-eag of §3a). The AF lanes 4-5 are kernel/r090eagaf rows 2-3, changed only in
   run_id and `model.n_slices`. They are trained eager (`compile=false model.compile_blocks=false`), paired to the control's
   training path. They are scored by kernel/r090eagaf/score.sbatch (bd40af29) and its rows 8-10, 12-14, changed the same way,
   eager, native / area / posed, 200 full_test cases at 100,000 points; area is checked to equal native and labelled
   degenerate. The AF smoke gates on every §4 item except the block-wise check, which runs beside but does not gate, since
   the arm has no compiled region. The supervisor's reasons: the control's seeds agree, so the no-harm check has teeth; it is
   the shared control of DTT-POSE-EXP and DTT-KNOBS; and the 128-slice decision at the freeze is worth about 800 GPU-h in the
   final rerun, so about 40 more GPU-h for sensitivity is worth it.
2. **Cap 500 GPU-h** (was 450).
3. **Launcher:** `$T/kernel/slicesall_af`, a new launcher dir, so the running kernel/slicesall launcher is not edited. It holds
   byte copies of kernel/slicesall's lanes.sbatch, ckpt_preflight.py, site.env and submit.sh (the same job prefix
   `isla.slicesall.`), plus its own lanes.tsv (rows 4-5) and score_rows_af.tsv. `$T/kernel/slicesall_beag` holds byte copies
   of kernel/r090eagaf's lanes.sbatch, site.env, lanes.tsv and ckpt_preflight.py for the smoke's control unit (cluster/setup_af.sh).
4. Rules, verdict and predictions are unchanged (P6, P7 and P12 now refer to AF-eag).
