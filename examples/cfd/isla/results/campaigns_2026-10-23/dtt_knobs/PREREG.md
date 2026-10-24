# DTT-KNOBS stage 2: retrained knobs of ISLA-DTT's boundary read on AirfRANS (pre-registration)

Coordinator DTT-KNOBS (research-lane, launched 2026-10-23 23:34 EDT by the supervisor 01a13c2c). Written 2026-10-24
02:58-03:01 EDT and corrected 03:07-03:12 EDT after a second read (`PREREG_REVIEW.md`; first version `PREREG.md.v1`),
**before any stage-2 job: nothing of stage 2 has been built or submitted, and no arm snapshot, checkpoint, log or score
exists.** Approval in principle: supervisor 02:54 EDT (six arms, code base, six
conditions, provisional cap 950 GPU-h, set at approval). Stage 1: `STAGE1.md` (md5 1a28dc7c). Implementation plan:
`s2plan/PLAN.md` and `s2plan/REPORT.md` (DK-S2PLAN, read-only, 02:47 EDT).

## 1. Questions

ISLA-DTT reads the boundary after every slice block through a Barnes-Hut forest at `TREE_READ_THETA` 1, with a
16-function kernel basis and a read after each of 12 blocks. Stage 1 found that on checkpoints trained at theta 1 every
inference-time change of the read is worse, so only retraining can say what a setting costs. Stage 2 retrains on
AirfRANS (eager, the standing's protocol, two seeds) and asks:

- **Q1, cheaper reads.** Does ISLA-DTT trained with a cheaper read (theta 2; reads after every second block; a basis
  of 8) predict as well as the standing? A cheaper read that costs nothing is a candidate for the scope freeze.
- **Q2, Peter's knob rule.** Retrained at theta 2, 1 (the standing), 0.5 and 0 (the exact read), does the error
  converge as theta tightens (the tree read earns credit as a convergent approximation), or is theta a pure cost knob?
- **Q3, the far-field gap.** Does a finer read (theta 0.5, exact) narrow ISLA-DTT's gap to interacting ISLA in the wake
  and beyond one chord, where stage 1 located it?
- **Q4, the bundle thread.** Does ISLA-DTT trained without ISLA's scales but with the inlet speed standardized
  (DTT-NOSC-STD, the mirror of ATTRIB-090's NOSC-STD) read like B? If so, DTT-NOSC's cost (C_f) is an interaction with
  the raw speed, not a cost of 0.9.0's bundle.

## 2. What was seen before registering (disclosure)

- All of stage 1 (`STAGE1.md` and the seven child reports, reviews and revisions), including every inference-time
  number on the control's checkpoints: theta 2 costs 1.30-1.91x at inference, theta 0.5 1.18-1.44x, the exact read
  1.79-4.68x (AirfRANS, native, per seed); B1 (every second block, mean-replaced) 15.5 / 11.5x; basis projected onto 8
  directions 1.11 / 1.28x. None of these is a retrain result.
- The control's scores in full (`scratch/isla-ref090-2026-10-20/eag_af_dtt.md`), the interacting ISLA standing
  (`eag_af_isla.md` there), r090fix_af_dtt's (ATTRIB's B), ATTRIB-090 RESULTS.md sections 2-3c (DTT-NOSC ÷ B 1.264 /
  1.225 / 1.158; NOSC-STD's reading for interacting ISLA; the mirrors), DK-GAP's regional table (`gap/gap_table.md`).
- The control lanes' step times and link history from sacct, train.log and metrics.jsonl (DK-S2PLAN, monitoring only).
- Nothing of any stage-2 arm exists.

## 3. Arms

`T` is the LHR task root. Every new snapshot is `cp -a $T/code_cmpaf_fix` (SNAPSHOT_HEAD 92d9d7800 + td0152 + the
COMPILE-AF model.py, md5 45b24fb8; tree_read.py c99dbc37) with one file replaced, an appended SNAPSHOT_HEAD line, the
changed file's md5 and its diff recorded in JOBS.md and `s2/patches/<arm>.diff`, then `chmod -R a-w`. Exact diffs:
`s2plan/PLAN.md` section 1. **One change per arm and nothing else** (supervisor's condition 1).

| prio | arm | the one change | snapshot | trained | control |
|---|---|---|---|---|---|
| 1 | K-THETA2 | tree_read.py:126 `TREE_READ_THETA = 2.0` | code_dttk_theta2 | eager | r090eag_af_dtt 42 / 43 |
| 2 | K-THETA05 | `TREE_READ_THETA = 0.5`; the reference's coherence test kept (it is part of what theta 0.5 means in this code) | code_dttk_theta05 | eager | r090eag_af_dtt |
| 3 | K-B1 | model.py 938 / 1913 / 1917: build and apply the tree read only after 0-based blocks 1, 3, 5, 7, 9, 11 (1-based 2, 4, ..., 12); the other six reads are not built (the recipe's DDP has no `find_unused_parameters`) | code_dttk_b1 | eager | r090eag_af_dtt |
| 4 | K-BASIS8 | tree_read.py:122 `KERNEL_READ_BASIS = 8` (heads, width, key, hidden unchanged) | code_dttk_basis8 | eager | r090eag_af_dtt |
| 5 | K-EXACT | `TREE_READ_THETA = 0.0` (the box test admits no node: every point reads every cell through the same path) | code_dttk_exact | eager (gated, section 8) | r090eag_af_dtt |
| 6 | DTT-NOSC-STD | ATTRIB's NOSC-STD config `isla_volume_airfrans_att_noscstd` (recipe_att090b) plus `++model.query_independent=true`; no code change | code_att090fix (exists) | fix-trained (`model.compile_blocks=true`), as ATTRIB's arms | ATTRIB's B r090fix_af_dtt 42 / 43 |

Everything else equals the control's row in `$T/kernel/r090eagaf/lanes.tsv` (rows 2-3): `airfrans_uvolume_2d`,
`isla_volume_airfrans`, extra `train_split=full_train val_split=full_test +training.val_interval=10 model.hidden=512
model.n_slices=512 model.compile_blocks=true ++model.query_independent=true compile=false model.compile_blocks=false`,
lr 3.75e-4, 500 epochs, 100,000 cells, bf16 (site.env LANE_PRECISION), seeds 42 and 43, recipe_r090. Arm 6's row is
ATTRIB's `$T/kernel/att3` NOSC-STD row with `++model.query_independent=true` and `model.compile_blocks=true`, recipe
recipe_att090b, as ATTRIB's DTT-NOSC (att2 rows 14-15) was built. A `diff` of each row against its control's row is
recorded in JOBS.md and shows only run id, group and code.

The arms keep the two code facts of stage 1 (the bf16 training forest; the lug's octant ties), exactly like the control,
so arms 1-5 are compared with their control within one code. Lane DTT-POSE fixes them separately (`CODE_FACTS.md`).

**Arm 6, like with like.** Arm 6 runs on code_att090fix + recipe_att090b, its reference B on code_cmpaf_fix +
recipe_r090, as in ATTRIB-090. code_att090fix's isla/ differs from code_cmpaf_fix's only by ATTRIB's ablation switches
(`query_schema`, `query_embedding`, `query_encoding`), inactive in this config, and its tree_read.py only by an optional
`query_vecs` path, off without a query_schema (DK-S2PLAN's diff, 02:44 EDT; ATTRIB section 9b made the same check for
its B). Arm 6's one change is relative to DTT-NOSC (att2_af_dttnosc, the raw inlet speed): the inlet speed
standardized, `(U_inf_mag - 62.116) / 17.914`. B is the frame's reference, as in ATTRIB.

The K-B1 change is the one real code change. It is also committed, as a record, on branch `psharpe/isla-dttknobs-b1` off
`origin/psharpe/isla-geo-bundle` (`git commit -s`, hooks on, plain push, never merged); the snapshot is what runs.

## 4. Launch, smoke, scoring, checks

- **Launchers** (copies of r090eagaf's, `s2plan/PLAN.md` section 2): `$T/kernel/dttknobs_s2` (arms 1-5, recipe_r090)
  and `$T/kernel/dttknobs_s2n` (arm 6, recipe_att090b); lanes.sbatch and ckpt_preflight.py byte copies (md5 8eecec8f /
  e34bd611); job names `isla.dttknobs.s2.<arm>-seed4x`; all five bad nodes in `exclude_nodes.txt`; 03:59:00 normal-QoS
  chained singleton links.
- **Concurrency.** Stage 2 needs 12 lanes at once (one running link and one pending successor each). COMMON_RULES'
  "at most 2 jobs of yours queued or running" was a stage-1 rule and does not apply to stage-2 lanes; submissions are
  still staggered two lanes at a time (LANE_RULES), short QoS stays at 2 campaign jobs, and lanes are **submitted in
  priority order** (arm 1 first), so a full queue delays arm 6 first (condition 6).
- **Smokes** (before any lane; short QoS through the campaign's slot dirs; r090eagaf's smoke.sbatch and smoke_wrap.py
  copied, md5 704f8188 / 32a2991e, paths and job name `isla.dttknobs.s2.smoke` changed). Units of 50 training steps
  (K-EXACT 200) through the lane launcher with NO_CHAIN; job (a) K-THETA2, K-THETA05, K-B1, K-BASIS8; job (b) K-EXACT
  eager, K-EXACT fix-trained (`model.compile_blocks=true`, 200 steps; it prices the alternative of section 8) and
  DTT-NOSC-STD, the last with a 50-case compiled-against-eager forward check on the smoke's checkpoint (ATTRIB's
  `cmpaf_diag.py`, as ATTRIB section 11 did for configs that change the graph; pass if every case's prediction
  deviation is at most 1e-5 relative, against the 1.5e-7 that ATTRIB's MIXOFF check read). Each unit passes only if: rc 0, no traceback, no non-finite loss, no OOM; its resolved_config.yaml equals
  the control's (`runs/r090eag_af_dtt_seed42`; arm 6: `runs/att3_af_noscstd_seed42`'s plus query_independent and
  compile_blocks=true) except run_id and the smoke paths; the unit prints, from its own process, the effective setting
  (`_tree_read_theta`, `len(kernel_reads)` and the read blocks, `kernel_basis.net[-1].out_features`), which must match
  its arm; arm 6's resolved config shows `U_inf_mag_std` in `global_schema` and its input has mean about 0 and std about
  1 over the smoke's batches. **Each unit records its median step and peak memory** (condition 5), against the
  control's 0.93 s (eager) and 0.57 s (fix-trained). A failed smoke unit holds its arm, not the others.
- **Event guard** (as SLICES-AF): validation loss above 4x its running minimum after epoch 10, a non-finite loss, or a
  BLOCKED marker stops that lane and is reported; it is not a verdict. Links are never cancelled or resubmitted except a
  duplicate after NODE_FAIL (keep the oldest).
- **Scoring exactly as the control:** r090eagaf's score.sbatch unchanged (bd40af29), `-J isla.dttknobs.s2.score`, the
  five nodes excluded, `SCORE_ROWS=$T/kernel/dttknobs_s2/score_rows.tsv`, `EVAL_ROOT=$T/evals_dttknobs_s2`; float32,
  `infer_split=val` (= full_test, 200 cases), 100,000 points, the final checkpoint, eager; protocols **native and posed**.
  Area is not scored: on AirfRANS it equals native by construction (degenerate). Each row's code is the arm's snapshot.
- **Checks before reading** each row: `"epoch": 500`; 200 `infer_step` rows with the control's sample ids;
  `SNAPSHOT_USED` = the arm's snapshot (a theta checkpoint loads silently in the wrong snapshot); DATASET_USED as the
  control's; the score log's extra equal to the lane's, plus `compile=false model.compile_blocks=false` for fix-trained
  rows (arm 6, and K-EXACT if fix-trained), which are scored eager as r090fix's rows were. A failed check makes the row
  invalid and its rule "incomplete as registered".
- **Regional scoring** (for R4): one more native inference per run with DK-SENS's probe (`sens/lhr/probe_sens.py`, md5
  323f5396, its `--regions` mode, which imports `af_regions` from `$T/kernel/dttknobs_gap/reduce_gap.py`, md5 6be03f4b,
  unchanged) at the arm's own read, on the arm's snapshot, all 200 cases; gate: its per-case metrics equal the scorer's
  to 1e-5. For K-B1 the probe's read hooks map `kernel_reads` to the read blocks; regional errors need no hooks.
- **Coherence count for K-THETA05** (condition 1, descriptive, not a rule): on all 800 full_train and 200 full_test
  cases, the number of nodes the box test admits and coherence refuses at theta 0.5, per case, both in float32 and
  under bf16 autocast (the forest training builds, `CODE_FACTS.md` section 1), from DK-COST's decoupled-coherence count
  (`cost/cost_probe.py`) on a cpu-short job. Reported beside R1.

## 5. Rules (registered)

CM is the scorer's case-mean relative L2 (`infer_summary`, else the mean of the 200 `infer_step` rows) on the 200
full_test cases, native, per field: pressure, in-plane velocity (`velocity_l2`; never velocity_z), nut. All three
fields are read; none is the headline.

**R1, per field, arms 1-5 against the control** (SLICES-AF's rule, LRE-AZ's, unchanged; condition 2).
- `D = ln( mean_seeds CM(arm) / mean_seeds CM(control) )`.
- `SE = sqrt((s_arm^2 + s_c^2) / 2)`, s = a run pair's two-seed sample sd (n - 1) of ln CM. The control's own:
  s_c = 0.0022 / 0.0395 / 0.0078 (pressure / velocity / nut; seeds 0.012428 / 0.0123887, 0.00737341 / 0.00697252,
  0.0207112 / 0.0204842).
- **Helps** if D < ln 0.90 and |D| > 2 SE. **Harms** if D > ln 1.10 and |D| > 2 SE. **Otherwise no change.**
- With these spreads the ln 1.10 bar (0.095) binds on every field unless an arm's own seeds are far apart (velocity's
  2 SE is 0.056 at s_arm = 0, 0.079 at s_arm = s_c).
- **Sub-bar flag** (SLICES-DRV's, as amended there): per field, up if arm ÷ control > 1.03 with both arm seeds above
  both control seeds; down if < 0.97 with both below. It changes no label; its decision use is in R3.
- Beside, not verdicts: the same D, SE and label on posed (non-degenerate, it rotates and translates the input; the
  same runs, no independent number); seed-matched ratios (K-B1 and K-BASIS8 change the initialization, so pairing is
  not assumed to cancel seed effects); the case median and the share of the 200 cases where the arm beats the control;
  the arm ÷ interacting ISLA (r090eag_af_isla) ratios beside the control's 0.994 / 1.128 / 1.342.
- Slots: 2 runs per arm, 4 per comparison. Three fields and two protocols are readings of those runs.

**R2, the ladder (Peter's knob rule; condition 3)**, per field, from the R1 labels L_t of K-THETA2 (t = 2), K-THETA05
(0.5) and K-EXACT (0) against the control (t = 1), and D_t (D_1 = 0; seed-mean ln CM, as R1). The adjacent steps are
(2, 1), (1, 0.5), (0.5, 0); a step **rises** if D_tighter - D_looser > 2 SE of that pair (SE as R1, from the two
pairs' own spreads).
- **Converges** (the tree read earns credit as a convergent approximation on that field): L_0 = helps, L_0.5 is helps
  or no change, L_2 is harms or no change, and no adjacent step rises.
- **Pure cost knob** (at theta <= 1; the approximation costs nothing there): L_0 = no change and L_0.5 = no change.
  Sub-label **free to theta 2** if also L_2 = no change and K-THETA2 raises no up flag on the field.
- **Exact read worse:** L_0 = harms. The tree's aggregate is better to train with than the exact read on that field,
  so theta is not a convergent approximation knob there.
- Otherwise **mixed**.
- **Overall reading:** the label that at least two of the three fields carry; otherwise mixed. The sub-label "free to
  theta 2" is carried overall only if at least two fields carry it. If K-EXACT is not run or not read (section 8), R2
  is "incomplete as registered" and the three-point ladder (2, 1, 0.5) is reported beside it with no label. If K-EXACT
  is fix-trained (only on the supervisor's decision, section 8), L_0 is read against r090fix_af_dtt with that control's
  spread, and every R2 label carries "the (0.5, 0) step crosses training paths".

**R3, decision use for the cheaper arms** (K-THETA2, K-B1, K-BASIS8; condition 4).
- **Candidate for the scope freeze** (about 10-27), as a free cost cut: R1 reads no harm (no change or helps) on all
  three fields **and** no field raises an up flag. Then stage 3 (section 7) is proposed for it.
- A harm on any field, or an up flag on any field: **not a free cost cut**; it goes to the scope freeze as a
  cost-against-accuracy trade, with its measured saving (smoke and lane step, memory, and DK-COST's car-scale
  projection), or is dropped.
- K-B1 changes where the model reads, an architecture change: its adoption also needs Peter's word, whatever it reads.
- **For the tighter arms** (K-THETA05, K-EXACT): helps on at least one field and no harm on any makes a tighter theta a
  candidate for the scope freeze as an accuracy-for-cost trade, weighed against its car-scale cost (DK-COST: at theta 0.5 the coupled read is 3.2x the pairs at 200k
  tokens on DrivAerML Volume and 400k points do not fit a GB300; the exact read is out of reach at car scale).

**R4, the far-field gap** (DK-GAP's regions; Q3), per run, native, pooled over the 200 cases, DK-GAP's definitions:
zone `wake` (downstream of the trailing edge along the inflow, |offset| < 0.1, d >= 0.01) and band `>= 1` chord; e_R =
sqrt(sum_R |p - t|^2 / sum_R |t|^2) per field. Two cells are read, the two that were robust in both seeds in stage 1:
in-plane velocity in the wake and nut beyond one chord. Per cell and arm, R1's shape on e_R:
D_R = ln( mean_seeds e_R(arm) / mean_seeds e_R(control) ), SE_R = sqrt((s_arm,R^2 + s_c,R^2) / 2) with each pair's
two-seed sd of ln e_R (stage 1: ISLA-DTT's own seeds differ by up to 1.72x on nut beyond one chord, so this bar can be
wide):
- **narrows** if D_R < ln 0.90 and |D_R| > 2 SE_R; **widens** if D_R > ln 1.10 and |D_R| > 2 SE_R; otherwise
  **level**. Seed-matched ratios are reported beside.
- Beside: every field in both regions and in the wall-distance bands, per seed, as ratios to the control and to
  interacting ISLA (seed-matched), and the gap ratio e_R(arm) / e_R(ISLA) against the control's (stage 1: wake velocity
  1.95 / 1.99, nut beyond one chord 1.48 / 2.47).
- Q3 is answered "a finer read narrows the far-field gap" if K-THETA05 or K-EXACT narrows a cell and neither arm
  widens either cell. If K-EXACT is not read, Q3 is answered from K-THETA05 alone, labelled so.

**R5, DTT-NOSC-STD against B** (Q4; ATTRIB-090's frame, sections 10 and 12 of its PREREG).
- B = r090fix_af_dtt 42 / 43 (0.0127425 / 0.0125662, 0.00710411 / 0.00714819, 0.0198947 / 0.023094); DTT-NOSC ÷ B =
  1.264 / 1.225 / 1.158 (RESULTS.md 3b), so G_f = ln(DTT-NOSC ÷ B) = 0.234 / 0.203 / 0.147.
- Δ = ln( mean_seeds CM(DTT-NOSC-STD) / mean_seeds CM(B) ) and Δ_N = ln( mean_seeds CM(DTT-NOSC-STD) /
  mean_seeds CM(DTT-NOSC) ). σ_f is ATTRIB section 5's pooled within-arm sd of ln e for the ISLA-DTT form on AirfRANS
  (the arms of RESULTS.md 3c's pool, 6 dof, 0.042 / 0.029 / 0.064) with DTT-NOSC-STD's two seeds added: **7 dof**,
  computed once with ATTRIB's reduce2.py convention. SE = σ_f (1/2 + 1/2)^(1/2) = σ_f for both comparisons.
- **A (differs from B)** on f: |Δ| >= 2 SE and |exp Δ - 1| >= 0.05 (ATTRIB's R1). **N (differs from DTT-NOSC)**:
  |Δ_N| >= 2 SE and |exp Δ_N - 1| >= 0.05, with Δ_N < 0.
- **Reading per field:** **standardization removes DTT-NOSC's loss on f** if N and (not A, or Δ < 0);
  **standardization leaves it (the scales carry it)** if A with Δ > 0 and not N; **both share it** if A with Δ > 0
  and N; **unresolved on f** if neither A nor N.
- φ_f = Δ / G_f (the share of DTT-NOSC's loss against B that standardization leaves) is reported beside, and Δ / D_DTT
  too (ATTRIB section 10's denominator, D_DTT = 0.103 / 0.156 / 0.026, the CONT-to-B step). G_f replaces D_f as the
  denominator because the question here is whether standardization removes DTT-NOSC's loss against B, and D_DTT on nut
  (0.026) would make the ratio unstable.
- For the bundle thread (Q4), on pressure and nut (where C_f >= ln 1.05): a "removes" reading on f means DTT-NOSC's
  cost C_f is an interaction with the raw speed, **not a cost of 0.9.0's bundle**, on f; "leaves" or "shares" locates
  it in the scales, as far as two seeds can show; "unresolved" stays unresolved.
- Beside: DTT-NOSC-STD ÷ DTT-CONT and ÷ DTT-NOSC per field and seed; posed; the interacting-ISLA mirror's reading
  (ATTRIB 2c) beside it for the "does either form use the scales" question. Labels: fix-trained.

## 6. Predictions (seed means; probabilities mine, written before any stage-2 result)

| | prediction | p |
|---|---|---|
| P1 | K-THETA2: no harm on all three fields | 0.45 |
| P2 | K-THETA2: an up flag on at least one field | 0.60 |
| P3 | K-THETA05: helps on at least one field | 0.20 |
| P4 | K-B1: no harm on all three fields | 0.40 |
| P5 | K-BASIS8: no harm on all three fields and no up flag (a scope-freeze candidate) | 0.45 |
| P6 | K-EXACT, if trained and read: helps on at least one field | 0.25 |
| P7 | K-EXACT, if trained and read: harms on at least one field | 0.20 |
| P8 | R2 overall, if K-EXACT is trained and read: pure cost knob 0.45, converges 0.20, exact read worse 0.10, mixed 0.25 | - |
| P9 | R4: a finer read narrows the far-field gap | 0.25 |
| P10 | R5 on pressure: "standardization removes DTT-NOSC's loss" | 0.60 |
| P11 | K-EXACT's eager smoke projects the pair above 300 GPU-h (section 8 stops it) | 0.75 |
| P12 | R5 on nut: "unresolved" (nut's sd 0.064 makes 2 SE 0.128 against G_nut 0.147) | 0.45 |

Why: stage 1's inference costs (1.3-1.9x at theta 2) are those of a model adapted to theta 1, and a model trained at
theta 2 adapts to its own aggregate, so most of it should be recovered; the up flag is likely because pressure's
control seeds are 0.3 % apart. Basis 8 keeps 99 % of the kernels' variance in stage 1 (5-6 directions). B1 drops block
0's read, the largest on AirfRANS. The gap sits in the wake and far field, where far points read only the coarsest
clusters at theta 1, so a finer read may help there; but no inference theta narrowed it, and the gap is also on the
training cases.

## 7. Stage 3, named now (for later approval)

The second dataset for any cheaper arm that R3 makes a scope-freeze candidate: **DrivAerML Volume**, where the read's
cost lives (stage 1: 400k points fit a GB300 only at theta >= 1.5). Arms: ISLA-DTT at the reference read and at the
candidate's setting, two seeds each, on DrivAerML Volume's canonical configuration at that time (Peter's whole-boundary
ruling: the domain meshes as named conditions, as FLOOR-DRV builds them), scored on its standing's protocol.
- **Trigger:** an AirfRANS cheaper arm reads no harm on all three fields and raises no up flag (R3).
- **Cost** (estimate, to be re-priced by a smoke when proposed): ISLA-DTT's step there is about 2.9x interacting ISLA's
  (1,202 ms against 422 at 100k points), so about 130 GPU-h per run, about 520 GPU-h for the four runs.
- **Reading:** R1 and its flag, with that dataset's control spread.
Not submitted under this PREREG.

## 8. Cost, cap, schedule

Measured on the control lanes (DK-S2PLAN, sacct): eager AirfRANS ISLA-DTT costs **110.8 / 110.1 GPU-h per run** (median
step 0.93 s, 100,000 steps, about 31 h of wall), not the brief's 70 (that was the compiled path, 66 GPU-h, 0.57 s).

| arm | step (s), basis | GPU-h, 2 seeds | wall per run |
|---|---|---|---|
| K-THETA2 | 0.89 (DK-COST 891 / 935 ms) | about 210 | about 30 h |
| K-THETA05 | 0.92 | about 220 | about 31 h |
| K-B1 | 0.89 (DK-CAP: -4 %) | about 212 | about 30 h |
| K-BASIS8 | 0.93 | about 220 | about 31 h |
| K-EXACT, eager | 1.43-1.73, extrapolated | about 330-390 | 45-54 h |
| DTT-NOSC-STD, fix-trained | 0.567 (DTT-NOSC measured) | about 136 | about 21 h |
| smokes, scoring, regional inference | | about 12 | |

- **Arms 1-4 and 6: about 1,010 GPU-h. With K-EXACT eager: about 1,340-1,400.** Both are above the provisional 950.
- **K-EXACT gate** (condition 5): its smoke sets its GPU-h, as 2 x (median step x 100,000 / 3600 + 1 h) x 4. If that is
  above 300 GPU-h, or its peak memory above 200 GiB, K-EXACT is **not submitted**, and I tell the supervisor at once.
  The estimate (330-390) makes that likely (P11). The pre-written alternative, for the supervisor's decision only:
  K-EXACT fix-trained (`model.compile_blocks=true`, about 245-310 GPU-h, 34-43 h), priced from its own smoke unit by the
  same formula and held to the same 300 GPU-h gate, read against r090fix_af_dtt as R2 says.
- **Cap.** I ask for 1,050 GPU-h for arms 1-4 and 6 with overheads, and K-EXACT approved separately from its smoke.
  At a cap of 950, arms 1-4 run with their overheads (about 874) and DTT-NOSC-STD waits for an increase. If failures
  or reruns project past the cap, I ask first.
- **Concurrency.** The arms need about 12 nodes at once (section 4); I ask for that explicitly with the approval.
- **Schedule** (if approved by about 04:00 EDT 10-24): snapshots, launchers and md5s 04:00-05:00; smokes 05:00-05:45;
  lanes from about 06:00 in priority order. DTT-NOSC-STD read about 04:00-08:00 10-25; arms 1-4 scored about
  14:00-20:00 10-25; K-EXACT eager (if run) about 04:00-12:00 10-26. HANDOFF_DTTKNOBS.md at 18:00 EDT 10-26 lists
  anything unread; the harness deadline is 23:34 EDT 10-26.
- Registry row `^isla\.dttknobs\.` stays (expected_end 2026-10-26); it is removed after the last job.

## 9. Amendments

Allowed only before results, as LANE_RULES define them (nobody writing has seen any score or validation curve of the
arms the amended rule reads), marked "Amendment before submission" or "Amendment before results", with what was known.

### Amendment before submission (supervisor), received 2026-10-24 03:40 EDT (by `date`)

Approval of md5 84a2528018a81dc20a5f39292b1882e0 as registered, with this one amendment of the supervisor's own gate.
**What was known:** no stage-2 snapshot has run, no stage-2 job has been submitted and no stage-2 number exists. Snapshots
and launchers were being built (DK-S2BUILD), nothing submitted. K-EXACT's eager cost was known only as an extrapolation,
about 330-390 GPU-h for the pair (section 8; DK-S2PLAN from DK-SENS's inference times).

1. Arms 1-4 and 6 are approved, cap **1,050 GPU-h** for them with overheads. Arms 1-5 train eager as registered.
2. **K-EXACT: eager only** (the whole ladder on one training path). Section 8's gate is replaced: K-EXACT's eager pair is
   submitted if its smoke projects it **at or below 400 GPU-h** (section 8's formula) **and** its peak memory at or below
   **200 GiB**. Above either, it is not submitted and the supervisor is told at once, with the fix-trained smoke unit's
   projection beside it. The fix-trained alternative is no longer pre-approved for any reading: its smoke unit stays
   only to price that message. K-EXACT's cap is **400 GPU-h**; stage 2 in all is capped at **1,450 GPU-h**. If failures
   or reruns would project past either cap, I ask first. R2's fix-trained K-EXACT clause is void.
3. Concurrency: up to 12 lanes at once (one running link and one pending successor each), submitted in the registered
   priority order, staggered two lanes at a time; the campaign's short QoS at most 2; user-wide short QoS within 12.
4. If K-EXACT is submitted, the registry row's expected_end becomes 2026-10-27.
5. When every smoke unit passes its registered checks, the lanes are submitted without a further approval. A failed
   unit or a tripped K-EXACT gate is reported at once; one message reports all lanes submitted, with their job ids.
The rules R1-R5 and the predictions are unchanged.
