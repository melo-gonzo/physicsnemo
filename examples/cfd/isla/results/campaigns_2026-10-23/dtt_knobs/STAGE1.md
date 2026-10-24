# DTT-KNOBS stage 1: ISLA-DTT's boundary read, its knobs and its accuracy gap (inference only)

Coordinator DTT-KNOBS, drafted 2026-10-24 02:25 EDT, revised 02:46 EDT after STAGE1_REVIEW.md (by `date`). Scratch:
`scratch/isla-dtt-knobs-2026-10-23/`. Every number below comes from a child report in the named subfolder
(`<part>/REPORT.md`, its tables and JSON, its `JOBS.md`). Six of the seven reports (sens, cost, cap, gap, order,
treeprec) were second-read by a reviewer (`<part>/REVIEW.md`), confirmed with corrections, and revised
(`REVISION_NOTE.md`; `REPORT_v1.md` is the first version). design/ is a read-only design study and was not reviewed;
its one fact that the rest relies on, the coherence bound, was checked in cost/ and sens/. This file was second-read too
(`STAGE1_REVIEW.md`).

## The answer in brief

1. **On theta-1 / leaf-4-trained checkpoints, theta acts as a trained constant, not an inference-time knob.** Read
   natively, both datasets score best at theta 1 on every field and both seeds. The error rises on both sides of
   theta 1, and the exact read (theta 0) is the worst native setting: 1.8-4.7x the theta-1 error. So at inference the
   error does not converge as theta tightens, and Peter's knob rule can be tested only by a ladder of retrained
   models. Leaf 2 and 8 are inert on AirfRANS (<= 0.1 %) and cost up to +33 % on the lug.
2. **A cheaper read pays only at car scale.** On AirfRANS and the lug a larger theta saves pairs but no eager forward
   time. On DrivAerML Volume with the whole boundary, 400k points fit a GB300 at theta >= 1.5 and at nothing
   <= 1.25; at theta 1 the limit is 349k.
3. **At inference the read's slack is in the kernel basis.** The 16-function basis has effective rank 4 (AirfRANS)
   and 1.5 (lug). Projected onto 8 directions it costs 1.11 / 1.28x on AirfRANS but 1.9 / 3.6x on the lug. Every read
   carries part of the trained model, except the lug's first two reads, which are unused in both seeds. Cutting heads
   costs 6-9x. These are inference rankings of models trained with every read and head; how they transfer to a retrain
   is untested. The 12 per-block reads alone (shared forest, walk and basis excluded; eager) are 9 % of the AirfRANS
   step and 26 % of the lug's.
4. **The gap to interacting ISLA is a fit gap, not a generalization gap, and no inference-time setting of the read
   explains it.** On AirfRANS it is a bias in the wake (in-plane velocity 1.95 / 1.99x ISLA's error there, seeds 42 /
   43) and beyond one chord. On the lug the float32 gap of 1.39 is real under the standing's protocol. Ruled out as
   causes: the bf16 training forest (it widens the gap slightly, it does not make it), the stored SDF, the old query
   embedding and every inference-time theta. Whether a retrained finer read closes it is untested (stage-2 arm 2).
5. **Two code facts found on the way** (no fix proposed; a fix touches the reference path). Under bf16 training,
   ISLA-DTT's forest is built from bf16-rounded coordinates. And the lug's posed score differs from native at every
   theta > 0, most likely because most of its normals sit on an octant boundary in the canonical frame (not
   confirmed).

Stage-1 compute: **21.76 GPU-h of the 30 cap** (table in `JOBS.md`, each part sacct-checked by its reviewer). Nothing
is running. No commit, branch or worktree.

## (a) Inference sensitivity: theta and leaf on trained checkpoints (`sens/`, 7.71 GPU-h)

Runs r090eag_af_dtt_seed42/43 (AirfRANS, 200 cases) and r090fix_lug_dtt_seed42/43 (lug, 600 cases), float32, eager,
the scorer's protocol. The gate reproduced every per-run standing score (AirfRANS to <= 1.8e-6). The table gives error
÷ theta-1 error on the same weights, native, as [seed 42, seed 43]:

| theta | AirfRANS pressure | AirfRANS velocity (in-plane) | AirfRANS nut | lug von Mises |
|---|---|---|---|---|
| 0 (exact) | [2.71, 2.36] | [1.79, 2.04] | [3.54, 4.68] | [4.62, 3.47] |
| 0.25 | [2.07, 1.86] | [1.58, 1.73] | [3.17, 3.98] | [2.90, 2.53] |
| 0.5 | [1.41, 1.40] | [1.18, 1.23] | [1.27, 1.44] | [1.97, 2.00] |
| 0.75 | [1.07, 1.03] | [1.03, 1.05] | [1.04, 1.08] | [1.24, 1.29] |
| 1.25 | [1.18, 1.20] | [1.04, 1.07] | [1.07, 1.07] | [1.39, 1.41] |
| 1.5 | [1.35, 1.35] | [1.10, 1.14] | [1.16, 1.17] | [2.12, 1.94] |
| 2 | [1.86, 1.91] | [1.30, 1.33] | [1.38, 1.44] | [3.20, 2.79] |

- Leaf at theta 1: leaf 2 and 8 change AirfRANS by <= 0.1 %. On the lug, leaf 8 changes it by <= 0.8 %, and leaf 2 by
  +2 % (seed 42) and +33 % (seed 43).
- Regions (DK-GAP's definitions): the AirfRANS error beyond one chord is 1.00x its theta-1 value for theta 1.25-2 in
  both seeds, within 1 % at 0.75, and grows below that (1.16x on pressure at 0.5). No inference theta narrows the wake
  or far-field gap; the gap beyond one chord is smallest at theta 1.
- Pairs per point on AirfRANS: 21.6 at theta 1, 15.0 at 2, 34.7 at 0.5, 1,007 exact. On the lug: 118, 54, 300 and
  6,525.
- Eager forward time (warm median, forest build and walk included, along with the probe's diagnostics) is flat for
  every theta > 0: AirfRANS 475-498 ms, lug 195-252 ms. The exact read takes 751 / 353 ms and 20.5 / 21.7 GiB.
- This is E2's result from the other side. Dense-trained checkpoints moved 2-20x when read through the tree; these
  tree-trained ones move 1.8-4.7x when read exactly.

## (b) Cost model (`cost/`, 3.63 GPU-h; random weights, bf16 training steps)

- **Pairs per point.** Under the box test, pairs = A theta^-p leaf^-r (log2 N - c) for cluster nodes, plus
  B theta^-q leaf^s for near cells. On DrivAerML p = 0.91, and nodes are 93-99 % of pairs at leaf 4. Held-out error at
  the largest N is 5-25 % median, bounded by a draw-to-draw sawtooth of about ±25 %. The form fails on the HiLift
  surface form (46 % held-out median), because there bf16 adds a near-field term.
- **Step time and memory (DrivAerML).**
  - Step time = 203 ms + 5.16 ms per million pairs per block + 158 ms per 100k points + 184 ms per 100k tokens.
  - Peak memory = 3.2 GiB + 0.265 GiB per million pairs per block + 35.0 GiB per 100k points + 36.0 GiB per 100k
    tokens (held-out memory error <= 0.4 %).
  - The probe's configuration is close to the brief's but not identical (whole boundary on code_geobundle2,
    mean-square loss, geometry memoized). It reproduced 1,180 ms / 132 GiB at 100k (brief: 1,202 / 129) and
    1,790 / 189 at 200k (brief: 1,833 / 183).
- **400k points on DrivAerML Volume** (274 GiB usable):
  - Fit, measured: theta 3 at 1,837 ms / 248 GiB, theta 2 at 2,070 / 260, theta 1.5 at 2,734 / 273 (at the limit).
  - Theta 1.25 does not fit at any leaf.
  - Largest Q that fits: 349k at theta 1, 380k at 1.25, 431k at 2, 462k at 3.
  - Interacting ISLA measured 796 ms / 190 GiB at 400k, so the settings that fit take 2.3-3.4x its time.
  - The blocks alone (35 GiB per 100k points) cap Q near 565k even with no read.
- **Leaf size** is a small effect either way. In pairs, leaf 2 is lowest everywhere. In step time at theta 3, leaf 16
  is 5-9 % faster than leaf 4 on the lug, AirfRANS and AeroZoo, and leaf 8 is 3 % faster on DrivAerML.
- **The normal-coherence test is vacuous at theta >= 0.919 (3D) / 0.765 (2D),** because inside one sign octant
  |mean n| >= 1/sqrt(d). Code and data agree: no node is refused at theta >= 1. Below that bound, theta tightens two
  tests at once. On DrivAerML at theta 0.5 the coupled read costs 3.2x the box-only read at 200k tokens (3.9x at
  800k); on AirfRANS and the synthetic case, 1.0x.
- **The walk fails past 2^31 pairs** (argsort; observed at coherence 0.25, box 1, N 200k, 50k points). Theta 0.5 at
  400k points on DrivAerML would exceed it (projected).

## (c) Where the read's capacity goes (`cap/`, 4.68 GPU-h; inference ablations, not retrains)

- **Update size.** On AirfRANS, read ÷ stream is 0.09-0.36 in blocks 0-3 and 0.02-0.03 by block 11. On the lug,
  blocks 0 and 1 are dead in both seeds: read ÷ stream <= 7.6e-4, and cutting them moves von Mises by < 0.1 %.
- **Attention.** AirfRANS lists hold 22 entries (89 % clusters) with 4-7 effective sources; lug lists hold 118 entries
  with 8-54. Exact near-field cells get 0.1-8 % of the attention.
- **Kernel basis.** Effective rank 4 (AirfRANS) and 1.5 (lug) of 16. Projected onto 12 directions it costs 1.05 /
  1.09x on AirfRANS and 1.08 / 1.31x on the lug; onto 8 directions, 1.11 / 1.28x and 1.9 / 3.6x.
- **Heads.** The top 4 carry 55-100 % of each block's update energy, yet keeping only them costs 6-9x at inference.
- **Fewer reads** (whole set, the cut reads replaced by their mean):
  - Reading every second block (2, 4, ..., 12) costs 15.5 / 11.5x on AirfRANS and 21 / 31x on the lug.
  - Keeping only the last 6 costs 25 / 24x and 36 / 49x.
  - Every second block beats keeping the last k for k <= 6. On the subset, keeping the last 8 or 9 does better.
- **Step cost** (bf16, eager): see brief item 3 for the shares. Reading every second block saves 4 % of the time on
  AirfRANS and 12 % on the lug; 4 heads plus basis 8 saves 0-4 %.

## Read order: the monopole against the exact read (`order/`, 3.07 GPU-h; AirfRANS, 50 cases per seed)

- **Size of the error.** At theta 1 each block's read update is 3.9-27 % (median over points) from the exact read on
  the same states, mostly from aggregating keys and values.
- **Convergence.** Where the model reads, the error converges slowly: theta^0.33-0.69 from 2 down to 0.5. The median
  reaches 1 % only at theta 0.08-0.26, pooled over seeds; seed 43's block 6 and seed 42's block 9 need theta below
  0.0625.
- **Second-order correction (H1).** H1's median error is worse than the monopole's at every theta >= 1 in every block
  (1.01-3.81x at theta 1), and its p95 is worse at every theta >= 0.5: the series is outside its range on these
  states. End to end, H1 at theta 1 scores 5.6-24x worse. So no H1 kernel lane, unless a retrain with H1 (or with
  monopole logits plus the bounded value term) beats the monopole at theta >= 1.

## (d) The accuracy gap (`gap/`, 1.51 GPU-h; `treeprec/`, 1.16 GPU-h)

- **Fit, not generalization.** On training cases ISLA-DTT ÷ ISLA is 0.987 / 1.083 / 1.413 (AirfRANS pressure /
  velocity / nut) and 1.434 (lug), against 0.994 / 1.129 / 1.342 and 1.394 on test. The AirfRANS training subset
  covers Re 2-3e6 only, but the test's low-Re quartile reads alike.
- **AirfRANS: wake and far field.**
  - Robust in both seeds [s42, s43]: in-plane velocity 1.95 / 1.99x in the wake, and nut 1.48 / 2.47x beyond one
    chord, where ISLA-DTT's own seed noise is 1.72.
  - The nut wake share rests on seed 43 (wake nut 1.02 / 1.35).
  - Near the wall nut is 1.20x but carries <= 5 % of the gap.
  - The difference is smooth, a bias rather than tree noise, except for nut beyond 0.1 chord (roughness 0.076 /
    0.112), which is where the nut gap sits. It holds at bf16 and on the training cases.
  - ISLA-DTT behaves as if it had an additive error floor: the ratio is largest where ISLA's own error is small.
- **Lug.** The float32 gap (1.394) covers 96 % of cases and sits 90 % on surface nodes. It lies below the bf16
  read-out noise, so a bf16 score reads the forms level (1.043 [1.096, 0.989]). That is arithmetic masking; whether
  bf16 training causes the gap is untested.
- **Ruled out as causes:**
  - The inference theta (above).
  - The bf16 training forest against the float32 scoring forest. Reading the training forest makes native error
    slightly and consistently worse (0.2-0.7 % on velocity, nut and the lug; pressure level). It widens the gap (lug
    1.394 -> 1.399), so the float32 standing does not inflate the gap.
  - The stored SDF and the old query embedding (ATTRIB-MIRRORS 3c: neither moves any field).
- **Unattributed.** C_f = 0.131 / 0.121 (pressure / nut) of ISLA-DTT's bundle cost stays unattributed. The gap is nil
  on pressure, where C_f is largest, and DK-GAP reads the 0.9.0 gap as mostly not a bundle effect. That is a reading,
  not a measurement: C_f nut 0.121 could be about 40 % of log(1.342) = 0.294.

## Code facts (no fix proposed)

- **bf16 forest.** `points @ frame` in `BoundaryForest.build` and `point_pairs` runs under the caller's autocast
  (tree_read.py lines 267 and 366 at f13eb5ce0; 247 and 346 in code_cmpaf_fix). model.py line 1907 calls
  `_tree_read_pairs` after the autocast-disabled block (lines 1871-1876) has closed.
  - Under bf16 training and in-training validation, the forest's coordinates, boxes and node measures are bf16, while
    diameter_sq and the walk's box table stay float32. Every float32 score therefore reads a different forest.
  - 82 % (AirfRANS) and 99 % (lug) of points get a different list; pairs per point differ by -34 % to +10 % across
    datasets. Native error moves at most 0.7 %.
  - The bf16 forest is not pose-invariant: the lug's posed error doubles under it. There are two candidate
    mechanisms, the uncentred `points @ frame` and the bf16 cast of the lab-frame inputs before the matmul; which
    dominates is untested.
  - On the HiLift surface form, bf16 adds a near-field term that theta does not shrink (2.4x the float32 pairs at
    400k cells).
  - To reproduce the training forest in a float32 probe, wrap the build and walk in autocast; swapping only the
    matmuls does not reproduce it.
- **Lug posed != native** at every theta > 0: posed ÷ native is 1.53 / 1.98 at theta 1 and 1.0000 at theta 0, and the
  posed float32 lists hold 1.29x the pairs.
  - Measured in job 364211 (the lug bf16 run, native pose; the diagnostic ran under its autocast, and the float32
    share was not measured): 84 % of the lug's cells (median over cases, range 76-91 %) have a normal component below
    1e-5 in the canonical frame. For those cells `normal_octants`' sign test rests on rounding residues, natively as
    well as posed.
  - Inferred, not tested: that the posed flip of those signs builds the different forest. Posed octant labels were
    not compared, no tie-breaking rule was tried, and the frame's own near-ties (skew 1.2e-6 against the 1e-9
    threshold) are not excluded.
  - This may touch TIE-EXP's scope, which this campaign has not opened.

## What this means for stage 2

- An inference probe cannot price a retrain: on these checkpoints theta acts as a trained constant, and the capacity
  ablations rank models trained with every read and head. Only stage 2 can say what a cheaper or narrower read costs.
- A cheaper read matters at car scale. AirfRANS can say what a setting costs in accuracy, not what it saves in time.
- Knob rule: with the standing at theta 1, retrained arms at theta 2 and 0.5 (and the exact read, affordable on
  AirfRANS) form a ladder. If the error falls toward the exact-trained model's as theta tightens, the tree read earns
  credit as a convergent approximation. If the exact-trained model is no better, theta is a pure cost knob on
  AirfRANS.

## Proposed stage-2 arms (for the PREREG; AirfRANS eager, the standing's protocol, two seeds each)

Control: the standing r090eag_af_dtt seeds 42 / 43, provided the arms train on its code (code_cmpaf_fix + recipe_r090)
with one setting changed each. Cost per run: ATTRIB-090's 71 GPU-h per AirfRANS ISLA-DTT lane (PREREG section 8).
cost/'s measured AirfRANS bf16 training steps at 100k points (921 / 935 / 891 ms at theta 0.5 / 1 / 2) put arms 1-4
at the same cost. Wall about 18 h per run.

| priority | arm | change | why | GPU-h (2 seeds) |
|---|---|---|---|---|
| 1 | K-THETA2 | theta 2 | the cheaper read; the car setting that fits 400k; costs 1.3-1.9x at inference, which a retrain may recover | ~140 |
| 2 | K-THETA05 | theta 0.5 | the ladder's tight rung; also tests whether a finer retrained read closes the far-field gap (DK-GAP's candidate 2, untested, not triggered) | ~140 |
| 3 | K-B1 | read after blocks 2, 4, ..., 12 only | fewer reads; best at inference among forms with <= 6 reads | ~140 |
| 4 | K-BASIS8 | kernel basis 8 (8 heads kept) | the narrower read where the slack is; heads are left alone because cutting them costs 6-9x at inference | ~140 |
| 5 | K-EXACT | theta 0 (the exact read) | the ladder's limit, the knob rule's reference | ~220, extrapolated (wall about 27 h) |
| 6 | DTT-NOSC-STD | no scales, inlet speed standardized (the supervisor's 23:44 arm) | closes the bundle thread; fix-trained against ATTRIB's B, as ATTRIB's arms are | ~140 |

- **K-THETA05 and coherence.** On the AirfRANS test cases coherence refuses nothing at theta 0.5 (identical pairs and
  metrics with coherence held), but the margin is thin: it binds below 0.505, and the 800 training cases were not
  checked. The PREREG would either check the training cases or train with coherence off, as in DESIGN's D1.
- **K-EXACT cost.** Its cost is the inference ratio (751 ÷ 484 ms = 1.55x); no theta-0 training step was measured.
  cost/'s AirfRANS memory fit gives about 73 GiB, which fits a GB300.
- **Totals.** All six arms: about 920 GPU-h; arms 1-4 alone: about 560.
- **Second dataset.** Decided after the AirfRANS reads, for the arm that wins. My candidate is DrivAerML Volume at
  the winning theta, since that is where the cost lives. The lug costs about 240 GPU-h per run and its two-seed spread
  is 21 %. The lug float32-training arm (DK-GAP's candidate 1, p about 0.55) is left out on cost.

## Decisions needed (asked separately)

1. **Approval per arm and the stage-2 cap.** I recommend arms 1-4 now (about 560 GPU-h), arm 5 if the cap allows,
   and arm 6 as you decide.
2. **The code base.** I recommend training on code_cmpaf_fix (0.9.0) with one constant changed per arm, so that the
   standing is the control and no control pair is retrained. The read is identical on psharpe/isla-geo-bundle.
   Training on the geo-bundle instead needs a new control pair (about 140 GPU-h). Either way the changes would sit on
   psharpe/isla-dttknobs-arms as a record, with snapshots verified by md5.

## For project memory (suggested)

- A theta-1 / leaf-4-trained ISLA-DTT checkpoint is most accurate read at theta 1, leaf 4. No inference theta in
  [0, 2] or leaf in {2, 8} improves it natively (AirfRANS, lug), and the exact read scores 1.8-4.7x worse.
- ISLA-DTT's coherence mask is vacuous for theta >= 0.919 (3D) / 0.765 (2D); the theta-1 read never uses it.
- Under bf16 training, ISLA-DTT's forest is built from bf16-rounded coordinates; when scored in float32 it is built in
  float32. The float32 score differs by at most 0.7 % native. A probe that reproduces training's forest wraps the build
  and walk in autocast.
- Lug ISLA-DTT's first two reads (blocks 0-1) are unused in both seeds.
- On the lug, float32 errors sit below the bf16 read-out noise (about 0.0033), so a bf16 score compresses a float32
  form gap toward 1.
- Operational: `sbatch --export=ALL,X='a,b'` truncates X at the first comma.

## For the book

No standing changes: stage 1 trained nothing. The DTT ledger row could gain one sentence. On the 0.9.0 checkpoints the
tree read acts as a trained constant (the exact read scores 1.8-4.7x worse); the AirfRANS gap to ISLA is a fit bias in
the wake and far field; and the bf16-forest and coherence facts. I can write a NOTEBOOK_DRAFT if you want one.

## On disk

This folder (a few MB). On LHR, `$T/kernel/dttknobs_gap/out` (4.1 GiB) and `out_bf16` (1.3 GiB) hold saved predictions
kept for stage 2's regional comparisons. Every other `$T/kernel/dttknobs_*` dir is under 110 MB.
