# NOTEBOOK_DRAFT: DTT-KNOBS stage 2, arms 1 to 5 (draft for the supervisor; DK-NOTEBOOK2, 2026-10-26 06:40 EDT)

Revised after NOTEBOOK2_REVIEW.md, 2026-10-26 07:03 EDT (DK-NOTEBOOK2-REV; the previous version is
`NOTEBOOK_DRAFT_STAGE2.v1.md`, md5 59af6071; what changed, by item, is in `NOTEBOOK2_REVISION_NOTE.md`).

Sources, all in `scratch/isla-dtt-knobs-2026-10-23/`: `PREREG.md` (md5 39d8272f: the registered rules, the supervisor's
amendment before submission and the coordinator's 17:35 EDT clarification in section 9), `RESULTS.md` (md5 e47571da, the
current version; its three sections each second-read and corrected; it differs from ad1ec340 only in the knob-rule
sentence of R2 and two "no rung of the theta ladder" wordings), the reviews `RESULTS_R5_REVIEW.md`,
`RESULTS_A14_REVIEW.md` and `RESULTS_EXACT_REVIEW.md` (all "confirmed with corrections"; only the corrected wordings are
used), `STAGE3_PROPOSAL.md` (md5 67304a2c) and `reduce/out_final/reduce_s2.md` (md5 adfd6f8a; `.json` e289de96). The
book was read at origin/psharpe/isla fa1cab63d (fetched 06:35 EDT); the head at 07:00 EDT, 9eb786e20, changes no book
file. Nothing in the book was edited.

The book already holds stage 1 (`{#sec-nb-dtt-knobs-stage1}`, from `NOTEBOOK_DRAFT_STAGE1.md`, whose draft label was
`-s1`), the registration (`{#sec-nb-dtt-knobs-prereg}`) and R5, arm 6 (`{#sec-nb-dtt-knobs-r5}`). This entry therefore
covers arms 1 to 5 (R1 to R4) and refers to R5 without repeating it. It goes at the end of `18-notebook.qmd`, and the
chapter changes follow it.

<!-- BEGIN ENTRY -->

## 2026-10-25 to 2026-10-26 — DTT-KNOBS stage 2, the retrained read: ISLA-DTT retrained on AirfRANS at θ = 2, 0.5 and 0, with a read after every second block, and with a basis of 8; only the basis of 8 crosses the registered bar (it harms eddy viscosity, 1.287); θ reads as a pure cost knob, free to θ = 2; a finer read narrows the far-field gap; the read after every second block is a scope-freeze candidate as a free cost cut, pending Peter's word and a DrivAerML Volume check {#sec-nb-dtt-knobs-stage2}

**Question.** Stage 1 found that on checkpoints trained at θ = 1 every inference-time change of ISLA-DTT's read is worse
(@sec-nb-dtt-knobs-stage1), so only retraining can price a setting. Stage 2 asks: does a cheaper trained read predict as
well as the standing (Q1)? Retrained at θ = 2, 1, 0.5 and 0, does the error converge as θ tightens, or is θ a pure cost
knob (Q2, Peter's knob rule)? Does a finer read narrow the gap to interacting ISLA in the wake and beyond one chord
(Q3)? The rules are registered in @sec-nb-dtt-knobs-prereg (PREREG md5 39d8272f). Q4 (DTT-NOSC-STD) was read on
2026-10-24 (@sec-nb-dtt-knobs-r5).

**Design.** Arms 1 to 5 each make one change and train eager against r090eag_af_dtt (the standing's eager-trained ISLA-DTT
pair): K-THETA2 (θ = 2), K-THETA05 (θ = 0.5), K-B1 (the read after 1-based blocks 2, 4, ..., 12 only), K-BASIS8 (8 kernel
basis functions, not 16) and K-EXACT (θ = 0, the exact read). Arm 6 (DTT-NOSC-STD) is fix-trained against
r090fix_af_dtt. Each arm
has two seeds (42, 43) and 500 epochs. Every arm trained on the code its control trained with, the 0.9.0 snapshot
`code_cmpaf_fix` with `recipe_r090` (arm 6 on `code_att090fix` with `recipe_att090b`). The lanes started at 05:29 EDT
2026-10-24, before DTT-FAST and DEFAULT-DTT were merged, and they keep stage 1's two code facts, as the control does.
Scoring: float32, eager, the 200 `full_test` cases, native and posed; scorers 382281 (arms 1 to 4) and 400809
(K-EXACT), regional jobs 382386, 382387 and 400920. Every section-4 check passed on all 20 rows of arms 1 to 5. The
control reproduces its registered seeds to 1.8e-6, and the regional gate passes on every run.

**Result 1: R1 and R3.** Arm ÷ control, ratio of arithmetic seed means, native, [seeds 42, 43] where cited. "Flag" is
the sub-bar flag: a shift of more than 3 % with both arm seeds beyond both control seeds.

| arm | pressure | in-plane velocity | eddy viscosity | R3 reading (registered) |
|---|---|---|---|---|
| K-THETA2 | 1.071 [1.063, 1.079], no change, flag up | 1.040, no change | 0.980, no change | not a free cost cut (up flag) |
| K-THETA05 | 0.936, no change, flag down | 0.950, no change, flag down | 0.932, no change, flag down | not a candidate (no field helps) |
| K-B1 | 1.013, no change | 0.988, no change | 0.991, no change | **candidate for the scope freeze (free cost cut)**; architecture change, so Peter's word too |
| K-BASIS8 | 1.054, no change | 1.027, no change | **1.287 [1.297, 1.278], harms**, flag up | not a free cost cut (harms) |
| K-EXACT | 0.957 [0.956, 0.957], no change, flag down | 0.976 [0.933, 1.022], no change | 0.879 [0.756, 1.003], no change | not a candidate (no field helps) |

: DTT-KNOBS stage 2, R1 and R3 (results/campaigns_2026-10-23/dtt_knobs_s2/reduce_s2.md, "R1" and "R3"). {#tbl-nb-dtt-knobs-s2-r1}

Posed gives the same labels everywhere (the same runs, not an independent number). Arm ÷ interacting ISLA, against the
control's 0.994 / 1.129 / 1.342: K-THETA05 0.931 / 1.072 / 1.251, K-EXACT 0.951 / 1.102 / 1.179, K-B1 1.007 / 1.115 / 1.330,
K-THETA2 1.065 / 1.173 / 1.315, K-BASIS8 1.048 / 1.159 / 1.727.

**Result 2: R2, the ladder.** D is the ln of the arm's ratio to the control (θ = 1, D = 0).

| field | D at θ = 2, 0.5, 0 | adjacent step that rises | registered label |
|---|---|---|---|
| pressure | +0.069, −0.066, −0.044 | (0.5 to 0): +0.022 against 2 SE 0.009 | pure cost knob |
| in-plane velocity | +0.039, −0.052, −0.024 | none | pure cost knob, free to θ = 2 |
| eddy viscosity | −0.020, −0.070, −0.129 | none | pure cost knob, free to θ = 2 |

: DTT-KNOBS stage 2, R2 (reduce_s2.md, "R2"). {#tbl-nb-dtt-knobs-s2-r2}

Overall the registered reading is **pure cost knob, free to θ = 2**: all three fields carry the label, and two carry the
sub-label. Pressure's rising step enters only the "converges" test. The down flags of K-THETA05 and K-EXACT enter no
label, because a flag changes no label. K-THETA2's up flag is part of the sub-label's registered test, which is why
pressure does not carry "free to θ = 2". In the knob rule's terms, R2 reads pure cost knob (registered gloss: the
approximation costs nothing at θ ≤ 1); its "converges" label does not apply (the exact-trained model is not better past
the bar, and pressure rises from θ = 0.5 to 0). R2 is a retrained-ladder test; it does not test the book's knob-rule
credit for pose invariance, which rests on the read's error at fixed weights vanishing as θ → 0 (stage 1 measured that
error, 3.9-27 % per block at θ = 1, falling with θ).

**Result 3: R4 and Q3, the far-field gap** (pooled e_R, as registered; D_R against 2 SE_R). K-THETA05 narrows in-plane
velocity in the wake (−0.122 against 0.112; seed ratios 0.899 / 0.870), and K-EXACT narrows eddy viscosity beyond one
chord (−0.248 against 0.213; 0.816 / 0.751). The other eight cells read level, and no arm widens a cell. DK-GAP's
case-normalized statistic, beside, gives the same ten labels. So Q3 reads as registered: **a finer read narrows the
far-field gap** (from K-THETA05 and K-EXACT).

**Predictions.** P1 (K-THETA2 no harm, 0.45) held; P2 (K-THETA2 up flag, 0.60) held; P3 (K-THETA05 helps, 0.20) failed;
P4 (K-B1 no harm, 0.40) held; P5 (K-BASIS8 a candidate, 0.45) failed; P6 (K-EXACT helps, 0.25) failed; P7 (K-EXACT harms,
0.20) did not occur; P8 (R2 pure cost knob, 0.45) held; P9 (a finer read narrows the gap, 0.25) held; P11 (K-EXACT's
smoke above 300 GPU-h) held at 310, and under the amended 400 gate the arm ran.

**Beside the rules (not verdicts).**
- *The ladder in plain terms.* No rung of the θ ladder moves a field past the registered 10 % bar. Within the bar, pressure
  falls a resolved 6 to 7 % per rung from θ = 2 to 0.5. It then rises a resolved 2 % to the exact read: the two arms'
  pressure seeds agree within 1 %. So on pressure the most accurate rung is θ = 0.5, not the exact read. Velocity and
  eddy viscosity move by less than 2 SE at every rung. K-THETA05's gains on them are seed-separated, and on eddy viscosity
  the exact read has the lowest mean (0.879), with seeds 0.756 and 1.003. E5's earlier ladder on older code
  (research/dtt/REPORT.md) put θ = 0.5 ÷ θ = 1 at 0.95 / 0.99 / 1.00, inside its seed spread.
- *K-THETA05 does not gain everywhere.* Its case means sit below the control's on every field, in both seeds, but it
  wins only 68 to 75 % of cases. Its eddy-viscosity error is higher in the outer zone (1.10 / 1.13) and in the 0.01 to 0.1
  chord band (1.12 / 1.07). In the wake, pressure (1.02 / 0.99) and eddy viscosity (1.02 / 1.01) are level.
- *Each finer read narrows a different cell.* The exact read leaves wake velocity level (1.001 / 1.000; ÷ ISLA 1.97 /
  1.96). In DK-GAP's form, ISLA-DTT ÷ ISLA beyond one chord falls from 1.48 / 2.47 to 1.15 / 1.10, still 10 to 15 % above
  interacting ISLA. Seed 43's 0.45 of the control in that form rests on a few cases, and the registered narrowing clears
  2 SE_R by 16 %. K-THETA05's wake-velocity gap ÷ ISLA is 1.73 / 1.68 against the control's 1.95 / 1.99 (DK-GAP's form).
- *K-B1.* Beside R1, its eddy viscosity beyond one chord is worse in both seeds (1.24 / 1.13; D_R +0.167 against 0.211,
  level). It saves 5.0 % of the AirfRANS step (0.887 against 0.934 s) and 5.7 to 6.1 % of the GPU-h against the arms
  trained alongside it. Stage 1 put the 12 per-block reads at 9 % of the AirfRANS step (a lower bound) and gave no
  car-scale share, so K-B1's saving on the car is not known.
- *Basis 8 is not slack for eddy viscosity in training.* At inference stage 1 put it at 1.07 / 1.34 on eddy viscosity.
  Retraining recovered pressure and velocity but not eddy viscosity, and the arm saves no time (0.934 s).
- *Cost of the read.* θ = 2 saves no AirfRANS step time (0.935 against 0.934 s), and θ = 0.5 adds 0.5 % (0.939 s). The
  exact read steps 1.354 s (1.45 times, with 1,007 pairs per point against 21.6) at 85.3 GB peak, against the other eager
  arms' 47.6 to 50.4 GB. On the car, stage 1's cost model measured the code before DTT-FAST: there θ = 0.5 reads about
  7.0 times θ = 1's pairs, and 400,000 points fit a GB300 only at θ ≥ 1.5. DTT-FAST now trains 400,000 whole-boundary
  points at θ = 1 in 1,792 ms and 212 GiB. On the current code, θ = 2's case therefore rests on step time, not on the
  memory fit, and θ = 0.5's memory fit is not measured.

**Consequence.** Nothing is adopted, and ISLA-DTT keeps θ = 1, 16 basis functions and a read after every block. K-B1 is
the only arm R3 makes a candidate for the scope freeze. Stage 3, K-B1 on the DrivAerML Volume task with the whole boundary
and DTT-FAST's read, two seeds against a fresh control pair, is proposed for approval (STAGE3_PROPOSAL.md: about 520
GPU-h, or about 260 if a current-code control pair appears). It is a new lane's work and is not submitted, and K-B1's
adoption also needs Peter's word. A basis of 8 goes to the scope freeze as a trade or is dropped, and so does θ = 2, whose
case is the car's step time. The tighter reads gain below the bar, so R3 does not make them candidates.

**Cost.** Stage 2 used 1,311.4 GPU-h of its 1,450 cap: arms 1 to 4 and 6 used 993.0 of 1,050, and K-EXACT used 318.4 of
400. The campaign used about 1,333 GPU-h.

**For the book.** No standing changes: the standings are the r090eag rows, and no arm replaces one. The chapter changes:
05 states what the retrained settings measure. 12's gap paragraph says that a finer read narrows the gap. 12's scales
sentence and 21's scales row are scoped to pressure. 20's open question becomes the car-scale check of K-B1, with
θ = 2's unmeasured car-scale step saving beside it, and its In-flight callout goes.
The ledger rows DTT, DTT-KNOBS and ATTRIB-090 gain stage 2 (records copied to results/campaigns_2026-10-23/dtt_knobs_s2/).

*Arms 1 to 4 read by the coordinator at 09:36 EDT 2026-10-25 and corrected at 10:08 after RESULTS_A14_REVIEW.md; K-EXACT,
R2 and Q3 read at 06:27 EDT 2026-10-26 and corrected at 06:34 after RESULTS_EXACT_REVIEW.md; this entry drafted by
DK-NOTEBOOK2 (NOTEBOOK_DRAFT_STAGE2.md). Read by the supervisor on <date> (to be filled).*

<!-- END ENTRY -->

## Chapter changes

All numbers are typed here for review. In the chapter, STYLE.md wants each one computed from a file in `results/` as
the page renders. The supervisor first copies the records to `results/campaigns_2026-10-23/dtt_knobs_s2/`:
`RESULTS.md`, `RESULTS_A14_REVIEW.md`, `RESULTS_EXACT_REVIEW.md`, `reduce/out_final/reduce_s2.md` and `.json`,
`STAGE3_PROPOSAL.md` and this draft. Then add `"dtt_knobs_s2": "campaigns_2026-10-23"` to `records.py`'s topic map.
`dtt_knobs_r5/` stays as it is, since 12 and 20 parse its R5 section. Chapter prose carries no dates, run identifiers or
node codes, and it cites no notebook entry. Every quoted passage below was found with `git grep -F` at fa1cab63d.

**In-flight callouts.** One is removed (change 8) and none is added. The new 12 sentence sits inside the paragraph that
the existing far-field callout (12, line 640) already covers, and nothing running replaces the new 05 or 20 numbers.
Stage 3 gets a callout only when it launches (text in change 8).

1. **14-program-status.qmd, ledger row `DTT` (line 563), fifth column**, after "... (1.53 / 1.98 at theta 1, cause
   inferred, not tested); both handed to DTT-POSE". Append:
   "; then retrained on AirfRANS (DTT-KNOBS stage 2, eager against r090eag_af_dtt on the 0.9.0 code, two seeds each,
   @sec-nb-dtt-knobs-stage2): R2 reads theta a pure cost knob, free to theta 2 (velocity and eddy viscosity; pressure pays
   a seed-separated 7 % at theta 2, its up flag), since neither theta 0.5 (0.936 / 0.950 / 0.932) nor the exact read
   (0.957 / 0.976 / 0.879) helps any field past the 10 % bar, so its "converges" label does not apply (registered gloss
   of the reading: the approximation costs nothing at theta <= 1; pressure also rises from theta 0.5 to 0); R2 is a
   retrained-ladder test and does not test the knob-rule credit for pose invariance, which rests on the read's error at
   fixed weights vanishing as theta -> 0 (stage 1 measured that error, 3.9-27 % per block at theta 1, falling with
   theta); Q3: a finer read narrows the far-field gap (theta 0.5 in-plane velocity in the wake, the exact read
   eddy viscosity beyond one chord); a read after every second block reads no change and no flag (1.013 / 0.988 /
   0.991), a scope-freeze candidate as a free cost cut pending Peter's word and stage 3 on DrivAerML Volume; a basis of 8
   harms eddy viscosity (1.287); without the scales and with the inlet speed standardized (DTT-NOSC-STD, fix-trained
   against r090fix_af_dtt) ISLA-DTT has 1.208 / 1.203 / 1.177 times the error (about 20 % worse), so the scales
   carry it as far as two seeds can show, and on pressure ISLA-DTT uses the scales where interacting ISLA does not
   (@sec-nb-dtt-knobs-r5)".
   Sixth column: after `@sec-nb-dtt-knobs-stage1;` add ` @sec-nb-dtt-knobs-r5; @sec-nb-dtt-knobs-stage2;`. Sources:
   RESULTS.md (all three sections), reduce_s2.md. Callouts: none.

2. **14-program-status.qmd, ledger row `DTT-KNOBS` (line 715).** Second column: replace "RUNNING (launched by" with
   "DONE 2026-10-26 (launched by". Whether the lane closes now, or at its registry row's removal, is the coordinator's
   call. In the same column replace "Stage 2 RUNNING:" with "Stage 2:" and "12 lanes RUNNING since 05:29 EDT" with
   "12 lanes launched at 05:29 EDT", so the row does not call stage 2 running and done at once ("Stage 2 RUNNING:" also
   occurs in row HL-DUP, line 704, which stays). Then replace the closing "133.3 GPU-h; @sec-nb-dtt-knobs-r5)" with:
   "133.3 GPU-h; @sec-nb-dtt-knobs-r5. Arms 1 to 4 read 2026-10-25 09:36 EDT (corrected 10:08, RESULTS_A14_REVIEW.md);
   K-EXACT, R2 and Q3 read 2026-10-26 06:27 EDT (corrected 06:34, RESULTS_EXACT_REVIEW.md); every section-4 check
   passed: R1 no change on every field for theta 2 (pressure 1.071, up flag), theta 0.5 (0.936 / 0.950 / 0.932, down
   flags on all three), every second block (1.013 / 0.988 / 0.991) and the exact read (0.957 / 0.976 / 0.879, pressure
   down flag); a basis of 8 harms eddy viscosity (1.287); R2 pure cost knob, free to theta 2; R3: only the read after
   every second block is a scope-freeze candidate (free cost cut; Peter's word needed, an architecture change); R4 /
   Q3: a finer read narrows the far-field gap; P1, P2, P4, P8, P9 and P11 held, P3, P5 and P6 failed, P7 did not occur;
   stage 2 used 1,311.4 of 1,450 GPU-h; stage 3, the read after every second block on DrivAerML Volume (about 520 GPU-h),
   proposed for approval, nothing built; nothing adopted; @sec-nb-dtt-knobs-stage2)".
   Sixth column: replace "@sec-nb-dtt-knobs-prereg" (at the row's end) with "@sec-nb-dtt-knobs-prereg;
   @sec-nb-dtt-knobs-r5; @sec-nb-dtt-knobs-stage2". Sources: RESULTS.md, STAGE3_PROPOSAL.md. Callouts: none.

3. **14-program-status.qmd, ledger row `ATTRIB-090` (line 686), second column.** Replace "where it lies is not
   attributed; @sec-nb-attrib090-stage2)" with "where it lies is not attributed by this study; 2026-10-24: DTT-KNOBS R5
   (DTT-NOSC-STD, ISLA-DTT without the scales and with the inlet speed standardized, ÷ B 1.208 / 1.203 / 1.177) places
   it in the scales on pressure and eddy viscosity, as far as two seeds can show, with no sign that it is an interaction
   with the raw speed; @sec-nb-attrib090-stage2; @sec-nb-dtt-knobs-r5)". Source: RESULTS.md, R5, "Registered
   readings" (the corrected wording of RESULTS_R5_REVIEW.md item 3). Callouts: none.

4. **05-forward-pass.qmd, @sec-arch-tree-read, the paragraph "$\theta$ is a constant of the class"** (lines 2121 to
   2123). Replace "Whether a model trained at a smaller $\theta$ is more accurate is open (@sec-oq-dtt-theta)." with:
   "Trained at each setting on the AirfRANS interior task, without compilation, two seeds each, the opening angle is a
   cost setting there. The rule registered for the comparison reads $\theta$ as a pure cost knob, free up to
   $\theta = 2$ on velocity and eddy viscosity. Models trained at $\theta = 0.5$ and with the exact read are not more
   accurate than the $\theta = 1$ model past its 10 % bar on any field (0.936 / 0.950 / 0.932 and 0.957 / 0.976 / 0.879,
   pressure / velocity / eddy viscosity), and one trained at $\theta = 2$ is not less accurate past it, though its
   pressure error is 6 and 8 % higher (seeds 42 and 43). Within the bar, $\theta = 0.5$ lowers every field's mean error
   by 5 to 7 %, with both seeds lower. The finer reads act in the far field: $\theta = 0.5$ lowers the in-plane velocity error in the wake, and
   the exact read the eddy-viscosity error beyond one chord (@sec-airfrans-dtt-reference). The exact read costs 1.45
   times the $\theta = 1$ step, and $\theta = 2$ saves no step time on AirfRANS. A read after every second layer moves no
   field (1.013 / 0.988 / 0.991) and saves 5 % of the step. A kernel basis of 8 functions in place of 16 raises the
   eddy-viscosity error 1.29 times."
   Keep the passage's following sentence ("The tree's error is also what moves ISLA-DTT ..."). Sources:
   `dtt_knobs_s2/reduce_s2.json` (R1 ratios, R2 labels), `dtt_knobs_s2/RESULTS.md` (K-EXACT "Cost"; arms 1 to 4 "Cost per
   arm"). Callouts: none.
   For the supervisor: STYLE.md's knob rule credits ISLA-DTT's pose invariance as exact when θ → 0, "and the model is
   run where that error is small". R2 reads pure cost knob (registered gloss: the approximation costs nothing at
   θ ≤ 1); its "converges" label does not apply (the exact-trained model is not better past the bar, and pressure rises
   from θ = 0.5 to 0). R2 is a retrained-ladder test; it does not test the book's knob-rule credit for pose invariance,
   which rests on the read's error at fixed weights vanishing as θ → 0 (stage 1 measured that error, 3.9-27 % per
   block at θ = 1, falling with θ). The retrained error at θ = 1 is within the bar of the exact-trained model's on
   every field (the exact-trained pressure error is a resolved 0.957 of it). Whether the credit stands in 02, 03, 04,
   05, 06, 09, 12, 20, 21 and 22 is your call or Peter's, and I draft no change to it. NOTEBOOK2_REVIEW.md's "knob rule"
   section, in short:
   - STYLE.md:225-236 credits a property exact as a knob goes to zero, "with an error that converges to zero, and the
     model is run where that error is small". Every chapter that credits the pose invariance credits the read's error,
     or a regrouping's, at fixed weights, "which vanishes as θ → 0": 02:733, 03:1620, 04:103, 04:723, 05:2029,
     06:155, 09:835, 12:479, 12:946, 20:1156, 21:190, 21:279 and 22:436; 05:2017-2024 adds the exact-read test at
     θ = 0 and the float64 pose test.
   - Only 20:1228-1231 ties the rule to retraining ("if the error falls toward the exact-trained model's as θ
     tightens, the tree read is a convergent approximation; if the exact-trained model is no better, θ is a cost
     setting on AirfRANS"), and change 8 cuts it. No passage makes the credit depend on R2.
   - R2 measures neither the property (the posed scores are a beside reading) nor the read's convergence at fixed
     weights, which stage 1 measured (05:2113-2116: 3.9 % to 27 % per layer at θ = 1, falling as θ^0.33 to θ^0.69).
   - The open point: RESULTS.md ad1ec340 (lines 168-170) and this draft's first version wrote that the tree read "does
     not earn credit as a convergent approximation", which negates Q2's parenthetical for "converges" and in a notebook
     or ledger line can be read as withdrawing STYLE.md's credit. The current RESULTS.md (e47571da, lines 168-173) and
     this version give the registered reading and its scope instead.

5. **12-airfrans.qmd, cell `para-airfrans-dtt-gap`** (lines 636 to 637). Replace "Whether a model trained with a finer
   read narrows it is open (@sec-oq-dtt-theta)." with:
   "A finer read in training narrows it in part, each finer setting in a different region, by the rule registered for
   it. Trained at $\theta = 0.5$, ISLA-DTT's in-plane velocity error in the wake is 0.90 and 0.87 of the $\theta = 1$
   model's (seeds 42 and 43, pooled over the cases); per case, the form of the figures above, it is 1.73 and 1.68 times
   ISLA's against 1.95 and 1.99, and its eddy viscosity beyond one chord does not move measurably. Trained with the exact
   read, its eddy-viscosity error beyond one chord is 0.82 and 0.75 of the $\theta = 1$ model's (pooled); per case, 1.15
   and 1.10 times ISLA's against 1.48 and 2.47, the 1.10 resting on a few cases. Its wake velocity is unchanged (per
   case, 1.97 and 1.96 times ISLA's). Over the whole domain neither change passes the 10 % bar registered for the
   comparison (@sec-arch-tree-read)."
   "Pooled" is R4's registered e_R; "per case" is DK-GAP's case-normalized form, which normalizes each case by its own
   target energy and is the form of this paragraph's 1.95 / 1.99 and 1.48 / 2.47. The two must not be equated: in the
   per-case form seed 43's exact-read model is 0.45 of the control beyond one chord, against 0.75 pooled, so that 1.10
   rests on a few cases (RESULTS_EXACT_REVIEW.md item 4; reduce_s2.md, "K-EXACT: wall_band"; K-THETA05's per-case ratios
   to the control in the wake are 0.88 / 0.84, "K-THETA05: zone"). Add `dtt_knobs_s2/RESULTS.md` (sections "Arms 1-4", R4,
   and "K-EXACT", "Q3 and R4") and
   `dtt_knobs_s2/reduce_s2.md` (R4 and the per-zone tables) to the cell's Data line. Callouts: the far-field callout
   after this cell (line 640) stays and covers this sentence. These are airfoil-alone runs too.

6. **12-airfrans.qmd, cell `para-airfrans-dtt-reference`** (lines 575 to 576). Replace "So on AirfRANS ISLA-DTT uses the
   scales where interacting ISLA does not; the difference between the two forms is resolved on pressure" with "So on
   AirfRANS, on pressure, two seeds show that ISLA-DTT uses the scales where interacting ISLA does not: the difference
   between the two forms is resolved there". The passage crosses the join between line 575, a plain string that ends
   "... the difference between ", and line 576, an f-string that begins "the two forms is resolved on pressure
   ({r5_d[0]} ...". Edit both pieces and keep the `f` prefix on the second, which formats `r5_d`: for example line 575
   `"(@sec-airfrans-attribution). So on AirfRANS, on pressure, two seeds show that ISLA-DTT uses the scales where "` and
   line 576 `f"interacting ISLA does not: the difference between the two forms is resolved there ({r5_d[0]} against two
   standard errors of {r5_d[1]}), and on velocity and eddy "`. The rest of the sentence ("({r5_d[0]} against two
   standard errors of {r5_d[1]}), and on
   velocity and eddy viscosity it is within two seeds' noise") stays. Source: RESULTS.md R5, "Beside the rule" ("For the
   scope freeze: on pressure, two seeds show ..."). Callouts: none.

7. **20-limits.qmd, `tbl-open-questions`, the row at line 906, and its cost (lines 885 to 888).** Replace the row with
   `("Does a read after every second layer cost ISLA-DTT nothing on the car, and what does $\\theta = 2$ save there?
   (@sec-oq-dtt-theta)", "ISLA-DTT on the DrivAerML Volume task with the whole boundary, the read after every second
   layer against the read after every layer, two seeds each, under its standing's protocol; the training step at
   $\\theta = 2$ against $\\theta = 1$ on the same task", f"about {fmt_h(q_knobs)}; the $\\theta = 2$ step not priced")`.
   Replace the comment and the three lines
   `KN = ...`, `assert ...` and `q_knobs = ...` with
   `# ISLA-DTT reading after every second layer, on DrivAerML Volume: four runs, as its registration prices them (section 7)`
   and `q_knobs = int(re.search(r"about (\d+) GPU-h for the four runs", " ".join(R.load("dtt_knobs_s2",
   "PREREG.md").split())).group(1))`. That needs PREREG.md (md5 39d8272f) copied into `dtt_knobs_s2/`. Its section 7 is
   unchanged from the copy in `dtt_knobs/`, so `R.load("dtt_knobs", "PREREG.md")` also works. Source: PREREG.md section 7,
   STAGE3_PROPOSAL.md ("about 520 for the four runs"; about 260 if a current-code control pair appears).

8. **20-limits.qmd, @sec-oq-dtt-theta (cell `para-oq-dtt-theta`, lines 1213 to 1237, and the callout at 1240 to
   1244).** Keep the heading and label, since 20, line 1163 and the table refer to it. Replace the printed paragraph with:
   "**The open question is whether a read after every second layer costs ISLA-DTT nothing on the car, and what
   $\theta = 2$ saves there.** ISLA-DTT trains and is scored at $\theta = 1$, with a kernel basis of 16 functions and a
   read after every layer (@sec-arch-tree-read).
   Retrained on the AirfRANS interior task, two seeds each, a read after every second layer moves no field by the rule
   registered for it (1.013 / 0.988 / 0.991, pressure / velocity / eddy viscosity) and shifts none past 3 % in both seeds,
   which makes it a candidate for a free cost cut. The opening angle and the basis are measured there as well: up to
   $\theta = 1$ the opening angle is a cost setting, and 8 basis functions raise the eddy-viscosity error 1.29 times
   (@sec-arch-tree-read). On AirfRANS
   the change saves 5 % of the step. The twelve per-layer reads take at least 9 % of the AirfRANS step, and their share
   of a car's step is not measured, so what it saves where the read's cost lives is unknown. On AirfRANS its eddy-viscosity
   error beyond one chord is 1.24 and 1.13 times the every-layer model's (seeds 42 and 43), within the noise of the two
   seeds. Whether that grows on a larger domain is not measured. Reading after every second layer changes where the model
   reads, an architecture change. Trained at $\theta = 2$, ISLA-DTT moves no field by the same rule, but its pressure
   error is 6 and 8 % higher (seeds 42 and 43), so $\theta = 2$ trades accuracy for cost. It saves no step time on
   AirfRANS (0.935 against 0.934 s). Its saving would lie on the car, where ISLA-DTT trains 400,000
   whole-boundary prediction points at $\theta = 1$ on one GB300 (@sec-using-dtt-cost), so its case there rests on step
   time, which is not measured at $\theta = 2$. What would settle it: ISLA-DTT on the DrivAerML Volume task with the
   whole boundary, reading after every second layer against every layer, two seeds each, under its standing's protocol,
   with the step and memory of both at 100,000, 200,000 and 400,000 prediction points, and the training step at
   $\theta = 2$ against $\theta = 1$ at the same counts. Cost: about {fmt_h(q_knobs)} GPU-hours, including a control
   pair on the current read; the step at $\theta = 2$ is not priced."
   Data line: `dtt_knobs_s2/RESULTS.md` (arms 1 to 4: R1, "K-B1", "Cost per arm", "Car-scale cost of a tighter or
   looser read") and `STAGE3_PROPOSAL.md`; the θ = 2 sentence also cites @sec-using-dtt-cost's DTT-FAST table. The setup
   lines `NAT`, `inf_ratio`, `x0` and `x2` go, since the new paragraph does not use them. Besides the replaced opening,
   "What would settle it" and "Cost" sentences, this cuts five sentences of 20:1222-1237:
   (a) "Read at other settings, a model trained there is less accurate: ... so no setting of the read can be priced on
   these checkpoints" (1223-1226): the inference result, still stated in 05:2108-2112;
   (b) "Not measured: whether a model trained at $\\theta = 2$, ... or with the exact read is more accurate"
   (1226-1228): now measured (change 4);
   (c) "The second also tests Peter's knob rule, ... $\\theta$ is a cost setting on AirfRANS" (1228-1231): the only
   passage that ties the knob rule to retraining; see the supervisor's note under change 4, since cutting it is part of
   the credit question;
   (d) "A finer trained read may also narrow ISLA-DTT's gap to interacting ISLA in the wake and beyond one chord
   (@sec-airfrans-dtt-reference)" (1231-1232): answered by change 5;
   (e) "The cost side is measured at car scale: on the DrivAerML Volume task with the whole boundary, 400,000 prediction
   points fit one GB300 at $\\theta \\geq 1.5$ and not at $\\theta = 1$ (@sec-cost-sweep)" (1232-1234): it describes the
   read before DTT-FAST, which trains 400,000 whole-boundary points at θ = 1 (@sec-using-dtt-cost; RESULTS.md, arms 1
   to 4, "Car-scale cost"; RESULTS_A14_REVIEW.md item 4).
   **Remove the In-flight callout** at lines 1240 to 1244 ("These numbers come from models trained at $\theta = 1$ and
   read at other settings. ISLA-DTT trained on AirfRANS at $\theta = 2$, 0.5 and 0, ..."): its arms are read.
   **To add only if stage 3 launches**, directly after the cell, and to remove when it is read:
   ```markdown
   ::: {.callout-note title="In flight"}
   This question rests on ISLA-DTT reading after every second layer on the AirfRANS interior task alone. The same
   change on the DrivAerML Volume task with the whole boundary is being measured, and the ledger (@sec-ledger) tracks it.
   :::
   ```

9. **20-limits.qmd, cell `para-oq-dtt-frame`** (line 1163). Replace " (seeds 42 / 43), so pricing a small $\\theta$ needs
   a model trained there (@sec-oq-dtt-theta). On the lug the " with " (seeds 42 / 43), so pricing a small $\\theta$ needs
   a model trained there (@sec-arch-tree-read). On the lug the ". After change 8, @sec-oq-dtt-theta asks about the read
   count and θ = 2 on the car, and the retrained results for a small θ are in 05 (change 4). Callouts: none.

10. **21-design.qmd, `tbl-principled-ledger`, the row "Its own physical scales" (lines 247 to 249), its evidence string
   (line 249).** The row sets "the scales add nothing measurable there for interacting ISLA" against "while ISLA-DTT
   without them loses accuracy on every field", the comparison between the forms that RESULTS_R5_REVIEW.md item 2
   withdrew. Replace "while ISLA-DTT without them loses accuracy on every field, its velocity reading flagged
   (@sec-airfrans-dtt-reference);" with "while ISLA-DTT without them loses accuracy on every field, its velocity
   reading flagged, and the difference between the two forms is resolved on pressure alone
   (@sec-airfrans-dtt-reference);". Source: RESULTS.md R5, "Beside the rule" (the plain
   2-SE check resolves the difference on pressure only, 0.186 against 0.161). Callouts: none.

**Not changed, checked.**
- 04's `para-remesh-theta` and 05's inference-θ sentences ("Read at another opening angle, a model trained at θ = 1 is
  less accurate ...") stay: they are true of the θ = 1-trained checkpoints.
- 05's `para-arch-read-constants` (16 basis functions) and the "read after every layer" of @sec-arch-tree-read stay:
  nothing is adopted.
- 20's @sec-oq-scales stays: its "on every field" is R5's registered per-field reading, and 20:979-983 scopes the
  difference between the forms to pressure ("The difference between the two forms is resolved on pressure alone").
  21's scales row keeps its "on every field" too, and change 10 adds the same scope there.
- 12's @sec-airfrans-read-only stays (a smaller instrument), and so do 06's @sec-using-dtt-cost and 19's @sec-cost-sweep
  (DTT-FAST's numbers).
- No standing table or figure changes.
- `git grep` finds no use of "flat basin", "uniformly better" or "first measurement" in any chapter (18 excluded).
