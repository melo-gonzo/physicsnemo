# NOTEBOOK_DRAFT: DTT-KNOBS stage 1 (draft for the supervisor; DK-NOTEBOOK, 2026-10-24 03:00 EDT)

Revised after `NOTEBOOK_REVIEW.md` by DK-NOTEBOOK-REV, 2026-10-24 03:36 EDT (first version `NOTEBOOK_DRAFT_STAGE1.v1.md`;
changes by item in `NOTEBOOK_REVISION_NOTE.md`).

Sources, all in `scratch/isla-dtt-knobs-2026-10-23/`: `STAGE1.md` (md5 1a28dc7c, the authoritative summary, second-read in
`STAGE1_REVIEW.md`), the six child reports `sens/`, `cost/`, `cap/`, `gap/`, `order/`, `treeprec/` `REPORT.md` (each revised
after its `REVIEW.md`), `CODE_FACTS.md`, and the stage-2 pre-registration `PREREG.md` (md5 84a25280). Every number below appears in one of them, in its corrected form. Book read at
origin/psharpe/isla 1e7f03d7d (ISLA 0.11.0). Nothing in the book was edited. The entry goes at the end of `18-notebook.qmd`;
the chapter changes follow it.

<!-- BEGIN ENTRY -->

## 2026-10-24 — DTT-KNOBS stage 1, inference only: on its trained checkpoints ISLA-DTT's opening angle acts as a trained constant (the exact read scores 1.8 to 4.7 times worse than θ = 1), a cheaper read saves time only at car scale (400,000 DrivAerML Volume points fit one GB300 at θ ≥ 1.5 and not at θ = 1, where the memory fit puts the limit at 349,000), the read's inference-time slack is in its kernel basis, and the gap to interacting ISLA is one of fit that no inference-time setting of the read explains {#sec-nb-dtt-knobs-s1}

**Question.** Peter wants ISLA-DTT practical in the long term (his ruling of 23:26 EDT 2026-10-23, @sec-nb-surface-bc).
Which of its boundary read's settings trade accuracy for cost, and where does its accuracy gap to interacting ISLA come
from? Stage 1 trains nothing: it reads the trained checkpoints at other settings and prices the read. Nothing is adopted.

**Design** (scratch/isla-dtt-knobs-2026-10-23/, STAGE1.md and six child reports). The checkpoints are the AirfRANS pair
r090eag_af_dtt (eager-trained, the standing's ISLA-DTT arm) and the lug pair r090fix_lug_dtt, seeds 42 and 43, each trained
at θ = 1 with leaves of 4 cells, read against r090eag_af_isla and r090fix_lug_isla. Every read ran on the code its run
trained with, in float32, on the standing's sets in the scorer's order (AirfRANS `full_test`, 200 cases; lug `test`, 600),
native and posed; a setting changed only on a copy of the model in the probe's process. Gate: each harness reproduced the
standing's per-run scores before any other number was read, and every scoring part passed (AirfRANS within 1.8e-6
relative; lug 0.00221933 / 0.00183431 against the standing's 0.002219 / 0.001834). The cost model, which scores nothing,
ran random weights and bfloat16 training steps on psharpe/isla-geo-bundle code with the whole boundary. There was no
acceptance bar beyond the gate.
21.76 GPU-h of the 30-GPU-h cap on LHR; production jobs 363160 and 364211 (sensitivity), 363435 and 364009 (cost), 363020
and 363328 (capacity), 363336, 363640 and 363641 (read order), 363147, 363156 and 363365 (gap), 364468 (forest precision).

**Result 1: θ acts as a trained constant.** Read natively, both datasets score best at θ = 1 on every field and both
seeds, the error rises on both sides of θ = 1, and the exact read is the worst native setting (sens/sens_table.md):

| θ | AirfRANS pressure | AirfRANS velocity (in-plane) | AirfRANS eddy viscosity | lug von Mises |
|---|---|---|---|---|
| 0 (exact) | 2.71 / 2.36 | 1.79 / 2.04 | 3.54 / 4.68 | 4.62 / 3.47 |
| 0.25 | 2.07 / 1.86 | 1.58 / 1.73 | 3.17 / 3.98 | 2.90 / 2.53 |
| 0.5 | 1.41 / 1.40 | 1.18 / 1.23 | 1.27 / 1.44 | 1.97 / 2.00 |
| 0.75 | 1.07 / 1.03 | 1.03 / 1.05 | 1.04 / 1.08 | 1.24 / 1.29 |
| 1.25 | 1.18 / 1.20 | 1.04 / 1.07 | 1.07 / 1.07 | 1.39 / 1.41 |
| 1.5 | 1.35 / 1.35 | 1.10 / 1.14 | 1.16 / 1.17 | 2.12 / 1.94 |
| 2 | 1.86 / 1.91 | 1.30 / 1.33 | 1.38 / 1.44 | 3.20 / 2.79 |

Each entry is the native error at θ ÷ the same checkpoint's error at θ = 1, seeds 42 / 43, float32, eager. This is E2
seen from the other side: checkpoints trained with the dense read moved 2 to 20 times when read through the tree, and these
tree-trained ones move 1.8 to 4.7 times when read exactly. So the knob rule's test, whether the error converges as θ
tightens, needs models trained at each θ. The leaf size is not shown to be trained in: leaves of 2 or 8 cells change
AirfRANS by at most 0.1 %; on the lug, leaf 8 changes it by at most 0.8 % and leaf 2 by +2 % (seed 42) and +33 % (seed
43). Pairs per point at θ = 1 / 2 / 0.5 / 0 are 21.6 / 15.0 / 34.7 / 1,007 on AirfRANS and 118 / 54 / 300 /
6,525 on the lug, yet the eager forward time is flat for every θ > 0 (AirfRANS 475 to 498 ms, lug 195 to 252 ms; the exact
read 751 and 353 ms).

**Result 2: a cheaper read pays only at car scale** (cost/cost_table.md). Pairs per point follow A θ^-p leaf^-r (log2 N −
c) for cluster nodes plus a near-cell term; on DrivAerML p = 0.91, and on the volume datasets nodes are 93 to 99 % of
pairs at leaf 4. The step costs 5.16 ms and 0.265 GiB per million pairs per block on DrivAerML (held-out memory error at
most 0.4 %). At 400,000 prediction points on the DrivAerML Volume task with the whole boundary, a GB300 (274 GiB usable)
ran θ = 3 (1,837 ms / 248 GiB), θ = 2 (2,070 / 260) and θ = 1.5 (2,734 / 273, at the limit); θ = 1 ran out of memory, and
the memory fit puts θ = 1.25 at 285 GiB. By that fit, the largest counts that fit are 349,000 at θ = 1, 380,000 at 1.25,
431,000 at 2 and 462,000 at 3 (projections, not runs). The settings that fit take 2.3 to 3.4 times interacting ISLA's
796 ms at 400,000 (measured with the car alone). The normal-coherence test is vacuous
at θ ≥ 0.919 in 3D and 0.765 in 2D, because inside one sign octant |mean n| ≥ 1/√d; no node is refused at θ ≥ 1
on any dataset; below that bound θ tightens two tests at once.

**Result 3: at inference, the read's slack is in the kernel basis** (cap/cap_table.md; inference ablations of models
trained with every read and head, not retrains). The 16-function basis has effective rank 4 on AirfRANS and 1.5 on the
lug. Projected onto 8 directions it costs 1.11 / 1.28 times on AirfRANS and 1.9 / 3.6 times on the lug. Keeping the 4
strongest heads costs 6 to 9 times, although they carry 55 to 100 % of each block's update energy. Every read carries
part of the trained model, except the lug's first two (blocks 0 and 1), which are unused in both seeds. Reading after
every second block costs 15.5 / 11.5 times on AirfRANS and 21 / 31 times on the lug, the best of the forms with at most 6
reads. The 12 per-block reads alone are 9 % of the AirfRANS step and 26 % of the lug's (eager, with the shared forest, walk
and basis still running, so lower bounds).

**Result 4: at θ = 1 the read is far from the exact read** (order/order_table.md, AirfRANS, 50 cases per seed). Each
block's read update differs from the exact read on the same states by 3.9 to 27 % (median over points), mostly through the
clusters' mean values. The difference falls slowly where the model reads, as θ^0.33 to θ^0.69 from θ = 2 to 0.5,
and the median reaches 1 % only at θ 0.08 to 0.26 (pooled over seeds). A second-order cluster correction (H1) has a worse
median than the monopole's at every θ ≥ 1 in every block (1.01 to 3.81 times at θ = 1) and a worse p95 at every θ ≥ 0.5;
end to end at θ = 1 it scores 5.6 to 24 times worse. No H1 kernel is worth building unless a model retrained with it beats
the monopole at θ ≥ 1.

**Result 5: the gap to interacting ISLA is one of fit** (gap/gap_table.md, treeprec/treeprec_table.md). ISLA-DTT ÷ ISLA on
the first 200 / 600 training cases is 0.987 / 1.083 / 1.413 on AirfRANS (pressure / velocity / eddy viscosity) and 1.434 on
the lug, against 0.994 / 1.129 / 1.342 and 1.394 on test; the AirfRANS training subset covers Re 2 to 3e6 only, and the
test's low-Re quartile reads alike. On AirfRANS the gap is a smooth bias, not per-point tree noise, except eddy viscosity
beyond 0.1 chord, where its gap sits (roughness 0.076 / 0.112): in-plane velocity is 1.95 / 1.99 times ISLA's error in
the wake (seeds 42 / 43), and eddy viscosity 1.48 / 2.47 times beyond one chord, where ISLA-DTT's own two seeds differ by
1.72 times. The eddy-viscosity gap's share in the wake rests on seed 43 (wake ratios 1.02 / 1.35). Near the wall eddy
viscosity is 1.20 times but carries at most 5 % of the gap. On the lug the float32 gap covers 96 % of cases and sits 90 %
on surface nodes. Scored in bfloat16 the lug's two forms read level (1.043 [1.096, 0.989]), mostly because bfloat16
read-out noise (about 0.0033, twice the float32 errors) compresses the gap toward 1; ISLA-DTT is still worse in 73 % of
cases and on the surface (1.11). This says nothing on whether bfloat16 training causes the gap.
Ruled out as causes: every inference-time θ (beyond one chord the gap is smallest at θ = 1, 1.41 / 1.72 / 2.05, against
1.64 / 1.91 / 2.22 at 0.5); the bfloat16 training forest, whose reading widens the gap slightly (lug 1.394 to 1.399,
AirfRANS eddy viscosity 1.342 to 1.351); and the stored signed distance and the old query embedding (ATTRIB-090 §3c).
C_f = 0.131 / 0.121 (pressure / eddy viscosity) of ISLA-DTT's bundle cost stays unattributed; DK-GAP's reading that the
gap is mostly not a bundle effect is a reading, since C_f on eddy viscosity could be about 40 % of log(1.342) = 0.294.
Whether a retrained finer read closes the gap is untested.

**Two code facts found on the way** (CODE_FACTS.md; handed to DTT-POSE, no fix proposed). Under bfloat16 training the
forest is built from bfloat16-rounded coordinates: `points @ frame` in `BoundaryForest.build` and `point_pairs` runs under
the caller's autocast, so the trees' coordinates, boxes and node measures are bfloat16, while the box diameters and the
walk's box table stay float32. Every float32 score therefore reads a different forest: 82 % of AirfRANS points and 99 % of
lug points get a different list, and native error moves by 0.2 to 0.7 % (pressure level). The bfloat16 forest is not
pose-invariant: on the lug's posed set it doubles the error (1.90 [1.95, 1.85]); which of two candidate mechanisms
dominates is untested. Second, the lug's posed float32 score differs from native at every θ > 0 (1.53 / 1.98 at θ = 1,
1.0000 at θ = 0, with 1.29 times the pairs posed). 84 % of the lug's cells have a normal component below 1e-5 in the
canonical frame (measured under bfloat16 autocast in job 364211; the float32 share was not measured), so their sign
octant rests on rounding residues. That the pose flips those signs and so builds a different forest is inferred, not
tested: posed octant labels were not compared, no tie-break was tried, and the frame's own near-ties (skews about 1.2e-6
against the 1e-9 threshold) are not excluded.

**Consequence.** An inference probe cannot price a retrain: θ is trained in, and the capacity ablations rank models
trained with every read and head. Stage 2 is pre-registered and not run: retrained AirfRANS arms at θ = 2 and θ = 0.5,
reads after every second block, a kernel basis of 8, the exact read, and ISLA-DTT without scales on a standardized inlet
speed, each against the standing's pair (scratch/isla-dtt-knobs-2026-10-23/PREREG.md, md5 84a25280: about 1,010 GPU-h
without the exact read, which adds about 330 to 390 GPU-h and is submitted only if its smoke prices it at 300 or less).
The θ = 2, 1, 0.5, 0 arms form the knob rule's ladder: if the error falls toward the exact-trained model's as θ
tightens, the tree read earns its credit as a convergent approximation; if the exact-trained model is no better, θ is a
pure cost knob on AirfRANS. The second dataset waits for the AirfRANS reads.

**For the book.** No standing changes. The DTT ledger row gains the trained-constant reading and the two code facts; 05
states what θ and the forest's precision do on a trained model; 04 and 20 read the exact read's error ratio on the
standing's own pair; 19 adds the DrivAerML Volume memory ladder; 12 and 16 say where the gap sits; 20 gains the open
question of a retrained θ (records copied to results/campaigns_2026-10-24/dttknobs/).

*Summary second-read in scratch/isla-dtt-knobs-2026-10-23/STAGE1_REVIEW.md; each child report second-read in its own
REVIEW.md and revised. Read by the supervisor on <date> (to be filled).*

<!-- END ENTRY -->

## Chapter changes

Numbers in the new chapter text are typed here for review. In the chapter, STYLE.md wants each computed from a file in
`results/` as the page renders, so the supervisor first copies the source files into the book's results tree; I suggest
`results/campaigns_2026-10-24/dttknobs/` with `sens/sens.json`, `sens/sens_regions.md`, `cost/cost.json`,
`cost/cost_table.md`, `cap/cap.json`, `order/order.json`, `gap/gap.json`, `treeprec/treeprec.json` and `CODE_FACTS.md`.
Chapter prose below carries no dates, node codes or run identifiers, and cites no notebook entry.

STYLE.md puts an In-flight callout beside a number that a running experiment will replace. Two changes add one now:
change 9 for DTT-FAST (ledger row 700, RUNNING), whose faster read and query-chunked training will replace 19's ISLA-DTT
cost numbers, and change 10 for AF-WHOLE (row 697, RUNNING), whose far-field runs will replace 12's airfoil-alone
ISLA-DTT ÷ ISLA gap. Stage 2 is pre-registered but not running, so its callout (change 8) is drafted for the day it
launches, at the one place a stage-2 result will replace what the chapter says; no stage-2 result replaces any other
number below, since the inference numbers stay true of the θ = 1-trained checkpoints.

1. **14-program-status.qmd, ledger row `DTT` (line 563), fifth column, after the E6 clause** ("... reading the final
   boundary state costs 37 to 72%"). Append:
   "; afterwards, on the 0.9.0 checkpoints trained at theta 1 with leaf 4 (AirfRANS eager-trained, lug fix-trained, two
   seeds each; DTT-KNOBS stage 1, inference only, @sec-nb-dtt-knobs-s1), the converse of E2: theta acts as a trained
   constant, every field and seed scoring best natively at theta 1 among the eight settings from theta 0 to 2 and the exact
   read 1.8 to 4.7x worse, while leaf 2 or 8 is inert on AirfRANS (at most 0.1 %) and costs up to 33 % on the lug; the normal-coherence test is
   vacuous at theta >= 0.919 (3D) / 0.765 (2D), so the theta-1 read never uses it; under bf16 training the forest is built
   from bf16-rounded coordinates (boxes and node measures bf16, box diameters and the walk's box table float32), so every
   float32 score reads a different forest (82 % / 99 % of AirfRANS / lug points get a different list, native error 0.2 to
   0.7 % apart, pressure level), and the lug's posed score differs from native at every theta > 0 (1.53 / 1.98 at theta 1,
   cause inferred, not confirmed); both code facts handed to DTT-POSE".
   Sixth column: add `@sec-nb-dtt-knobs-s1`. Sources: `sens/sens_table.md`, `cost/REPORT.md` result 1, `CODE_FACTS.md`,
   `treeprec/treeprec_table.md`. Callouts: none.

2. **14-program-status.qmd, ledger row `DTT-KNOBS` (line 701), second column.** Replace "RUNNING (launched by ...
   @sec-nb-surface-bc)" with:
   "RUNNING (launched by the supervisor 23:33 EDT 2026-10-23 on Peter's ruling of 23:26 EDT; stage 1 DONE 2026-10-24,
   inference only, 21.76 of 30 GPU-h, @sec-nb-dtt-knobs-s1: on the trained checkpoints theta acts as a trained constant,
   a cheaper read saves time only at car scale (400,000 DrivAerML Volume points fit a GB300 at theta >= 1.5 and not at
   theta 1, where the memory fit puts the limit at 349,000), the read's inference slack is in the kernel basis, and the
   gap to interacting ISLA is a fit gap that no inference-time setting explains; stage 2, retrained AirfRANS arms (theta 2, theta 0.5, every second
   read, basis 8, the exact read, no scales with a standardized inlet speed), is pre-registered and awaits the
   supervisor's approval of arms and cap; nothing adopted)".
   Fifth column: replace "Each probe's harness reproduces the standing's scores to float32 roundoff before any other
   number is read" with "Each probe's harness reproduces the standing's scores to float32 roundoff before any other number
   is read: met (ISLA-DTT on AirfRANS within 1.8e-6 relative; lug 0.00221933 / 0.00183431 against 0.002219 / 0.001834)".
   Sixth column: add `@sec-nb-dtt-knobs-s1`. Sources: `STAGE1.md`; the gate in `sens/REPORT.md` and `gap/REPORT.md`.
   Callouts: none. If the ledger has no DTT-POSE row yet, its owner's row is the place for the two code facts' status;
   I do not draft it.

3. **05-forward-pass.qmd, @sec-arch-tree-read, numbered list item 4**, after "...is below $\theta$." (the coherence
   test). Insert:
   "At $\theta \geq 0.919$ in three dimensions, and $0.765$ in two, the coherence test admits every node: inside one sign
   octant $\lvert \bar n \rvert \geq 1/\sqrt d$, so the box test alone decides which clusters a point reads."
   Source: `cost/REPORT.md` result 1, `CODE_FACTS.md` (Related fact), `sens/REPORT.md` (Coherence). Callouts: none.

4. **05-forward-pass.qmd, @sec-arch-tree-read**, three edits. (a) After the paragraph that ends
   "(`segment_softmax_read`)." insert a paragraph:
   "Under bfloat16 autocast, as in training, the two products `points @ frame` in `BoundaryForest.build` and
   `point_pairs` return bfloat16. The trees' coordinates and boxes and the node measures are then bfloat16, while the box
   diameters and the walk's box table stay float32, so a bfloat16 training step and a float32 score of the same case
   build different forests. On the AirfRANS interior task and the lug, 82 % and 99 % of the points get a different list,
   and reading the training forest changes the float32 native error by 0.2 to 0.7 % (pressure level). The bfloat16
   forest also depends on the pose: on the rotated and translated lug it doubles the error."
   So that 05 does not contradict (a), two edits to the paragraph "Three properties hold by construction":
   (b) replace "And a rotation, reflection or translation of the case gives the same forest and the same lists" with
   "And in float32 or float64 a rotation, reflection or translation of the case gives the same forest and the same
   lists" (the cited test runs in float64);
   (c) replace "The frame can tie: two equal principal variances, or an axis without a third moment, whose sign then
   follows whichever of the dataset's axes it overlaps most." with "The frame can tie: two equal principal variances, or
   an axis without a third moment, whose sign then follows whichever of the dataset's axes it overlaps most. So can the
   octant split, at a normal component that is zero but for rounding in that frame: its sign, and so the cell's tree,
   then rests on rounding."
   Sources: `CODE_FACTS.md` §1 and §2, `treeprec/treeprec_table.md`, `treeprec/treeprec.json`. I checked that
   origin/psharpe/isla's `tree_read.py` still has `points @ frame` at line 267 (and `point_pairs`' product at 366,
   CODE_FACTS §1), so this describes the current code; its `normal_octants` gives a component at exactly zero the sign of
   the next nonzero one, which leaves the rounding residues of (c). Callouts: none from stage 2; if DTT-POSE changes this
   code, the paragraph changes with it.

5. **05-forward-pass.qmd, @sec-arch-tree-read**, after "...so its kernel, keys and values are learned as the forest
   groups the cells." Insert:
   "Read at another opening angle, a model trained at $\theta = 1$ is less accurate, not more. On the AirfRANS interior
   task and the lug, the two seeds of each pair score best at $\theta = 1$ on every field, against $\theta$ = 0, 0.25, 0.5,
   0.75, 1.25, 1.5 and 2, and the exact read at $\theta = 0$ has 1.8 to 4.7 times the error at $\theta = 1$. At
   $\theta = 1$ each layer's read differs from the exact read on the same states by 3.9 % to 27 % (the median over points,
   by layer, AirfRANS), mostly through the clusters' mean values, and the difference falls slowly, as $\theta^{0.33}$ to
   $\theta^{0.69}$ from $\theta = 2$ to 0.5. The leaf size matters less: leaves of 2 or 8 cells change the AirfRANS error
   by at most 0.1 %; on the lug leaf 2 costs 2 % and 33 % (seeds 42 and 43) and leaf 8 changes it by at most 0.8 %."
   Sources: `sens/sens.json` and `sens/REPORT.md` (the θ settings and leaf ratios), `order/order_table.md` and
   `order/REPORT.md` (3.9-27 %, the exponents, the error in the values). Callouts: none (stage 2 adds models trained at
   other θ; it replaces none of these numbers).
   For the supervisor: this sentence bears on STYLE.md's knob rule, which credits a property exact as θ → 0 when "the
   model is run where that error is small". Stage 1 shows the per-layer error at θ = 1 is not small and that the
   end-to-end error does not converge at inference; whether the rule's credit holds for ISLA-DTT is what the retrained
   ladder tests (change 8). The existing sentences that credit pose invariance through the rule (04, 05, 09, 12, 20, 21,
   22) are unchanged here; that call is yours or Peter's.

6. **04-invariances.qmd, cell `para-remesh-theta`**, the sentence "The weights were trained at $\theta = 1$, so a smaller
   $\theta$ at inference also changes the function: the native pressure error of one seed rises from {...} to {...}."
   Replace with:
   "The weights were trained at $\theta = 1$, so a smaller $\theta$ at inference also changes the function: on the
   standing's pair, the exact read's native pressure error is 2.71 and 2.36 times its error at $\theta = 1$ (seeds 42 and
   43)."
   This cuts the one-seed number from the pair trained under the compiled-path fault and reads the standing's own pair.
   The refinement numbers at θ = 0 in the same paragraph stay (stage 1 did not re-measure refinement); keep their
   source note and add the new one. Source: `sens/sens.json` (AirfRANS, native, θ 0 ÷ θ 1). Callouts: none.

7. **20-limits.qmd, cell `para-oq-dtt-frame`** (@sec-oq-dtt-frame). Three edits.
   (a) Replace "whose native error rises at $\theta = 0$ ({...} to {...}, pressure / velocity / eddy viscosity, one seed), so
   pricing a small $\theta$ needs a model trained there." with:
   "; on the standing's pair the exact read has 2.71 / 2.36 times the error at $\theta = 1$ on pressure, 1.79 / 2.04 on
   velocity and 3.54 / 4.68 on eddy viscosity (seeds 42 / 43), so pricing a small $\theta$ needs a model trained there
   (@sec-oq-dtt-theta)."
   (b) Replace "That frame moves with a rigid motion, so the forest, and with it every point's list of far clusters at a
   finite opening angle $\theta$, is the same in every pose. Where two principal variances are equal, or the third moment
   along an axis vanishes, the axis or its sign is decided by rounding (the code then takes the sign of whichever of the
   dataset's axes it overlaps most), and a rotation can regroup the clusters." with "That frame moves with a rigid
   motion, so in float32 or float64 the forest, and with it every point's list of far clusters at a finite opening angle
   $\theta$, is the same in every pose. Where two principal variances are equal, or the third moment along an axis
   vanishes, the axis or its sign is decided by rounding (the code then takes the sign of whichever of the dataset's axes
   it overlaps most); where a cell's normal has a component that is zero but for rounding in that frame, so is the sign
   octant that picks its tree; and a rotation can regroup the clusters."
   (c) Replace "Where the frame ties, the error at $\theta = 1$ is not measured." with:
   "On the lug, where 84 % of the cells have a normal component below $10^{-5}$ in the forest's frame (measured in
   bfloat16), ISLA-DTT's posed error at $\theta = 1$ is 1.53 and 1.98 times its native error (seeds 42 and 43) on the code
   that pair trained with, the posed forest holds 1.29 times the pairs, and at $\theta = 0$ the two are equal. Whether the
   octant split's sign test on those near-zero components or a near-tie of the frame itself regroups the clusters is not
   separated. On boundaries with near-equal principal variances the error at $\theta = 1$ is not measured."
   Then add to "What would settle it": "and, on the lug, the posed forest's octant labels compared with the native ones
   under a tie-breaking rule".
   Sources: `sens/sens.json`, `sens/sens_table.md` (posed rows), `CODE_FACTS.md` §2. Callouts: none from stage 2. For the
   supervisor: the lug numbers are on the 0.9.0 code; the book's code now carries the geometry bundle (a new
   `normal_octants` rule for exactly zero components, the canonical frame in column products). If DTT-POSE re-measures the
   lug on the current code, STYLE.md would put an In-flight callout beside (c); that is outside this draft's brief.

8. **20-limits.qmd, a new open question**, a section after @sec-oq-dtt-frame and a row in `tbl-open-questions` after the
   frame row.
   Row: `("Does ISLA-DTT's error fall toward the exact read's when it is trained at a smaller opening angle?
   (@sec-oq-dtt-theta)", "ISLA-DTT trained on AirfRANS at $\\theta$ = 2, 0.5 and 0 (the exact read) against the
   $\\theta = 1$ standing, two seeds each, under the standing's protocol", "about 760 to 820")`. The figure is the stage-2
   pre-registration's (`PREREG.md` md5 84a25280, section 8): about 210 GPU-h at $\theta = 2$, 220 at 0.5 and 330 to 390
   for the exact read, eager, two seeds each. The exact read is submitted only if its smoke prices it at 300 GPU-h or
   less (PREREG gives P = 0.75 that it does not), so the supervisor may note that gate beside the row.
   Section:
   "### ISLA-DTT's opening angle in training {#sec-oq-dtt-theta}

   **The open question is whether ISLA-DTT is more accurate when it is trained at a smaller opening angle.** ISLA-DTT
   trains and is scored at $\theta = 1$ (@sec-arch-tree-read). Read at $\theta$ = 0, 0.25, 0.5, 0.75, 1.25, 1.5 or 2, a
   model trained there is less accurate on the AirfRANS interior task and the lug, and the exact read scores 1.8 to 4.7
   times worse, so the opening angle cannot be priced at inference. At $\theta = 1$ each layer's read is 3.9 % to 27 % from the exact read
   on the same states (median over points, AirfRANS). Whether a model trained with a tighter read is more accurate, so
   that the tree read converges toward the exact one in training as it does as an operator, or whether $\theta$ only
   trades cost, is not measured. Nor is whether a tighter trained read narrows ISLA-DTT's gap to interacting ISLA beyond
   one chord. What would settle it: ISLA-DTT trained on AirfRANS at $\theta = 2$, 0.5 and 0 against the $\theta = 1$
   standing, two seeds each, scored under the standing's protocol. If the error falls toward the exact-trained model's as
   $\theta$ tightens, the tree read is a convergent approximation; if the exact-trained model is no better, $\theta$ is a
   cost setting on AirfRANS. On the DrivAerML Volume task, the cost side is measured: 400,000 prediction points fit one
   GB300 at $\theta \geq 1.5$ and not at $\theta = 1$ (@sec-cost-sweep)."
   Sources: `sens/sens.json`, `order/order_table.md`, `cost/cost_table.md`, `STAGE1.md` (stage-2 arms). The
   @sec-cost-sweep reference assumes change 9 lands; otherwise drop the last sentence.
   In-flight callout to add **when stage 2 launches**, not before (it replaces the "not measured" of this section), and
   to remove when its arms are read:

   ```markdown
   ::: {.callout-note title="In flight"}
   These numbers come from models trained at $\theta = 1$ and read at other opening angles. ISLA-DTT trained on AirfRANS
   at $\theta = 2$, 0.5 and 0 is being measured, and the ledger (@sec-ledger) tracks it.
   :::
   ```

9. **19-capacity.qmd, cell `para-cost`**, after "...and it runs out of one GPU's memory at {DOOM:,}, where ISLA's step is
   {...} ms at {...} GiB." Insert:
   "How far a cheaper read moves that limit was measured on the Volume task with the whole boundary (the car and the
   tunnel box), random weights and bfloat16 training steps: 400,000 prediction points fit one GB300 at $\theta \geq 1.5$
   (2,734 ms at 273 GiB at $\theta = 1.5$, 2,070 ms at 260 GiB at 2, 1,837 ms at 248 GiB at 3) and run out of its memory
   at $\theta = 1$, where the memory fit (held-out error at most 0.4 %) puts the largest count at 349,000. The settings
   that fit take 2.3 to 3.4 times interacting ISLA's 796 ms at 400,000, measured with the car alone. By the same fit the
   slice blocks alone, at 35 GiB per 100,000 prediction points, cap the count near 565,000 with no boundary read. A
   larger $\theta$ costs accuracy on a model trained at $\theta = 1$ (@sec-arch-tree-read), so these settings need models trained there."
   Source: `cost/cost_table.md` (last section, the 400k projection) and `cost/cost.json`; 796 ms / 190 GiB is RENDER-090
   job 336005, `isla_volume_090.json`, as cost/REPORT.md cites it. Scope to keep in the caption or sentence: the probe ran
   another code snapshot and loss than the sweep (cost/REPORT.md, Checks), and reproduced 1,180 ms / 132 GiB at 100,000
   against the training configuration's 1,202 / 129. The 349,000 and 565,000 are projections of the memory fit
   (`cost/cost_table.md`, "largest Q that fits"); of the 400,000 rows, θ = 3, 2 and 1.5 ran and θ = 1 ran out of memory.
   Callouts: none from stage 2 (its arms train on AirfRANS). Add one for DTT-FAST (ledger row 700, RUNNING), whose fused
   read and query-chunked training will replace these numbers and the cell's other ISLA-DTT step and memory figures
   (`cost/REPORT.md` projects θ = 1 at about 430,000 with half the bytes per pair), directly after the cell `para-cost`:

   ```markdown
   ::: {.callout-note title="In flight"}
   These numbers come from ISLA-DTT's read as it is. A faster read and query-chunked training are being measured, and
   the ledger (@sec-ledger) tracks it.
   :::
   ```

   Remove it when DTT-FAST's numbers replace these.

10. **12-airfrans.qmd, @sec-airfrans-dtt-reference**, after the paragraph of cell `para-airfrans-dtt-reference`. A new
    paragraph:
    "**ISLA-DTT's gap on AirfRANS is one of fit, and it sits in the wake and beyond one chord.** On the first 200 training
    cases ISLA-DTT ÷ ISLA is 0.987 / 1.083 / 1.413 (pressure / velocity / eddy viscosity), against 0.994 / 1.129 / 1.342
    on the test set, for the eager-trained pairs. Its in-plane velocity error is 1.95 and 1.99 times ISLA's in the wake
    (seeds 42 and 43), and its eddy-viscosity error 1.48 and 2.47 times beyond one chord, where ISLA-DTT's own two seeds
    differ by 1.72 times. Near the wall eddy viscosity is 1.20 times ISLA's but carries at most 5 % of the gap. The
    difference varies smoothly in space, a bias rather than per-point tree noise, except for eddy viscosity beyond 0.1
    chord, where its gap sits. No opening angle read at inference narrows it, and the forest that bfloat16 training
    builds widens it slightly (eddy viscosity 1.351)."
    Scope: the training subset covers Re 2 to 3e6 only (the test's low-Re quartile reads alike); say so in the
    sentence or a margin note. I left out the eddy-viscosity wake share, which rests on seed 43. Sources: `gap/gap.json`,
    `gap/gap_table.md`, `gap/rough.md`, `gap/REPORT.md` ("where its gap sits"), `sens/sens_regions.md`,
    `treeprec/treeprec.json`. Callout: AF-WHOLE (ledger row 697, RUNNING) trains both forms with the far field and will
    replace this airfoil-alone gap. The callout at 12's line 278 sits beside the standing table, not in this section, so
    repeat its text directly after this paragraph (after the cell, if the paragraph is printed inside it):

    ```markdown
    ::: {.callout-note title="In flight"}
    These numbers come from ISLA and ISLA-DTT reading the airfoil alone. The same two trained with the far field
    as well (@sec-data-airfrans) are being measured, and the ledger (@sec-ledger) tracks it.
    :::
    ```

    Remove both callouts when AF-WHOLE's results land.

11. **16-lug.qmd, @sec-lug-limits, cell `para-lug-limits`, bullet "The standing is of the interacting form"**, after
    "...while the standing's ISLA trained eagerly ({source})." and before "Which form is ISLA's interior model is open
    (@sec-oq-dtt)." Insert:
    "The gap is one of fit (1.434 on the first 600 training cases), covers 96 % of the test cases and sits 90 % on the
    surface nodes. Scored in bfloat16, the 16-bit floating-point format training runs in (an 8-bit exponent and an
    8-bit significand), the two forms read level (1.043), mostly because bfloat16 read-out noise (about 0.0033, twice the
    lug's float32 errors) compresses the gap toward 1; even so ISLA-DTT is worse in 73 % of the test cases and on the
    surface nodes (1.11)."
    16 does not define bfloat16 elsewhere (git grep finds no match), so this is its first use there; the definition
    follows 01's wording. Sources: `gap/gap.json`, `gap/gap_table.md`, `gap/gap_table_bf16.md`, `gap/REPORT.md`
    (the quadrature model, 73 %, 1.11). Callouts: none.

Not changed, checked: no chapter says the read is cheap or that θ or the leaf size "changes nothing" (git grep over the
chapters for "cheap", "changes nothing", "leaf" and "constant of the class"); 05's per-point source count and step ratio at
θ = 1 (research/dtt/REPORT.md) and @sec-arch-cost's tree-read times stand; 09's and 06's θ = 1 descriptions stand; 22's
refinement numbers stand (stage 1 did not re-measure refinement).
