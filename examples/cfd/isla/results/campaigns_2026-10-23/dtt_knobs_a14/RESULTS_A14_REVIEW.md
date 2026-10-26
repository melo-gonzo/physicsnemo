Verdict: CONFIRMED WITH CORRECTIONS

DK-A14-REVIEW, a read-only second read of RESULTS.md (md5 44396226), section "Arms 1-4: R1, R3, R4", against PREREG 39d8272f
and reduce_s2.py e2b605d2. Recomputed independently (stdlib scripts in a temporary directory outside the repo, since removed) from s2data/ and ctrl/.
K-EXACT's outputs were not read.

What holds: every registered number and label. R1's 12 native cells (means, ratio, D, s_arm, s_c, 2 SE, label, flag
with the both-seeds rule, seed ratios, arm ÷ ISLA arithmetic) match. Posed gives the same labels and flags. Section 4
passes on all 16 rows (epoch 500; 200 ids equal to the control's, in order; SNAPSHOT_USED = code_dttk_<arm>; DATASET_USED;
the extra byte-equal to lanes.tsv). sacct: 382281_0-3, 382386 and 382387 COMPLETED 0:0. The regional gate passes on 8 runs
(max 4.2e-6), the fetched per_case md5s equal LHR's, and SETTING matches each arm. R3 follows PREREG's text, the
tighter-arm rule included. R4's 8 cells reproduce in the pooled registered form, and DK-GAP's form gives the same labels.
P1-P5 and P9 are scored correctly. GPU-h and steps match sacct, basis8_seed43's NODE_FAIL link included.

Corrections:
1. P1, l.95: "K-THETA05 is uniformly better ... and better in the wake" overclaims. Its case means are below the control's
   on all three fields, in both seeds. But it wins only 68-75 % of cases, and the regional tables show nut worse in the
   outer zone (1.10 / 1.13, registered form) and in the 1e-2..0.1 band (1.12 / 1.07). In the wake, pressure (1.02 / 0.99)
   and nut (1.02 / 1.01) are level; only velocity improves. Fix: "better on every field's case mean, in both seeds
   (seed-separated), and on in-plane velocity in the wake; not uniformly (nut in the outer zone 1.10 / 1.13)".
2. P1, l.97-98: "the first measurement in which a tighter trained read is more accurate on AirfRANS" is contradicted by
   the earlier DTT research. E5 (research/dtt/REPORT.md on origin/psharpe/isla, code_dtt12) trained theta 1, 0.5, 0.25 and
   the dense read on AirfRANS. Theta 0.5 ÷ theta 1 was 0.95 / 0.99 / 1.00 (pressure seeds 0.85 / 1.04), inside its seed
   spread. Fix: "the first seed-separated measurement ...; E5's earlier ladder on older code saw a few percent, unresolved".
3. P1, l.98: "theta 0.5 costs 3.2x the pairs at 200k tokens" misattributes the factor. In cost/REPORT.md, 3.2x is theta 0.5's
   coupled read (with coherence) over the box-only read at the same theta: 5,830 against 1,851 pairs per point. Against
   theta 1 (833 pairs per point) it is about 7.0x. Fix: state both factors.
4. P1, l.99 and l.106: "400k points do not fit a GB300" and "400k points fit at theta >= 1.5 only" describe the code before
   DTT-FAST. DTT-FAST, merged 2026-10-24, runs 400k DrivAerML whole-boundary points at theta 1 in 1,792 ms / 212 GiB
   (dtt-fast final2/TABLE.md l.150). So "theta 2's saving lives at car scale" no longer rests on the fit. Fix: date both
   statements to the code before DTT-FAST, give DTT-FAST's figure, and say that theta 0.5's fit on the new code is not measured.
5. P2, l.102-103: stage 1's "1.11 / 1.28x" is a geometric mean over the three fields. Basis 8's nut at inference was already
   1.07 / 1.34 (cap_table.md, basis_kw_8). Retraining recovered pressure and velocity but not nut. Fix: "not slack for nut
   in training" in place of "not slack in training", and quote the per-field inference figures.
6. P2, l.100-101: K-B1 "free" holds for R1. Beside it, nut beyond one chord is worse in both seeds (1.24 / 1.13; D_R +0.167
   against 2 SE 0.211, so "level"); say so. Its 7.6 % GPU-h saving includes about 2 % that every arm shows against the
   control pair (K-THETA2 216.4 at an equal step). Against the arms trained alongside it, the saving is 5.7-6.1 %, in line
   with the 5.0 % step saving. Also, sacct gives the control pair 110.8 / 110.0 (220.8), not 110.1 / 220.9. "A larger share
   on the car" is not a number stage 1 gives (cap: 9 % AirfRANS, 26 % lug); give its source, or soften it.
7. P2, l.55 and l.82-83: probe_s2reg.py is DK-SENS's registered probe with four changes (REGIONAL.md, diff 43dbafe8); say
   so. The 1.73 / 1.68 against 1.95 / 1.99 are in DK-GAP's case-normalized form; label them.
8. P2, clarity: a reader new to the campaign needs one line per arm (theta 2 is a coarser tree read, B1 reads after every
   second block, basis 8 is a narrower kernel basis), the control's name (r090eag_af_dtt, eager), "ratio < 1 is better",
   and what "flag", "scope freeze" and "stage 3" mean. R3's trade readings call for "its measured saving (smoke and lane
   step, memory, car-scale projection)". The section gives the lane step only; add the peak memory (49.6-50.4 GB for
   K-THETA2).
