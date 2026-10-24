Verdict: CONFIRMED WITH CORRECTIONS

DK-PREREG-REVIEW (lane-reviewer, read-only, no jobs), written 2026-10-24 03:04 EDT by `date`, on PREREG.md md5 5aa4f0a7.
Numbers that check out against their sources: s_c 0.0022 / 0.0395 / 0.0078 and the control seeds (eag_af_dtt.md); velocity 2 SE 0.056 / 0.079;
control / ISLA 0.994 / 1.128 / 1.342 (eag_af_isla.md); B's seeds; G_f = ln 1.264 / 1.225 / 1.158 = 0.234 / 0.203 / 0.147 (RESULTS 3b);
C_f >= ln 1.05 on p and nut (§12); stage-1 inference ranges and the wake / >= 1 chord ratios (STAGE1, gap_table); R4's zone and band
definitions equal reduce_gap.py af_regions (md5 6be03f4b; probe_sens 323f5396); costs and steps equal PLAN §3 (sums 1,010 and
1,340-1,400); the K-EXACT gate formula gives 326-393 GPU-h for 1.43-1.73 s. The six conditions are each addressed; P8 sums to 1.
The problems below, all fixable before submission (PREREG §9 allows it):

P1 (should be fixed before approval)
1. L170, σ_f dof. 3c's pooled σ is 6 dof; DTT-NOSC-STD's two seeds add 1 (n - 1), giving 7, not 8. Fix: "7 dof", and name the
   pooled arms (3c's list plus DTT-NOSC-STD), so σ_f cannot be re-chosen after results.
2. L171, R5 'moves'. ATTRIB's R1 (PREREG §5), which the frame claims to use, is |Δ| >= 2 SE **and** |exp(Δ) - 1| >= 0.05; R5 drops the
   5 % floor. Fix: "moves f if |Δ| >= 2 SE and |exp Δ - 1| >= 0.05 (ATTRIB R1)".
3. L173-178, R5's non-move branch. On nut, 2 SE = 0.128 against G_nut = 0.147, so an arm with Δ = 0.12 (φ 0.82, almost DTT-NOSC) does
   not move nut and reads "the input standardization carries it". Section 10 sentence 2 then gives "is not in 0.9.0's bundle": one label
   for φ near 0 and for φ near 0.85. Fix: "standardization carries it" only if φ_f <= 0.3, or if the arm does not move f **and**
   |G_f - Δ| > 2 SE; otherwise "unresolved on f". The bundle-thread sentence follows only from a resolved label.
4. L172-175, φ's denominator. ATTRIB §10's φ is Δ / D_f (the CONT -> B step; for ISLA-DTT D_DTT = 0.103 / 0.156 / 0.026); G_f is a
   different, defensible choice (D_DTT on nut makes φ unstable), but "mirrored" and "ISLA-DTT's step" say otherwise. Fix: say G_f
   replaces D_f and why; word the labels "standardization removes / leaves DTT-NOSC's loss against B"; report φ against D_DTT beside it.
5. L157-160, R4 is not noise-aware. A seed-matched 0.90 / 1.10 bar is used where the control's own seeds differ by 1.72x in nut >= 1
   chord (gap_table §3), and by 0.87-1.16 in most cells; R1 itself (L125-126) says seed pairing is not assumed. Fix: R1's shape on each
   cell, D_R = ln(mean_seeds e_R(arm) / mean_seeds e_R(control)), SE from both pairs' spreads, narrows if D_R < ln 0.90 and |D_R| >
   2 SE (widens mirrored); keep the seed-matched ratios beside.
6. L132-133, R2 'rises' mixes units. "CM ... exceeds CM ... by more than 2 SE" compares a CM difference with an SE of ln CM. Fix: "a step
   rises if D_tighter - D_looser > 2 SE of that pair (seed-mean ln CM, D_1 = 0)".
7. L52, L58-63, arm 6 like-with-like is not stated. L62-63 "the comparisons are within one code" is false for arm 6 (code_att090fix +
   recipe_att090b against B on code_cmpaf_fix + recipe_r090). Fix: add PLAN §1's argument (isla/ differs only by the query_schema /
   query_embedding / query_encoding switches, inactive here; tree_read.py only by an optional query_vecs path, off; ATTRIB §9b made the
   same check for B), and state that arm 6's one change is relative to DTT-NOSC (the inlet speed standardized), B being the frame's
   reference, as in ATTRIB.
8. L97 contradicts L93 for arm 6. Fix-trained rows are scored eager by appending `compile=false model.compile_blocks=false` (PLAN §2,
   as r090fix's rows), so "the score log's extra equal to the lane's" fails and would void R5. Fix: "equal to the lane's, plus
   `compile=false model.compile_blocks=false` for fix-trained rows".

P2 (notes)
9. L123: "its decision use is in section 6": section 6 is the predictions; the decision use is R3 (L146-148). Fix: "in R3".
10. L193-196: P11 gives 0.75 that K-EXACT is not run, but P6-P8 put pure cost knob + converges + exact worse at 0.75. Fix: mark P6-P8
    "conditional on K-EXACT being trained and read", or give P8 incomplete >= 0.75. No prediction covers R5 on nut.
11. L80-81, L236-237: the fix-trained K-EXACT alternative (245-310 GPU-h) can itself exceed 300 and has no smoke unit. Fix: add PLAN's
    200-step fix-trained K-EXACT unit to job (b), apply the same 300 GPU-h gate to it, and price the supervisor's decision from it. The
    gate as written is the supervisor's (stop and tell at > 300 GPU-h); the 200 GiB memory stop is an extra and is fine.
12. L232-238: the cap section does not say what runs if the supervisor keeps 950 (arms 1-4 + overheads are about 874; arm 6 does not
    fit). Fix: "at 950, arms 1-4 only; DTT-NOSC-STD waits for an increase". Also ask explicitly for the concurrency lift (L74-77, 12 nodes).
13. L164: "narrows either cell and widens none" is ambiguous across the two arms, and for K-EXACT not run. Fix: "K-THETA05 or K-EXACT
    narrows a cell and neither arm widens either cell; if K-EXACT is not read, from K-THETA05 alone, labelled so".
14. L141: say whether the overall R2 label carries the sub-label "free to theta 2" (proposed: only if two fields carry it). L237: with
    a fix-trained K-EXACT, the step (0.5, 0) crosses training paths; say so in the label.
15. L152-154: "a candidate" for the tighter arms: name it ("a candidate for the scope freeze as an accuracy-for-cost trade").
16. L80-81: PLAN §1 recommends a 50-case cmpaf_diag compiled-against-eager check for DTT-NOSC-STD in the smoke; it was dropped silently.
    Add it or say why not.
17. L23-24 (clarity): "the accuracy gap is not in 0.9.0's bundle" reads as Q3's gap to interacting ISLA. Fix: "DTT-NOSC's cost (C_f)
    is an interaction with the raw speed, not a cost of 0.9.0's bundle", as R5 L177 says.

One change per arm holds for arms 1-5 (PLAN §1 diffs). The control and training path are stated for every arm. Condition 1
(coherence kept, refusals counted on 800 + 200 cases) and condition 6 (priority order) are met as written. No result was seen
by me; I opened nothing under scratch/isla-tie-2026-10-22/.
