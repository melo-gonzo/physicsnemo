# AF-FIX campaign: ISLA-DTT on AirfRANS with its far field, with the encoding fixed (pre-registration)

Lane AF-FIX (research-lane, supervisor 01a13c2c). Drafted 2026-10-25, 08:05-08:35 EDT (PREREG_DRAFT.md md5 553e4714).
Carried out by lane AF-CAMPAIGN (scratch/isla-af-campaign-2026-10-25). The supervisor approved the campaign at 08:51 EDT
10-25; the approval and every change since the draft are in section 9 (amendment before submission), and the lines
changed in sections 2-8 are marked [A1]. Adoption of any change is Peter's.

## 0. What was seen before drafting (disclosure)

- AF-WHOLE's records in full (FINAL_REPORT, RESULTS, read.md, FACTS, the octant probe), including W-DTT's and B-DTT's
  scores, which are this campaign's control and beside.
- NEARWALL's DESIGN, PREREG_CONTINGENCY, GEOMETRY_NOTE, RESULTS; DTT-POSE's DESIGN; REYNOLDS-OUT's DESIGN.
- This lane's probes: the read-mass probe (`PRESSURE_PROBE.md`; W and B checkpoints, 16 full_test cases each, with
  their per-band errors), the tie count (`tie/REPORT_TIE.md`) and the precision gate (`prec/REPORT_PREC.md`).
- **Not seen:** any DTT-POSE-EXP score, validation curve or Q/P probe (its arms are code_dttposex on airfoil-only
  AirfRANS and the lug); any SLICE-NORMAL-EXP or SLICES-ALL result. I will open none before this file is frozen.

## 1. Question

AF-WHOLE: ISLA-DTT trained with the far field (W-DTT) is worse than airfoil-only ISLA-DTT (B-DTT) by 1.94 / 1.53 / 1.89
(pressure / velocity / nut), and its pose breaks (1.53-2.52). Three defects of the encoding are measured with the far
field (this lane): (T) the octant split ties on the far field's axis-aligned normals; (N) the softening 1e-6 L
(3.05e-4 chords) puts 20 % of the points in the linear part; (C) the far field's 99.86 % of the measure takes 99.75 % of
the slices' mass and 79-97 % of the tree read's. **Q: does fixing each, in turn, reduce W-DTT's error, and does the
whole fix close the gap to B-DTT?**

## 2. Arms (nested: each adds one fix to the one before)

All ISLA-DTT, AirfRANS with its far field, AF-WHOLE's W-DTT configuration exactly (`kernel/afwhole/lanes.tsv` rows 2-3:
`isla_volume_airfrans`, `airfrans_uvolume_2d`, `++model.query_independent=true`, hidden 512, 512 slices, lr 3.75e-4,
500 epochs, 100,000 points, bfloat16 training, recipe `$T/recipe_afwhole`, the refit constants), seeds 42 and 43. Only
the code snapshot differs.

| arm | code | change against the arm before | reads against |
|---|---|---|---|
| **F** | `$T/code_affix_f` = `$T/code_dttposex` (DTT-POSE's forest ported to 0.9.0: faces with the tie band, centred float64 frame coordinates, banded box test, built outside autocast) + rule 1 of `DESIGN.md` §1: one set of face trees per declared condition, each about its condition's own measure-weighted centroid, tol and R from it (bitwise code_dttposex on a single-condition boundary) | the forest (T) | W-DTT (AF-WHOLE's `afwhole_af_dtt_seed42/43`) |
| **FN** | `$T/code_affix_fn` = code_affix_f + the viscous softening: the query's second channel asinh(s / eps) with eps = nu / (U L) in frame units, i.e. asinh(d U / nu), U and nu the same ones as the Reynolds input (`_case_scales`) | the near-wall length (N) | F |
| **FNC** | `$T/code_affix_fnc` = code_affix_fn + per-condition quadrature: every boundary token's log-weight in the blocks and in the read's per-pair log m is log a_i - log A_k + beta_k (A_k the full-resolution measure of its condition, beta_k a learned scalar per declared condition, init 0); the forest is still built from the raw cell measures. [A1] Plus the read's backward fix (section 9, A1.5): the Triton read returns the gradient of log m | the measure (C) | FN |
| **FNΩ** [A1] | `$T/code_affix_fnomega` = code_affix_fn + the solid-angle read prior: each pair's log m in ISLA-DTT's tree read gains -(d - 1) log r, r the pair's point-source distance as the read's invariants compute it (`dist.clamp_min(eps)`, in units of L; a unit change is a per-point constant, which cancels in the point's softmax); sources are cells and node centroids alike; computed without gradient (geometry); the blocks and the slices unchanged | the read's reference weight (Omega), in place of C | FN |

- Beside, never in a rule: B-DTT (`r090eag_af_dtt_seed42/43`, airfoil only), the gap target.
- **F needs the supervisor's approval of a forest change.** DTT-POSE's forest is frozen (supervisor, 2026-10-24: "Any
  change to the forest's semantics needs the supervisor's approval first"). Rule 1 changes the grouping, the trees' origin
  and where tol and R are measured from. It is bitwise DTT-POSE on a single-condition boundary (checked in §3).
- **Alternative F0** (no forest change; DTT-POSE exactly as DTT-POSE-EXP tests it): `$T/code_dttposex` as it is. TIE measured
  the faces' residual with the far field on 38a7915cc (the 0.13.0-line forest; the 0.9.0 port keeps 0.9.0's
  `canonical_frame`, the same function to float64 roundoff, so F0's residual is inferred to be the same): 4.1 / 15.8 /
  26.5 % of posed pairs with no native match, against 0.004 / 0.026 / 6.0 % for rule 1 (a counterfactual). F0's R3 is
  therefore more likely to read broken (my p 0.45 for "holds").
- **NEARWALL's registered contingency is superseded** (confirmed by the supervisor, 08:51 EDT 10-25 [A1]): its arm V was interacting W-ISLA, a model
  0.13.0 removed. FN is a new test of the same softening on ISLA-DTT, nested after F; in ISLA-DTT the softening reaches
  only the points' seed (the read's log(r / |d_q|) is unsoftened), so FN is a weaker test of ii than V would have been.
- **Fourth arm FNΩ** (approved by the supervisor, 08:51 EDT 10-25, +about 220 GPU-h): code_affix_fn + the solid-angle
  read prior (each pair's bias gains -(d - 1) log r) in place of C. It isolates the read from the slices. [A1]
- [A1] F0 is not run: the supervisor approved rule 1 for arm F (section 9, A1.2).
- Why 0.9.0 ports: the control W-DTT ran on code_cmpaf_fix (0.9.0); a port keeps everything else identical, as
  DTT-POSE-EXP did. Each fix is ported to 0.13.0 at adoption, function-equal (REYNOLDS-OUT's bitwise check pattern): F is
  DTT-POSE's 38a7915cc; N in 0.13.0 is a generic constructor argument `distance_softening: {nu: 1, U: -1}` (units grammar
  over declared scales; AirfRANS adds `nu: {rank: 0, defines: nu}`); C has no fluid term in either version.
- Paired reading is valid for every step: F and FN add no parameters, and C's beta is created with zeros (no RNG draw),
  so a seed gives the same initialization and data order in an arm and its control. The smoke confirms it: the step-0
  parameter tensors (beta aside) are identical across the arms.

## 3. Snapshots and checks before submission

- `code_affix_f`, `code_affix_fn`, `code_affix_fnc`, `code_affix_fnomega` [A1]: `cp -r --preserve=timestamps $T/code_dttposex` plus the changed
  tree_read.py (F; the forest's condition index comes from model.py) and model.py (N, C; for FNΩ tree_read.py; for FNC
  also segment_softmax.py, A1.5), and the new test files, SNAPSHOT_HEAD, MD5SUMS, chmod a-w. Built from a worktree branch `psharpe/isla-affix-ports` (record
  only, `git commit -s`, hooks on, plain push, never merged).
- Tests: the whole ISLA test directory of each snapshot on LHR cpu-short, one at a time; plus new tests: N's channel
  invariant to a consistent rescale of length and viscosity, and equal to (i) where no viscosity is declared; C's
  refinement invariance (splitting a cell), similarity, beta = 0 with one condition equal to today's function, and the
  learned shares' gradient finite.
- Precondition gate for N (NEARWALL PREREG_CONTINGENCY §1): on code_affix_fn, the channel's float32 compute rounding on
  AirfRANS with its far field at p99.9 <= 0.1 in the native and posed frames, rerun with `prec/prec.py` on the arm's own
  snapshot (0.9.0 geometry: NEARWALL measured 0.0018 native and 0.0098 at T = L_body; 0.13.0 geometry: 1.8e-7 / 1.7e-7,
  `prec/tables.md`).
- Forest check for F (rule 1): TIE's harness (`tie/groups.py`) on code_affix_f, 24 cases: far-field and airfoil cells
  changing face under pose, and posed pairs with no native match (expected near the measured counterfactual, median
  2.6e-4); and bitwise equality with code_dttposex's forest on 4 airfoil-only cases. The same harness on code_affix_fnc:
  its forest equals code_affix_f's bitwise (C must not reach the forest). [A1] The same on code_affix_fn and code_affix_fnomega: their forests equal
  code_affix_f's bitwise.
- A CPU forward on 2 real cases per snapshot: finite; N's seed column equals asinh(d U / nu) to 1e-6 relative; C's
  log-weights equal log a - log A_k on beta = 0, and with one condition C's forward equals code_affix_fn's to float32
  rounding (1e-6 relative; not bitwise: the constants cancel in every softmax only up to rounding). [A1] FNΩ: each pair's log m equals FN's minus (d - 1) log r of that
  pair (float32 rounding), and the blocks' inputs equal FN's bitwise.
- [A1] **The read's log-m gradient (FNC; gating; section 9 A1.5)**, on a GB300 (short QoS, `isla.affix.check`) with the
  snapshot's own tests: (g1) the read alone, random plans at the model's widths (H 8, KC 4, C 8), float32 and bfloat16
  inputs: the Triton path's gradient of log m is finite, nonzero, and within 1e-5 of max |reference| of the eager read's in
  float32 (on float32 copies of the same inputs); its other gradients (ker, q, k, v) and its output are bitwise those of
  code_dttposex's segment_softmax.py on the same inputs (the fix adds a store only). (g2) The model: a small two-condition
  ISLA-DTT (the AirfRANS config's conditions, two real cases), CUDA, float32 eager: beta's gradient with the Triton read
  against the same forward with the read forced eager (`segment_softmax.supported` patched to False) within 1e-4 relative
  per component (measured value reported); beta's gradient through the read alone (full minus the same backward with the
  read's beta term detached) finite and nonzero. Beside, not gating: (g2) on code_affix_fnc with code_dttposex's
  segment_softmax.py (the bug), where the read's part is expected to be exactly zero.
- [A1] **The read path in training**: every smoke unit of an ISLA-DTT arm logs how many reads ran the Triton path and how
  many the eager path; pass only if every read of every unit ran the Triton path (expected from the code: CUDA tensors,
  Triton importable, H 8, KC 4, C 8 are powers of two, bfloat16 not float64).
- **Smoke** (one short-QoS job, `isla.affix.smoke`, 40 min): F, FN, FNC and FNΩ [A1] seed 42, 50 steps each, written to a smoke
  directory; passes on rc 0, finite losses, a resolved config equal to W-DTT's except run_id and the code path, the
  step-0 parameter tensors (beta aside) equal across the four arms [A1] and a fifth unit on code_cmpaf_fix with W-DTT's row
  (smoke only, never scored), and the first batches' case ids and a hash of their points equal across all five [A1]. A smoke
  step of F above 1.05x the code_cmpaf_fix unit's goes back to the supervisor for the cap before the lanes. Cost per arm: median step and peak memory against W-DTT's smoke
  (AF-WHOLE 361876).

## 4. Scoring (exactly AF-WHOLE's)

kernel/afwhole/score.sbatch unchanged in function: native, area (checked equal to native, then labelled degenerate) and
posed, float32, full_test (200 cases), the epoch-500 checkpoint. Near-wall dumps and AF-WHOLE's R2 decades with the
3.05e-4 chord split. Checks before reading: `"epoch": 500`, 200 rows, SNAPSHOT_USED, the freestream guard, the same scored
points as W-DTT (coordinate hash). A failed check makes the row invalid and its rule "incomplete as registered".
Event guard as AF-WHOLE's (validation loss above 4x its running minimum after epoch 10, non-finite loss, BLOCKED, a hung
link): stop and tell the supervisor; never resubmit.

## 5. Rules (AF-WHOLE's and NEARWALL's, per step)

CM = the scorer's case-mean relative L2, native; fields pressure, in-plane velocity, nut. For each step X against its
control Y (F/W-DTT, FN/F, FNC/FN, FNΩ/FN [A1]):
- **R1:** `D = ln(mean_seeds CM(X) / mean_seeds CM(Y))`, `SE = sqrt((s_X^2 + s_Y^2) / 2)` with s the pair's two-seed sd
  of ln CM. Helps if D < ln 0.90 and |D| > 2 SE; harms if D > ln 1.10 and |D| > 2 SE; else no effect. Paired, with
  r_s = CM(X, s) / CM(Y, s): helps (paired) if r_42 and r_43 < 1/1.10; harms (paired) if both > 1.10; a harms (paired)
  counts as harms. Sub-bar flag as NEARWALL's. **"Helps" below means R1's helps or helps (paired)**; the label says which.
- **R2 (descriptive):** per field and decade, R1 on the pooled band e_R; for FN/F the split below / above 3.05e-4 chords;
  beside, the share of W-DTT's band gap to B-DTT that the arm closes, `1 - ln(e(X) / e(B)) / ln(e(W) / e(B))`.
- **R3 pose** (AF-WHOLE's R3 verbatim, per arm): posed / native of CM. **Pose holds** if every field and seed lies within
  0.002 of one; **"holds at the standing's level"** when the 0.002 bar fails only on a field where the arm's worst seed is no
  farther from one than B-DTT's worst seed (B-DTT pressure 1.0017 / 1.0022). Otherwise "pose broken", reported as a
  defect. F's R3 is the tie fix's registered test.
- **R4 cost:** smoke step and memory against W-DTT; lanes' GPU-h.
- **Beside:** FNC / B-DTT and FNΩ / B-DTT [A1] per field, the remaining gap.
- What a reading leads to (no decision of mine): F helps or restores pose: evidence for DTT-POSE-EXP's adoption on the
  whole boundary. FN helps with no field harmed: (ii) goes to Peter with the AeroZoo question (section 7). FNC helps on
  pressure with no field harmed: C goes to Peter, with its blast radius (every multi-condition dataset). [A1] FNΩ helps
  on pressure with no field harmed: Omega goes to Peter with its narrower blast radius (ISLA-DTT's read on every interior
  dataset); FNC and FNΩ are each read against FN only, never against each other in a rule (beside: FNC / FNΩ per field,
  descriptive). Any harms: that fix is not proposed. A lost arm or row: incomplete as registered, escalated.

## 6. Predictions (mine, before any arm exists)

| id | prediction | p |
|---|---|---|
| A1 | the smoke passes first time | 0.80 |
| A2 | F R3: pose holds or at B-DTT's level | 0.80 |
| A3 | F / W-DTT helps on at least one field (R1 or paired) | 0.65 |
| A4 | FN / F velocity: helps / no effect / harms | 0.30 / 0.62 / 0.08 |
| A5 | FN / F nut: helps / no effect / harms | 0.30 / 0.62 / 0.08 |
| A6 | FN / F pressure: helps / no effect / harms | 0.10 / 0.80 / 0.10 |
| A7 | FNC / FN pressure: helps / no effect / harms | 0.40 / 0.45 / 0.15 |
| A8 | FNC / FN harms on some field | 0.20 |
| A9 | FNC / B-DTT below 1.10 on pressure (seed means) | 0.25 |
| A10 | FN's smoke step within [0.98, 1.03] of F's; FNC's within [0.98, 1.05] | 0.85 |
| A11 | F's smoke step at most 1.00x the code_cmpaf_fix unit's (faces and rule 1 cut the pairs) | 0.65 |
| A12 [A1] | FNΩ / FN pressure: helps / no effect / harms | 0.25 / 0.63 / 0.12 |
| A13 [A1] | FNΩ / FN velocity: helps / no effect / harms | 0.15 / 0.75 / 0.10 |
| A14 [A1] | FNΩ / FN nut: helps / no effect / harms | 0.20 / 0.70 / 0.10 |
| A15 [A1] | FNΩ / FN harms on some field | 0.18 |
| A16 [A1] | FNΩ R3: pose holds or at B-DTT's level | 0.78 |
| A17 [A1] | FNΩ's smoke step within [0.98, 1.03] of FN's | 0.85 |
| A18 [A1] | the read-gradient checks (g1), (g2) pass on the first run of the fixed FNC | 0.85 |

Why: the tie fix removes a pose break and a case-to-case split of whole far-field edges (A2, A3); rule 1 removes the
airfoil's 28-chord lever arm that faces alone leave (measured as a counterfactual only, hence 0.80). In ISLA-DTT the tree
read already codes log(r / |d_q|) unsoftened, so the softening reaches only the points' seed, and its effect is likely
smaller than NEARWALL predicted for interacting ISLA (A4-A6). The measure is the measured mechanism for the slices and
the read, but its causal link to the pressure harm is inferred (`PRESSURE_PROBE.md` §2), hence A7 below even.
[A1, written by AF-CAMPAIGN at 08:57 EDT 10-25, before any job of the campaign] FNΩ moves only the read's prior; the
learned kernel can already approximate -(d - 1) log r (its basis takes log r), so the function class is about the same
and the effect comes from where training starts and what it finds easily. The read in W-DTT gives the airfoil 10-12 % of
the attention next to the wall; with the prior an airfoil cell at r near |d_q| gains about log(200 / 1e-6) nats over a far-field
cell, so the near-wall read starts on the airfoil. That should help pressure more than velocity and nut, whose near-wall
error FN targets; but the slices stay 99.75 % far field, which the measured mechanism also blames, so most mass stays
on "no effect" (A12-A14).

## 7. AeroZoo

- Built and checked (tests, CPU forward, smoke, as §3) only when the supervisor launches it; not part of this cap.
- F and C change AeroZoo's ISLA-DTT (F: every forest; C: inlet, outlet and walls get their own shares). N changes it only
  if AeroZoo's config declares the viscous length (it should, for one rule across flows: `physics.kinematic_viscosity`
  defines nu, `bc_velocity` defines U; eps then moves from 1e-6 L to 5e-9-1e-4 L across cases, NEARWALL).
- **Check (no-harm, not attribution):** FNC on AeroZoo's full split (azc_full_dtt's configuration on code_r090 plus the
  three fixes: `code_affix_fnc_r090`), seeds 42 and 43, against the standings azc_full_dtt_seed42/43; R1 per field
  (velocity, pressure, nut, tke, omega) with the harms half only: adoption needs no field "harms".
- Cost about 2 x 270 GPU-h (ISLA-DTT AeroZoo lanes: 3,770 GPU-h for 14, AZ-DECIDE) plus scoring.

## 8. Cost, cap, timeline

- [A1] **Cap approved: 920 GPU-h** for the four arms (section 9), the smoke's cost rule kept.
- **AirfRANS (3 arms x 2 seeds):** W-DTT's lanes took 110.4 and 110.3 GPU-h (8 and 10 links), so about 6 x 110 = 660 GPU-h,
  plus smoke 2.7 (four units), scoring 3, near-wall dumps 2, tests (cpu-short), about 670: **cap 700 GPU-h** (a 5 % margin;
  the smoke's cost rule above protects it). FNΩ adds 220 (cap 920).
- **AeroZoo check:** about 545 GPU-h (cap 600). [A1] Not launched: gated on this campaign's reads and Peter's adoption decisions (section 9, A1.4).
- **Timeline** (W-DTT ran about 30 h of wall per lane):
  - approval by about 10:00 EDT 10-25; three ports (F's forest, N, C), their tests, the forest check and the smoke by
  about 15:00 (a writer child in a worktree, and one cpu-short suite at a time); lanes from about 15:00;
  - AirfRANS lanes end about 10-26 21:00 - 10-27 05:00 EDT, scores and reads by about 10-27 08:00 if the queue allows
    six lanes at once (DTT-POSE-EXP, SLICE-NORMAL-EXP and SLICES-ALL also hold AirfRANS lanes): **at the scope freeze**;
  - with fewer nodes, or a failed smoke, the read slips past the freeze;
  - the AeroZoo check slips past the freeze either way: an ISLA-DTT AeroZoo lane is about 270 GPU-h, about 68 h on one
    4-GPU node (AZ-DECIDE tables_partial.md, projected to 250 epochs), so run in parallel from 10-25 afternoon it ends
    about 10-28 midday, and gated on the AirfRANS read about 10-30.
- **Registry row** `^isla\.affix\.` re-added at the first job (it was removed at 08:33 EDT 10-25, after the probes),
  expected_end an ISO date; job names `isla.affix.<arm>-seed4x`,
  `isla.affix.smoke`, `isla.affix.score`; the five bad nodes excluded.

## 9. Amendment before submission (A1, 2026-10-25 08:57 EDT, AF-CAMPAIGN)

Written before any job of this campaign: no port, snapshot, test, check, smoke, lane or score exists. What the writer
(AF-CAMPAIGN) has seen: everything AF-FIX listed in section 0 (AF-FIX's files, AF-WHOLE's records including W-DTT's and
B-DTT's scores), DTT-POSE-EXP's PREREG and JOBS (its snapshot and launcher; none of its results, curves or probes), and
TEST-BUILD's BUGS.md rows W3-B2 and W1-B1. No number from any arm this file reads exists.

**The supervisor's decisions (08:51 EDT 10-25), recorded as given:**
- A1.1 The campaign is approved: arms F, FN and FNC, plus the fourth arm FNΩ (code_affix_fn plus the solid-angle read
  prior in place of C), each with seeds 42 and 43. **Cap 920 GPU-h**, with the draft's smoke cost rule (section 3).
- A1.2 Rule 1 (face trees per declared condition, each about its condition's own centroid) is approved for arm F as a
  change to DTT-POSE's frozen forest. On AirfRANS, declared conditions and connected components coincide. Grouping for
  multi-condition datasets in canonical ISLA is Peter's decision later. F0 is not run.
- A1.3 FN supersedes NEARWALL's registered contingency, whose arm was interacting W-ISLA, removed in 0.13.0.
- A1.4 The AeroZoo check (section 7) is not launched now. It is gated on this campaign's reads and on Peter's adoption
  decisions.

**Changes by AF-CAMPAIGN (before results):**
- A1.5 **The read's log-m gradient (supervisor's steer, about 08:55 EDT 10-25).** TEST-BUILD's W3-B2: in 0.9.0's
  `segment_softmax.py` (code_dttposex 181991c0, the same file as 92d9d7800) the Triton read's backward returns None for
  `logm`, while the eager read differentiates it. **Which path FNC trains on (from the code):** `segment_softmax_read` takes
  the Triton path when every tensor is on CUDA, Triton imports, H, KC, C, H KC and H C are powers of two and q is not
  float64. The read's widths are H = 8, KC = 4, C = 64 / 8 = 8 (tree_read.py constants), the lanes train on GB300 under
  bfloat16 autocast (`compile=false`, `compile_blocks=false`: eager, but the read's Triton kernels are not torch.compile),
  so **every training read is the Triton one** (confirmed per smoke unit, section 3). Unfixed, beta_k would learn only
  through the blocks and the head frame while its value still moved the read, so FNC would not test C as registered.
  **Choice: a minimal backward fix in FNC's port**, recorded as part of FNC's code: `_ssr_bwd_point_kernel` gains an
  output `dlogm` (float32, one per pair) = the sum over heads of the logit gradient dz it already computes in float32,
  stored only when log m needs a gradient (a constexpr flag, so the kernel without it is today's); `backward` returns it
  in logm's slot (cast to logm's dtype, float32 here). Forward, dker, dq, dk and dv are untouched (checked bitwise,
  section 3). The fix lives only in code_affix_fnc: in F, FN and FNΩ log m takes no gradient (FNΩ's -(d - 1) log r is
  geometry, built without gradient), so their segment_softmax.py stays code_dttposex's and their controls' read is the
  same code. Why not force the eager read: it gathers per pair and keeps every pair's terms for backward (memory and time
  at 41 pairs per point x 100,000 points x 12 blocks, unmeasured), it would have to be forced on FN too for a fair step,
  and it carries W1-B1 under bfloat16 autocast (`k[bj].to(acc)` sums the gathered gradients in bfloat16: dv 29 % off
  float64). With the fix, **the eager read runs only as the float32 reference of the check** (on float32 copies of the
  inputs), so W1-B1 never enters training or the check. The gradient check (g1, g2) is added to section 3 as gating.
- A1.6 **Ports and record.** Base of the record branch `psharpe/isla-affix-ports`: 92d9d7800 (code_cmpaf_fix's archive),
  with a first commit holding code_dttposex's model.py (fc8d0fa9, which includes COMPILE-AF's 45b24fb8 change) and
  tree_read.py (00d83cb9). Commits then follow the nesting: F, FN, FNC; FNΩ's commit comes last and its tree is FN's plus
  Omega (it drops C; the message says so). Each snapshot holds its commit's isla files (md5 checked against `git show`);
  formatting by the hooks, if any, is recorded, and the single-condition forest check below is what proves F equals
  code_dttposex's forest.
- A1.7 **The whole ISLA test directory per snapshot** (section 3) passes if every failure or error is either on
  code_dttposex with the same assertion message, or in one of these classes declared now, each attributed by the
  failing test's own setup: (E-F) a golden or forest test of an ISLA-DTT model with two or more conditions (F changes
  the forest there); (E-N) a golden of a model that declares a viscosity role (FN changes its query channel); (E-C) in
  FNC, a golden of a model with two or more conditions, a golden of one condition whose tolerance is below float32
  rounding (the -log A shift cancels only to rounding), and a state-dict key, parameter-count or load test (beta is a new
  parameter); (E-Ω) in FNΩ, an ISLA-DTT golden. Anything else fails the check. The new tests must all pass.
- A1.8 **Job names and checks.** Gating GPU checks run as `isla.affix.check` (short QoS; the read-gradient tests and the
  precondition gate), cpu-short checks as `isla.affix.tests`, `isla.affix.forest` and `isla.affix.fwd`; all match the
  registry row `^isla\.affix\.` (added 08:55 EDT 10-25, expected_end 2026-10-27). Lanes `isla.affix.<arm>-seed4x` with
  <arm> in f, fn, fnc, fnomega; run ids `affix_<arm>_seed4x`; launcher `$T/kernel/affix` (AF-WHOLE's kernel/afwhole
  files with only the job-name prefix, lanes.tsv rows and code column changed), `isla.affix.smoke`, `isla.affix.score`.
- A1.9 **FNΩ's rows** are added to sections 2, 3, 5 and 6, and its predictions (A12-A17) and A18 were written before any job of the
  campaign.
