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

## 10. Amendment before submission (A2, 2026-10-25 11:41 EDT, the supervisor's)

Written before any lane, and before the smoke (384133, running) has reported; no number of any trained arm exists.
- A2.1 (supervisor, 11:40 EDT 10-25) The §3 CPU-forward check "with one condition C's forward equals code_affix_fn's to
  float32 rounding (1e-6 relative)" becomes: **"with one condition, C's forward equals code_affix_fn's in float64 to
  1e-12 relative"** (outputs, max |diff| / max |ref| per field). Reason: the 1e-6 float32 bar underestimated rounding
  through 12 blocks. Measured beside (float32, LHR cpu-short 383954, 2 airfoil-only cases, reference width, 20k points):
  max 1.5e-6 to 6.1e-6, rms 4e-7 to 1.9e-6. The float64 run (384134): 2e-15 to 1.1e-14 (passes).
- A2.2 (record) FNC's port was fixed twice before any job of an arm: a56c10c92 (the log shares in a submodule registered
  last, so parameters() keeps every other parameter's order; ae181604f put beta first and shifted the release goldens'
  randomization) and **3171904db** (beta gathered by a one-hot mask, no scatter in its backward, after an Inductor CPU
  compile failure in test_isla_compile [global_source-cpu]; forward bitwise a56c10c92's; the shift also in
  _forward_tensors). code_affix_fnc is 3171904db. Snapshots of the superseded commits are kept read-only under
  `$T/affix_campaign/superseded_code_affix_fnc_*`, never trained.

## 11. Addendum A3 (2026-10-27, lane AF-CONFIRM): arm FC, F plus C without N

Written by lane AF-CONFIRM (research-lane, supervisor 01a13c2c; folder `scratch/isla-af-confirm-2026-10-27`) at about
09:00 EDT 10-27, before any job of arm FC: no FC snapshot, check, smoke, lane or score exists. **What the writer has
seen:** every AF-CAMPAIGN record, including all scores and reads of F, FN, FNC and FNΩ (RESULTS.md, read.md,
FINAL_REPORT.md), W-DTT's and B-DTT's scores, and the near-wall bands. Sections 1-10 stand as written; nothing here
changes a registered verdict. The question: FNC / FN measured C only on top of N, and N harmed nut (FN / F 1.156);
**does C still help on F alone?**

### A3.1 Arm and control

| arm | code | change against the control | reads against |
|---|---|---|---|
| **FC** | `$T/code_affix_fc`, from **5ca0d8137** on branch `psharpe/isla-affix-fc` (pushed, never merged), its parent F's c0fa1fc28 | C exactly as FNC's 3171904db (the per-condition log shares log a_i - log A_k + beta_k, beta in the submodule registered last, gathered by a one-hot mask; the read's log-m gradient fix, segment_softmax.py cb9ad796), without N (no change to the distance channel; test_isla_affix_softening.py absent) | F (`affix_f_seed42/43`, the campaign's lanes, final checkpoints `ISLA.0.500.mdlus` kept, scores in `scores/evals_affix`) |

- How C was separated from N (commit message of 5ca0d8137): the tree diff from FN (4c604956b) to FNC (3171904db) was
  applied three-way on F; four model.py hunks conflicted where C's lines sit beside N's softening, and each kept C's
  lines and dropped N's. FC's model.py differs from FNC's exactly as F's differs from FN's (N's lines only).
- Snapshot: as section 3 and A1.6 (lhr/build_snap.sh): `cp -r --preserve=timestamps $T/code_dttposex` plus the commit's
  isla model and test directories (git archive), FNC's rule for segment_softmax.py (the branch's, with the fix),
  SNAPSHOT_HEAD, MD5SUMS checked against `git archive`, `chmod -R a-w`. Overlay proof: `diff -rq` against code_affix_fnc
  (expected: model.py, test_isla_affix_softening.py absent, MD5SUMS, SNAPSHOT_HEAD) and against code_affix_f.
- Configuration: F's lane rows exactly (`$T/kernel/affix/lanes.tsv` rows 0-1), only run_id (`affix_fc_seed42/43`) and
  the code column changed. Launcher `$T/kernel/affixfc` = a copy of `$T/kernel/affix` with only the job-name prefix
  (`isla.affixfc.`), the lanes.tsv rows, the scorer and near-wall rows and their output roots (`$T/evals_affixfc`,
  `$T/evals_affixfc_nw`) changed; md5s logged. Seeds 42 and 43.
- Paired reading valid: C's beta is created as zeros (no RNG draw) after every other parameter; the smoke confirms that
  FC's step-0 parameters (beta aside) and first batches equal F's.

### A3.2 Checks before the lanes (section 3 as amended; every job `isla.affixfc.*`, registry row `^isla\.affixfc\.`)

- (c1) The whole ISLA test directory of code_affix_fc on LHR cpu-short: passes under A1.7, the allowed failures being E1
  and class E-C (expected: exactly FNC 3171904db's two, the parameter count and the reference checkpoint's missing key
  condition_log_shares.beta); `test_isla_affix_quadrature.py` and `test_isla_affix_forest.py` pass.
- (c2) On a GB300 (short QoS): `test_isla_affix_read_grad.py` (g1, g2) of the snapshot passes; **FC's forest equals F's
  bitwise** (probe_affix.py captures: the 24 far-field cases x 2 protocols against the campaign's `cap/afw_f`, and the 4
  airfoil-only cases against `cap/af1_f`; every field and the pair lists). C must not reach the forest.
- (c3) CPU forwards, 2 cases each: all finite; FC's boundary log-weights on beta = 0 equal F's minus log A_k of their
  condition (as cmp_fwd.py's C check); **with one condition FC's outputs equal F's in float64 to 1e-12 relative** (A2.1;
  float32 beside); the forest inside forward equals F's.
- (c4) Smoke (one short-QoS job, 4 GPUs): units the W-DTT control row (affix_ctl:2), F seed 42 (affix:0) and FC seed
  42 (affixfc:0), 50 steps each. Passes on rc 0, finite losses, FC's resolved config equal to F's except run_id and the
  run paths, the step-0 parameters (beta aside) and the first 3 batches equal across the three units, every read on the
  Triton path. **Cost rule:** FC's median step above 1.05x F's unit in the same job goes back to the supervisor before
  the lanes (FC / ctl beside).
- A failed check stops the lane before any lane job, reported to the supervisor.

### A3.3 Scoring and rules

- Scoring as section 4, exactly F's: the same scorer (score.sbatch) and rows with fc for f (native, area, posed; float32;
  full_test 200 cases; the epoch-500 checkpoint), near-wall dumps for FC, the reduction over FC, F, W-DTT and B-DTT. Every
  check of section 4 applies (epoch 500, 200 rows with W-DTT's sample ids, SNAPSHOT_USED code_affix_fc, DATASET_USED,
  recipe_afwhole, the freestream guard, the near-wall points hash and the dumps reproducing the scorer). The event guard
  as section 4: stop and tell the supervisor; never resubmit.
- **R1, FC against F, per field (pressure, in-plane velocity, nut), exactly section 5:** D = ln(mean_s CM(FC) / mean_s
  CM(F)), SE = sqrt((s_FC^2 + s_F^2) / 2); helps if D < ln 0.90 and |D| > 2 SE, harms if D > ln 1.10 and |D| > 2 SE, else
  no effect; helps (paired) if r_42 and r_43 < 1/1.10, harms (paired) if both > 1.10 (counts as harms); the sub-bar flag.
  Labels and thresholds unchanged. Posed beside.
- R2 (descriptive) for FC / F as section 5 (decades, the split at 3.05e-4 chords, the share of W-DTT's band gap closed).
- R3 pose for FC, section 5's rule verbatim. R4 cost: smoke step against F's; lanes' GPU-h against F's.
- **Beside, never in a rule:** FC / FNC per field (seed means, with R1's machinery printed descriptively: the N step
  measured on top of C, in reverse) and FC / B-DTT per field (seed means).
- Independent numbers: each R1 cell holds 4 CMs (2 seeds x 2 arms); F's two runs are shared with the registered F / W-DTT
  and FN / F steps; the three fields and the posed column come from the same runs.
- What a reading leads to (no decision of mine; section 5's mapping for C, applied to FC): **FC / F helps on pressure
  with no field harmed: C (without N) goes to Peter with its blast radius (every multi-condition dataset), and the AeroZoo
  no-harm check (section 7, with FC for FNC, launched only on the supervisor's steer) follows.** Any field harms: C is
  not proposed without N. Helps on other fields only, or no field helps: reported as read, with no proposal of mine; the
  supervisor decides. A lost seed or row: incomplete as registered, escalated.
- The read: `harness/read_affix.py` copied to `scratch/isla-af-confirm-2026-10-27/harness/read_affix_fc.py`, extended
  only to add the FC / F step (R1, posed, R2, R3) and the two beside rows; its md5 is recorded in that folder's JOBS.md
  before the read, and it must reproduce the campaign's read.md tables for the four registered steps unchanged.

### A3.4 Predictions (before any FC job)

| id | prediction | p |
|---|---|---|
| B1 | checks (c1)-(c3) pass on their first run | 0.85 |
| B2 | the smoke passes first time, cost rule included | 0.85 |
| B3 | FC's smoke step within [0.98, 1.04] of F's | 0.85 |
| B4 | FC / F pressure: helps / no effect / harms | 0.75 / 0.23 / 0.02 |
| B5 | FC / F velocity: helps / no effect / harms | 0.72 / 0.26 / 0.02 |
| B6 | FC / F nut: helps / no effect / harms | 0.65 / 0.32 / 0.03 |
| B7 | FC / F harms on some field | 0.05 |
| B8 | FC / F helps on all three fields | 0.55 |
| B9 | FC R3: pose holds or at B-DTT's level | 0.85 |
| B10 | beside: FC / FNC below 1 on nut (seed means) | 0.60 |
| B11 | beside: FC / FNC within [0.85, 1.15] on all three fields | 0.60 |
| B12 | beside: FC / B-DTT below 1.10 on pressure (seed means) | 0.60 |

Why: FNC / FN was 0.610 / 0.640 / 0.653 with |D|/2SE 2.06-2.58, and FNC / F (descriptive, from the campaign's CMs) is
0.672 / 0.698 / 0.755. If C and N act independently in log error, FC / F is near the first; if C's gain partly undid N's
harm, near the second. Both are well below 1/1.10, and F's and FNC's seed spreads are small (F pressure 0.0178 / 0.0186),
so helps is the mode on every field, nut least sure (F already helps nut against W-DTT, and part of C's nut gain may be
the repair of N's harm, B10). The measured mechanism (the far field's 99.86 % of the measure taking the slices' and the
read's mass) does not involve N.

### A3.5 Cost

Two lanes at F's and FNC's measured cost (114.3 and 114.7-115.2 GPU-h per lane), about 230 GPU-h, plus smoke about 0.4,
GPU check about 0.2, scoring about 0.7 and near-wall about 0.3: **about 232 GPU-h against the lane's Part A cap of 260**.
Past 260 the lanes are not cancelled; the overrun is reported to the supervisor. Intermediate checkpoints of the FC lanes
are pruned to their final ones after the read (`$T/kernel/prune/prune_intermediates.py`, dry run first, logged).

## 12. Amendment A4 (2026-10-27, lane AF-CONFIRM): Part B, the AeroZoo no-harm check with FC

Written by lane AF-CONFIRM at about 10:20 EDT 10-27, before any job of Part B: no code_affix_fc_r090 snapshot, check,
smoke, lane or score exists, and no FC lane on AirfRANS has a scored number yet (A3's lanes 423644/5 are queued or in
their first link). **Authorised** by the supervisor's steer of 09:4x EDT 10-27, after Peter's ruling, verbatim: "Adopt
rule 1 now; continue investigating C but then ask me for a verdict." Peter also ruled N inadmissible (LANE_RULES,
2026-10-27). **What the writer has seen:** everything A3 lists, AZ-DECIDE's tables (`tables_final.md`: the standings'
five test-side fields per seed), AZ-RESCORE's tables, COMPILE-AF's results, CERT's verdicts, and the standings' run
configs and sacct cost. This section replaces section 7's check by the same check with FC for FNC, because N is not a
candidate. Section 7's other lines stand.

### A4.1 Arm, control and execution

| arm | code | against |
|---|---|---|
| **FC-AZ** (`azfc_full_fc_seed42/43`) | `$T/code_affix_fc_r090`, from **e066a8101** on `psharpe/isla-affix-fc-r090` (pushed, never merged), its parent FC's 5ca0d8137 | the standings `azc_full_dtt_seed42/43` (code_r090, epoch 250) |

- **Code:** code_r090 (archive of 92d9d7800) plus DTT-POSE's forest, rule 1 (F) and C, nothing else. The affix ports
  start from code_dttposex, whose model.py also carries COMPILE-AF's block compile options (mix-order reduction off,
  code_cmpaf_fix); e066a8101 removes exactly that hunk (the code_r090 -> code_cmpaf_fix diff), so the compiled blocks
  get the standings' Inductor options. COMPILE-AF measured that AeroZoo takes the fused geometry kernel, which leaves no
  mix-order reduction to fuse (its section 2: unaffected, at most 4.9e-4). segment_softmax.py is the branch's (the log-m
  gradient fix, cb9ad796, as code_affix_fc).
- **Snapshot:** `cp -r --preserve=timestamps $T/code_r090` plus the commit's isla model and test directories (git
  archive), SNAPSHOT_HEAD, MD5SUMS checked against `git archive`, `chmod -R a-w`; `diff -rq` against code_r090 and
  code_affix_fc logged (expected against code_affix_fc: model.py only, by the compile hunk).
- **Configuration and execution: the standings' exactly.** Their `kernel/azi/lanes.tsv` rows 14-15 (aerozoo_uvolume,
  isla_volume_aerozoo, `train_split=full.train val_split=full.val +training.val_interval=10 dataloader.num_workers=12
  dataloader.prefetch_factor=12 ++model.query_independent=true`, lr 3.75e-4, 200,000 points, 250 epochs), recipe_r090,
  `.venv-recipe`, bfloat16, on LHR, with the config's `model.compile_blocks: true` (the standings trained their blocks
  compiled; `compile=false` for the whole model, as theirs). Only run_id and the code column change. Launcher
  `$T/kernel/affixfcaz` = a copy of `$T/kernel/azi` (lanes.sbatch byte-identical, 8eecec8f, the same file as
  kernel/affix's) with the job-name prefix `isla.affixfcaz.`, the two rows, and the node-list exclude of the five bad
  nodes (the standings ran before that list; it changes no computation). Seeds 42 and 43.
- Paired reading: beta is zeros, created after every other parameter, so a seed gives the standing's initialization and
  data order; the smoke checks it.

### A4.2 Checks before the lanes (jobs `isla.affixfcaz.*`, registry row `^isla\.affixfcaz\.`)

- (b1) The whole ISLA test directory of code_affix_fc_r090 on LHR cpu-short: the failures and errors must be exactly
  code_affix_fc's (E1 and the two E-C failures, the same assertion messages); anything else fails.
- (b2) GB300, short QoS: `test_isla_affix_read_grad.py` (g1, g2) of the snapshot passes.
- (b3) CPU forwards on 2 AeroZoo full.val cases (fwd_affix.py, recipe_r090, isla_volume_aerozoo, 20,000 points), code_r090
  and code_affix_fc_r090 in separate processes: both finite; FC's boundary log-weights on beta = 0 equal code_r090's minus
  log A_k of their condition, for each of inlet, outlet, no_slip and slip present; the forest's tree count logged
  (rule 1: face trees per condition).
- (b4) Smoke (one short-QoS job, 4 GPUs, 60 min): the standing's row on code_r090 (smoke only, from a copy launcher
  `$T/kernel/affixfcaz_ctl`) and FC-AZ seed 42, 50 steps each, compiled blocks as the lanes. Passes on rc 0, finite
  losses, FC-AZ's resolved config equal to the standing's (`runs/azc_full_dtt_seed42`) except run_id and the run paths,
  the step-0 parameters (beta aside) and first 3 batches equal across the two units, every tree read on the Triton path.
  **Cost rule:** the standings took 287.9 and 287.8 GPU-h (sacct, 21 links each). The projection 2 x 287.9 x (FC-AZ's
  median step / the control unit's) + 8 GPU-h (checks, smoke, scoring) must be at most the 600 GPU-h cap, i.e. a step
  ratio of at most 1.028; above it, the supervisor decides before any lane.
- A failed check stops Part B before any lane, reported to the supervisor.

### A4.3 Scoring and the rule

- Scoring: AZ-DECIDE's scorer (`$T/kernel/azdecide/eval_azi/eval_test.sbatch`, which produced the standings' scores in
  `p200k_fp32_test_azc`) copied with only its output root (`$T/p200k_fp32_test_affixfcaz`), its rows file, its working
  directory and the job name changed: full.test (716 cases), 200,000 points, float32, the epoch-250 checkpoint, the
  lane's own code column. **The standings are rescored in the same job** (units 2-3, code_r090), so all four rows come
  from one scorer copy and one draw. The stored AZ-DECIDE scores stand beside; if a rescore differs from its stored
  case-mean by more than 1e-3 relative on any field, I stop and report before reading.
- Checks before reading: `"epoch": 250` in each config row, 716 cases with the standings' sample ids, SNAPSHOT_USED,
  the event guard as section 4 (validation loss above 4x its running minimum after epoch 10, non-finite loss, BLOCKED, a
  hung link: stop and tell the supervisor; never resubmit).
- **R1, harms half, on every field the standings score: velocity_l2 (3-D), pressure_l2, nut_l2, tke_l2, omega_l2**
  (the scorer's case-mean relative L2, "training" space, the five fields of AZ-DECIDE's tables). D = ln(mean_s
  CM(FC-AZ) / mean_s CM(standing)), SE as section 5; **harms** if D > ln 1.10 and |D| > 2 SE, or harms (paired) if both
  seeds' ratios exceed 1.10. The helps half is printed beside, never in the rule. Velocity components x, y, z beside.
- Independent numbers: 4 CMs per field; the five fields come from the same runs.
- **What a reading leads to** (no decision of mine): **no field harms: the package (DTT-POSE's forest, rule 1 and C)
  passes section 7's no-harm check on AeroZoo, and C goes to Peter for his verdict with Part A's read.** Any field harms:
  reported as a harm of the package on AeroZoo. This is a no-harm check, not an attribution: rule 1 is already adopted,
  F and C are not separated here, and the verdict on C is Peter's. A lost seed or row: incomplete as registered,
  escalated.

### A4.4 Predictions (before any Part B job)

| id | prediction | p |
|---|---|---|
| C1 | checks (b1)-(b3) pass on their first run | 0.80 |
| C2 | the smoke passes first time (cost rule aside) | 0.75 |
| C3 | the smoke's step ratio is at most 1.028 (the cost rule passes) | 0.45 |
| C4 | no field harms | 0.72 |
| C5 | harms on velocity / pressure / nut / tke / omega | 0.06 / 0.10 / 0.10 / 0.12 / 0.08 |
| C6 | beside: R1 helps (or helps paired) on pressure | 0.35 |
| C7 | beside: R1 helps (or helps paired) on velocity | 0.30 |

Why: AeroZoo's boundary is a box (inlet, outlet, slip walls) around no_slip bodies, so, as on AirfRANS, the outer
pieces hold most of the measure, which C rescales to one share per condition; on AirfRANS C helped every field by
0.61-0.65. But the bodies' share differs, the standings are already good (velocity 0.11), and F's per-condition face
trees are untested in 3-D at this scale, so harm keeps real mass (C5). AirfRANS' smoke put F at 1.04x its control and
C at 1.005x more; 3-D forests with four conditions may cost more, hence C3 near even.

### A4.5 Cost

Two lanes at the standings' cost, 575.7 GPU-h, plus checks about 0.5, smoke about 2 and scoring about 3 (four units on
one node, about 0.6 h): **about 581 GPU-h against Part B's cap of 600**, with the smoke's cost rule above. Past 600 the
lanes are not cancelled; the overrun is reported to the supervisor. After the read, the lanes' intermediate checkpoints
are pruned to their final ones (dry run first, logged).

## 13. Amendment A5 (2026-10-27, lane AF-CONFIRM): Part B as the C-only no-harm check on code_r090 (arm C-AZ)

Written by lane AF-CONFIRM at about 12:16 EDT 10-27, before any lane of Part B, and before any check or smoke of the
new arm. Decision B1 is the supervisor's, of 12:07 EDT 10-27, after A4's smoke failed. Peter's 09:4x ruling
stands: "continue investigating C but then ask me for a verdict". Rule 1 is on hold.

**What the writer has seen since A4:**
- FC-AZ's smoke 425238: the control passed; FC-AZ ran out of memory at step 19.
- The diagnostics and probes in `scratch/isla-af-confirm-2026-10-27/ROOT_CAUSE_RULE1.md`: pair counts, forest bounds
  and the blast radius over AeroZoo, DrivAerML Volume, the lug, Tensile2d and AirfRANS.
- No Part B arm has trained, and no score of any Part B arm exists. Part A's FC lanes have not started.

A5 replaces A4's arm. Everything else in A4 stands as written.

### A5.1 Arm

| arm | code | against |
|---|---|---|
| **C-AZ** (`azfc_full_c_seed42/43`) | `$T/code_affix_c_r090`, from **646243133** on `psharpe/isla-affix-c-r090` (pushed, never merged), parent 92d9d7800 = code_r090 | the standings `azc_full_dtt_seed42/43` |

**What C-AZ is:** code_r090, with code_r090's own octant forest unchanged (no DTT-POSE forest, no rule 1), plus C.
- The C part is as in FNC's 3171904db: per-condition log shares, beta registered last and gathered by a one-hot mask,
  the same shift in `_forward_tensors`, and the read's log-m gradient fix (segment_softmax.py cb9ad796).
- One change is needed by the octant forest, whose nodes may hold several conditions. A node's log m is
  log Omega_m + logsumexp_k (log f_{m,k} + s_k), its cells' shifted measure.
  - f_{m,k} comes from float64 prefix sums, without gradient.
  - On a node of one condition it is bitwise FNC's formula.
  - Tested: pure nodes bitwise; mixed nodes equal the float64 log of the shifted measure to 2e-6; a mixed node's
    shift gradient equals its conditions' shifted shares.
- Snapshot: `cp -r code_r090` plus the commit's isla model and test directories (lhr/build_snap_c_r090.sh;
  MD5SUMS == git archive, 39 files; model c049a86b, tree_read a9388a65, segment_softmax cb9ad796).
- Configuration, execution, launcher and seeds are A4.1's, with only the code column and the run_ids changed: LHR,
  kernel/azi rows 14-15, recipe_r090, .venv-recipe, bf16, compile_blocks true with 0.9.0's Inductor options.
- Paired reading: beta starts at zero and comes last, as in A4.

### A5.2 Checks before the lanes (jobs `isla.affixfcaz.*`)

- (b1) The whole ISLA test directory of code_affix_c_r090, and of code_r090 beside it as the base, on cpu-short.
  - Allowed: code_r090's own failures and errors, with the same assertion messages.
  - Also allowed: class E-C (the parameter count, and the reference checkpoint's missing key
    condition_log_shares.beta).
  - Anything else fails the check.
  - The two affix test files must pass; read_grad skips on CPU.
- (b2) GB300: `test_isla_affix_read_grad.py` (g1, g2) passes.
- (b3) CPU forwards on 2 AeroZoo full.val cases, code_r090 and code_affix_c_r090 in separate processes, as A4's b3:
  - both finite;
  - C's log-weights are code_r090's minus log A_k, one constant per condition;
  - **the forest equals code_r090's bitwise** (every captured forest hash and the pair lists), so C does not reach the
    forest.
- (b4) The smoke, as A4's b4 with C-AZ for FC-AZ: the standing's row against C-AZ seed 42, 50 steps, compiled blocks.
  - Passes on rc 0, finite losses, configs equal except run_id and paths, step-0 parameters (beta aside) and batches
    equal, and every read on Triton.
  - **Cost rule unchanged:** a step ratio above 1.028 goes back to the supervisor.

### A5.3 Scoring, rule and mapping

A4.3 unchanged: the same scorer copy, with the standings rescored in the same job; R1's harms half on velocity,
pressure, nut, tke and omega over 716 cases; the helps half beside; mean_stable (515) beside, never gating.

Mapping:
- **No field harms:** C passes section 7's no-harm check on AeroZoo, and goes to Peter for his verdict with Part A's read.
- **Any field harms:** a harm of C on AeroZoo, which is now attributable to C alone, since the forest is the
  standings'.
- Lost seed or row: incomplete as registered.

### A5.4 Predictions (before any C-AZ job)

| id | prediction | p |
|---|---|---|
| D1 | checks (b1)-(b3) pass on their first run | 0.85 |
| D2 | the smoke passes first time (cost rule aside) | 0.85 |
| D3 | the cost rule passes (step ratio <= 1.028) | 0.80 |
| D4 | no field harms | 0.75 |
| D5 | harms on velocity / pressure / nut / tke / omega | 0.05 / 0.09 / 0.09 / 0.10 / 0.07 |
| D6 | beside: helps (R1 or paired) on pressure | 0.35 |
| D7 | beside: helps (R1 or paired) on velocity | 0.30 |

Why:
- C adds only a shift per token, plus a K-wide logsumexp per node; on AirfRANS, C cost 1.005-1.016x its control. Hence
  D3 is higher than A4's C3.
- The effect predictions follow A4's reasoning. The forest is now the standings', which removes F's untested 3-D risk,
  so harm loses a little mass.

### A5.5 Cost

As A4.5: about 581 GPU-h, cap 600. Part B has used 2.90 GPU-h so far (FC-AZ's smoke and the rule-1 probes).

### A5.6 Part A (record; A3 unchanged)

Part A's arms F and FC use rule 1 as committed in c0fa1fc28, i.e. before the eta fix (V2, 3e6cc1636, kept as the record).
On AirfRANS, V2's pair lists differ from rule 1's in 48 of 48 case-draws, though the pair counts agree to 0.3%
(ROOT_CAUSE_RULE1.md section 2). Part A's report states this.

### A5.7 Record (2026-10-27 12:40 EDT, before any lane of C-AZ): a port fix

The first runs of b1 (428109) and b4 (428114) on 646243133 failed, so neither check passed as registered.

What failed:
- (i) A condition given without cells (AeroZoo cases with slip 0) has -log A_k = inf. In a node's logsumexp, -inf + inf
  gave NaN: test_isla_conditions, every-condition-trains [isla_dtt]. It also gave NaN losses on some smoke steps.
- (ii) test_isla_tree_read's exact-read reference took the old 4-argument _tree_read_pairs.

The fix, e36111772:
- such a condition's shift is 0, and a condition that a node does not hold is masked to -inf;
- nodes of one condition are unchanged, bitwise;
- the test's exact read takes and adds the cells' shift.

The design is unchanged. The code of C-AZ is **e36111772**: snapshot code_affix_c_r090 rebuilt (model e20f5a72,
tree_read 53c950c0). The superseded snapshot is kept read-only at
`$T/affix_confirm/superseded_code_affix_c_r090_6462431`. Checks b1-b4 are rerun on it; the base run of code_r090's suite
(428110: 0 failures) stands.
