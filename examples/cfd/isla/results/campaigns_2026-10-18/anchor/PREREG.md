# E3b ANCHOR pilot on AirfRANS: pre-registration (2026-10-18, written before any lane is submitted)

## Question

Can a query-independent ISLA whose interior writers are **anchors derived from the boundary alone** (Lambert-LOG,
`REPORT.md` §1) match interacting ISLA on AirfRANS uniform, where the requested points themselves write into the
slices? Secondary: does it beat ISLA-DTT (boundary read) and the passive floor, and is any shortfall due to the
anchors' *law* (diagnostic arm with CFD nodes as anchors) or to the writers' *role*?

## Code and setup

- Branch `psharpe/isla-anchor` (worktree `~/gh/physicsnemo-isla-anchor`) off `psharpe/isla` ea397e04f (A2), head
  abf8e1d5f. ISLA suite on CPU: 216 passed, 51 skipped (CUDA); recipe suite: 507 passed, 12 skipped.
- One snapshot for every arm: LHR `$T/code_anc` (td015 inside) + `$T/recipe_anc`, MD5SUMS md5 de1affeb… / 4c6beb38….
- Every lane copies the `afu_a2s_isla` row of `$T/kernel/airfrans_u/lanes.tsv` (airfrans_uvolume_2d,
  isla_volume_airfrans, 512/12/512, compile_blocks, lr 3.75e-4, 100,000 points, 500 epochs, full_train / full_test)
  except the arm's settings. Seeds 42 and 43. Table: `$T/kernel/anchor_af/lanes.tsv`.

| arm | lanes | settings beyond the a2s row |
|---|---|---|
| A interacting ISLA (control) | anc_isla_seed42/43 | none |
| B passive | anc_passive_* | `query_independent=true n_anchors=0` (no writers but the boundary, no read) |
| C ANCHOR | anc_anchor_* | `query_independent=true n_anchors=50000 anchor_boundary_budget=1000 anchor_extent=domain_diameter` |
| D ISLA-DTT | anc_dtt_* | `query_independent=true` (the current class's boundary read) |
| E mesh-node diagnostic | anc_mesh_* | dataset `airfrans_uvolume_2d_meshanchor`; `query_independent=true n_anchors=50000 +forward_kwargs.anchor_points=boundaries.anchor_nodes` |

### Choices I made (please confirm with the PREREG)

1. **t_max = 565.685 chords, from the far-field boundary as data.** The converted `.pdmsh` files keep only the
   airfoil; the raw AirfRANS cases' `_freestream.vtp` is the square [-200, 200]² in all 1,000 cases (checked on LHR
   2026-10-18; bbox diagonal 565.685 / 565.686). Its diameter is recorded in the three `airfrans_uvolume_2d*.yaml`
   files as the global leaf `domain_diameter` (SetGlobalField, last transform); only C reads it. Same rule as a
   closed domain (diameter of the union of the boundaries). Not a multiple of the body size. Cost: with
   h ≈ 2e-3 c, about 40 % of C's anchors fall beyond the prediction crop (≈ 3.7 c), 8 % beyond the far field's
   inscribed circle; REPORT §3 E4 shows coverage near the body barely changes out to 256× the diameter.
2. **Boundary budget N_b = 1000 for h = A / N_b** (2-D). The AirfRANS reader takes the airfoil whole (843–1,217
   segments per case, mesh-dependent), so the recipe has no boundary budget; 1000 is the nominal resolution
   (h ≈ 2.0e-3 c). Ruling (2) says the budget, not the count; this sets the number. **This is the one choice the
   rulings do not settle.**
3. **M = 50,000 anchors, Q = 100,000 supervised points** (the same supervision as A). Evidence: QI arm C on
   AirfRANS, 50k-node supports gave 1.01 against the full support. 151k tokens per case.
4. **E's anchors**: SplitAnchorNodes moves a uniform random 50,000 of the stored interior nodes into
   `boundaries.anchor_nodes` before the 100,000 prediction points are drawn from the rest (disjoint; fresh each
   epoch; the inference seed fixes them at eval). They use C's anchor embedding (built-in wall-distance channel
   only) and are all kept. Research-only: needs the volume mesh.
5. Implementation notes: anchors split across boundary meshes by largest remainder of M·N_k/N, parents by measure
   within each mesh (ruling 3); fresh anchors every training step, a fixed stream (seed 0) in eval (ruling 4); the
   fluid-side test uses the model's own nearest-token search; rejected candidates are kept as passive tokens
   (`log_w = -inf`), so shapes are fixed. In 2-D `normalize(n + s)` is uniform in angle over the half-plane, not
   cosine-weighted (it is the cosine law in 3-D); this is the construction REPORT E2 prototyped.
   `n_anchors=None` keeps every existing configuration's parameters and state dict, so the checkpoint version is
   unchanged (no mapper needed).

### Rulings recorded with the approval (supervisor, received 2026-10-18 23:59 EDT)

- t_max = 565.685 c confirmed: the far field is part of the boundary-value problem, so it enters as data.
- N_b = 1000 confirmed **for the pilot only**. Open point before any adoption: h must not follow the input mesh
  (otherwise refinement keeps adding anchors in ever-thinner wall layers and the anchor law does not converge), so it
  must come from a model budget; the single-input design's `boundary_tokens` budget would give a different h. A later
  h-sensitivity check is planned.
- Cost note: on AirfRANS (≈1k boundary tokens) C costs 1.45× and D 1.41× an A step, with C's peak memory 54.7 GB
  against A's 37.6; the AeroZoo estimate was C ≈ 1.2× against D's 1.9×, so the cost case for anchors rests on AeroZoo.

## Metrics

- **AF-UNIFORM protocol** (`$T/kernel/anchor_af/score.sbatch`, a copy of the AF-SCORE scorer): float32 inference of
  the final checkpoint, full_test (200 airfoils), 100,000 points, native and posed (area runs too; uninformative
  here). Per-field mean rel-L2 of pressure, velocity, nut; seed means.
- **QUERY-POP probe** on every final checkpoint (`$T/kernel/anchor_af/qpop/`): conditions native_h, half,
  far-field companions, wall-heavy companions, 4 chunks, full; ratio of the mean rel-L2 to native_h and the largest
  prediction change ÷ RMS target on a fixed random half of the native points. Validity: `full` reproduces infer.py
  to ≤ 1e-4 relative.
- **Cost**: step time (median over epoch-0 steps 10–200 and a mid-run epoch) and peak memory (`Mem:` in train.log).

## Rules (decided now)

- **Validity.** C's QUERY-POP ratios must be 1 and its changes at float32 roundoff (≤ 1e-4 of the RMS target) in
  every condition. If not, C's accuracy is not read; the bug is reported.
- **Proceed.** C proceeds to the AeroZoo stage if the geometric mean of the seed-mean ratios C/A over pressure and
  velocity is ≤ 1.05 and C/A on nut (seed means) is ≤ 1.10 (native protocol). See the amendment below.
- **Does not replace.** C does not replace interacting ISLA if C/A ≥ 1.10 on pressure or on velocity with both seeds
  agreeing (each seed's C/A on that field ≥ 1.10, paired by seed).
- **Otherwise Peter decides.**
- C vs D and C vs B are reported (ratios per field), as is E vs A and C vs E (law vs role, REPORT P7).
- Caveat recorded now: the two a2s seeds differ by 6 % on pressure, 5 % on velocity and 9 % on nut, so a two-seed
  mean ratio carries about ±4 % noise; the 1.05 line is inside it.

### Amendment before submission (2026-10-19, before any lane was submitted)

- Old proceed condition: C/A ≤ 1.05 on pressure **and** on velocity, and ≤ 1.10 on nut (seed means).
- New proceed condition: geometric mean over pressure and velocity of the seed-mean ratios C/A ≤ 1.05, and C/A on
  nut ≤ 1.10.
- Reason: the two a2s seeds differ by 5–9 % per field, so a per-field 1.05 line on two-seed means sits inside the
  noise; pooling the two primary fields halves the chance that one field's noise alone decides. The validity check
  stays first; the does-not-replace rule is unchanged.

### Clarification of the validity check (2026-10-19 01:30 EDT, after approval and launch, before any final result)

The probe's smoke (subagent jobs 275463/275575/275632, float32, 1-epoch checkpoints) gave the ANCHOR arm ratios of
exactly 1.000000 but largest changes up to 2.1e-4 of the RMS target on one case (passive, DTT, mesh ≤ 6e-5; ISLA
1.5). The same probe in float64 with eager blocks (`QPOP_FLOAT64=1`, job 276051, 12 cases) gives 1.4e-13 for ANCHOR,
5e-14 for passive and 1.8 for ISLA, so the float32 excess is roundoff (151k tokens, compiled blocks). The validity
check therefore reads: ratios 1 to 1e-4 in float32 **and** changes ≤ 1e-9 of the RMS target in the float64 probe
(the float32 changes are reported, not gated). This is how "within float tolerance" is measured, not a change of
the accuracy rules. To be confirmed by the supervisor.

## Predictions (my probabilities)

| # | prediction | p |
|---|---|---|
| P1 | C, B, D and E pass the validity check; A's far / wall / chunk ratios differ from 1 by > 1 % on some field | 0.97 |
| P2 | "Proceed" holds for C (amended rule) | 0.45 |
| P2b | "Does not replace" holds | 0.25 (so "Peter decides" ≈ 0.30) |
| P3 | C ≤ D on all three fields (seed means). The pre-A2 DTT trailed ISLA by 11 / 26 / 21 % (p / v / nut) | 0.65 |
| P4 | B is the worst arm on velocity, and B/A ≥ 1.10 on velocity | 0.70 |
| P5 | E/A ≤ 1.05 on pressure and velocity (CFD-law writers recover ISLA) | 0.55 |
| P6 | C closes ≥ 50 % of the B−A gap on every field | 0.55 |
| P7 | if E ≈ A (≤ 1.05) and C/E > 1.05 on some field, the anchor law is at fault, not the writers' role | (diagnostic, no p) |
| P8 | full-run median step time: C and E ≤ 1.6× A, B ≤ 1.05× A (smokes: 1.45×, 1.45×, 1.00×; D 1.41×) | 0.80 |
| P9 | posed / native within 0.5 % for every arm and field (exact covariance) | 0.95 |

## Smokes (done before this pre-registration was sent)

One epoch on full_train_n100 (25 steps per rank, 4 GB300) plus a validation pass on full_test and the final
checkpoint, through the lanes' own launcher (`lanes.sbatch`, NO_CHAIN) into `$T/kernel/anchor_af/smoke_runs/`;
jobs 275344, 275345, 275469 (normal QOS; 275224 in short was cancelled as it started, after a long QOSGrpNodeLimit
wait). GPU pytest of the ANCHOR and CUDA tree-read tests on code_anc: 25 passed (CUDA float32 query independence
included). The mesh-node arm first failed (the datapipe's generator is on CUDA in the training loader);
fixed in abf8e1d5f (recipe only), snapshot updated in place before any lane, smoke rerun green.

| arm | median step (steps 8–25) | × A | peak mem (GB, rank 0) |
|---|---|---|---|
| A isla | 0.455 s | 1.00 | 37.6 |
| B passive | 0.455 s | 1.00 | 37.6 |
| C anchor | 0.661 s | 1.45 | 54.7 |
| D dtt | 0.641 s | 1.41 | 41.5 |
| E mesh | 0.661 s | 1.45 | 54.7 |

## Compute

10 lanes on LHR, QOS normal, 03:59 singleton chain links. From the smokes (200 steps per epoch, 500 epochs, +10 %
for validation): A ≈ 56, B ≈ 56, C ≈ 81, D ≈ 78, E ≈ 81 GPU-h per lane, ≈ 700 GPU-h in all (+ ≈ 20 for scoring and
the probe); ≈ 14 h (A, B) to ≈ 20 h (C, D, E) of node time per lane, so ≈ 1–1.5 days of wall with queueing. Two launch waves about 13 minutes apart: wave 1 A, C, E (6 lanes), wave 2 B, D (4 lanes).

## Supervisor confirmation (2026-10-19 01:29 EDT, after launch, before any result)

The float64 form of the query-independence validity check (largest change at most 1e-9 of the RMS target in the float64
probe; float32 ratios 1 to within 1e-4) is confirmed. It was added after launch because the float32 probe smoke changed one
case by 2.1e-4 while the same probe in float64 changed it by 1.4e-13: the excess is float32 summation roundoff, not
dependence on the requested points. Exact independence is a mathematical property of the design, so float64 is the right
instrument. The lanes were already training when this was added, but no lane result had been seen.
