# AF-FIX design: fixing ISLA's encoding of AirfRANS's far field (2026-10-25)

Lane AF-FIX (research-lane, supervisor 01a13c2c), written 08:20-08:35 EDT 10-25. Peter's ruling of 2026-10-23 stands:
canonical ISLA reads the whole boundary, so a harm means the encoding is fixed. Evidence files: `tie/REPORT_TIE.md` and
`tie/RESULTS_TIE.md` (task 1, child AF-FIX/TIE), `prec/REPORT_PREC.md` and `prec/tables.md` (task 2's measurements, child
AF-FIX/PREC), `PRESSURE_PROBE.md` and `out/readmass_tables.md` (task 3), `PREREG_DRAFT.md` (task 4), `JOBS.md`.
"Measured" means a probe on real cases; "inferred" means my reading.

## 1. The octant tie (task 1)

**Measured** on all 200 full_test cases of AirfRANS with its far field, posed by the scorer's `airfrans_uvolume_2d_posed`,
through the real forward (a wrapper on `BoundaryForest.build`; the groups equal the forest's own `tree_cells` in every
case and both poses). Code: (a) b86c7391d (0.13.0, octants), (b) 38a7915cc (DTT-POSE, faces with the tie band).

| cells that change group under pose (min / median / max) | (a) octants | (b) faces |
|---|---|---|
| far field, all (1,578-1,873 cells) | 263 / 492 / 812 (15.4 / 28.9 / 45.1 %), in all 200 cases | **0 in all 200 cases** |
| straight edges y = +-200, x = 200 | about 42 % of each | 0 (margin 0.996 from the band edge) |
| arc | 0 | 0 |
| airfoil | 0 / 2 / 9 | 1 cell in 3 cases |

AF-WHOLE's 0.9.0 numbers reproduce exactly (median 29 %). **The face grouping removes the far-field tie.** Where faces
can still tie (all measured):
- the band edge: 3 airfoil cells near +-45 degrees in 200 cases, because tol and the gap both move under pose (tol up to
  16 %); arc cells at +-135 degrees came within 1.1e-6 of the edge and did not flip;
- the sign of the chosen axis is not certified for 4-10 tiny trailing-edge and wake cells per case (tol 0.6-1.8, because
  tol is measured from the whole boundary's centroid, 28 chords from the airfoil); none flipped;
- the frame's axis-1 sign: 0 flips, but in one case |skew_1| = 1.5e-8 lies below its own pose change (2.7e-8);
  `CANONICAL_SKEW_TIE` = 1e-9 is about 100x below the measured noise.

**The forest's pair lists are still not pose-invariant with faces** (24 cases x 100,000 scorer points, share of posed
pairs with no native match): octants 0.61 / 0.71 / 0.92; faces **0.041 / 0.158 / 0.265**. 96 % of the residual is airfoil
entries: the airfoil sits 27.8 chords from the forest's origin (the whole boundary's centroid), so pose moves its tree
coordinates by 5.5e-6 to 4.1e-5 chord, more than its trailing-edge cells (5.4e-6 chord) (measured); the float32 Morton
grid over a 400-chord box adds about 2e-5 chord (inferred). One case (89.338) has a far-field walk tie (measured: the posed walk opens one outlet node for every point; inferred,
not traced: the node lies at a distance about equal to its diameter). For scale: B-DTT's octant forest left 9.5 % unmatched on airfoil-only
AirfRANS and its posed / native stayed at 1.002 (DTT-POSE DESIGN §3).

**Proposed tie-free(r) rules** (no threshold on rounded inputs is fully tie-free; these remove the structured decisions):
1. **One set of face trees per declared condition, each in coordinates about that condition's own measure-weighted
   centroid, with tol and R measured from it** (the whole boundary's canonical frame kept for the axes). Measured as a
   counterfactual: unmatched pairs 3.8e-5 / 2.6e-4 / 0.060 (whole-boundary frame; 3.0e-5 / 1.5e-4 / 0.061 with each
   condition's own frame), with 0.74-0.82x the pairs. On a boundary with one condition it is DTT-POSE's forest exactly,
   so DTT-POSE-EXP's airfoil-only and lug results carry over. No node then mixes cells of two conditions. TIE proposed the
   same with geometric connected components in place of declared conditions; on AirfRANS they coincide. I prefer
   conditions (declared, no topology pass, no welding of coincident vertices across meshes); components are the
   name-free alternative.
2. **A derived error bound on the frame's skew** in place of `CANONICAL_SKEW_TIE`: a skew inside its bound is a tie, and
   its axis takes the sign that makes det F = +1 given the certain axes (invariant under proper rigid motions). Untested;
   not needed by AirfRANS today (0 flips), but the margin is thin in one case. `canonical_frame` is shared with the token
   draw, so this touches interacting ISLA too (DTT-POSE §5's reason to leave it).
3. The band edge (3 cells in 200 cases) is a floor I would accept.

**Rule 1 changes a frozen forest.** DTT-POSE's forest is frozen (supervisor, 2026-10-24, DTT-POSE FINAL_REPORT: "Any
change to the forest's semantics needs the supervisor's approval first: `BoundaryForest.build`, ... `normal_error_bound`").
Rule 1 changes the tree grouping, the trees' origin and where tol and R are measured from, so it needs that approval
before any port is built; the campaign's F arm asks for it (`PREREG_DRAFT.md` §2, with F0 as the frozen alternative).

Side finding of TIE, checked and refuted: its interior-offset observation (posed interior translated by t_i != t_b) is an
artifact of its two separately drawn pipelines. In the scorer's own path the posed wall nodes lie at d = 0 exactly and
the interior follows the airfoil's rigid map to 5.5e-7 chord (6 cases, LHR cpu-short 381862, `out/posecheck_381862.log`).

## 2. The near-wall length (task 2)

Measured on ISLA 0.13.0 (b86c7391d) by AF-FIX/PREC, job 381746 (`prec/tables.md`). **0.13.0's geometry forms x - p in
float64 and rounds once, so the float32 compute rounding NEARWALL measured on 0.9.0 has gone** (about 3e-7 at most in every
channel, frame and dataset), except where the float32 on-wall band sets d32 = 0 (none on AirfRANS in the scored frames;
25-162 of 71.8 M AeroZoo points). The registered precondition (PREREG_CONTINGENCY §1: compute p99.9 <= 0.1 on AirfRANS,
native and posed) therefore passes for every candidate by about 10^5, and no longer separates them. Pose invariance and
near-wall resolution do.

| option | definition (0.13.0 vocabulary) | AirfRANS with far field: first-node t; linear share | gate (AirfRANS native / posed p99.9) | pose change p99.9 / max (AirfRANS posed) | AeroZoo | DrivAerML Volume | lug, Tensile2d | bitwise unchanged | application-agnostic | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| i (today) | eps = 1e-6 L | 0.0069; 20 % | 1.7e-7 / 1.6e-7 pass | 3.3e-4 / 7.4e-4 | max 0.063-0.17 from band flips | 0.59 % off-wall linear | none linear | everything | yes | the harm's measured suspect |
| **ii (viscous length from declared scales)** | constructor argument `distance_softening: {nu: 1, U: -1}`, a monomial of declared scales in the `units` grammar; eps = its value / L per case in float64; None = i. AirfRANS: `nu: {rank: 0, defines: nu}` (the SetGlobalField leaf, consumed, not an input) and `U_inf` `defines: U` as now; AeroZoo, if it opts in: `physics.kinematic_viscosity` `defines: nu`, `bc_velocity` `defines: U` | **8.3; 0.56 % (wall nodes only)** | 1.8e-7 / 1.7e-7 pass | 0.018 / 0.10 | changes every case if declared (eps 8.5e-10 to 1.6e-4 m); p99.9 1.8e-7, max 1.5 / 2.8 on band-flip points (below float32's coordinate resolution on the highest-Re cases) | unchanged (no viscosity declared; the files' nu = 1.0 is a placeholder) | unchanged | by construction (not measured): DrivAerML, the lug, Tensile2d, every surface task, and AeroZoo unless it opts in | yes: no fluid name in model.py; the names live in the configs | **recommended** |
| F1 (precision floor, stored frame) | eps = 1e-6 max(\|q\|, max\|v\| of the closest cell) | 9.1; 0.56 % | pass | **8.4 / 12** (AeroZoo 2.2 / 5.4; DrivAerML 2.4 / 4.4 at T = L_body) | changes | changes | changes | none | yes, but reads the frame origin | rejected: fails pose (the leading edge sits at the stored origin, eps down to 6e-12 chord) |
| F2 (precision floor about the centroid) | eps = 1e-6 max\|x - c\|, c the boundary's measure-weighted centroid | **0.077; 6.0 %** | pass | 0.0027 / 0.0079 | changes (posed max 0.62) | changes | changes | none | yes | rejected: pose-invariant, but the centroid is 28 chords from the airfoil, so eps = 2.8e-5 chord stays above the first nodes |
| M (multiscale, per-condition gauges) | one more channel per condition k, asinh(d_k / (1e-6 L_k)) | airfoil channel 4.9; 0.56 % | pass | 0.018 / 0.10 | **open pieces' channels change sign with the frame: inlet / outlet pose change 31-35** (d_k of a lone sheet is signed by its own orientation about the origin) | changes | changes; a one-cell condition (Tensile2d `pinned`) has L_k = its cell's size, so re-meshing changes it | none (input width grows, checkpoints incompatible) | yes | rejected unless rebuilt with an unsigned or whole-boundary-signed d_k, and it still fails discretization on one-cell conditions |

Notes:
- ii is NEARWALL's (ii) with the Reynolds plumbing removed: 0.13.0 moved the Reynolds number to the recipe (`re_z`) and
  dropped the viscosity role, so the model needs a generic way to name a length built from declared scales. The `units`
  grammar already is that (outputs use it: nut `{U: 1, L: 1}`), and `defines` already lets a rank-0 global supply a scale
  without becoming an input. So ii adds one constructor argument and no vocabulary. The function equals NEARWALL's
  asinh(d U / nu) exactly where both are defined. The model cannot check that the monomial is a length (ISLA's scales carry
  no dimensions); the config's author vouches for it, as for every `units` entry.
- ii is two codes, chosen by what the config declares (NEARWALL's objection). In 0.13.0 that is how every scale works:
  undeclared scales leave the model unscaled. I do not think it is a flag.
- The precision trade NEARWALL reported for 0.9.0 (ii at 3.1 / 5.9 at T = 0.1 L / L) is now only the float32 on-wall band
  and the input's own rounding (input p99.9 0.41 / 1.8 at T = 0.1 L / L). No eps can repair them below u32 |x|; they matter only
  for a case stored far from its origin.
- Today's i reaches 0.16-0.17 on AeroZoo at T = L_body and T = L (band flips), above the 0.12 that `soft_log_length`'s
  docstring states. A docstring fix, whatever else is chosen.
- A floored ii (eps = max(nu / U, 1e-6 L), NEARWALL's vii) equals i on AirfRANS with its far field: no fix.

**Recommendation: ii.** It resolves the AirfRANS first nodes (t 8.3 instead of 0.0069) independently of where the far
field was drawn, passes the gate, changes no dataset that does not declare it, and puts no fluid term in model.py. Peter decides adoption.

**NEARWALL's registered contingency is superseded, not carried out** (the supervisor to confirm). It registered arm V =
interacting W-ISLA on `code_nearwall_ii` against W-ISLA, with a W-DTT pair only as an option. 0.13.0 removed interacting
interior ISLA ("query_independent gone, interacting interior ISLA with it"), so V tests a model that no longer exists. The
campaign's FN is a new test, on ISLA-DTT, nested after F, where the softening reaches only the points' seed, and its
registration predicts mostly "no effect" (A4-A6). An FN "no effect" is therefore weaker evidence for ii than NEARWALL's V
would have been for interacting ISLA. For ISLA-DTT note that its
tree read already codes log(r / |d_q|) unsoftened, so the softening reaches only the points' seed; the read itself is
79-97 % far field in W-DTT (`PRESSURE_PROBE.md`), a second suspect for W-DTT's near-wall harm.

## 3. The pressure harm (task 3)

Summary of `PRESSURE_PROBE.md` (job 381714, the eight trained runs, 16 full_test cases each):
- **Measured:** the far field takes the slices and the read. W-DTT: on average over the slices, 99.75 % of a slice's mass is far
  field; 0.4 of 512 slices have an airfoil majority; the anchors a mid-field point reads lie about 60 chords away, and the tree read gives
  the airfoil 10-12 % of the attention even 1e-6 chord off the wall (B-DTT: 100 %). W-ISLA: the airfoil holds 0.09 % of
  the slices' mass (no slice has an airfoil majority); the prediction points' own routing mass fills the slices (63 %).
  The routing logits could undo the measure's 6.6 nats (at least 20 available: the cosine part alone spans
  20, and the geometric bias is unbounded); training did not.
- **Inferred:** that this causes the 0.01-0.1 chord pressure harm. The across-case correlation does not support it
  (little variation to test), and the inference-time reweighting is off-distribution (pressure errors rise 2.3-18x in W-ISLA and 23-55x in
  W-DTT). Only a retrain
  can test it.
- **Proposed fix C:** each declared condition its own quadrature, with learned shares: log w_i = log a_i - log A_k +
  beta_k (beta init 0, so conditions start equal). Principle: a measure-weighted mean over a truncated domain tends to the
  far field alone as the truncation radius R grows; with C each condition's share no longer goes to zero. What C does not
  fix: the length unit L (the whole boundary's RMS pairwise distance) still grows with R, so every relational invariant
  in units of L (slice anchors, the read's r and log r) still depends on R. C enters the blocks and the read's per-pair
  log m only; the forest is built from the raw measures (with rule 1 a node holds one condition, so C shifts a node's
  log m by a constant and leaves its virtual cell unchanged). Changes every multi-condition
  dataset in both tasks; nothing stays bitwise after training. Alternative Omega (ISLA-DTT's read only): an unprojected solid-angle
  prior a / r^(d-1) as the read's reference weight; approximately in the read's function class (the learned kernel can
  approximate -(d-1) log r), narrower blast radius, does not fix the slices.

## 4. The campaign (task 4)

`PREREG_DRAFT.md`: ISLA-DTT on AirfRANS with its far field, nested arms on 0.9.0 ports so that AF-WHOLE's W-DTT stays the
control: F (DTT-POSE forest + rule 1's per-condition trees), FN (+ ii), FNC (+ C); two seeds; AF-WHOLE's rules per step;
about 700 GPU-h; lanes end about 10-26 21:00 to 10-27 05:00 EDT if six lanes run at once, so the read lands at the scope
freeze. An AeroZoo no-harm check of the bundle (about 545 GPU-h, about 68 h of wall per lane) slips past the freeze
either way (run in parallel from 10-25 afternoon it would end about 10-28 midday).

## 5. What needs Peter

1. **ii, the viscous near-wall length** (`distance_softening: {nu: 1, U: -1}`, opt-in by declared scales). Yes or no to
   adopting it if the campaign's FN reads no harm, and whether AeroZoo should declare it too (one rule across flows; it
   changes every AeroZoo case).
2. **C, per-condition quadratures with learned shares**: the largest change. It alters what "measure-weighted" means
   in every block and every dataset with more than one condition. If FNC helps on AirfRANS, adopting it needs checks on
   every dataset family.
3. **Rule 1, the forest per declared condition**: a small extension of DTT-POSE (bitwise DTT-POSE on single-condition
   boundaries). Names or connected components is a taste question. The forest is frozen by the supervisor, so the
   supervisor approves the change before it is built.
4. For the supervisor: confirm that NEARWALL's contingency (interacting W-ISLA arm V) is superseded by the campaign's FN.
