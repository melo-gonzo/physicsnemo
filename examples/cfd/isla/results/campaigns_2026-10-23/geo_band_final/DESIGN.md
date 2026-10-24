# GEO-BAND design: derived, frame-robust tie and on-wall decisions in ISLA's exact query geometry (2026-10-24)

Base: `psharpe/isla` at 921472488 (geometry.py 7472ecf0). Evidence: `RESULTS.md` (phase A, jobs 368478 and 368608),
`DERIVATION.md` (item 1), `PHASE0.md` (cost baseline). Every number below was measured on 921472488's geometry.py with
one switch changed at a time, ops renamed, fast and reference both run.

## One finding first: in float32 the band is the inputs' storage rounding, and that term cannot be centred

The band is E_k = u (|x| + R_k) + C u64 (|x| + R_k). For float32 inputs the first term is 1 - 1.7e-7 of it. It bounds
the rounding of the coordinates as they were given, so it must be measured in the frame they were given in: a posed
float32 case really is rounded more coarsely, by u32 times the translation. Measuring it about the boundary's centroid
makes it smaller than the rounding it stands for, and then exact ties stop being ties under the pose (GEO-THIN's case
1909, reproduced: 170 lug nodes with |du| = 1 when both terms are centred). Centring only the query's magnitude (cA)
keeps the vertex term but is not a bound either, and it moves native outputs wherever the centroid is far from the
queries: AirfRANS's centroid is the far field's centre, so 216 near-airfoil nodes become on-wall natively, and
DrivAerML gains 12,296 ties over two draws with |du| up to 1.

So the brief's expectation that a centroid-measured band removes case 1884's band-edge node holds only for cA, which I
do not recommend. With the storage term kept in the given frame (as the brief allows), centring the arithmetic share
changes no float32 decision, and node 6997 stays a band-edge node under the scorer's pose. That node is answered
correctly in both poses: untied natively (gap / band 1.12), tied posed (0.86), each certified for its own inputs; u is
discontinuous across the band edge by construction. Rotation alone, with no translation, still flips u at band edges
on DrivAerML (180 to 213 nodes per 400k under every variant), because re-rounding moves the gap between two features by
about the band itself. No threshold rule removes those flips; only a u that is continuous in gap / band would (not
proposed: it changes the function's definition at every near-tie, a model change of its own).

## The rule (one canonical rule, no flag): variant VF of phase A

G1. **Derived constants** (`DERIVATION.md`). The arithmetic share is C u64 (|x'| + R'_k), C = 91 in three dimensions
(worst case 89: the accurate face foot 88 units, not 77; plus the switch threshold, 1; plus 1 for the centring and the
second-order terms), C = 17 in two (an edge point, 15). `_ERICSON_UNITS = 13` and `_FOOT_UNITS = 12` are replaced by
the derived values with their derivations in the comments; the switch threshold stays 1 unit and is certified by
construction (C >= threshold + 88). K = 168 is unchanged (a worst case, second-order terms included).

G2. **Centred arithmetic.** After the unique cells are known, c = the measure-weighted centroid of the unique cells of
nonzero measure (float64 measures and midpoints, summed exactly in int64 fixed point, so c is a function of the set of
cells, bitwise, whatever their order), rounded to the queries' precision. Every float64 coordinate of the search
(queries, cell vertices; the BVH, the seed, the ball search, Ericson's and the accurate weights, the pair distances,
the winding number) is taken less c; the float32 copies the BVH and the 2-D brute force read are the centred values cast
once. For float32 inputs the subtraction is exact in float64. The exact predicates (`_zero_area`) and the normals the
on-wall rule and tie directions read (`n64`) stay on the coordinates as given.

G3. **The band:** E_k = u (|x| + R_k) [storage, as given] + C u64 (|x'| + R'_k) [arithmetic, centred], used wherever
`coef` is now (seed bound, radius, 2-D margin, pair errors, ties, on-wall). In float32 it differs from today's by
1.45e-7 relative; in float64 it widens from 14 to 92 units where |x| ~ |x'|.

G4. **Certified distinct directions.** Replaces `distinct = (tc != tc[head])`: two tie directions are distinct when
|c_i - c_j| > tau_i + tau_j, tau = sqrt(2) s (2 where s >= 1), s a bound on the sine of the computed direction's error
against the exact direction of the stored inputs: for a normal, the vertices' storage rounding u|v| through the
certified cross product (`_thin_bounds`' delta and |X| - |E|) plus the float64 normal's rounding; for a point
direction, (E_k + Ericson's misplacement on a face) / D_k. A tie needs one certified-distinct pair; the least-norm point
is then taken over all of the group's directions, as now.

G5. **Float64 rebuild of the differentiable closest point.** `_closest_features` returns `diff = (x - v0) - sum_{i>0}
w_i (v_i - v0)` evaluated in float64 from the coordinates as given and cast once, in place of `closest`;
`exact_boundary_geometry` reads it as `diff`. Autograd runs through the same expression (the envelope gradient is
unchanged up to rounding). This removes NEARWALL's compute rounding.

## What it changes (float32, the recipe's precision; full tables in RESULTS.md)

- Native, against 921472488: no winding sign and no on-wall decision changes on any dataset. d changes at rounding
  level from G5 (DrivAerML 266k of 400k queries, |dd| <= 4.8e-7, relative <= 1e-3; AirfRANS relative <= 0.7 % at the
  first nodes, where the old float32 cancellation was that wrong); u changes up to 6.5e-3 at 7,932 DrivAerML queries
  (G4: near-coincident directions no longer tie), 1.6e-5 on the lug, 6.2e-4 on AirfRANS, 0 on Tensile2d.
- Float64 queries: 141 DrivAerML ties change (|du| <= 1.7e-3), 43 lug, 3 AeroZoo; posed tie flips 6 -> 0 on DrivAerML.
- Pose (dumps, translation one RMS radius): tie flips lug 90,221 -> 25, Tensile2d 231 -> 0, AeroZoo 76 -> 8; u at those
  nodes moved <= 2.6e-5 either way. DrivAerML u flips > 1e-3: 3,000 -> 3,706 (G4's threshold also grows with the
  storage term; each flip is bounded by the certified direction uncertainty). End to end: lug 24 posed cases unchanged
  (median 1.55e-7, 1884 1.74e-5); AirfRANS 24 self-posed median 5.9e-5 -> 4.4e-5 (G5).
- Far from the origin (NEARWALL's compute part): DrivAerML 0.0156 -> 1.9e-7 (asinh change p99.9 at T = L), AeroZoo 7.5e-4
  -> 1.9e-7, AirfRANS off-wall queries 0.29 -> 1.8e-7 at T = 0.1 L. What remains on AirfRANS is the input's own
  rounding (4,529 queries on the wall in float32 at 30 chords).
- Cost (DrivAerML whole-boundary call, GB300): reference +13 ms (+2.8 %), fast +14 ms (+3.5 %).
- The fast path: no change of its own. G1-G4 live in `_closest_candidates_impl`, which both kernels run; G5 is outside
  the op. Fast against reference: 816 of 816 probe rows bitwise (every dataset, precision and variant).

## Versions

G1-G5 change interacting ISLA's function (float32: the rounding-level d changes and the tie changes above; more for
float64 queries), so they go into ISLA 0.12.0 with DTT-POSE's forest (psharpe/isla-dtt-pose 0f06e4065, version commit
4242962e0, frozen). I do not bump the version; the supervisor folds it at merge. The branch psharpe/isla-geo-band starts
from origin/psharpe/isla and touches geometry.py and the geometry tests only (no model.py, no tree_read.py), so it
merges with DTT-POSE's branch without conflict (checked with `git merge-tree` before I report).

## Trained models are not harmed: inference check, pre-registered here

- Checkpoints: TIE-EXP's `tie_lug_isla_seed{42,43}` and `tie_t2dz_isla_seed{42,43}` (epoch 500; 0.11.0's geometry rules;
  `$T/runs/`), scored as TIE-EXP scored them (kernel/tie2's scorer on `$T/code_tie2`, 200k cells, float32, eager; lug
  600 cases native and posed, Tensile2d 100 cases native and posed), once with the model's `exact_boundary_geometry`
  replaced by 921472488's (O) and once by the branch head's (N), each a renamed-op module, in separate processes, the
  process printing which ran.
- Preflight: O reproduces TIE-EXP's registered scores (RESULTS of isla-tie-2026-10-22) to 4 significant figures;
  otherwise I stop and report.
- Criterion, per dataset, seed, protocol and scored field: |log(score_N / score_O)| <= 0.01, and the posed / native ratio
  moves by at most 0.01. 0.01 is the low end of the program's two-seed gaps (1-16 %, LANE_RULES) and 1/17 of TIE-EXP's
  lug sigma (0.176). Met on every row: no retrain. Exceeded on any row: I report it and propose retraining that
  dataset's two seeds; you decide.
- Reported beside, not gated: per case the largest prediction change over the case's RMS.
- Expected (from the change table): lug and Tensile2d move by about 1e-6 relative; the check passes.
- G5's largest changes are on AirfRANS and DrivAerML, not in TIE-EXP's datasets. If you name one canonical whole-boundary
  ISLA checkpoint of each, I run them under the same criterion (one GB300 short job).

## Phase C plan (after approval)

1. Worktree `/home/psharpe/gh/physicsnemo-isla-geo-band` on new branch psharpe/isla-geo-band off origin/psharpe/isla; a
   lane-writer implements G1-G5 in geometry.py from `probes/variant.py` (VF) and this design, with the derivations as
   comments (budget 3 h).
2. Tests, new `test_isla_geometry_band.py`: (a) far from the origin (a mesh translated by 1e3 L, float32 queries at 1e-4
   L: |d32 - d_exact| within a few u32 d; fails at 921472488); (b) pose of a planar star (a flat fan rotated in float32:
   the on-wall centre query is untied in both poses, u the plane normal; fails at 921472488's bitwise test); (c) float64
   queries: the derived arithmetic share holds against exact rational distances on random thin cells and caps (a guard);
   (d) renumbering: cells and vertices permuted, d and u bitwise (a guard of the fixed-point centroid; 921472488 already
   passes it); (e) the fast path bitwise on (a)-(d), CUDA. Existing tests that pin `_ERICSON_UNITS` / `_FOOT_UNITS` are
   updated to the derived constants.
3. Probe re-run at the branch head against `probes/variant.py`'s VF: bitwise on every dump (so the implementation is
   the measured rule).
4. The whole ISLA directory and the recipe tests on cpu-short; the geometry, geometry-fast, degenerate, thin, band and
   release files on GB300 under auto and forced; the inference check above.
5. `git commit -s`, hooks on; plain push of the new branch; no merge.

## Decisions requested

1. The storage term in the frame the inputs are given in (VF, recommended: a true bound; case 1884's node stays a
   band-edge node, answered correctly in each pose), or the query's magnitude about the centroid (VFA: removes case
   1884's excess and cuts DrivAerML's translated flips 3,706 -> 1,353, but is not a bound and changes 216 native
   AirfRANS on-wall decisions and 12k DrivAerML ties).
2. G4 (certified distinct test) in or out. Recommended in: it removes the planar-star membership flips (lug 90k -> 25)
   at the price of 24 % more DrivAerML u flips under a translated pose.
3. The inference check's criterion as above, and whether to add one AirfRANS and one DrivAerML checkpoint.
