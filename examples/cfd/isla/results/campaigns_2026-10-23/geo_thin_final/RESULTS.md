# GEO-THIN results (phase A; 2026-10-24, from 06:10 EDT)

Code under test: `$T/code_geodegen3` (= `1086cf42e`, geometry.py md5 `dd492ac1`), whose geometry.py, geometry_fast.py and
geo_kernel.py equal `psharpe/isla` at `7213c70c5` (FAST-MERGE's merge; checked 05:00). Every probe prints the md5 of the
geometry.py it ran and the op name each kernel called (`isla_closest_candidates` / `_fast`); cross-variant probes load
each variant with both ops renamed (LANE_RULES). Jobs: `JOBS.md`. LHR `T=.../agents/2026-08-09-mt2-stage0`, `K=$T/kernel/geothin`.

## Notation

For a triangle: L its longest edge, A its area, rho = 2A / L^2 (the altitude over the longest edge, over L: a sliver or a
needle has rho << 1), R_cell its greatest vertex norm. u is the unit roundoff of the queries' precision (2^-24 for
float32, 2^-53 for float64), u64 = 2^-53. The op's bound E = (u + 13 u64)(|x| + R). Failure index of Ericson's closest
point, from its mechanism (below): Phi_u = u64 L / (rho^2 (u + 13 u64) 2 R_cell), its misplacement of the closest point
(about u64 L / rho^2 for a query on the cell) over the on-wall band of a query at the cell. The census's `W` columns are
8 Phi (`lhr/census.py`, docstring).

## A1. Census of the stored boundary meshes (`lhr/census.py`, job 366291; table `work/census_table.md`, raw `lhr/out/census/`)

All meshes are float32. Per mesh, after the weld. HiLift: one wing (284.6M cells) in chunks, thinness only, no weld.

| meshes (cases) | rho min | cells with Phi32 >= 1.25 (>= 12.5) | cells with Phi64 >= 1.25 | normal error > 4 eps32 | coplanar folds / axis-plane overlaps | other |
|---|---|---|---|---|---|---|
| DrivAerML Volume whole boundary (dump + run_2, run_100 stored) | 0.038 | 0 | 0 | 0 | 0 / 0 | 110 exact T-junction vertices on the tunnel floor's edges (dump) |
| DrivAerML Volume vehicle (dump), Surface vehicle (run_1, 2, 100, 200, 300) | 0.037 | 0 | 0 | 0 | 0 / 0 | 7.4k-7.7k valence-1 edges |
| DrivAerML run_1 single-solid STL (not read by ISLA) | 5.1e-5 | 0 | 23,246 | 0 | 0 / 0 | |
| SHIFT-SUV estate vehicle (runs 1, 2, 4, 5; run 3 has no vehicle file) | 0.039 | 0 | 0 | 0 | 0 / 0 | 3,190 valence-1 edges |
| AeroZoo (the recipe's dump) | 6.5e-7 | 2 (1) | 156 | 0 | 0 / 0 | 4 near folds (3e-4 rad from flat), 4 edges of valence 4 |
| AeroZoo stored cases 0, 1, 3 (domain boundaries) | 1.7e-6 | 15, 0, 0 | 2590, 4, 57 | 0 | 0 / 0 | |
| lug (20 cases) | 0.29 | 0 | 0 | 0 | 0 / 0 | |
| DeepJEB++ (20 cases, 1 in 3000) | 1.7e-4 | 0 | 1,169-1,779 per case | 0 | 0 / 0 | |
| HiLift wing geo_LHC001_AoA_10 (surface task) | 2.5e-11 | 18,774 (11,842) | 26.5M | 0 | (HL-DUP's) | 228,548 zero-area cells |
| AirfRANS airfoil, Tensile2d (2-D) | segments | - | - | - | 0 folds, 0 zero-length | |

- No stored mesh has a cell whose float64 normal is uncertain beyond 4 float32 ulps (max 6.4e-6 eps32), so the plane
  bound's failure does not occur on stored data.
- Surface tasks (DrivAerML Surface, SHIFT-SUV, HiLift, DeepJEB++) call no exact geometry; only DrivAerML Volume,
  AeroZoo, lug, AirfRANS and Tensile2d do.
- AirfRANS's far field (AF-WHOLE's facts: a closed chain in all 1,000 cases) is a closed polyline of segments; 2-D has no
  thin cells.

## A2. The exact geometry against an exact rational brute force on the recipe's draws (`lhr/real_check.py`)

Jobs 366241 (distance, cell, sign) and 366627 (plus the op's own pair distance, its on-wall decision, and u against the
exact direction for untied off-wall queries). Every query of each dump runs through the op (raises counted); the brute
force checks a sample: all queries for AirfRANS, Tensile2d and the 20 lug cases; for DrivAerML and AeroZoo 3,000-4,000
random + 1/4 on-wall + every query a cell with rho < 1e-3 could hold (rigorous lower bound).

| | queries run | checked | raises | wrong d | wrong cell | wrong sign (of) | pair distance > 2E (max) | on-wall wrong | u wrong (of) |
|---|---|---|---|---|---|---|---|---|---|
| float32, reference and fast | 1,083,000+ | 310,545 | 0 | 0 | 0 | 0 (244,388) | 0 (0.00 E) | 0 | 0 (16,240) |
| float64, reference and fast | same | 310,545 | 0 | 0 | 0 | 0 (244,388) | 0 (0.18 E) | 0 | 0 (18,342) |

No standing and no final rerun is exposed through the draws checked. AeroZoo's 4 near folds: the recipe's queries within
10 cell sizes of them (13-39 each) get bitwise the same d and u without the thinner cell (`lhr/out/real2_366627/folds.log`).
### A2b. Adversarial queries on real meshes (`probes/gen_realthin.py` + `real_check.py`; jobs 366781, 366823)

Per mesh, the 50 cells of largest Phi32; on each, 40 queries on the face and 60 off it along its normal at 1e-2, 1 and
1e2 times its Ericson misplacement, both sides; 5,000 queries, cast to float32 (then run in float32 and in float64).

| mesh | cells Phi32 > 1 / Phi64 > 1 | float32 (both kernels) | float64 (both kernels) |
|---|---|---|---|
| AeroZoo, the recipe's dump | 2 / 161 | 0 wrong d, cell, sign, pair distance (max 0.01 E), u | device-side assert (no candidate), 366781 and 366823 |
| AeroZoo stored case 0 | 17 / 3,074 | 0 wrong (max 0.10 E) | device-side assert, both kernels |
| AeroZoo stored case 3 | 0 / 70 | 0 wrong | 0 wrong d, cell, sign; 5 pair distances beyond 2E (max 64 E), not decisive |
| DrivAerML whole boundary | 0 / 0 | 0 wrong | 0 wrong |

So a stored mesh breaks the reference when queried in float64 (CUDA assert, CPU IndexError), and float32 queries
survive even on its worst cells. The recipe runs float32 queries (or bfloat16 cast to float32), so no standing is
exposed. The `wall` column of real_check.py (12 and 26 in float32, 7 in float64 on DrivAerML) is the check's own
artefact: it used the global R for the on-wall band, where the op uses each cell's R_k; the op's pair distance is within
0.1 E there, so its on-wall decision is its own rule applied correctly.

## A3. Mechanisms on adversarial sets (`probes/probe_mech.py`, jobs 366479, 366629, 366756)

Variants (ops renamed): asis; noplane (the float32 lower bound off); nocorner (the float64 Ericson rejection off);
neither; exactw (both off, the closest-point weights exact); minedge (both off, the nearer of Ericson's point and the
edges'). Sets: GEO-DEGEN's sliver soups (10 per offset h); the unit cube with a cap or needle of width h in its bottom
face (axis-aligned; unshifted and offset 1024; and randomly rotated and shifted, 3 rotations each); coplanar overlaps;
TIE-FIX's set. Tables: `work/mech_totals.txt` (soups), `work/mech2_totals.txt` (boxes, overlaps), `work/mech3_totals.txt`
(rotated boxes). The first box run (366479) had an inconsistently oriented cube: its sign counts are void.

| set | precision | asis | which step |
|---|---|---|---|
| soups, sliver offset 1e-16 / 1e-13 | float64 | 10 / 8 of 10 raise | plane bound and float64 rejection drop the true cell (no candidate: IndexError); both off: 65 / 3 wrong d (weights) |
| soups 1e-10, 1e-7 | both | 0 | |
| rotated cap or needle, rho 1e-3 ... 1e-6 | float64 | 3 of 3 raise at every rho | as above; both off: 2,079-2,733 on-wall wrong per 10,227 queries, up to 3,072 wrong d (weights) |
| rotated cap, rho 1e-6 | float32 | 844 wrong cells, 1,727 on-wall wrong of 10,227 | weights (exactw: 0) |
| rotated needle, rho 1e-5 / 1e-6 | float32 | 1 / 21 wrong cells, 47 / 435 on-wall wrong | weights (exactw: 0) |
| rotated cap or needle, rho >= 1e-4 (needle >= 1e-3... see table) | float32 | 0 | |
| axis-aligned caps, needles, bent caps, rho 1e-7 ... 1e-13 | float64 | 296-388 wrong d, up to 388 wrong cells per 2,209 | weights, and the float64 rejection (stage "rejected") |
| axis-aligned, float32 | float32 | 0 wrong; at offset 1024 the cap rounds onto the edge (a T-junction): u changes at 115 queries, up to 0.56 | T-junction (tie rule) |

With exact weights and both prunings off, every thin-cell set is correct (0 raises, 0 wrong d, cells, signs, on-wall).
The winding number gave no wrong sign on any clearly off-wall query (d* > 2E) in any set; a query on a thin face that the
op wrongly takes as off the wall gets an arbitrary sign, which is the on-wall failure, not the winding number's.

Why: Ericson's face-region weights are ratios of Gram determinants va, vb, vc ~ rho^2 L^4 computed from products of size
L^2 l^2 (l the query's distance to the vertices), so their float64 rounding moves the closest point along the face by
about u64 l^2 / (rho^2 L); for a query on or near the face that exceeds the on-wall band, and the pair distance is too
large (the true cell is rejected by the ball search's float64 test, loses to another cell, or the query is taken as off
the wall, which reverses u). The plane bound fails only where the float64 normal itself is uncertain (rotated caps in
float64; never on stored data).

Coplanar overlaps (nonzero area, no zero-area cell): same orientation: u changes at ties in float32 at offset 1024 (3-9
queries of 2,511, up to 0.21), never in float64, d correct. Opposite orientation (a partial fold, or a reversed cell inside
a face): the cells cancel in w by the documented fold rule; against the cube without them, 21-181 queries change u by up
to 1 and 67-115 change the sign of d (first run). TIE-FIX's float32 set: 5 of 1,946 queries change u, up to 0.39, d correct
(the cast midpoint makes a sliver along the edge: a near-coplanar overlap).

## A4. The least-norm tie direction at near-duplicate normals (`probes/probe_ties.py`, job 366479 `ties.log`)

- Real tie nodes (20 lug cases, Tensile2d): every vertex moved by one ulp of float32 at random, the geometry rerun (3
  trials): |du| at tied queries max 3.9e-5 (lug), 6.4e-6 (Tensile2d); |du| / |dn| (the normals' own change) at most 2.6;
  the active set of the least-norm point changed for 0 of the persisting groups. The tie flag itself is decided by
  rounding: 1,639 tied lug queries become 5,777-5,823 (planar stars whose normals were bitwise equal stop being so), with
  u continuous through it.
- `_min_norm_points` alone, each captured group's normals perturbed at s = u64 and u32: max |dx| / s 1.3e6 and 113 (lug),
  2.2 and 1.0 (Tensile2d); active set changed 0 times.
- Synthetic n1, n2 = n1 rotated by theta, n3 at beta: |dx| / s grows as 1/theta until theta ~ sqrt(s), where the active
  set flips (35 of 64 trials at theta 1e-7, s = u64) and |dx| reaches about sqrt(s) (Hoelder-1/2: near-duplicate normals
  make a thin triangle in the space of normals). The value stays the least-norm point either way.

## T-junctions on DrivAerML's whole boundary (supervisor's condition 3; `probes/tjunc.py`, job 366938)

On the canonical dumps after the boundary merge (isla_volume, seeds 42 and 43; 200,000 queries each): 110 T-junction
vertices on 74 valence-1 edges of the tunnel floor (exact collinearity). Queries whose closest point is a T-junction
vertex (within the op's bound): 0 of 200,000 in each draw. With every T-junction removed (each cell over such an edge
split into a fan at its T vertices; the same set), d, u and the sign are bitwise unchanged at all 200,000 queries of
each draw, under both kernels (seed 42) and the reference (seed 43). `lhr/out/dumps_366938/tjunc_*.log`.

## Phase C checks of 91bd3236e (jobs 367730, 367731, 367849)

Every model-visible output (d, u, the winding sign, the tie and on-wall decisions) and every gradient is bitwise unchanged on every real dump under both kernels; the op's internal closest-point weights differ at 11 of 200,000 DrivAerML whole-boundary queries per draw, all on-wall queries on ordinary tunnel-floor cells, by 1-2 ulps, and the new point is closer to exact in 20 of 22 rows.

- Check a (against 7213c70c5, ops renamed): GB300 104 of 104 small and canonical-dump cases bitwise under each kernel
  (AirfRANS with its far field, 4 draws; AeroZoo; the 20 lug cases; Tensile2d; icosphere, torus, ellipse), and
  DrivAerML whole boundary (2 draws x 2 precisions, 200k queries) bitwise except the weights above; CPU one thread 60
  of 60 bitwise under each kernel.
- Adversarial float64 queries on the stored AeroZoo meshes: 0 raises, 0 wrong (the base: CUDA assert).
- Adversarial CPU sets: 0 raises, 0 wrong d, cell, on-wall or sign of 517,296 queries (86 set-precisions).
- Fast against reference: 0 of 2,558,096 queries and 0 of 16.7M candidate pairs differ; FAST-PLUGIN 0 of 552,953.
- GPU test files (geometry, tree-read, geometry-fast, degenerate, thin, release): 278 passed, 1 skipped, forced and auto.
- Cost: reference 412 to 474 ms, fast 343 to 405 ms.

## Final head 6bac57cbd (merge of psharpe/isla cce0fb53d), snapshot `$T/code_geothin3`

- GB300 (368128): the geometry, tree-read, geometry-fast, degenerate, thin and release files, 282 passed, 1 skipped, both
  forced (FAST-PLUGIN: 0 of 584,953 queries differ) and under auto. Check a against 7213c70c5: 104 of 104 small and
  canonical cases bitwise under each kernel (AirfRANS with its far field, 4 draws; AeroZoo; the 20 lug cases;
  Tensile2d), and DrivAerML whole boundary (2 draws x 2 precisions x 200k) bitwise except the 11 internal weight rows
  per draw, as at 91bd3236e.
- cpu-short: the whole ISLA directory (368129) and the recipe tests: counts in FINAL_REPORT.

## Cost by part (GB300, 367918; `probes/prof_parts.py`, synchronized timers, median of 5 after 3 warm-up)

| dump (200k queries) | kernel | call 7213c70c5 -> head (ms) | ball search | _closest_weights (of which _accurate_weights) | _thin_bounds | guarded pairs |
|---|---|---|---|---|---|---|
| DrivAerML whole boundary | reference | 409 -> 472 | 232 -> 266 | 4.1 -> 31.3 (23.7) | 8.5 | 12,243 of 1,287,536 (0.95 %) |
| DrivAerML whole boundary | fast | 338 -> 400 | 87 -> 119 | 4.0 -> 31.7 (24.4) | 8.5 | same |
| DrivAerML vehicle | reference | 337 -> 394 | 151 -> 178 | 4.1 -> 34.7 (27.4) | 8.5 | 7,921 of 964,764 (0.82 %) |
| AeroZoo | reference | 142 -> 173 | 91 -> 92 | 4.0 -> 33.7 (26.7) | 2.4 | 93 of 493,449 (0.02 %) |

F3's cost is per call, not per row (26.7 ms for 93 AeroZoo rows, 23.7 ms for 12,243 DrivAerML rows): the accurate
point's expansion arithmetic is about a hundred small torch kernels, launched twice (the seed and the pairs). F2's cost
is the extra per-cell record and arithmetic in the traversal kernels (+32 to +34 ms on DrivAerML, +1 ms on AeroZoo).

## Would a tighter certified bound on Ericson's error help? (GB300, 368137; `probes/probe_guard.py`, analysis only)

_ERICSON_K scaled by f (as a bound f times tighter would be); guarded pairs and the call's time:

| dump | f = 1 | 0.1 | 0.01 | 0.001 |
|---|---|---|---|---|
| DrivAerML whole, reference | 12,243 (0.95 %), 469 ms | 4,590, 470 ms | 4,231, 473 ms | 0, 449 ms |
| DrivAerML whole, fast | 12,243, 403 ms | 4,590, 402 ms | 4,231, 404 ms | 0, 375 ms |
| DrivAerML vehicle, reference | 7,921 (0.82 %), 386 ms | 303, 386 ms | 6, 375 ms | 0, 364 ms |
| AeroZoo, reference | 93, 165 ms | 78, 165 ms | 55, 166 ms | 36, 166 ms |
| lug (first case, 20k), reference | 23,548 (33.5 %), 63 ms | 20,200, 63 ms | 371, 64 ms | 0, 37 ms |

- Time drops only where no pair is guarded at all (the launches are skipped), and then by the accurate point's cost
  (20-27 ms). That takes a bound 1000 times tighter than K = 168 on DrivAerML and the lug. The largest error observed
  against exact rational arithmetic is 0.0115 of the bound, so a valid bound can be at most about 87 times tighter. No
  certified bound reaches zero guarded pairs on these data.
- On DrivAerML the pairs still guarded at f = 0.01 are on-wall queries on the tunnel's large floor cells (rho about
  0.42, L / (|x| + R_k) about 0.75): Ericson's first-order error there is about u64 L / rho^2, several units of
  u64 (|x| + R_k) whatever the constant, against the guard's one unit. On AeroZoo the guarded pairs are the genuinely
  thin cells (rho median 4e-5), which the guard exists for.
- So the cost lever is the implementation, not the bound: one fused kernel for the guard and the accurate point (a Warp
  or Triton function over the guarded rows) would remove the launch overhead, about 25 ms per call on every dataset,
  and F2's per-cell record could be packed into the existing corner records. Recorded for the 0.12.0 follow-up; nothing
  committed (the supervisor: do not narrow the switch threshold).

## Fourth limit: the medial-axis band edge (lug case 1884; `probes/probe_1884.py`, cpu-short 368087 and 368125)

DTT-POSE's random-weight forward on the scorer's lug set differs posed against native by 1.74e-5 relative RMS on case
1884 (median 1.5e-7 over 24 cases). The cause is one node of the query geometry:
- Node 6997 (interior, d = -5.19 mm): its two nearest distinct features are the interior of cell 4526 and an edge of
  cell 1854. Exact gap 2.61e-8 native and 2.90e-8 posed (the pose re-rounds the float32 coordinates). The tie band
  E_0 + E_1 = (u32 + 13 u64)(2|x| + R_0 + R_1) is 2.32e-8 native and 3.36e-8 posed, because its |x| term follows the
  translation. Gap over band 1.12 (untied: u the face normal) against 0.86 (tied: u the midpoint of two directions 90
  degrees apart). |du| = 0.707 at that one node of 11,520, which is the whole forward excess (0.707 / sqrt(11,520) =
  0.0066 = RMS |du|). d moves by at most 3e-8, with 0 sign flips and 0 on-wall changes. 1 band-edge node in DTT-POSE's
  24 cases.
- Tie membership differs at many more nodes without moving u. Native against posed: tied 1,394 against 7,317 on case
  1884; 1,070 against 5,252 on 1859 and 2,018 against 7,326 on 1909 (4,204 and 5,334 nodes differ on the last two).
  Yet u moves by at most 1.5e-5 there: these are on-wall stars whose tied normals nearly coincide (the
  near-duplicate-normal limit, A4). The pose breaks the bitwise equality of normals that left a planar star untied.
- The band about the boundary's area-weighted centroid c (analysis variants of geometry.py, ops renamed, 368125):

| band | case 1884: max abs du (nodes above 1e-3) | 1859 | 1909 | tie membership differing (1884 / 1859 / 1909) |
|---|---|---|---|---|
| as is: (u32 + 13 u64)(abs x + R_k) | 0.707 (1) | 8.9e-6 (0) | 1.5e-5 (0) | 5,951 / 4,204 / 5,334 |
| cA: abs x replaced by abs(x - c) | 2.2e-5 (0) | 8.9e-6 (0) | 1.5e-5 (0) | 5,950 / 4,204 / 5,334 |
| cB: abs x and R_k both about c | 2.2e-5 (0) | 8.9e-6 (0) | 1.0 (9) | 5,950 / 4,204 / 5,340 |

  - cA removes node 6997's flip (untied in both poses), as the supervisor predicted. It does not change the membership
    counts: they come from the bitwise distinct-direction test at planar stars, not from the band's frame.
  - cB, which also measures R_k about the centroid, makes the band smaller than the float32 storage rounding of the
    vertices in the frame they are given. On case 1909 nine mid-plane nodes of the plate (exactly equidistant from two
    opposite faces natively: tied, u = 0) see a re-rounding gap of up to 3.6e-9 in the posed frame, which the smaller
    band no longer covers: untied, |du| = 1.
  - So a pose-invariant band must still cover the inputs' own storage rounding, u32 times the stored coordinates'
    magnitude in their own frame: the vertex term stays absolute (as in cA). The irreducible part is the inputs' float32
    storage rounding under the pose. For the follow-up lane.
