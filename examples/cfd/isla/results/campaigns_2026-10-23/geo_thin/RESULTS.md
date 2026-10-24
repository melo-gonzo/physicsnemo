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
