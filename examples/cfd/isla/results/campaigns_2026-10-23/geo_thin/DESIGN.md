# GEO-THIN design: thin cells of nonzero area in ISLA's exact boundary geometry (2026-10-24)

Base: `psharpe/isla` at `7213c70c5` (geometry.py md5 `dd492ac1`, as `1086cf42e`). Evidence: `RESULTS.md` (phase A).
Notation as there: rho = 2A / L^2, u the queries' unit roundoff, u64 = 2^-53, E_k = (u + 13 u64)(|x| + R_k) the op's
error bound of the distance to cell k, of which 13 u64 (|x| + R_k) is the float64 arithmetic's share (`coef`).

## What fails, and where (phase A in one paragraph)

Three steps of the reference read Ericson's float64 formulas or a float64 normal as if they were accurate on every cell;
on thin cells they are not. (1) The ball search's float32 lower bound reads the float64 normal and decides each edge
plane's outward side by a rounded sign test; where the normal is uncertain (rotated caps in float64; never on stored
data) it rejects the true cell. (2) Its float64 test rejects cells whose Ericson distance exceeds the radius; on a thin
cell that distance can be too large. (1) and (2) leave a query with no candidate: IndexError on CPU, a device-side
assert on CUDA (GEO-DEGEN's 8 of 10 soups; every float64 rotated cap down to rho = 1e-3). (3) Ericson's face-region
weights are ratios of Gram determinants ~ rho^2 L^4 built from products of size L^2 l^2, so the closest point moves along
the face by about u64 l^2 / (rho^2 L); for a query on or near the face this exceeds the on-wall band and the pair
distance is wrong: wrong d, wrong cell, or a query on the wall taken as off it (u reversed). In float32 it starts at
rho ~ 1e-6 to 1e-5 (rotated caps and needles), in float64 at rho ~ 1e-3; the failure index is Phi_u (RESULTS). The
winding number, the tie rule and the 2-D path did not fail on thin cells. Stored data: no failure on the recipe's draws
(310,545 queries checked, both precisions, both kernels). But adversarial float64 queries on the thinnest faces of the
stored AeroZoo meshes (the recipe's dump and stored case 0) raise a device-side assert under both kernels; in float32
the same queries are answered correctly (RESULTS A2b).

## Principle

Each of the three steps keeps its current arithmetic wherever that arithmetic is certified to be within the bound the
next step assumes, and otherwise uses a formulation whose error does not grow with 1/rho. No tolerance constant: every
certificate is a derived bound on IEEE float64 rounding (as GEO-DEGEN's Shewchuk filter, ONWALL's band). A choice the
certificate leaves to rounding is only ever between two answers that are both within the bound. On a cell without the
defect nothing changes, bitwise.

## F1. The ball search's float32 lower bound (`_ball_candidates`)

- Edge planes: the outward in-plane normal of the edge p0 -> p1 of a cell (a, b, c) taken cyclically is
  normalize(cross(p1 - p0, n)), n the cell's normal from (b - a) x (c - a): outward by the vertex order alone. The rounded
  test `m . (opp - p0) > 0` and its flip go. In exact arithmetic it never flips; it flips only where rounding decides it
  (rho below about the normal's error), and then wrongly. Bitwise unchanged wherever the test was certain.
- Certified normal: per cell, the a-priori bound of each component of (b - a) x (c - a) in float64, ERR (|l| + |r|)
  (Shewchuk, as `_zero_area`'s filter, which already forms these products), gives delta = |bound| / |c|, a bound on the
  angle between the computed and the exact normal. The lower bound's pad, 16 eps32 (|x| + R) = 32 u32 (|x| + R), was
  written for planes accurate to their float32 cast; its float32 arithmetic (cast of the normal and of the offset, three
  products and three sums) uses at most 9 u32 (|x| + R), and a normal error delta adds 2 delta (|x| + R). A cell with
  delta > u32 (its float64 normal is not accurate to its own float32 rounding) gets NaN planes, so the kernel never
  rejects it by them (as GEO-DEGEN's zero-area cells). Stored data: 0 cells (max delta 6.4e-6 eps32).

## F2. The ball search's float64 test (`_triangle_distance2` in `_ball_candidates_kernel`, `_ball_winding_kernel`)

The kernel rejects a cell when Ericson's float64 distance d~ exceeds the radius. It now rejects it when d~ - eps exceeds
the radius, with eps a certified bound on Ericson's error for that query and cell:
- Ericson's point misplacement is at most Delta = K u64 l^2 / (rho_low^2 L) (l the largest distance from the query to
  the cell's vertices, rho_low a lower bound of rho from the certified cross product, K from the propagation of
  rounding through d1..d6, va, vb, vc and the division; the estimate gives K about 176, which phase C derives in full,
  states in the docstring, and checks against exact rational weights on the adversarial sets, where the largest
  observed error over the bound must be below 1);
- the misplacement is along the face (the weights sum to 1), so for a face-region point d~^2 = d^2 + |misplacement|^2 and
  d~ - d <= min(Delta, Delta^2 / (2 (d~ - Delta)_+)); an edge or vertex point is computed on that edge or vertex, and a
  wrong region costs at most Delta.
- Per cell, kappa = K u64 / (rho_low^2 L) (float64; inf for a cell with no certified area) goes to the kernel beside the
  corners; per query it forms eps from l and d~. A few extra flops per tested cell.
A kept cell beyond the radius changes no output (its pair distance, computed accurately, exceeds d_min by more than the
tie band; GEO-DEGEN's argument for NaN corners), so this only adds candidates, and only on thin cells. The fast path uses
the same kernels.

## F3. The closest point on its cell (`_closest_weights`, for the seed and for every candidate pair)

Per pair, after Ericson's weights w~ (as now):
1. Guard: eps (as in F2, computed in torch from the pair's l, rho_low, L and d~) against one unit of the arithmetic
   bound, u64 (|x| + R_k). A pair with eps below it, and every pair of a cell with no thin geometry, keeps w~ bitwise.
2. Guarded pairs: the accurate closest point, the least of
   - the three edge points (`_segment_t` on the canonical pairs (0,1), (0,2), (1,2), as now: bitwise the edge points
     every neighbour computes), and
   - the face foot, in the frame of the longest edge AB: e = (B - A) / |B - A|, n the cell's normal from its exactly
     rounded cross product (TwoProduct and an expansion sum, `_zero_area`'s code, for the guarded cells only), m = n x e
     (pointing to the apex C by the vertex order), s1 = (x - A) . e, s2 = (x - A) . m, h = (x - A) . n; the weights
     lambda_C = s2 / hc (hc = |exact cross| / |B - A|), lambda_B = (s1 - lambda_C tc) / |B - A| (tc = (C - A) . e),
     lambda_A = 1 - lambda_B - lambda_C, taken when all three are >= 0. The point sum lambda v is then accurate to
     O(u64 (|x| + R_k)) whatever rho (the error of lambda_C along e is cancelled by lambda_B's), where Ericson's is
     O(u64 l^2 / (rho^2 L)).
   Its distance error e_R <= 12 u64 (|x| + R_k) (derivation in the docstring).
3. Switch: the guarded pair takes the accurate weights when the two distances differ by more than the arithmetic
   bound, |d_acc - d~| > (13 + 12) u64 (|x| + R_k): Ericson's result is then provably outside the bound the tie test
   assumes (in either direction: a point misplaced along the face can also fall short of the true distance near a thin
   edge). Otherwise it keeps w~ (Ericson within the bound of an accurate answer). Rounding decides the switch only
   between two answers that are both within the bound.
The seed's bound and the radius read the same function, so the radius stays a true upper bound. Ties, features and
`reached` read the chosen weights: an accurate edge point is bitwise the neighbours' (same call, same ordered pair).

## Not changed, with the evidence

- The winding number: no wrong sign on any clearly off-wall query (d* > 2E) in any set; its triple product loses
  accuracy only for queries inside the on-wall band of a thin face, where (with F3) the query is on the wall.
- Normals read by the on-wall and tie directions (n64): their float64 error (about u64 / rho for caps) is below their
  sensitivity to the inputs' own rounding (u |v| / (rho L)), which the tie rule already absorbs (A4).
- Overlapping coplanar cells and T-junctions (recorded as limits, counts in RESULTS): stored meshes hold 0 coplanar folds
  and 0 axis-plane overlaps; AeroZoo's 4 near folds change no recipe query; DrivAerML's whole boundary has 110 exact
  T-junction vertices on the tunnel floor's edges. Same-orientation overlaps change u at ties (float32, up to 0.21 at
  offset 1024), opposite-orientation ones flip the sign by the documented fold rule. A principled fix (the surface as a
  set: an arrangement of the coplanar cells) is neither cheap nor exposed. Not attempted.
- Near-duplicate normals at ties: the least-norm direction is Hoelder-1/2 in the normals (|dx| up to about sqrt(s) at
  theta ~ sqrt(s)), the active set flips there by rounding, but the value is the least-norm point either way; on the
  lug and Tensile2d a one-ulp move of every vertex changes u by at most 3.9e-5, 2.6 times the normals' own change.
  Recorded as a limit; no change.

## Expected effect on real data, cost, version

- Stored meshes: F1 flags 0 cells (max delta 6.4e-6 eps32); F2 keeps a few more borderline candidates (output-neutral);
  F3's switch fires only where Ericson is wrong beyond the bound, which the real-data check never saw (largest pair
  error 0.18 E in float64, 0.00 E in float32). Expected: every output bitwise unchanged on all real dumps; checked as
  GEO-DEGEN's check a (head against `7213c70c5` with renamed ops, every op output, w, d, u and gradients, CPU 1 thread
  and GB300, both kernels). If any real output changes, I report counts and the largest change, and propose batching it
  with DTT-POSE's ISLA-DTT change in one version (0.12.0); otherwise no version change and no new custom op.
- Cost on the DrivAerML whole-boundary call (16.65M cells, 200k queries; 406 ms reference, 341 ms fast today): F1 adds
  the bound's norm per cell (the products exist); F2 one float64 array per cell and a few flops per tested cell; F3 a
  guard per pair (~2.5 pairs per query) and the accurate path on guarded pairs. Expected a few ms; measured in phase C
  for both kernels.
- Tests: a new `test_isla_geometry_thin.py` (GEO-DEGEN's soups no longer raise; rotated caps and needles in float32 and
  float64 against an exact rational brute force, on and off the face; the fast path bitwise equal to the reference on
  them, CUDA; adding a coplanar cap to a closed cube leaves d unchanged), then the whole ISLA directory and the recipe
  tests on LHR cpu-short, and the geometry, geometry-fast, degenerate, thin and release files on GB300 under auto and
  forced.
