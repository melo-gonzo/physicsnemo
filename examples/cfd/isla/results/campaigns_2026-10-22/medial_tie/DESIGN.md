# MEDIAL-TIE: principled tie rules for ISLA's geometry features (design note and CPU prototype; 2026-10-22)

Status: design written, CPU prototype committed on `psharpe/isla-medial-tie`
(worktree `/home/psharpe/gh/physicsnemo-isla-medial-tie`, base `psharpe/isla` @ 817e5ddbf). Not merged, not pushed.
The experiment in section f is proposed only, for the supervisor to decide.

- `bf44c8b77` ISLA tests: ruff-format of the two test files (formatting only, AST unchanged; the hook failed on untouched lines).
- `ea49d23ea` ISLA geometry: closest-feature ties take the least-norm generalized gradient; normal octants are antipodal.

## 0. Where the two conventions live (read 2026-10-22)

- `exact_boundary_geometry` (geometry.py) returns `(d, u)`. Before this lane, `_closest_features` asked `_closest_cell`
  for one cell: `signed_distance_field` (a float32 Warp BVH) in 3-D, `argmin` over a brute force in 2-D. Both keep the
  first minimum they meet. The closest point is then rebuilt from that cell's vertices (float64 barycentric weights),
  `u = sign(plane) n_f` on a face, `(x - p)/|x - p|` on an edge or a vertex, and `d` is signed by the feature's
  pseudo-normal (vertex: angle-weighted sum of cell normals; edge: sum of the two triangles' normals).
- Call sites: model.py:1651 (`forward`, the union of the named meshes) and model.py:1771 (`forward_batched`, per
  case). Both store `wall_distance` and `wall_direction`.
- What reads `u`: only `q_nhat = sign(d) u` (model.py, `_forward_tokens`, `sign(0) = +1`). It enters (1) the query
  seed `n . g_k` (`qt.seed`); (2) the blocks as the token's `n_hat` (`rel_hat . n`, `n . m_s`, and the slice means
  `m_s` of `n_hat` when the queries are not passive); (3) the tree read's pair invariants `rho . n_q`, `n_q . n_j`,
  `n_q . u`. Every use is linear in `n_q`, and nothing renormalizes it (checked in blocks.py and tree_read.py).
  So a shortened or zero `n_q` is a well-defined input.
- `normal_octants` (tree_read.py:173) codes `n_k > 0` per component in the boundary's canonical frame. Its one
  caller is `BoundaryForest.build`, which builds one cluster tree per octant.

## a. The tie set and its tolerance

**Definition.** For a query `x` and every cell `k`, let `D_k` be the computed distance from `x` to cell `k` and `E_k`
an error bound on it. Cell `k` is in the tie set `T(x)` when both of these hold:

1. `D_k - E_k <= min_j (D_j + E_j)`: the intervals overlap, so `k` may be the closest cell. This is "the features
   within `d_min + tau`", with `tau = E_k + E_min ~ 2E`, written so that it does not depend on which minimum is
   "the" minimum.
2. The closest point `p_k` is a local minimum of the distance *on the surface*. When `p_k` lies on a vertex or an
   edge, every cell sharing that vertex or edge must reach the same point (a count against the feature's valence).
   This rule is purely topological and has no tolerance. Without it, a point beside a convex vertex v "ties" with
   the neighbouring cell, whose closest point is v itself: v lies on the cell that holds the true foot, so
   `|x - v| - D_min` is second order in the lateral offset and falls inside `E`. Measured on the float32
   translated-wall test (1000 near-wall points, offsets 1e-5 to 1e-3): without rule 2, 0.6 % of points tied at
   shifts 0 and 1, and 8.7 % (2-D) or 2.5 % (3-D) at shift 100, with |u| down to 0.76. With rule 2, none tied.
   Rule 2 keeps genuine medial ties only: distinct local minima at equal distance.

The tie is **distinct directions**: cells sharing one vertex or edge give the same direction bitwise (below) and
count once.

**The bound `E_k`, derived rather than tuned.** The coordinates arrive in precision `p` (unit roundoff
`u_p = 2^-24` in float32, `2^-53` in float64), and the prototype computes every candidate in float64.

- Representation. Each stored coordinate is within `u_p |X_i|` of the exact one, so `|x - X|_2 <= u_p |X|_2`, and
  the same holds for each vertex. The distance to a simplex is 1-Lipschitz in the query and in the vertices: moving
  every vertex by at most `eta` moves every point of the cell, a convex combination, by at most `eta`. So the
  representation term is `u_p (|x|_2 + max_{v in k} |v|_2)`.
- Arithmetic, float64, first order. The differences `x - v` and `v_i - v_j` each round once relative to
  `|x| + |v|`. The closest-point weights inside a Voronoi region enter `D` only at second order, because `D` is
  stationary in the closest point along the cell; region-boundary weights are exact zeros and ones. The combination
  `sum w_j v_j` rounds within `gamma_d max |v|`, the difference `x - p` once more, and the norm within
  `gamma_{d+1} D` with `D <= |x| + max |v|`. Summed: `(3d + 4) u_64 (|x| + max |v|)`.
- So `E_k = (u_p + (3d + 4) u_64) (|x|_2 + max_{v in k} |v|_2)`. In float32 the representation term dominates by
  `2^29`.

The representation term has to be there. Rigid-motion covariance is about the moved coordinates, rounded to `p`.
ATTRIB-090 and the 2026-10-18 note both trace the lug's medial flips to "float32 rounding of the posed
coordinates", which is this term.

**Scaling with coordinate magnitude.** `E` is absolute and grows with `M = |x| + |v|`, the distance from the
*origin*, not with the size `L` of the geometry or the wall distance `d`. In float32, `tau ~ 2E ~ 4 u_p M`:

| M / L | tau / L (float32) | relative rounding of a near-wall d = 1e-6 L (soft-log's epsilon) |
|---|---|---|
| 1 | 2.4e-7 | ~12 % (the soft_log_length docstring assumes this case) |
| 10 | 2.4e-6 | ~120 % |
| 100 | 2.4e-5 | the channel is noise |

The last column is the same bound applied to `d` itself: `|delta d| <= 2 u_p M`. That is why geometry stored in raw
metres far from its own centre is a suspect for AeroZoo's percent-level float32 floor. For example, a body at
M ~ 5 m with wall-resolved first cells at d ~ 1e-5 m has `2 u_p M / d ~ 6 %`. ISLA is frame-free, so it should
not care where the origin is. In float32 it does, because rounding is relative to the origin. The remedy belongs
in the data path, not in this rule: recentre (for example on the gauge centroid) in float64 before the cast to
float32. The in-model computation cannot undo rounding that is already in its inputs. Consequence for the tie rule:
the tie band widens with M, as it should, because the data cannot resolve the closest feature any better.

## b. The rule

Let `c_k` be candidate `k`'s unit direction from the surface to `x`: `sign(plane_k) n_k` on a face, and
`(x - p_k)/|x - p_k|` on an edge or a vertex. The gradient of the signed distance from candidate `k` is
`sign(d) c_k`. The Clarke generalized gradient of `d` at `x` is `conv{sign(d) c_k : k in T}`, and the rule takes
its **minimum-norm element**:

    nu = sign(d) m,   m = argmin { |g| : g in conv{c_k : k in T} },   and u := m at a tie.

- **Two candidates:** `m = (c_1 + c_2)/2`. They are unit vectors, so the foot of the perpendicular from 0 on their
  chord is its midpoint. `|m| = cos(theta/2)`, which is zero at a plate's mid-plane (`c_2 = -c_1`).
- **Three or more:** the projection of 0 onto the polytope. By Caratheodory, `m` lies in the relative interior of a
  face spanned by at most `d` candidates, where it is that face's least-norm affine point with non-negative weights.
  Every such feasible point lies in the hull, so `m` is the least-norm feasible point over all faces of at most `d`
  vertices. In 2-D those are pairs; in 3-D, pairs and triangles, with weights `G^{-1} 1 / 1^T G^{-1} 1` and `G` the
  Gram matrix. `m = 0` when 0 lies in the hull, which the optimality test `c_k . m >= |m|^2` (to a slack of
  `2 d eps`) detects. The prototype enumerates faces, which is exact and fine for the 2 to 6 distinct directions
  that occur. Production would use Wolfe's algorithm, or the 2-D closed form (the midpoint of the two angular
  extremes, or 0 when they span at least pi).
- **Not the mean.** The mean depends on multiplicities and on interior points. With directions at 0, 10 and 90
  degrees, the minimum-norm element is the midpoint of the 0 and 90 degree directions, (1/2, 1/2); the mean is
  pulled toward 0 degrees. The cube's three-face corner gives `m = (1, 1, 1)/3`, the centroid, because there the
  three directions are the vertices of the face nearest 0 (tested).
- **What `u` becomes:** `u = m`, no longer a unit vector, with `|m| <= 1`, and 0 when the candidates cancel (no
  preferred direction). `|d|` is unchanged: it is the minimum. `d`'s sign is the vote `sign(sum_k s_k)` of the
  candidates' pseudo-normal tests `s_k = sign((x - p_k) . N_k)`; the votes agree off the surface on a closed
  manifold. Downstream, `n_q = sign(d) m` feeds every reader listed in section 0 linearly. A zero `n_q` gives
  their odd invariants the sign-free value 0 at a mid-plane, which is what option 1 (the nematic) of
  DESIGN_NOTE_medial_ties.md wanted, without a structural change. The cost is that `|n_q| < 1` is a new signal:
  the network can see the medial surface (open question 3).
- On-wall nodes (`|x - p| <= eps`) keep `u` = the pseudo-normal of the found feature (open question 1).

## c. Properties (argued, then tested; each test fails on the base and passes on the branch)

1. **Independent of vertex order and search order.** Each cell's vertices are put in the lexicographic order of
   their positions (`_canonical_cells`), and the permutation's parity carries the orientation, so the normal is
   `parity x` the normal of the sorted cell. Every closest point is therefore computed by one floating-point path
   per cell, set by positions alone. A triangle's edge-region point is recomputed on the edge (`_segment_t` on its
   two sorted endpoints), so both triangles that share an edge give it bitwise; a vertex region gives the vertex
   exactly (weights exactly 1 and 0). The tie set is built from a minimum, interval tests and valence counts, all
   set functions. The distinct directions are de-duplicated and sorted (`torch.unique`) before the QP, so the QP's
   rounding path is fixed too. The sign votes are integer sums.
   Test: `test_tie_direction_is_the_least_norm_gradient_and_independent_of_the_order` covers square diagonals and
   the centre, 4 vertex permutations and 3 cyclic shifts of the cell list. It asserts bitwise equality of `d` and
   `u`, and that `nu` is exactly the midpoint, or 0 at the centre.
   Base probe: under cell rolls `nu` jumps by up to 2 (the centre picks opposite sides).
2. **Exact equivariance under a winding flip.** A flip reverses each cell's vertex order. The canonical order is
   unchanged, so every `p_k`, `D_k`, `E_k` and valence is bitwise unchanged: the same tie set. Only the parity
   flips. Then `n_k -> -n_k` exactly, because IEEE negation is exact. The pseudo-normals are sums of exactly negated
   terms in the same order, so they are exactly negated. `plane_k -> -plane_k` exactly, so face-region
   `c_k = sign(plane_k) n_k` is unchanged, and edge or vertex `c_k = (x - p_k)/D_k` is unchanged. So `m` is bitwise
   unchanged, the votes negate, and `d -> -d`, `nu -> -nu` bitwise.
   Only exception: a vote sum of exactly 0, where the candidates disagree on the side. That happens only within
   `E` of the surface; both orientations then take +1.
   Test: `test_tie_direction_flips_exactly_with_the_winding[2, 3]`, a box in a generic pose, so its ties hold to
   rounding only, with bitwise `d_F = -d` and `nu_F = -nu` at every tie point.
   Base probe: `|nu_F + nu|` reaches 1.41 at 2 of 37 points in 2-D and 23 of 46 in 3-D.
3. **Covariance under rigid motions.** In exact geometry, `T` is defined by distances, which isometries preserve;
   `c_k -> R c_k`; and the minimum-norm point commutes with linear isometries, because `R conv S = conv RS` and
   `|Rg| = |g|`. Numerically, an exact tie is within `E` in every pose, because each pose's `E` bounds that pose's
   rounding, so the tie set is the same and `nu` moves with `R` to rounding.
   Not covered: near-ties whose exact gap lies between the two poses' bands. `E` depends on `|x|`, so a translation
   moves the band edge. That is the hard-tau discontinuity of item 4.
   Test: `test_tie_direction_is_covariant_under_rigid_motions[2, 3]`, 3 random rotations, each proper and improper,
   with random translations, `atol 1e-12`. It also asserts the minimum-norm values: the midpoint, `(1, 1, 1)/3` at
   the cube's corner diagonals, and 0 at the centre.
   Base probe: `|nu - R nu_0|` reaches 1. Branch: 4e-16.
4. **Continuity.** Before, `nu` jumps across the medial set M by `|u_1 - u_2|`, up to 2 at a mid-plane, at a
   location set by rounding. With the rule, `nu` is discontinuous on the two edges of the tie band
   `{|D_1 - D_2| <= E_1 + E_2}`, which has width about `2(E_1 + E_2)/|u_1 - u_2|`. Each jump is about
   `|u_1 - u_2|/2`, so jump times width is about `E_1 + E_2`: the discontinuity has the size of the data's
   resolution, and its location is reproducible.
   **No continuous alternative exists without a new scale.** `grad d` has different one-sided limits on M, so any
   continuous field must differ from `grad d` on a neighbourhood of M, and that neighbourhood's width is a length.
   The problem supplies only `E` without a choice. A ramp over `E` is formally continuous but has Lipschitz constant
   `1/E`, so it is numerically the same as the jump. A ramp relative to the local distance (`gap/d < kappa`) needs a
   dimensionless `kappa`, which is a tuned constant. The soft-min gradient (option 3 of the 2026-10-18 note) needs a
   length `tau`. So a continuous rule needs a new constant; the rule above is the principled discontinuous one.

## d. The search change

The search must return every cell within `min_j(D_j + E_j) + E_k`, not just the first minimum.

**Prototype (committed; no new kernels).** `_closest_candidates` is a `torch.library.custom_op`
(`physicsnemo::isla_closest_candidates`, with a fake implementation), so the whole-model `fullgraph=True,
dynamic=True` compile still traces (the interior compile test passes). It runs without gradient and is chunked over
the queries, at `2^20` (query, cell) pairs per chunk.

- 2-D: an exact brute force over all segments, in float64.
- 3-D: the existing float32 BVH gives a seed cell; its exact float64 distance plus `E` is an upper bound; cells whose
  axis-aligned bounding box lies within that bound plus `E` are searched exactly in float64. That broad phase is
  dense, `Q x F` in float64.
- Then the valence test, de-duplication, and the per-query QP, a Python loop over the tie queries only.
- The differentiable rebuild of `d` keeps one cell (the first minimum) and uses canonical vertices.

Measured on CPU (4 threads, float32 inputs, nice 19), against base:

| case | base | branch |
|---|---|---|
| 2-D, NACA 798 segments, 20k queries | 0.53 s | 4.9 s |
| 3-D, sphere 20,480 triangles, 20k queries | 0.15 s | 13.6 s |

The prototype is correct but too slow for training. The geometry runs in every forward pass.

**Production (Warp; proposed, not written).**

1. Pass 1 is the existing `mesh_query_point`, giving `d_1` (float32) and its cell.
2. Pass 2 enumerates every triangle within `r = d_1 + delta`, where `delta` is the float32 BVH bound plus `E`, with
   a distance-pruned BVH traversal: a node is pruned when its box is farther than `r`. For each candidate, compute
   the exact float64 closest point on the canonical triangle and keep those within the bound.
3. Use a count, an exclusive scan and a fill, so the output is CSR and needs no fixed capacity constant.
4. In torch: the valence test (all cells sharing a vertex or edge reach the same point, as in the prototype), then
   the closed forms vectorized for at most 3 distinct directions (pair midpoint; triangle foot with a barycentric
   check), and Wolfe's algorithm on the rare remainder.

Cost:

- Time is `O(Q (log F + k))`, with `k` the triangles within `r`. That is the cap of the ball near the foot, about
  `sqrt(2 d delta)/h` across, so `k` is a few. Ties add a handful.
- An AABB query with the existing `mesh_query_aabb` over `[x +- r]` also works without a new traversal, but costs
  `(2d/h)^2` for a node at distance `d` over a flat wall, because the box face lies in the wall. Prefer the ball
  traversal.
- Memory is the CSR, `Q x k` entries of 12 bytes (index plus float64 distance): about 50 MB at `Q = 10^6`, `k = 4`.
- The extra pass roughly doubles the BVH query time.

## e. The octant fix (committed)

**Rule.** Bit `k` of a normal's octant is set when `n_k > 0`. A zero component takes the sign of the next nonzero
component, cyclically: `k+1, k+2, ...` mod `d`. Unit normals always have a nonzero component.

**Antipodal (proof).** Under `n -> -n` the zero pattern is unchanged. A nonzero component flips its sign, so its bit
flips. A zero component consults the same `j`, whose sign flips, so its bit flips too. Every bit flips:
`oct(-n) = 2^d - 1 - oct(n)`, the antipode.

**Coherence is kept.** Two normals in one octant have, per component, either the same strict sign or a zero, so
`n . m >= 0`. Opposite normals never share an octant, as before.

**Forest flip symmetry (proof).** The canonical frame is built from positions and measures only, so a winding flip
leaves it unchanged. Then `normals_f -> -normals_f` exactly: the same matmul with negated inputs. The cells of
octant `k` become the cells of octant `2^d - 1 - k`, so the partition into trees is the same set of blocks, and each
tree is built from the same points in the same input order, so it is the same tree. Only the trees' order in the
forest table changes (antipodal octants are visited in reverse), which changes the read at summation rounding only.

**Effect.** This only matters for normals with exact zero components in the canonical frame, which needs the frame
to align exactly with face normals.

- Exactly symmetric boundaries can do that. The 2 x 1 rectangle of the test has frame `[[0, 1], [1, 0]]`. Under the
  base rule it had three trees whose blocks change under the flip; now it has two that map onto each other.
- A subdivided `cube_surface` gives a frame about 0.01 rad off the axes, so it has no exact zeros, and both rules
  give identical forests.
- Before, `+e_k` landed alone in the octant `2^k`, and every `-e_k` landed in octant 0 together with the (-,-,-)
  normals: an asymmetric partition. Now every `+e_k` lands in octant `2^d - 1` and every `-e_k` in 0.

Tests: `test_normal_octants_are_antipodal_at_zero_components[2, 3]` (every normal with components in {-1, 0, 1})
and `test_boundary_forest_partition_is_the_same_under_a_winding_flip`.

## f. Proposed pre-registered experiment (for the supervisor; not run)

**Pre-steps, no training.**

- (i) A cpu-short LHR job: the geometry-only flip residual on the 60 lug test meshes, base against branch, with
  ATTRIB-090's `geo_flip_probe.py` approach.
- (ii) B's checkpoints, scored eager with the branch geometry: the prediction-level compensated flip residual
  (`flip_probe.py`). Its median should fall from 1.5e-4 to the float32 floor, because the geometry is then exactly
  equivariant for any weights.
- (iii) A one-epoch timing smoke of the branch on the lug, under 1 GPU-h. **Gate:** if the search adds more than
  20 % to the step time, cache the geometry per case first, or build section d's kernel. The geometry is
  deterministic and the meshes are not augmented.

**Arms.**

- B: the base, eager-trained. Reuse REF-090-CLOSE's eager-trained `r090eag_lug_isla` and the Tensile2d
  counterpart if they sit on the same base and recipe.
- TIE: this branch, eager-trained.
- Lug and Tensile2d, seeds 42 and 43, REF-090's recipe, epoch-500 checkpoints, eager scoring, 600 test cases on the
  lug.

**Cost.** TIE costs about 63 + 10 = **73 GPU-h** (REF-090's per-lane figures, two seeds each). A fresh B would add
73 more, for **146 GPU-h**. Add about 1 GPU-h for pre-step (iii), plus whatever the search overhead adds unless the
gate is met first.

**Metrics.** Native case-mean error; posed / native; the flip residual (pre-step (ii), and the same on the TIE
checkpoints).

**Reading rule (pre-register the sigma before scoring).** Two seeds per arm, so the standard error of the log ratio
is `sigma_log`. The lug's pooled `sigma_log` is 0.036 (ATTRIB-090 section 4). Tensile2d's comes from the REF-090
records.

- **F (symmetry):** the TIE checkpoints' compensated flip residual has median at most 1e-6, with both seeds, on both
  datasets. Then "exact flip symmetry restored". Otherwise "a residual source remains", to be located before
  anything else is read.
- **N (native):** `r = TIE / B` (seed means).
  - `|log r| <= 2 sigma` (lug: r in [0.93, 1.075]) means "no cost".
  - `log r > 2 sigma` means "costs native accuracy".
  - `log r < -2 sigma` means "gain".
- **P (pose; lug primary):** the excess `e = posed/native - 1`. B's is about 0.37 (0.00198 / 0.00145).
  - `e_TIE <= 0.5 e_B`: "the medial ties carry most of the lug's pose sensitivity".
  - `e_TIE >= 0.8 e_B`: "they do not; look elsewhere".
  - Between: "partial".
  - Tensile2d is the control: predicted |change in e| within 2 sigma.
- **Decision:** adopt (with section d's kernel) if F passes and N is not "cost" on either dataset. P decides whether
  the 2026-10-18 note's other options are still needed.

## g. Open questions

1. **On-wall nodes** (s = 0, 33 % of lug nodes) still take the pseudo-normal of the found feature. ATTRIB-090 saw
   their `nu` flip to within 6.7e-4 rad, probably the same first-found choice among features within `E` of a point
   on the surface. A rule there needs a decision. My recommendation is the pseudo-normal of the lowest-dimensional
   feature in the tie set (vertex, then edge, then face), which is again a set function.
2. **Vote sum 0** takes +1, which is not antisymmetric. It only happens within `E` of the surface.
3. **Non-unit `nu`** at ties is a new input signal (`|nu| < 1` on the medial surface). The alternative, normalizing
   `m` when it is nonzero, keeps unit vectors but is not the Clarke element and is still 0 at cancellation. I
   recommend the unnormalized `m`.
4. **`E` depends on `|x|`**, so it is not translation-invariant. It cannot be, since the representation error is
   not. A float64 recentring in the data path would shrink it and the near-wall d error with it (section a). This
   is worth checking on AeroZoo separately.
5. **Gradient:** at ties `u` is detached (no gradient), and `d`'s gradient uses one cell. The geometry is data, so
   training is unaffected.
6. **Speed:** the prototype is 9x (2-D) to 90x (3-D) slower on CPU (section d). This blocks the training experiment
   until the cache or the kernel exists.

## h. Records

- Tests, CPU, serial, nice 19, `PYTHONPATH=td0152`:
  - `test_isla_interior_geometry.py` plus `test_isla_tree_read.py`: 75 passed, 7 skipped (CUDA).
  - `test_isla.py -k 'geometr or interior or equivariance or gradient or scale'`: 18 passed.
  - `test_isla_compile.py -k 'interior and not budget'`: 1 passed, 1 skipped.
- Base check: with the base `geometry.py` and `tree_read.py` checked out (the patch saved and re-applied), all 8 new
  tests fail. Branch: all 8 pass.
- No existing test changed behaviour. The only edit to an existing test is the renamed chunk constant
  (`_TORCH_SEARCH_CHUNK` became `_SEARCH_PAIRS`, now counted in pairs).
- Probes: `probe_base_vs_branch.py` (the base numbers in section c) and `probe_timing.py` (section d), in this
  folder.
- The optional LHR measurement (task item 4) was not run, to stay in the time budget. It is pre-step (i) above.

## i. Proposed, not implemented: the fold sign and the token draw (GEO-ROBUST phase 1, 2026-10-22)

Both defects come from f32floor (scratch/isla-az-decide-2026-10-21/f32floor/README.md). Neither proposal is
implemented; each needs the supervisor's go-ahead and a pre-registered rerun of f32floor's A32-A64 arms.

### i.1 The fold sign: a generalized winding number

**The defect.** `d` is signed by the pseudo-normal of the closest feature. At a fold (a zero-thickness fin,
coincident cells of opposite orientation, two roles welded back to back) the closest edge's or vertex's normals
cancel: the angle-weighted sum is exactly 0 in float64, so the sign is undefined in exact arithmetic and roundoff
picks it. f32floor found 12-427 such points per AeroZoo case, 0.002-0.32 L from the wall. Float64 only makes the
arbitrary choice reproducible. Separately, 51-118 fluid points per case carry a negative `d` in both precisions,
which points at open or non-manifold walls, where a local pseudo-normal does not decide inside from outside anyway.

**Proposal.** Sign `d` by the generalized winding number (Jacobson, Kavan and Sorkine-Hornung 2013), a global
quantity that does not divide by a local sum:

    w(x) = (1 / 4 pi) sum_f Omega_f(x)     (2-D: (1 / 2 pi) sum_s theta_s(x)),
    sign(d) = sign(sigma / 2 - w(x)),       sigma = sign(V),  V = (1/3) sum_f a_f n_f . (c_f - g),

with `Omega_f` the signed solid angle of cell `f` seen from `x` (Van Oosterom and Strackee 1983; in 2-D the
signed angle a segment subtends), `a_f, n_f, c_f` its measure, oriented unit normal and centroid, and `g` the
gauge centroid (so `V` is the enclosed signed volume of a closed surface and frame-free for an open one).

- **Why it is the right sign.** For a closed oriented surface `w` is 0 on the side the normals point to and
  `sigma` (+1 with outward normals) on the other, so `sign(sigma/2 - w)` is "positive on the side the normals
  point to", the current convention, and equals the pseudo-normal sign everywhere off the surface in exact
  arithmetic. Existing behaviour on watertight meshes is unchanged.
- **Folds.** Two coincident sheets of opposite orientation contribute `+Omega` and `-Omega`: the fold drops out of
  `w`, and both sides of a zero-thickness fin take the sign of the region around it (fluid: positive). That is the
  physical answer, and it no longer depends on which side roundoff picks.
- **Open and non-manifold walls.** `w` is harmonic off the surface and jumps by exactly 1 across it; with holes it
  is fractional and `1/2` is the standard inside threshold (Jacobson 2013). The sign is ambiguous only where
  `|w - sigma/2|` is below the evaluation error, a thin set spanning the holes, instead of on every fold.
- **Robust to precision.** Away from the surface `w` is a sum of smooth terms; rounding moves it by about
  `u sum_f |Omega_f| / 4 pi`, which is far below `1/2` (float32: about 1e-6 for well-shaped meshes). A sign flip
  needs `w` within that of `sigma/2`.
- **Exactly antisymmetric under a winding flip.** Every `Omega_f`, `n_f` and so `V` and `w` negate exactly (IEEE
  negation, same summation order), `sigma -> -sigma`, so `sign(sigma/2 - w)` negates bitwise. This also removes
  open question g.2 (the vote sum 0 at ties takes +1): ties take the same `w` sign as everything else.
- **Translation.** `w` uses only differences `x - v`; `V` uses `c_f - g`. Invariant.

**Fast evaluation (Barill, Dickson, Schmidt, Levin and Jacobson 2018, "Fast winding numbers for soups and clouds").**
The BVH that the tie search already builds (`_ball_candidates`) carries, per node, the dipole moment
`m = sum a_f n_f`, its area-weighted centre `p` and radius `r`, from one bottom-up pass over the levels. A query
uses the far-field term `m . (p - x) / (4 pi |p - x|^3)` when `|p - x| > beta r` (Barill's `beta = 2`, a
second-order expansion if more accuracy is needed) and descends otherwise, with the exact `Omega_f` in float64 at
the leaves: the same stackless Warp traversal as `_ball_candidates_kernel`, with a different prune test and an
accumulator instead of a list. Error: Barill report about 1e-3 at `beta = 2` with first-order terms; the sign only
needs `|w - sigma/2|` above it.

**Cost at AeroZoo size (F = 1M triangles, Q = 200k).** About `O(log F + near field)` node visits per query,
several hundred at `beta = 2`; at about 40 flops per visit that is 2e4 flops per query, 4e9 per call. On a GB300
that is milliseconds in float32 (about 5-20 ms, bound by gathers), roughly 4-10 ms more in float64 for the near
field. The moments pass is `O(F)`, about 5 ms. Memory: 7 floats per node, 2M nodes, 56 MB (float32). The same
traversal on CPU: about 1-3 s per call at this size (Barill's CPU numbers), acceptable for tests only.

**How it fits `exact_boundary_geometry`.** `_closest_candidates` returns a per-query sign from `w` next to the tie
outputs (it already has the BVH, the float64 simplices and the query norms). `exact_boundary_geometry` uses it for
every off-wall query: `d = sign_w * dist` for untied and tied queries alike; on-wall nodes keep `d = 0`. The
face-interior shortcut `d = plane` stays (its sign is the cell's own normal, which agrees with `w` off folds); at a
fold cell's interior the plane sign is also local, so the cleanest rule is `d = sign_w * |plane|` there too. The
vertex and edge pseudo-normals are then no longer needed for the sign and can be dropped (`_vertex_pseudo_normals`,
`_edge_pseudo_normals`, `_feature_normals` for the sign), which pays for part of the cost. `u` is unchanged.
Tests to add: the fin (two coincident opposite sheets) gives positive `d` on both sides in float32 and float64; a
winding flip negates `d` bitwise everywhere; a cube with one face removed signs the far field consistently; and
f32floor's sign-flip count (12-427) goes to 0 in the rerun.

### i.2 The token draw: a combinatorial order, cached per mesh

**The defect.** `sample_boundary_cells` orders each condition's cells by the Morton code of their centroids on a
21-bit grid over the bounding box, in the dataset's axes, and draws a systematic PPS sample along that line (start
0.5 at evaluation). Two things in it move with roundoff and pose: a centroid near a grid line changes cell when its
coordinates are rounded differently (5-5,794 of 200,000 drawn cells change between float32 and float64), and the
grid follows the dataset's axes, so a rotation reorders the line and redraws the sample. The draw is the larger
noise source on AeroZoo (S32: 2-15 % per case).

**Proposal.** Keep the systematic PPS draw (its stratification along a local order is what keeps the draw noise
low) and replace the Morton line by an order built from the mesh's connectivity only:

- `pi` = a breadth-first (or reverse Cuthill-McKee) order of the welded cell-adjacency graph of each condition, from
  its lowest-index cell, ties within a level broken by cell index, components concatenated by lowest index. It is a
  function of the connectivity and the numbering, not of coordinates: exactly invariant to rotation, reflection,
  translation and scale, and to the precision of the coordinates. Consecutive cells along `pi` are geodesic
  neighbours (BFS levels are rings), so systematic sampling along it stratifies the sample over the surface, as
  the Morton line did.
- The PPS line is the float64 cumulative sum of the inclusion probabilities along `pi` (as now). The only remaining
  coordinate dependence is the measures: rounding of posed float32 coordinates moves each measure by about `u`
  relative, independently, so the cumulative sum moves by about `u A / sqrt(F)` and a hit changes cell with
  probability about `u n / sqrt(F)` per token: 0.01 % at `n = 2e5`, `F = 1e6`, so about 2 of 200,000 tokens per
  draw under a rotation, and none between float32 and float64 compute on the same inputs (the line is computed in
  float64 from the same float32 data). Quantizing the probabilities to a fixed grid (for example `2^-40` of the
  share) before the sum would make even that a measure-zero event.
- The draw still depends on the mesh's numbering, like any draw from a stored mesh. It does not depend on how the
  case is posed, which is the invariance the model claims.

**Cost.** The order depends on the mesh only, and poses move coordinates, not connectivity, so it is computed once
per mesh and cached with the case (or in the datapipe's per-case cache): a BFS over the cell graph, `O(F)` work,
about 0.1 s per 1M cells on CPU (scipy `breadth_first_order` on the CSR adjacency, which the weld already gives),
4 MB of int32 per 1M cells. Per forward pass it replaces the Morton encode and the argsort (`O(F log F)`) by a
gather, so the step gets slightly cheaper. In-model without the cache, a GPU level-synchronous BFS needs about one
launch per level (the graph diameter, about `sqrt(F)`, 1-3k levels at 1M cells), 0.1-0.5 s per mesh: too slow per
step, so the cache is part of the proposal.

**Alternatives considered.** (1) Morton in float64 (f32floor's cheap fix): removes the float32-against-float64
flips, but not the rotation dependence or the flips from rounding of posed coordinates. (2) Morton in the
boundary's canonical frame (`tree_read.canonical_frame`): rotation-covariant, but the frame and the grid lines both
move with rounding, so posed draws still change near grid lines, about as often as now. (3) Priority sampling with
hashed cell indices (key `U_c^(1/pi_c)`, `U_c` = hash(cell index, seed); Efraimidis-Spirakis): exactly invariant,
no per-mesh cost, but it is a Poisson-type sample with no stratification, so it would raise the draw-to-draw noise
that S32 already measures at percent level; measure before choosing it.

**Tests to add.** The kept cells and weights are bitwise equal under a rotation and reflection of the case in
float64, equal between float32 and float64 compute on the same float32 inputs, and differ under rotation in float32
by at most a handful of cells; the draw's stratification (the spread of kept-cell counts over equal-measure patches)
is no worse than the Morton draw's on the sphere and the lug.
