# GEO-DEGEN design: zero-area cells in ISLA's exact boundary geometry (2026-10-24 02:06 EDT)

Base: `psharpe/isla-geo-fast` at `0570fc6c8` (worktree `/home/psharpe/gh/physicsnemo-isla-geo-fast`). The supervisor's
decision (the supervisor's, not Peter's), quoted: fix it, do not refuse such meshes. A zero-area cell is the set its
vertices span; its closest point is the closest point on that set; it never supplies a normal or an on-wall direction; its
winding-number contribution is zero.

## The three cases

The search reads every cell with its vertices in canonical order (`_canonical_cells`: the lexicographic order of the
positions, after the weld), so coincident vertices are adjacent and carry one index.

| case | canonical vertices | the set |
|---|---|---|
| two coincident vertices | `(P, P, Q)` or `(Q, P, P)` | the segment PQ |
| three coincident vertices | `(P, P, P)` | the point P |
| three distinct collinear vertices | `(A, B, C)`, ordered along the line | the segment AC |

On a line, the lexicographic order of points is their order along it (the first coordinate in which the direction is
nonzero decides it). So in all three cases the set is the union of the cell's two edges between consecutive canonical
vertices, `[v0, v1]` and `[v1, v2]` (one of them a point for two coincident vertices, both for three). The closest point is
the nearer of theirs, the first at equal distances.

This differs from GEO-FAST's proposal (the closest point on the longest edge, `[v0, v2]`) on purpose. The first version of
this fix (02:20 EDT) took `[v0, v2]`. It was correct in d and in the closest cell, but at a T-junction cell `(A, M, B)` lying
over two mesh edges AM and MB it changed `u` against the same mesh without the cell: 91 of 2,964 float32 queries, by up to
0.54, on the box offset by 1024. The segment AB was a feature that no mesh cell shares, so it joined ties that the
local-minimum test (every cell of a feature reaches the point) should reject. The two-edge rule gives features `A`, `M`,
`B`, `AM`, `MB`, the mesh's own. Each point is computed by the same `_segment_t` call, on the same ordered pair, as a
triangle sharing that edge, so it is bitwise the same point. With it, adding zero-area cells to a closed mesh leaves u
unchanged: 0 of 102,800 queries differ.

In two dimensions, a zero-length segment (two welded vertices, one index) is the point. The 2-D search already handles
it (`_segment_t` gives t = 0), and the fix adds only the rules on normals, solid angles and stars below.

## The flag: an exact test (`_zero_area`)

The flag is computed on the coordinates the search reads: the points cast to the queries' precision, held exactly in
float64. A triangle has zero area exactly when its cross product `(b - a) x (c - a)` is zero, so when its three
components, the 2-D orientation determinants of its projections, are all zero.

1. Filter: each component is evaluated in float64 from the rounded differences, with Shewchuk's (1997) a-priori bound
   `|err| <= (3 + 16e) e (|l| + |r|)`, where `e = 2^-53` and `l`, `r` are the two products. A component whose value exceeds
   the bound is proven nonzero. This is a derived bound on IEEE rounding, not a tolerance, and it clears every cell of
   nonzero area except the very thinnest (sin of the angle below about 1e-15).
2. Exact: the remaining cells are evaluated exactly. Each component of `a x b + b x c + c x a` (equal to `(b - a) x
   (c - a)`) is six products of input coordinates. Each product is split exactly into two doubles (Dekker's TwoProduct,
   by Veltkamp splitting, no fused multiply-add), and the twelve terms are summed exactly into a nonoverlapping
   expansion (Shewchuk's Grow-Expansion of Knuth TwoSums). A nonoverlapping expansion is zero exactly when every term is.

Why it is exact: TwoSum and TwoProduct are error-free under round-to-nearest unless a result overflows or the product's
error term underflows.
- For float32 coordinates (every ISLA dataset), every value is a multiple of `2^-149` and at most `2^128` in magnitude. So
  in float64 the products are multiples of `2^-298` (no underflow) and below `2^257` (no overflow): exact always.
- For float64 coordinates, it is exact when every nonzero magnitude lies in `[2^-485, 2^510]`, about `[1e-146, 3e153]`.
- Torch evaluates each elementwise operation as its own kernel, so no multiply-add is fused (torch's own `linalg.cross`
  does fuse, which is why the cross product of two equal edge vectors comes out near 6e-18 rather than 0).
- The flag is computed inside the custom op, which torch.compile does not trace, so inductor cannot fuse it either.

The filter costs about 30 elementwise operations over the cells and one host synchronisation. The exact path runs only
on the cells the filter leaves.

## Changes (all in `geometry.py`; every one is guarded by `any_flat`, true only when some cell is flagged)

- **Ball search (`_ball_candidates(flat=)`)**: a flagged cell gets NaN plane rows and NaN corners, so the kernel's float32
  lower bound and its float64 Ericson distance (undefined on such a cell: NaN, or a vertex distance) both fail to reject
  it. It stays a candidate wherever the box test reaches it; its box contains its segment, so the candidate set still
  holds every cell within the radius. The kernels are unchanged.
- **Seed**: the radius's bound uses `_closest_weights` with the flag, so it is a true distance to the cell's set.
- **`_closest_weights(q, simplex, flat)`**: on flagged rows the weights are `(1 - t1, t1, 0)` on `[v0, v1]` or `(0, 1 - t2, t2)`
  on `[v1, v2]`, whichever is nearer (`_segment_distance2`, the first at equal distances). A zero-length edge gives t = 0.
- **Exact phase and tie test**:
  - A flagged cell is never on_cell, because its middle weight is 0. Off the wall its direction is `(x - p) / |x - p|`.
  - Stars: a flagged cell's features are its distinct vertices (each counted once) and its edges between consecutive
    distinct vertices; its edge `v0 v2` is not among them. So a zero-area cell lying along mesh edges or at a vertex
    takes part in the "every cell of the feature reaches it" test like any other cell of that feature.
  - `has_n` is false for flagged cells. This is TIE-FIX's rule that the on-wall direction skips zero-area cells. It was
    broken for the order `(Q, P, P)`, where torch's fused cross product of two equal edge vectors normalizes to a random
    unit vector, which passed the old finite-normal test.
- **On-wall query whose closest cell is flagged**: it is marked tied, so its direction is the odd least-norm point of the
  normals of the cells within the bound that have one. That is the one normal when one cell holds the query, and the
  edge or vertex rule when several do. When only zero-area cells hold the query, no normal exists and `u = 0`. That is the
  least-norm element of the generalized gradient of the distance to a segment or a point at a point of it (a disk or a
  ball about 0), the same as the existing rule "zero when the directions cancel". `d = 0` as on any on-wall query. Tested.
- **Winding number**: flagged cells get parity 0 in the winding sum and the oriented areas, so their solid angle, area
  and moments are exactly 0, and the volume sign sigma excludes them.
- **Fast path**: the same functions on the same values. The flag is computed once per cell and permuted with the cells
  into the tree order (`flat[order]`, then `[leaf_order]` for the preorder tree), so the arithmetic is the reference's.
  No new custom op and no new kernel.

## Unchanged without zero-area cells (the claim, checked bitwise in RESULTS check a)

When no cell is flagged, `any_flat` is False and every changed line takes its old branch: the same operations on the same
values as `0570fc6c8`. The only additions are the filter's reads and one synchronisation, which write nothing the search
reads. So every output (the six op outputs, w, d, u and their gradients) is bitwise unchanged.

## Out of scope, reported (found 02:00 EDT; supervisor told by progress update)

Thin cells of nonzero area break the reference too, through the same two causes.
- On float64 soups whose slivers are a midpoint moved off the line by 1e-13 relative, 8 of 10 soups raise IndexError
  (10 of 10 at 1e-16).
- With the plane bound off, none raise, but 3 of 14,000 queries get a farther point, because Ericson's float64 weights
  misplace the closest point on thin cells.
- Float32 slivers (offset 1e-7 relative or more) and float64 at 1e-10: 0 wrong of 14,000.

Fixing this changes outputs on inputs without a zero-area cell, which this brief excludes. Proposed follow-up: certify the
plane bound per cell (skip it where the normal's error bound exceeds the pad), and take a thin cell's closest point as
the nearest of its face projection and its three edge points. Probes: `probes/probe_thin.py`, `probes/diag_thin.py`.

## Check bound for d (RESULTS check b)

`|d|` is rebuilt in the queries' precision from float64 weights: the weights are rounded, then three products and two
sums, the difference and the norm. So its error against the exact distance is a few units `u` (unit roundoff of the
queries' precision) of `|x| + R`, where R is the greatest vertex norm. The check requires `||d| - d_exact| <= 8 u (|x| + R)`
and reports the largest ratio. Wrong closest cell: the exact distance of the op's cell exceeds the exact least by more
than the op's own tie bound, `E_idx + E_min`, with `E = (u + 13 u_64)(|x| + max |v|)`.

## Amendment (02:45 EDT, commit `1086cf42e`): a zero-area cell that lies in the other cells is left out

Found by the GB300 tests of `a9592e9a3` (job 364908) and the cpu-short suite (364909): TIE-FIX's
`test_on_wall_direction_skips_zero_area_cells` failed. The cell `(a, b, mid)` lies on the sphere's edge `ab`. As a segment of
its own, its closest points are computed on `[a, mid]` and `[mid, b]`, which reach the points of edge `ab` only to rounding,
not bitwise. So at the off-wall query `1.01 mid` it made a spurious tie, and `u` changed in its last bits (1e-16).

Rule: before the search, `_covered` drops every zero-area cell whose set lies in the cells of nonzero measure, by an exact
test on the welded indices, with no arithmetic: a point that is one of their vertices; a segment whose edges between
consecutive distinct vertices are their edges, or whose edge from first to last vertex is (on a line the canonical order
is the order along it, so that edge holds the middle vertex). The set adds nothing to the surface, so the outputs are
those of the surface without the cell, bitwise (the returned cell index is mapped back to the input's numbering). In two
dimensions a zero-length segment at a vertex of another segment is dropped the same way.

What stays: zero-area cells that the index test cannot place in the surface (dangling segments and points, and segments
lying inside a face but off its edges) are kept as their segments or points by the rules above. The second kind is a
coplanar overlap. Its `u` can differ from the surface without it at ties, exactly as an overlapping coplanar triangle of
nonzero area does at `0570fc6c8` (RESULTS, check b, control row). That is the existing tie rule on overlapping cells, not a
zero-area rule, and is out of scope.

The unchanged-input claim still holds: `_covered` runs only when `any_flat`, and `kept` is `None` otherwise.

## Answer to the brief's questions, in one place

- Flag: exact (Shewchuk filter, then error-free products and an expansion sum), no tolerance constant; exact for all float32
  coordinates and for float64 coordinates with magnitudes in `[2^-485, 2^510]`.
- On-wall query held only by zero-area cells: `d = 0`, `u = 0` (the least-norm element of the distance's generalized
  gradient at a point of a segment or a point). Tested (`test_query_on_zero_area_cells_alone`, and 3,024 such queries in
  check b, 0 with `u != 0` or `d != 0`).
- Rule chosen against GEO-FAST's proposal: the two edges between consecutive canonical vertices, not the longest edge, for
  the reason in "The three cases"; the same set, and the same closest distance up to rounding.
