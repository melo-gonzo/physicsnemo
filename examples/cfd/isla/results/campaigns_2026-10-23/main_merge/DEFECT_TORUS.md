# DEFECT: the on-wall direction at a quad's diagonal (found by MAIN-MERGE, 2026-10-24)

## How it was found
Upstream #2148 (63bd06e44, "Mesh: fix reversed cell orientation in primitive generators") reversed the winding of
`physicsnemo.mesh.primitives.surfaces.torus.load`. Before, its cells were `[idx, next_j, next_i]` and
`[next_j, next_both, next_i]`. After, they are `[idx, next_i, next_j]` and `[next_j, next_i, next_both]`, so the
normals point outward. Points and triangles are the same, and only the vertex order changed.

Two ISLA tests take "edge midpoints" as `pts[cells[:, :2]].mean(1)`. Under the new winding, the second triangle block's
first two vertices `(next_j, next_i)` span each quad's **diagonal**. So the tests now query diagonal midpoints, which
they never queried before. Both then failed at the merge head, on x86 CPU (LHR cpu-short 376127) and on GB300
(376126):
- `test_on_wall_direction_is_a_function_of_the_input_mesh[split]`: u from float32 inputs != u from float64;
- `test_float32_points_within_the_bound_are_on_the_wall[torus]`: u changes under vertex renumbering and cell
  reordering (d is unchanged).

On the supervisor's ruling (22:46 EDT), both tests are pinned to the base winding with `_flipped`, and
`test_on_wall_direction_at_quad_diagonals` (strict xfail, two cases) records the defect. Commit 15e9063aa is on
psharpe/isla-main-merge-1024.

## What differs, where, and by how much
The defect is pre-existing: the same counts come from 5b70c8686's ISLA code with 5b70c8686's torus, and from the merged
code in either winding (probes/torus_probe2.py, probes/torus_probe3.py). Only the diagonal midpoints differ. The quad
sides, vertices and centroids are bitwise stable.

| case | mesh | queries | differ | max abs diff in u |
|---|---|---|---|---|
| renumbering (the test's permutation of vertices, cell shuffle, roll(1, -1) of each cell) | torus 48 x 24, generic pose (`orthogonal(3, 7)`), float32 | 1152 diagonal midpoints, all on the wall | 51 (52 under the original test's RNG stream) | 1.79e-6 |
| precision (float32 inputs against float64 copies) | split torus 28 x 12 (vertices moved by 1e-5 of their radius), posed | 336 diagonal midpoints, 35 on the wall in both | 5 | 1.61e-6 |

d is bitwise equal in every case. u is a unit vector, so a difference of 1.8e-6 is about 15 float32 ulps. The effect is
small but real: it breaks the claim that the on-wall u is a function of the input mesh alone.

## Untested guess at the mechanism
A torus quad is a planar isosceles trapezoid, so at its diagonal the two triangles' normals are equal in exact
arithmetic. Computed from float32 vertices, they differ by rounding (about 1e-6). In `_closest_candidates_impl`
(geometry.py, around lines 2005-2060), the on-wall directions are the cells' float64 normals, `wall_n[pk[tie]]`.
`_certified_tied` then keeps a tie only for certified-distinct directions (`|c_i - c_j| > tau_i + tau_j`).

The guess: two normals that are near-equal but not certified distinct collapse to one representative, or the tie is
dropped. Which normal survives then depends on the cells' order or the candidate order, not on the positions alone. That
would give u = n1 in one numbering and u = (n1 + n2) / 2 or n2 in another, a step of about |n1 - n2| / 2, which is
about 1e-6, as observed.

A second suspect is `face_normals`, if the caller computes it from the cells in the given vertex order rather than the
canonical (`_canonical_cells`) one. But float64 rounding alone would move u by about 1 float32 ulp, not 15.

Check: on a failing query, print `tq`, `tc`, `s` and `tied` for both numberings.

## Probes (each runs in its own process; PYTHONPATH=<tree>:td0152, CUDA hidden)
- `probes/torus_probe.py <worktree>`: the two original tests' queries, with the winding as loaded and flipped.
- `probes/torus_probe2.py <worktree>`: every edge kind (sides and diagonal) under both windings. Run it at the merge
  tree and at the 5b70c8686 tree.
- `probes/torus_probe3.py <worktree>`: the xfail test's two cases, counted.
