# ISLA impact of the upstream/main merge a682959bc (main c8f684288..289e57382, 22 commits)

**Merge verdict: BLOCK.** Two ISLA tests that pass on the ISLA tip 5b70c8686 fail on the merged tree. The cause is main's torus winding fix (#2148). ISLA's model, recipe and book compute the same numbers as before on real data. The only training-side numbers that can move belong to the baseline models' SDF inputs and to float64 meshes rotated by `Mesh.rotate`, and both changes are small.

I checked the call sites by grepping 5b70c8686 in `physicsnemo/experimental/models/isla`, the recipe (`src/`, `conf/`, `datasets/`, `tests/`, `tools/`), the book (`*.qmd`, `records.py`, `standing.py`, `palette.py`) and `test/experimental/models/isla`. I ran CPU probes against `git archive` copies of c8f684288, 289e57382, 5b70c8686 and a682959bc, one process per version, and deleted the archives afterwards.

## Findings

1. **P1: two ISLA tests fail on the merged tree.**
   - **Where:** `test/experimental/models/isla/test_isla_interior_geometry.py`, in `test_on_wall_direction_is_a_function_of_the_input_mesh[split]` (around line 1236) and `test_float32_points_within_the_bound_are_on_the_wall[torus]` (around line 1332).
   - **Cause:** main's #2148 reverses every cell of `primitives/surfaces/torus.py`, so its normals now point outward. Both tests use the raw `torus.load` winding.
   - **Evidence:**
     - On 5b70c8686 both tests pass (2 passed each).
     - On a682959bc, `-k "torus or split"` gives 2 failed, 9 passed.
     - A probe of the split case shows the cause. With the new winding, 5 of 409 on-wall points have a float32 `u` that differs bitwise from float64 `u` cast to float32, by up to 1.6e-6. With the cells flipped back to the old winding, 0 of 414 differ.
     - In the second test, renumbering the vertices leaves `d` bitwise equal but not `u`.
   - **What it means:** ISLA's code did not change, so this is a bitwise property of `exact_boundary_geometry` that held on the old torus and does not hold for the reversed winding. It was already latent on the ISLA branch and the new primitive exposes it.
   - **Smallest fix:** in those two tests, build the torus with `cells[:, [0, 2, 1]]`, or with the existing `_torus()` helper, which orients it explicitly. The other choice is to make the float32 geometry bitwise winding- and renumbering-invariant, which those docstrings claim.
2. **P2: six ISLA-only recipe configs still depend on a side effect.** Main (#2076) added `validate_args: false` to the six YAMLs it owns that pin z with a zero-width `Uniform`. These ISLA-only files have the same pattern without it: `drivaer_ml_surface_1case`, `drivaer_ml_volume_reference`, `_reference_area`, `shift_suv_surface`, `shift_suv_surface_area`, `shift_suv_surface_posed`. They still build only because an import-time `@torch.compile` in `nn.functional` turns distribution validation off for the whole process (pytorch#157926). This is not a regression; the merge only makes the inconsistency visible.
3. **P2: `from_zarr` is now stricter.** `from_zarr` and `io_zarr._open_group` now reject a store without the root `physicsnemo_mesh_type` attr, unless it sits inside a finished DomainMesh store. The datapipes readers use them for any `zarr.json` path. All ISLA datasets are `.pmsh`/`.pdmsh` memmaps, so nothing in use is affected. A zarr dataset written without `to_zarr` would now fail to load.

## Impact table

Classes: (a) not used by ISLA, recipe or book; (b) used, behaviour unchanged; (c) used, numbers may change; (d) used, API break.

| File (PR) | What main changed | ISLA / recipe / book use | Class |
|---|---|---|---|
| `mesh/primitives/surfaces/torus.py` (#2148) | Every triangle reversed, normals now outward | ISLA tests (raw `torus.load`) | **c (tests only)**: finding 1. Tests using the `_torus()`/`_oriented` helper get bitwise the same cells (probe) and pass. |
| `mesh/primitives/planar/structured_grid.py` (#2148) | Both triangles of each quad reversed to CCW, same diagonal | Recipe `_box_face` (AddBoxBoundaries, SHIFT-SUV) | b: `_box_face` re-winds into the box using the first cell, so the final cells are identical. |
| `mesh/primitives/surfaces/sphere_icosahedral.py` (#2129) | `radius` now applied when `subdivisions=0` | ISLA tests (subdivisions 1–3), book none | b: probe shows bitwise-equal points and cells at 1–3; only `subdivisions=0` changes. |
| `circle_2d`, `cube_surface`, `cube_surface.subdivide(linear)` | Docstring only, or untouched | ISLA tests | b: bitwise equal (probe). |
| Other primitives (cone, disk, hemisphere, plane, l_shape, text, volumes, lumpy_ball, …) | Orientation and degenerate-cell fixes | Not used; the recipe builds its own hemisphere | a |
| `mesh/mesh.py` (#2103, #2128) | Empty data containers and `_cache` built with `_new_unsafe`; `slice_cells` gives new `point_data`/`global_data` containers sharing the same tensors; `Mesh.merge` skips copying the first mesh; `cell↔point_data` conversion rewritten; `pad` zeroes `_effective_measure`; `pad_to_next_power` rounding fix | Constructor (ISLA, recipe, book), `slice_cells`/`slice_points`/`merge` (recipe) | b. Recipe call sites reassign the result, so the container split cannot be seen there. `pad` and the conversions are not called (a). |
| `mesh/utilities/_scatter_ops.py` (#2103) | Fast path for the unweighted mean; the comment says values match exactly | Indirect (normals, adjacency) | b |
| `nn/functional/normalization.py` → `mesh/utilities/_normalization.py` (#2147, R100) | Moved; `nn.functional.safe_normalize` removed with no shim | No ISLA, recipe, book or `examples/cfd/isla` import of either path in a682959bc | b (old path breaks only for outside users) |
| `nn/functional/weighted_multinomial.py` → `utils/weighted_multinomial.py` (#2147) | Moved; lazy shim warns with `LegacyFeatureWarning`; with replacement above 2^24 categories it now inverts a float64 CDF where `torch.multinomial` used to raise | Recipe, indirectly through datapipes `SubsampleMesh`; ISLA's own sampling does not use it; no old-path import anywhere in scope | b (≤2^24 identical; >2^24 errored before) |
| `nn/functional/__init__.py` | Shim above; `safe_normalize` dropped | ISLA imports `equivariant_ops` (unchanged) | b |
| `mesh/geometry/_cell_normals.py` | Import path only | ISLA `compute_cell_normals`, `cell_normals` | b |
| `mesh/spatial/bvh.py` (#2105) | Traversal: per-dimension AABB test, no −1 child filter, CPU queries above 32768 split into chunks | ISLA, recipe and book use only `_compute_morton_codes` and `_MORTON_SPREAD_STEPS` (unchanged); traversal is not called | b |
| `mesh/spatial/sdf.py` (#2125) | Degenerate-face repair offsets toward the original off-edge vertex (keeps plane and orientation) instead of an arbitrary perpendicular | Recipe `src/sdf.py` through `ComputeSDFFromBoundary` (drivaer_ml_volume*, aerozoo_uvolume*, highlift_volume) | **c, baselines only**: SDF value, sign and normals can change near faces with height below 1e-4 of the longest edge. ISLA configs do not read the SDF. |
| `nn/functional/geometry/sdf.py` | `.contiguous()` before Warp | ISLA imports `_closest_point_weights` (unchanged) | b |
| `mesh/boundaries/_facet_extraction.py` (#2165, #2128) | `operator.index` on cell width for dynamic compile; `_effective_measure` no longer carried onto facets | No facet or boundary extraction call in scope | a |
| `mesh/transformations/geometric.py` (#2128, #2119) | Each field transformed in its own dtype (mismatched dtypes used to raise); rotation trig in the mesh dtype; similarity test scale-normalized | Recipe `_transform_tensordict`, `transform`, `translate`, `scale`; datapipes `RandomRotateMesh` | b for float32 meshes and for FixedRandomPose (builds its own R). **c, tiny:** float64 meshes through `Mesh.rotate` (axis_aligned mode) now get float64 trig, about 1e-7 relative change in R. |
| `mesh/projections/_project.py`, `_embed.py`, `_extrude.py` (#2128) | `project` transfers explicit measures after replacing data; embed keeps measures; extrude guards | Recipe RestrictToPlane `project(..., keep_dims)` with the default flags (all False) | b (that code path is unchanged); embed and extrude a |
| `mesh/domain_mesh.py` (#2119) | `apply_to_meshes` keeps each result's device (`device=None`) | Recipe infer `apply_to_meshes(scale)` | b |
| `mesh/io/io_zarr.py` (#2130) | Refuses unfinished or attr-less stores; uint64 scalar read fix | `from_zarr` in merge_global_data and the readers, only for zarr paths | b with note: finding 3 |
| `mesh/io/io_pyvista.py` (#2130) | Cell-centroid mode uses vertex means for linear cells; `to_pyvista` vertex cells | Recipe `from_pyvista(pv.read(...))` in default mode | b |
| `mesh/calculus/*` LSQ, `utils/_small_linalg.py` lstsq, `nn/functional/derivatives/mesh_lsq_gradient` (#2141) | All LSQ systems solved at once and shared across fields | No gradient or LSQ calls; `small_det` unchanged | a |
| `mesh/subdivision/*` (#2123, #2103) | Tet orientation, int32, memory | Only through `cube_surface.subdivide(linear)` in ISLA tests (bitwise equal) | b |
| `nn/functional/neighbors/radius_search/*`, `knn/_cuml_impl.py` (#2156, #2125) | 1D–3D points, empty queries | Not used in scope (only by `examples/cfd/isla/research` scripts) | a |
| `mesh/sampling/sample_data.py`, `neighbors/_adjacency.py`, `_edge_lookup.py`, `tessellation`, `remeshing`, `generate`, `visualization` | Edge cases, docstrings | Not used | a |
| `domain_parallel/shard_tensor.py` (#2157) | Imports `torch._dynamo` before AOT `runtime_wrappers` | Recipe does not import it; ISLA's `nn.functional` import loads dynamo first anyway | b (protective) |
| `datapipes/transforms/mesh/transforms.py` | Main's side: `weighted_multinomial` imported from `physicsnemo.utils` (merged cleanly) | Recipe | b |
| `datapipes/transforms/subsample.py`, `cae/transolver_datapipe.py`, `DISTRIBUTIONS.md` | Same import move; documentation | Not imported by the ISLA path | a/b |
| Recipe `datasets/*.yaml` (6) + `README.md` (#2076) | Main's side: `validate_args: false` on the zero-width-z `Uniform` | Recipe | b; finding 2 |
| Recipe `src/train.py` (#1972) | Main's side: scaler saved and restored, `scheduler.step()` before the save, final epoch saved | Recipe | b: merged file equals ISLA's side (ISLA already had it in 49aa7126a) |
| `calculus/measure.py`, `fields`, `cluster_tree`, `_lbvh`, `_ragged`, `_gather`, `_cell_areas`, `equivariant_ops`, datapipes readers, `mesh_dataset`, `keys`, `core`, `distributed`, `optim`, `metrics`, `experimental/models/globe/model.py` (book parses it), `pyproject`/`uv.lock` (tensordict pin) | Unchanged by main | Heavily used | b (untouched) |
| Strata, GraphCast/MGN distributed, darcy datapipes, domino utils, docs and tutorials, CI | Various | Not used | a |

## Conclusions

- **What changes numbers:**
  - **ISLA tests (P1):** the torus winding change makes two tests fail, and they need a fix before the PR.
  - **Baselines' SDF inputs:** they change only near sliver faces. Transolver and GeoTransolver volume runs read them; ISLA does not.
  - **Float64 meshes through `Mesh.rotate`:** their rotation matrices change by about 1e-7 relative.
- **Measures, LSQ, BVH and facets:** ISLA calls none of the code main fixed for measures (`measure.py` is untouched; padding, extrude and facet extraction are not called) or for LSQ gradients. The BVH change is only a traversal ISLA does not use.
- **Old import paths:** nothing in ISLA, the recipe, the book or `examples/cfd/isla` uses an old path in a682959bc. `weighted_multinomial` has a warning shim. `nn.functional.safe_normalize` has none, and nothing in scope uses it.
- **Files both sides changed:** these merged cleanly, and main's side only adds the `validate_args` and import-path lines. `train.py` comes out identical to ISLA's side.
- **Not run:** the full ISLA suite on the merged tree. I ran the torus and split tests of `test_isla_interior_geometry.py` (2 failed, 9 passed) and `test_isla_geometry_fast.py` (20 passed, 100 skipped). Every other ISLA test input built from main's primitives is bitwise unchanged (probe).

Nothing I made stays on disk; the probe archives in `scratch/isla-main-merge-2026-10-24/impact/` are deleted.

**Decision needed:** pin the old torus winding in the two tests (`torus.load(...).cells[:, [0, 2, 1]]` or the `_torus()` helper) so the merge can land, or first make `exact_boundary_geometry` bitwise invariant for both windings? I recommend pinning the winding in the merge and tracking the invariance as an ISLA follow-up.

{"verdict": "block", "question": "Two ISLA tests fail only because main's #2148 reversed torus.load's winding: pin the old winding in those two tests to land the merge (recommended), or first make exact_boundary_geometry's float32 output bitwise invariant for both windings?"}