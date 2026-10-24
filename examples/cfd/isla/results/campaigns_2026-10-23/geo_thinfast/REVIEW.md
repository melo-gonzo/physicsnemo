# GEO-THINFAST review: psharpe/isla-geo-thinfast at d42fc83d5 (base 3f3fef69f)

## Findings, worst first
1. `physicsnemo/experimental/models/isla/geometry_fast.py:441` · **P0** · A column-major `q` breaks the fused path: it returns wrong weights, and the fast op can crash. `out = torch.empty_like(q)` keeps a dense permuted layout (strides `(1, n)`), but the kernels write `out + 3*r + k` row-major. `q.contiguous()` fixes only the input. Real callers reach this: `query_points.double()` and `q64 - origin` (geometry.py:1898) both keep strides `(1, N)`, so `w0` (geometry.py:1925) gets a column-major `q64` whenever a float32 or float64 query tensor is column-major, for example `pts.T` or `query_points[i]` taken from a transposed batch in model.py. `geometry_kernel="auto"` uses this path on CUDA. Evidence comes from probes run under TRITON_INTERPRET=1 on CPU with `_usable` patched; the layout bug does not depend on the device. `probes/review_layout.py`: with contiguous `q`, the fused path is within 3e-16 of the reference; with column-major `q` (same values), 300 of 300 rows differ by up to 1.0. `probes/review_layout_op.py`: `exact_boundary_geometry(kernel="fast")` runs on contiguous queries but fails on the same queries column-major with `IndexError: index 81711 is out of bounds` (geometry.py:1965). `w0` is wrong, so the radius is too small and one query gets no candidate. **Fix:** fall back to the reference unless `q.is_contiguous() and simplex.is_contiguous()`, which is decided from metadata before the forward. Only allocating `out` contiguous is not enough: the reference's `q.norm(dim=-1)` then reduces a strided axis, and P0 never measured that order, so `unit` (and so the guard and switch) may not match. **Test:** add a column-major case (`q.T.contiguous().T`) to `test_guard_kernel_equals_the_torch_weights_bitwise` and to `test_fast_search_equals_the_reference_on_thin_cells`.
2. `geometry_fast.py:274, 292, 354` · **P1** · The int32 row offsets lower the op's size limit from about 2^31 pairs to about 238M. Past that, `9 * r` overflows to a negative offset, so the kernels silently read or write outside the buffers. The second call runs over all candidate pairs (`q64[li]`, which is not chunked), so a few million queries at tens of candidates each reach this on a GB300, and the reference would still fit in memory there. **Fix:** cast `r` with `.to(tl.int64)` in both kernels (after the `tl.load` in `_accurate_kernel`); `guarded` can stay int32. The writer flagged this risk.
3. Pre-existing, P2 for the lane: the reference itself depends on layout. On CPU, `exact_boundary_geometry(kernel="reference")` returns `d` that differs on 140 of 2,000 icosphere queries, by up to 8.9e-16, between contiguous and column-major float64 queries. This is not this branch's change, but "bitwise for every input" holds only per layout.
4. Tests, P2: NaN coordinates are untested (writer's note). So are `n < 256` (every set has more than 256 rows; partial and full last blocks are both covered), `n > 0` with zero guarded rows (`_accurate_kernel` with `count == 0`), and a `-0` `t` from underflow through the clamp (it is the same `max.f64` in both paths, but never exercised). The rest is covered: all six Ericson regions and the face default (random queries on faces, edges, vertices, near and far), the edge recomputation, the guard on and off, the switch taken (asserted on caps and needles) and never taken (`switch=inf`), zero-area rows, empty input, float32-derived and float64 inputs, and the CPU fallback (1 passed locally).
5. Style, P2: `geometry_fast.py:130`'s section comment still says "the thin-cell guard" although it now heads Ericson's weights and everything after it. `_load_rows`, the kernel signature and `l2` are 150–230-character lines, where the base file had one line over 120. `test_the_band_units_are_the_derived_constants` points its message at a git-ignored scratch path.

## Where I found nothing
- **Bitwise forms, checked line by line against `_closest_weights`, sdf `_closest_point_weights`, `_segment_t`, `_segment_distance2`, `_thin_bounds`, `_ericson_error`, `_accurate_weights`, `_cross_expansion`, `_two_sum` and `_two_product`:**
  - Every sum, norm and vertex-axis sum uses P0's order.
  - The cross product is the measured fma form, and `-x` is written `x * -1.0`.
  - The `scalar / tensor` in `kappa` is computed as `(1.0 / ...) * KAPPA`; `1 - t` and `zeros` come out as `+0`.
  - clamp, clamp_min, minimum and amax all propagate NaN.
  - Ericson's region chain runs in the same order.
  - The `ends` table matches, including `argmax` over `w == 0` with its default of 0.
  - The longest-edge `argmax` and the four-way `argmin` take the first index and treat NaN as first. Torch's index tie-break makes those results independent of reduction order.
  - The scatter layouts of `slots` match.
  - The Grow-Expansion and its summation order match.
  - The switch recomputes `dist` from the stored weights; flat rows are never guarded, so the flat replacement in kernel 1 is safe.
  - NaN payloads in `unit`, `l2`, `kappa` and `dist` reach only comparisons.
- **Compaction:** one atomic per block plus a block cumsum lists each guarded row exactly once. Kernel 2 writes only those rows, so the output is deterministic even though the list order is not. Kernel 1 writes every row below `n`, programs past the count return, and nothing synchronizes with the host.
- **Execution choice:** `_usable` (device and Triton) and `n == 0` are checked before any launch, with no try/except and nothing in a state dict. The 2-D path is unchanged. `geometry.py` changes only by the `fast` argument, its dispatch, the two call sites and docstrings; the reference arithmetic is untouched.
- **Hygiene:** the commit messages follow the repository's style, with no co-author trailers.

Probes kept: `scratch/isla-geo-thinfast-2026-10-24/probes/review_layout{,_op}.py`. No GPU was used.

Merge verdict: BLOCK
{"verdict": "block"}
