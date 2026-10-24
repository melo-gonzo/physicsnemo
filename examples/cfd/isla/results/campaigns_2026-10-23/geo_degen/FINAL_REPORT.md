Status: DONE

**Outcome.** The zero-area cell fix is in the reference and the fast path. It is bitwise unchanged on every input without a
zero-area cell, and correct on 291,100 degenerate-input queries, where `0570fc6c8` raised on 38 sets and gave 4,629 wrong
distances. Commits on `psharpe/isla-geo-fast` (pushed, not merged; `git commit -s`, hooks on):
- `a9592e9a3`: a zero-area cell is the segment or the point its vertices span.
- `1086cf42e`: a zero-area cell lying in the other cells is left out.

ISLA version and `_refuse_earlier_state_dict` are unchanged, and there is no new custom op.

**Design** (`DESIGN.md`, md5 18c7bec9; the supervisor's decision, not Peter's: fix, do not refuse):
- **Flag** (`_zero_area`): the cross product is tested exactly zero. A float64 filter with Shewchuk's error bound runs first;
  the cells it leaves get exact products summed as a nonoverlapping expansion. There is no tolerance constant. It is exact for
  all float32 coordinates, and for float64 magnitudes in [2^-485, 2^510].
- **Ball search**: flagged cells get NaN planes and corners, so no lower bound drops them.
- **Closest point** (`_closest_weights(flat=)`): the nearer of the two edges between consecutive canonical vertices.
  - This differs from GEO-FAST's "longest edge" on purpose. That rule was right in d, but at a T-junction it created a feature,
    the long edge, that no mesh cell shares, which changed u at 91 of 2,964 queries.
- **Exact phase and ties**: a flagged cell contributes its distinct vertices and edges to the stars, has no normal (TIE-FIX's
  rule, now also for the order (Q, P, P), where torch's cross product gave a unit vector of rounding), and parity 0 in the
  winding number.
- **On-wall query held only by zero-area cells**: d = 0 and u = 0, the least-norm element of the generalized gradient. Tested.
- **Covered cells** (`_covered`, an index test, no arithmetic): a zero-area cell along the edges or at a vertex of other
  cells is dropped. The outputs are then bitwise those of the mesh without it. This fixed TIE-FIX's test, which `a9592e9a3`
  broke in the last bits off the wall.
- **Fast path**: the same code on the same values, with the flag permuted into the tree order.

**Checks** (tables in `RESULTS.md`, md5 f151d5ea):
- **a, unchanged** (head against `0570fc6c8` loaded with both ops renamed; every op output, w, d, u and their gradients):
  - CPU, 1 thread: 0 of 427,884 queries differ in 50 cases, under each kernel.
  - GB300: 0 of 2,265,906 differ in 102 cases, under each kernel. The cases are the small set, DrivAerML vehicle and whole
    boundary, AeroZoo, the 20 lug cases, AirfRANS and Tensile2d.
- **b, correct on degenerate input** (152 set-and-precision runs, 291,100 queries, 1,919 zero-area cells, on CPU and GB300
  with identical counts):
  - 0 raises, 0 non-finite, 0 distances beyond 8 u(|x|+R) (largest 2.27), 0 wrong cells, 0 wrong signs of 239,824 checked, and
    0 bad of 3,024 queries held only by zero-area cells.
  - Adding covered cells to closed meshes changes 0 of 110,948 queries.
  - The base on the same sets: 38 raise (GEO-FAST's 5 + 33), 4,629 distances beyond the bound, 3,999 wrong cells, 14 wrong signs.
- **c, fast against reference, bitwise**: 0 of 1,958,096 queries differ over 86 runs (adversarial sets and every dump; tie sets
  included), 0 on the degenerate sets, and 0 of 520,761 queries in the forced-fast GPU tests.
- **d, tests**:
  - New file `test_isla_geometry_degenerate.py`: 18 CPU tests plus 16 CUDA tests. At `0570fc6c8`, 13 of the CPU tests fail.
  - Whole ISLA directory on LHR cpu-short (365178): 358 passed, 100 skipped, 0 failed.
  - Recipe tests: 506 passed, 12 skipped.
  - GB300 geometry, tree-read, geometry-fast and degenerate files: 208 passed, 1 skipped, both forced and under auto.
- **Cost** on DrivAerML whole boundary (16.65M cells): `_zero_area` takes 2.4 ms. The geometry call goes from 406 to 412 ms
  under the reference kernel, and from 340.5 to 341.1 ms under the fast kernel.

**What it means.** Canonical ISLA now takes meshes with zero-area cells, as CAD exports produce, without a crash or a wrong
answer. No trained result changes, since real dumps hold no such cells (0 of 16.65M on DrivAerML) and every other output is
bitwise unchanged. Confidence is high: the bitwise checks are exact, and the brute force shares no code with geometry.py.

**Two limits, both outside this brief, which stay as they were at the base:**
- **Thin cells of nonzero area** break the reference the same way. On float64 soups whose slivers sit 1e-13 off the line,
  8 of 10 soups raise; with the plane bound off, 3 of 14,000 queries are wrong. I sent this to the supervisor at 02:10.
  - A fix changes non-degenerate outputs, so it needs its own lane. Proposed fix: a per-cell certified plane bound, and the
    closest point taken as the least of the face projection and the three edge points.
  - The float32 cast of TIE-FIX's midpoint is such a thin cell (5 of 1,646 queries, the same as at the base).
- **Overlapping coplanar cells** change u at ties. Zero-area cells inside a face but off its edges are kept as segments
  (905 of 9,768 float32 queries change). The control, triangles of nonzero area in the same place, changes 891 at the head and
  at `0570fc6c8` alike.

**Rulings relied on.**
- The supervisor's fix-not-refuse decision and its principle.
- Peter's boundary-only input ruling of 2026-10-17.
- LANE_RULES on cross-version op renaming, suites on LHR, and registry rows.

For project memory: "ISLA geometry: a zero-area cell is the set its vertices span; covered ones are dropped exactly; thin
nonzero-area cells remain a known reference defect (GEO-DEGEN, 2026-10-24)."

**GPU-h**: 0.66 of the 4 GPU-h cap, from 364908 and 365177 (one 4-GPU node each, short QoS). Two cpu-short jobs ran:
364909 and 365178. Nothing is running, and the registry row has been removed.

**Left.** The supervisor merges the branch, and decides on a thin-cell lane.

**On disk.**
- The worktree `/home/psharpe/gh/physicsnemo-isla-geo-fast` (GEO-FAST's, about 1 GB).
- On LHR: `$T/code_geodegen3` (57 MB, the head snapshot) and `$T/kernel/geodegen` (424 KB).
- Scratch: 1.2 MB, holding DESIGN, RESULTS, JOBS, probes, logs and the registry backups.
