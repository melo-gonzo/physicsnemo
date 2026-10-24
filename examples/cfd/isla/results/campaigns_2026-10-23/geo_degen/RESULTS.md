# GEO-DEGEN results (2026-10-24 03:12 EDT)

Branch `psharpe/isla-geo-fast` (worktree `/home/psharpe/gh/physicsnemo-isla-geo-fast`), base `0570fc6c8`, pushed:
`a9592e9a3` (the fix) and `1086cf42e` (covered zero-area cells left out). Head geometry.py md5 `dd492ac1`; GB300 snapshot
`$T/code_geodegen3` (all 3,173 files of `physicsnemo/` and `test/` equal `1086cf42e`, md5-compared). Design: `DESIGN.md`.
Jobs: `JOBS.md`. Every probe prints the md5 of the geometry.py it ran (head `dd492ac1`, base `0570fc6c8` = `0ea245be`).
All numbers below are at the head `1086cf42e` unless a row says otherwise.

## Check a: unchanged on inputs without a zero-area cell (bitwise, 0 differences required)

`probes/probe_a.py` (GEO-FAST's probe_c1.py): the head against `0570fc6c8`'s geometry.py loaded by `probes/refload2.py`
(both custom ops renamed, `_geometry_ref`; each side prints the op it ran). Compared per query: all six op outputs (cell,
tied, tie direction, sign, on-wall, weights), w, d, u, dd/dq + du/dq, and d/dp; float32 and float64. Run once with
`kernel="reference"` on both sides and once with `kernel="fast"` on both sides.

| where | cases | queries (per kernel) | kernel reference: differing | kernel fast: differing | log |
|---|---|---|---|---|---|
| CPU, 1 thread: 20 lug, icosphere, torus (each with and without repeated cells), ellipse | 50 | 427,884 | 0 | 0 | `work/a2_cpu_{ref,fast}.log` |
| GB300 (365177): the 50 small cases + DrivAerML Volume vehicle | 52 | 827,884 | 0 | 0 | `lhr/out_checks2/a_small_dmlv_*.log` |
| GB300: DrivAerML Volume whole boundary | 2 | 400,000 | 0 | 0 | `a_dmlw_*.log` |
| GB300: AeroZoo, 20 lug cases + an AeroZoo 20k case, AirfRANS, Tensile2d | 48 | 1,038,022 | 0 | 0 | `a_az_lug_af_t2d_*.log` |

The same check at `a9592e9a3` (364908, local 02:17-02:28) was also 0 everywhere. Cost on DrivAerML whole boundary
(16.65M cells, 200k queries, GB300, `cost_dmlw.log`): `_zero_area` 2.4 ms; the geometry call (two interleaved passes)
reference 406.0 to 411.7 ms, fast 340.5 to 341.1 ms. Real dumps have 0 zero-area cells (GEO-FAST's count, and
`_zero_area` here on DrivAerML: 0 of 16,652,301).

## Check b: correct on degenerate input

`probes/probe_degen.py` against `probes/brute.py`, an independent float64 brute force over every cell (zero-area cells as
their segments or points, no geometry.py code). Bound: `||d| - d_exact| <= 8 u (|x| + R)` (u the unit roundoff of the
precision, R the greatest vertex norm). Wrong cell: the op's cell farther than the exact least by more than the op's own tie
bound `E_idx + E_min`. Sign: off-wall queries with `|sigma/2 - w_exact| > 0.05`, against the exact winding sign. `only0`:
on-wall queries held only by zero-area cells (required: d = 0 and u = 0). `inv`: closed meshes with zero-area cells added,
against the same mesh without them: |d| beyond the bound or |du| > 64 u. `fast`: kernel fast against reference, bitwise.

Totals, CPU (1 thread, `work/b2_cpu.log`) and GB300 (365177, `lhr/out_checks2/b_degen.log`), identical counts:

| | sets | queries | zero-area cells | covered (dropped) | raise / device assert | non-finite | d beyond bound | wrong cell | sign wrong (of checked) | only0 bad (of) | fast differing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| head, CPU | 152 | 291,100 | 1,919 | 499 | 0 | 0 | 0 (max 1.96 u(\|x\|+R) f32, 1.79 f64) | 0 | 0 (239,824) | 0 (3,024) | 0 |
| head, GB300 | 152 | 291,100 | 1,919 | 499 | 0 | 0 | 0 (max 1.30 f32, 2.27 f64) | 0 | 0 (239,824) | 0 (3,024) | 0 |
| `0570fc6c8`, CPU (`work/b_cpu_base0570.log`) | 152 | 291,100 | | | 38 sets raise IndexError | 0 | 4,629 (of the 253,100 queries in sets that ran) | 3,999 | 14 | 0 | |

The base's raises are GEO-FAST's: float32 5 of 40 soups, float64 33 of 40, with 260 and 80 queries beyond the bound in the
soups that ran, exactly GEO-FAST's counts.

Per set (head): `work/b2_cpu_table.md` (CPU), `lhr/out_checks2/b_degen_table.md` (GB300); base: `work/b_cpu_base0570_table.md`.
Sets, float32 and float64 each: GEO-FAST's 40 soups; probe_c2's soup; isolated lattice soups of the three cases (5 each);
the three cases added to a closed box and the two coincident cases to a closed icosphere (along edges, at vertices, over
straight runs of two edges), unposed and offset by 1024; dangling segments at a box's corners; zero-area cells inside a
box's faces; TIE-FIX's `(a, b, midpoint)` on an icosphere edge; a fold (finned cube, cells on the fin's edges and corners);
medial axes (a plate's mid-plane with cells on both faces; equidistant queries between a segment and a square).

`inv` (adding zero-area cells to a closed mesh changes nothing): 0 of 110,948 queries on the box, icosphere, fold, plate
and float64 TIE-FIX sets, whose zero-area cells are all covered (the base: 75 of them differ). Not zero, and not degenerate-specific:
- zero-area cells inside faces, off the edges (kept as segments): 905 of 9,768 float32, 1 of 9,768 float64. These are
  coplanar overlaps. The control row, triangles of nonzero area placed inside the same faces (no zero-area cell), gives
  891 float32 and 384 float64 with the head and the same 891 and 384 at `0570fc6c8`: overlapping coplanar cells change u at
  ties in the existing tie rule. The base, for the zero-area version, has 404 and 0 (it ignores the cells as planes).
- TIE-FIX's set in float32: 5 of 1,646. In float32 the midpoint is not exactly on the edge after the cast, so the cell has
  nonzero area (correctly not flagged: `zero_area` 0); it is a thin cell, the out-of-scope defect below. The same 5 at the
  base. In float64 it is flagged, covered, and 0 of 1,646 differ.

## Check c: fast against reference, bitwise

| check | where | sets | queries | differing |
|---|---|---|---|---|
| `probes/probe_c2.py`: candidate pairs (so the tie sets), tie groups and directions, op outputs, w, d, u | GB300 (365177): 32 adversarial sets, AeroZoo, the lug cases, AirfRANS, Tensile2d (`c_adv_az_lug_af_t2d.log`), DrivAerML vehicle (`c_dmlv.log`), whole boundary (`c_dmlw.log`) | 86 | 1,958,096 (250,286 on the wall, 165,999 tied, 1,372,648 tie members) | 0 |
| probe_degen's `fast` column: op outputs, d, u and gradients | CPU 1 thread and GB300 | 152 + 152 | 291,100 + 291,100 | 0 |
| check a with `kernel="fast"` (head fast against base fast, which GEO-FAST showed equal to base reference) | CPU, GB300 | 50 + 102 | 427,884 + 2,265,906 | 0 |
| fast_plugin (fast forced at every op call, compared with the reference) in the GPU tests | GB300 | 367 calls | 520,761 | 0 |

On CPU with 4 threads (an early run, `work_probe_degen_cpu_try1.log`) the float32 gradients differed between the kernels
on 1,449 queries of the icosphere sets: the reference's own run-to-run variation at more than one thread, which GEO-FAST
recorded. At 1 thread and on CUDA it is 0.

## Check d: tests

| suite | where | head `1086cf42e` | notes |
|---|---|---|---|
| new `test_isla_geometry_degenerate.py` (flag; the three cases' distance in f32/f64 on CPU and CUDA; adding cells changes nothing; query held only by zero-area cells; 2-D zero-length segments) | local CPU | 18 passed, 16 skipped (CUDA) | at `0570fc6c8`: 13 failed, 5 passed |
| whole ISLA test directory | LHR cpu-short (365178, `lhr/tests_isla.geodegen.tests2_365178.log`) | 358 passed, 100 skipped, 0 failed | GEO-FAST at `0570fc6c8`: 340 / 84; the new file adds 18 + 16. At `a9592e9a3` (364909): 1 failed (TIE-FIX's test, fixed by `1086cf42e`) |
| recipe tests | local CPU | 506 passed, 12 skipped | |
| geometry, tree-read, geometry-fast, degenerate files, fast forced (fast_plugin) | GB300 (365177) | 208 passed, 1 skipped | FAST-PLUGIN 0 of 520,761 queries differ |
| same files under auto | GB300 | 208 passed, 1 skipped | |

## Out of scope, found here: thin cells of nonzero area

At `0570fc6c8` and at the head alike (no zero-area cell is involved): float64 soups with 8 slivers, the middle vertex moved
off the line by a relative 1e-13, raise IndexError in 8 of 10 soups (1e-16: 10 of 10); float32 slivers and float64 at 1e-10
and above: 0 of 14,000 wrong. With the plane bound off, none raise, but 3 of 14,000 queries get a farther point (the float64
Ericson weights misplace the closest point by up to 3e-4 relative). `probes/probe_thin.py`, `probes/diag_thin.py`. Reported
to the supervisor at 02:10 EDT. A fix changes outputs on inputs without zero-area cells, so it is a separate lane.

## GPU

364908 (5:06) and 365177 (4:52), one 4-GPU node each, short QoS: 0.66 GPU-h of the 4 GPU-h cap. CPU: 364909 (15:11) and
365178 (11:21), cpu-short.
