# GEO-THINFAST results

## P0: torch's float64 CUDA arithmetic on GB300 (jobs 369877, 369899, 369906; torch 2.12.0+cu130, Triton 3.7.0)

Probes `probes/prims.py`, `prims2.py`, `prims3.py` (logs `$T/kernel/geothinfast/out/prims*_*.log`): candidate explicit
forms in Triton (`enable_fp_fusion=False`) against torch, bitwise (int64 views), up to 4M rows of mixed magnitudes
(2^±40), exact zeros of both signs and NaNs.

| torch expression | form that matches on every row | notes |
|---|---|---|
| `.sum(-1)` over a contiguous last axis of 3 (`(x*y).sum(-1)`, `(x**2).sum(-1)`, `(P,3,3).sum(-1)`) | `((x0 + x2) + x1) + 0.0` | two threads per output (block width last_pow2(3) = 2); identity 0 turns -0 into +0 |
| `.norm(dim=-1)`, (P,3) and (P,3,3) | `sqrt((x0*x0 + x2*x2) + x1*x1)` | squares rounded, no fma; every fma form fails |
| `.sum(1)` over the vertex axis of (P,3,3) | `((a0 + a1) + a2) + 0.0` | sequential; never -0 |
| `torch.linalg.cross` (= `torch.cross`) | comp i: `fma(a1, b2, -(a2*b1))` with an exact negation | nvcc contracts; 0 of 4M rows differ incl. NaN rows |
| `.square()`, `** 2` | `x * x` | |
| `scalar / tensor` | `tensor.reciprocal() * scalar` | torch/_tensor.py `__rdiv__` |
| `/`, `sqrt` | Triton fp64 `/`, `tl.sqrt` | correctly rounded both |

Traps found: Triton's unary minus is `0 - x` (semantic.py `minus`), so `-(+0.0)` is +0.0 where torch's neg gives -0.0;
use `x * -1.0` (a sign-bit xor differs from torch's neg on NaN). Triton contracts `a*b + c` to fma by default (77 % of
rows match torch); `enable_fp_fusion=False` gives 100 %.

## What was built (branch psharpe/isla-geo-thinfast, pushed at b959a6735)

`geometry_fast.closest_weights`: the whole three-dimensional `_closest_weights` (Ericson's weights from sdf.py, the edge
recomputation, the thin-cell guard, the zero-area segments) in one Triton kernel over the pairs, and the accurate point
with the switch in a second kernel over the guarded pairs only (listed by the first, no host sync), each operation in
torch's CUDA form and order (P0), launched with `enable_fp_fusion=False`, negation as `x * -1.0`. Used only by the fast
op on CUDA with Triton, on contiguous inputs (a metadata decision before launch); otherwise the reference torch code,
which is unchanged. Row offsets int64. Writers' dev jobs: 0 of 12.4M+ resampled pairs and every test set differ
(REPORT_WRITER.md, REPORT_WRITER2.md); an independent review (REVIEW.md) found the column-major and int32 issues, fixed
in 1730022cb.

## Checks at code_geothinfast1 (b959a6735), GB300 job 370582 (logs $T/kernel/geothinfast/out/checks_370582/)

Bitwise = int views (signed zeros and NaN payloads count), on the op's six outputs, d, u, grad wrt queries and vertices,
float32 and float64; `geometry_kernel="auto"` resolved to 'fast' on the GB300.

| check | runs | queries | differing |
|---|--:|--:|--:|
| check_tf, DrivAerML: dmlv_whole s42, s43, drivaer_ml_volume_reference (vehicle), drivaer_ml_volume_whole, 200k each; fast and auto vs reference | 16 | 3,200,000 | 0 |
| ... per-call weights hook (seed and every candidate pair) | 32 calls | 14,360,548 pairs | 0 |
| check_tf, every other dump: AirfRANS whole s42 (+posed), s43-45, airfrans_uvolume_2d, AeroZoo 200k (+posed), lug20 all 20 cases (+posed), Tensile2d (+posed), GEO-THIN's adversarial AeroZoo sets adv_az_case0 and adv_az_dump, and GEO-THIN's 88 adversarial thin-cell sets (thinsets.py = probe_mech.py's sets), each also with column-major queries | 1,224 | 13,327,248 | 0 |
| ... per-call weights hook | 2,320 calls | 32,177,452 pairs | 0 |
| FAST-PLUGIN (fast_plugin.py forced over the whole ISLA directory but test_isla_compile.py; every reference call also run fast) | 694 calls (563 3-D) | 673,850 | 0 |
| cross-version, the 2-D prefilter margin 32 -> 40 (dump_out.py at code_geothinfast1 and at code_geoband2 = psharpe/isla's geometry.py, separate processes; 2-D both kernels, native and posed; 3-D reference) | 294 + 4 (DrivAerML) | | 0 |

Tests, the whole ISLA directory:
- GB300 (370582), split over GPUs by file and keyword: test_isla_compile.py `-k "not cpu"` 9 passed, `-k cpu` 5 passed;
  the rest under auto 606 passed, 1 skipped (ISLA_AF_POSED_SAMPLE unset): 620 passed, 1 skipped, 0 failed.
- The rest again under fast_plugin (forced): 605 passed, 1 skipped, 1 failed:
  `test_isla_conditions.py::test_a_change_of_condition_subset_compiles_once_per_subset`, a RecompileError whose guard is
  fast_plugin's own counter (`G['__import_fast_plugin'].stats['calls_compiling'] == 0`): an artifact of the plug-in's
  global mutation under torch.compile; the test passes under auto in the same job.
- CPU, cpu-short 370585: 413 passed, 200 skipped (CUDA), 0 failed.

## Timing, GB300 job 370584 (lhr/timing_table.md; reduce lhr/reduce_timing.py)

GEO-BAND's phase-0 harness (FAST-MERGE / GEO-FAST final2 protocol: bfloat16, eager, medians after 5 warm-up calls or
steps, 4 runs at once, GPUs rotated), arms interleaved: base = 2e9cd7fda under auto (fast; 3 passes), head =
b959a6735 under auto (fused; 3 passes), reference (2 passes). Means over passes:

| case (Q / F) | geometry call: base fast / head fast / reference [ms] | head / base (call) | step: base / head / reference [ms] | head / base (step) |
|---|--:|--:|--:|--:|
| lug (9808 / 6,488) | 74.1 / 37.6 / 83.5 | 0.51 | 204 / 166 / 213 | 0.82 |
| AirfRANS 2-D (100000 / 1,018) | 31.7 / 31.6 / 33.2 | 1.00 | 356 / 357 / 361 | 1.00 |
| AeroZoo (200000 / 501,802) | 113.7 / 92.8 / 135.9 | 0.82 | 598 / 581 / 627 | 0.97 |
| DrivAerML vehicle (200000 / 16,953,702) | 384.4 / 345.2 / 450.4 | 0.90 | 867 / 829 / 942 | 0.96 |
| DrivAerML whole boundary (200000 / 16,958,756) | 436.1 / 403.3 / 514.3 | 0.92 | 930 / 897 / 1,009 | 0.96 |

The lug's fast call (37.6 ms) is back at FAST-MERGE's pre-thin-fix 36.7 ms; on DrivAerML the fused path removes 33-39
ms of the ~80 ms the thin-cell fix and the bands added (what remains is F2's ball-search test, +32-34 ms per GEO-THIN,
and the bands' +14 ms). Peak memory unchanged (same per case across arms). AirfRANS (2-D) does not use the kernel.
