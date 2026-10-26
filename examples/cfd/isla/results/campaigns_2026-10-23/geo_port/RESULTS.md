# ISLA-GEO-PORT results

## 1. The 602-run cross-version set, BASE against head (eager, each version in its own process)

Run twice: at 6fcb1c6e1 (F2; jobs 397456 GPU, 397457 CPU) and at the final head a7a8d1338 (jobs 397953 GPU, 397954 CPU,
head geometry.py md5 0dcb76f4). Every count below is identical in the two runs; the table names the first jobs.

BASE side: snapshot code_103b546ee (geometry.py md5 20156177; bf71190af touches no ISLA file, so this is also the
geometry of BASE + bf71190af). Head: code_6fcb1c6e1 (geometry.py md5 d00091b9). Harness: lhr/xv_dump.py, xv_cmp.py,
thinsets.py (md5s in JOBS.md); each process printed the op it ran (`TAG base called _closest_candidates ran
['physicsnemo::isla_closest_candidates']`, and so on for head and for `_fast`). Sets as PR-P7: GPU big 20 runs
(PROFILE drv_whole, drv_body, aerozoo, lug, airfrans_whole, airfrans; GEO-THIN dmlv_whole_s42@pose, s43@renum), GPU
small 326 runs (af_whole_s42..45 with pose/renum, aerozoo_uvolume, lug20_az, Tensile2d, AirfRANS 2-D, adv_az_case0,
adv_az_dump, two HiLift slabs, the 88 thin sets, the 20k-segment ellipse and its renumbering), CPU 256 runs (one
thread). Each run in float32 and float64. Compared per run: the six op outputs, w, d, u, d's gradients in the
queries and the vertices, u's gradients in the queries and the vertices.

| device / sets | kernel | runs | runs bitwise in every tensor | runs differing only in u's gradients | edge-region (slide) rows | grad_q_u rows differ | outside slide rows | grad_p_u vertex rows differ | outside slide-cell vertices | runs differing elsewhere |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GB300 big (job 397456) | reference | 20 | 4 | 16 | 503,855 | 503,855 | 0 | 1,002,435 | 0 | 0 |
| GB300 big | fast | 20 | 4 | 16 | 503,855 | 503,855 | 0 | 1,002,435 | 0 | 0 |
| GB300 small | reference | 326 | 30 | 296 | 186,583 | 186,583 | 0 | 101,378 | 0 | 0 |
| GB300 small | fast | 326 | 30 | 296 | 186,583 | 186,583 | 0 | 101,378 | 0 | 0 |
| CPU, 1 thread (job 397457) | reference | 256 | 16 | 240 | 93,331 | 93,331 | 0 | 35,488 | 0 | 0 |
| CPU, 1 thread | fast | 256 | 16 | 240 | 93,331 | 93,331 | 0 | 35,488 | 0 | 0 |

- 602 runs per kernel, 1,204 in all: every op output, w, d, u and d's gradients bitwise in every run.
- u's gradient changes at every edge-region query row (3-D, off the surface, untied, two nonzero weights) and at no
  other row; in the vertices only at the welded canonical vertices of those rows' closest cells.
- The bitwise runs are the 2-D runs (AirfRANS, Tensile2d, the ellipse), where F2 does not apply, and the four float32
  HiLift slab runs, whose 50,000 queries are all on the wall in float32 (no slide rows).
- Harness check before: BASE against itself on CPU (job 395318), 256 + 256 runs bitwise.
- Not covered by these sets: float16/bfloat16 queries (F1; covered by the new test), the chunked least-norm path (F3;
  no set reaches it; covered by the K = 100 test), zero cells / zero queries (new tests).
- Per-run logs: out_gb_397456/xv_*_cmp.log, out_cpu_397457/*_cmp.log.

## 2. Timings (job 397456, one GB300 per (case, kernel) stream, forward call, median of 10 calls after 3 warm-up,
two repetitions in BASE, head, BASE, head order; peak = max allocated above the inputs)

| case | kernel | BASE ms (rep 1, 2) | head ms (rep 1, 2) | BASE peak GiB | head peak GiB |
|---|---|---|---|--:|--:|
| DrivAerML whole domain (Q 200,000, F 16,652,301) | reference | 524.6, 525.6 | 465.4, 460.5 | 17.16 | 13.34 |
| DrivAerML whole domain | fast (the model's default on CUDA) | 404.6, 404.2 | 348.2, 345.8 | 17.44 | 12.40 |
| AeroZoo (Q 200,000, F 1,354,543) | reference | 184.0, 184.7 | 181.7, 182.5 | 1.41 | 1.10 |
| AeroZoo | fast | 110.1, 110.7 | 110.4, 112.5 | 1.44 | 1.03 |
| lug (Q 9,048, F 6,660) | reference | 75.2, 75.5 | 72.4, 77.8 | 0.05 | 0.04 |
| lug | fast | 34.2, 33.4 | 33.1, 35.0 | 0.04 | 0.04 |

PR-P7's library (fast traversal, no Triton) measured 521.9 to 397.4 ms and 17.16 to 12.40 GiB on the same DrivAerML
call. ISLA's fast path (with Triton) goes 404 to 347 ms (-14 %) and 17.44 to 12.40 GiB (-29 %); its reference path
525 to 463 ms (-12 %) and 17.16 to 13.34 GiB (-22 %). AeroZoo and the lug: time unchanged within noise, peak -22 to
-28 % on AeroZoo. Raw: out_gb_397456/m2_*.json.
