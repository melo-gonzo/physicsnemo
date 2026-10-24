# GEO-FAST results (2026-10-24 01:40 EDT)

Branch `psharpe/isla-geo-fast` (worktree `/home/psharpe/gh/physicsnemo-isla-geo-fast`), on `f13eb5ce0`, pushed, not merged.
Head `0570fc6c8`; GB300 snapshot of the head content `$T/code_geofast3c` (geometry.py md5 0ea245be, geometry_fast.py 37d1bbf7,
model.py c0975d69, test_isla_geometry_fast.py f73983d8). Design: `DESIGN.md` (md5 4e38af75, approved 19:48 EDT with decision 3
changed). Jobs and every log: `JOBS.md`, `lhr/out_*`.

## Commits

| commit | what | gate |
|---|---|---|
| `fb263fa29` | the reference's search radius seeded by a float32 nearest-cell traversal of its own cell tree (`_nearest_cells`), not by `signed_distance_field`'s per-call Warp mesh BVH | C1 bitwise; CPU dir 321 passed 52 skipped (= f13eb5ce0) |
| `b30ff5c33` | the fast op `physicsnemo::isla_closest_candidates_fast` (preorder tree by one kernel, cells in leaf order, ball search on it, winding on packed records); `exact_boundary_geometry(kernel=)`, `ISLA.geometry_kernel` | C1, C3 bitwise; CPU dir 340 passed 70 skipped |
| `543400873` | the fast path's weld scan and canonical cells in Triton (`geometry_fast.py`, importlib, HAS_TRITON), dedupe by packed-key sorts; auto = fast iff CUDA and Triton | C1, C3 bitwise; CPU dir 340 passed 84 skipped |
| `0570fc6c8` | the fast path builds the lower-bound planes as the reference does (a Warp planes kernel differed on coincident-vertex cells) | C1, C2, C3, C6 bitwise; CPU dir 340 passed 84 skipped |

ISLA version and `_refuse_earlier_state_dict` unchanged. The reference's numerics are unchanged: C1 is bitwise at every commit.

## Correctness at the head (0 differences required, 0 found)

Every comparison is bitwise (bound 0 ulp). C1 = the head's reference against f13eb5ce0 (refload.py, op renamed); C2/C3 =
`kernel="fast"` against `kernel="reference"` (two ops of one module). Probes print the md5 of the geometry they ran.

| check | where | sets | queries | differing |
|---|---|---|---|---|
| C1: op outputs, w, d, u, dd/dq + du/dq, d/dp | CPU 1 thread (`work/c1_cpu_head.log`) | 50 (20 lug, icosphere, torus, each +/- repeated cells, ellipse; x f32/f64) | 427,884 | 0 |
| C1, same | GB300 (363374) | 98: 50 small, DML vehicle 2, DML whole 2, AeroZoo 2, 20 lug x 2, an AeroZoo 20k case x 2 | 2,043,456 | 0 |
| C3: fast vs reference, the same outputs and gradients | CPU 1 thread (`work/c3_cpu_head.log`) | 50 | 427,884 | 0 |
| C3, same | GB300 at 543400873 (362140) | 98 | 2,043,456 | 0 |
| C2: candidate pairs (so the tie sets), tie groups and directions, op outputs, w, d, u | CPU (`work/c2_cpu_adv2.log`) | 36 adversarial + AirfRANS | 160,074 (74,550 on the wall, 52,657 tied, 279,930 tie members) | 0; 1 set raises in both (the defect below) |
| C2, same | GB300 (363374, 363922) | 84: 32 adversarial, DML vehicle 2, DML whole 2, AeroZoo 2 + AirfRANS 2, 20 lug x 2 + an AeroZoo 20k case x 2, Tensile2d 2 | 1,952,096 | 0; float32 zero-area soup raises in both |
| C6: a trained 0.9.0 checkpoint's outputs, eval, float32 and bf16 autocast | GB300 (363375) | AirfRANS r090_af, Tensile2d r090_t2dz, lug r090_lug, AeroZoo azc_full, DrivAerML r090_vol (seed 42 each) | first training batch each | 0: every field bitwise |
| tests, fast forced at every call + comparison plugin | GB300 (363374) | geometry, tree-read, geometry-fast files | 291 calls, 442,077 queries | 0; 174 passed, 1 skipped (env-gated) |
| tests under auto | GB300 (363374) | same files | | 174 passed, 1 skipped |
| whole ISLA test directory | CPU (`work/pytest_all_step3c.log`) | | | 340 passed, 84 skipped (f13eb5ce0: 321 / 52; the new file adds 19 CPU and 32 CUDA-only tests) |

Adversarial sets (probe_c2.py): icosphere, raw torus and box at vertices, edge midpoints and centroids, unposed and posed at
offset 1e3; centroids moved by E/4 and 4E; box mid-planes, diagonals and centre; a plate's mid-plane; the torus core circle;
near-duplicate normals (a flat patch perturbed by 1e-9); a triangle soup; a soup with zero-area cells; the finned cube
(opposite coincident cells); an icosphere with repeated cells; each in float32 and float64.

One non-bitwise observation, not the fast path's: on CPU with 8 threads the float32 vertex gradient differs run to run in the
reference itself (680 of 20,000 AirfRANS rows, reference against reference; 0 at 1 thread; `lhr/probe_threads.py`).

## Reference defect (reported 2026-10-24 01:08 EDT, not fixed: awaits the supervisor)

On a cell with two coincident vertices the reference raises IndexError or returns a farther cell silently: the ball search's
lower bound (torch's cross product of equal vectors is 6e-18, not 0) and `_closest_weights` in the vertex order (b, b, a) both
mishandle it. Degenerate soups: float32 5 of 40 raise, 260 of 35,000 queries silently wrong; float64 33 of 40 raise, 80 of
7,000 wrong; CUDA float32 raises a device-side assert. Real dumps: 0 such cells in 36,136,773 (26 cases). Proposed fix (A + B)
in the report; `lhr/probe_defect{,2,3}.py`. The fast path reproduces the reference's behaviour there exactly.

## Timing (GB300, LHR 363375, one node, two passes with the GPUs rotated; `lhr/final2_table.md`)

q.4's harness: recipe_r090 train.py under `timed_train2.py`, REF-090 settings, bfloat16, eager, medians after 5 warm-up calls or
steps; pass 1 / pass 2. base = code_cmpaf_fix; head = 0570fc6c8 content, under kernel="reference" (forced) and under auto
(the fast path; every call logged as "fast").

| case | geometry call base | f13eb5ce0 | head, reference | head, auto (fast) | step base | f13eb5ce0 | head, reference | head, fast |
|---|---|---|---|---|---|---|---|---|
| AirfRANS 2-D (100k / 1,018) | 13.1 / 10.2 ms | 30.5 / 30.5 | 30.5 / 30.8 | 30.5 / 29.1 | 367 / 361 ms | 463 / 382 | 389 / 385 | 389 / 382 |
| lug (9.8k / 6,488) | 7.5 / 7.3 | 46.3 / 42.7 | 43.5 / 43.7 | 35.4 / 36.1 | 141 / 136 | 183 / 172 | 177 / 177 | 172 / 175 |
| AeroZoo (200k / 0.50M) | 18.2 / 17.1 | 96.2 / 94.2 | 104.4 / 98.2 | 81.6 / 82.8 | 505 / 503 | 587 / 581 | 603 / 583 | 577 / 577 |
| DrivAerML Volume, vehicle (200k / 16.95M) | 97.5 / 98.5 | 373.6 / 372.0 | 366.0 / 360.6 | 296.0 / 295.4 | 570 / 577 | 851 / 858 | 862 / 843 | 786 / 776 |
| DrivAerML Volume, whole boundary (200k / 16.96M) | 2,423 / 2,441 | 2,778 / 2,758 | 421.7 / 423.9 | 351.5 / 349.0 | 2,936 / 2,935 | 3,270 / 3,247 | 928 / 920 | 847 / 839 |

Step against the base (mean of passes): AirfRANS f13 1.16x (pass spread 1.04-1.27; head 1.06x either kernel), lug 1.28x ->
head fast 1.25x, AeroZoo 1.16x -> 1.15x, DrivAerML vehicle 1.49x -> 1.36x, whole boundary 1.11x -> 0.29x (head reference
0.31x). Training-process peak memory is the same for every version (34.4 / 8.1 / 126.3 / 128.4 / 128.4 GiB: the model's).
The geometry call's own peak on DrivAerML (prof_phase, 363374/362140): f13eb5ce0 18.3 GiB, head reference 16.8, head fast 17.7.
The earlier run at 543400873 (362819) agrees within 3 percent.

Per phase on DrivAerML vehicle (prof_phase, 362140, same GPU): reference 332 ms against fast 259 ms; the fast path removes
25 ms of weld and canonical cells, 16 of dedupe, 10 of seed, 9 of tree, 18 of moments; the ball and winding traversals (150
ms) and the exact phase (35 ms) are the reference's.

## What is not done

- The exact phase in Triton (DESIGN d, step 3): not built; it is the shared torch code (35-45 ms at DrivAerML size).
- The optional 2-D kernel: not built; on 2-D cases the fast path differs only in the weld and canonical cells.
- The joined ball and winding traversal on packed records (s3a, 362005) was built and dropped: on CUDA its `w` was not
  bitwise (fused multiply-adds placed differently by the compiler), with no speed gain.
