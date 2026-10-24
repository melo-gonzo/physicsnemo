Status: DONE. The fast path is built and bitwise equal to the reference everywhere. One decision is still yours: the reference defect fix, reported 01:08 EDT 10-24 and not applied.

**Design as approved** (DESIGN.md md5 4e38af75; 19:48 EDT, with your change to decision 3).
- Warp runs the traversals; Triton (through importlib, HAS_TRITON) runs the weld scan and the canonical cells. Each kernel was kept only because it is bitwise.
- `geometry_kernel="auto"` picks the fast path when the points are on CUDA and Triton imports, and logs the choice once. A forced value overrides it.
- The seed fix went into the reference first, as its own commit.
- Not built: the Triton exact phase, which stays the shared torch code, and the optional 2-D kernel.
- Dropped: a packed joined ball-and-winding traversal (job 362005). Its `w` was not bitwise on CUDA and it gave no speedup.

**Commits** on `psharpe/isla-geo-fast` (pushed, not merged). Worktree: `/home/psharpe/gh/physicsnemo-isla-geo-fast`, 974 MB.
- `fb263fa29` (merge-ready alone): the reference's seed is a nearest-cell search on its own cell tree. Before, it was a per-call Warp mesh BVH, which degenerates on the whole-boundary box.
- `b30ff5c33`: the fast op `physicsnemo::isla_closest_candidates_fast` and the `geometry_kernel` selection.
- `543400873`: the Triton kernels, and the gate on Triton.
- `0570fc6c8`: the fast path builds the lower-bound planes exactly as the reference does.
- ISLA version and `_refuse_earlier_state_dict` are unchanged.

**Correctness at the head** (counts and logs in RESULTS.md). Every check was bitwise, so the stated bound is 0 ulp.
- C1, the head's reference against `f13eb5ce0`, comparing all six op outputs, `w`, `d`, `u` and their gradients: 0 differing queries of 427,884 on CPU and 0 of 2,043,456 on GB300.
- C2/C3, fast against reference: candidate pairs (so the tie sets), tie groups and directions, op outputs, `w`, `d`, `u` and gradients.
  - 0 differing queries of 1,952,096 on GB300, over 84 sets: DrivAerML Volume vehicle and whole boundary, AeroZoo, AirfRANS, Tensile2d, the 20 lug cases, and 32 adversarial sets.
  - 0 of 160,074 in 36 sets on CPU.
  - The adversarial sets cover on-wall points in float32 and float64, posed and offset copies, E/4 and 4E offsets, medial axes, near-duplicate normals, soups, a fold, and repeated cells.
- C6, one trained 0.9.0 checkpoint per dataset (AirfRANS, Tensile2d, lug, AeroZoo, DrivAerML): every output field is bitwise equal, in float32 and in bf16 autocast.
- Tests:
  - Whole ISLA directory on CPU: 340 passed, 84 skipped. f13eb5ce0 gives 321 / 52; the new test file adds 19 CPU tests and 32 CUDA-only tests.
  - GB300, geometry, tree-read and geometry-fast files: 174 passed, 1 skipped, both forced and under auto. A plugin compared both ops at all 291 calls (442,077 queries): 0 differences.
- CPU and fast: every fast-path piece is also bitwise on CPU (the C3 and C2 counts above). On CPU the Triton pieces fall back to the reference's torch code; the preorder tree, packed winding and dedupe still help. Measured on LHR's CPU with 8 threads (job 362859), reference to fast: AeroZoo at 20k queries 8.98 s to 4.28 s, the lug 296 to 284 ms, AirfRANS 392 to 388 ms.

**Timing** (GB300, LHR 363375, one node, GPUs rotated, two passes). Geometry call in ms: base / f13eb5ce0 / head fast.
- DrivAerML Volume, vehicle: 98 / 373 / 296. Step against the base: 1.49x at f13eb5ce0, 1.36x at the head.
- DrivAerML Volume, whole boundary: 2,432 / 2,768 / 350. The step falls from 2.94 s at the base to 0.84 s (0.29x the base). The head's reference path alone gets 0.92 s, from the seed fix.
- AeroZoo: 18 / 95 / 82. Step 1.16x to 1.15x.
- lug: 7.4 / 44.5 / 35.8. Step 1.28x to 1.25x.
- AirfRANS: 11.7 / 30.5 / 29.8. Step 1.06x at the head with either kernel; the 2-D path is unchanged.
- Peak memory of the training process is the same for every version. The geometry call's own peak on DrivAerML: 18.3 GiB at f13eb5ce0, 17.7 GiB fast.
- Against DESIGN's estimates the fast path fell short on the vehicle case (296 against about 210 ms), because the Triton exact phase was not built.

**What it means.** The fast path cuts the bundle's added step cost on the vehicle-only DrivAerML case by about a quarter (from +49% to +36%), and by less elsewhere. The whole-boundary result is larger and comes from the reference itself: every ISLA version paid about 2 s per call there, and the seed fix in `fb263fa29` removes it. That makes the canonical whole-boundary DrivAerML step about 3.2x faster than at 0.9.0. I am confident in the correctness results, since every check is exact. The timings agree within 3% with an earlier run (job 362819).

**Reference defect, reported, not fixed.** A cell with two coincident vertices makes the reference raise IndexError (a device-side assert on CUDA) or silently return a farther cell.
- Degenerate soups: 260 of 35,000 float32 queries and 80 of 7,000 float64 queries were wrong without raising.
- Real dumps: 0 such cells in 36.1M (26 cases).
- Proposed fix: keep such cells in the ball search, and compute their closest point on the longest edge.
- The fast path matches the reference's behaviour there exactly.

**Rulings relied on.** The whole boundary in every table (Peter, 17:08 EDT 10-23); the approved design; your 00:32 instruction not to change the reference.

**For project memory.** Whole-boundary DrivAerML costs about +2 s per geometry call at every version before `fb263fa29`. On CPU the reference's float32 vertex gradient varies between runs at more than one thread (bitwise only at 1 thread).

**Left for the main session.**
- Decide on the reference defect fix.
- Merge `fb263fa29`, and later the rest.
- Change the book's documented cost limitation: draft numbers are in RESULTS.md.
- Optional: the Triton exact phase (about 35 ms at DrivAerML size) and the 2-D kernel.

**GPU and disk.** 7.76 GPU-h of the 30 cap, in 16 jobs (359447 to 363922), all done; nothing is running. The registry row has been removed. LHR: `$T/kernel/geofast/` (499 MB, outputs and dumps) and snapshots `$T/code_geofast{1,1b,1c,2,3a,3b,3c}` (57 MB each; `3c` is the head). Local: scratch folder 13 MB.
