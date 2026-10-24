# TOKEN-DRAW: the boundary-token draw as a function of the input mesh (GEO-ROBUST phase 2b; 2026-10-22)

Written 15:51-16:10 EDT, 2026-10-22. Worktree /home/psharpe/gh/physicsnemo-isla-token-draw, branch
psharpe/isla-token-draw (from psharpe/isla-medial-tie d3d9da49d). Supervisor's decisions on MEDIAL-TIE DESIGN i.2
as given in the task: BFS/Cuthill-McKee and hashed priority sampling rejected; change (1) required; change (2)
measured and kept only if it cuts pose changes by >= 10x on non-degenerate cases at < 2% of AeroZoo's 0.51 s step.
*Supervisor's note (2026-10-23 03:40 EDT), the reasons given with that decision at 14:48 EDT 10-22 and recorded then
in the program's architecture agenda:* the breadth-first and Cuthill-McKee orders depend on the cells' numbering and
need a per-mesh cache, which the boundary-only input ruling excludes; hashed priority sampling depends on the
numbering and is not stratified over the surface.

## 1. Changes

**(1) Committed, 2042c50cf.** `sampling.draw_inputs(mesh)` computes the measures and centroids the draw reads in
float64 from `mesh.points` as given: the stored `_effective_measure` cast, else `compute_cell_areas` of the float64
vertices, and the float64 vertex means. It never reads the mesh's cached geometry, which is in the mesh's dtype.
`ISLA.forward` passes these to `sample_boundary_cells` and casts the Horvitz-Thompson weights to the tokens' measure
dtype at the end. The body of `sample_boundary_cells` is unchanged; it was already float64 from the probabilities on.

**(2) Implemented, not committed: awaiting the supervisor's reading of the rule (section 4).** In
`sample_boundary_cells`, `frame = tree_read.canonical_frame(cat(centroids), cat(measures))` over every cell of
every condition, in float64, and Morton codes of `x @ frame`. The rule is the same everywhere: no flag, no fallback.
Start, stratification and water filling are unchanged.

Tests, all in test/experimental/models/isla/test_isla_sampling.py:
- `test_the_draw_is_the_same_at_every_model_precision`: a float32 and a float64 ISLA on the same float32 meshes
  (cells around (300, -200, 100)) keep bitwise the same cells, and the tokens' measures agree after the cast.
  On d3d9da49d it fails at `torch.equal(kept32, kept64)`.
- `test_the_draw_is_independent_of_the_numbering`: renumbering the vertices and cells of float32 meshes keeps the
  same physical cells with bitwise the same weights. On d3d9da49d it fails only at the import of `draw_inputs`: the
  base draw is already numbering-invariant on these meshes (no Morton ties), so this test pins the property.
- `test_the_draw_is_the_same_in_every_pose` (with (2)): a proper and an improper rotation plus a translation in
  float64 keep the same cells, with weights equal to 1e-12. It fails without (2).

## 2. Measurement (drawn cells that change, of k = 200,000; start 0.5)

The protocol is scripts/drawdiag.py (md5 2e2952ed) on the LHR snapshot code_r090 / recipe_r090. Base
`sample_boundary_cells` is byte-identical to code_r090's (md5 435c4cdf), and the variants differ only in their
inputs. Inputs are as the float32 scorer delivers them: the stored float32 points, and `_effective_measure` set to the
float32 cell areas (ComposeQuadratureMeasure), recomputed after posing.
- **Pose**: FixedRandomPose's rotation (key = case number, salt 0) and a translation of 1 RMS radius, applied in
  float32 through `Mesh.transform` and `Mesh.translate`. I applied them to the delivered meshes, not as the first
  transform. For DrivAerML the scorer's CenterMesh then removes the translation, which this protocol does not do, so
  it is harsher there.
- **Precision**: float32 against float64 compute on the same float32 inputs, cast and with fresh caches.
- **Renumbering**: random permutations of the vertices and the cells.

Variants: base = d3d9da49d; v1 = (1); v2 = (1) + (2). Degenerate = frame eigenvalue gaps below 3e-3 of the largest.

| case (family) | cells | pose base / v1 / v2 | v2 cut | precision base / v1 / v2 | renumbering base / v1 / v2 |
|---|---|---|---|---|---|
| AZ 003 (duct, non-box) | 720k | 120,411 / 120,387 / 224 | 538x | 2,594 / 0 / 0 | 0 / 0 / 0 |
| AZ 022 (streamlined, box) | 383k | 51,875 / 51,879 / 32 | 1621x | 664 / 0 / 0 | 0 / 0 / 0 |
| AZ 026 (zero_to_cad, box) | 832k | 91,562 / 91,560 / 200 | 458x | 40 / 0 / 0 | 14 / 14 / 12 |
| AZ 050 | 291k | 38,678 / 38,688 / 51 | 758x | 165 / 0 / 0 | 0 / 0 / 0 |
| AZ 088 | 685k | 64,467 / 64,472 / 205 | 314x | 3,330 / 0 / 0 | 0 / 0 / 0 |
| AZ 092 (zero_to_cad) | 345k | 41,693 / 41,693 / 53 | 787x | 51 / 0 / 0 | 0 / 0 / 1 |
| AZ 100 (duct) | 899k | 137,095 / 137,066 / 1,670 | 82x | 406 / 0 / 0 | 0 / 0 / 0 |
| AZ 035 (blobs, crossflow) | 450k | 53,332 / 53,323 / 106 | 503x | 12 / 0 / 0 | 0 / 0 / 0 |
| AZ 044 | 460k | 67,063 / 67,060 / 225 | 298x | 10 / 0 / 0 | 0 / 0 / 0 |
| AZ 047 | 379k | 57,862 / 57,844 / 236 | 245x | 154 / 0 / 0 | 0 / 0 / 0 |
| AZ 087 (cars) | 383k | 47,312 / 47,324 / 1,277 | 37x | 33 / 0 / 0 | 0 / 0 / 0 |
| AZ 157 (airframes) | 391k | 43,065 / 43,059 / 123 | 350x | 46 / 0 / 0 | 0 / 0 / 0 |
| AZ 020 (blender_bluff, box) | 838k | 93,476 / 93,469 / 223 | 419x | 405 / 0 / 0 | 0 / 0 / 0 |
| DrivAerML run_1 | 17.7M | 185,914 / 185,922 / 25,759 | **7.2x** | 120,618 / 0 / 0 | 0 / 0 / 0 |
| DrivAerML run_2 | 16.4M | 186,937 / 186,980 / 2,578 | 73x | 28,129 / 0 / 0 | 0 / 0 / 0 |
| AZ 041 (opposed_sphere), degenerate | 1.39M | 163,397 / 163,459 / 165,927 | 1.0x | 21,352 / 0 / 0 | 0 / 0 / 0 |
| AZ 131 (opposed_sphere), degenerate | 1.52M | 158,424 / 158,470 / 157,402 | 1.0x | 1,009 / 0 / 0 | 0 / 0 / 0 |
| AZ 186 (opposed_hemisphere), degenerate | 1.52M | 156,990 / 157,000 / 153,417 | 1.0x | 2,893 / 0 / 0 | 0 / 0 / 0 |

- **Precision.** With v1 and v2 the change is 0 on every case, and the weights are bitwise equal after the final
  cast. Base changes 10-21,352 cells on AeroZoo, consistent with f32floor's 5-5,794, and 28k-121k on DrivAerML.
- **Pose.** v1 is no better than base, as expected: the grid still follows the dataset's axes. On the 15
  non-degenerate cases v2 cuts the changes by 7.2x to 1621x, with a median of about 420x and 39x on the pooled
  counts. On the degenerate frames, where both eigenvalue gaps are 3e-4 to 3e-3 or one gap is 1e-4 (the opposed
  layouts, whose inlet and outlet are mirror images), v2 neither helps nor hurts. The frame's own pose error there is
  1.4e-5 to 3.2e-5, against 1.5e-8 to 2.4e-7 elsewhere.
- **What v2 leaves** (out/decomp_337688_*.json) is float32 rounding of the posed coordinates moving centroids across
  Morton grid lines, plus, less, the rounding of the posed float32 areas. For example, on run_1 with only the posed
  centroids changed (native measures) 24,606 cells change, and with only the posed measures 2,059. On run_1 rotation
  alone changes 20,642 and translation alone 6,614; on run_2, 6,370 and 23,249; on AZ 087, 18 and 60 (1,277 with
  both). No coordinate-built order can remove this, because posing in float32 changes the input coordinates.
- **Renumbering.** Base, v1 and v2 all change 0 cells, except where the Morton codes tie (AZ 026: 306-318 cells
  share a code, 12-14 drawn cells change in every variant, because ties are broken by index), and one cell on AZ 092
  under v2: the frame's float64 sums run in the cells' order, so a permutation moves the frame by roundoff. Both
  are kept as residual risks; neither is new except that one cell.

## 3. Cost (GB300, LHR 337688, nvl72d027-T06; median of 30, eager, inputs and draw, after warm-up)

| case | cells | base | v1 | v2 | v2 - v1 | share of a 0.51 s step |
|---|---|---|---|---|---|---|
| AZ 041 | 1.39M | 5.46 ms | 5.44 ms | 11.55 ms | 6.1 ms | 1.2 % |
| AZ 100 | 0.90M | 5.79 ms | 5.76 ms | 9.91 ms | 4.1 ms | 0.8 % |
| AZ 026 | 0.83M | 5.70 ms | 5.69 ms | 9.86 ms | 4.2 ms | 0.8 % |
| DrivAerML run_1 | 17.7M | 10.1 ms | 12.6 ms | 84.7 ms | 72 ms | (not an AeroZoo step) |

(1) costs nothing measurable at 1M cells and 2.5 ms at 17.7M. (2) costs 0.8-1.2 % of the AeroZoo step, under the 2 %
cap. It costs 72 ms at DrivAerML's 17.7M cells (the float64 covariance, third moments and matmul over all cells).

### 3a. TOKEN-COST: v2 without the float64 GEMMs (3d9acfa63; written 20:04 EDT, 2026-10-22)

**Where the 72 ms went** (profile LHR 339555, run_1 on GB300). Three float64 GEMMs took 69.5 of the draw's 84 ms:
the covariance `(w x).t() @ x`, `(3, N) @ (N, 3)`, 34.1 ms; the sign rule's projection `x @ axes` 17.9 ms; and the
Morton coordinates `x @ frame` 17.9 ms. GB300 has few float64 units, and cuBLAS runs these skinny shapes on its
float64 tensor-core kernels (cutlass d884gemm) far below memory bandwidth. The same quantities as column products
and sums take 1.3, 1.0 and 0.9 ms. Everything else was already small: weights 0.40 ms, centering 0.51 ms, the GPU
`eigh` 0.19 ms, the moments 1.0 ms. There was one host sync (`eigh`'s error check), no CPU `eigh`, and no repeated
gather or argsort.

**Change.** In `canonical_frame` the covariance is `stack([(w x * x[:, j]).sum(0) for j])` and the projection is
the new `tree_read.frame_coordinates(x, axes)`, which computes `sum_k x[:, k] * axes[k]`. The draw's Morton
coordinates use `frame_coordinates(x, frame)`. There is one code path on every device (supervisor's decision B,
19:50 EDT). The sign rule and the outputs are otherwise unchanged. No caches across calls. Diff: +20 / -4 source
lines in tree_read.py and sampling.py.

**Timing** (LHR 341593, nvl72d062-T09; median of 30 after two warm-up calls, eager; draw_inputs plus
sample_boundary_cells; both draws timed in the same process on the same GPU, with c2db0445e's frame and
`x @ frame` patched in for the "before" column, verbatim):

| case | cells | v1 (337688) | before: c2db0445e | after: 3d9acfa63 | saved | after - v1 | canonical_frame before / after | Morton projection before / after |
|---|---|---|---|---|---|---|---|---|
| AZ 100 | 0.90M | 5.76 ms | 8.68 ms | 6.93 ms | 1.7 ms | 1.2 ms | 3.16 / 0.86 ms | 0.93 / 0.09 ms |
| AZ 041 | 1.39M | 5.44 ms | 9.90 ms | 6.82 ms | 3.1 ms | 1.4 ms | 4.60 / 0.87 ms | 1.42 / 0.10 ms |
| AZ 026 | 0.83M | 5.69 ms | 7.99 ms | 6.30 ms | 1.7 ms | 0.6 ms | 2.92 / 0.77 ms | 0.86 / 0.08 ms |
| DrivAerML run_1 | 17.7M | 12.6 ms | 84.6 ms | 17.7 ms | 66.9 ms | 5.1 ms | 53.9 / 4.19 ms | 17.8 / 0.87 ms |

"after - v1" compares against v1 from job 337688, on a different node, so it is approximate. At 17.7M cells the
frame now costs about 5 ms, 0.9-1.2 % of the DrivAerML Volume step (422-552 ms) instead of 13-17 %. At about 1M cells
it costs 0.6-1.4 ms, 0.1-0.3 % of AeroZoo's 0.51 s step.

**What remains, at 17.7M** (column timings from 339555): weights 0.40 ms, centering 0.51 ms, covariance 1.31 ms,
`eigh` 0.19 ms plus one host sync, the sign projection 1.0 ms, the moments 1.0 ms, and the Morton projection 0.87 ms.
The essential work is one float64 pass for the weighted mean, one for the weighted second moments, one for the
projection and its third moments, and the Morton projection. These passes are memory-bound in principle
(424 MB per read of a float64 (N, 3) array), but eager mode builds a temporary per op. A fused kernel, or Inductor
under the model's compile, could bring the roughly 5 ms down to about 1 ms. That is not attempted here: it needs a
kernel or a compile region, and the remaining cost is about 1 % of a step. `eigh`'s host sync is inherent to
`torch.linalg.eigh` (its info check); a CPU `eigh` would need a copy, which syncs as well. It was already present in
c2db0445e, so the sampling module docstring's "without a host synchronization" has been inaccurate since then.
The supervisor's item 5 (one shared projection, with negated columns for the signs) does not apply. The sign rule
projects the centered coordinates of all cells, while the Morton step projects each condition's uncentered centroids.
Sharing one projection would change the Morton inputs by rounding and save at most 0.9 ms.

**Results** (scratch out/cost_*_341593_*.json, out/cost_local_cpu.json). canonical_frame's float64 output moved at
ulp level, because the column sums round differently from the CPU's dgemm (no column form reproduces dgemm's
summation order) and from cuBLAS's. Per case, on CPU, the old (GEMM) frame against the new one:

- **The 15 non-degenerate real meshes:** max axis difference 3.3e-16 to 6.1e-15. Every axis keeps its order and its
  sign. The smallest relative eigengap is 0.023 (AZ 044) to 0.33.
- **The 3 degenerate frames:** AZ 041 4.1e-13 (gap 3.1e-4), AZ 131 5.8e-13 (gap 1.3e-3), and AZ 186 5.1e-12
  (gap 1.1e-4). Each is of order |delta cov| / gap. Order and signs are kept.
- **The tests' cases** (the meshes of the three draw tests in test_isla_sampling.py, and clouds of 400 (d=2), 500,
  3000 and 100k cells): 1e-17 to 7.8e-16, with order and signs kept.
- **Sign margins** (|m3| / sum w |p|^3 on the axes the third moment signs): every case's smallest is 5.9e-7 or more.
  The smallest is AZ 100's shortest axis (|skew| 6.8e-7, 674x the tie threshold), and the next is AZ 041's at
  9.0e-6. No case is below 1e-12, and no axis falls back to the lab rule on any case.
- **Draw (start 0.5, k = 200,000):** bitwise the same cells and weights on all 18 real meshes (the 3 degenerate ones
  included: 0 changed cells) and on every test case, on CPU.
- **GB300**, old against new GPU draw: bitwise the same cells and weights on AZ 100, AZ 041, AZ 026 and run_1.
  The frame differs by 1.1e-16 to 2.7e-15, except 8.4e-14 on the degenerate AZ 041.
- Local tests unchanged and passing: test_isla_sampling.py, test_isla_tree_read.py and test_isla.py, with
  104 passed and 6 skipped (CPU, serial).

ISLA-DTT's tree read gets the new `canonical_frame` too (its frame moves by the same ulps). Its own `points @ frame`
and `normals @ frame` in `BoundaryForest.build` and `interaction_lists` are still GEMMs; I left them alone (out of
scope). They are the same skinny float64 shape, so `frame_coordinates` would cut those too.

## 4. Decision on (2)

The cost criterion passes. The 10x criterion passes on all 13 non-degenerate AeroZoo cases (37x-1621x), on
DrivAerML run_2 (73x), and pooled (39x), but fails on DrivAerML run_1 (7.2x). Whether "at least 10x on the
non-degenerate cases" means every case or the pool is the supervisor's reading, so I asked (contact_supervisor
need_decision, 15:51 EDT). My recommendation is to implement (2). Every AeroZoo case clears the bar by a wide margin,
and run_1's residual is coordinate rounding that the scorer's float32 posing creates and no Morton grid removes.
(2) is in the worktree, uncommitted, with its test.

Later: (2) was committed as c2db0445e. With 3d9acfa63 (section 3a) it costs 5.1 ms at DrivAerML's 17.7M cells, against
72 ms before, which is 0.9-1.2 % of that step. At about 1M cells it costs 0.6-1.4 ms. The draws are unchanged
bitwise on every case measured.

*Supervisor's decision (2026-10-22 16:04 EDT; recorded here 2026-10-23 04:02 EDT):* the bar is read on the pool (39x;
14 of the 15 non-degenerate cases above 10x, DrivAerML run_1 at 7.2x from the float32 posing that no Morton grid
removes), and (2) is approved on principle, on the condition that its cost at DrivAerML's size come down. TOKEN-COST
met that condition (3a).

## 5. Records

- LHR jobs: cpu-short 337521 (smoke, AZ 035), 337563 and 337564 (AZ), 337565 (DrivAerML run_1, run_2); short
  337688 (1 node x 4 GB300, 1 min 51 s: timing on 4 GPUs plus the CPU decomposition). About 0.12 GPU-h allocated.
- LHR $T/kernel/tokendraw/: drawdiag.py, drawdiag.sbatch, drawdiag2.sbatch, drawtime.sbatch, drawtime2.sbatch,
  out/ (under 1 MB). Local copies: scripts/ and out/ here.
- Registry row `^isla\.tokendraw\.` added 15:04 EDT; removed when the lane ends.
- TOKEN-COST (19:45-20:10 EDT): short 341593 (isla.tokendraw.cost, 1 node x 4 GB300, 1 min 56 s: GPU timing and
  comparison on 4 GPUs, then the CPU comparison on the 18 real meshes on the same node), about 0.13 GPU-h allocated.
  It read profile 339555 (TOKEN-DRAW). Snapshot $T/kernel/tokendraw/code_tc_c2db0445e_wt1 (read-only):
  code_td_c2db0445e plus the working-tree tree_read.py (md5 8faa68bd) and sampling.py (md5 7d6b1103). Committed
  3d9acfa63 differs from it only in the frame_coordinates docstring's timings. Scripts drawcost.py (md5 f9d79b18)
  and drawcost.sbatch are in $T/kernel/tokendraw/ and in scripts/. Outputs are $T/kernel/tokendraw/out/cost*341593*,
  copied to out/ here. Registry row set to expected_end 2026-10-23 at 19:58 and removed at 20:03 EDT (backups
  ACTIVE_EXPERIMENTS.tsv.bak2 and .bak3 here).
