# ISLA-DTT report

## Status (2026-10-06 night PDT)

Where the program stands after one day: the tree read is implemented, exact at theta = 0 and exactly query
independent at every theta, and verified at unit and checkpoint level; it is not yet usable, for two measured
reasons. (1) **Accuracy**: with the dense-trained kernel, the cluster monopole is far too crude at every theta
(E2 below). (2) **Cost**: at DrivAerML Volume scale the per-point lists are 550 to 1,400 sources and the first
implementation spends 20 to 35 times the read-only step on them. Both are being worked: a decomposition probe of
the monopole error is running, and a profile of the step is queued.

### Done

- Brief and preregistration (`BRIEF.md`, `PREREG.md`; E4's prediction amended on the change of form).
- Tree read `physicsnemo/experimental/models/isla/tree_read.py`, `ISLA(kernel_read_theta=...)`, on the pi lane's
  dense global read; canonical `psharpe/isla` merged (afdb09a55). Three forms in one day:
  1. GLOBE's dual tree (a80cfc0f4): every unit contract passed; at DrivAerML scale 1.6e9 pairs per block at
     theta = 0.5 (out of memory on a 277 GiB GB300). A leaf of eight points in a sparse region of the volume spans a
     box of order a metre and is never well separated from the boundary leaves near it under (D_T + D_S) / r.
  2. Per-point walk of GLOBE's Morton tree (a59f3a44e): 3.4e8 pairs per block at theta = 1 (1,700 sources per
     point). GLOBE's nodes are count-midpoint ranges of the Morton order, so they straddle Morton boundaries and
     their boxes are loose for a surface. Also found here: the `pair_dot` Triton kernel's 32-bit pair offsets wrap
     above 2.7e8 pairs (NaN losses); fixed with 64-bit offsets.
  3. Per-point walk of `BoundaryTree` (07762f7e7, current): balanced, each cluster split at its median along its
     own principal axis (sign fixed by the third moment), bounding spheres, measure-weighted virtual cells; the
     admission is sphere diameter < theta x gap. The tree is frame-free up to ties, so at finite theta a rotated or
     rescaled case gives the same clusters (tested). 21 CPU tests in float64 plus 2 CUDA tests pass; the ISLA suite
     passes under the merged tree (283 passed, job 106325 on the second snapshot; the third snapshot's run is in
     flight, job 106421).
- E1 (exactness) holds at unit level (1e-10) and at checkpoint level: on the four G checkpoints at 100,000 points
  in float32, theta = 0 reproduces the dense read to 1e-6 relative L2 per field and the protocol error to four
  digits (job 106327, Morton tree; the tree does not matter at theta = 0).

### E2 verdict: the dense-trained kernel does not survive the monopole (fails at every theta)

Four G checkpoints, 200 held-out airfoils, 100,000 points, float32, Morton tree (job 106327; the BoundaryTree
repeat is job 106423). Ratio of the tree read's protocol error to the dense read's, pressure / velocity / eddy
viscosity, and the mean relative L2 deviation of the predictions:

| theta | seed 42 | seed 43 | seed 44 | seed 45 | deviation p / v / nut (seed 42) | pairs per block |
|---|---|---|---|---|---|---|
| 0 | 1.000 / 1.000 / 1.000 | same | same | same | 1e-6 / 5e-7 / 9e-7 | 1.0e8 (dense) |
| 0.25 | 2.81 / 1.85 / 7.45 | 2.46 / 1.58 / 3.87 | 2.23 / 1.66 / 5.22 | 1.93 / 1.52 / 6.10 | 0.056 / 0.019 / 0.16 | 1.7e7 |
| 0.5 | 5.86 / 2.13 / 19.2 | 4.50 / 2.11 / 14.3 | 5.10 / 2.17 / 12.3 | 3.17 / 1.86 / 11.0 | 0.12 / 0.024 / 0.42 | 1.1e7 |
| 1.0 | 8.22 / 2.40 / 27.7 | 6.44 / 3.01 / 22.0 | 7.70 / 3.19 / 18.0 | 4.29 / 2.49 / 14.9 | 0.17 / 0.028 / 0.60 | 7.3e6 |

The preregistered bar (ratio within [0.99, 1.01] at theta <= 0.5) fails by a factor of two to twenty; eddy
viscosity, whose points sit far from the airfoil in the wake and read the whole boundary through coarse clusters,
suffers most. The deviation does fall with theta (0.17 to 0.056 on pressure from theta 1 to 0.25) but far too
slowly: first-order convergence from a 17% start. Mechanism candidates, in the preregistered order of suspicion:
the mean normal of a cluster that spans both faces of a thin airfoil is the renormalization of a near-zero vector
(a random direction), and the kernel reads the normal through three cosines; the mean key of a cluster is a
first-order surrogate of a softmax over its cells; the centroid position smears log r. The decomposition probe
(`scoring/dtt_smear.py`, job 106424: the dense pair list grouped by the admitted clusters with one input at a time
replaced by its cluster mean) measures each.

### Cost at DrivAerML Volume scale (one GB300, 512 / 512, compiled blocks, bf16, 200,000 cells and points)

| form | sources per point | pairs per block | step | forward | peak memory |
|---|---|---|---|---|---|
| read-only, no read (floor) | - | - | 0.48 s | 0.18 s | 126 GiB |
| Morton tree, theta 1 | 1,700 | 3.4e8 | 18.8 s | 5.6 s | 233 GiB |
| BoundaryTree, theta 1 | 550 | 1.1e8 | 10.3 s | 3.3 s | 160 GiB |
| BoundaryTree, theta 0.5 | 1,400 | 2.8e8 | 16.4 s | 5.0 s | 213 GiB |
| BoundaryTree, theta 0.25 | - | - | out of memory | | |

(jobs 106296, 106326, 106422.) The tight tree cut the pair count three to five times. The remaining cost is per
pair: 1.1e8 pairs through 12 block reads in 10 s is about 8 ns per pair and block, against a bandwidth bound of
roughly 0.1 ns for the bytes each pair moves, so the eager per-pair tensors and the Triton segment kernels, not the
pair count, set the step. A profile is queued (`scoring/dtt_profile.sbatch`).

### In flight

- Job 106421: ISLA suite on the third snapshot. Job 106423: E2 repeated with the BoundaryTree. Job 106424: the
  decomposition probe (seeds 42 and 43 at theta 0.5, seed 42 at 0.25 and 1.0, 12 cases each).
- Lanes `dtt_vol_p_seed42/43` (E3's P arms, read-only without a read, 512 / 512, lr 3.75e-4, 200,000, 500
  epochs): training at 0.474 s per step and 130 GB per rank, about 51 s per epoch, so roughly 7 hours. The TG
  lanes wait for a usable form.

### Decision pending

If the decomposition shows the normal smear dominates, the next form clusters by orientation as well as position
(split a cluster along the principal axis of its normals while its normal spread exceeds theta, then by position;
admit only clusters whose normal spread is below theta), which keeps a thin body's two faces apart at every scale.
If the key or value smear dominates, the dense read cannot be approximated at usable cost by any geometric
clustering, and the honest alternative is to train with the tree at a fixed theta (the clusters become part of the
model, as in multipole attention) and judge it on E3 alone. Either way the per-pair cost must drop by an order of
magnitude first.

## Implementation notes

`BoundaryTree.build` is level synchronous: per level the nodes' means and covariances by `index_add_` in float64,
`torch.linalg.eigh` for the principal axis, the sign from the third moment, two stable sorts to reorder the cells
within their nodes, a median split; then measure-weighted aggregates of every node by prefix sums, and the
bounding radius level by level. `point_tree_pairs` is a level-synchronous walk: a frontier of (point, node) pairs
from every point's root entry, three boolean splits per level (admitted, opened leaf, descend), the leaves
expanded with `_ragged_arange`, the output sorted by point. Per block the node's key and value are the
measure-weighted means of its cells' through float64 prefix sums over the tree order (`node_means`),
differentiable in the cells' states. The per-block read reuses the truncated read's sparse machinery (`PairPlan`,
`pair_scores`, `pair_weighted_sum`, Triton on CUDA) with a segment max for the softmax shift and no null; the
dense read's parameters are shared, so the two forms are one model.

## Environment notes

Local WSL cannot load CUDA (`/usr/lib/wsl/lib/libcuda.so.1` is a corrupt 188 KB file; ld.so aborts); CPU float64
tests run in `~/.venvs/isla-cpu` with an empty stub `libcuda.so.1` on `LD_LIBRARY_PATH` so warp falls back to CPU.
Cluster venv `.venv-recipe-td14` (tensordict 0.14.2, torch 2.12.0+cu130) is the venv of every post-merge snapshot;
its `bin/` entry points were copied from `.venv-recipe` and executed the old interpreter until ISLA Main rewrote
them on 2026-10-06 (the first two reference lanes died on that; the launcher now calls
`python -m torch.distributed.run`).
