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

### Decomposition of the monopole error (job 106424, position-only BoundaryTree)

The dense pair list grouped by the admitted clusters, with one input of the pairs inside admitted clusters
replaced by its cluster mean; mean relative L2 deviation from the dense read over 12 held-out airfoils, pressure /
velocity / eddy viscosity. "none" reproduces the dense read and "all" the tree read to 1e-6 (harness checks).

| replaced by the cluster mean | seed 42, theta 0.25 | seed 42, theta 0.5 | seed 43, theta 0.5 | seed 42, theta 1.0 |
|---|---|---|---|---|
| nothing (dense) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| position (centroid) | 0.010 / 0.005 / 0.048 | 0.026 / 0.011 / 0.14 | 0.013 / 0.011 / 0.19 | 0.040 / 0.018 / 0.24 |
| normal | 0.20 / 0.10 / 0.88 | 0.24 / 0.12 / 0.90 | 0.23 / 0.10 / 0.90 | 0.32 / 0.16 / 0.85 |
| key (per block) | 0.16 / 0.035 / 0.15 | 0.24 / 0.042 / 0.26 | 0.23 / 0.059 / 0.18 | 0.30 / 0.048 / 0.35 |
| value (per block) | 0.22 / 0.070 / 0.50 | 0.30 / 0.096 / 0.68 | 0.30 / 0.094 / 0.64 | 0.37 / 0.11 / 0.80 |
| all four (the tree read) | 0.24 / 0.095 / 0.81 | 0.31 / 0.12 / 0.92 | 0.40 / 0.11 / 1.05 | 0.37 / 0.14 / 0.98 |

The position smear is small (1% to 4% on pressure). The normal, key and value smears are each of the order of the
whole deviation, and they do not add: any one of them alone breaks the read. The normal result was expected (the
mean normal of a two-faced cluster is a random direction). The key and value results mean the dense-trained kernel
weights vary across an admitted cluster by far more than the first-order monopole can carry, and the cells' states
(keys and values) vary with them, so the mean of the products is not the product of the means. These clusters are
two-faced, so one-sided clusters may fare better; the probe is repeated with the orientation-aware tree (job
106466). Whatever it shows, the dense-trained kernel is not approximable at a usable theta by first-order
aggregates alone; see the decision below.

### Two more forms of the tree (2026-10-06 evening)

- Position-only BoundaryTree, E2 repeated (job 106423): worse than the Morton tree (pressure ratio 6.8 to 16 at
  every theta, 5.5e6 pairs at theta 0.5): its clusters are chordwise strips of a thin airfoil, two-faced at every
  level, and it admits fewer of them.
- Orientation-aware BoundaryTree with the largest chord as the normal spread (04b598294): on DrivAerML Volume one
  stray normal condemned every cluster above it, the walk read 4,000 sources per point and ran out of memory at
  theta 1 (job 106437). Replaced by the measure-weighted RMS chord, sqrt(2 (1 - |mean normal|)), the statistic
  the monopole error depends on, at no cost (2783a7e09, snapshot `code_dtt6`).

### Where the step's time goes (profile, job 106425, theta 1, position-only tree, 10.3 s step)

float64 prefix scans of the node means (`aten::cumsum`, 74 ms each, 74 per step with the recompute and the
backward) 7.2 s of 10.3 s; gathers (`aten::index`) 2.7 s; the basis MLP's matmuls and their backward about 1.5 s;
the Triton segment kernels 0.6 s. Fixed in `code_dtt5`: node means by one leaf scatter and pairwise sums up the
tree in float32; pair invariants computed once without gradient and kept, only the basis MLP checkpointed. The
profile is repeated on the sixth snapshot (job 106467).

### In flight

- Job 106465: E2 with the orientation-aware (RMS) tree. Job 106466: the decomposition with it. Job 106464: the
  DrivAerML cost ladder with it. Job 106467: the profile. Jobs 106435 / 106436: the same E2 and decomposition with
  the largest-chord tree (accuracy only; its cost is known to be unusable).
- Lanes `dtt_vol_p_seed42/43` (E3's P arms, read-only without a read, 512 / 512, lr 3.75e-4, 200,000, 500
  epochs): training at 0.474 s per step and 130 GB per rank, about 51 s per epoch, so roughly 7 hours. The TG
  lanes wait for a usable form.

### Decomposition with orientation-coherent clusters (job 106436, largest-chord tree, same 12 cases)

| replaced by the cluster mean | seed 42, theta 0.25 | seed 42, theta 0.5 | seed 43, theta 0.5 | seed 42, theta 1.0 |
|---|---|---|---|---|
| position (centroid) | 0.004 / 0.001 / 0.019 | 0.014 / 0.004 / 0.081 | 0.010 / 0.005 / 0.10 | 0.023 / 0.010 / 0.14 |
| normal | 0.008 / 0.002 / 0.014 | 0.022 / 0.006 / 0.078 | 0.009 / 0.005 / 0.055 | 0.041 / 0.014 / 0.11 |
| key (per block) | 0.018 / 0.005 / 0.042 | 0.054 / 0.011 / 0.16 | 0.051 / 0.015 / 0.082 | 0.088 / 0.017 / 0.25 |
| value (per block) | 0.019 / 0.005 / 0.043 | 0.048 / 0.010 / 0.13 | 0.047 / 0.014 / 0.11 | 0.071 / 0.016 / 0.19 |
| all four (the tree read) | 0.018 / 0.005 / 0.041 | 0.054 / 0.012 / 0.18 | 0.053 / 0.017 / 0.13 | 0.091 / 0.020 / 0.28 |
| error ratio to dense, p / v / nut | 1.30 / 1.04 / 1.37 | 2.49 / 1.21 / 3.94 | 2.29 / 1.48 / 2.64 | 3.90 / 1.56 / 6.21 |

Orientation coherence removes the normal smear (0.24 to 0.008 on pressure at theta 0.25) and with it most of the
damage: the tree read's deviation falls from 0.24 to 0.018 at theta 0.25 and the pressure error ratio from 9.8 to
1.30, at the same pair count (9.6e6 against 9.8e6 per block). What remains is the key and value smears, each the
whole residual (0.018 and 0.019 against 0.018 together) and first order in theta (0.018, 0.054, 0.091 at 0.25,
0.5, 1). The preregistered bar needs a deviation near 0.0025, about theta 0.03, at a pair count the Volume task
cannot afford (the ladder is 2.8e8 pairs per block at theta 0.5 already).

### Decision (2026-10-06 evening): train with the tree

Route A, approximating the dense-trained read, is closed: with the right clusters the first-order aggregate of the
keys and values is a 2% to 5% deviation at theta 0.25 to 0.5, and second-order corrections would buy a factor,
not the thirty needed. Route B is taken: the read-only model is trained with the tree read at a fixed theta, so
the kernel, keys and values are learned under the aggregation (the clusters become part of the model, as in
multipole attention). What stands: exactness at theta = 0 (the dense read), exact query independence at every
theta (a point's walk sees the tree and the point alone), the tree frame-free up to ties, theta the one structural
constant. What changes: theta is no longer a tolerance on a defined model but a model constant like a kernel
radius, chosen by cost under the preregistered bar (step within 6x the read-only floor), and the verdict is E3
alone (TG against I, with P as the floor). Lanes at theta 1 and 0.5, two seeds each, launch when the profile on
the sixth snapshot shows the step within the bar; theta 0.25 only if the cost allows.

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
