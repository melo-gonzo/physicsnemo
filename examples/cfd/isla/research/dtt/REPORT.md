# ISLA-DTT report

## Status (2026-10-06 night PDT)

- Brief and preregistration written (`BRIEF.md`, `PREREG.md`; E4's prediction amended on the change of form below).
- Tree read implemented (`physicsnemo/experimental/models/isla/tree_read.py`, `ISLA(kernel_read_theta=...)`) on
  the pi lane's dense global read; canonical `psharpe/isla` (upstream main merged) merged in at afdb09a55.
- The first form (a80cfc0f4) walked GLOBE's dual tree (a tree over the points as well, the symmetric criterion
  (D_T + D_S) / r < theta, far-field entries expanded to their points). It passed every unit contract and was
  unusable at scale: on DrivAerML Volume (200,000 cells, 200,000 points, job 106296) it built 1.6e9 pairs at
  theta = 0.5 and 7e8 at theta = 1 and ran out of memory on a 277 GiB GB300 in the first block's read (and inside
  the traversal itself at theta = 0.25). Mechanism: a leaf of eight points in a sparse region of the volume spans a
  box of order a metre, so no boundary leaf within twice that distance is ever well separated from it, and each
  such point leaf reads the whole car cell by cell. The dual criterion charges the target's own diameter, which is
  irrelevant once the targets are expanded to points.
- Second form (this commit): a per-point walk of the boundary tree (`point_tree_pairs`), as a body walks the tree
  in an N-body code. A node is admitted when its box diagonal is below theta times the gap between the point and
  the box; an opened leaf contributes its cells; the point otherwise descends to both children. There is no tree
  over the points, so a point's list depends on the boundary and the point alone: query independence is exact at
  every theta, not only at zero. Unit level: 19 CPU tests in float64 pass (plus 2 CUDA tests on the cluster): theta
  = 0 equals the dense read to 1e-10 in two and three dimensions; a dense checkpoint loads strictly; every point's
  list covers the boundary measure exactly once and admits only separated clusters at every theta; on a sphere the
  deviation falls from theta = 1 to 0.25 and vanishes at 0; query independence exact at theta 0, 0.5 and 1;
  covariance, similarity, refinement and measure-scale contracts exact at theta = 0; every parameter receives a
  gradient at theta = 0.5.
- The read-only floor of the Volume task without a read, blocks uncompiled, 512 / 512 configuration, one GB300:
  0.70 s per step and 163 GiB peak (the compiled reference is the comparison for E3; the cost probe is rerun with
  the recipe's `compile_blocks: true`).
- In flight on LHR after the redeploy (snapshot `code_dtt2`): the snapshot's verification, E1b/E2 (the four G
  checkpoints at theta in {0, 0.25, 0.5, 1}, `scoring/dtt_theta.py`), and the cost probe at DrivAerML Volume scale
  (`scoring/dtt_bench.py`: compiled floor and theta 1 / 0.5 / 0.25, three real cases each, step time, peak memory,
  pairs per block).

## Implementation notes

GLOBE's `ClusterTree` (Morton LBVH, leaf size `TREE_READ_LEAF_SIZE = 8`) is built over the cells with the measure
fractions as areas; `compute_source_aggregates` gives every node's measure-weighted centroid, mean normal
(renormalized) and mean rank-1 inputs, and `node_total_area` its measure (its log is the softmax bias, the
quadrature weight of the cluster). `point_tree_pairs` is a level-synchronous walk: a frontier of (point, node)
pairs starting at every point's root entry, three boolean splits per level (admitted, opened leaf, descend), the
leaves expanded with `_ragged_arange`, the output sorted by point. Per block the node's key and value are the
measure-weighted means of its cells' through float64 prefix sums over the Morton order (`node_means`),
differentiable in the cells' states. The per-block read reuses the truncated read's sparse machinery (`PairPlan`,
`pair_scores`, `pair_weighted_sum`, Triton on CUDA) with a segment max for the softmax shift and no null; the
dense read's parameters are shared, so the two forms are one model. Known approximation at finite theta: the
far-field logit is evaluated at the cluster's mean invariants and mean key (a monopole in both); the renormalized
mean normal of a cluster that spans both faces of a thin body or a closed curve is a poor direction (its length is
small); the orientation-binned aggregate is the planned remedy if E2 needs it.

## Environment notes

Local WSL cannot load CUDA (`/usr/lib/wsl/lib/libcuda.so.1` is a corrupt 188 KB file; ld.so aborts); CPU float64
tests run in `~/.venvs/isla-cpu` with an empty stub `libcuda.so.1` on `LD_LIBRARY_PATH` so warp falls back to CPU.
The cluster venv `.venv-recipe` has tensordict 0.12.4; the merged tree's pyproject wants 0.14.2 (reading datasets
is fine; saving meshes is not); `.venv-recipe-td14` (tensordict 0.14.2, torch 2.12.0+cu130) exists and is the venv
of the `code_dtt2` jobs.
