# LIB-TREE results (2026-10-25, 01:10 EDT)

Branch `pr/cluster-tree-warp` (worktree `/home/psharpe/gh/physicsnemo-pr-cluster-tree-warp`), base upstream/main
289e57382, head 42de1669d9fe908afc7cdab452827a6bac9a7ff7, pushed to `origin` (physicsnemo-private). All GPU numbers:
LHR GB300, torch 2.13.0+cu130, warp 1.17.0. Job logs and JSON under `results/` (out_376887, out_377360, out_377545,
cpu_*.log); the scripts are in `lhr/`; JOBS.md lists every job.

## Commits
| sha | what |
|---|---|
| 35ca57336 | P1: grouped Warp Morton build behind `from_points(implementation=)`, new `ClusterTree.from_point_groups` (`_lbvh_warp.py`) |
| 1d56696ff | P2: Warp point walk behind `find_point_interaction_pairs(implementation=, count_only=)`; streams grouped by target, then root (`_point_walk_warp.py`) |
| d89f35209 | P3: the Warp build's boxes in one bottom-up kernel (`_lbvh_warp.node_boxes`) |
| 42de1669d | float64 walks use Warp under `None` (measured below); CHANGELOG one bullet; the boxes' exceptions stated |

## Bitwise proofs
- Library tests (`test/mesh/spatial/test_cluster_tree.py`, +700 lines): Warp build against torch on every field (2-D/3-D,
  fp32/fp64/bf16/fp16, bf16 autocast, leaf 1/4/8, duplicates, tiny and empty); `from_point_groups` against per-group
  `from_points` + `merge`; Warp walk against torch entry for entry (mask, NaN/inf targets, fp32/fp64/bf16 autocast,
  count_only); the torch order against ISLA's per-tree concatenation (old level-major walk written in the test); ISLA's
  threshold test; fallbacks and refusals; non-default CUDA stream for build and walk; lazy import of the Warp modules.
- ISLA equivalence (probe `lhr/isla_side.py` with ISLA's own code at 5b70c8686 in one process, `lhr/lib_side.py` with
  the library in another; real dumps of DTT-FAST): every per-octant tree field, the forest table (range_start/count,
  left/right, boxes, diameters, measure, cells, roots), the near and far streams in order, and the counts are equal
  in sha256 to ISLA's fast build and walk, for `implementation="torch"` and `"warp"`, at P2 (job 376887) and at the
  head with P3's boxes (job 377360):

| dump | autocast | cells / nodes | queries | near / far entries | lib build warp / ISLA fast (ms) | lib walk warp / torch (ms) |
|---|---|---|---|---|---|---|
| drv_body | bf16 | 200k / 140,086 | 400k | 13.8M / 264.3M | 5.98 / 8.67 | 23.7 / 110.4 |
| drv_body | fp32 | 200k / 139,772 | 400k | 13.4M / 264.6M | 5.98 / 8.50 | 23.7 / 110.8 |
| aerozoo | bf16 | 200k / 138,388 | 400k | 8.3M / 86.2M | 6.45 / 8.92 | 10.3 / 40.8 |
| airfrans (2-D) | bf16 | 688 nodes | 180,593 | 0.40M / 3.35M | 5.44 / 6.29 | 1.28 / 6.47 |
| airfrans_whole (2-D) | bf16 | 1,858 nodes | 180,593 | 0.34M / 6.59M | 4.63 / 6.70 | 1.51 / 7.82 |
| lug | bf16 | 4,192 nodes | 9,048 | 0.10M / 0.99M | 5.49 / 7.77 | 1.37 / 6.70 |

- GLOBE: every `ClusterTree.from_points` output of a forward (all 14 fields) is identical in sha256 between upstream/main
  and the branch (80k and 200k cells, jobs 376887 and 377360). GLOBE's CUDA forward is not run-to-run deterministic
  (its own outputs differ between two identical calls), so outputs are compared on CPU: GLOBE's two non-regression
  configurations with every `from_points` forced to Warp (`lhr/nonreg_warp.py`, CPU Warp, 6 Warp builds) give outputs
  equal in every bit to the default torch build (job 377361). Training losses of 15 steps are equal in every bit
  base vs branch.

## GLOBE training step (DrivAerML vehicle, K sampled cells, globe_surface_drivaerml config, bf16 autocast, AdamW)
| K | base step (ms, 2 runs) | branch step (ms, 2 runs) | change | from_points on K points base → P2 → head (ms) | peak GiB |
|---|---|---|---|---|---|
| 80,000 | 5490.4, 5484.8 | 5469.4, 5463.0 (head 5469.1) | −21 ms (−0.4 %) | 15.2 → 4.8 → 3.7 | 56.96 both |
| 200,000 | 16308.5, 16315.3 | 16287.7, 16285.9 | −25 ms (−0.15 %) | 17.3 → 5.1 | 87.26 both |
GLOBE builds two trees per forward through `from_points`; its step is dominated by its kernel evaluations, so the
gain is about 10 ms per tree.

## Walk: torch against Warp on ISLA's real 200k-cell DrivAerML forest (job 376887; ms, peak GiB)
| case | 100k | 400k | 800k | 1.6M |
|---|---|---|---|---|
| bf16 autocast torch / warp | 31.5 / 7.3 | 110.2 / 23.6 | 215.4 / 45.2 | 429.0 / 88.4 (109 / 25 GiB) |
| float32 torch / warp | 32.5 / 7.3 | 110.8 / 23.8 | 215.8 / 45.3 | 429.9 / 88.5 |
| float64 torch / warp | 38.3 / 30.9 | 137.8 / 115.5 | 275.5 / 227.8 | 557.2 / 453.4 (109 / 25 GiB) |
| count_only float32 torch / warp | 25.0 / 3.0 | 82.3 / 8.6 | 159.0 / 15.9 | 316.0 / 30.8 (88 / 0.3 GiB) |
Warp is faster in float64 at every size, so `None` picks Warp for float64 too (LIB-MOVE's 512 vs 583 ms compared ISLA's
per-tree reference walk with its fast walk, not the library's forest walk).

The torch walk's one stable sort per stream (the per-target order) costs, against upstream/main's torch walk on the
same merged forest (job 377360, `lhr/torchwalk.py`; same pairs as multisets): 25.2 → 32.4 ms at 100k, 85.2 → 110.9 at
400k, 165.2 → 216.3 at 800k (+29-31 %), peak 23.3 → 27.4 GiB at 400k (+17 %).

## Build: from_points on K vehicle centroids, torch / Warp at P2 / Warp at head (P3 boxes), ms (job 377360)
| K | f32 leaf 1 | f32 leaf 4 | f64 leaf 1 | f64 leaf 4 |
|---|---|---|---|---|
| 20k | 13.3 / 4.24 / 4.34 | 15.1 / 4.27 / 3.49 | 14.0 / 4.14 / 4.09 | 14.4 / 4.28 / 3.50 |
| 80k | 15.8 / 4.61 / 3.41 | 13.7 / 4.47 / 3.33 | 15.3 / 4.25 / 3.33 | 13.5 / 4.15 / 3.28 |
| 200k | 18.3 / 4.86 / 4.67 | 16.6 / 4.79 / 4.01 | 17.2 / 4.80 / 3.71 | 15.2 / 4.81 / 3.66 |
| 1M | 24.5 / 7.93 / 6.18 | 18.3 / 7.32 / 5.95 | 26.6 / 8.08 / 6.91 | 19.1 / 7.45 / 6.06 |
| 4M | 24.9 / 18.7 / 17.7 | 20.1 / 17.0 / 15.8 | 29.3 / 23.5 / 18.5 | 23.3 / 20.6 / 16.4 |
P3 is faster at every size but one tie (20k f32 leaf 1, within noise) and lowers peak memory (4M f64 leaf 4: 1.04 →
0.70 GiB), so it is adopted. At 4M points the build is near torch's (1.1-1.4x), as LIB-MOVE found for 20M.

## Suites
| suite | where | branch head 42de1669d | base |
|---|---|---|---|
| test/mesh/spatial | CPU (LHR cpu-short, 377361) | 391 passed, 279 skipped | — |
| test/models/globe | CPU (377361) | 258 passed, 65 skipped | — |
| test/mesh/spatial + test/models/globe | GB300 (377360) | 993 passed, 0 skipped | 871 passed at upstream/main (376887; 8 errors from a snapshot without examples/, fixed later) |
| ISLA dir + mesh/spatial + globe, merged into psharpe/isla fc6ed19d0 (merge-tree 99b7a341, clean) | GB300 (377545) | 1696 passed, 7 skipped, 2 xfailed, 0 failed | 1582 passed, 7 skipped, 2 xfailed |
| same | CPU (377542) | 1110 passed, 585 skipped, 2 xfailed | 1054 passed, 527 skipped, 2 xfailed |
pre-commit on the six changed files and `pre-commit run import-linter --all-files`: pass.

GPU use: 376887 29 min, 377360 3 min, 377545 35 min (two of its four GPUs idle) on one 4-GPU node: about 4.5 GPU-h of
the 6 GPU-h cap.
