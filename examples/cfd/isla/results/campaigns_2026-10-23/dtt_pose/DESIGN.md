# DTT-POSE design: ISLA-DTT's forest as a function of the boundary alone (2026-10-24)

Written 05:58-06:00 EDT. Lane DTT-POSE. Mechanisms with counts: `MECHANISMS.md`. The design is prototyped, uncommitted,
in `/home/psharpe/gh/physicsnemo-isla-dtt-pose` (branch psharpe/isla-dtt-pose, at 7213c70c5 after the supervisor's
merge request; no commit yet). Snapshot `$T/code_dttpose_proto1` (tree_read.py md5 5905eeda, model.py f651e394);
the base `$T/code_dttpose_base` = 7213c70c5. Every number below is from GB300 probes on real cases (JOBS.md),
one code version per process, the forest's tree_read.py named in each log.

## 1. What is wrong, in one paragraph

The forest makes three discrete choices from float32 numbers: each cell's orientation class (the sign octant of its
normal in the canonical frame), the Morton trees over the cells' canonical coordinates, and the walk's admission test
`theta * dist > diam`. Under bf16 autocast the coordinates, boxes and node measures are bf16, and every choice moves
(training's forest differs from the scored one at 86-100 % of points, and posed from native at 77-94 % of pairs). In
float32 the lug decides two choices by rounding in every pose: 86 % of its normals have exact zero components in the
frame, which `c > 0` reads as signs of rounding residues (11 % of cells flip octant under pose), and its mesh-node
queries make `dist = diam` an exact equality at theta 1 for many pairs (6.7 % of lists move under a 1e-6 change of
theta). Forcing the native octant labels restores 94 % of the posed lists; banding the box test restores the rest but
a point. The canonical frame itself is pose-equivariant to rounding on every probed case and is not the mechanism.

## 2. The changes (all in tree_read.py's forest and its call in model.py)

| | change | invariance it restores |
|---|---|---|
| D1 | Build and walk with autocast disabled: `BoundaryForest.build` and `point_pairs` run their bodies inside `torch.autocast(enabled=False)`, in the inputs' precision (float32), frame arithmetic in float64. | Precision: the forest of a bf16-autocast forward is bitwise the float32 forward's (lists, boxes, measures, diameters). Also removes the bf16 node measures that biased every node mean and log-measure by up to 2^-9. |
| D2 | Centred canonical coordinates in float64, rounded once: `x_f = ((x - c) @ F)` by `frame_coordinates` in float64, `c` the measure-weighted centroid, `F = canonical_frame(x.double(), measure)` kept in float64 (the function is unchanged; only its input dtype). Queries the same way. The forest stores `frame` (float64) and a new `origin`. | Pose: the trees' coordinates carry rounding relative to the boundary's extent, not its lab position; the float32 product's error grew with the translation. |
| D3 | Orientation classes: the cube face of the normal (`normal_faces`: the axis of the largest canonical-frame component and its sign, 2d faces) in place of the 2^d sign octants, with a tie band: a component within `2 (tau_i + eta)` of the largest is tied and the face goes to the lowest tied axis. `tau_i` bounds cell i's normal rounding (`normal_error_bound`, from its vertices, about the boundary's centroid); `eta` bounds the frame's rotation by the rounding of its inputs (`frame_rotation_bound`, Davis-Kahan, `4u tr C / min eigengap`). | Pose (and precision of the inputs): the face is not decided by rounding. Axis-aligned faces, the lug's 86 %, are far from every face boundary (margin about 1); only normals near 45 degrees are near one, and those within the bound go to the fixed axis. |
| D4 | Box test with its error band: each tree's `node_diameter_sq` becomes the squared separation diameter `D' = (D (1 + theta k u) + 2 (1 + theta) k u R) (1 + 2u) / (1 - (k + d + 3) u)`, `k u = 3u + eta` the coordinates' error per unit length, `R` the node box's reach from the centroid, so a node is admitted only when it is certainly well separated; the walk itself is unchanged. | Pose: exact ties of `dist = diam` (the lug's lattice) are opened in every pose instead of falling either way. |
| D5 | model.py: in `forward`, for ISLA-DTT only, the kept cells' `tau` (float64, from the meshes' vertices as given, centroid from the draw's own float64 measures and centroids) is carried as a boundary token field `normal_tol` and passed to `tree_read_pairs` (new argument `normal_tol`, default `None`: the normals are exact as given, `tau = (d + 2) u`, the case of `_forward_tensors`' bare tokens). | Pose: the band is the bound of the normals the forest actually reads. |

**Why faces and not octants with a band.** I measured both (`probe/raw/v4_*.jsonl`). Octants with the same per-cell band
still flip 66 [5, 168] lug cells and leave 22 % of pairs unmatched: the lug's small components (1e-7 to 5e-5) straddle
any band edge continuously. Faces have margins of order 1 for axis-aligned cells. A face holds normals with
`|n_k| >= 1/sqrt(d)`, the same bound as an octant, so the coherence test's vacuity bound (theta >= 0.919 in 3D) and
the no-opposite-normals property are unchanged; `-n` lands on the opposite face (antipodal), bitwise.

**One judgment call: the bands are pose-invariant.** `tau` and `R` are measured about the boundary's centroid, not
from the lab origin. A rigorous bound for the given float32 inputs would use the lab magnitudes `|x|`, but then each
pose has its own band, and a band edge that moves with the pose decides the cells and pairs near it: measured, the
lab-magnitude bands left 0.15 % of lug pairs unmatched (5 / 24 cases identical) and 0.014 % on AirfRANS with outliers
to 31 %, against 0.07 % (14 / 24 identical) and 0.002-0.02 % for the centroid-relative ones; DrivAerML 7.6 / 7.0 %
against 0.35 / 1.07 %. The bands are therefore the error bounds of the arithmetic in the boundary's own frame. A case
placed far from the origin (translation much larger than its extent) adds input rounding of order `u |c|` that no
function of the boundary can follow; the posed protocol (one RMS radius) is within the bands' margin (the measured
posed-native normal differences reach at most 0.19 (lug) and 0.36 (AirfRANS) of the lab-magnitude bounds).

## 3. Measured effect (prototype against 0.11.0 at 7213c70c5, theta 1, leaf 4)

End to end (`probe/probe_impl.py`; a small ISLA-DTT, hidden 64, 2 layers, every parameter random, read included;
jobs 366658, 366705): posed against native in float32, and a bf16-autocast forward against the float32 one.

| | lug, 12 test cases, scorer's posed set | AirfRANS, 12 full_test cases, self-posed | DrivAerML Volume, 2 val cases, self-posed, draw fixed |
|---|---|---|---|
| forward, posed vs native, RMS diff / RMS (scalar output): 0.11.0 | 2.1e-6 [9e-7, 4.8e-6] | pressure 7.1e-5, nut 3.3e-5 | 3.3e-5 / 3.3e-5 |
| same, prototype | **1.6e-7 [1.1e-7, 2.2e-7]** (float32 roundoff) | 7.1e-5, 3.3e-5 (unchanged: not the forest) | 3.3e-5 / 3.3e-5 (not the forest) |
| forest pairs unmatched, posed vs native: 0.11.0 | 52 % [42, 76] | 9.5 % [0.0007, 13] | 5.0 % [4.2, 5.8] |
| same, prototype | **0.0002 % [0, 2.0]**, 4 of 12 identical | **0.001 % [0.0003, 0.43]** | **0.52 % [0.35, 0.69]** |
| bf16 forward's forest bitwise equal to float32's: 0.11.0 / prototype | 0 / **12 of 12** | 0 / **12 of 12** | 0 / **2 of 2** |
| pairs per point: 0.11.0 / prototype | 117 / 112 (-4 %) | 21.6 / 21.1 (-2 %) | 706 / 586 (**-17 %**) |
| trees | 8 / 6 | 4 / 3 | 8 / 6 |
| `tree_read_pairs` eager (build, walk, basis), GB300: 0.11.0 / prototype | 115 / 95 ms | 65 / 59 ms | 230 / 192 ms (20k points) |

Forest-only probe on more cases (`probe/raw/v4_*.jsonl`, variant P2 = this design without D5's model plumbing, tau
computed the same way): lug 24 posed cases 0 % [0, 1.72] pairs unmatched (mean 0.072 %), 14 of 24 identical; lug 8
self-posed similar; AirfRANS 24 cases 0.0008 % [0.0003, 0.43]. Renumbering of the cells changed no list in any case,
before or after. On synthetic meshes (prototype, CPU): a random cloud rotated, reflected, scaled and translated, and an
axis-aligned box (normals exactly along the axes) in four float32 poses give identical forests and lists, and a bf16
autocast build is bitwise the float32 one.

**What remains (the generic residual), and a contradiction with the brief.** The brief expects the forest equal under
pose point for point on real cases. That holds on synthetic and structured meshes and on 4-14 of 24 lug cases, not on
every real case: posing in float32 moves every input by its rounding, and a discrete function of continuous inputs has
some cells within rounding of each threshold (a face boundary, a Morton grid line, the box test), whatever the rule.
What the design removes is every structured dependence (exact zeros, exact equalities, bf16), leaving a handful of
cells per case: DrivAerML's 0.35-0.69 % of pairs is mostly Morton-grid lines at 200k cells (one cell flipping face moves
only a few nodes; DrivAerML's posed draw itself changes 12k-32k of its 200k tokens upstream, TOKEN-DRAW's residual).
I propose phase C's real-case checks as measured bounds (section 7), not equality.

## 4. Cost (DrivAerML Volume, whole boundary, 200k tokens)

- New work per forward: `tau` for the 200k kept cells (one float64 gather of their vertices, elementwise), the
  centroid from the draw's existing float64 measures and centroids (two reductions over all cells, 17.7M), one extra
  3x3 `eigvalsh` (`eta`; one host sync, as the frame's `eigh` already has), two float64 column products (tokens and
  points), and the separation diameters per node (elementwise). Measured as part of `tree_read_pairs`: the forest
  build, walk and basis together are faster than 0.11.0 (230 -> 192 ms at 20k points), because 6 trees replace 8 and
  the faces give fewer pairs.
- Pairs: -17 % per point on DrivAerML (706 -> 586; cases 702 -> 593, 711 -> 580). Projected with DK-COST's fitted model
  (step = 203 ms + 5.16 ms per Mpairs per block + 158 ms per 100k points + 184 ms per 100k tokens; memory = 3.2 GiB +
  0.265 GiB per Mpairs + 35.0 / 36.0 GiB per 100k points / tokens) from DK-COST's theta-1 leaf-4 828 pairs per point
  scaled by 0.83:

| | pairs per point | Q 200k: step / peak | Q 400k: step / peak | largest Q that fits 274 GiB |
|---|---|---|---|---|
| 0.11.0 (DK-COST, measured where run) | 828 | 1741 ms / 189 GiB (1790 / 189) | 2912 / 303 (OOM) | 349k |
| this design (projected) | 687 | 1596 ms / 182 GiB | 2621 / 288 (does not fit) | 374k |

  The ratio 0.83 is from two DrivAerML cases at 20k points under float32; DK-COST's 828 is under bf16 at 200k points.
  A measured training step at 200k needs a GPU job of about 0.5 GPU-h; I can run it in phase C if wanted.
- Memory of the forest itself: boxes float32 instead of bf16 (2 x 4 bytes x d per node more; about 5 MB at 200k
  tokens), `origin` 3 floats, `normal_tol` one float64 per token (1.6 MB).

## 5. Not proposed, with reasons

- **`CANONICAL_SKEW_TIE` (1e-9) stays.** Measured: it never decides anything on the probed cases (every axis's skew
  is at least 12 times its posed-native change; no frame column flipped in 56 posed forests). A rigorous worst-case
  bound of the skew's rounding (about 2e-6) exceeds the lug's two near-symmetric skews (1e-6) in 20 / 7 of 24 cases,
  and the tie's fallback (the lab axis of largest overlap) is pose-dependent: a derived threshold would make the lug's
  frame pose-dependent, where today it is not. No pose-invariant rule can sign the axis of a mirror-symmetric body
  (the body is its own mirror image), so a principled threshold needs a different fallback, which is a frame question
  for the token draw as much as the forest: `canonical_frame` is shared with `sample_boundary_cells`, and changing it
  changes interacting ISLA's draw. I recommend leaving it to a separate decision; the forest's `eta` band covers the
  frame's rotation, not its signs.
- **A shared Morton grid across the faces' trees** (one quantization box for all trees, so a cell changing face moves
  only the nodes holding it): it cut AirfRANS's worst case from 0.43 % to 0.03 % of pairs but did not change the lug
  or DrivAerML medians, and it needs a `bounds` argument in `ClusterTree.from_points` (physicsnemo.mesh core, GLOBE's
  tree). Left as an option.

## 6. What this touches and what it changes

- **Interacting ISLA: bitwise unchanged.** `canonical_frame`, `frame_coordinates`, `sampling.py` and `geometry.py` are
  not changed. model.py computes and carries `normal_tol` only when `self.kernel_reads is not None` (ISLA-DTT). The
  release goldens `isla_default_v1`, `isla_interior_v1` and the backbone golden must pass unchanged; `isla_dtt_v1` is
  regenerated.
- **ISLA-DTT computes a different function** wherever the forest changes, which is every case (faces replace octants).
  Its checkpoints (0.9.0 snapshots and any 0.11.0) would read a different forest than they were trained with.
- **Every ISLA-DTT result** (standing scores, DTT-KNOBS stage 2 on code_cmpaf_fix) is on the old forest. Retraining is
  outside this lane.
- `normal_octants` is removed (`normal_faces` replaces it; ISLA is unreleased, no shims). `BoundaryForest.frame` becomes
  float64 and gains `origin`. The tree_read docstrings on the orientation split and the symmetries are updated.

## 7. Phase C plan (after approval)

Commits on psharpe/isla-dtt-pose (merged with origin/psharpe/isla at 7213c70c5): (1) the forest (D1-D4) with its
tests; (2) model.py's `normal_tol` (D5) with its tests; (3) the version. Tests (test_isla_tree_read.py, test_isla_release.py):
- forest equality under rotation, reflection, rescale and translation (as now, plus float32 poses of an axis-aligned
  box mesh with lattice queries where theta-1 ties occur, and a cylinder with normals at 45 degrees, both failing on
  7213c70c5); equality under renumbering; bitwise equality of the bf16-autocast and float32 builds (lists, boxes,
  diameters, measures) on CPU and CUDA; `normal_faces` antipodal and tie rule; `normal_error_bound` covering the
  measured normal change of posed meshes; the separation diameter opens an exact tie;
- an ISLA-DTT forward with random weights posed equal to native to 1e-6 on a synthetic box case; interacting goldens
  unchanged;
- the whole ISLA directory and the recipe tests on CPU on LHR (cpu-short), the tree-read and release files with CUDA
  on GB300;
- a verification job on real cases (this probe on the committed code): lug 24 cases forward posed / native <= 1e-6 on
  every case; bf16 forest bitwise on every case of the three datasets; pairs unmatched reported.

## 8. What DTT-FAST's fast build and walk must change to stay bitwise with this reference

1. Run outside autocast (build and walk); boxes, diameters and measures in the inputs' precision (float32).
2. Coordinates: `F = canonical_frame(points.double(), measure)` (float64), `origin = sum(w~ x)` in float64 with
   `w~ = clamp_min(measure.double(), 0) / sum`, `x_f = frame_coordinates(x.double() - origin, F).to(points.dtype)`
   for cells and queries (column products in that order).
3. Classes: `normal_faces(frame_coordinates(normals.double(), F), tol + eta)`, faces `0 .. 2d - 1` in that order
   (trees for k in range(2d)), `tol` = the `normal_tol` argument (float64) or `(d + 2) u_normals`,
   `eta = frame_rotation_bound(x - origin, w~, u_points)`.
4. After each `ClusterTree.from_points`: `node_diameter_sq = separation_diameter_sq(tree, theta, 3u + eta, d, u)`
   (computed in float64 from the float32 boxes, then cast); the walk reads it as it reads `node_diameter_sq` now.
5. `BoundaryForest` fields: `frame` float64, new `origin`.
Their existing bf16-autocast bitwise test against the reference stays valid (both are now autocast-independent).

## 9. Version handling (supervisor decides)

Proposal: **0.12.0**, `_version = 12`; `_refuse_earlier_state_dict` refuses state-dict version 11 **only for an
ISLA-DTT module** (`module.query_independent`), since interacting ISLA computes the same function at 0.12.0 as at
0.11.0; earlier versions are refused as now. Alternative: refuse 11 for every ISLA (simpler; interacting 0.11.0
checkpoints, if any are kept, would have to be reloaded with 0.11.0 code). I recommend the first.

## 10. Decisions requested

Approve D1-D5 as above, with: the pose-invariant (centroid-relative) bands of section 2; `CANONICAL_SKEW_TIE`
unchanged (section 5); no shared Morton grid; real-case checks as measured bounds (section 3, 7); version 0.12.0
with the DTT-only refusal. Compute so far: 2.1 GPU-h of the 12 cap (JOBS.md); phase C needs about 1-2 more.
