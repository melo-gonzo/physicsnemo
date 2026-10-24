# DTT-POSE phase A: why ISLA-DTT's forest depends on pose, precision and numbering (2026-10-24)

Written 05:45 EDT. Probes: `probe/probe_pose.py` (versions in `probe/`; LHR copies `$T/kernel/dttpose/probe_pose_v*.py`)
runs a small random-weight ISLA-DTT forward on real cases up to `tree_read_pairs`, captures the forest's exact inputs,
and builds forests under variants with a copy of `BoundaryForest.build`/`point_pairs`. The copy is checked bitwise
against the real 0.11.0 code on every case (`*_copy_bitwise_equal_ref`: true in all 106 forests). Code: 0.11.0
(`code_dttfast_integ_ref0110` = 1e7f03d7d, whose ISLA code equals 045dd14fe; `tree_read.py` md5 57891d30, unchanged at
7213c70c5). Cases: the lug, 24 test cases (every 25th), posed by the scorer's dataset `ginot_lug_volume_posed`, and 8
more posed by the probe itself (FixedRandomPose, translate 1, after the pipeline); AirfRANS, 24 full_test cases (every
8th), `airfrans_uvolume_2d_posed`; DrivAerML Volume, 2 val cases (run 15, 55), whole boundary, 200k tokens, 20k
points, posed by the probe with the native token draw held fixed (the posed dataset file subsamples different interior
points, Kabsch residual 3.1, so it cannot be compared point for point). Theta 1, leaf 4, as trained. Raw records
`probe/raw/*.jsonl`; tables `probe/tables_366118.md`. Jobs in `JOBS.md`.

Metrics. "Points differ": a point's list (cells and nodes, each node as its cell set) is not the same multiset. This is
a harsh metric: one cell moved anywhere changes a coarse node every point reads. "Pairs unmatched": the share of the
posed forest's (point, entry) pairs with no identical native pair. Medians [min, max] over cases.

## 1. Octants (lug, float32)

| quantity | lug, 24 posed (scorer's protocol) | lug, 8 self-posed |
|---|---|---|
| cells whose sign octant flips, posed vs native | 754 [201, 1415] (11.1 % [4.1, 19.7]) | 928 [300, 1019] (12.9 %) |
| native cells with smallest canonical-frame normal component < 1e-6 / 1e-5 / 1e-4 | 27 / 84 / 100 % | 30 / 84 / 100 % |
| native cells whose smallest component is below the bound on its rounding (tau, below) | 85.9 % [78.2, 89.9] | 86.1 % |
| points whose list differs, 0.11.0 | 100 % in every case | 100 % |
| pairs unmatched, 0.11.0 | 55 % [40, 80] | 65 % [37, 82] |
| points that differ when the posed forest is built with the native octant labels | 5.9 % [4.1, 11.7] | 6.7 % [4.2, 14.9] |
| posed nodes (cell sets) without a native twin, labels forced | 0 [0, 9] | 0 [0, 9] |
| same, labels forced and the box test given its error band (section 3) | 1 [0, 641] points | 0 [0, 720] |

- **The float32 share is the bf16 one.** 84 % of the lug's cells have a component below 1e-5 in float32 too (DK-SENS's
  84 % was a bf16 measurement): the lug's faces are parallel to its principal axes, so the exact components are 0 and
  the stored ones are rounding residues (median 1.5e-6). The bound on a cell normal's rounding error from its float32
  vertices, tau = u (6 V (|e1| + |e2|) + 4 |e1||e2|) / |e1 x e2| + 5u, has median 1.4e-5 on the lug, and 86 % of cells
  sit below it. The sign of these components is rounding, natively and posed.
- **The flips account for the posed forest's different lists.** With the native labels forced, 94 % of the points get
  their native list back, and every posed node has a native twin in 22 of 24 cases. What is left is the box test
  (section 3): with labels forced and the box test banded, the median case differs at 1 point. The two cases with
  641 / 720 left are band-edge cases (generic, section 4). CODE_FACTS's inference is right, and there is a second,
  smaller mechanism.
- **The canonical frame is pose-equivariant to rounding on the lug.** Canonical coordinates posed against native differ by
  at most 1.9e-7 [1.2e-7, 2.6e-7] of the extent; no column changes sign in any of the 32 cases. The two near-symmetric
  axes have skews 1.1e-6 and 1.3e-6 (median), reproduced under pose to 2.5e-8 (max 1.1e-7, ratio at least 12), so the
  third-moment signs are decided by the mesh's own asymmetry (the split hex faces), not by rounding, and
  `CANONICAL_SKEW_TIE = 1e-9` never comes into play. A rigorous first-order bound on the skew's rounding
  (3 sum w p^2 delta / m2^1.5, delta = 2u|x|) is 2e-6 / 1.1e-6, above the skew in 20 / 7 of 24 cases, and its root-sum-
  square form is 3e-8 / 2e-8. See DESIGN.md section 5.
- AirfRANS: 1 [0, 8] cells flip; no component below 1e-5 (0 %); the bound covers the posed-native normal difference with
  ratio at most 0.36 (lug 0.19). DrivAerML: 18 / 25 cells flip of 200k; 0.01-0.02 % below 1e-5.

## 2. bf16

The probe reproduces training's forest by wrapping build and walk in `torch.autocast(bfloat16)` (as DK-TREEPREC).

| quantity | lug | AirfRANS | DrivAerML |
|---|---|---|---|
| native: points whose bf16 list differs from the float32 list | 100 % [94, 100] | 86 % [50, 100] | 100 % |
| posed vs native under bf16: pairs unmatched | 83 % [66, 94] | 77 % [47, 90] | 94 / 93 % |
| posed vs native under float32 (0.11.0): pairs unmatched | 55 % | 9.7 % [0.0008, 15.5] | 4.2 / 5.8 % |

Isolation (octant labels held at the native float32 ones and the box test banded, so only the coordinates vary; points
whose list differs posed vs native):

| coordinates of the trees and the queries | lug | AirfRANS |
|---|---|---|
| float32 `points @ frame` (0.11.0 without autocast) | 1 [0, 641] points | 13 [3, 778] points |
| autocast's cast of `points` and `frame` to bf16 before the product | 100 % | 87 % [73, 92] |
| float32 product with the frame alone rounded to bf16 | 100 % [59, 100] | 34 % [1, 70] |
| float32 uncentred product, its result rounded to bf16 | 100 % [98, 100] | 87 % [73, 91] |
| float32 centred product `(x - c) @ frame`, its result rounded to bf16 | 1 [0, 250] points | 4 [0, 91] points |

- **Both candidate mechanisms break pose invariance on their own.** Under autocast, `points @ frame` casts both operands
  to bf16 (the lab coordinates lose 8 bits relative to their lab magnitude, the frame is rotated by about 4e-3) and
  rounds the product to bf16. Each of the three roundings alone makes nearly every posed list differ. A centred product
  rounded to bf16 would be pose-invariant (as float32 is) but still a bf16 forest, different from the float32 one: it
  restores pose and not precision.
- **Which forest quantities depend on autocast** (GB300, under autocast): the boxes `aabb_min/max` (bf16), the node
  measures `node_total_area` and so `forest.measure` (bf16, `.to(dtype)` of the float64 prefix sums): these are the
  denominators of every block's `node_means` and the nodes' log-measures, so under bf16 a node's mean key and value is
  scaled by up to 2^-9 and its log measure shifted by up to 2^-9 against the exact sum of its cells' (the monopole is
  no longer exact for a constant). `diameter_sq` is float32 on CUDA (autocast upcasts `pow`/`sum`; bf16 on CPU). The
  walk's box table is float32 (`cat` promotes) but holds bf16-rounded values. Centroids, mean normals and rank-1 means
  are float32 (aggregated from the float32 lab points). The octant labels: natively 0 of the lug's flips come from bf16
  (the native frame is close to the lab axes), posed 4,586 [969, 7,397] (against 754 in float32).
- The forest's inputs themselves (`points / gauge`, `n_hat`, `log_frac`, `query_points / gauge`) are float32 under
  autocast (elementwise ops and `logsumexp`): every bf16 effect is inside `BoundaryForest.build` and `point_pairs`.

## 3. Other dependences on pose, precision or numbering

| mechanism | lug | AirfRANS | DrivAerML | kind |
|---|---|---|---|---|
| box-test ties: native points whose list changes when theta moves by a factor 1 +- 1e-6 | 6.7 % [4.5, 11.1] | 0.001 % [0, 0.004] | 0.07 % | structured (lug), rare elsewhere |
| same at 1 +- 1e-5 | 9.0 % | 0.013 % | 0.7 % | |
| Morton codes duplicated inside a tree (ties broken by input order) | 0 | 0 | 0 | none seen |
| renumbering of the cells (random permutation): points whose list differs | 0 (32 cases x 2 poses) | 0 (48) | 0 (4) | none seen |
| token draw changed by the pose (upstream of the forest; TOKEN-DRAW's residual) | 0 (kept whole) | 0 (kept whole) | 12,050 / 31,714 of 200k | generic |

- **Box-test ties are the lug's second mechanism.** The lug's prediction points are its hexahedral mesh nodes and its
  boundary cells split the same hex faces, so distances and box diagonals are commensurate and at theta = 1 the test
  `min_dist^2 theta^2 > diam^2` is an exact equality for many (point, node) pairs: 6.7 % of points change list under a
  1e-6 change of theta, against 0.001 % on AirfRANS. Rounding decides them, and differently in each pose.
- **Numbering.** The forest is a function of positions; ties broken by index never occurred (no duplicate Morton codes),
  and the frame's float64 sums move by 1e-16 under renumbering, which changed nothing. Kept as a residual risk.
- **Why AirfRANS posed equals native today.** Its normals have no zero components in its frame (none below 1e-5), so only
  0-8 cells flip octant, and its points are not on a lattice commensurate with its cells (box ties 0.001 %). A flip of
  one extreme cell still requantizes that octant's Morton grid (9.7 % of pairs unmatched), but the read's change is
  small: the standing posed / native is 1.00 at 0.00-0.25 %. Under bf16 training the same set loses 77 % of its pairs.

## 4. What no construction removes (the generic residual)

Posing in float32 changes the inputs by their rounding (u |x|, u = 6e-8). Any discrete choice made from them (a face
label, a Morton cell, the box test) has some cells or pairs within rounding of its threshold, and those are decided by
rounding whatever the rule. A tie band moves the threshold away from the structured coincidences (exact zeros, exact
equalities), not away from a continuum of values. The structured ties are what makes the lug's posed forest different
in every list; the residual is a handful of cells per case. The prototype's numbers are in DESIGN.md.
