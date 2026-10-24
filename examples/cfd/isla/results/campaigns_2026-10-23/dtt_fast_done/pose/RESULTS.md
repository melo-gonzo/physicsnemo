# DTT-FAST POSE: the fast forest build and walk on DTT-POSE's forest (2026-10-24)

Branch `psharpe/isla-dttfast-pose`, head **b40c9494d** (pushed): `--no-ff --signoff` merge of `origin/psharpe/isla-dtt-pose`
0f06e4065 into the DTT-FAST head 5e82a9b98, hooks on. ISLA 0.12.0 comes in only through the merge. Jobs: `JOBS.md`.
Snapshot `$T/code_dttfast_pose_a` (md5-equal to b40c9494d's physicsnemo/ and ISLA tests).

## 1. The merge and the fast path

- Reference semantics are DTT-POSE's, unchanged: `BoundaryForest.build` / `point_pairs` bodies run under
  `torch.autocast(enabled=False)`, centred float64 canonical coordinates, `normal_faces` with the tie band,
  `separation_diameter_sq`, `frame`/`origin`, D5's `normal_tol` from model.py.
- DTT-FAST's execution structure kept: `build=`/`walk=`/`kernel=`, `tree_read_walk` + `walk_range_pairs`, the one
  `tree_read_kernel` attribute. `normal_tol` is threaded through `_tree_read_walk` on both the unchunked forward
  (`_forward_tokens`) and `_QueryChunks` (`boundary.get("normal_tol")`).
- Fast build (`tree_build.py`): takes `points_f` and `face` computed by the reference's own code (frame, origin,
  `frame_coordinates`, `normal_faces(tol + eta)`) inside the same autocast-disabled `_build`; only the topology kernels
  are its own, over `2 d` faces in place of `2**d` octants. After `forest_tables`, each tree's `node_diameter_sq` is
  replaced by `separation_diameter_sq(tree, theta, 3u + eta, d, u)` exactly as the reference does, and the forest's
  `diameter_sq` is their concatenation. Fast walk (`tree_walk.py`): unchanged code; it reads `forest.diameter_sq`
  (the widened one) and builds its opening-dtype table from it as the reference's `_traversal_tables` does, and runs
  inside `_point_pairs`, outside autocast, on the reference's float64-centred query coordinates.
- Tests (`test_isla_tree_read.py`): DTT-POSE's adversarial tests parametrized over `kernel in {reference, fast}`
  (pose at finite theta, float32 poses of the frame-aligned lattice, bf16-autocast bitwise on CPU and CUDA,
  renumbering, box-test ties); new `test_fast_forest_and_walk_are_the_reference_on_face_and_box_ties` (CPU and CUDA:
  native, two float32 poses, renumbered, bf16 autocast, box ties; every forest field bitwise and lists entry for
  entry); new `test_query_chunks_read_the_forest_with_the_normals_error_bounds` (forward and `query_chunks` pass the
  same float64 `normal_tol` and build the same forest and lists); `origin` added to `_FOREST_FIELDS`; the frame-aligned
  model-forward test records through `walk_range_pairs`. Mutation check: with the fast build's separation diameter
  removed, the new test and `test_fast_forest_build_is_the_reference_forest` fail (`diameter_sq`, every tree's
  `node_diameter_sq`).

## 2. Forest equivalence on the real dumps (job 369730, `out/a/forest_table.md`)

`lhr/job/forest_probe.py`: per dump, the model's forward to the tree read records its inputs (boundary tokens,
measure fractions, D5's `normal_tol`, 20k points); every forest field, every tree's fields and (qi, si) raw are hashed
for b40c9494d's forced reference and its fast build + fast walk, and for DTT-POSE 0f06e4065 in its own process from
the head's tokens (DTT-POSE's own recorded tokens are bitwise the head's in every case and precision). Posed = the
recorded positions, normals and rank-1 inputs rotated (`orthogonal(3, 20)`) and translated in float64, rounded.

| | DrivAerML vehicle | DrivAerML whole | AeroZoo | AirfRANS | lug |
|---|---|---|---|---|---|
| float32, native and posed: every field, fast vs both references | bitwise | bitwise | bitwise | bitwise | bitwise |
| bf16 autocast, native and posed | bitwise | bitwise | bitwise | bitwise | bitwise |
| float64, native and posed: decision inputs (cells, topology, boxes, `diameter_sq`, `coherent`, frame, origin) | bitwise | bitwise | bitwise | bitwise | bitwise |
| float64: node means and measures (`center`, `normal`, `vecs`, `measure`, trees' `node_total_area`) | differ (*) | differ (*) | differ (*) | bitwise | bitwise |
| lists (qi, si raw, n_node), every precision and pose | equal | equal | equal | equal | equal |
| bf16 forest vs float32 forest, both versions | bitwise | bitwise | bitwise | bitwise | bitwise |
| trees; pairs per point (float32, 20k points) | 6; 572.2 | 6; 700.6 | 6; 199.0 | 3; 20.5 | 6; 119.0 |

(*) The two **references** (b40c9494d forced and 0f06e4065, same code path, same tokens, two processes) differ from
each other in exactly these fields too, so in float64 at 33k-cell trees the reference's own CUDA float64 `cumsum`
(`ClusterTree.from_points`' area prefix, `compute_source_aggregates`' prefix) is not reproducible across processes;
the fast build's difference is of the same kind and touches no decision. Its effect on a step is bounded in
section 3 (float64 chunked vs unchunked at 1e-13 or below). In float32 the same sums are cast to float32 and every
field matched.

Forest build + walk time, float32, 200k points (median of 3, GB300): DrivAerML vehicle reference 101 + 85 ms, fast
15 + 10 ms (DTT-POSE's own 99 + 84); whole 110 + 97 vs 17 + 11 (DTT-POSE 98 + 93); AeroZoo 84 + 59 vs 14 + 5;
AirfRANS (181k) 32 + 19 vs 9 + 1.3; lug (9k) 67 + 43 vs 14 + 1.5. Pairs per point at 200k: 571 / 700 / 199 / 20.6 / 119.

## 3. One training step, chunked (b40c9494d) against DTT-POSE's unchunked (job 369730, `out/a/compare_*.md`)

`lhr/job/diag4_probe.py` (integration's diag3_probe, parameters loaded as a plain dict because 0.12.0 refuses their
0.11.0 state-dict metadata), PROFILE's parameters, 2,000 points, chunks of 700, mse; bf16 families of 8 one-ulp
samples per version, samples 0-3 and 4-7 in two cold processes each. Distances are d(run, float64 reference) / group
max, ratios against the reference's bf16 run.

| case | float64: head vs reference (outputs / blocks / max other group) | float32: head vs ref, distance to float64 (blocks) | bf16 blocks ratio, ref family / head family (min-max) | bf16 kernel_basis | bf16 other groups above ref max |
|---|---|---|---|---|---|
| DrivAerML vehicle | 5.1e-15 / 7.1e-14 / 8e-15, loss equal | 1.8e-4 / 1.8e-4 | 0.99-1.01 / 1.00-1.01 | 1.00-1.16 / 1.04-1.16 | none |
| AeroZoo | 6.5e-15 / 3.2e-14 / 2e-15, loss equal | 5.9e-4 / 5.8e-4 | 0.92-1.13 / 0.92-1.13 | 0.96-1.07 / 0.95-1.12 | none |
| AirfRANS | 3.2e-15 / 1.9e-13 / 5e-15, loss equal | 4.0e-5 / 2.4e-5 | 0.90-1.05 / 0.74-1.01 | 1.00-1.47 / 0.94-1.52 | outputs 1.21 vs 1.17 |
| lug | 5.7e-15 / 6.0e-14 / 8e-15, loss equal | 5.4e-5 / 5.4e-5 | 0.99-1.00 / 0.99-1.00 | 0.80-1.00 / 0.82-0.91 | other 1.12 vs 1.10 |

Pair lists: the head's equal the reference's in every precision ("same" in every row). float64 is at roundoff (bound
1e-12: met, worst 1.9e-13). float32: the head's distance to float64 equals or is below the reference's own. bf16: the
families overlap on every group with the head's median at or below the reference's; three group maxima (AirfRANS
kernel_basis 1.52 vs 1.47 and outputs 1.21 vs 1.17, AeroZoo kernel_basis 1.12 vs 1.07, lug other 1.12 vs 1.10) lie
2-5 % above the reference family's maximum of 8, the size of pass 3's chunked-step tail (5e82a9b98 vs 1b25eb899); the
forest is bitwise here, so this is the chunked step, not the forest. Per tensor, 0-8 of ~430 gradient tensors are
above every reference run in every head run, 23-33 below.

## 4. Cost (job 369730 GPU 3 and 1, `out/a/cost`, `cost_table.py`)

The recipe's own `_run_epoch` (`final/lhr/step.py`), bf16, compile_blocks, 3 warm + 15 timed steps, median per run;
3 alternations in rotated order; DrivAerML vehicle and AeroZoo at 200k points, the lug at all points.

| case | DTT-POSE 0f06e4065 (unchunked recipe) | b40c9494d | DTT-FAST 5e82a9b98 (old forest) | b40c9494d vs 5e82a9b98, paired | peak GiB (POSE / new / old) |
|---|---|---|---|---|---|
| DrivAerML vehicle 200k | 1764 [1751, 1798] ms | 1134 [1128, 1140] (-35.7 %) | 1192 [1192, 1194] (-32.5 %) | -57 [-67, -52] ms (-4.8 %) | 177.7 / 104.7 / 106.0 |
| AeroZoo 200k | 1248 [1209, 1254] | 762 [754, 763] (-39.3 %) | 767 [765, 775] (-38.7 %) | -11 [-13, -4] | 155.9 / 124.7 / 125.5 |
| lug (9,048) | 396 [393, 439] | 273 [239, 274] (-39.0 %) | 252 [247, 262] (-37.5 %) | +21, +27, -23 (noise) | 6.8 / 5.4 / 5.5 |

The parts: the forest build and walk are 25 ms of the vehicle step on the fast path against 186 ms on the reference
(section 2); the rest of the difference to DTT-POSE is DTT-FAST's fused read and basis and the chunked recipe. Against
5e82a9b98 only the forest changed; the vehicle's 57 ms is consistent with its fewer pairs (571 per point here;
DTT-POSE measured -17 % per point against the octant forest), not separately profiled.

## 5. Tests

- CPU, whole ISLA directory (job 369663, cpu-short): **421 passed, 133 skipped, 0 failed**. Recipe tests: 2 failed
  (`test_campaign_c_hooks.py::test_initialize_from_checkpoint_*`, `TypeError: int() argument ... 'NoneType'`), the same
  two tests with the same E lines as on 5e82a9b98 (integration job 368338) and DTT-POSE's base (its job 367075).
- GB300, whole ISLA directory with every CUDA test (job 369730 GPU 0): **556 passed, 1 skipped, 5 failed**. The 5 are
  `test_isla_kernels.py::test_compile_blocks_float32_matches_eager_at_every_token_count[12001-12003]` and
  `..._query_independent_float32_at_every_token_count[6001, 6002]`, `assert 2.7e-07 > 5e-06` style; the same 5 tests
  fail with the same assertion on both parents in the same job (5e82a9b98 and 0f06e4065, `tokencount_*.log`); KERNEL-TESTS
  368568 also saw 5 failures on 1b25eb899: pre-existing on both parents, not this branch. Every tree-read test, CUDA included (the new face/box-tie
  test's CUDA case and the bf16 bitwise test on both kernels), passed.

## 6. Left

- Nothing of the brief's tasks; the parent merges DTT-FAST's pass-4 head into this branch later.
- The 5 GPU failures are pre-existing on both parents; they need their own fix outside this lane.

# Re-merge and 0.13.0 (2026-10-24, brief `BRIEF_REMERGE.md`)

Branch `psharpe/isla-dttfast-pose`, head **38a7915cc** (pushed, plain push). Files in `remerge/`; job in `JOBS.md`.

## Commits

- **90874cdc9**: `--no-ff --signoff` merge of `origin/psharpe/isla-dtt-fast` **4d281a702** (DTT-FAST pass 5, which
  contains `origin/psharpe/isla` b4b10d5a9: GEO-BAND's 0.12.0, DEFAULT-DTT, GEO-THINFAST) into b40c9494d, hooks on.
  Conflicts and their resolution:
  - `model.py`: the refusal text and the version comment are psharpe/isla's (GEO-BAND's 0.12.0). `_tree_read_walk`
    takes both `normal_tol` (DTT-POSE) and `walk_points` (DTT-FAST). `_QueryChunks` passes `normal_tol` and
    `walk_points=False`.
  - `test_isla_release.py` and `data/isla_default_v1.mdlus`: psharpe/isla's (`git checkout --theirs`).
  - `tree_build.py`: DTT-FAST's whole-table per-tree fields, over DTT-POSE's `2 d` faces (`n_face`, `by_face`). The
    separation diameter still replaces every tree's diameters after `forest_tables` (`BoundaryForest._build`).
  - `tree_read.py`: DTT-FAST split `point_pairs` into `frame_pairs(points_f)`, and its lazy walk computes
    `query_r[i] @ forest.frame`. That line **auto-merged** and would have walked uncentred coordinates against
    DTT-POSE's centred trees. Resolved: `BoundaryForest.frame_points` is DTT-POSE's formula
    (`frame_coordinates(points.double() - origin, frame).to(points.dtype)`, outside autocast). It feeds `frame_pairs`
    from `point_pairs` and from `tree_read_walk`. `frame_pairs` walks outside autocast, as DTT-POSE's `point_pairs` did.
    So `walk_points_range` and `walk_pair_counts` return what `point_pairs` returns. The `_walk_context` docstring
    now says this.
  - Tests: `test_query_chunks_read_the_forest_with_the_normals_error_bounds` reads the chunked (lazy) walk through
    `walk_points_range`. `test_walk_of_a_range_of_points_...` also asserts that the walk's lists equal the forest's own
    `point_pairs`.
  - Mutation check (local CPU): putting the auto-merged `query_r[i].detach() @ forest.frame` back fails 6 tree-read
    tests, among them the new assertion (cpu-reference, cpu-fast), the float32-pose test, the covariance tests and the
    bf16 test.
- **38a7915cc** `ISLA 0.13.0: ISLA-DTT's boundary forest as a function of the boundary alone`, in 4242962e0's form,
  with 3f3fef69f's texts kept:
  - `__model_checkpoint_version__ = "0.13.0"` and `_version = 13`.
  - The refusal docstring now covers 0.7.0 to 0.12.0 (with 12 for 0.12.0). The message gains "before 0.13.0, ISLA-DTT's
    boundary forest split by sign octants and built under autocast in the caller's precision and uncentred coordinates,
    with the boxes' own diameters and no bound on each token's normal", and keeps every earlier entry.
  - The class comment names what changed in 0.13.0.
  - `test_isla_release.py`: the relabelled-checkpoint test gains "0.12.0", the state-dict test gains ("0.12.0", 12)
    and is renamed `..._0_7_0_to_0_12_0_...`, `_version == 13` is pinned, and the strings say 0.13.0.
  - The `.mdlus` is regenerated by the test's procedure (deleted, then recreated by
    `test_isla_reference_checkpoint_matches_golden`).

## The forest code equals DTT-POSE's (`remerge/forest_code_diff.txt`)

GEO-BAND did not touch `tree_read.py`, `tree_build.py` or `tree_walk.py`: `git diff --stat 1b25eb899 b4b10d5a9` lists
only `model.py`.

- `python3 remerge/forest_code_diff.py 0f06e4065 90874cdc9 physicsnemo/experimental/models/isla/tree_read.py` compares
  every function and method by source (AST). **Identical (29):** `canonical_frame`, `frame_coordinates`,
  `normal_faces`, `normal_error_bound`, `frame_rotation_bound`, `separation_diameter_sq`, `node_means`,
  `tree_pair_invariants`, `BoundaryForest` (fields), `node_cells`, `n_nodes`, the constants, `_KernelBasis`,
  `_chunked_basis` and others.
- **Changed (6):** `BoundaryForest.build` and `_build` (only the `build=` argument and the fast branch before the
  reference body, which is unchanged), `point_pairs` (now `frame_points` + `frame_pairs` + `_frame_pairs`), and
  DTT-FAST's `TreeReadPairs`, `tree_read_pairs` and `_TreeKernelRead.forward`. `remerge/fn_diff.py` prints these
  diffs. `_frame_pairs` is DTT-POSE's `_point_pairs` body with the fast-walk branch and `count_only` added.
- `python3 remerge/added_lines_present.py 0673bff1d 90874cdc9 .../tree_read.py`: 141 of 147 added lines are present.
  The 6 missing are the old `_build(...)` and `_point_pairs(...)` signatures, a docstring line that now continues, and
  the `points_f = frame_coordinates(points.detach().double() - self.origin, self.frame).to(points.dtype)` line, whose
  expression is now `frame_points`' `return` (the same expression, inside the same autocast-disabled block).
- `python3 remerge/added_lines_present.py 40b411df5 90874cdc9 .../model.py` (D5, `normal_tol`): 22 of 23 are present.
  The missing line is the forward's `_tree_read_pairs(tokens, q_tokens, s[..., 0].abs(), g_unit,
  boundary.get("normal_tol", None))`, which on DTT-FAST passes `q_wall` for `s[..., 0].abs()` (as b40c9494d did).
- `diff <(git diff 5e82a9b98 b40c9494d -- model.py | grep '^[-+]') <(git diff 4d281a702 90874cdc9 -- model.py | grep '^[-+]')`:
  DTT-POSE's model.py delta onto DTT-FAST is the same as at b40c9494d. It differs only in the version lines (now
  GEO-BAND's) and the two signatures that gained `walk_points`.

## The checkpoint (`remerge/mdlus_compare.py`, output `remerge/mdlus_compare.txt`)

`python remerge/mdlus_compare.py remerge/isla_default_v1_0.12.0.mdlus test/.../data/isla_default_v1.mdlus` (the 0.12.0
file is a copy of 90874cdc9's):

    members: ['args.json', 'metadata.json', 'model.pt'] same
    args.json: equal
    metadata.json: differing keys [('mdlus_file_version', '0.12.0', '0.13.0')]
    model.pt: 59 vs 59 tensors, names in order same, bitwise-different tensors: 0 []
    model.pt: state-dict metadata entries differing: {'': ({'version': 12}, {'version': 13})}

## Checks on 38a7915cc (jobs 372087 GB300 and 372086 cpu-short; snapshot `$T/code_dttfast_pose_r`)

The snapshot is built by `remerge/lhr/snap/mk_r.sh`: `code_dttfast_integ_p5c` (4d281a702) plus an overlay of
`git diff 4d281a702 38a7915cc -- physicsnemo test pyproject.toml`. Every file is checked against the manifest of
38a7915cc's git archive, with no extra files except tensordict. The recipe is `recipe_dttfast_integ_p5c`, since
38a7915cc changes no recipe file.

| check | result |
|---|---|
| whole ISLA test directory on a GB300 with every CUDA test (372087, nvl72d019-T16, 5 streams as integration p5 c1) | **728 passed, 1 skipped, 0 failed** (3 + 5 + 6 + 183 + 531; base 4d281a702: 709 + 1) |
| ISLA directory on CPU (372086) | **477 passed, 244 skipped, 0 failed** (base 371488: 461 / 241) |
| recipe tests on LHR (372086) | 597 passed, 5 failed: the same 5 tests with the same E lines as the base's 371488 (FAILED list md5 6769338c, E lines md5 98315664 in both). These are the 2 known `test_campaign_c_hooks` TypeErrors and the 3 `isla_volume_vehicle` config tests from the snapshot's extra file (integration pass 5) |
| recipe tests locally on the worktree | **597 passed, 15 skipped, 0 failed** |
| local: `test_isla_tree_read.py`, `test_isla_release.py` | 95 passed, 50 skipped; 32 passed |

**Forest, fast against reference** (372087, `remerge/out/r/forest_table2.md`; probe `remerge/lhr/forest_probe.py`,
which is lhr/job's plus the lazy walk; table by `remerge/lhr/forest_compare2.py`). Five real dumps, 20k points,
native and posed. The head forced to the reference (`build="reference"`, `walk="reference"`) and its fast build and
walk ran in one process, and 0f06e4065 in its own process from the head's tokens. md5s printed: head model.py b1871185,
tree_read.py afcde57e; 0f06e4065 model.py 432b66df, tree_read.py 99aac192. DTT-POSE's own recorded tokens equal the
head's in every case and precision.

| | DrivAerML vehicle | DrivAerML whole | AeroZoo | AirfRANS | lug |
|---|---|---|---|---|---|
| float32, native and posed: every forest field, fast vs head reference vs 0f06e4065 | bitwise | bitwise | bitwise | bitwise | bitwise |
| bf16 autocast, native and posed | bitwise | bitwise | bitwise | bitwise | bitwise |
| float64: every decision field (cells, ranges, children, boxes, `diameter_sq`, `coherent`, frame, origin, levels) | bitwise | bitwise | bitwise | bitwise | bitwise |
| float64: `center`, `normal`, `measure`, (`vecs`), trees' `node_total_area` | differ (*) | differ (*) | differ (*) | bitwise | bitwise |
| lists (qi, si raw, n_node), every precision and pose, all three | equal | equal | equal | equal | equal |
| lazy walk (`walk_points=False`, 3 ranges by `walk_points_range`; `walk_pair_counts`) = `point_pairs`, 12 runs per case | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 |
| bf16 forest vs float32 forest, both versions | bitwise | bitwise | bitwise | bitwise | bitwise |
| trees; pairs per point (float32) | 6; 572.2 | 6; 700.6 | 6; 199.0 | 3; 20.5 | 6; 119.0 |

(*) As in section 2: the two references (the head forced and 0f06e4065) differ from each other in exactly these
float64 fields too. This is CUDA float64 `cumsum` in the node means, and it reaches no decision.

**One training step** (372087, `remerge/out/r/table_*.md`; integration p5's `eq_probe.py`, mse, 2,000 points, pass 2's
parameters with the metadata dropped). The head's chunked step (chunks of 700, the lazy walk) ran against the head's
own unchunked step (`tree_read_kernel` auto), and against its unchunked step with `tree_read_kernel="reference"`, each in
its own process. Columns: the chunked lists raw-equal to the walk of every point; pairs identical in float64 and
float32; float64 max relative difference over the groups; float32 max over the gated groups of (chunked's distance to
the float64 run) / (unchunked's distance to it); the loss ratio, reported but not gated.

| case | raw | pairs | float64 | float32 ratio | loss ratio |
|---|---|---|---|---|---|
| DrivAerML vehicle | True | identical | 7.1e-14 | 1.01 | 1.00 |
| DrivAerML whole | True | identical | 3.1e-13 | 1.14 | 0.98 |
| AeroZoo | True | identical | 3.2e-14 | 1.02 | 0.49 |
| AirfRANS | True | identical | 1.9e-13 | 1.15 | 1.00 |
| lug | True | identical | 6.0e-14 | 1.00 | 1.00 |
| (against unchunked `reference` kernel) | True everywhere | identical everywhere | the same to 2 digits | the same, except AirfRANS 1.22 (`kernel_reads` 3.1e-7 against 2.6e-7) | the same |

float64 is at roundoff (bound 1e-12; worst 3.1e-13). In float32 every ratio is at most 1.22, inside the about-2 bound.
The numbers match DTT-FAST pass 5's on 4d281a702 against b4b10d5a9 (7.1e-14, 3.1e-13, 3.2e-14, 1.9e-13, 6.0e-14).
No step differs, so the bf16 families were not rerun (brief).

GPU-h: 4 x 19:55 = **1.33 of 2**.
