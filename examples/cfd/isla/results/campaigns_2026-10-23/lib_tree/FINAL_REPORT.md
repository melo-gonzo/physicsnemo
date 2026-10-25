Status: DONE (LIB-TREE, written 01:12 EDT 10-25; P1, P2 and P3 all landed)

**Branch.** Library-only branch `pr/cluster-tree-warp`, based on upstream/main 289e57382, head **42de1669d**, pushed to `origin`. Nothing went to `public` and no PR was opened.
- Commits: 35ca57336 (P1), 1d56696ff (P2), d89f35209 (P3), 42de1669d (float64 rule, CHANGELOG, doc caveats).
- Files changed: mesh/spatial `cluster_tree.py`, `bvh.py`, the new `_lbvh_warp.py` and `_point_walk_warp.py`; `test/mesh/spatial/test_cluster_tree.py`; one CHANGELOG bullet.
- Two lane-writer children wrote the code, one after the other, each the worktree's only writer. Briefs: `WRITER_BRIEF.md`, `WRITER2_BRIEF.md`; reports in `writer/`.
- It merges cleanly into psharpe/isla fc6ed19d0.

**API as built.**
- `ClusterTree.from_points(..., implementation=None)`: `None` means Warp on CUDA, torch otherwise, falling back to torch where Warp cannot build. A forced `"warp"` raises there, and also runs on CPU. `split="midpoint"` refuses `"warp"`.
- `ClusterTree.from_point_groups(points, groups, *, n_groups, leaf_size=1, areas=None, implementation=None) -> (forest, trees)`:
  - `trees[g]` equals `from_points(points[groups == g])`, an empty tree for an empty group;
  - `forest` equals `merge(trees)`, except that `sorted_source_order` indexes `points` and `source_points` is `points`.
- `find_point_interaction_pairs(..., implementation=None, count_only=False)`:
  - both streams are grouped by target, then by root, then in the walk's order; the torch path gets this with one stable sort;
  - `count_only` returns `(n_near, n_far)` per target;
  - `None` means Warp on CUDA in every dtype, float64 included (measured below).
- Warp is imported lazily, and the walk's kernels sit in their own `fuse_fp=False` module.
- Caveat now in the docs: P3's Warp boxes can differ from torch only in the sign of a zero tie (+0.0 against -0.0) and in NaN payloads. torch's own CUDA amin does not fix the sign either.

**Bitwise proofs.**
- **Library tests** (about 700 lines in `test_cluster_tree.py`): Warp against torch on every field and on every walk entry; the grouped build against per-group `from_points` plus `merge`; the torch order against ISLA's per-tree concatenation; ISLA's threshold, fallback and CUDA-stream tests, ported.
- **ISLA equivalence.** ISLA's own code at 5b70c8686 ran in one process (`lhr/isla_side.py`), the library in another (`lhr/lib_side.py`), on 6 real dumps (DrivAerML bf16 and fp32, AeroZoo, AirfRANS and AirfRANS whole in 2-D, lug). Every tree field, the forest table, the near and far streams in order, and the counts are sha256-equal to ISLA's fast build and walk, under both implementations. This held at P2 (job 376887) and again at the head (job 377360).
- **GLOBE.** All `from_points` trees in a forward are sha256-equal to upstream/main's, and the 15-step losses are bit-equal. GLOBE's CUDA forward varies from run to run on its own, so its outputs were compared on CPU: the non-regression configurations with Warp forced are bit-equal to torch (job 377361).

**GLOBE step on a GB300** (DrivAerML, its published config, bf16; `RESULTS.md`):
- 80k cells: 5488 → 5466 ms (−0.4 %).
- 200k cells: 16312 → 16287 ms (−0.15 %).
- `from_points` alone on 80k points: 15.2 → 3.7 ms. Peak memory is unchanged.
- GLOBE builds 2 trees per forward, and its kernel evaluations dominate the step, so the gain is real but small.

**Other measurements.**
- ISLA's DrivAerML forest: the library build takes 6.0 ms against ISLA's fast build at 8.7 ms; the Warp walk at 400k targets takes 23.7 ms against 110 ms for torch.
- float64 walk, torch against Warp, at 100k to 1.6M targets: 38 vs 31 ms up to 557 vs 453 ms, with Warp's peak memory 4.3x lower. Warp is faster at every size, so `None` uses it for float64. LIB-MOVE's 512 vs 583 ms compared different code: ISLA's per-tree reference walk, not this forest walk.
- P3 is about 1 ms faster per build at 80k-1M points (for example 4.6 → 3.4 ms) and uses less memory, so it is adopted.
- Cost of the order contract: the torch walk is 29-31 % slower than upstream's and peaks 17 % higher (85 → 111 ms and 23.3 → 27.4 GiB at 400k targets). It returns the same pairs.

**Suites** (all on LHR; full table in `RESULTS.md`):
- CPU, head: test/mesh/spatial 391 passed / 279 skipped; test/models/globe 258 passed / 65 skipped.
- GB300, head: test/mesh/spatial plus GLOBE, 993 passed, 0 skipped.
- Head merged into psharpe/isla (ISLA directory, mesh/spatial and GLOBE): GB300 1696 passed (base 1582); CPU 1110 passed / 585 skipped (base 1054 / 527); 0 failures.
- pre-commit on the changed files and the import-linter: pass.
- GPU use: about 4.5 GPU-h of the 6 GPU-h cap. Jobs 376887, 377360 and 377545 ran on GB300; 376774, 376888, 377361 and 377542 on cpu-short (`JOBS.md`).

**For you to decide.**
1. Whether this becomes an upstream PR (Peter's call). My recommendation is yes, as is.
2. Whether the slower torch walk is acceptable. It is the CPU default and the fallback. My recommendation is to accept it, because CSR-ready per-target streams are the point of the change. The alternative is to make the order opt-in.
3. A note for the later ISLA switch. ISLA's "reference" build and walk call the library without `implementation=`, so on CUDA they now run Warp too. ISLA's fast-against-reference tests would then compare Warp with Warp unless ISLA passes `implementation="torch"`.

**Rulings relied on.**
- Peter's library direction (c) and `n_spatial_dims`.
- PR #2112's conventions.
- The supervisor's local-run rule of 22:58 and 23:17: local runs under the local_heavy flock, a few minutes at most, and whole suites on LHR. I passed it on to both writers. Nothing new needs to go into memory.

**Still on disk; nothing is running.**
- The worktree `/home/psharpe/gh/physicsnemo-pr-cluster-tree-warp`. Its branch is pushed but not merged; your call.
- This folder, about 1 MB. The raw dumps are deleted.
- On LHR, read-only snapshots `$T/code_libtree_*` (6 × 17 MB) and `$T/kernel/libtree` (24 MB).
- I removed the registry row.
