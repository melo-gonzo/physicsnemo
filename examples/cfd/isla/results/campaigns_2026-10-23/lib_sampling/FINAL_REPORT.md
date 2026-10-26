Status: DONE

The branch is built and pushed: ISLA now uses the canonical frame and the systematic sampling from NVIDIA/physicsnemo#2174. In eager mode it is bitwise equal to BASE on real cases of all 8 dataset families, on CPU and on GB300, except in the zero-measure edge, which no real case reaches.

**Branch.** `psharpe/isla-lib-sampling` was pushed to origin with a plain push. `git ls-remote origin psharpe/isla-lib-sampling` was empty before the push and gives 7f7e1c09a after it. It is not merged into psharpe/isla. It contains all of DRAW-EXACT's 5b5e72c2f (merged whole).
- BASE = **5c39a6b52**, built from psharpe/isla 9f279664c plus two `--no-ff --signoff` merges:
  - c2b8cd50f merges psharpe/isla-draw-exact 5b5e72c2f. It had one textual conflict, tests-p4's slow mark in test_isla_sampling.py, kept on its test.
  - 5c39a6b52 merges #2174's 6fc6a8c24 and was clean.
- HEAD = **7f7e1c09a**. Every commit is `-s` with hooks on, and none has a co-author trailer.

| commit | change | eager vs BASE |
|---|---|---|
| a c60621385 | the frame is imported from `physicsnemo.mesh.spatial._frame`; tree_read's copies are deleted; the goldens' TARGETS are retargeted | bitwise, tested on CPU as its own arm |
| b 9acd340ad | the draw uses the library's `pps_inclusion_probabilities` and `systematic_sample` (its scan is `mesh_cumsum`); `isla_cumsum`, `_systematic_positions` and ISLA's pps are deleted; the zero-measure fill is applied per condition; one new edge test | bitwise outside the edge |
| c 3ebf9465b | 7 test ids that only duplicate the library's tests are dropped (DROPPED_TESTS.md) | tests only |
| d 7f7e1c09a | test fix: the planted-hit and run-to-run determinism tests rewritten for the exact int64 line (they failed on BASE) | tests only |

No compatibility shims or re-exports were added. ISLA keeps `draw_inputs`, `water_fill` and the multi-condition composition. Per-file md5s (HEAD and BASE) are in MD5_CHANGED.txt. Golden data, the recipe and the book are unchanged.

**Behaviour change (brought in on purpose, per the 20:33 steer).** The change is confined to one edge: a condition with zero-measure cells whose water-fill share is at least its count of positive-measure cells.
- On HEAD, every positive cell is kept at pi = 1, and the zero-measure cells fill the rest of the share at weight 0.
- On BASE, every cell got pi = 1, the line overran, and cells were dropped.
- The new test `test_zero_measures_fill_a_condition_past_its_positive_cells` fails on BASE's code: positive cells are missed, and the HT sum is 32.94 against a true 46.99 (edge_test_on_base.txt). It passes on HEAD.
- On real data, 0 of 38 cases reach the edge, and no cell of zero measure appears anywhere (drv 6, az 8, af 8, lug 8, t2d 8 cases).

**A/B results** (tables in RESULTS.md; jobs and md5s in JOBS.md; per-pair JSON in results/):
- **Eager, GB300 (393521, 393763):**
  - Draws are bitwise in every family, for example drv 15/15 and az 19/19.
  - Forward, loss and every gradient are bitwise per family: drv 931/931, az 968/968, af 923/923, lug 881/881, t2d 940/940, DrivAerML surface 663/663, HiLift 715/715, SHIFT-SUV 665/665.
- **Eager, CPU (393947, 393764):** HEAD and commit a are each bitwise against BASE.
  - Draws: drv and az at the full 200k budget, plus af, lug and t2d.
  - Forward and gradients: af, lug, t2d and the three surface families.
  - The DrivAerML and AeroZoo forwards do not fit in cpu-short memory (OOM at 180 GB), so they are proven on GB300 only.
- **Compiled** (bf16, compile_blocks, draw under `torch.compile(fullgraph, dynamic)`):
  - Draws are bitwise against BASE, cold and warm, in all families, and the compiled draw equals the eager draw.
  - Forward and gradients are bitwise when both arms share one Inductor cache: af 923/923, az 968/968.
  - With separate cold caches they differ. These are compile draws: the DrivAerML surface family, which has no draw, also differs, and BASE against itself, cold against warm, differs too.
  - The 14:13 aggregated certification was therefore not run (my judgment; DECISIONS_NEEDED.md).
  - DrivAerML's compiled forward hit the probe's 80 GiB chunk cap; only its compiled draws are proven.

**Suites:**
- HEAD, CPU tier 394351: 716 passed, 516 skipped, 5 xfailed, 0 failed.
- HEAD, GB300 tier 394339: 1248 passed, 5 xfailed, 0 failed.
- BASE failed 1 CPU test and 2 GB300 tests. Both are tests-p4 tests written before DRAW-EXACT: one was dropped in c, the other fixed in d.
- Recipe tests on CPU: 626 passed, 19 skipped.
- pre-commit on the changed files and import-linter `--all-files` pass.
- The goldens and the .mdlus pass with no data changed.

**Book.** 3 references break the render: 05-forward-pass.qmd:376 and :471, and 20-limits.qmd:1185. 5 more have stale wording. All are listed with file:line in BOOK_LIST.md; I did not edit the book.

**Compute.** 2.54 of the 2.5 GPU-h (0.04 over). The `^libsamp\.` registry row has been removed, and nothing is running.

**Rulings relied on:** Peter's 20:20 instruction, assume #2174 merges as-is; the supervisor's 20:33 steer; LANE_RULES (separate processes for custom ops, snapshots, the GB300 tier, the 14:13 ruling); the brief's ban on shims.
**For memory:** nothing new.

**Left for the ISLA supervisor:**
- Merge psharpe/isla-lib-sampling in place of 5b5e72c2f.
- Apply the book fixes.

**On disk:**
- Worktree /home/psharpe/gh/physicsnemo-isla-lib-sampling (1.1 GB, clean, pushed). Remove it after the merge.
- This folder (1.2 MB).
- On LHR, $T/kernel/libsamp (47 MB: read-only snapshots, logs, JSON; the raw .pt files are deleted).
