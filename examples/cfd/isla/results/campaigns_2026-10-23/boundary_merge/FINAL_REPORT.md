Status: DONE (MERGE-PREP, 2026-10-24 07:30 EDT). Part 1 is on psharpe/isla as 1b25eb899. Part 2 is ready to merge, with
part 1 already merged into it: psharpe/isla-boundary-merge at 2b15490b2, pushed. Its book needs the edits in
`book/BOOK_CHANGES.md` first. Details: `REPORT.md`.

**What ran.** All on LHR; GPU jobs were short QoS with the five bad nodes excluded. 6.98 GPU-h of the cap, which the
supervisor raised from 4 to 9.
- GPU jobs: timing 366130; GPU tests 366327, 366600, 366724 (compile), 367119 (2d).
- cpu-short ISLA suites: 366325, 366326, 367120. cpu-short refused the recipe suites (user submit limit), so I ran them
  locally as short checks.
- Commits on psharpe/isla-align8: 1062d6086 and 17c9df5c0.
- Commits on psharpe/isla-boundary-merge: 6215af9a7, 658289991, f6c8e5429, 8dcfc24f5, e94d91c86 and 2b15490b2.
- All commits used `-s` with hooks on and were pushed with plain pushes. The registry row is removed.

**Results.**
- 1a: e60be94d4 ran `F.pad` (a clone) even when nothing was padded, and its surface gradients were not bitwise.
  1062d6086 makes that path a strict no-op in eager mode. It is bitwise against the base on 20 cases, with the same aten
  compute ops, and adds a test. Files: `probes/bitwise_aligned*.txt`.
- 1b: interleaved timing (`results/TIMING.md`). Where nothing is padded, the fix runs exactly the base's kernels (eager
  GPU time 1.000 on all three cases). Where it pads, GPU time is 0.87-0.92 of the base on the lug and Tensile2d and
  0.96-1.00 on DeepJEB++, and wall time is level or faster. Accepted.
- 1c: the merge of 7213c70c5 had no conflicts. On 17c9df5c0:
  - ISLA directory on CPU: 367 passed, 0 failed.
  - Recipe tests: 523 passed.
  - test_isla_compile on GB300: 14 of 14 passed.
  - Release, geometry and padding files on GB300: all passed.
  - All 293 cells of the ten chapters that run ISLA pass; 3 cells print something new (listings, and roundoff-level
    residuals).
- 2a: four conflicts with ce598a72d.
  - model.py: one docstring hunk; both sides' text kept.
  - The three reformatted test files: a 3-way merge of their ruff-formatted versions. Syntax-tree checks show every
    top-level node is the version of the side that changed it (`merges/resolve_tests.txt`).
  - f6c8e5429 makes a vehicle-only surface forward bitwise 0.11.0's; the supervisor kept it. Interior and surface
    forwards are bitwise those of psharpe/isla on 28 of 28 cases. ISLA stays 0.11.0.
- 2b: 8dcfc24f5 adds the request contract (cover the body; spread falls as one over the square root of the cell count).
- 2d: on 2b15490b2, all with 0 failures:
  - ISLA directory on CPU: 374 passed.
  - Recipe tests: 553 passed.
  - GB300: test_isla_compile 14 passed; release, surface conditions and token padding 45 passed; geometry files 166
    passed.
  - Per-path checks are in `merges/m1..m5_*.tsv`.
- 2c, the book: chapter 06 fails 11 of 32 cells on the merged tree, because its surface demos lack the now-required
  `prediction_boundaries`. Seven one-argument edits fix all 32. Five cells print changed content, listed in
  `book/BOOK_CHANGES.md`. One of them, chapter 20's `para-limits-whole-boundary`, needs a code change: it counts
  SHIFT-SUV under `isla_surface`, so it prints a wrong count.

**What it means.** Both merges keep every forward of the current psharpe/isla bitwise where nothing changes by design. The
padding costs nothing where nothing is padded. I am confident in the bitwise results and the tests. The timing results
are from one case per dataset.

**Rulings relied on.** Peter's whole-boundary ruling (17:08 EDT 10-23); option A for the surface task (23:26 EDT); a
performance fix may cost nothing elsewhere (10-18); LANE_RULES on heavy local work, hooks, and never merging into
psharpe/isla. Supervisor decisions of 05:58 EDT: f6c8e5429 stays; part 2 merges together with a book branch; the cap
rose to 9 GPU-h.

**For project memory.** `F.pad` with zero padding still copies (aten `constant_pad_nd` clones), so a pad path is free only
if it returns its input. Making `prediction_boundaries` required means every surface call site, chapter 06 of the book
included, must pass it.

**Left, for the supervisor.**
- A book lane applies `book/BOOK_CHANGES.md` on a branch off 2b15490b2.
- Decide the SHIFT-SUV gap: `isla_surface` now names tunnel pieces that SHIFT-SUV does not store, so ISLA cannot train
  on SHIFT-SUV with it (an inference from the resolver; not run). A body-only SHIFT-SUV config is one option.
- Whole-model-compile timing of the padding was not measured: its compile exceeded the 720 s guard.
- Nothing is running.

**On disk.**
- Worktrees `/home/psharpe/gh/physicsnemo-isla-align8` (merged; remove when you wish) and
  `/home/psharpe/gh/physicsnemo-isla-boundary-merge` (pushed, about 1.1 GB each).
- LHR snapshots under `$T`: `code_mergeprep_{a8,a8m,bm,bm2,isla7213}` and `recipe_mergeprep_{a8m,bm,bm2}` (about 57 MB
  per code snapshot).
- The scratch folder is 91 MB.
