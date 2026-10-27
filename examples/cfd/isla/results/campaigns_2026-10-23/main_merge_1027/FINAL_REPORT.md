# MAIN-MERGE-1027: final report (13:57 EDT 10-27)

Status: DONE. Branch `psharpe/isla-mainmerge-1027` at **12bd38519**. It is one merge commit, made with
`git merge --no-ff --signoff upstream/main`, with parents 6f1e527f1 and aad6ce655. Author Peter Sharpe, signed off,
no AI trailers, hooks run. I pushed it with a plain push. psharpe/isla is still at 6f1e527f1 (checked 13:55), so the
branch fast-forwards it. No PR, no GitHub writes, no messages.

## What the merge brings
- 4025d12ea (#2167): vectorized graph partitioning.
- 96b7cd1c7 (#2191): CI's root-CA change: uv 0.13.5, certifi and virtualenv pins in uv.lock, Dockerfile and workflows.
- aad6ce655 (#2174's squash merge; its tree equals the final head eacbfa7e8 on every #2174 file).

psharpe/isla carried #2174's older head e18c5e6b6. That head is an ancestor of eacbfa7e8, and psharpe/isla had changed
none of #2174's files since. So every #2174 file takes main's final form, and **no hunk from our side is kept** there.
After the merge, physicsnemo/mesh/, docs/api/mesh/ and test/mesh/ equal upstream main.

## Conflict hunks
Each is also explained in the commit message.
1. `CHANGELOG.md` (Added). Our Muon `cuda_graph` entry stood against main's empty side; the base had nothing at that
   spot. I kept both: the Muon entry, then main's `sample_cells_systematic` entry, which both sides share and which now
   appears once. Main's other new entries merged cleanly: #2167 under Changed and Fixed, #2191 under Security.
2. `docs/api/mesh/sampling.rst`. e18c5e6b6's example stood against eacbfa7e8's, which adds `set_cell_measures` and the
   "sample as a mesh" lines. I took main's.
3. `physicsnemo/mesh/sampling/systematic_sampling.py` (add/add). I took main's. It changes four things:
   - `_pps_inclusion_probabilities` takes `amin` in place of `argmax`;
   - `_systematic_sample` clamps its start into [0, 1], with NaN read as 0;
   - `sample_cells_systematic` rejects a Python start outside [0, 1);
   - `sample_cells_systematic` keeps zero-measure cells out of its Morton grid.
4. `test/mesh/sampling/test_systematic_sampling.py` (add/add). I took main's.

`bvh.py` merged cleanly: `_compute_morton_codes` now takes `amin` and `amax`.

## Imports
These resolve on the merged tree, checked in a process with the worktree on PYTHONPATH:
- ISLA's `_pps_inclusion_probabilities`, `_systematic_sample`, `_cumsum`, `_frame.canonical_frame` and
  `frame_coordinates`;
- `bvh._compute_morton_codes`;
- the recipe's `domain_transforms.py`.

## Checks
- **Direct bitwise probe** (`probe_draws.py`, CPU, one process per tree; A is a `git archive` of 6f1e527f1):
  - **4,620 of 4,620** outputs are bitwise.
  - The inputs cover 300 seeded cases each of `_pps_inclusion_probabilities` (ties, zeros, a 30-sigma log spread,
    shares up to 1.5n), `_systematic_sample` (starts 0, 0.5, random and 1-2^-53; with and without order), and
    `_compute_morton_codes` (float32 and float64, isotropic or not, 2-D, constant columns, duplicates, signed zeros).
  - They also cover 60 cases of ISLA's own `sample_boundary_cells` over 1-4 conditions.
- **ISLA goldens A/B on CPU** (`goldens.record` in each tree, then `compare`): **110 of 110 case variants PASS
  bitwise**. Each side's log names its own `systematic_sampling.py`: md5 590c0a50 for A, e1df2988 for B.
- **ISLA tests on CPU, locally** (local_heavy lock, `nice -n 19`, serial; sampling, determinism, goldens and release):
  232 passed, 177 skipped (CUDA), 1 xfailed, rc 0. Log: `local_isla_tests.log`.
- **ISLA GB300 tier on torch 2.13** (job 429301, the whole directory): 8 invocations rc 0, with **1,298 passed, 6
  skipped, 1 xfailed**. These are the same counts as MAIN-MERGE-1026's 414702.
- **Loader-leaf A/B** (job 429231, GB300, torch 2.13, 83 configs, 6f1e527f1 against 12bd38519): **81 BITWISE, 13,172 of
  13,172 leaves**. The 2 drvref baseline rows fail to compose on both sides with the same KeyError, as before. Table:
  `LEAF_AB.md`.
- **Recipe tests on CPU** with no SLURM_* variable set (0 in the environment): **648 passed, 34 skipped**, rc 0. Log:
  `recipe_tests.log`.
- **Library tests on CPU** for the changed files (`test_systematic_sampling`, `test_bvh`, `test_lbvh`, `test_frame`,
  `test_distributed_graph`): 192 passed, 89 skipped, rc 0. Log: `lib_tests.log`.
- **Full book render** (book_render lock): rc 0, 23 HTML files, no `quarto-unresolved-ref` in the log or the HTML, and no
  warning other than the Jupyter-kernel TCP notice. The 05-forward-pass listings now show main's `amin` rank and the
  start clamp. Log: `render_full.log`.
- **Hooks**: `pre-commit run --files` on all 16 changed files passed (ruff, ruff format, interrogate, markdownlint,
  license, large files, import-linter), and `ruff check` on the 5 changed .py files passed.
- **Snapshots** were copied on LHR from existing snapshots plus the changed files, and verified against `git archive`:
  3,229 of 3,229 tracked files for each code snapshot, and 157 of 158 for the recipe. The one recipe difference is
  `dataset_paths.yaml`, which carries the LHR roots by design. The tier's snapshot equals code_b12bd. Details in
  `JOBS.md` and `lhr/snapshot_tars.md5`.

GPU use: **2.73 of 4 GPU-h** (429231: 2.19; 429301: 0.54).

## For the supervisor
- ISLA's `sampling.py` still builds its Morton grid over every cell. Main's `sample_cells_systematic` now spans only
  the cells of positive measure. That is why ISLA's draws stay bitwise. Adopting main's grid would change ISLA's draw
  wherever a boundary has zero-measure cells, so it would be an architecture-level change for Peter. I did not make it.
- uv.lock comes from main (#2191: certifi 2026.10.23 and virtualenv 21.14.13); pyproject.toml is unchanged. The LHR
  t213 venv was not rebuilt; the tests do not depend on these packages.

## Left on disk
- The worktree `/home/psharpe/gh/physicsnemo-isla-mainmerge-1027`. Remove it after the supervisor merges.
- On LHR: `$T/kernel/mm1027` (128 MB, after I deleted the 17 GB of leaf .pt files) and
  `$T/kernel/isla_tests/snap/mm1027_b12bd_t213_gpu` (63 MB).
- `scratch/isla-mainmerge-1027`: small. I deleted the raw probe and golden records and the base archive copy.
- The registry row is added and removed. Nothing is running.
