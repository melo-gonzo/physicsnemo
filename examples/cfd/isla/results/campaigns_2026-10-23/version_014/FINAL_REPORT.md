# VERSION-014: ISLA 0.14.0's version commit

Lane of ISLA Autoresearch (01a13c2c). Worktree /home/psharpe/gh/physicsnemo-isla-014-v, branch psharpe/isla-014-v
(pushed). Jobs: JOBS.md (LHR). Smoke launcher: smoke/JOBS.md.

## Phase 1 (2026-10-27, 13:59-14:45 EDT): DONE

### Commit
- **d2159561b** `Merge psharpe/isla daf1e605b (upstream main aad6ce655: #2167, #2191, #2174's final form) into
  psharpe/isla-014`, `--no-ff --signoff`, on psharpe/isla-014 8097ed00f (HEAD checked).

### Conflict hunks: none
psharpe/isla's only new commits since 6f1e527f1 are the upstream-main merge (12bd38519, daf1e605b). Its 16 files
(.github/*, CHANGELOG.md, Dockerfile, docs/api/mesh/sampling.rst, physicsnemo/mesh/sampling/systematic_sampling.py,
physicsnemo/mesh/spatial/bvh.py, physicsnemo/nn/module/gnn_layers/distributed_graph.py, test/mesh/sampling/
test_systematic_sampling.py, test/models/test_distributed_graph.py, uv.lock) and the 26 files psharpe/isla-014 changes
since 6f1e527f1 are disjoint (`comm -12` empty), so git merged with no conflict and no hunk needed a choice: every
file takes one side whole. ISLA-relevant pieces: `_pps_inclusion_probabilities` takes the first uncapped rank as an
`amin` (was `argmax`), `_systematic_sample` clamps the start (NaN -> 0), `sample_cells_systematic` validates a Python
start; bvh's Morton bounds use `amin/amax` (was `min/max(...).values`). All value-identical; the golden A/B below
exercises them (the B side loads the new systematic_sampling.py, md5 e1df2988).

### Checks
| check | result | where |
|---|---|---|
| golden A/B, CPU, d2159561b against 8097ed00f (each in its own process; base = git archive 8097ed00f) | **110 of 110 PASS bitwise** | ab/ab_m1.txt, ab/run_ab.sh |
| GB300 whole ISLA dir, torch 2.13 (429946) | 1,318 passed, 6 skipped (mix-order controls, by design), 1 xfailed, 0 failed | JOBS.md |
| GB300 whole ISLA dir, torch 2.12 (429948) | 1,324 passed, 1 xfailed, 0 failed | JOBS.md |
| CPU full tier, torch 2.13 (429949) | 775 passed, 533 skipped, 1 xfailed, 0 failed | JOBS.md |
| recipe tests, local CPU, torch 2.13, no SLURM_* variable set | 648 passed, 34 skipped (same as 8097ed00f) | recipe_tests_d215956.txt |
| pre-commit hooks + `ruff check` on the 5 changed .py files | all pass | |

The counts equal 72b624ae4's tiers (ASSEMBLY-014) on both torches.

Snapshot: `$T/kernel/isla_tests/snap/v014_d215956` was built on LHR as a copy of the verified snapshot t213gpu_72b624a
(72b624ae4) plus the 5 changed files (`snap_on_lhr.sh`; no 33 MB VPN upload). Its MD5SUMS equal `git archive
d2159561b physicsnemo pyproject.toml test` file for file (3,229 files), and its 138 tensordict 0.15.3 files equal the
base snapshot's. 2.13 jobs use RERUN-ENV's sbatch copies `$T/kernel/rerunenv/isla_tests/` with
VENV=$T/venvs/isla-rerun-t213 (`submit_on_snap.sh`).

### Smoke harness, ready for Phase 2
`$T/kernel/v014/smoke` (smoke/JOBS.md): AF-PORT-014's eager launcher, base = the d2159561b snapshot, head from env
HEAD_CODE, MERGE-2171-TIDY's islice wrapper, and a `djbc` unit (DeepJEB++ with model.compile_blocks=true). Harness job
430368 (base only, 20 steps): the step cap now holds on AeroZoo's val (3 of 73) and DeepJEB++ (20 of 13,824), eager and
compiled, all rc 0. RUN_TIMEOUT (1080 s) stays as the time bound.

### Compute
GB300: 429946 (8:20), 429948 (7:59), 430368 (14:53) = 2.08 GPU-h. CPU: 429949.

## Pre-staging (supervisor's reply to the Phase 1 report; 2026-10-27, 14:44-15:05 EDT; CPU only)

The supervisor's reply asked for option (iii) on a side branch, the version commit for (i) and for (iii), pushes, a
handoff, and an end to this lane; a fresh lane runs Phase 2 (HANDOFF_PHASE2.md).

| branch | head | parent | content |
|---|---|---|---|
| psharpe/isla-014-v-octant | 123d48c4f | d2159561b | revert of eca42af08: 0.13.0's octant forest, keeping 1909e1cde's guards, the bias removals, name-keyed fills |
| psharpe/isla-014-v-face-0140 | 041dd470e | d2159561b | the version commit, option (i) |
| psharpe/isla-014-v-octant-0140 | 121dcc278 | 123d48c4f | the version commit, option (iii) |

### Option (iii), 123d48c4f
- Conflict hunks of `git revert eca42af08` on d2159561b. Every hunk comes from 1909e1cde's guards on top of eca42af08:
  - tree_build.forest_tables, docstring: the octant wording, with the guard's "cells or nodes past Warp's
    `_WARP_MAX_ELEMENTS`".
  - tree_build.forest_tables, stats: the octant `n_oct` slices. The depth check stays merged with the node-count
    check below them, as 1909e1cde wrote it.
  - test_isla_tree_read.py: 1909e1cde's walk-guard test stays, and the face file's `_KERNELS` lines go.
  - The binary and manifest data were taken from HEAD and then regenerated.
- The result against daf1e605b (0.13.0):
  - model.py is identical.
  - tree_build.py, tree_walk.py and tree_read.py differ only by the guards and cb73af61b's `bias=False`.
  - The tests differ only by the two guard tests, plus cases.py's name-keyed fills and bias removals.
- Goldens were regenerated with `record --regenerate goldens_v1.pt isla_dtt_v1.pth isla_dtt_checkpoint_output_v1.pth
  isla_release_v2.pth` (octant/regen.log). The changed cases:
  - goldens_v1.pt: **36 of 90 cases change**, the same 36 that eca42af08 changed (G-FOREST 14, G-WALK 8, G-READ
    tree_read 4, G-MODEL model_dtt and query_chunks 6, G-PARTS parts_dtt 4).
  - isla_release_v2.pth: dtt_f64, dtt_2d_f64, dtt_draw_f32 and query_chunks change; rich_surface_f32 and surface_f64
    are bitwise.
  - The ISLA-DTT forward and checkpoint-output goldens change.
  - Why: every case that builds a forest changes, and nothing else does.
  - Against daf1e605b's goldens, **G-FOREST and G-WALK (22 cases) are bitwise 0.13.0's**, which confirms that the
    forest is 0.13.0's. The 30 cases that differ from 0.13.0 are exactly the parameter-drawing cases that 83757fb16
    moved (octant/data_diff_vs_daf.txt).
- **A/B against d2159561b: 54 of 110 variants pass bitwise.** The 56 failing variants are exactly the 36 regenerated
  cases (ab/ab_oct.txt, ab/oct_fail_cases.txt).
- CPU full tier on torch 2.13, job 430744: 761 passed, 530 skipped, 1 xfailed. The face forest's tests are gone: the
  file has 8 fewer test functions, so 14 fewer pass and 3 fewer skip.

### The version commit (041dd470e for (i), 121dcc278 for (iii)); script version_commit.py
- model.py:
  - `__model_checkpoint_version__ = "0.14.0"` and `_version = 14`, with the version comment for 0.14.0.
  - The `_version` and `_refuse_earlier_state_dict` docstrings now cover 0.7.0 to 0.13.0 (13 for 0.13.0).
  - The refusal text says "current layout (before 0.14.0, with two more parameters)". It gains, for (i):
    "before 0.14.0, ISLA-DTT's boundary forest split by sign octants and built under autocast in the caller's
    precision and uncentred coordinates, with the boxes' own diameters and no bound on each token's normal, and the
    slice blocks' geo_logit bias and ISLA-DTT's kernel basis's last bias". For (iii) it gains "before 0.14.0, the slice
    blocks' geo_logit bias and ISLA-DTT's kernel basis's last bias".
- test_isla_release.py:
  - `EARLIER_RELEASES` gains `"0.13.0": 13`.
  - The refusal tests expect 0.14.0 and 14.
  - `test_isla_0_7_0_to_0_13_0_...` is renamed, and its docstring now covers the changes before 0.14.0.
- Fixtures: both .mdlus files and both checkpoint-output goldens were regenerated with the record tool.
  - On both branches, **isla_default_v1.mdlus keeps 57/57 tensors bitwise and isla_dtt_v1.mdlus 84/84**.
  - Only `mdlus_file_version` (0.13.0 -> 0.14.0) and the root state-dict version (13 -> 14) change.
  - Both checkpoint-output goldens come out byte-identical (face0140/, octant0140/).
- Checks:
  - test_isla_release.py: 82 passed, 1 xfailed on each branch (83 items, 3 more than before).
  - Golden A/B against its parent: **110/110 bitwise** on each (ab/ab_face0140.txt, ab/ab_oct0140.txt).
  - pre-commit (ruff v0.17.0 pinned) passes. The shared venv's older ruff 0.15.15 flags model.py's import order,
    as it does on d2159561b.
- LHR snapshots v014_041dd47 and v014_121dcc2 are built and verified. No GPU job has run on either.

### Compute, disk
- Total: GB300 2.08 GPU-h (429946, 429948, 430368); CPU-only jobs 429949 and 430744.
- The registry row was removed at 15:05 EDT: nothing is running.
- On disk:
  - the worktree /home/psharpe/gh/physicsnemo-isla-014-v (1.1 GB; Phase 2 uses it);
  - LHR snapshots `$T/kernel/isla_tests/snap/v014_*` (4 x 63 MB) and `$T/kernel/v014/smoke` (launcher plus harness output);
  - this folder (0.3 MB; raw records deleted).
