# VERSION-014 Phase 2, forest option (ii) "neartie" (speculative; 2026-10-27 15:20-16:38 EDT)

Lane VERSION-014-P2 of ISLA Autoresearch (01a13c2c). Peter has not chosen the forest yet: this stages option (ii)
so the rerun can launch early if he approves. Nothing is merged into psharpe/isla or psharpe/isla-014.
Worktree /home/psharpe/gh/physicsnemo-isla-014-v (now on psharpe/isla-014-v-neartie-0140, clean). Jobs: JOBS.md.

## Commits (both pushed, plain push)

| branch | head | parent(s) | content |
|---|---|---|---|
| psharpe/isla-014-v-neartie | 7eab6ad86 | d2159561b + 981ca5aaf | `git merge --no-ff --signoff origin/psharpe/isla-014-neartie` (FOREST-NEARTIE's fix) |
| psharpe/isla-014-v-neartie-0140 | **6e9e2b94e** | 7eab6ad86 | the version commit: `git cherry-pick -n 041dd470e`, version comment extended, fixtures regenerated |

## Hunks
- Merge: no conflict and no hunk needed a choice. d2159561b changes nothing under physicsnemo/experimental/models/isla
  or test/experimental/models/isla relative to 8097ed00f (981ca5aaf's base), so tree_read.py, test_isla_tree_read.py
  and goldens_v1.pt/.json take the neartie side whole (+205/-22, 4 files).
- Version commit: 041dd470e's model.py, test_isla_release.py and both .mdlus, plus one edit. The version comment's
  0.14.0 note said "the separation diameter widened by the coordinates' error bound"; it now reads "... by the
  coordinates' error bound and by the frame's rotation in proportion to each node's own size, never past a
  frame-free ball". The refusal text is unchanged: it lists what came *before* 0.14.0 ("with the boxes' own
  diameters"), which still holds (0.13.0's octant forest used the boxes' own diameters).

## Golden A/B and fixtures (CPU, local, each side in its own process)
- A/B 7eab6ad86 against d2159561b (`ab/run_ab.sh`, BASE = git archive d2159561b, TAG=neartie): **86 of 110 variants
  bitwise; the 24 failing are the 12 G-FOREST forest_* cases (x fast/reference), `out.diameter_sq` only**. The
  FAIL lines are identical, max diffs included, to FOREST-NEARTIE's own A/B against 8097ed00f
  (`ab/ab_neartie.txt` vs `../isla-forest-neartie-2026-10-27/ab/compare.txt`).
- Fixtures regenerated (`record --regenerate isla_default_v1.mdlus isla_dtt_v1.mdlus
  isla_default_checkpoint_output_v1.pth isla_dtt_checkpoint_output_v1.pth`; neartie0140/regen.log), `diff_mdlus.py`:
  - against the parent 7eab6ad86: isla_default_v1.mdlus **57/57** tensors bitwise, isla_dtt_v1.mdlus **84/84**; only
    mdlus_file_version 0.13.0 -> 0.14.0 and state-dict version 13 -> 14 change (neartie0140/diff_vs_parent.txt);
  - against face-0140 041dd470e: 57/57 and 84/84, metadata equal (neartie0140/diff_vs_face0140.txt);
  - both checkpoint-output goldens byte-identical: **the DTT fixture's outputs did not move** with the fix.
- test_isla_release.py + test_isla_tree_read.py locally: 185 passed, 50 skipped, 1 xfailed
  (neartie0140/release_treeread_tests.txt).

## Tiers on 6e9e2b94e (LHR snapshot `$T/kernel/isla_tests/snap/v014_6e9e2b9`)
Built by `snap_on_lhr.sh` as a copy of v014_d215956 plus 8 changed files; MD5SUMS == `git archive 6e9e2b94e
physicsnemo pyproject.toml test` (3,229 files); 138 tensordict 0.15.3 files == base snapshot's.

| tier | job | result | expected (d2159561b + 3 release + 6 neartie) |
|---|---|---|---|
| GB300 whole ISLA dir, torch 2.13 | 431336 (8:43) | 1,327 passed, 6 skipped (mix-order, by design), 1 xfailed, 0 failed | 1,318 + 9 |
| GB300 whole ISLA dir, torch 2.12 | 431341 (8:26) | 1,333 passed, 1 xfailed, 0 failed | 1,324 + 9 |
| CPU full, torch 2.13 | 431342 (9:10) | 784 passed, 533 skipped, 1 xfailed, 0 failed | 775 + 9 |
| recipe tests, local CPU, no SLURM_* set | - | 648 passed, 34 skipped (= d2159561b) | recipe_tests_6e9e2b9.txt |
| pre-commit (ruff v0.17.0 pinned etc.) on the 4 changed .py | - | all pass | |

The shared venv's older ruff 0.15.15 flags one import-order (I001) each in model.py and tree_read.py; both flags are
present on d2159561b too (checked with `git show d2159561b:<f> | ruff check --stdin-filename`).

## Smokes (head 6e9e2b94e against base d2159561b, torch 2.13, bf16, 4 ranks, 100 steps; median iter_time_s of last 50)
Harness `$T/kernel/v014/smoke` (smoke.sbatch 2bfa92da, smoke_wrap.py 6c794225 with the step cap, units.tsv 9ea7af32).
Table: smoke/smokes_last50_neartie.md (+ .json); raw logs smoke/runs/<job> (18 MB).

| family | job | base s / peak GiB | head s / peak GiB | head/base |
|---|---|---|---|---|
| drv | 431343 | 1.350 / 170.16 | 1.353 / 170.14 | 1.002 |
| az | 431343 | 0.898 / 163.06 | 0.897 / 158.83 | 0.998 |
| af | 431596 | 0.413 / 31.89 | 0.419 / 31.86 | 1.016 |
| lug | 431596 | 0.345 / 9.02 | 0.356 / 9.02 | 1.032 |
| ten | 431596 | 0.351 / 4.84 | 0.344 / 4.84 | 0.978 |
| drs | 431597 | 0.374 / 84.81 | 0.374 / 84.81 | 1.001 |
| suv | 431597 | 0.377 / 84.52 | 0.376 / 84.52 | 0.997 |
| hl | 431597 | 0.390 / 81.38 | 0.372 / 81.38 | 0.955 |
| djb | 431597 | 0.177 / 21.69 | 0.177 / 21.69 | 1.001 |
| djbc (compiled) | 431597 | 0.118 / 17.07 | 0.118 / 17.07 | 1.001 (4 graphs, 3 recompiles, 0 breaks, both) |

No family over 1.05, so no flip ran. All runs rc 0, 100/100 steps, no traceback, no non-finite loss. The DTT
families' first losses agree to 3-4 digits (the forest's diameters differ); the surface families are an A/A test and
read 0.955-1.001. AeroZoo's peak is 2.6% lower on head (fewer pairs), the others within 0.03 GiB.

## Compute
GB300: 431336 + 431341 + 431343 + 431596 + 431597 = 104.6 node-min x 4 GPUs = **7.0 GPU-h** (budget 10). CPU: 431342.
Registry row `^isla\.v014\.` was added at 15:25 EDT and removed at 16:37 EDT; nothing is running.

## What stays on disk
- Worktree /home/psharpe/gh/physicsnemo-isla-014-v (about 1.1 GB, shared with the other VERSION-014 branches).
- LHR: snapshot `$T/kernel/isla_tests/snap/v014_6e9e2b9` (63 MB, read-only); smoke output `$T/kernel/v014/smoke/out_{431343,431596,431597}`.
- This folder: neartie0140/ (logs), ab/ab_neartie.txt (the A/B records and base_d215 source copy were deleted), smoke/runs (18 MB).
