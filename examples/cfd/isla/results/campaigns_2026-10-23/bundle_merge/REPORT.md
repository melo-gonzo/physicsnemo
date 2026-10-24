# BUNDLE-MERGE report (2026-10-24, 02:20 EDT)

The geometry robustness bundle plus GEO-FAST's seed fix (fb263fa29) is merged with origin/psharpe/isla (d1bb2e887)
and bumped to ISLA 0.11.0 on psharpe/isla-geo-bundle, which is pushed at ce598a72d and tested. It does not include
b30ff5c33 or later. Nothing was merged into psharpe/isla. Since ce598a72d contains d1bb2e887 (still the tip of
psharpe/isla at 02:17), merging it into psharpe/isla is a fast-forward, and `git merge-tree` reports it clean.

## Commits on psharpe/isla-geo-bundle (f13eb5ce0..ce598a72d)

| commit | what |
|---|---|
| fb263fa29 | fast-forward: the search radius seeded by a nearest-cell search on the cell tree (GEO-FAST step 1) |
| 5bd43842c | `git merge --no-ff --signoff origin/psharpe/isla` (d1bb2e887), no conflict |
| ce598a72d | ISLA 0.11.0: the geometry robustness bundle (version bump, `git commit -s`, hooks on) |

The commit ce598a72d was amended once, before the push, to add a paragraph to its message about the formatting. Its
content is unchanged.

## Equality checks of the merge 5bd43842c

- `git diff fb263fa29 5bd43842c -- physicsnemo test/experimental/models/isla` is empty. Every model and test file
  equals fb263fa29's.
- `git diff d1bb2e887 5bd43842c -- examples` is empty. Every book, results and recipe file equals
  origin/psharpe/isla's.
- Exactly 12 paths differ from d1bb2e887, all under physicsnemo/experimental/models/isla/ (geometry.py, model.py,
  sampling.py, tree_read.py) or test/experimental/models/isla/ (_meshes.py, data/isla_dtt_v1.pth,
  data/isla_interior_v1.pth, test_isla.py, test_isla_interior_geometry.py, test_isla_nd.py, test_isla_sampling.py,
  test_isla_tree_read.py).
- 449 paths differ from fb263fa29: 434 under examples/cfd/isla, 14 under examples/cfd/external_aerodynamics, and
  .pre-commit-config.yaml (a markdownlint exclude). None is under physicsnemo/ or test/.
- So every path of the merge equals one of its parents, and rerere played no part.

## Version change (ce598a72d: 3 files)

- physicsnemo/experimental/models/isla/model.py:
  - `__model_checkpoint_version__` goes from "0.9.0" to "0.11.0", and `_version` from 9 to 11.
  - The history comment gains a line for 0.11.0: least-norm tie directions, the winding-number sign of the wall
    distance, the float64 canonical-frame token draw, the on-wall rule, float64 cell normals and the cell-tree seed of
    the search radius, with the same parameters and a new forward. It also says why 0.10.0 is skipped: that number
    belongs to the unmerged head-frame candidate, whose checkpoints exist.
  - The `_version` comment and the docstring of `_refuse_earlier_state_dict` now list 9 (0.9.0) and 10 (the
    candidate).
  - The refusal message adds the clause "before 0.11.0, the geometry without ..." and notes that state-dict version
    10 is the candidate. The check itself is unchanged (`version != ISLA._version`), so it refuses 1, 8, 9 and 10.
    There is no shim, flag or upgrade path.
- test/experimental/models/isla/test_isla_release.py:
  - The reference-checkpoint docstring now says 0.11.0.
  - The relabelled-checkpoint refusal adds the cases "0.9.0" and "0.10.0" and expects "current version 0.11.0".
  - The state-dict refusal test, renamed `test_isla_0_7_0_to_0_10_0_state_dicts_are_refused_on_every_loading_path`,
    adds the cases ("0.9.0", 9) and ("0.10.0", 10). The ("0.9.0", 9) case is the requested test that a torch
    version-9 state dict is refused through `Module.load`, `load_state_dict` and a DDP-style wrapper, with the
    versions named.
  - The ruff-format hook reformatted the whole file to its 88 columns, since the file was stored unformatted. To check
    that this was formatting only, the parent's copy was ruff-formatted and diffed against the new file: they differ
    by the version edits alone.
- test/experimental/models/isla/data/isla_default_v1.mdlus was regenerated (see below).
- Hooks (`pre-commit run --files` on all three files, and again at commit) pass: ruff check, ruff format, license,
  large files and import-linter.

## Regenerated test data

Only `isla_default_v1.mdlus` stores a version. The `.pth` files are plain output dicts or backbone states.

| file | how | md5 before | md5 after |
|---|---|---|---|
| data/isla_default_v1.mdlus | the procedure in test_isla_release.py's docstring: file deleted, then `test_isla_reference_checkpoint_matches_golden[cpu]` run twice (the first run recreated it from `_instantiate_model(seed=7, **CONFIGS["default"]).save()` and failed on purpose; the second passed) | 67091061df402d1ad1e7d18902a6fb2b | 59af2e3f143dbd122190f2ef70c401e6 |

`regen/compare_mdlus.py` (output in regen/compare_mdlus.out) compares the old and new files:

- Both hold args.json, metadata.json and model.pt.
- args.json is unchanged.
- metadata.json differs only in `mdlus_file_version`, from 0.9.0 to 0.11.0.
- All 59 tensors in model.pt are bitwise equal, and the only change in torch's `_metadata` is the root version, from 9
  to 11.

No golden output changed: data/isla_default_checkpoint_output_v1.pth (cc4d7089) is kept and passes against the new
checkpoint on CPU and CUDA. The other data files keep their md5s: isla_backbone_v1 0d6fe827, isla_default_v1.pth
08ab6fa6, isla_dtt_v1 6b8eed5b and isla_interior_v1 860c23e3.

## Tests

The snapshots on LHR are read-only, with MD5SUMS and SNAPSHOT_HEAD files:

- $T/code_bundlemerge: a git archive of ce598a72d (tar md5 b62d0dc3), plus td0152 from code_surfbc.
- $T/recipe_bundlemerge: a git archive of the ce598a72d recipe (tar md5 eda167af), plus the 14 cluster-only dataset
  files of recipe_surfbc2.

In the code snapshot, model.py (1febf1e3), geometry.py (6ba44b1d, the same as GEO-FAST's fb263fa29 snapshot) and the
mdlus (59af2e3f) all equal the worktree's.

| run | where | result |
|---|---|---|
| 364583 isla.bundlemerge.tcpu | LHR cpu-short, 8 CPUs, 13:20 | ISLA directory: **325 passed, 52 skipped, 0 failed**. 51 skips need CUDA; 1 needs ISLA_AF_POSED_SAMPLE. |
| 364583 (same job) | recipe tests | 521 passed, 12 skipped, 2 failed: test_campaign_c_hooks.py::test_initialize_from_checkpoint_{loads_weights_only, rejects_architecture_mismatch} |
| local | recipe tests at ce598a72d, CPU, nice 19 | **523 passed, 12 skipped, 0 failed** (recipe_tests_local.log) |
| 364584 isla.bundlemerge.tgpu | LHR GB300 short QoS, 1 node (nvl72d003-T08), 2:03 | interior geometry 81 passed, 1 skipped (ISLA_AF_POSED_SAMPLE); tree read 42 passed; release 33 passed. No CUDA test skipped. |
| local | test_isla_release.py, CPU | 25 passed |

The 2 recipe failures on LHR come from the environment, not from this branch:

- Both fail with `TypeError: int() argument must be ... not 'NoneType'` at `int(os.environ.get("SLURM_NPROCS"))`.
  `save_checkpoint` calls `DistributedManager.initialize()`, which sees `SLURM_PROCID` inside an sbatch step without
  srun, where `SLURM_NPROCS` is unset.
- Surface-bc's LHR recipe run 362050 failed these 2 tests with the same message.
- physicsnemo/distributed/ and physicsnemo/utils/checkpoint.py are byte-identical in ce598a72d, fb263fa29 and
  d1bb2e887.
- Both tests pass locally at ce598a72d.

GPU-h: 364584 ran 4 GPUs for 2:03, which is **0.14 GPU-h** of the 2 GPU-h cap. The other job (364583) was CPU only.

## Files

This folder holds JOBS.md, REPORT.md, FINAL_REPORT.md, commit_msg.txt, regen/ (compare_mdlus.py and its .out),
recipe_tests_local.log, and lhr/ (the sbatch scripts, build_snapshots.sh, cluster_only.txt, snapshot_tars.md5, the
two job logs and out_gb_364584/).

On LHR, these stay: $T/code_bundlemerge (57 MB), $T/recipe_bundlemerge (1.4 MB) and $T/kernel/bundlemerge (55 KB).
The tars were removed.
