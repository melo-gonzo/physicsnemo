Status: DONE

psharpe/isla-geo-bundle is pushed at **ce598a72d**. It holds the geometry robustness bundle, GEO-FAST's seed fix
(fb263fa29) and origin/psharpe/isla (d1bb2e887), bumped to ISLA 0.11.0. It does not include b30ff5c33 or later. The
CPU suite and the GB300 files pass. Nothing was merged into psharpe/isla. Since ce598a72d contains d1bb2e887, which was
still the tip of psharpe/isla at 02:17 EDT, the supervisor's merge is a fast-forward (`git merge-tree` reports it clean).
Details are in REPORT.md in this folder.

**Commits** (f13eb5ce0..ce598a72d):
- fb263fa29: fast-forward.
- 5bd43842c: `merge --no-ff --signoff origin/psharpe/isla`, no conflict.
- ce598a72d: "ISLA 0.11.0: the geometry robustness bundle". Committed with `-s` and the hooks on. I amended its message
  once before the push; the content is unchanged.

**Equality checks** of the merge 5bd43842c:
- `git diff fb263fa29 M -- physicsnemo test/experimental/models/isla` is empty, so every model and test file equals
  fb263fa29's.
- `git diff d1bb2e887 M -- examples` is empty, so the book, results and recipe equal origin/psharpe/isla's.
- The 12 paths that differ from d1bb2e887 are all model or test files. The 449 that differ from fb263fa29 are all under
  examples/, plus .pre-commit-config.yaml (equal to d1bb2e887's).

**Version change** (model.py, test_isla_release.py, data/isla_default_v1.mdlus):
- `__model_checkpoint_version__` goes from 0.9.0 to 0.11.0, and `_version` from 9 to 11.
- The history comment gains a 0.11.0 line and a note on why 0.10.0 is skipped.
- The refusal message names the 0.11.0 geometry and the head-frame candidate's version 10. The check is still
  `!= _version`, so it refuses 1, 8, 9 and 10, with no shim.
- Tests: the relabelled-checkpoint refusal adds 0.9.0 and 0.10.0, and the state-dict refusal test adds torch versions
  9 and 10 (renamed `..._0_7_0_to_0_10_0_...`).
- The ruff-format hook reformatted all of test_isla_release.py, which was stored unformatted. Formatting the parent's
  copy and diffing it against the new file leaves only the version edits.

**Regenerated data**: isla_default_v1.mdlus only (md5 67091061 before, 59af2e3f after). I used the procedure in
test_isla_release.py's docstring: delete the file, then run the test twice. Its 59 tensors are bitwise equal to the old
file's, args.json is unchanged, and the only changes are `mdlus_file_version` (0.9.0 to 0.11.0) and the root state-dict
version (9 to 11), as regen/compare_mdlus.out shows. No golden changed: isla_default_checkpoint_output_v1.pth passes
against the new file on CPU and CUDA.

**Tests**:
- LHR cpu-short 364583: the ISLA directory gave **325 passed, 52 skipped, 0 failed**. 51 of the skips need CUDA, and 1
  needs ISLA_AF_POSED_SAMPLE.
- The recipe tests in the same job gave 521 passed, 12 skipped, and 2 failed in test_campaign_c_hooks.py. Both failures
  come from the environment, not the merge. They raise `TypeError` at `int(os.environ.get("SLURM_NPROCS"))` inside
  `save_checkpoint`, because an sbatch step without srun leaves that variable unset. Surface-bc's run 362050 failed
  both with the same message. physicsnemo/distributed/ and checkpoint.py are byte-identical across ce598a72d,
  fb263fa29 and d1bb2e887. Run locally at ce598a72d, the recipe suite gives **523 passed, 12 skipped, 0 failed**.
- GB300 short-QoS job 364584 (nvl72d003-T08): interior geometry 81 passed with 1 skipped (the ISLA_AF_POSED_SAMPLE
  probe), tree read 42 passed, release 33 passed. No CUDA test was skipped.
- Hooks pass on all three changed files.

**GPU-h**: 0.14 (4 GPUs for 2:03), against a cap of 2.

**Rulings I relied on**: Peter's ruling of 2026-10-23 17:08 to adopt the bundle, with TIE-EXP's adopt read, and the
lane rules (no shims, the hooks on, tests on LHR). For project memory: ISLA 0.11.0 is the bundle. The version string
0.10.0 and torch state-dict version 10 stay reserved for the HEAD-NORM candidate, so if HEAD-NORM merges later it
needs 0.12.0 or later.

**Left**: the supervisor merges psharpe/isla-geo-bundle into psharpe/isla. Nothing is still running, and the registry
row has been removed. On disk: on LHR, $T/code_bundlemerge (57 MB), $T/recipe_bundlemerge (1.4 MB) and
$T/kernel/bundlemerge; locally, this folder (132 KB) and the existing worktree /home/psharpe/gh/physicsnemo-isla-geo-bundle.
