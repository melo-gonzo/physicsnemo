Status: DONE

**Commit.** `3f3fef69f` "ISLA 0.12.0: the exact geometry's derived, frame-robust tie and on-wall bands" is on top of 2e9cd7fda. It was made with `git commit -s` and the hooks on; all hooks passed, and the only trailer is Signed-off-by. Pushed with a plain push (2e9cd7fda..3f3fef69f), not merged. `git merge-tree` with origin/psharpe/isla 0d46b2f9b is clean; since eba038f08, that branch has changed only the book and records.
- It touches the same 3 files as 4242962e0: model.py, data/isla_default_v1.mdlus and test_isla_release.py. Nothing comes from DTT-POSE's branch.
- `__model_checkpoint_version__` is "0.12.0" and `_version` is 12. The history comment gains a 0.12.0 line.
- The `_refuse_earlier_state_dict` docstring now reads "0.7.0 to 0.11.0 ... 11 for 0.11.0". Its message gains "before 0.12.0, the exact geometry's first-order tie and on-wall bands and its arithmetic in the caller's frame".
- The check is unchanged (`version != _version`), so every state-dict version before 12 is refused, with the version named.
- test_isla_release.py: the docstring says 0.12.0; the relabelled-checkpoint test gains a "0.11.0" case and expects "current version 0.12.0".
- The state-dict test, renamed `..._0_7_0_to_0_11_0_...`, gains ("0.11.0", 11), pins `_version == 12` and expects "before version 0.12.0".

**Golden check** (`review/golden_check.py`, output `review/golden_check.txt`, old file `review/isla_default_v1.old.mdlus`). I regenerated the checkpoint with the test's own procedure: delete the file, let the test recreate it, run the test again.
- All 59 of 59 tensors are bitwise equal: same dtype and shape, every byte.
- The only differences are `mdlus_file_version` (0.11.0 to 0.12.0) and the root state-dict metadata version (11 to 12). The other 48 metadata entries are equal.
- args.json is byte-identical. The output golden `isla_default_checkpoint_output_v1.pth` is unchanged and still passes. test_isla_release.py passes locally on CPU (27 passed).

**Test-change review** (`REVIEW_TESTS.md`, `review/ast_compare.txt`). I compared `ast.dump` per definition and found no added or removed tests and no changed skips or marks. Every assert expression and assert message is identical. There are six changes, and all six are expected; none weakens a test:
- Five tolerances in thin `_check` (e_all, e_near, e_k), coplanar-cap and the interior float64 off-wall test. Each changes from `(u + 13 u64)(|x| + R)` to the code's new band `u (|x| + R) + C u64 (|x'| + R')`, with C = 91 in 3-D and 17 in 2-D. The `8 u` accuracy bound and the 1/2, 2, 1/4 and 4 factors are unchanged.
- The switch-disable monkeypatch changes from `_FOOT_UNITS = -inf` to `_SWITCH_UNITS = inf`. The threshold is still 1 unit, both forms disable the switch, and the non-vacuity assert stays.
- One note: the tests now read `geometry._BAND_UNITS` where the old ones wrote the literal 13, so nothing pins 91/17 any more.
- test_isla_kernels.py's diff comes from the eba038f08 merge, not from GEO-BAND.

**Suite results.** All ran on LHR from snapshot `$T/code_bandmerge1`, which is code_geoband2 plus the 3 files. Every one of the 3178 files in `git archive 3f3fef69f physicsnemo test pyproject.toml` matches the snapshot by md5. I read every log myself.

| job | suite | result | GEO-BAND |
|---|---|---|---|
| 369879 | GB300 (nvl72d005-T07), whole ISLA directory, every CUDA test | 550 passed, 1 skipped (ISLA_AF_POSED_SAMPLE, same line), 0 failed | 548 / 1 / 0 |
| 369880 | cpu-short, whole ISLA directory | 411 passed, 132 skipped, 0 failed; SKIPPED lines identical | 409 / 132 / 0 |
| 369988 | cpu-short, recipe tests (recipe_mergeprep_bm; the recipe is unchanged since e94d91c86) | 551 passed, 12 skipped, 2 failed (test_campaign_c_hooks); FAILED and E lines identical | 551 / 12 / 2 |

The +2 in both ISLA runs are the two new 0.11.0 refusal cases. GPU use was 2.0 GPU-h (30 min on a 4-GPU node) of the 3 GPU-h cap. Job records are in `JOBS.md`, and the launchers are in `lhr/`: tests_cpu.sbatch 4df4b02f and whole_gb.sbatch ae8e2b02.

**What it means.** psharpe/isla-geo-band at 3f3fef69f is GEO-BAND as ISLA 0.12.0, tested and ready for you to merge. I am confident in this: the only model-visible change is the version, the golden is bitwise, and the counts match GEO-BAND's plus the version tests.

**Rulings relied on:**
- LANE_RULES: hooks on; tests on LHR; the whole directory on a GB300 with the five bad nodes excluded.
- Your brief: GEO-BAND is 0.12.0, and DTT-POSE's forest becomes 0.13.0 if adopted.

**For project memory:** "ISLA 0.12.0 = GEO-BAND's derived bands (3f3fef69f, `_version` 12). DTT-POSE's 4242962e0 also claims 0.12.0; if adopted, redo its version commit as 0.13.0 / 13."

**For you to decide:**
1. Merge psharpe/isla-geo-band (3f3fef69f).
2. Optionally pin `assert geometry._BAND_UNITS == {2: 17, 3: 91}` in test_isla_geometry_band.py. It is a one-line follow-up that I left out to keep the commit to the version change.
3. DTT-POSE's 4242962e0 will conflict with this commit in model.py, the mdlus file and test_isla_release.py. Its version commit must be redone as 0.13.0 rather than merged.

**Left and on disk.** Nothing is running, and the registry row has been removed (backups `ACTIVE_EXPERIMENTS.tsv.bak` and `.bak2`).
- LHR: `$T/code_bandmerge1` (57 MB, read-only) and `$T/kernel/bandmerge` (launchers and logs, small).
- Worktree `/home/psharpe/gh/physicsnemo-isla-geo-band` (about 1.1 GB): clean and pushed. Remove it after the merge.
- This scratch folder holds 1.3 MB.
