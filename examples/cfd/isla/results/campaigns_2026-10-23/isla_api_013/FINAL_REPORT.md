Status: DONE. psharpe/isla-api-013 is ISLA 0.13.0 and is pushed (plain push, not merged). The head is **7c88b2e1f**.

**Commits** (fc6ed19d0..7c88b2e1f, all `-s` with hooks on, 58 files, +1405/-1073):
- 0b3d135d5 (a): the interior task is ISLA-DTT; `query_independent` goes.
- 092ee6a0f (e): `dim` becomes `n_spatial_dims`.
- 21e50b9bd: an early 0.13.0 commit. The writer made it before my steer reached it, so I reverted it in 3b3ff7963.
- df623dfbc: the `live(model)` test helper, and the three draw-under-compile tests restored (TEST-AUDIT items).
- 7c957ac7e: `boundary_tokens` becomes `n_boundary_tokens`, in its own commit.
- 65cce1c2e: merge of origin/psharpe/isla fc6ed19d0. It was clean. The new tests needed no renames, and the strict xfail is untouched and still xfails.
- cbaa9d81e and bf5d2e29e: REYNOLDS-OUT steps 1 and 2. They were written in a side worktree and cherry-picked; that worktree is now removed.
- b64e01981: a recipe test fix (`n_boundary_tokens` in the new transform test).
- **25bea52cb: ISLA 0.13.0.** `_version` is 13, 0.7.0 to 0.12.0 are refused, and the .mdlus golden is regenerated. Its 59 tensors are bitwise the 0.12.0 file's. Only `mdlus_file_version`, the state-dict version and the stored arguments differ (dim→n_spatial_dims, boundary_tokens→n_boundary_tokens, query_independent and reynolds gone).
- 7c88b2e1f: one test-only commit after the version commit. It fixes the audit's P2: the read was not live in `test_a_global_source_adds_no_direction`. The branch was already pushed, so I added the commit rather than reorder history. ISLA 0.13.0 is unchanged.

**Mapping table** (DESIGN.md, plus its addendum):
- Renamed: `dim`→`n_spatial_dims` (constructor, attribute, recipe keys); `boundary_tokens`→`n_boundary_tokens` (your pick).
- Deleted: `query_independent`; `reynolds`.
- Schema keys: `defines` replaces `velocity_scale` and the density/viscosity roles; `velocity_moments` becomes `rms_moments`.
- Kept, because they already match Mesh or GLOBE, or Mesh has no such concept: `boundary_meshes` (your pick), `prediction_boundaries` (your pick), `prediction_points`, `global_data`, `output_schema`/`global_schema`/`boundary_schemas`, `interior`, `hidden`/`n_layers`/`n_slices`/`compile_blocks`, `output_stats`, RankSpec's `rank`/`symmetric`/`parity`, and the `units`/`scale` keys.
- Cases left for Peter: the three name picks, which are each easy to reverse; and the name `defines`, which lives in one constant, `scales.DEFINES`.

**Deleted code (a):**
- `query_independent` (argument, attribute, stored argument);
- `_QueryTokens.routing_mass` and `qt.logw`;
- the interacting branch of `_query_tokens`;
- `_log_count`.

`query_chunks` now refuses surface models. An interior model with rank-2 boundary inputs is now refused at construction, because ISLA-DTT has no rank-2 pair read. The recipe's chunked step keys on `interior` (`_isla_dtt`).

The Reynolds code also went (z, the viscosity role). It is replaced by the recipe's `ComputeBoundaryScales` and float64 `ComputeLogReynolds`, appended only in the AirfRANS and AeroZoo ISLA configs.

**Deleted tests**, one line each with its reason, in DELETED_TESTS.md and DELETED_TESTS_RE.md:
- the interacting golden `isla_interior_v1.pth` and `CONFIGS["interior"]`;
- the form-separation and form-storage tests;
- the routing-mass refinement test;
- the interacting fullgraph-compile and padding `[interior]` cases;
- the conditions `[isla]` case and the scales `[isla-*]` cases;
- the model-side z assertions and the `reynolds` refusals.

The assertion audit (ASSERTION_AUDIT.md) found no tolerance loosened and every deletion recorded. Its P1 and one P2 are fixed. Two refinement contracts now check refinement with the exact read (θ=0); the audit judged this principled. The aligned-padding check for `[interior]` is weaker, with its reason recorded.

**Suites** (logs in logs/lhr; job IDs in JOBS.md):
- ISLA directory on CPU, cpu-short 377644: 453 passed, 240 skipped, 2 xfailed, 0 failed.
- ISLA directory on GB300 with every CUDA test (377645, g0): 698 passed, 1 skipped (an environment-gated sample), 2 xfailed, 0 failed.
- Recipe tests (378081): 613 passed, 2 failed. Both failures are in test_campaign_c_hooks and come from DistributedManager under the SLURM environment. They fail identically on the base (377137).
- Smokes on GB300: 6/6 SMOKE OK. They cover the five interior configs (volume, aerozoo, airfrans with n_spatial_dims 2, lug, tensile2d) and the surface config. Each run asserts that the resolved kwargs have no `query_independent`, `dim`, `boundary_tokens` or `reynolds`, then runs 3 training steps with finite outputs. Volume and surface needed a rerun (378080) because my snapshot overlay had a bad `dataset_paths.yaml`.
- Datapipe check of `re_z` against 5b70c8686's in-model z: AeroZoo 4/4 and AirfRANS 4/4 bitwise, in float64 and float32, over 2 dumps with 3+1 runs of the old side.
- GPU use: 1.25 of the 3 GPU-h cap. Registry row removed; nothing is running.

**BOOK_CHANGES.md**:
- 30 cells break in 05, 06, 09 and 19: 17 root causes (old names `dim`, `boundary_tokens`, `reynolds`, `query_independent`; interacting parameter counts; one `listing` marker) and 13 cascades.
- Two source listings in 05 print differently.
- BOOK_PROSE.md lists 86 prose passages that present the removed API as a default or a choice. The interior standings text stays as it is.

**For you:**
1. Accept 7c88b2e1f, the test-only commit after the version commit, or ask me to redo the history. The 21e50b9bd/3b3ff7963 pair also stays in the history.
2. Peter's picks are pending: `n_boundary_tokens`, `boundary_meshes`, `prediction_boundaries`, and the name `defines`.
3. The interior task now refuses rank-2 boundary inputs; no config uses them.

**On disk:** worktree /home/psharpe/gh/physicsnemo-isla-api-013 (1.1 GB, branch pushed); LHR snapshots `$T/code_islaapi_{092e,65cc,7b60,25be,base}` (read-only).
