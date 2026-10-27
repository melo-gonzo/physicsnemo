# MAIN-MERGE-1026: final report (22:19 EDT 10-26)

Status: DONE. Branch `psharpe/isla-mainmerge-1026` at **4100a2553**, pushed with a plain push. It fast-forwards
psharpe/isla, which is still at 604d675e3 (checked 22:17). No PR, no GitHub writes, no messages.

## Commits on top of psharpe/isla 604d675e3
1. `bfaa9bee4` Merge upstream/main (3908973f6): #2169, #2166, #2120, #2033 final, #2172 final, Conformal UQ 1-4.
2. `0a0cb6033` Merge #2171's final head bb83f5d55 (SHA checked after `git fetch upstream refs/pull/2171/head`).
3. `4100a2553` Recipe: event-timed `iter_time_s`, plus `epoch_seconds` and `peak_memory_gib` (our own commit).

## Conflicts and how each was resolved
Merge 1 (each hunk is also explained in the commit message):
- `physicsnemo/datapipes/{mesh_dataset.py, readers/mesh.py, transforms/mesh/transforms.py}` and their tests
  (`test_mesh_readers.py`, `test_mesh_dataset_cache_host.py`, `test_center_mesh.py`): I took main's final #2033 whole.
  Main already makes the drops psharpe/isla made after the 1b1f8a236 head (cache_host_views, use_measure_weighting,
  read_only_memory_maps), so these files held nothing of ours that main lacks. `physicsnemo/datapipes` now equals
  main. Main adds docstrings, `operator.index` validation of `interior_n_points_range` (no recipe config sets it), and
  CenterMesh's `global_data.copy()` in place of `.clone()` (same values). Our later fixes sit outside these files and
  stay as they were: the recipe subsamplers' `_devices_match` (bf71190af, 737f1c48c, in `domain_transforms.py`), and
  #2172/#2176's tracing guards in physicsnemo/mesh and `utils/_tracing.py`, which already match main's #2172.
- `datasets/drivaer_ml_volume.yaml`: psharpe/isla already had `boundary_subsample: cells` from #2120's head. Only the
  comment above that line conflicted. I took main's shorter comment and kept the rest of our file, including the
  `ComputeFreestreamDirection` step and the ground comments. Taking main's whole file would have dropped that step.
  So #2120 changes no input of the DrivAerML Volume baselines: b_gt_drv and b_t_drv are bitwise. `highlift_volume.yaml`
  gains `boundary_subsample: cells` from #2120, and no rerun lane reads that file.

Merge 2:
- `src/train.py`, the loop in `_run_epoch`: main's `for i, batch in enumerate(dataloader)` with
  `recursive_to_device`, as the head has it, plus our chunked ISLA-DTT step (`_chunked_step`,
  `chunked_forward_backward` with `_average_gradients_`) and its `if not chunked` guards on `zero_grad` and `backward`.
  The hand prefetch is gone everywhere, our chunked step included; `grep` finds no `next_batch` or `load()` left. The
  chunked step's values go through the same `_start_step_readback` and `log_step` as the unchunked step's. The head's
  history reverts the old head that psharpe/isla carried (cf4a1eead, 1666e0724, 21340864e). Those parts merged
  without conflict: the old `iter_time_s`, `epoch_seconds` and `peak_memory_gib` and their README and CHANGELOG text.
- `tests/test_train_helpers.py`: I kept the head's imports and readback tests, our
  `test_float32_matmuls_stay_full_precision`, and the `logging` and `nullcontext` imports that our other tests use.
  The old head's tests of the prefetch order, `iter_time_s` and the summary went with the code they tested.
- Commit 3 adds back what the final rerun's pricing reads (`price/parse_runs.py` reads `iter_time_s` and
  `epoch_seconds`):
  - Each step records a CUDA timing event just before its readback starts, so the event comes before the readback's
    own event on the stream.
  - `log_step` reads `elapsed_time` between consecutive marks once the record has landed, so no step syncs. Off CUDA
    it uses the host clock.
  - The summaries carry `epoch_seconds` and `peak_memory_gib` again (the maximum over ranks, after the epoch-end drain).
  - New tests: `test_iter_time_is_the_whole_iteration` (CPU and CUDA, train and val; on CUDA it catches a GPU sleep
    that `step_time_s` misses), `test_iter_time_survives_the_epoch_end_drain`, and `test_summary_records_epoch_cost`.
    The head's `test_train_steps_do_not_sync_with_the_device` still passes on GPU.

## Checks (LHR; jobs and logs in JOBS.md; GPU use 4.71 of 6 GPU-h)
- **Loader-leaf A/B** (LEAF_AB.md): 83 configs against 604d675e3, after merge 1 (414597) and again after commit 3
  (415086). Both runs: 81 configs bitwise, 13,172 of 13,172 leaves equal. The other 2 configs are GT and T on
  `drivaer_ml_volume_reference`, which raise the same KeyError on both sides, as in ISLA-PREMERGE.
- **ISLA test tiers** on bfaa9bee4 (commits 2 and 3 change no file under physicsnemo/ or test/):
  - GB300 on torch 2.13 (414702): 1,298 passed, 6 skipped, 1 xfailed, rc 0.
  - CPU full on 2.13 (414857): 758 passed, 530 skipped, 1 xfailed, rc 0.
  - GB300 on 2.12 (414877): 1,304 passed, 1 xfailed, rc 0.
- **Library suites**, cpu-short on 2.13 (414880; base 604d675e3 in 415029):
  - test/mesh: 4,013 passed. GLOBE: 250 passed.
  - test/datapipes `-k "not domino"`: 823 passed, 1 failed. The failure is `test_mesh_datapipe[cuda]`, known with CUDA
    hidden.
  - test/utils: 929 passed, 31 failed in `test_checkpoint.py`.
  - Recipe: 647 passed, 2 failed (`test_campaign_c_hooks`, a TypeError on SLURM_NPROCS in a batch job).
  - Base 604d675e3 shows the same failing tests with the same messages. The recipe failures come from the Slurm
    environment: the recipe tests pass in full on GPU with the Slurm variables unset (683 passed, 6 skipped, 415080).
  - After commit 3, recipe on CPU (415087): 654 passed, the same 2 failed.
- **#2171 smoke** (415080; 4 ranks, torch 2.13, 20 train and 3 val steps; pre = bfaa9bee4, post = 4100a2553, then the
  first version again as an A/A check; `smoke_415080.md`):
  - All 9 runs exit rc 0, and `iter_time_s` is logged at every step.
  - ISLA-DTT lug, eager (`compile_blocks=false`): every train and val value is bitwise across pre, post and pre.
  - AeroZoo GeoTransolver: bitwise across pre, post and pre.
  - ISLA-DTT lug, compiled (`compile_blocks=true`, as the rerun trains it): not bitwise even pre against pre (cold
    compile draws, at most 3.6e-3 relative). Post against pre is inside that spread (at most 4.9e-3).
  - Median `iter_time_s`, pre then post: GT 0.452/0.449 then 0.425 s; lug compiled 0.284 then 0.278 s.
- **Hooks**: pre-commit passed on every changed file at each commit (ruff, interrogate, markdownlint, license,
  import-linter). No `--no-verify`, no SKIP.
- **Book**: full render of bfaa9bee4 under book_render.lock: rc 0, 23 HTML files, no `quarto-unresolved-ref`.
  Commits 2 and 3 touch no book file or anything the book imports, so I did not render again.

## For the supervisor
- Peak memory: I kept `epoch_seconds` and `peak_memory_gib` (#2171's final head drops them) because pricing reads
  `epoch_seconds`. Drop them from commit 3 if you want strictly the head plus `iter_time_s`.
- `iter_time_s` now times the GPU stream between per-step events. It is no longer the old head's sync-to-sync host
  clock. The medians agree to within the size of the removed per-step sync.
- `step_time_s` stays close to `iter_time_s` under the final head (GT 0.413 against 0.425 s). The host still waits
  somewhere other than the readback, maybe the datapipe or the optimizer. I did not investigate.
- The 31 `test/utils/test_checkpoint.py` failures exist on psharpe/isla 604d675e3 too, with the same messages. They are
  not from this merge, but they deserve a lane if nobody has one.
- On disk: worktree /home/psharpe/gh/physicsnemo-isla-mainmerge-1026 (remove after the merge); LHR $T/kernel/mm1026
  (137 MB after deleting the leaf .pt files); scratch/isla-mainmerge-1026 (small). Registry row removed; nothing runs.
