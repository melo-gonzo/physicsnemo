# MERGE-2171-TIDY: final report (23:00 EDT 10-26)

Status: DONE. Branch `psharpe/isla-2171tidy-1026` at **5e98b9e3d**, pushed with a plain push. It fast-forwards
psharpe/isla, which was still at 1a86c09f9 at 23:00. No PR, no GitHub writes, no messages.

## Commit
`5e98b9e3d` Merge #2171's tidied head 867050087 (`git merge --no-ff --signoff`; author Peter Sharpe, Signed-off-by,
no AI trailers). Before merging, `git fetch upstream refs/pull/2171/head` gave FETCH_HEAD = 86705008796313cd0dd8e6632e4b694be781a086.
At 23:00 the PR head was still that SHA, so it has not moved. Between bb83f5d55 and the head there are 2 commits
(52bde3c1e, 867050087), which touch CHANGELOG.md, src/train.py and tests/test_train_helpers.py. **The head changes no
README text**, so the README keeps psharpe/isla's lines on `iter_time_s`, `epoch_seconds` and `peak_memory_gib` as they were.

## Conflict hunks and how each was resolved (also in the commit message)
- CHANGELOG.md, the readback entry: I took the head's wording ("once they reach the host" and the `step_time_s` note).
  Our separate `iter_time_s`/`epoch_seconds`/`peak_memory_gib` entry follows it, unchanged.
- train.py, imports: the head's `ExitStack` and `NamedTuple`, plus our `pathlib.Path`.
- train.py, `log_step`: the head's `log_step(step: _PendingStep)` with its `.item()` comment. It unpacks a new field,
  **`span`**, which is our iteration-time span: the pair of CUDA timing events, or the host-clock float off CUDA.
  - `_PendingStep` is `i, step_dt, mem_gb, lr, span, host_values, landed`, so `*readback` still fills the last two fields.
  - The span's events come before the readback's event on the stream, so they are complete whenever a record is logged.
    The `query()` guard never waits, and no host sync is added.
- train.py, before the loop: I kept our peak-memory reset and `iteration_mark()`, then added the head's `log_landed()`.
  The head's ExitStack callback on `with grad_ctx` runs `log_landed` when a step raises. `log_landed` logs only records
  whose `landed` event has completed, so the error path waits for nothing, iteration times included.
- train.py, the readback: our mark/span lines, then the head's `readback = _start_step_readback(...)` and
  `_PendingStep(i, step_dt, mem_gb, lr, span, *readback)`.
- test_train_helpers.py, imports: the head's `os`, `socket` and `timedelta`, plus our `logging`, `time` and
  `nullcontext`. `ast`, `inspect` and `textwrap` are gone, because the head removed the AST readback test.
- test_train_helpers.py, end of file: the head's `test_step_readback_is_the_all_rank_mean` (2-rank gloo), then our
  `test_float32_matmuls_stay_full_precision`.
- These merged cleanly:
  - ISLA-DTT's chunked step (`_chunked_step`, `chunked_forward_backward`, `_average_gradients_`, the `if not chunked` guards).
  - The summaries' `epoch_seconds` and `peak_memory_gib`.
  - Our `test_iter_time_is_the_whole_iteration`, `test_iter_time_survives_the_epoch_end_drain` and `test_summary_records_epoch_cost`.
  - The head's `test_error_logs_only_the_landed_records` and its fix to the `_InFlight`/`_hold_readbacks` helper.
- No hand prefetch: `grep` finds no `next_batch`, `.load()` or `prefetch` in train.py.
- New test of ours: `test_error_path_records_carry_iter_time` (CPU and CUDA). When a step raises, the landed records
  it logs carry `iter_time_s`, and there is no drain wait.
- Against 1a86c09f9, `git diff --stat` shows 3 files: CHANGELOG.md, src/train.py and tests/test_train_helpers.py.
  Nothing changed under physicsnemo/, test/, the recipe's datasets/ or conf/, or examples/cfd/isla/. **So I ran no
  loader-leaf A/B and no book render, and none was needed.**

## Checks
- Hooks: `pre-commit run --files` on the 3 files passed: ruff format, interrogate, markdownlint, license, large files,
  import-linter. The ruff-check hook skips examples/ by its config, so I also ran `ruff check` and
  `ruff format --check` on both .py files directly: both clean. No `--no-verify`, no SKIP.
- Recipe tests on CPU, locally (CUDA hidden, nice 19, under local_heavy.lock taken with -n): **648 passed, 34 skipped,
  rc 0**. Log: local/recipe_cpu_5e98.log.
- Recipe tests on GPU (LHR job 415489, GB300 nvl72d037-T10, torch 2.13, every SLURM_* variable unset): **686 passed,
  6 skipped, rc 0**.
  - The 6 skips are HiLift data tests that need dataset roots.
  - All 14 CUDA, sync and gloo tests in test_train_helpers passed (junit in smoke_415489/recipe_gpu.xml). Among them:
    `test_train_steps_do_not_sync_with_the_device`, `test_iter_time_is_the_whole_iteration[train/val-cuda]`,
    `test_iter_time_survives_the_epoch_end_drain[cuda]`, `test_summary_records_epoch_cost[cuda]`,
    `test_error_path_records_carry_iter_time[cuda]` and `test_step_readback_is_the_all_rank_mean`.
- Smoke (415489; 4 ranks, torch 2.13, bf16, 20 train and 3 val steps; pre = 1a86c09f9, post = 5e98b9e3d, then the
  first version again as an A/A check). Details in smoke_415489.md.
  - All 6 runs exit rc 0.
  - Lug ISLA-DTT eager (`compile_blocks=false`): every train and val step value, and every summary loss and metric,
    is bitwise across post, pre and post.
  - AeroZoo GeoTransolver: bitwise across pre, post and pre.
  - `iter_time_s` is logged on every train and val step. `epoch_seconds` and `peak_memory_gib` are in both summaries,
    and peak memory is equal across the runs: lug 8.861 GiB train and 1.798 GiB val; GT 77.739 GiB train and 5.408 GiB val.
  - Median `iter_time_s` over steps 5-19, in run order:
    - lug, post/pre/post: 0.365, 0.366 and 0.360 s.
    - GT, pre/post/pre: 0.474, 0.427 and 0.425 s. The first, cold run reads slower.
- GPU use: 0.99 of the 2 GPU-h cap (415489, 14:49 on 4 GPUs).

## #2171's CI state (read-only `gh pr view`, 23:00 EDT)
- The head is still 867050087 and the PR is still a draft.
- Passed: Test with uv (Python 3.12, ubuntu-latest), multi-GPU selection, pre-commit.ci.
- In progress: Testmon and Coverage. I did not wait for them.
- Skipped: Test with pip, multigpu-dynamic and multigpu-static, the coverage publishing jobs, and the multi-GPU report.

## What stays on disk
- The worktree /home/psharpe/gh/physicsnemo-isla-2171tidy-1026, to remove after the supervisor merges.
- scratch/isla-2171tidy-2026-10-26 (small): logs, junit, smoke metrics, scripts.
- I deleted LHR $T/kernel/tidy2171 after writing this report, and removed my registry row. Nothing is running.
