Status: DONE

**What ran.** On LHR, GB300 job 368568 (short QoS, one node, five bad nodes excluded, 30:15). It ran the whole ISLA test directory with
every CUDA test and `-rfEs`, one pytest per GPU and one code version per process, each log printing its version and md5s. The
versions were the head 921472488, 1b25eb899, 7213c70c5, and my candidate. The cpu-short job 368574 (16:18) ran the whole directory
on the candidate. The snapshots `$T/code_kerneltests_{head,1b25,7213,c1}` are git archives verified by md5, and their launchers
are in `lhr/`. Details are in JOBS.md and RESULTS.md, with the logs in `logs_368568/`.

**Failures at the head: exactly the 5 that DTT-FAST reported, and nothing else.** Each fails on the forced-on positive control,
`assert err > 5e-6`:
- test_compile_blocks_float32_matches_eager_at_every_token_count at n = 12001, 12002, 12003: err = 2.68e-7, 2.72e-7, 2.67e-7;
- test_compile_blocks_query_independent_float32_at_every_token_count at nq = 6001, 6002: err = 1.82e-7, 1.86e-7.

At 1b25eb899 the same 5 fail with the same message (2.67e-7 to 2.73e-7, 1.82e-7 and 1.85e-7). At 7213c70c5 all 469 pass.
The test file is identical at 7213c70c5 and 1b25eb899 (md5 fd8e36ff), so the cause is the token padding. It hides the unaligned
count: the compiled blocks see 12032 tokens, a multiple of the 4-row tile, so forcing mix-order reduction on does no harm
there. This is a premise of the tests, not a code defect, and the fixed-path assertions passed throughout.

**Commit.** 396459aec on psharpe/isla-kernel-tests, off origin/psharpe/isla 921472488. It changes only test_isla_kernels.py
(+54 −3), was committed with `git commit -s` and the hooks on, and was pushed with a plain push. Not merged.
- Both guards set `TOKEN_ALIGNMENT` to 1 by monkeypatch. test_isla_token_padding.py already does this for its unpadded
  reference. The forward's padding then does nothing, so the compiled blocks (`ISLA._block_forwards`, built with
  `_BLOCK_COMPILE_OPTIONS`) receive the unaligned count.
- A recorder wrapped around `_block_callables()` asserts the count the blocks receive: `[n]`, and `[6000 + nq, 7000]`.
- Both original assertions stay, with unchanged tolerances: forced on, the bug shows; fixed, the run matches eager and builds no
  mix-order kernel.
- New test `test_compile_blocks_see_an_aligned_token_count`, for the surface and ISLA-DTT tasks, on CPU and CUDA. In a padded
  forward, 301 cells, and 301 cells plus 7 queries, reach the compiled blocks as 320 tokens.
- No test was deleted.

**Tests at my head.** The candidate snapshot carries the committed file byte for byte (md5 59f6827a) on 921472488.
- GB300, every CUDA test: 526 passed, 1 skipped, 0 failed. The head had 522 tests; the 4 new ones make 526.
- CPU, LHR cpu-short: 397 passed, 122 skipped, 0 failed.
- The one GB300 skip is the same in all four versions: test_isla_interior_geometry.py:158, which runs only when
  ISLA_AF_POSED_SAMPLE is set.
- The hooks passed on the changed file (ruff check and format, license, large files, import-linter).

**GPU-h:** 4 GPUs × 30:15 = 2.02 GPU-h, inside the 3 GPU-h cap.

**Deviation:** the GB300 job's limit was 45 minutes, not about 60, so that one 4-GPU node stayed inside the 3 GPU-h cap. It ran 30 minutes.
The first measurement and the head run shared that one job: the candidate on GPU 3 is the commit, checked by md5. That saved a
second GPU job.

**Rulings relied on:** LANE_RULES on tests and GPU tests; Peter's no-hook-bypass rule; nothing to push to GitHub beyond this branch.
New for project memory: a test that needs an unaligned token count inside the compiled blocks sets `TOKEN_ALIGNMENT` to 1 by
monkeypatch and asserts the count the blocks receive.

**Left and still running:** nothing is running, and the registry row ^isla\.kerneltests\. has been removed. The supervisor merges
396459aec. The worktree /home/psharpe/gh/physicsnemo-isla-kernel-tests stays until then. The scratch folder is under 1 MB, and the
LHR snapshots are 4 small read-only code trees.
