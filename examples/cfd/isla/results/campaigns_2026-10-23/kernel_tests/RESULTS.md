# KERNEL-TESTS results (2026-10-24; logs copied to logs_368568/)

GB300 job 368568 (nvl72d030-T05, 30:15, short QoS, five bad nodes excluded), whole `test/experimental/models/isla/`, one
pytest per GPU, `-rfEs`, each log printing SNAPSHOT_HEAD, MD5SUMS-OK, model.py / test_isla_kernels.py md5 and
`VERSION ran <path> TOKEN_ALIGNMENT <value>` (torch 2.12.0+cu130):

| GPU | code | model.py md5 | test_isla_kernels.py md5 | TOKEN_ALIGNMENT | result |
|---|---|---|---|---|---|
| 0 | head 921472488 | 21a4d086 | c9569ae0 | 32 | 5 failed, 517 passed, 1 skipped (28:47) |
| 1 | 1b25eb899 (padding merge) | 865ff9ac | fd8e36ff | 32 | 5 failed, 469 passed, 1 skipped (28:15) |
| 2 | 7213c70c5 (before padding) | d89d3cd3 | fd8e36ff | None | 469 passed, 1 skipped (21:40) |
| 3 | candidate = commit 396459aec | 21a4d086 | 59f6827a | 32 | 526 passed, 1 skipped (28:34) |

The one skip everywhere: `test_isla_interior_geometry.py:158: set ISLA_AF_POSED_SAMPLE to a forward-kwargs .pt saved by the
posed-inference reproduction` (an opt-in data test, not a CUDA skip).

Failures at the head and at 1b25eb899, all `test_isla_kernels.py`, all the forced-on positive control `assert err > 5e-6`:

| test | head err | 1b25eb899 err | 7213c70c5 |
|---|---|---|---|
| test_compile_blocks_float32_matches_eager_at_every_token_count[12001] | 2.678e-07 | 2.672e-07 | pass |
| ...[12002] | 2.718e-07 | 2.734e-07 | pass |
| ...[12003] | 2.671e-07 | 2.677e-07 | pass |
| test_compile_blocks_query_independent_float32_at_every_token_count[6001] | 1.821e-07 | 1.848e-07 | pass |
| ...[6002] | 1.861e-07 | 1.817e-07 | pass |

Attribution: the token padding (1b25eb899). The test file is identical at 7213c70c5 and 1b25eb899 (md5 fd8e36ff), the
failures appear with the model change, and the assertion messages agree at head and 1b25eb899. With the padding the
compiled blocks see 12032 / 12032 tokens, a multiple of the 4-row tile, so mix-order reduction forced on is harmless
(error at roundoff, 1.8e-7 to 2.7e-7): the padding hid the shape, not a code defect. The fixed-path assertions (match eager,
no mix-order kernel) did not fail.

Fix (commit 396459aec, test file only): both tests set `TOKEN_ALIGNMENT` to 1 by monkeypatch (the construction
test_isla_token_padding.py already uses for its unpadded reference), so the forward's padding is a no-op and the compiled
blocks (`_block_forwards`) receive the unaligned count; a recorder wrapped around `_block_callables()` asserts the count
they receive (`[n]`; `[6000 + nq, 7000]`). Both original assertions are kept with unchanged tolerances. New test
`test_compile_blocks_see_an_aligned_token_count[surface|interior_dtt][cpu|cuda]`: 301 cells (surface) and 301 + 7 queries
(ISLA-DTT) reach the compiled blocks as 320 tokens.

CPU, cpu-short job 368574 (16:18) on the candidate snapshot: 397 passed, 122 skipped (CUDA skips and the
ISLA_AF_POSED_SAMPLE one), 0 failed.
Hooks on the changed file (`pre-commit run --files`, and again at commit): ruff check, ruff format, license,
large files, import-linter passed.
GPU-h: 4 GPUs x 30:15 = 2.02 GPU-h (cap 3).
