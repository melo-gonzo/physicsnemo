# HiLift on torch 2.13: `Cannot transform nonzero effective measure on a zero-measure cell`

RERUN-ENV, 2026-10-26. Code: psharpe/isla bf71190af. The supervisor approved the capture job (cap raised to 16.7 GPU-h)
at about 05:50 EDT.

## The failure
- Config: `model=isla_surface_highlift dataset=highlift_surface` (medium split, deduplicated, with its far field and
  symmetry plane), 4 ranks, bf16, `model.compile_blocks=true`.
- Torch 2.13 failed in 3 of 3 runs (398482, and 399508 twice). Each time rank 3 raised at step 15-17, inside the
  datapipe's transform chain: recipe `nondim.py:324` `scale`, then `geometric.py:726`, then `measure.py:216`. The other
  ranks then hung.
- Torch 2.12 passed in 3 of 3 runs.
- The loader alone (`benchmark_io`, one epoch on each torch, job 399380) never hit it: 0 of 2,752 transfers.

## (1) Minimal repro, and which bits differ
Capture job 400202 ran `lhr/capture.sbatch`, with `lhr/capture_wrap.py` and `lhr/cap_check.py` (log
`logs/rerunenv.hlcapture_400202.log`). It ran one training epoch (128 steps × 4 ranks) for seeds 42 and 43 on both
torches, recording every cell that carries a measure and has zero or near-zero area.
- **The offending cells are exactly degenerate.**
  - Their float32 vertices v0 and v1 are bitwise equal, for example `[-187.0809326171875, -470.2018127441406,
    141.93902587890625]` twice, with the third vertex 0.42 away.
  - Their vertices are bitwise equal under 2.12 and 2.13, at the same transfer of the same mesh: the same cell count and
    the same fingerprint.
  - Re-measured with `compute_cell_areas` (CUDA and CPU, float32 and float64), their area is 0.0 under both torches.
- **Their effective measures are equal and nonzero** under both torches, for example 0.002777. The measure came from the
  cell's area before its vertices collapsed: the measure is about 1,376 times the cached area of 2.0e-6.
- **What differs is not arithmetic.** It is what `Mesh.cell_areas` returns:
  - Torch 2.12 returns the area cached before the collapse: 2.0e-6, carried forward by `translate` and `scale`, which
    keep cached areas.
  - Torch 2.13 recomputes it from the vertices, gets 0, and the check raises.
  - The cell-area arithmetic, the nondim scale and the measure transfer compute the same bits on both torches.
- **Why 2.13 recomputes.**
  - Torch 2.13's `torch/_dynamo/convert_frame.py` (`compile_inner`) holds the process-wide
    `torch.compiler._is_compiling_flag` for a whole compile session (`_compile_session_context`). So
    `torch.compiler.is_compiling()` reads True in every thread while any thread compiles.
  - Probe `lhr/iscompiling_probe.py`, output `logs/iscompiling_probe.txt`: another thread read True 1,541 of 10,716
    times on 2.13 and 0 times on 2.12. `torch.compiler.is_dynamo_compiling()` read 0 times on both, and True inside the
    traced frame on both.
  - #2172's `Mesh._cached` returns `compute()` without reading or writing `_cache` when `is_compiling()` is True. So
    while the training thread compiles, which happens through about step 40 because `compile_blocks` recompiles on new
    shapes, the datapipe's threads drop the geometry cache.
  - A sliver whose float32 vertices later collapse then has no cached area left to carry forward. Its later check reads
    0 against its old nonzero measure.
- **Minimal repro**, CPU only with no data: `lhr/hl_repro.py`, output `logs/hl_repro.txt`.
  - It builds a 1e-5 sliver, sets its measure to its area, and translates it by 1,000, so v0 and v1 collapse.
  - It does that in a second thread while the first thread sits in a `torch.compile` backend, then calls `scale(2.0)`
    after the compile.
  - Torch 2.13 raises the same ValueError. Torch 2.12 does not, and neither torch does when the mesh is built in the main
    thread.

## (2) Can 2.12 hit it?
Over one training epoch, with seeds 42 and 43, 4 ranks each, 516 transfers per rank and 129 of them with measures:

| torch | seed | zero-area cells with a measure (the raise), per rank | collapsed or near-zero cells with a measure (sin ≤ 8 eps, cached area nonzero), per rank |
|---|---|---|---|
| 2.12 | 42 | 0 / 0 / 0 / 0 | 7,152 / 7,102 / 6,981 / 7,067 |
| 2.12 | 43 | 0 / 0 / 0 / 0 | 6,971 / 7,027 / 7,013 / 6,815 |
| 2.13 | 42 | 210 / 64 / 332 / 159 | 6,951 / 7,039 / 6,662 / 6,915 |
| 2.13 | 43 | 387 / 61 / 195 / 324 | 6,601 / 6,969 / 6,831 / 6,500 |

- About 7,000 cells per rank and epoch, about 55 per sample, collapse this way on both torches.
- On 2.12 the cache always carries their area, so the check never fires: 0 of about 56,000 such cells.
- 2.12 would hit it only if some code dropped a mesh's cache between setting the measure and the collapse. That did not
  happen in these two epochs; I did not audit every transform for it.
- The hits on 2.13 are the cells handled in a datapipe thread while the training thread was compiling.

## (3) Where the pieces come from
- **The check** (`measure.py:213-218`, `old_areas == 0` and `measures != 0` raises): `git blame` gives e3dae21a3,
  "Unify complete integration measures for mesh cells and point samples (#1970)". Peter wrote it on 2026-09-28; it is in
  upstream/main and was already in psharpe/isla 9f279664c before the premerge.
  - The HL-DUP HiLift lanes (code_h090 = 75ac723d8, which has the check) trained 500 epochs on 2.12 without hitting it.
- **The cache bypass** (`mesh.py:2710`, `if torch.compiler.is_compiling(): return compute()`, and its twin at
  `measure.py:110` in `set_cell_measures`): `git blame` gives 3cf768033, "Compute Mesh geometric properties without the
  cache under torch.compile". That is **#2172**, merged into psharpe/isla by bf71190af.
- **The trigger**: torch 2.13's process-wide compile-session flag.
- The check is right: in their float32 geometry those cells have zero area. The bug is #2172's use of a process-wide
  predicate for a question about the current frame. Under 2.13 it drops the geometry cache, which is semantic state here
  because it carries areas across transforms, in every thread that runs while another thread compiles.

## (4) Proposed fix: in #2172 (the fork owns it); no library code edited here
`PROPOSED_FIX_2172.diff` (against bf71190af) changes three things:
- `Mesh._cached` asks `torch.compiler.is_dynamo_compiling()` instead of `torch.compiler.is_compiling()`, with a
  docstring line saying why.
- `set_cell_measures` changes the same way.
- New tests in `test/mesh/mesh/test_cache_threads.py`:
  - a mesh in a second thread keeps its `cell_areas` cache while the first thread is inside a compile session;
  - the collapse repro: `scale` after the compile must not raise.

Checked on LHR CPU by `lhr/fixtests.sbatch`, job 400986 (log `logs/fixtests_400986.log`), with the fix applied as a
test-only plugin (`lhr/fix_cached.py`). The suites are test/mesh/mesh/test_compile.py, test/mesh/calculus/test_measure.py
and the new tests:

| torch | fix | result |
|---|---|---|
| 2.12 | no | 55 passed, 12 skipped (CUDA) |
| 2.12 | yes | 55 passed, 12 skipped |
| 2.13 | no | the 2 new tests fail (the same ValueError, and the cache dropped); #2172's own tests pass |
| 2.13 | yes | 55 passed, 12 skipped |

Notes for whoever lands it:
- `is_dynamo_compiling()` is False under non-strict `torch.export`, where `is_compiling()` was True through
  `_is_exporting_flag`. If `_cached` must also bypass the cache under export, use
  `torch.compiler.is_dynamo_compiling() or torch.compiler.is_exporting()`. `is_exporting()` is also process-wide, but it
  is held only during an export.
- The test file needs the SPDX header before it is committed.
- **The same 2.13 behaviour reaches every `torch.compiler.is_compiling()` guard.**
  - bf71190af has 246 in physicsnemo/, 47 of them in mesh, ISLA and datapipes.
  - In datapipe threads during a compile, these guards now skip validation (for example the raise at measure.py:213
    itself), or take compile-only branches.
  - Only `_cached` drops state; the rest are silent. A sweep to a thread-correct helper is a library question for Peter.
- No recipe workaround is proposed, following Peter's rule: fix the library, never patch around it in the recipe.
