Status: DONE

**Outcome.** The fast path now fuses the closest-weights step into Triton, and its outputs are bitwise equal to the reference. Branch `psharpe/isla-geo-thinfast` is pushed (plain push) at `b959a6735` and not merged. The reference torch code is unchanged. Two Triton kernels run on CUDA when Triton is present and the inputs are contiguous; any other input runs the torch code.
- Kernel 1: the whole 3-D `_closest_weights`, that is Ericson's weights, the edge recomputation, the thin-cell guard and the zero-area segments.
- Kernel 2: `_accurate_weights` and the switch, on the guarded pairs only.

**Commits** (`git commit -s`, hooks on), written by lane-writer children:
- `d7585a6c4`: guard and accurate point fused.
- `b65d65351`: whole weights fused.
- `453dc7636`: guarded-pair compaction.
- `1730022cb`: fixes from the review. The fast path runs only on contiguous inputs, and row offsets are int64.
- `d42fc83d5`: pins `_BAND_UNITS == {2: 17, 3: 91}`.
- `dc4b1e484`: widens the 2-D prefilter margin from 32 to 40 units and corrects the 2-D band comment to 15 + 1 + 1.
- `b959a6735`: comment re-wrap.
- Merges: `755cfbb37` (geo-band 3f3fef69f) and `a61a89407` (psharpe/isla b66013630).

**Equality with torch.** Probe P0 (jobs 369877, 369899, 369906) measured how torch's float64 CUDA kernels round on GB300:
- `sum(-1)` is `((x0+x2)+x1)+0`; `sum(1)` is sequential; `norm` uses no fma; `cross` uses an fma.
- `scalar/tensor` is computed as `reciprocal*scalar`.
- Triton writes `-x` as `0-x`, which loses the sign of zero.
- Details are in `RESULTS.md`. The kernel matches these forms with no reference change, so I had no reason to stop.

**Bitwise results** (GB300 job 370582 on snapshot `$T/code_geothinfast1` = b959a6735). Comparisons use int views. They cover every op output, d, u and both gradients, in float32 and float64, forced fast and under auto (which resolved to fast). Every row below gave 0 differences:
- All GEO-THIN and GEO-BAND dumps: DrivAerML ×4, AirfRANS ×5, AeroZoo, lug ×20, Tensile2d. Each also posed, and with column-major queries.
- The adversarial AeroZoo sets and GEO-THIN's 88 adversarial thin-cell sets.
- Totals: 1,240 runs and 16,527,248 queries.
- Per-call weights hook: 46,538,000 pairs.
- FAST-PLUGIN check over the ISLA test directory: 673,850 queries.
- Margin change against psharpe/isla's geometry, each version in its own process: 298 runs.

**Timing** (job 370584, PHASE0's harness, interleaved passes; `RESULTS.md`, `lhr/timing_table.md`):

| case | geometry call, fast: base → head [ms] | reference [ms] | step, base → head [ms] |
|---|---|---|---|
| lug | 74.1 → 37.6 (0.51) | 83.5 | 204 → 166 (0.82) |
| AirfRANS 2-D | 31.7 → 31.6 (1.00) | 33.2 | 356 → 357 |
| AeroZoo | 113.7 → 92.8 (0.82) | 135.9 | 598 → 581 (0.97) |
| DrivAerML vehicle | 384.4 → 345.2 (0.90) | 450.4 | 867 → 829 (0.96) |
| DrivAerML whole boundary | 436.1 → 403.3 (0.92) | 514.3 | 930 → 897 (0.96) |

- The lug's fast call is back at its pre-thin-fix 36.7 ms.
- The fast path now beats the reference by 22–55 % on every 3-D case. AirfRANS (2-D) does not use these kernels.

**Tests:**
- Whole ISLA directory on GB300, every CUDA test (370582, split by file and keyword over 3 GPUs): 620 passed, 1 skipped (environment-gated), 0 failed.
- The same directory, without `test_isla_compile.py`, under the forced plug-in: 605 passed, 1 failed. The failure is `test_a_change_of_condition_subset_compiles_once_per_subset`, a recompile guarded on fast_plugin's own counter, so it is an artifact of the plug-in. It passes under auto in the same job.
- CPU (cpu-short 370585): 413 passed, 200 skipped, 0 failed.

**Review:** a read-only lane-reviewer found a P0 bug: column-major `q` gave wrong weights or an IndexError. It also found int32 offset overflow beyond about 238M pairs. Both are fixed and tested (`REVIEW.md`, `REPORT_WRITER2.md`).

**GPU use:** 4.26 of 6 GPU-h: 3 probes 0.09, the writers' dev jobs 1.41, checks 1.17, timing 1.59. The registry row was added at 13:15 and removed at 15:22.

**For you to decide:**
1. Merging into psharpe/isla.
2. Whether a version bump is needed. I see none: no output, gradient or state dict changes.
3. A pre-existing quirk, unchanged by this branch (review item 3): the reference's own outputs differ at roundoff between contiguous and column-major queries. The fast path falls back to the reference there, so it inherits this.
4. Book changes, with numbers, are drafted in `NOTEBOOK_DRAFT.md`. They cover the 06 cost table and paragraphs, removing the In-flight callout, and the ledger row, and they complete Peter's "remove the documented limitation / change it".

**For project memory:** torch's float64 CUDA forms on GB300 are those listed above. Triton's `-x` is `0 - x`. Python floats meet fp64 tensors exactly in fp64. GEO-THINFAST's fused weights are bitwise equal to the reference (2026-10-24).

**On disk:**
- Nothing is running.
- Worktree `/home/psharpe/gh/physicsnemo-isla-geo-thinfast` (1.1 GB; pushed and clean; remove it after the merge).
- LHR: `$T/code_geothinfast1` (57 MB) and `$T/kernel/geothinfast` (2.3 MB of logs).
- Scratch folder: 2.9 MB.
- Rulings relied on: Peter 2026-10-23 17:08, the supervisor's steers of 13:51 and 14:40, and LANE_RULES.
