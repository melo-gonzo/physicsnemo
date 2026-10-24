Status: DONE

The AirfRANS excess is systematic, the DrivAerML whole-boundary excess is noise, and neither comes from
`query_chunks`' structure.

**What causes it.** The chunked step runs the boundary side on the boundary tokens alone: 1,017 on AirfRANS, an odd
count. At that count, under PyTorch's default `allow_bf16_reduced_precision_reduction=True`, cuBLASLt computes the
slice anchors (`z_pos`, `m_s` of every block, and the frame's anchors) with split-K partial sums rounded to bf16. That
is an extra bf16 rounding point. The unchunked step, at 3,017 tokens, does not have it (profiler, job 367990).

**How it shows.** Over 14 one-ulp parameter samples at 2k points, 6 chunked runs land at 2.1-6.8x the reference's bf16
blocks error, and no reference run goes above 1.00x. At all 180,593 points, 1 of 5 lands at 6.1x. The tail sits in
blocks 0-3's geometry and routing weights.

**Why the structure is exact.** Running the chunked step on the unchunked token tensor reproduces the reference's
loss bitwise. Dropping the leaves (one graph) changes nothing. Point permutations could not decide this, because they
leave the forward bitwise unchanged.

**DrivAerML whole boundary.** Its 200,000 boundary tokens are already a multiple of 32. The reference's own one-ulp
spread reaches 6.6x (mse) and 16.2x (relative_l2) on the loss, and 3.35x on the blocks; every chunked sample lies inside.
At 100k and 200k points (float32 truth, since float64 ran out of memory) every gradient group is at or below 1.6.

**Remedy, prototyped, not committed.** `remedy/remedy_pad_757a5a877.patch` pads `_QueryChunks`' boundary side the way
psharpe/isla 1b25eb899's `forward` pads (`_pad_tokens`, 32), using 1b25eb899's own definitions. Measured:
- AirfRANS 2k: blocks 0.74-1.01 over 14 samples (relative_l2 0.82-1.00 over 4).
- AirfRANS at all points: 0.73-0.98 over 5 samples.
- DrivAerML and the lug: unchanged, as expected.
- Cost: +3 MB peak, step time unchanged.
- Tests: the whole ISLA test directory passes (398 passed, 0 failed). The recipe tests have only the 2 known
  `test_campaign_c_hooks` failures.

Turning `allow_bf16_reduced_precision_reduction` off also removes the tail (0.92-1.00).

I am fairly sure of all this: 14-sample families at 2k, a profiler that names the one distinct operation, and two
independent cures. The open risk is that padding only steers cuBLAS's shape heuristics, so another token count could
hit a bf16-partial reduction again. The flag is the only setting that rules that out everywhere.

**What ran.** All on LHR, short QoS. GPU jobs 367557, 367780, 367888, 367990, 368049: **2.22 of 4 GPU-h**.
cpu-short jobs 367513, 367632, 367633, 367756, 367868, 367879, 368050. No commits and no pushes. I made the read-only
snapshot `$T/code_dttfast_bf16diag_pad`.

**Files.** All in `scratch/isla-dtt-fast-2026-10-23/bf16diag/`: `RESULTS.md` (tables and verdict), `JOBS.md`,
`STATUS.md`, `out/<job>/` (compare tables, logs, profiles), `lhr/` (probes and launchers), `remedy/`.

**Rulings I relied on.** The CHILD_RULES equivalence standard: bf16 is reported beside the reference's own distance,
and moving a rounding point counts as a change. LANE_RULES' compiled-regions ruling: the one-ulp perturbation is the
noise yardstick, and a scalar loss is reported but not gated.

**For project memory, if the parent agrees.** Judge bf16 equivalence against a one-ulp family of the reference (7 or
more samples), not one run.

**Decisions for the parent.**
- Whether to apply the padding patch. It is needed anyway when merging 1b25eb899; after that merge only the
  `__init__` hunk remains. Afterwards, rerun the bf16 equivalence with the one-ulp yardstick.
- Whether to ask the supervisor about setting `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction =
  False` in the recipe. It is program-wide and changes the reference's bf16 numerics at some shapes; it needs a timed
  benchmark first.

**What is left.** Nothing is running. I deleted the raw `.pt` outputs on LHR (about 210 GB) and the local source
copies. About 2 MB stays in this folder and 1.8 MB under `$T/kernel/dttfast_bf16diag`.
