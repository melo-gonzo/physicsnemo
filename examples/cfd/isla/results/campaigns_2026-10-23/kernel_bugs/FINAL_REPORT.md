Status: DONE (KERNEL-BUGS, written 11:02 EDT 10-25)

**Branch** `psharpe/isla-kernel-bugs`, worktree `/home/psharpe/gh/physicsnemo-isla-kernel-bugs`, base b841486a5. Pushed with a plain push; head **0d0ce638b**. Not merged. All commits are `-s` with hooks on, author Peter Sharpe, no AI trailers. The diff touches only `physicsnemo/experimental/models/isla/` and its tests: 10 files, +142/−87.

**Fixes** (one lane-writer wrote them; a lane-reviewer approved with nits in REVIEW.md, and I fixed its nits in 0d0ce638b):
- **W3-B2, 14518b280.** The Triton backward now returns `dlogm[p] = Σ_h dz[p,h]`, computed in float32. The by-point kernel does this only when a `LOGM_GRAD` constexpr is set, and the flag is set only when `needs_input_grad[1]`. The pin is now `test_both_segment_softmax_reads_differentiate_the_log_measure_alike`, which checks the 2x rule and that `dker`/`dq` are bitwise with and without the flag.
  - Accuracy on GB300 (job 383338), relative L2 distance to float64: Triton 2.95e-7, eager float32 1.12e-6. So the Triton gradient matches eager to float32 rounding.
  - `fused_kernel_read` is documented as not differentiable in `logm`. It now refuses a `logm` that requires a gradient, with a `ValueError` that names both remedies (0d0ce638b).
- **W1-B1, b1e15f1ec.** The eager read now casts before the gather: `q.to(acc)[qi]`, `k.to(acc)[bj]`, `v.to(acc)[bj]`. Against float64, bf16 grad.v goes from 2.9e-1 to 1.8e-3, grad.q from 2.6e-2 to 2.0e-3, grad.k from 1.6e-2 to 3.2e-3.
  - New CPU test: one bf16 ulp (2^-7). It fails on the base, where dv is off by 0.58.
  - The CUDA-vs-CPU comparison of bf16 grad.k and grad.v, switched off since BUGS_W1, is back on.
- **W2-1, b0412eef3.** `resolve_geo_kernel`'s `auto` now picks the fused region only when CUDA autocast is off or bf16; under fp16 it picks eager. The cache key is (type, index, autocast dtype), and the log line names the dtype. The choice is made from the autocast state before the forward, so it is the same on every rank, and no error is caught. The `[cuda]` strict xfail now passes, and the disqualifier test asserts fp16→eager, bf16→fused, fp16→eager, none→fused.
- **W2-2, b4aa97706.** `_RoutingLogits.logits` normalizes fp16 rows in float32 and casts back. fp32 and bf16 do not reach the branch. The `[cpu]` strict xfail now passes.

**Bitwise evidence** (RESULTS.md):
- CPU A/B harness, each version in its own process (`writer/compare_*.txt`):
  - W3-B2, W2-1, W2-2: 110/110 bitwise.
  - W1-B1: exactly 6 tensors change, grad.q/k/v of `segment_softmax_read_bf16` and `kernel_read_bf16`. Every forward and every fp32/fp64 entry stays bitwise.
- CUDA goldens A/B on GB300 (job 382974): 155/155 PASS.
- Whole-model A/B on GB300 (interior ISLA-DTT case and the padded 200-cell surface model): fp32 and bf16 outputs and every parameter gradient are bitwise.
- Under CUDA fp16, outputs are bitwise but the gradients change, by up to about 1e-3 of their largest entry. W2-2 causes this: the normalize backward now runs in fp32 and rounds once. The base summed two fp16 terms, so the new gradient is the more accurate one. The brief asked only fp32 and bf16 to stay bitwise.

**Regenerated golden:** `goldens_v1.pt` only (md5 487ccbda → 0e98f2be), with the documented `--regenerate goldens_v1.pt`, in the W1-B1 commit with a one-line reason. Only those 6 bf16 gradient entries changed, out of 3,260. `data/` is 771,697 B, under 1 MB.

**Suites** on the final head 0d0ce638b:

| tier | job | passed / skipped / xfailed | rc |
|---|---|---|---|
| full CPU | 383618 | 708 / 472 / 5 | 0 |
| GB300 GPU | 383532 | 1196 / 0 / 5 | 0 |

- The same counts on b4aa97706: jobs 382656 and 382538.
- The 5 remaining xfails are all out of scope: W3-B1 ×2, the GEO-TIEORDER quad diagonals ×2, and the core `Module.load` gap.
- pre-commit passed on the changed files (COLUMNS=250).
- GPU use: 1.83 of the 3 GPU-h cap.

**Timing**, W3-B2 backward on the real DrivAerML body plan at 200k points (135.7M pairs, bf16, GB300, job 382397; median of 11 runs, two processes per side):

| logm | base | branch |
|---|---|---|
| no gradient | 29.6 ms | 29.6 ms |
| requires gradient | 29.6 ms (returns None) | 30.3 ms (+2 %, +1.0 GiB peak) |

Without the flag, `dker` and `dq` are bitwise the base's.

**For you:**
- **A new refusal.** With `measure_weights` that require a gradient, on CUDA, an unsampled interior forward through the fused tree read now raises; before, it dropped that gradient silently. The message names the remedy, `tree_read_kernel="reference"`. If you prefer one contract on every path, detach `log_frac` in `tree_read_walk` instead (maintainer's call; I did not).
- **Review notes, not fixed:**
  - Multithreaded CPU bf16 eager dk/dv now vary from run to run, as fp32 always has: CPU `index_put_` accumulation is not deterministic. Goldens record on one thread, so they are unaffected.
  - blocks.py:315 (`m_s / norm.clamp_min(eps)`) has the same fp16 eps hazard for a slice whose mean normal is exactly zero. Padding cannot cause it.
- **A launcher gotcha for project memory:** `submit.sh`'s `CAP_SINCE` is read in LHR local time (PDT). An EDT value makes sacct fail its check, and the cap then reads 0.
- **Registry:** I added the row `^isla\.kernelbugs\.` and have removed it now that the lane is done.

**On disk:**
- Worktree `/home/psharpe/gh/physicsnemo-isla-kernel-bugs` (about 1.1 GB, clean). Remove it after the merge.
- This folder, 756 KB; I deleted the writer's raw records.
- On LHR: `$T/kernel/kernelbugs/` (the probes) and the snapshots `$T/kernel/isla_tests/snap/kernelbugs-*` (8).
- Nothing is running.
