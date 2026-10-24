Status: DONE

Turning the flag off costs nothing, but I recommend keeping PyTorch's default. Neither effect it seemed to have is real: both came from a larger source. A cold compile of the bf16 step picks some kernels by timing, so at a fixed flag, fixed inputs and fixed parameters the step's numbers change from one process to the next. That changes the one-ulp families as much as the flag does.

**What ran.** All on LHR. No commits, no branch changed, nothing running now.
- First pass: GPU jobs 368758 (cost, 2.12 GPU-h) and 368799 (effects, 0.90 GPU-h); cpu-short 368749 and 369015.
- Follow-up, under your 0.8 GPU-h grant: GPU jobs 369180 (0.26) and 369325 (0.11); cpu-short 369146, 369254 and 369309.
- **3.39 GPU-h in total** (3.02 in the first pass, the 0.52 over-cap accepted, plus 0.37 of the grant).
- Versions: ISLA-DTT is the head 5e82a9b98, interacting ISLA is 1b25eb899. Snapshots md5-checked (`lhr/md5/`).
- Files: `RESULTS.md` (decision paragraph and sections 1-4), `JOBS.md`, `STATUS.md`, `out/`.

**Cost.** The recipe's bf16 step with compile_blocks, 4 interleaved alternations, median of 20 steps (`out/cost_table.md`).
- Turning the flag off moves step time by -0.53% to +0.09% on the 8 large settings, for both models. Every paired range is within ±1.2%, and peak memory is identical.
- The lug's +2.7% and +4.1% straddle zero and are noise.

**(a) The 2,752-token family** (AirfRANS whole boundary, 100k points, chunked step). Each arm ran in its own process; "cold" means it had its own empty Inductor and Triton caches.
- Flag off, the blocks group's maximum in three processes: 1.53 and 12.44 (cold) and 1.58 (on the flag-on control's warm cache).
- Flag on: 2.90 in two cold processes, 2.34 in a third.
- The reference family (maximum 1.44) comes from one process only, so it understates the reference's own spread.
- The flag-off "cure" of integration job 368662 (maximum 1.53) was one draw. A cold rerun repeated it bitwise, and another gave 12.44.

**Where the process-to-process variation comes from** (profiles `out/out_369180/prof_*.txt`).
- With the flag on, 29 split-K reductions keep bf16 partial sums, among them the 24 slice-anchor products on the 2,752-token axis. With it off there are none, and those products accumulate in float32.
- Two cold flag-off processes differ in one product, a `[2752,512,6]` cuBLAS gemv that one of them pads to 8 columns.
- That is Inductor's `pad_mm` pass, which pads when a `do_bench` timing favours padding and caches the decision. A warm cache therefore replays the earlier choice.
- With `TORCHINDUCTOR_SHAPE_PADDING=0`, two cold processes compile identical kernel lists, yet their losses still differ from the 4th digit, and 0 of 428 gradient tensors are bitwise equal. So a second timing-based choice remains, most likely Triton's autotuning of kernel configurations. I did not test that.

**(b) The reference's blow-up** (ISLA-DTT's unchunked reference, AirfRANS 2k, sample k = 2, flag on).
- It is real and repeats in a second cold process: the losses for samples 0 to 5 are bitwise equal to job 368799's, and `blocks.9.assign.weight` reaches 6.68e5 against 6.65e5 (true maximum 6e-4).
- A third cold process with the same run order shows no blow-up, and its sample-0 loss already differs. Neither do runs in other orders, compiled or eager. Eager runs give the same loss in different processes.
- So it belongs to one compile draw, not to the flag. Its flag-off counterpart in 368799 was a separate process, so a different draw.
- I could not name the first operation: the localization run did not draw the blow-up, and its replay matched the full step bitwise.
- The blow-up's own gradients place it in block 9's backward, running upstream into every earlier block. Nothing is non-finite, and the loss and outputs are normal.
- Likely mechanism, untested: `rel_hat = rel / dist` and `log(dist)` in `_relational_invariants`, with `dist` clamped only at 1e-12. A slice anchor that lands almost exactly on a token gives huge but finite gradients.

**How sure I am.**
- High on the cost, on the variation between processes, and on `pad_mm` being one source of it.
- Moderate on Triton autotuning being the second source, and on the 1/dist mechanism.

**Rulings relied on:** the one-ulp family yardstick (7 samples or more); a scalar loss is reported, not gated; all compute on LHR; the queue limits; the parent's grant of 11:42.

**For project memory, if the supervisor agrees:** the compiled bf16 step is not reproducible across cold processes (`pad_mm` timing, probably autotuning too). A bf16 equivalence verdict needs a reference family spread across processes, or pinned compile choices. Details are under "Ideas for DTT-KNOBS" in `RESULTS.md`; the 1/dist conditioning is an architecture question.

**What is left.** The flag decision is the supervisor's. A process-aware rerun of the 2,752-token comparison would settle the tail.

**On disk.**
- `flag/` (about 3.5 MB).
- On LHR, `$T/kernel/dttfast_flag/` (1.1 GB). Almost all of it is `keep_blowup/`: three records of the blow-up, kept for a later look. Delete them when no longer wanted. The other raw files are deleted.
