# DTT-FAST FLAG: what `allow_bf16_reduced_precision_reduction = False` costs and changes (2026-10-24)

## For the decision-maker

**Recommendation: do not switch the flag off on these numbers; keep PyTorch's default, and fix the bf16 yardstick
first.** Turning `allow_bf16_reduced_precision_reduction` off is free, but nothing measured here shows that it
removes a systematic error. Both effects it seemed to have came from a source this lane found to be larger: a cold
compile of the bf16 step picks some kernels by timing, so the step's rounding changes from one process to the next.

- **Cost: none measurable.** On the recipe's own bf16 training step (compile_blocks), turning the flag off changes the
  median step time by -0.5% to +0.1% on the eight large settings (DrivAerML vehicle and whole boundary at 200k,
  AeroZoo at 200k, AirfRANS at all points; ISLA-DTT and interacting ISLA). Every paired range is within about ±1.2%,
  and peak memory is identical. The lug's +2.7% and +4.1% straddle zero and are noise (section 1).
- **Process-to-process variation (section 4).** At a fixed flag, fixed inputs and fixed parameters, the compiled bf16
  step gives different numbers in different cold processes. Its one-ulp family moves with it, and the move is as
  large as anything the flag does:
  - At the 2,752-token setting (AirfRANS whole boundary, 100k points), the blocks maximum with the flag off is 1.53,
    12.44 and 1.58 in three processes. With the flag on it is 2.90 (two processes) and 2.34 (one).
  - Two sources were seen. Inductor's `pad_mm` pads a product only when a timing benchmark favours it, so two cold
    compiles of one graph can differ. With that turned off, two cold processes compile identical kernel lists and
    still differ, so a second timing-based choice remains, most likely Triton's autotuning of kernel
    configurations (not tested).
  - A warm compile cache replays the earlier choices. That is why some reruns matched to every digit.
- **The chunked step's tail at 2,752 tokens.** The flag-off "cure" of integration job 368662 was one draw of this
  variation: a cold rerun repeated it bitwise, another gave 12.44. The yardstick is also incomplete: the reference's
  own family comes from one process. Neither "the tail is gone with the flag off" nor "the chunked step has a
  systematic tail there" is established.
- **The reference's blow-up** (ISLA-DTT's unchunked reference, AirfRANS 2k, sample k = 2, flag on): block 9's
  `assign.weight` gradient reaches 6.6e5 against a true maximum of 6e-4. It is real and repeats in a second cold
  process (6.68e5, same loss bitwise), but it belongs to one compile draw.
  - It is absent in a third process with the same run order, and absent with compile_blocks on or off in other run
    orders. Its flag-off counterpart in job 368799 was also a different compile draw.
  - So the flag is not shown to cause it. The likely mechanism, not tested, is a small denominator in the block's
    geometry: `rel_hat = rel / dist` and `log(dist)`, with `dist` clamped only at 1e-12, so a slice anchor that lands
    almost exactly on a token gives a huge but finite gradient.
- **The reference's own bf16 error** otherwise moves with the flag in no single direction (section 2). Interacting ISLA
  on AeroZoo is worse with the flag off (blocks max 1.46 to 3.60); several shapes are bitwise unchanged. Each of those
  is one process per arm, so they carry the same caveat.
- **What would settle it:**
  - Judge bf16 equivalence against a family spread over several cold processes as well as over one-ulp
    perturbations, or fix the compile choices first (inductor's `pad_mm` off and a pinned autotune cache).
  - Rerun the 2,752-token comparison that way.
  - Look at the 1/dist conditioning as an architecture question (Ideas, below).

## 1. Cost (job 368758, one node, nvl72d019-T05)

Method: `lhr/step_flag.py`, which is FINAL-TABLE's `step.py` (the recipe's `train._run_epoch` on PROFILE's first-batch
dumps; bf16, compile_blocks, seed-42 init) plus `--rrr`, which sets the flag before anything runs; every run reports
the flag at start and end. Each run has 3 warm-up steps, then 20 timed steps. A setting runs 4 alternations on one GPU,
one process per run, in the order 1 0 | 0 1 | 1 0 | 0 1, so each alternation gives one paired difference from adjacent
runs. ISLA-DTT is 5e82a9b98 (the recipe's chunked step) and interacting ISLA is 1b25eb899. All 80 runs are ok. Table:
`out/cost_table.md` (`cost_table.py out/out_368758`).

| setting | flag on: median of 4 run medians, ms (min-max) | flag off | paired (off - on) / on: median (min to max) | peak GiB |
|---|---|---|---|---|
| ISLA-DTT DrivAerML vehicle 200k | 1165.5 (1165.2-1167.7) | 1165.6 (1162.8-1169.0) | -0.06% (-0.27 to +0.33) | 106.0 both |
| ISLA-DTT DrivAerML whole boundary 200k | 1299.0 (1294.6-1300.1) | 1299.1 (1295.4-1302.5) | +0.04% (-0.07 to +0.24) | 107.3 both |
| ISLA-DTT AeroZoo 200k | 742.8 (736.1-747.5) | 740.7 (739.2-745.6) | +0.09% (-1.00 to +0.42) | 125.5 both |
| ISLA-DTT AirfRANS 180,593 | 651.8 (642.8-656.6) | 649.9 (648.3-653.8) | -0.03% (-0.96 to +0.86) | 26.6 both |
| ISLA-DTT lug (all) | 227.2 (215.3-232.4) | 241.9 (217.8-267.4) | +4.11% (-1.93 to +24.17) | 5.5 both |
| interacting DrivAerML vehicle 200k | 757.7 (757.5-758.4) | 756.9 (755.6-757.8) | -0.11% (-0.25 to -0.07) | 126.8 both |
| interacting DrivAerML whole boundary 200k | 823.1 (821.6-823.7) | 823.1 (821.6-824.3) | +0.02% (-0.03 to +0.07) | 126.8 both |
| interacting AeroZoo 200k | 589.1 (587.0-593.2) | 586.8 (584.3-591.3) | -0.53% (-1.21 to +0.74) | 125.9 both |
| interacting AirfRANS 180,593 | 705.8 (705.0-706.5) | 704.8 (703.7-705.2) | -0.12% (-0.40 to -0.03) | 60.7 both |
| interacting lug (all) | 137.9 (132.5-143.3) | 141.4 (136.1-151.7) | +2.65% (-0.90 to +9.55) | 5.9 both |

Within-run spreads (the interquartile range of the 20 steps) are 2-10 ms on the large settings and 4-74 ms on the lug
(`out/cost_table.md`). The step times agree within 2% with FINAL-TABLE's 368334 on another node: ISLA-DTT 1,164 / 1,305 / 741 /
640 ms and interacting 760 / 821 / 589 / 707 ms.

Reproducibility, from the same runs: each run has the same inputs, seeds and initial parameters, so its 23-step mean
loss is a determinism check.
- Interacting ISLA gives one value per arm on each large setting, bitwise over 4 runs. The flag moves that value on
  all four large settings, by up to 6e-4 relative (DrivAerML whole boundary: 0.0822096 on, 0.0821577 off).
- On the lug, interacting ISLA gives the same two values in both arms, so there the flag changes nothing.
- ISLA-DTT gives 4 distinct values per arm on every setting, under both flags. Its training step is not
  run-to-run deterministic over 23 steps, whatever the flag.

## 2. Effect on the reference itself (job 368799)

Method: `lhr/flag_probe.py`, integration pass 3's copy of BF16-DIAG's `diag3_probe.py` with `DIAG_COMPILE=1`, so
compile_blocks is on as in training. It adds `DIAG_LABEL`, `DIAG_QI=0` (interacting ISLA, which loads the ISLA-DTT
state non-strictly: 1 key stays from the seed-42 init and 126 go unused) and records which flag was set. Every float32
parameter is moved one ulp (k = 0..6, where 0 is the stored parameters), at 2,000 points with the huber loss (the
recipe's). Each version and flag runs in its own process.
- ISLA-DTT's reference is the head 5e82a9b98 with `tree_read_kernel="reference"` on `forward_pass` (unchunked).
- Interacting ISLA is 1b25eb899.

The truth is each reference's float64 step. Ratio = the distance to the truth over the **flag-on** reference's
unperturbed bf16 distance. Each cell gives min / median / max over 7 samples. Tables: `out/out_368799/compare_*.md`
(`lhr/flag_compare.py`).

| reference, case | flag | loss | outputs | blocks | kernel_reads | kernel_basis | other |
|---|---|---|---|---|---|---|---|
| ISLA-DTT, AirfRANS (1,017 cells) | on | 1.00 / 1.22 / 1.37 | 0.83 / 1.00 / 1.12 | 0.71 / 1.00 / **1.2e8** | 0.97 / 0.97 / 1.00 | 0.79 / 0.88 / 1.05 | 0.92 / 0.97 / **1.2e7** |
| | off | 1.06 / 1.25 / 1.56 | 0.81 / 0.89 / 1.11 | 0.66 / 0.78 / 0.80 | 0.97 / 0.97 / 1.00 | 0.73 / 0.76 / 0.99 | 0.96 / 0.97 / 1.03 |
| ISLA-DTT, AeroZoo | on = off, bitwise | 0.81 / 1.00 / 1.09 | 0.85 / 0.99 / 1.04 | 1.00 / 1.21 / 1.43 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.95 / 0.98 / 1.01 |
| interacting, AirfRANS | on | 0.87 / 0.98 / 1.14 | 0.92 / 1.00 / 1.02 | 1.00 / 1.20 / 2.40 | | | 0.91 / 0.98 / 1.09 |
| | off | 0.65 / 0.86 / 0.93 | 0.89 / 1.02 / 1.09 | 0.98 / 1.31 / 1.84 | | | 0.75 / 0.89 / 1.06 |
| interacting, AeroZoo | on | 0.45 / 1.00 / 2.17 | 0.79 / 0.89 / 1.04 | 1.00 / 1.27 / 1.46 | | | 0.79 / 0.95 / 1.09 |
| | off | 0.36 / 0.80 / 2.43 | 0.73 / 0.80 / 1.01 | 1.03 / **1.78 / 3.60** | | | 0.82 / 0.96 / 1.13 |

The reference families of section 3 extend this table:
- AirfRANS whole boundary at 2k: on 0.20 / 0.41 / 1.52, off 0.32 / 0.88 / 1.33 in blocks.
- AirfRANS at 100k, AirfRANS whole boundary at 100k and DrivAerML whole boundary at 2k: on and off are bitwise or
  nearly equal.

Per tensor, no gradient tensor of interacting ISLA has every run of one flag above or below every run of the other (0
of 302 on AirfRANS, 0 of 307 on AeroZoo). For the ISLA-DTT reference on AirfRANS, 2 of 427 are above with the flag on.

**The blow-up** (`out/job_369015.log`, `lhr/tail_check.py`): ISLA-DTT reference, AirfRANS 2k, flag on, sample k = 2.
- Loss 0.95084 against a true 0.95013, and outputs within 1.9e-2 of the truth, as for every other sample.
- No gradient is non-finite. `blocks.9.assign.weight` has max |gradient| 6.6e5, against a true max of 6.1e-4.
- The excess runs upstream: `blocks.0-5.broadcast.weight` 3-9e4, `embed.1.weight` 7.3e4, the seed 6.3e4. The later
  blocks are untouched. So it arises in block 9's backward and propagates.
- The same sample with the flag off is normal (max 6.3e-3 on `head.weight`).
- Sample k = 4 (5.4x) sits in `blocks.2.assign.weight`: 2.9e-2 against a true 5.5e-4.
- BF16-DIAG's and integration's eager families (compile_blocks off, mse) never showed this. It appears here under
  compile_blocks with huber, and one sample is all there is.

## 3. Effect on the chunked step (job 368799)

The head 5e82a9b98 runs the recipe's chunked step under `"auto"` (chunks of 700 at 2k, one chunk of 100,000 at 100k)
against the reference's unchunked step; the reference and settings are those of section 2. The truth is the
reference's float64 step at 2k and the head's float64 chunked step at 100k. Pairs were md5-equal to the reference's in
every run. The table gives the blocks group (min / median / max over 7). The other groups stay within the reference's
spread except where noted.

| case (boundary cells; padded token count) | ref on | ref off | head on | head off | head tensors above every ref run (on / off) |
|---|---|---|---|---|---|
| AirfRANS, 1,017 (1,024), 2k | 0.71 / 1.00 / 1.2e8 | 0.66 / 0.78 / 0.80 | 0.71 / 0.79 / 0.81 | bitwise = head on | 3 / 1 of 427 |
| AirfRANS, 1,017, 100k | 0.56 / 1.18 / 2.20 | bitwise = on | 1.12 / 1.18 / **2.90** | bitwise = head on | 4 / 4 of 427 |
| AirfRANS whole, 2,744 (2,752), 2k | 0.20 / 0.41 / 1.52 | 0.32 / 0.88 / 1.33 | 0.21 / 0.32 / 0.60 | 0.22 / 0.51 / 0.73 | 0 / 1 of 428 |
| AirfRANS whole, 2,744 (2,752), 100k | 0.52 / 0.87 / 1.44 | 0.53 / 0.87 / 1.44 | 0.77 / 1.32 / **2.90** | 0.76 / 1.16 / **3.69** | 1 / 0 of 428 |
| DrivAerML whole, 200,000, 2k | 1.00 / 1.38 / 8.20 | bitwise = on | 1.34 / 1.38 / 2.85 | bitwise = head on | 0 / 0 of 431 |

AirfRANS whole at 100k, per sample (blocks):
- Head, flag on: 2.22, 0.89, 0.84, 0.77, 2.90, 1.32, 2.65. These equal integration g1 (368565) to every printed digit.
- Head, flag off: 0.93, 2.32, 1.16, 1.58, 0.76, 1.16, 3.69.
- Integration g2 (368662), flag off, same probe, state and setting: 0.90, 0.58, 0.96, 1.53, 1.15, 0.81, 0.50.
- The g2 and present flag-off losses differ in the 4th digit for every sample (for example k = 0: 0.7076281 against
  0.7071413).
- The tail tensors are `blocks.{1,2,3,6,7}.assign.weight` under both flags (`out/job_369015.log`).

Section 4 settles the irreproducibility. The flag-off result depends on the compile draw, not on the flag, and the
flag-on result does too (job 369180's cold flag-on control gives 0.44 / 0.81 / 2.34).

The whole-boundary scalar loss is ill-conditioned: the reference's unperturbed loss error is 1.3e-5, so the loss
ratios reach 150. It is reported in the tables, not gated.

## 4. Follow-up: compile draws and the blow-up (jobs 369180, 369325; tables 369254)

Granted by the parent at 11:42 EDT, up to 0.8 GPU-h; used 0.37. Every process had its own cold, process-private
Inductor, Triton and Warp caches, except the one warm arm named below.

**(a) The 2,752-token family** (head 5e82a9b98, chunked, AirfRANS whole boundary, 100k points, huber, compile_blocks;
the 7 one-ulp samples; the reference family and the truth are job 368799's). Probe: `lhr/flag2_probe.py`, which is
`flag_probe.py` plus a profiler on sample 6 and Inductor's cache counters. Tables: `out/out_369254/compare_afw_*.md`.

| arm | process and caches | blocks min / median / max | tensors above every reference run | loss k = 0 |
|---|---|---|---|---|
| 368565 (integration g1), flag on | cold | 0.77 / 1.32 / 2.90 | 1 of 428 | 0.70863235 |
| 368799, flag on | cold | 0.77 / 1.32 / 2.90 (bitwise g1's losses) | 1 | 0.70863235 |
| 369180 on, flag on | cold | 0.44 / 0.81 / 2.34 | 0 | 0.70812547 |
| 368662 (integration g2), flag off | cold | 0.50 / 0.90 / 1.53 | 0 | 0.70762813 |
| 369180 offA, flag off | cold | 0.50 / 0.90 / 1.53 (bitwise g2's losses) | 0 | 0.70762813 |
| 369180 offB, flag off | cold | 0.64 / 1.31 / **12.44** | 2 | 0.70799816 |
| 369180 offW, flag off | warm from 369180 on | 0.45 / 0.65 / 1.58 | 0 | 0.70785385 |
| 368799, flag off | warm from 368799's flag-on run | 0.76 / 1.16 / 3.69 | 0 | 0.70714128 |
| reference family (368799, one process), flag on | | 0.52 / 0.87 / 1.44 | - | |

Kernels used by the token-axis bf16 products. These are the profiles `out/out_369180/prof_*.txt` of sample 6; the
comparison counts every mm and bmm with its shapes and kernels.
- Flag on: 29 cuBLASLt split-K reductions keep bf16 partial sums (`splitKreduce_kernel<.., __nv_bfloat16, ..>`), among
  them the 24 slice-anchor products `[512, 2752] x [2752, 2]` and the boundary-side `[512, 2744] x [2744, 6]`.
- Flag off: none. The same products run `cutlass_80_*tensorop_s16816gemm*` (float32 accumulation and partials); the
  shapes and call counts are otherwise identical (1,062 calls, 497 of them on the 2,752-token axis).
- offA against offB, both flag off and cold: one product differs. 24 calls of `bmm [2752, 1, 512] x [2752, 512, 6]` run
  a cuBLAS gemv in offA, while offB pads it to `[.., 512, 8]` and runs an `nvjet` kernel. Four compiled graphs and
  their Triton kernels differ with it.
- That is Inductor's `pad_mm` pass: `torch/_inductor/fx_passes/pad_mm.py` pads when `ori_time > pad_time * multiplier`
  in a `do_bench` timing and caches the decision in the Inductor cache. So the decision is timing-dependent per cold
  process and replayed from a warm cache.
- offA against offW (offW on a warm cache): identical kernel lists.

With `TORCHINDUCTOR_SHAPE_PADDING=0` (job 369325, two cold flag-off processes, nopadA and nopadB):
- The kernel lists are identical: no mm, bmm or kernel name or count differs (`out/out_369325/prof_nopad*`).
- The results still differ. Losses differ from the 4th digit, for example k = 0 gives 0.70763028 and 0.70774627, and
  0 of 428 gradient tensors are bitwise equal in any sample (`out/out_369325/nopad_bitwise.txt`).
- So a second timing-dependent choice remains. Most likely it is Triton's runtime autotuning, which picks among
  configurations of one kernel by timing and changes its summation order. This was not tested.
- Runtime atomics alone would not explain it: offA and g2, and brepro and 368799 below, repeat each other's losses
  bitwise across processes.

**(b) The reference's blow-up** (ISLA-DTT reference: head 5e82a9b98 with `tree_read_kernel="reference"`, forward_pass,
AirfRANS 2,000 points, huber, flag on, sample k = 2). Probes: `lhr/blowup_probe.py` (the step with gradient hooks per
block, and a replay of one block), `lhr/flag3_probe.py` (flag2_probe with the same hooks and an in-process replay of
the very compiled callable). `blocks.9.assign.weight` is the maximum |gradient| (true 6.1e-4).

| process | run order, blocks | node | loss k = 2 | max \|grad\| `blocks.9.assign.weight` |
|---|---|---|---|---|
| 368799 dtt_af2k | float64, then bf16 k = 0..6; compiled | nvl72d039-T01 | 0.95084101 | **6.646e5** |
| 369180 brepro | the same, cold | nvl72d042-T03 | 0.95084101 (bitwise; k = 0..5 bitwise too) | **6.679e5** |
| 369325 bblk | the same, cold, with block hooks | nvl72d010-T09 | 0.95086992 (k = 0 already differs) | 1.6e-3, normal |
| 369325 bblke | the same, eager blocks | nvl72d010-T09 | 0.95072639 | normal |
| 369180 bcomp | bf16 k = 0, 1, 2 only; compiled | nvl72d042-T03 | 0.95083094 | normal |
| 369180 beager | the same; eager | nvl72d042-T03 | 0.95072639 | normal |

- The blow-up repeats whenever the compile draw does. brepro has 368799's losses bitwise for k = 0..5. Its gradients
  differ only at the level of the reference read's bf16 atomics (48 of 427 tensors bitwise), and its `blocks.9`
  gradient is 6.68e5 against 6.65e5.
- In the other draws every block's gradient is normal. The gain |grad in| / |grad out| is 1.06-3.16 per block,
  matching sample 0's to 3 digits, compiled and eager alike. Eager runs are deterministic: bblke and beager give the
  same k = 2 loss in different processes.
- The flag's part is not established. The flag-off rerun of k = 2 in 368799 was a separate process, so it was a
  different draw too.
- No operation could be named. The localization run (bblk) did not draw the blow-up, so its replay found nothing: the
  compiled callable replayed bitwise to the full step, and no op exceeded 100 times sample 0's.
- What the blow-up's own gradients show (`out/job_369015.log`, `out/out_369254/tail_brepro.txt`): it arises in block
  9's backward and runs upstream. Block 9's seeds (`assign.weight`) are at 6.6e5, and every earlier block's
  `broadcast.weight`, `embed` and the boundary seed are at 3e4 to 9e4; blocks 10 and 11 are untouched. The loss and
  outputs are normal and nothing is non-finite.
- Likely mechanism (a hypothesis, not tested): `_relational_invariants` (`blocks.py`) divides by the point-to-anchor
  distance, `rel_hat = rel / dist` and `log(dist)`, with `dist` clamped only at `EPS = 1e-12`. A slice anchor
  `z_pos` (a weighted mean of token positions) that lands almost exactly on a token gives derivatives of order
  1/dist: finite, and huge. Whether it lands there depends on the rounding of the bf16 slice weights, hence on the
  compile draw.
- Three raw records are kept on LHR for a later look: 368799's truth and k = 2, and brepro's k = 2, in
  `$T/kernel/dttfast_flag/keep_blowup/`.

## Ideas for DTT-KNOBS / the program (not implemented: they change the evaluation or the architecture)

- Make bf16 equivalence judgements across compile draws: the reference family over several cold processes, or both
  sides with `pad_mm` off and a shared, pinned autotune cache. A one-process family understates the reference's spread.
- The 1/dist conditioning of the relational invariants near a coincident anchor (a smooth floor on `dist` at a
  derived scale instead of 1e-12) is an architecture question for the supervisor, not this campaign's.
- Whether to set the flag off is still open. It costs nothing, but it is only worth the change to standing numerics
  once a draw-aware yardstick shows that it helps.

## Files

`out/cost_table.md`, `out/out_368758/` (80 runs' JSON and logs), `out/out_368799/compare_*.md` and logs,
`out/job_369015.log`, `out/out_369180/` (follow-up logs and profiles), `out/out_369254/` (follow-up tables),
`out/out_369325/`. Probes: `lhr/step_flag.py`, `lhr/flag_probe.py`, `lhr/flag2_probe.py`, `lhr/flag3_probe.py`,
`lhr/blowup_probe.py`, `lhr/flag_compare.py`, `lhr/tail_check.py`.
Launchers: `lhr/{dry,dry2,dry3,cost,eff,tail,fu,fucmp,fu2}/`. Snapshot md5 lists: `lhr/md5/`. Jobs: `JOBS.md`.
