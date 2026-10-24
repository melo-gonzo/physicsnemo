# DTT-FAST BF16-DIAG: the chunked step's bf16 excess (2026-10-24 09:33 EDT)

## Verdict

The AirfRANS excess is **systematic, not summation-order noise**, and it is not in `query_chunks`' structure. The chunked
step runs the boundary side on the boundary tokens alone: 1,017 on AirfRANS, an odd count. At that count, under
PyTorch's default `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True`, cuBLASLt computes the
slice anchors (`z_pos = einsum(a, r)` and `m_s = einsum(a, n_hat)` of every block, and the head frame's anchors) with a
split-K kernel whose partial sums are rounded to bf16. That is an extra bf16 rounding point. The unchunked step, at
its 3,017 tokens, does not have it. It shows as a heavy tail, not a shift. Over 14 one-ulp parameter perturbations at
2,000 points, 6 chunked samples land at 2.1-6.8x the reference's bf16 blocks error, and no reference sample goes above
1.00x. At all 180,593 points, 1 of 5 chunked samples lands at 6.1x. The tail sits in blocks 0-3's routing and geometry
weights.

Evidence that the structure is exact:
- With the boundary side run on the unchunked token tensor (form x), the chunked step reproduces the reference's loss
  bitwise, sample by sample.
- With no leaves at all (form n, one autograd graph), the step equals the leaf version.
- Turning reduced-precision reduction off, or padding the boundary side to a multiple of 32, removes the tail.

The DrivAerML whole-boundary excess (loss 5.2x) **is noise**. Its boundary count, 200,000, is already a multiple of 32.
The reference's own one-ulp spread on the loss reaches 6.6x (mse) and 16.2x (relative_l2), and 3.35x on the blocks
gradients. Every chunked sample lies inside that spread. The loss is a scalar: report it, do not gate on it.

**Remedy, prototyped:** pad `_QueryChunks`' boundary side the way `forward` pads its tokens on psharpe/isla 1b25eb899
(ALIGN8's `_pad_tokens`, `TOKEN_ALIGNMENT = 32`). It is a 5-line change plus 1b25eb899's two definitions, in
`remedy/remedy_pad_757a5a877.patch`. Measured:
- AirfRANS, 2,000 points: blocks ratio 0.74-1.01 over 14 samples (mse); relative_l2 0.82-1.00 over 4.
- AirfRANS, all 180,593 points: 0.73-0.98 over 5 samples.
- DrivAerML whole boundary (already aligned): unchanged.
- Lug: within the reference's spread.
- Cost: none measurable (+3 MB peak; one-step times equal within noise).
- Whole ISLA test directory: 398 passed, 0 failed.

## Method

The inputs, parameters and probes are integration pass 2's: PROFILE's dumps and `integration/.../p2/state/<case>.pt`.
The head is `code_dttfast_integ_p2b`, whose model files equal 757a5a877's, with recipe `recipe_dttfast_integ_p2d` and
`tree_read_kernel="reference"`. The reference is `code_dttfast_integ_ref7213` (7213c70c5) with recipe
`recipe_dttfast_integ_p1`. Each version runs in its own process, and every run prints its model-file md5s. Pairs are
md5-equal to the reference's in every run.

Distances are eq2's: max |run - truth| over a tensor group, divided by the group's max |truth|. The truth is the
reference's float64 unchunked step. At sizes where float64 does not fit, it is the head's float64 chunked step (equal to
the reference's to 1e-13), or the head's float32 step (about 1e-6, which is 1e-4 of a bf16 run's error). "Ratio" is a
run's distance over the reference's own unperturbed bf16 distance. "Shift" is the run's distance to that reference bf16
run, on the same scale.

The noise yardstick is the program's one-ulp rule (LANE_RULES, compiled regions): every float32 parameter moves one ulp
up or down, with signs seeded by the sample number k and the parameter name, identical in both versions. That gives a
family of k = 0..N bf16 samples per form.

Probes and tables: `lhr/diag_probe.py` and `lhr/diag_compare.py` (job a); `lhr/b/diag2_*`, `lhr/c/diag3_*`,
`lhr/d/diag4_probe.py` (profiler), `lhr/e/` (the remedy as code). Every launcher in `lhr/<tag>/` names its snapshots.

## 1. The permutation control (job a, 367557), and why it cannot decide

Permuting the prediction points leaves the reference's bf16 forward bitwise unchanged: the outputs shift by 0.00, and
the points enter every token-axis reduction with weight exactly zero. Gradients move by only 0.05-0.3 of the bf16
error. Every chunked run also shares one boundary-only forward. So neither family samples forward rounding.

| case (2,000 points, mse unless noted) | ref, 5 point orders | head chunked: 3 orders, chunks 700 or 1 | head unchunked |
|---|---|---|---|
| AirfRANS | 1.00 every group | blocks 6.36-6.40 (4 runs); others <= 1.01 | 1.00 |
| AirfRANS relative_l2 | 1.00 | blocks 4.00, 4.03 | |
| DrivAerML whole, 3 token draws | 1.00 | loss 7.69 / 1.60 / 2.14; gradient groups <= 1.20 | 1.00 |
| DrivAerML vehicle | 1.00 | <= 1.09 (mse), 1.26 (relative_l2) | 1.00 |
| AeroZoo | 1.00 | <= 1.26 (outputs) | 1.00 |

The token draw is degenerate on AirfRANS: its 1,017 cells are all kept, so every draw gives the same tokens. Tables are
in `out/a_367557/compare_<case>.md`, including per tensor and per `_QueryChunks` leaf.

Other findings of job a:
- The bf16 emulation of the parameter-gradient split (the float64 parts of each parameter's gradient rounded to bf16,
  then summed) gives 2.2e-4 on `blocks.1.geo_feat.weight`, against an observed 2.0e-2. So the split of the parameter
  gradients is not the cause.
- Per leaf, the float64 split emulation is about 2e-3 (no cancellation between the points' and the boundary's parts).
- The reference's own bf16 leaf gradients are already 3e-2 to 0.95 of the leaf's max from float64, so leaf
  comparisons are noise-dominated and cannot localize the excess. The decisive localization is in sections 2 and 3.

## 2. The one-ulp yardstick and the structure (job b, 367780; job c, 367888)

The cells give the blocks group ratio as min / median / max over the family. Families are per form at 2,000 points.

| family | AirfRANS mse | AirfRANS rl2 | DrivAerML whole mse: blocks; loss | DrivAerML whole rl2: blocks; loss |
|---|---|---|---|---|
| ref unchunked (14 / 4 / 7 / 4 samples) | **0.88 / 0.97 / 1.00** | 0.94 / 0.99 / 1.00 | 0.87-3.35; loss 0.45-6.57 | 1.00-2.05; loss 1.00-16.2 |
| head chunked c, chunks 700 (14 / 4 / 7 / 4) | **0.75 / 1.04 / 6.80**; 6 of 14 > 2 (6.40, 6.12, 6.80, 2.09, 2.14, 3.44) | 0.98 / 2.00 / 4.00 | 0.80-1.87; loss 1.48-6.66 | 0.71-1.04; loss 5.21-18.3 |
| x: c with the unchunked step's boundary forward (7 / 2) | 0.88 / 0.97 / 1.00; loss bitwise the reference's per sample | 1.00 / 1.00 / 1.00 | 1.00-3.33 (bitwise ref) | 1.00-1.17 |
| n: c with no leaves, one graph, one chunk (2) | 6.40, 1.01 (k = 0 bitwise c with one chunk) | | k = 0 bitwise c | |
| p: c with the boundary side padded to 32 (14) | **0.74 / 0.98 / 1.01** | | | |
| c with `allow_bf16_reduced_precision_reduction=False` (7) | **0.92 / 0.98 / 1.00** (ref: 0.88-1.00) | | | |

The other groups (loss, outputs, kernel_reads, kernel_basis, other) stay within 0.55-1.23 in every AirfRANS family at
2k, and within the reference's spread on DrivAerML. Tables: `out/b_367780/{af2k,dw2k}/compare.md`, `out/c_367888/compare_af2k_
{pad,rrr0}.md`. Per tensor (`out/b_367780/tail_367868.log`, `lhr/b/tail_tensors.py`), the tail samples carry 4-15x the reference
family's maximum in blocks 0-3: `geo_feat.weight`, `assign.weight` (the seeds), `assign.k_proj`/`v_proj`,
`geo_logit.weight`, `norm_assign.weight`, `mlp.0`.

## 3. Mechanism (profiler, job d, 367990)

The profile covers one AirfRANS bf16 step at 2,000 points, one chunk. It lists the matmuls whose cuBLASLt split-K
reduction keeps bf16 partial sums (`splitKreduce_kernel<.., __nv_bfloat16, ..>`), by input shape
(`out/d_367990/prof_*.txt`).

| run | bf16-partial split-K calls | of which the slice anchors `bmm [1,512,N]x[1,N,2]` over the boundary tokens |
|---|---|---|
| head chunked, N = 1,017 | 40 | **25** (2 per block x 12, plus the frame's anchors) |
| head chunked, padded to N = 1,024 | 15 | 0 |
| reference unchunked, N = 3,017 | 41 (slice pooling and readback bmm, geo_feat weight gradient) | 0 |
| either, `allow_bf16_reduced_precision_reduction=False` | 0 | 0 |

The other bf16-partial reductions of the chunked step also occur in the padded step, which is cured, or in the
reference: the geo_feat weight gradient `[256,N]x[N,6]` and `[512,1017]x[1017,3]`. That leaves the anchors as the
chunked step's one distinct extra rounding. The anchors set every token's geometry invariants (`rel = r - z_pos`,
distance, log distance) in each block, which is where the tail sits.

## 4. Size (jobs b, c, e)

| case | truth | ref | head chunked (chunks of 100k) | padded / patched | RRR off |
|---|---|---|---|---|---|
| AirfRANS, 180,593 points, mse, blocks | head f64 chunked | 0.94-1.00 (5) | 0.96-1.01, **6.09** (5) | 0.73-0.98 (5) | 0.90-0.98 (5) |
| AirfRANS, 180,593, relative_l2, blocks | same | 0.99-1.00 (2) | 0.99, **3.63** | | |
| DrivAerML whole, 100k, mse, worst gradient group (draws 1 / 2 / 1234) | head f32 chunked | 1.04 / 1.38 / 1.06 | 1.60 / 1.44 / 1.21 (2 samples each) | no-op (aligned) | |
| DrivAerML whole, 200k, relative_l2, worst gradient group | head f32 chunked | 1.00 | 1.05 (loss 1.69-2.58, ref 1.00-1.17) | no-op | |

On AirfRANS the excess neither grows nor vanishes with size: it is the same heavy tail. On DrivAerML every gradient
group stays at 1.6 or below. With 2 samples per draw, the reference's spread there is underestimated; at 2k its
family reaches 3.35. Float64 at 100k and 200k on DrivAerML ran out of memory (276 GiB), so the truth there is float32.
Tables: `out/c_367888/compare_{affull_pad,affull_rrr0,dw100k,dw200k}.md`, `out/b_367780/affull/compare.md`.

## 5. The remedy as code (job e, 368049; tests 368050)

The patch is `remedy/remedy_pad_757a5a877.patch` and the patched file is `remedy/model.py` (md5 in
`remedy/MD5SUMS`). It adds 1b25eb899's `TOKEN_ALIGNMENT` and `_pad_tokens` verbatim, plus the F import. In
`_QueryChunks.__init__` it pads the boundary tokens after the walk, and it gives the read's sources the real rows
`h[:, :n_cells]`. After merging 1b25eb899, only the `__init__` hunk remains. The snapshot is
`$T/code_dttfast_bf16diag_pad` (read-only, MD5SUMS and SNAPSHOT_HEAD inside).

| check | result |
|---|---|
| AirfRANS 2k mse, 14 samples (blocks; every group) | 0.74 / 0.98 / 1.01; every group <= 1.12 |
| AirfRANS 2k relative_l2, 4 samples | blocks 0.82-1.00 |
| AirfRANS 180,593 points, 5 samples | blocks 0.73-0.98; outputs 0.92-1.00 |
| DrivAerML whole 2k, 4 samples (200,000 tokens, nothing padded) | losses bitwise the unpatched run's for 3 samples; one differs in the 7th digit (the reference read's bf16 atomics, as pass 1's run-to-run spread) |
| lug 2k, 7 samples (unpatched / patched) | blocks 1.00-1.01 / 1.00-1.00; every group within the reference's spread |
| pairs | md5-equal to the reference's in every run |
| cost | peak 32.363 vs 32.360 GiB at 180,593 points; one-step wall times equal within noise (1.52 vs 1.51-1.69 s; not a timed benchmark) |
| whole ISLA test directory, cpu-short 368050 | **398 passed, 122 skipped, 0 failed** (as 367067 on p2b) |
| recipe tests, 368050 | 529 passed, 2 failed: the pre-existing `test_campaign_c_hooks` pair (manager.py:350/398 TypeError, as 367067) |

The change was not committed. The parent decides.

Caveat: padding steers cuBLAS's shape heuristics, and it fixes the one shape observed here. Another token count, or
the padded unchunked count after 1b25eb899, could in principle land a token-axis reduction on bf16 partials again.
Only the flag `allow_bf16_reduced_precision_reduction=False` removes bf16-partial split-K everywhere. That costs
nothing visible here, but it changes the reference's own bf16 numerics at some shapes: at 2k its losses change, while
at 180,593 points they are bitwise unchanged. It is a program-level precision choice, not this lane's.

## Ideas for DTT-KNOBS / the program

- Set `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` in the recipe for bf16 training.
  This would make every token-axis reduction's split-K partials float32 at every shape, for the reference and
  candidates alike. It needs a timed benchmark and the supervisor's decision, since it changes standing numerics.
- Judge bf16 equivalence against a one-ulp family (7 or more samples) of the reference, not one run. A single sample
  can sit at the tail or the bulk of the error distribution.

## Jobs and GPU-h

GPU: 367557 (0.83), 367780 (0.53), 367888 (0.41), 367990 (0.12), 368049 (0.33): **2.22 of the 4 GPU-h cap**.
cpu-short: 367513, 367632, 367633, 367756, 367868, 367879, 368050. Details are in `JOBS.md`.
