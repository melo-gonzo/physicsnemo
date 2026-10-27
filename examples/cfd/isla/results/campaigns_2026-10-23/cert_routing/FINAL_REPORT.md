# CERT-ROUTING final report (2026-10-27, written 07:05 EDT)

Status: DONE.
- The diagnosis is complete. There is no principled fix: the compiled path is more precise, not less.
- Part 3 ran as approved. It certifies no family: the positive control passed G2'' in 4 of 8 families, so as
  registered the gate is "without power". The 8 families keep training eager.

## Diagnosis (DIAGNOSIS.md; jobs 419517, 420516, 420636)

Inductor's default casts remove two of eager's bf16 roundings on the routing path. Eager rounds at both places,
and the compiled path is the more precise there.
- **C1, the measure term of the prototypes' attention** (`_RoutingLogits.prototypes`, blocks.py:197-199). Eager
  rounds the GEMM output times `scale` to bf16, rounds `log_w` to bf16 (`.to(att.dtype)`), and adds the two in
  bf16, then upcasts for the softmax. Inductor computes `scale*GEMM + log_w` in fp32, with fp32 `log_w`, inside the
  softmax kernel. The attention logits come out 65x closer to fp64 (2.8e-5 against 1.8e-3). This explains t2d,
  lug and vol, and in part af, az and suv.
- **C2, the routing logits' backward** (blocks.py:224 and 312-326). Eager accumulates the three gradient
  contributions of the bf16 logits in bf16 and multiplies by T in bf16. Inductor accumulates them in fp32 and
  rounds T*G once. This explains hl, suv and drs.

The proof runs in both directions:
- Compiled with eager's rounding put back by an opaque op recovers: t2d 1.27 -> 0.096, lug 0.157 -> 0.020, vol
  0.242 -> 0.122; hl 1.25 / 1.45 -> 0.49 / 0.61, suv 1.27 -> 0.50.
- Eager made strictly more precise in only that step reproduces the compiled excess:
  - C1: t2d 13.3x (compiled 13.3x), lug 3.6x (3.5x), vol 2.0x (2.1x), suv 1.55x (1.55x);
  - C2: hl 1.33x / 1.52x (1.25 / 1.45), suv 1.49x (1.27x).

Why removing a rounding error makes the gradient worse: the gated groups are ill-conditioned. In t2d block 2, 98-100%
of the error sits in one slice. Its weights are near uniform over the closed specimen boundary, so its measure-weighted
mean normal cancels: norm(m) = 0.0056 against terms of order 1. The anchor normal m/norm(m) is bf16 noise in both
paths (+-0.003 on 0.005), and its backward is amplified by 1/norm(m). Eager's own gradient of m there is 58% wrong.
Eager's rounding of `log_w` happens to push norm(m) up, which shrinks the gradient; the more precise compiled forward
lands on the other side. CERT's random-sign one-ulp family averages out over such a slice, so it samples neither the
noise nor eager's luck in it.

Ruled out (no effect):
- cuBLAS reduced-precision reductions;
- the pattern matcher, online softmax, split reductions and epilogue fusion;
- buffer reuse and in-place buffers;
- IEEE division and FTZ;
- the partitioner's recompute bans and the activation budget;
- autograd accumulation order.

## Fix: none (diff 0 lines; the worktree psharpe/isla-014-routing was made, left unused, and removed)

No allowed change makes the compiled path more accurate here, since it already computes these steps more precisely
than eager. The one change that would pass the old gate puts eager's compensating rounding back into the graph, for
example by keeping the measure-term addition out of it. It would bring t2d and lug back to eager's numbers, vol to
1.0x, and not hl. The supervisor declined it as not principled (05:15 EDT).

Because nothing changed, eager stays bitwise by construction. That made the GB300 and CPU test tiers and the
compiled timing check moot, so I ran none of them; Part 1's 4-rank smokes (15-36% saving) stand. Part 3's tool
checks that its eager reformulations, with no step changed, reproduce eager bitwise: 18 of 18 checks pass
(3 replacements x 2 precisions x 3 kinds), on CPU and in every GPU process.

## Part 3 (CERT_PLAN.md section 10, md5 6ecbfeff, approved 05:53 EDT; jobs 420755, 420756, 420828, 420829)

**This gate was redesigned after Part 2's results.** The reason is a diagnosed yardstick bias: eager made strictly
more precise in one step fails the eager-only gate by the compiled margin. The redesign is the reference family of
eager plus 9 pre-listed precision-raising or reordered eager implementations (REFORMULATIONS.md, with the disclosed
P6 correction), pooled and null-calibrated, with a calibrated fp64 bug detector and a positive control. It was
decided on fresh data: seeds 4 and 5, all groups, two cold processes per family.

Verdicts (`out/part3/PART3_VERDICTS.md`):

| family | G1'' fp32 | G2'' bf16 N / R (cN / cR) | G3'' fp64 | positive control | verdict | prediction (P certify) |
|---|---|---|---|---|---|---|
| af | pass | **1 / 1.007** (0 / 1.000), fwd11.routing c1 | pass | detected | not certified | 0.75 |
| az | pass | 0 / 1.000 (0 / 1.000) | pass | detected | not certified (gate without power) | 0.75 |
| drs | **fail** (N1 2 > 1) | **1 / 1.017**, sfwd0.routing c0 | pass | detected | not certified | 0.30 |
| hl | pass | 0 / 0.992 | pass | **passed** | not certified (gate without power) | 0.30 |
| lug | pass | 0 / 0.998 | pass | **passed** | not certified (gate without power) | 0.60 |
| suv | pass | 0 / 0.990 | **4 groups** (3.3e-12 > 2.9-3.2e-12) | **passed** | not certified | 0.50 |
| t2d | pass | 0 / 0.995 | pass | detected (R 1.78) | not certified (gate without power) | 0.75 |
| vol | **fail** (R1 5.46 > 1.79) | 0 / 0.994 | pass | **passed** | not certified | 0.75 |

Power rule: the positive control passed in 4 of 8 families (more than 2), so no family certifies, whatever its own
reading. Five families pass their own three gates: az, hl, lug and t2d, and suv on G1'' and G2'' only.

What the data say beside the verdict:
- The eager-only gate fails again on fresh seeds in all 8 families.
- The precision-raising members fail it as the compiled path does:
  - t2d: P1 11.5x against compiled 9.8x;
  - lug: P1 1.90x against 2.03x;
  - vol: P1 1.47x against 1.69x;
  - suv: P3 2.31x and P6 1.46x against 1.26x;
  - hl: P6 1.29x against 1.35x;
  - az: O1 2.97x against 1.02x.
- So the old gate cannot tell compiled from eager made more precise.
- The new gate is too wide to detect an added bf16 rounding of the softmax-over-points inputs in 4 families. At
  this resolution, per-group bf16 distances in these ill-conditioned routing groups do not separate "different
  rounding" from "lost precision".
- vol's fp32 failure has the same shape: fwd2.routing c0 is 7.9e-4 compiled against 7.1e-4 eager, in a group whose
  fp32 error is about 1e4 times fp32's unit roundoff. The fp32 reorderings O1 and O2 do not cover Inductor's fp32
  fusion order.

How sure I am: the diagnosis is firm, with both directions, two mechanisms and fresh-seed replication. The meaning of
Part 3's "without power" is less certain. It may only show that the control was too weak (one bf16 rounding, about
2x eager's own error in that step), not that the family is too wide for every real loss.

## For the supervisor's decision (not done)

- A decisive test of the compiled path would have to read well-conditioned quantities, for example a short training
  A/B of compiled against eager on several seeds, read on validation error. Per-group distances in degenerate slices
  cannot decide it either way. Training eager costs about 815 GPU-h more in the rerun (CERT).
- LANES.tsv: no change (the 8 stay eager; djb compiled, per CERT Part 2).

## Architecture note for Peter (not for 0.14.0; after the rerun)

Near-degenerate slices have an anchor normal that is bf16 noise in both paths.

On a closed boundary the integral of n dA is zero. So a slice whose routing weights are near uniform over a closed
piece has a mean normal m = sum_n a_ns n_hat_n of norm close to 0. ISLA normalizes it, m_s = m / max(norm(m), 1e-12):
its direction is rounding noise, and its backward is amplified by 1/norm(m).

Measured smallest norm(m) of the 512 slices:

| family | block | smallest norm(m) |
|---|---|---|
| t2d | 0 | 0.0023 |
| t2d | 2 | 0.0056 |
| lug | 2 | 0.0079 |
| vol | 4 | 0.0036 |
| vol | 3 | 0.0048 |
| hl | 0 | 0.056 (not degenerate; hl's excess is C2) |

In t2d block 2, eager's bf16 gradient of m is 58% wrong, nearly all of it from one slice.

This is the case SLICE-NORMAL-EXP's proposal (c) addresses (`scratch/isla-slice-normal-exp-2026-10-24/C_DESIGN.md`):
the smooth floor g_c(m) = m / sqrt(norm(m)^2 + L_s^2), with L_s = 2u norm(sum_n a_ns abs(n_hat_n)) and u = 2^-8. With
L_s of about 7e-3, (c) changes exactly these slices, whose norm(m) is at or below L_s. It would bound their gradient
and make their routing gradients well conditioned in both paths. Its constant u = 2^-8 is Peter's to accept under the
no-magic-constants rule. C2 (hl, suv, drs) is a separate cancellation, sum_n G[n,s] u_n over 200k tokens in the
prototypes' gradient, which (c) does not touch.

## For the book

The compiled-region certification chapter needs one new result: the compiled blocks are at least as precise as
eager op by op. The Part 2 failures come from eager's compensating bf16 roundings in near-degenerate slices, and a
redesigned, fresh-seed Part 3 could not decide the question because its positive control was not detected. The
DIAGNOSIS.md tables and the Part 3 table above are the sources. No notebook draft was written; ask if one is wanted.

## For project memory

- Compiled ISLA blocks (Inductor default casts) remove two eager bf16 roundings on the routing path (C1, C2). Per-group
  distances to fp64 in degenerate slices cannot certify the compiled path either way: eager made more precise fails
  the eager-only gate, and the pooled-family gate did not detect an added rounding.
- Any future equivalence gate of this kind should include a positive control of the size it must detect.

## GPU-h and what stays on disk

**GPU-h: 14.1 of the 25 cap**, by sacct x 4:

| job | GPU-h |
|---|---|
| 419517 (diagnosis) | 5.08 |
| 420516 (x1) | 1.13 |
| 420636 (x2) | 0.24 |
| 420755 (Part 3) | 3.19 |
| 420756 (Part 3) | 2.00 |
| 420828 (Part 3) | 2.01 |
| 420829 (Part 3) | 0.44 |

Job 420729 was a CPU dry run. Details are in JOBS.md. Nothing is running. The registry row is removed.

- **LHR:** `$T/kernel/certrouting` holds only the tools, queues and the job logs. The per-job output directories
  were deleted after their files were copied here.
- **Local:** this folder holds DIAGNOSIS.md, REFORMULATIONS.md, CERT_PLAN.md (section 10), JOBS.md and the tools
  in `lhr/`. It also holds the gzipped records in `out/` (d1, x1, x2, dry and part3) and `synth/` (the summarizer's
  synthetic test). The raw captures and taps were deleted once this report was written.
- No worktree and no venv remain. The branch psharpe/isla-014-routing was never pushed and is deleted.
