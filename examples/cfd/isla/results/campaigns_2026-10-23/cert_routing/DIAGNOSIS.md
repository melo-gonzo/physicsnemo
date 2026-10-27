# CERT-ROUTING diagnosis (written 2026-10-27 05:25 EDT)

Code: psharpe/isla-014 8097ed00f via CERT's LHR snapshot `$T/kernel/cert1027/code_8097`, torch 2.13, GB300.
Tool: `lhr/diag.py` (CERT's cert3.py load and capture; per captured block call, forward and backward with CERT's
fixed output gradients; every variant scored against the float64 eager reference, pooled by CERT's roles, l2) and
`lhr/patches.py` (the variants). Job 419517 (one node, 4 workers over an appendable queue, 1:16 h from 01:11 PDT).
Outputs: `out/d1/*_d<N>.jsonl` (summaries: `python3 summ.py d<N>` in `out/d1`), the Inductor code dumps
`out/d1/t2d_output_code.txt` and `code_{inj,lw}.txt`, and the tap summaries `out/d1/taps3/*_tapsum.jsonl`.
Groups studied: the Part 2 worst groups, t2d fwd2 c0 and fwd0 c0, lug fwd2 c1, hl sfwd0 c0/c1, vol fwd3/fwd4 c1,
and in d14 also af fwd11 c1 / fwd5 c0, az fwd9 c1, drs sfwd0 c0/c1, suv sfwd0 c1 / sfwd2 c0.

Numbers below: routing pooled l2 distance to float64 (base run); "ratio" = median of the variant's one-ulp draws
over the eager one-ulp family's maximum (CERT's G2 shape; > 1 fails).

## 1. The excess reproduces, and only Inductor's elided casts cause it

| group | eager | eager family max | compiled | compiled + `emulate_precision_casts` |
|---|---|---|---|---|
| t2d fwd2 c0 | 0.124 | 0.135 | 1.27 | 0.124 |
| t2d fwd0 c0 | 0.309 | 0.413 | 2.06 | 0.309 |
| lug fwd2 c1 | 0.075 | 0.093 | 0.157 | 0.076 |
| hl sfwd0 c0 | 0.014 | 0.031 | 0.035 | 0.013 |
| vol fwd3 c1 | 0.095 | 0.120 | 0.242 | 0.094 |

Not the cause (no change): cuBLAS reduced-precision bf16 reductions, Inductor's pattern matcher, online softmax,
split reductions, epilogue fusion, buffer reuse / in-place buffers, IEEE division, FTZ, the partitioner's
recompute bans, the activation budget (d2, d4, d7).

## 2. The cast: the log-measure term of the prototypes' attention (blocks.py:198-199)

`att = einsum(seeds, k_proj(x)) * scale; att = att + log_w[..., 0][:, None, :].to(att.dtype)`, then
`softmax(att.float())`. Under eager bf16 autocast the GEMM output, the scaled logits, `log_w` and their sum are each
rounded to bf16. Inductor fuses the chain into the softmax kernel and keeps `scale * GEMM + log_w` in float32 with
the exact float32 `log_w` (code dump: `triton_per_fused__softmax__to_copy_add_bmm_mul_..._3` loads `log_w` as fp32;
the `.to(bf16)` is elided). The compiled attention logits are 65x closer to float64 (2.8e-5 vs 1.8e-3).

Proof that this one cast is the cause, from both directions (d5, d13, d14):
- **compiled + that one rounding put back** (an opaque custom op rounds `log_w` to bf16 before the add;
  `INJ__F_lw`): t2d fwd2 1.27 -> 0.096, t2d fwd0 2.06 -> 0.415, lug 0.157 -> 0.020, vol fwd3 0.242 -> 0.122.
  The kernels of the two compiled graphs differ only in that kernel (diffed).
- **eager with only that rounding removed** (`E_lw32`: bf16 logits + float32 `log_w`; `E_att32`: scale and
  `log_w` added in float32), i.e. eager made strictly more precise in one step, reproduces the compiled excess:

  | group | eager | compiled | eager E_att32 | eager E_lw32 |
  |---|---|---|---|---|
  | t2d fwd2 c0 (ratio) | 0.95 | 13.3 | 13.3 | 14.0 |
  | t2d fwd0 c0 (ratio) | 0.66 | 3.31 | 3.23 | 3.11 |
  | lug fwd2 c1 (ratio) | 0.86 | 3.54 | 3.56 | 2.17 |
  | vol fwd3 c1 (ratio) | 0.76 | 2.12 | 2.02 | 1.71 |
  | vol fwd4 c1 (ratio) | 0.75 | 1.25 | 1.24 | 1.28 |
  | suv sfwd0 c1 (ratio) | 0.66 | 1.55 | 1.55 | 1.56 |
  | af fwd11 c1 (ratio) | 0.98 | 1.09 | 1.00 | 1.02 |
  | az fwd9 c1, routing (ratio) | 0.95 | 0.47 | 1.09 | 1.07 |
  | drs sfwd0 c0 (ratio) | 0.96 | 1.14 | 1.03 | 1.00 |
  | hl sfwd0 c0 (ratio) | 0.73 | 1.20 | 0.72 | 0.70 |

  So the compiled path does not lose precision in any operation: it loses a bf16 rounding error of eager's that
  happens to compensate. hl (and most of drs) is not explained by this cast; see section 4.

## 3. Why removing a rounding error makes the gradient worse: near-degenerate slices

Taps (opaque copy ops that record forward values and gradients without changing the roundings; the excess
survives them) on t2d fwd2 c0 show where the routing-gradient error arises (rel. l2 to float64):

| tensor | eager | compiled | compiled + F_lw |
|---|---|---|---|
| att logits (fwd) | 1.8e-3 | 2.8e-5 | 1.3e-3 |
| protos (fwd) | 1.1e-2 | 7.6e-3 | 7.4e-3 |
| m = sum_n a n_hat (fwd) | 1.3e-2 | 4.2e-3 | 4.6e-3 |
| m_s = m / max(norm(m), eps) (fwd) | 7.1e-3 | 1.4e-2 | 4.2e-3 |
| grad m_s | 1.3e-2 | 1.8e-2 | 9.4e-3 |
| grad m | 0.58 | 2.98 | 0.44 |
| grad a (softmax over points, 1st) | 0.58 | 3.36 | 0.45 |
| grad protos | 0.12 | 1.18 | 0.09 |

Nearly all of the error (98-100% of grad m, grad a, grad of the attention) sits in one slice, 232, whose
measure-weighted mean normal nearly cancels: its routing weights are almost uniform (effective 478 of 496 tokens)
over the closed specimen boundary, where the integral of n dA is zero, so norm(m) = 0.0056 in float64. Normalizing
`m` amplifies any error by 1/norm(m) in the forward and in its backward. Its y component is 0.00528 (float64) and
0.00854 (eager), 0.00232 (compiled), 0.00714 (compiled + F_lw): pure bf16 noise of about +-0.003 on a 0.005 signal.
A noise draw that shrinks norm(m) blows the gradient up (compiled, 4x the float64 norm); one that grows it shrinks
the gradient (eager). Rounding `log_w` is a coherent perturbation (the measure takes a few values that follow the
geometry), so it moves m coherently; CERT's one-ulp family flips input bits with random signs, which average out
over the hundreds of tokens of such a slice, so the family neither samples this noise nor eager's luck in it (its
spread is 0.118-0.135 where eager-more-precise lands at 1.3). lug and vol show the same concentration (TAPSUM:
in compiled, 76-82% of the m_s forward error and 88-97% of the grad m error sit in the 10 smallest-norm(m)
slices; in eager 30-38%).

The ill-conditioning is in the architecture, eager included: in bf16, eager's own gradient of m in t2d block 2
is 58% wrong, and the anchor normal of such a slice is noise in both paths.

## 4. hl (and part of drs): another cancellation, not yet pinned to one cast

None of the eager reformulations tried reproduce hl's excess. On hl the excess appears in grad protos (compiled 4.8e-2,
eager 9.5e-3) with grad m and grad m_s accurate, and it vanishes when any graph break isolates the routing logits
(1.4e-2 -> 0.7e-2, better than eager) or when the logits' gradient is rounded once more before the temperature
multiply. This points to the same kind of cancellation in grad protos = T sum_n G[n, s] u_n over 200k tokens; it is
not yet pinned to one cast.

## 5. What this means for a fix

- No allowed change makes the compiled path more accurate here: it already computes the attention logits more
  precisely than eager. The gate fails it because eager's bf16 rounding of `log_w` happens to compensate the
  error of a few ill-conditioned slices, and the one-ulp yardstick cannot see that. Eager made strictly more precise
  in that one step fails the same gate by the same margin (t2d 13x, lug 3.6x, vol 2.0x, suv 1.55x).
- The only code change that would pass is to put eager's rounding back into the compiled graph, for example by
  keeping the measure-term addition out of the graph. Measured (`P_att`): t2d and lug return to eager's numbers,
  vol to 1.0x (borderline), hl and probably drs not. That copies eager's compensating error; it does not make
  the compiled path more accurate, so I do not call it a principled fix.
- Principled options lie outside this lane's scope. One is a yardstick that also samples coherent roundings, such
  as the eager reformulations above (a gate change). The other is an architecture change that conditions the anchor
  normal of slices whose normals cancel: m_s divides by norm(m), which is about 0 for a near-uniform slice on a
  closed boundary.
