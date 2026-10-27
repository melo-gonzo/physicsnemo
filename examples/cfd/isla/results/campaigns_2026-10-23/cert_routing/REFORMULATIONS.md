# The reformulation set (written 2026-10-27 05:30 EDT, from the code alone, before any fresh-seed measurement)

The supervisor's 05:1x EDT reply asks for this list first. Source: blocks.py at 8097ed00f. "Eager" below means bf16
autocast on CUDA: GEMMs (einsum, Linear) run in bf16 with bf16 outputs; layer_norm, softmax and norm run in fp32;
other pointwise ops and reductions keep their input dtype.

## Scope: which roundings a correct compiled path may legitimately differ in

Under its default casts, Inductor keeps every GEMM in bf16 with a bf16 output, as autocast does. Inside a fused
kernel it computes pointwise ops and reductions in fp32 and rounds only what it stores. So a correct compiled path
can differ from eager in two ways:
- in the non-GEMM bf16 roundings that eager makes on its routing path (a precision raise);
- in the order of algebraically equivalent operations.

The set therefore removes, one step at a time, each non-GEMM bf16 rounding group of the routing path (precision
raises P), and adds the algebraically equivalent reorderings the code allows (O). Each member changes exactly one
step; eager's numerics elsewhere are untouched. GEMM precision is out of scope, since neither path changes it.

The routing path's eager bf16 roundings outside GEMM outputs (blocks.py line numbers):
- r1: `att * self.scale` (197).
- r2: `log_w.to(att.dtype)` (199).
- r3: `att + log_w` (199).
- r4-r9: `delta.mean`, `delta - mean`, `delta.square()`, `.mean()`, `.clamp_min`, `.sqrt()` (202, 209).
- r10: `ROUTING_TEMPERATURE * einsum` (224).
- r11: `logits + bias` (324, bf16 + bf16).
- In the backward, the gradients of these bf16 tensors are bf16:
  - r12: the logits' three gradient contributions (first softmax over points, geometry region, second logits) are
    accumulated in bf16, and the sum is rounded before the multiply by T;
  - r13: the attention logits' gradient is rounded at `att.float()` and again after `* scale`.

## Precision raises (bf16; in fp32 and fp64 they are no-ops by construction)

| id | variant (patches.py) | step changed | roundings removed |
|---|---|---|---|
| P1 | `E_lw32` | the measure term: bf16 `att*scale` + `log_w` kept in fp32 (promotes the sum) | r2, r3 |
| P2 | `E_att32` | the attention-logit step: GEMM output upcast, `* scale` and `+ log_w` in fp32 | r1, r2, r3 (and r13's second rounding) |
| P3 | `E_delta32` | the centered value term: `delta` upcast after its GEMM; mean, centering, mean square and root in fp32 | r4-r9 |
| P4 | `E_mul32` | the routing logits kept in fp32 after the GEMM: `T *` in fp32, so the bias sum and the fused region's pre-logits are fp32 | r10, r11 |
| P5 | `E_lb32` | only the bias sum: `logits.float() + bias.float()` | r11 |
| P6 | `E_gT32` | the backward of `T *`: the logits' gradient contributions are accumulated in fp32 and multiplied by T before one bf16 rounding (the forward stays bitwise eager; Inductor's fused backward of line 224 does this) | r12 |
| P7 | `E_gacc32` | the logits' gradient contributions accumulated in fp32, rounded once before `T *` (forward bitwise eager) | r12's accumulation roundings |

## Algebraically equivalent orders (bf16, fp32 and fp64)

| id | variant | reorder |
|---|---|---|
| O1 | `E_tfirst` | the temperature folded into the token rows before the GEMM: `einsum(T * unit(x), unit(protos))` |
| O2 | `E_sfirst` | the attention scale folded into the seeds before the GEMM: `einsum(seeds * scale, k_proj(x))` |
| O3 | `INJ__I_s1__I_geo__I_s2` (fp64 only) | the logits' gradient contributions accumulated in another order (identity autograd nodes on the three consumers) |
| O4 | `e64` (fp64 only) | the fused Triton geometry region in place of the eager reference region (CERT's e64) |

Excluded, with reasons:
- `torch.softmax(x, dim=1)` for the softmax over points: blocks.py:50-58 documents a ~1e-4 fp32 floor for the
  middle-dimension kernel, against ~1e-7 for the row kernel used, so it lowers precision.
- GEMMs in fp32: out of scope (above).
- `emulate_precision_casts`: it makes the compiled path copy eager; it does not reformulate eager.

## Families

- **bf16 reference family:** eager plus P1-P7 and O1-O2 (10 implementations), each with its own one-ulp draws,
  pooled.
- **fp32:** eager, O1 and O2 (P1-P7 equal eager in fp32).
- **fp64 reordering spread (G3 yardstick):** ref (eager region) against O1, O2, O3, O4 and ref's rerun. Per pooled
  group, the maximum distance to ref over those members.

## Addendum (05:50 EDT, after the exploratory runs x1 and x2): P6's implementation

P6 is described above as "the logits' gradient contributions accumulated in fp32 and multiplied by T before one bf16
rounding". Its first implementation, `E_gT32`, left one contribution rounded: the second logits `logits + bias` are a
bf16 sum, so their gradient reached the fp32 accumulator already rounded. While pinning HiLift (x2), after its numbers
were seen, I wrote the implementation that matches the description: `INJ__E_gT32__E_gTall`. It keeps that sum's
forward rounding (bitwise eager) and passes its gradient through in fp32. Part 3's proposal uses it as P6. The list
of steps is unchanged.
