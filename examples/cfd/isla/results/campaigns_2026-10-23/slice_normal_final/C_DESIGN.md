# Option (c): a smooth floor at bf16's rounding level for the slice mean normal (design note, 2026-10-25)

Written by SLICE-NORMAL-EXP at the supervisor's request (03:52 EDT 10-25), after the DR-A read showed that option (a)
harms (pressure ×1.167, wall shear ×1.121, worse on 48 of 48 cases; RESULTS.md §4.1). Nothing here has been run. M =
measured (with its source), E = estimated.

## 1. What (c) is

The reference normalizes each slice's mean normal m = Σ_n a_ns n̂_n as g(m) = m / max(|m|, 1e-12). Option (c) replaces
it with

    g_c(m) = m / sqrt(|m|² + L_s²),   L_s = 2u · | Σ_n a_ns |n̂_n| |,   u = 2⁻⁸ (bf16's unit roundoff), fixed in code,

where |n̂_n| takes absolute values per component and the outer |·| is the Euclidean norm. L_s is the slice's own
first-order bound on the rounding error of the anchor product a^T n̂ when its inputs are rounded to bf16 (each of a and
n̂ contributes at most u relative error per term; SLICE-NORMAL RESULTS §2 calls it L_wc). It is computed per slice from
the model's own weights, as m is.

One line in `_SliceBlock.forward`, plus one product for L_s:

    lvl = 2.0**-7 * torch.einsum("bns,bnc->bsc", a, n_hat.abs()).norm(dim=-1, keepdim=True)
    m_s = m_s / (m_s.square().sum(-1, keepdim=True) + lvl.square()).sqrt()

The fused kernel takes m_s as given (SLICE-NORMAL RESULTS §1), so it needs no change; the extra product is a (B,N,S) x
(B,N,3) contraction, like m's own.

## 2. Why it should not repeat (a)'s harm

- (a) changes **every** slice: the two m-invariants (rel̂·m, n̂·m) shrink by |m|, a median 0.04-0.18 (×6-23). The DR-A
  read says the network used the unit direction of diffuse slices (or could not absorb the per-slice rescaling through
  `geo_logit` / `geo_feat`, whose input is shared by all slices).
- (c) is the reference wherever |m| ≫ L_s. The relative change is 1 - |m| / sqrt(|m|² + L_s²) ≈ L_s² / (2|m|²): with
  L_s ≈ 7e-3 (M, SLICE-NORMAL proto and RESULTS §2), that is 0.1-1.3 % at the median slice, 3e-5 to 3e-4 on coherent
  slices (|m| 0.3-0.9; proto (iv): |g_c| / |g_ref| = 0.99973-0.99997), and large only for slices near or below the
  rounding level: 0.008 % of slices on DrivAerML, 0.1-0.4 % on AirfRANS and the lug, 2-3 % on AeroZoo ISLA-DTT (M,
  RESULTS §2, below L_wc).
- So (c) keeps the function the controls were trained with on nearly every slice, and changes it only where the
  reference's direction is rounding noise anyway (direction error err/|m| up to 0.83 rad, M).

## 3. Derivation of its bound

- d g_c / d m = (I - m m^T / (|m|² + L²)) / sqrt(|m|² + L²), whose spectral norm is 1 / sqrt(|m|² + L²) ≤ 1 / L_s. So
  the m-path gain is at most 1 / L_s; it is largest at m = 0, where g_c ≈ m / L_s is linear and smooth (unlike (b)'s
  kink at |m| = L_s).
- Bound on 1 / L_s: v = Σ_n a_ns |n̂_n| has non-negative components, and with Σ_n a_ns = 1 and ‖n̂‖₂ = 1,
  ‖v‖₂ ≤ Σ a ‖n̂‖₂ = 1 and ‖v‖₂ ≥ ‖v‖₁ / √d = Σ a ‖n̂‖₁ / √d ≥ 1 / √d. So L_s ∈ [2u / √d, 2u] and
  1 / L_s ∈ [128, 128 √d]: at most 181 in 2-D and 222 in 3-D. Measured gain on the closed 1024-gon: 4.1 against the
  reference's 2.9e10 (M, proto (i)).
- Gradient through L_s itself: L_s is 1-Lipschitz in a up to the factor 2u, and enters g_c smoothly; it may be kept
  (smooth) or detached (L_s as a per-slice scale with no gradient). Keeping it is the plain choice; detaching changes
  nothing in the forward.
- Forward noise in the two invariants, bf16 anchor product: at most about 0.07-0.19 on sub-level slices (proto (ii),
  split-K), against O(1) for the reference; (a) had 4e-4 to 2.5e-3.

## 4. What is not principled about it, and the alternatives

- u = 2⁻⁸ is a property of the training number format, fixed in code. It is not tuned and not derived from data, but it
  is a constant: the model computes the same function in float32 inference only because u is fixed rather than read from
  the runtime dtype. Peter's "no magic constants" ruling should be asked about this explicitly; the supervisor's sigma_s
  reading (A6) does not cover it, since u is not a statistic of the input.
- Reading u from the runtime dtype (c') would make float32 and bf16 inference different functions: rejected.
- (b), the hard clamp at L_s, has the same bound and constant but a kink; (d), the normal second moment, has no constant
  but loses orientation; (e), float32 anchor products, removes the noise, not the conditioning. (a) is now ruled out.
- Is any fix needed? The reference's failure is rare: one exact zero in 557 compiled bf16 draws, and no training spike
  traced to it in 267 runs (DTT-FAST SPIKE, M). What it bounds is a 1e9 gain that certification catches only by chance
  (FLAG). (c) is a guard, so its test is a no-harm check, as here.

## 5. What a test would cost (E), and a staged design

Same rule and shared controls as this lane (A §5, A3/A5); only arm lanes are new, on new snapshots (copies plus the
one changed blocks.py), with smokes and the m-check adapted (the arm's m_s must equal g_c(control's raw m) to rounding,
and the reference elsewhere).

| stage | arms | basis | GPU-h | calendar |
|---|---|---|---|---|
| 1 | DR-A-C (2 lanes, option A, fused kernel kept) | DR-A (a) lanes: 29.5 GPU-h each by sacct, 8.3 h | 60-65 | about 9 h plus scoring |
| 2 | AF-C (2) and LUG-C (2, compiled like the control) | this lane's AF and LUG arms (about 110 and 150 per lane) | 520-560 | AF about 29 h, LUG about 41 h |
| smokes and scoring | 3 smokes (LUG with C4, reported as F1), 3 scorers | this lane | 20-25 | |
| **total** | | | **about 600-650** | |

- Stage 1 alone is the cheap gate: DR-A is where (a) harmed, its controls exist, and its two-seed spread is small (s_c
  0.05 / 0.035), so a "harms" there is decisive and a clean read there is informative. Stage 2 would follow only if
  stage 1 reads no harm.
- If the (s) arms read clean on their marginal reading, a (c)+(s) arm could replace (c) alone in stage 2 (AF-CS, LUG-CS),
  at the same cost.
- Prediction (mine, before any (c) result): DR-A-C no change on both fields, 0.88; harms on any, 0.06.
