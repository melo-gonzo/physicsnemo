# GEO-BAND item 1: worst-case bounds for the exact query geometry's tie and on-wall bands (2026-10-24; v2 after the independent review, REVIEW_DERIVATION.md)

Code: `geometry.py` at 921472488 (md5 7472ecf0). Notation as GEO-THIN: `u = 2^-53` (float64 arithmetic), `u_in` the unit
roundoff of the queries' precision, `l` the greatest distance from the query to a vertex of the cell, `L` its longest
edge, `R` its greatest vertex norm, `d` the pair distance, `rho = 2A / L^2`. A *unit* is `u (|x| + R)` measured in the
frame the arithmetic runs in; with the centring of item 3 that is `u (|x'| + R')`, `x' = x - c`, `R' = max |v - c|`.

## Rounding model, and why the first-order counts become worst cases

Each float64 operation is `fl(a op b) = (a op b)(1 + delta)`, `|delta| <= u` (IEEE; torch and Warp evaluate each
elementwise operation by itself; where CUDA contracts `a * b + c` into a fused multiply-add the result has one rounding
fewer, so every bound below still holds). A quantity formed by `n` successive roundings carries a factor `prod (1 +
delta_i)^{+-1} = 1 + theta_n`, `|theta_n| <= gamma_n = n u / (1 - n u)` (Higham 2002, Lemma 3.1), *exactly*, with no
dropped term. So a first-order count `c u |t|` that counts every rounding of a term `t` becomes the worst case `gamma_c
|t|` once two conditions hold:

1. every product of two error terms is bounded too (`|e_1 e_2| <= (c_1 u)(c_2 u) |t_1 t_2|`), which adds a relative
   `c u <= 100 u = 1.1e-14` to the term;
2. no term is divided by a quantity that rounding can make small (a denominator near zero turns a second-order term into
   a first-order one).

Condition 2 is the only one that can fail; it is checked per computation below. Where it holds, the worst case is the
first-order count times `1 + 2e-14`, and the constants below are rounded up past that.

## (a) Ericson's regions, `K = 168` (already derived; GEO-THIN)

Steps 1-5 of the comment above `_ERICSON_K`. Condition 2: the one denominator is `denom~ = va~ + vb~ + vc~`, which
cancels on thin cells; the derivation keeps it as `denom~ >= (1 - theta) D` and solves `K = 101.25 / (1 - 66 / K)`, so the
amplification is in the bound, not dropped. Condition 1 adds `<= 25 u` relative to steps 1-2 (`(5 u L l)^2` against
`20 u L^2 l^2`). The worst case is `167.25 (1 + 3e-15)`: `K = 168` holds as a worst case. Checked against exact rational
arithmetic: largest observed ratio 0.0115 of the bound over 600k thin pairs (GEO-THIN).

## (b) Edge points (`_segment_t` and the edge-region rebuild in `_closest_weights`)

`y = fl(x - a)`, `s = fl(b - a)` (`|y~ - y| <= u |y|`, `|s~ - s| <= u |s|`; both exact for float32 inputs with exponents
within 29 bits), `t = fl(fl(y . s) / fl(s . s))`, clamped to [0, 1]:
- `|fl(y~ . s~) - y . s| <= (2 u + gamma_3) |y| |s|` (the two input roundings and the three-term dot product);
  `fl(s~ . s~) = |s|^2 (1 + theta)`, `|theta| <= 2 u + gamma_3`; the division `u`.
- The point's displacement along the edge, `|t~ - t| |s| <= 5 u |y| + 6 u t |s|`, and `t |s| <= |y|` (a projection),
  so `<= 11 u l` (clamping is 1-Lipschitz). The point is on the segment, so `0 <= |x - P~| - D <= 11 u l`.
- Assembly `fl(fl(1 - t) a + fl(t b))` (the third weight is an exact zero): `<= (u + gamma_2) R <= 3 u R`.
- `fl(|fl(x - P)|)`: `u d` for the difference, `gamma_{2.5}` for the norm (three squares, two sums, a root; Higham
  Sec. 3.1): `3.5 u d`, `d <= l`.
- Total `14.5 u l + 3 u R`; with `l <= |x| + R`: **17.5 units**. Condition 2 holds (`|s|^2` divides only terms that
  carry `|s|`; a zero-length segment is caught by `len2 > 0` and the exact zero-area test).
- In two dimensions (AirfRANS, Tensile2d) the dot products have two terms: `(2u + gamma_2)`, `gamma_2`; displacement
  `<= 9 u l`, assembly `3 u R`, difference and norm `3 u d`: `12 u l + 3 u R`, **15 units**.

## (c) The accurate face foot (`_accurate_weights`)

Recounted step by step as coded (`A, B` the longest edge, `e = (B - A) / |B - A|`, `n` the normal from the exactly
rounded cross product, `m = n x e`, `y = x - A`); corrected after the independent review (`REVIEW_DERIVATION.md`):
1. `n~`: each component rounded once from its exact expansion (`u`), the norm (`gamma_{2.5}`), the division (`u`):
   `5.5 u`. `e~`: `5.5 u`. `m~ = n~ x e~`: `15 u`.
2. `s1 = y . e`, `h = y . n`: `9.5 u l`; `s2 = y . m`: `19 u l`.
3. Along `m`: `lambda_C hc` errs by `19 u l + 9 lambda_C u L` (`|s2| = lambda_C hc`).
4. Along `e`: `9.5 u l + (10.5 lambda_C + 5.5 lambda_B) u L`. `tc`'s error (`9.5 u L`) and the product's rounding scale
   with `lambda_C`; the difference, the division and `|B - A|`'s rounding scale with `|lambda_B |B - A|| = |s1 - lambda_C
   tc|`, not with `L`. (The first draft of this file added the two maxima separately, `16 u L`, and wrongly said
   GEO-THIN had under-counted; GEO-THIN's `delta` was right.)
5. On an admitted face `lambda_B~ + lambda_C~ <= 1 + u`, so `delta = 28.5 u l + 19.5 u L`. `lambda_A~ >= 0` is decided
   from rounded weights, so the admitted point may lie `u h_A <= u L` beyond `BC`. Weights' sum and assembly: `5 u R`.
6. **Admitted face:** `|d_acc - D| <= 9.5 u l + delta + u L = 38 u l + 20.5 u L`, **79 units**.
7. **Rejected face** (the worst path, missing from the first draft and from the code's comment): the exact foot `F`
   lies inside the face, but some rounded weight is negative. Then `D = |h|` while the computed distance is the least
   edge candidate, `<= sqrt(h^2 + s^2) + e_edge`, `s <= delta + u L` the distance from `F` to the face's boundary:
   `e_acc = 43 u l + 20.5 u L + 3 u R`, **87 units**. The rule "the least of the candidates errs by at most the
   largest of their errors" fails here, because the exact minimum's face term has no computed counterpart. Condition 2
   (no small denominator) holds only on an admitted face; in the rejected branch the weights are unbounded, but the
   error across `AC` is damped by `hc / |AC|` and across `BC` it is at most `16 u L m_err / hc` with `s <= hc`, about
   67.5 units, inside the linear bound.
8. A switched pair's distance recomputed from the weights: `32 u l + 19.5 u L + 5 u R` (76 units) on an admitted face;
   87 after a rejected face.

So `e_acc = 87` units, not 77 (GEO-THIN) or 88 (first draft). `_FOOT_UNITS = 87`.

## (d) The pair distance's arithmetic share (the band's `C`)

| path | bound | units |
|---|---|---|
| unguarded Ericson point: the guard reads the computed distance, up to `a` = 9.5 units above `d(P')`, so its "below one unit" certifies the in-plane error to `(1 + sqrt(1 + 8a)) / 2` = 4.9; the weights' divisions and sum (3), assembly (3), difference and norm (3.5), Ericson's linear roundings | `<= 18.4` | 18.4 |
| edge point, (b), plus the guard (`a` = 17.5, 6.4) | `14.5 u l + 3 u R + 6.4` | 24 |
| vertex (difference and norm, the guard) | | 10 |
| zero-area triangle: an edge point and the choice between its two edges (`_segment_distance2`, `16.5 u l` each) | | 51 |
| switched pair, (c) 8 | `max(76, 87)` | 87 |
| guarded pair keeping Ericson's point: `tau + e_acc` | `1 + 87` | **88** |

Worst 88; the centring adds 1 (`x - c`, `v - c`; zero for float32 inputs whose exponents are within 29 bits of `c`'s):
89; the band's own float64 evaluation and the second-order terms take 2 more: **C3 = 91**, written `_SWITCH_UNITS +
_FOOT_UNITS + 3`. Two dimensions: an edge point, 15, plus the centring: 16, **C2 = 17**. The band's evaluation needs
no extra margin for its comparisons (rounding is monotone, so `fl(D - E) <= fl(D + E)` and `fl(d_min - E) <= 0` keep
their sense); its relative error of about 5 u is second order.

## (e) The switch threshold

Any threshold `tau` is certified once `C >= tau + e_acc + 1` (the centring): a kept pair has `|D_E - D| <= |D_E -
d_acc| + |d_acc - D| <= tau + e_acc`, and `fl(|d_acc - D_E|) <= tau` implies the exact difference is within `tau (1 +
2u)`. A switched pair uses the recomputed distance, (c) 8. The guard and the switch do not interact: only unguarded
pairs rely on the guard's bound. The unit the switch uses inside `_closest_weights` (from the centred coordinates passed
to it) is the unit of the band's arithmetic term. `switch` uses `>`, so a NaN distance keeps Ericson's weights (no
non-finite weights in 1.6M random float64 slivers, the review's CPU probe).

## (f) Other margins and conditions

- K = 168 (a): the form `min(Delta, Delta^2 / (2 (d~ - Delta)))` bounds the in-plane misplacement only; Ericson's
  step-4 roundings are linear in `u l` and belong to the share (the 18.4 above). The ball search's `_beyond` stays
  safe because the radius's pad of `3 err_max` exceeds the 23.5 units of those roundings at C = 91 (not at the
  package's former 14 for float64 queries).
- The 2-D brute force keeps segments within `2 (32 u_in (|x'| + R'max) + err_max)` of the least: two brute-force
  distances err by at most 12 units each (an edge point without the assembly) and the comparison's square and root by
  2.5, against 64; `2 err_max` covers `E_k + E_min`. The review counted `e_bf + A = 33.5 > 32` for float64 queries,
  treating the arithmetic share A as part of the 32; by my count `err_max` already covers it. Recorded as a
  disagreement; output-neutral either way, since a wider prefilter adds only candidates beyond the band.
- FMA: a contracted `a b + c` drops a rounding, so every bound holds; `_two_product` and `_two_sum` need it absent,
  and torch evaluates each operation by itself in the custom op, which torch.compile does not trace.
- G5 (the float64 rebuild of `x - p` outside the op) uses only differences, so neither the frame nor `sum w != 1`
  matters, and it does not feed the band.
- Float64 inputs: covered where every product stays normal, about `|v| < 1e76` with nonzero differences above
  `1e-76`, narrower than `_zero_area`'s `[2^-485, 2^510]`.

## What the bounds change

- Float32 and bfloat16-cast queries: the band is `u32 (|x| + R_k) + C u64 (...)`; going from 13 to 91 units moves it
  by `78 u64 / u32 = 1.45e-7` relative.
- Float64 queries: the band goes from `14 u64` to `92 u64` per unit (`6.6` times).
- Measured on every canonical dump: `RESULTS.md`, variant V1F (job 368608): no float32 decision changes; float64 ties change at 20 DrivAerML, 103 lug and 3 AeroZoo queries.
