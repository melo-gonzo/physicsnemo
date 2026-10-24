# Review of DERIVATION.md (md5 57110c08): independent re-derivation, 2026-10-24

**Verdict.** By my own count, C = 91 (3-D), C = 17 (2-D), K = 168 and the 1-unit switch are all certified (first order,
with the second-order terms bounded). The draft's argument does not show this: it over-counts the accurate foot's error
along the edge (25 u L; the true figure is 19.5 u L) and omits the worst path, a face foot inside the face that rounding
rejects. The two mistakes roughly cancel. By the draft's own counts the omitted path is 98 U, and C would have to be 100.
U = u64 (|x'| + R'). l <= |x'| + R' and L <= 2 R' are the triangle inequality in the centred frame, so they are valid.

## Items 1-6
1. **K = 168: agree.** Steps 1-2 (5 u L l; 22 u L^2 l^2 + u |va|) and the step-3 identity check. On the face
   sum |v_i| = D, so the u |v| terms and r give 3 u L D (the comment's 5 over-counts, harmlessly). Step 4 (Warp,
   4 u L + 5.5 u l) checks; K = 101.25 / (1 - 66/K) = 167.25. gamma_n and the error products add a relative O(30 u),
   against a 0.45% margin. Three flaws in the comment, none of which changes K:
   (a) min(Delta, Delta^2 / (2 (d~ - Delta))) bounds only the in-plane misplacement (steps 3, 6). Step 4's Warp roundings
   and the torch evaluation's are linear in u l and do not shrink with d: on a fat cell with the query at height L,
   eps ~ 4e-26 L while Warp's rounding is ~1e-16 L. `_beyond` is safe only because the radius's third err_max is
   >= 23.5 U, which holds at C = 91 but not for float64 queries at the package's 13 + 1.
   (b) `_ericson_error` receives the computed `dist`, up to a = 9.5 U above d(P'), so "eps < 1 unit" certifies only
   eps <= (1 + sqrt(1 + 8a)) / 2 = 4.9 U (6.4 U on edge pairs, where a = 17.5). (c) "over 166.8" should read 167.25.
2. **Edge point: agree.** 3-D: 5 u l + 6 u t |s| <= 11 u l along the edge, assembly 3 u R (the zero weight leaves one
   effective sum), difference and norm 3.5 u l: 14.5 u l + 3 u R = 17.5 U. 2-D: 9 u l, 3 u R, 3 u l: 12 u l + 3 u R = 15 U.
3. **Accurate foot: disagree with the draft; the code comment's delta is right.** Along e: 9.5 u l + u L (9.5 lambda_C
   [tc~] + lambda_C [product] + 5.5 lambda_B [difference, division, lab~]), since those roundings scale with
   |lambda_B~ lab| = |s1 - lambda_C tc|, not with L. Along m: 19 u l + 9 lambda_C u L (|s2| = lambda_C hc). On an
   admitted face lambda_B~ + lambda_C~ <= 1 + u, so delta = 28.5 u l + 19.5 u L; the draft's 16 u L adds the two
   maxima separately. Missing in both: lambda_A~ >= 0 lets P~ lie up to u h_A <= u L outside BC, so the admitted face
   gives 9.5 u l + delta + u L = 38 u l + 20.5 u L = **79 U** (the comment says 77, the draft 88).
   **Missing in both, and the worst case:** F inside the face, rejected by some lambda~ < 0. Then d = |h| and d_acc,
   the least edge candidate, is <= sqrt(h^2 + s^2) + e_edge, where s = dist(F, boundary) <= delta + u L:
   **e_acc = 43 u l + 20.5 u L + 3 u R = 87 U**. "The least of the candidates errs by at most the largest of their
   errors" fails here, because the exact minimum's face term has no computed counterpart. In this branch lambda~ is
   unbounded, so the draft's Condition 2 does not apply: (lambda_C~ - lambda_C)(tc - tc~) ~ (19 u l + 9 u L) 9.5 u L / hc.
   Per crossed line: for AC it is damped by sin = hc/|AC| to O(u) m_err; for BC it is <= 16 u L m_err / hc, and
   s <= min(hc, ...) keeps it at <= ~67.5 U, inside the 69.5 U linear bound. This argument must be written out.
   A switched pair, recomputed from the weights: face 32 u l + 19.5 u L + 5 u R = 76 U; edge after a rejected face 87 U.
4. **Arithmetic share (3-D): agree with 89 and C = 91, not with the route.**
   - Unguarded Ericson face: eps (<= 4.9, item 1b) + 9.5 = 18.4. My torch count: sum w - 1 2 u R, divisions u R,
     assembly 3 u R, difference and norm 3.5 u l; the draft has 13.5.
   - Edge: 17.5 + eps (<= 6.4), so <= 24; the draft omits eps. Vertex: 3.5 + eps.
   - Switched: <= 87. Guarded and kept: 1 + 87 = 88.
   - Zero-area triangle (absent from the draft): 17.5 + a wrong pick between its edges by `_segment_distance2`
     (2 x 16.5 u l), so <= 51.
   - Worst 88 + 1 (centring) = 89, which leaves 2 U for the band's evaluation and second order. 2-D: 15 + 1 <= 17.
5. **Switch: agree, given the corrected e_acc.** A kept pair has |D_E - d| <= |D_E - d_acc| + |d_acc - d| <=
   tau + e_acc. The directions are right: fl(|d_acc - dist|) <= unit implies the exact value is <= unit (1 + 2u). A
   switched pair needs the recomputed distance (above), not d_acc; the draft has this right. The guard and the switch
   do not interact: kept and switched bounds do not use eps, and only unguarded pairs rely on eps < 1 (item 1b).
   Same unit, confirmed: in variant.py both `_closest_weights` (q64[li] and tri, after `q64, simplex = q64 - origin,
   simplex - origin`) and `band()` (q_norm and reach, recomputed after centring) use u64 (|x'| + R'_k).
   P2: `switch` uses `>`, so a NaN `dist` keeps Ericson's weights (the guard uses `~(<)` for this reason). A CPU probe
   of 1.6M random float64 slivers produced no non-finite weights.
6. **Omissions.**
   - Band evaluation: rounding is monotone, so fl(D - E) <= fl(D + E) and fl(d_min - E) <= 0 need no margin, and err's
     relative error of about 5u is second order. The draft should say so.
   - FMA: right for the bounds. Add that `_two_product` (Dekker splits) and `_two_sum` must not be contracted; they rely
     on eager torch, one kernel per op (a custom op, which torch.compile does not trace).
   - G5 (rebuild=f64): uses only differences, so neither the frame nor sum w != 1 matters, and it does not feed the
     band. Fine, but state it.
   - 2-D brute force: 2 (32 u_in (|x'| + R'max) + err_max) must cover 2 (e_bf + A), with e_bf = 14 u l + 1 (centred
     copies) + 2.5 (reach's root, sum and square) = 17.5 and A = 16. Float32 queries are fine (A ~ 3e-8 u32 units).
     For float64 queries, 33.5 > 32 by my first-order count: raise 32 to about 40, or derive it.
   - 3-D ball radius: E + A + (Warp error - eps) <= 3 err_max holds at C = 91.
   - Centring: the 29-bit qualification is right, and the 1 U covers the inexact cases.
   - Ranges: float32 inputs stay normal in float64. Float64 inputs need L l < 2^511 and products normal, roughly
     |coordinates| < 1e76 and nonzero differences > 1e-76, narrower than `_zero_area`'s [2^-485, 2^510]. State it.

## Corrections, most important first
1. DERIVATION (c)6 and (d): add the rejected-face case. e_acc = 87 U; switched = max(76, 87); kept = 1 + 87;
   worst 89. (Without it, the draft's 25 u L gives 98 U and C = 100.)
2. DERIVATION (c)4-5: along e, 9.5 u l + (10.5 lambda_C + 5.5 lambda_B) u L; delta = 28.5 u l + 19.5 u L. Withdraw
   the claim that GEO-THIN under-counted, and add the u L from lambda_A~'s rounding.
3. DERIVATION Condition 2: it holds for an admitted face only; add the per-line argument for the rejected branch.
4. Comment above `_FOOT_UNITS`: steps 5-6 become 87 U (the rejected case and the u L); with VF, drop "the design's
   figure 12". `_closest_weights` docstring: the switch is certified by C >= tau + e_acc + 1.
5. Comment above `_ERICSON_K`: the Pythagorean form covers the in-plane part only, and step 4 is linear. `dist` must
   be a lower bound (or state the 4.9 U). 166.8 becomes 167.25. 5 u L D to 3 u L D is optional.
6. Comment above `_ERICSON_UNITS` and the `_closest_candidates_impl` docstring: 13 becomes C = 91 / 17, derived.
7. DERIVATION (d): eps on the edge and vertex rows, 4.9 on the unguarded row, a zero-area row; item 6: the 2-D
   brute-force margin for float64 queries.
