# Review of NEARWALL DESIGN.md (lane-reviewer, read-only, 2026-10-24)
Checked against RESULTS.md, the tables, out/*.json, the harness probes, geometry.py, model.py, and my own numpy
point-to-segment probe in /tmp (float32, frame offsets T = 0, 30, 305 chords).

## Findings (most severe first)

1. **P1. The precision argument is not established (DESIGN.md:30-35, §4 item 2; RESULTS.md:94-102).**
   - nwprec.py:73-79 translates in float64, casts q and the vertices to float32, and compares with d64 in the native frame.
     The cast re-rounds the case's own nodes at ulp(T), so the model is given a different geometry.
   - Probe: a node 1.6e-6 chords off the wall, stored at T = 30 chords, is up to 0.98 of its distance wrong (on the wall at
     305). That error is in the input: no encoding can undo it, and the target is as wrong.
   - ISLA's own share comes from the absolute closest point (geometry.py:1722 `closest = sum w_i v_i`, :1817
     `diff = query_points - closest`): up to 0.74 relative at 30 chords and 5.7 at 305.
   - In the closest cell's local frame, `(q - v0) - sum w_i (v_i - v0)` with the subtraction in the input's precision, the
     same float32 arithmetic stays below 6e-5 at every T.
   - So "any log code for these nodes depends on the frame's origin" does not follow; it describes the current kernel and
     float32 storage at a hypothetical offset. Fix: rerun A3 against d64 of the translated-then-cast float32 coordinates,
     and restate F as "no rounding beyond the input's".
2. **P1. F is judged at an unscored offset (DESIGN.md:34-35, §3 (ii), §4 item 1).** The 3-4 failures of (ii) and (iv)
   appear only at T = 0.1 L whole = 30 chords. At the posed offset (about L body) every candidate is at or below 0.036
   (precision table, AirfRANS rows). "F fails" needs that qualifier; the rules' F is float32 vs float64 on one input.
3. **P1. The physics is overclaimed (§4 item 2: "nearly linear in d, so the linear code matches the physics").**
   RESULTS.md:47-50, 74-75: the linear part reaches the turbulent boundary layer (nut / nu about 1 at t 0.3-1), sits at
   y+ up to about 25-90 (estimated), and "the buffer layer is where a log code helps". u is linear only below y+ about 5.
   Fix: the linear part covers the sublayer, the buffer layer and the start of the log layer.
4. **P2. A missed candidate.** `asinh(s_k / 1e-6)`, `s_k = d / L_k`, L_k = `ISLA._gauge` of the declared condition that
   holds the closest point, with d from the local-frame kernel of finding 1.
   - It keeps P, U, R and D (L_k converges under refinement), uses only the named meshes (model.py:80-90), adds no
     constant, and rounds frame-free.
   - AirfRANS's first nodes return to t of about 3.8 and DrivAerML's car nodes to about 33, with no wall declaration.
   - Cost: a jump of ln(L_j / L_k) where two conditions are equidistant (about 6.6 airfoil/far field, some 150 chords
     out). Untested; the jump may rule it out. Also, (ii) or (iv) with the local-frame kernel pass F as the rules define it.
5. **P2. First-node position.** DESIGN.md:32 and RESULTS.md:95 say 6.5e-9 L, DESIGN.md:49 says 5e-9 L. e_af.json: first
   off-wall t median 0.0052 (min 0.0047), that is 5.2e-9 L, 11.5x below 2^-24, not 10x.
6. **P2.** DESIGN.md:40 gives the margin as 16.8 x 2^-24, RESULTS.md:97 as 16 x (exactly 16.78); DESIGN.md:40 also says
   (i) keeps the docstring's "at most about 0.1", but A3 measures 0.17 at T = L whole.
7. **P2.** §4 item 1 cites "up to 4 for (ii), (iv), (v), (vi)", but (v) and (vi) were not run. The 1e-9 L arm was run
   (3.6 at T = 0.1 L, p_af.json) and is not tabulated. Label (v) and (vi) inferred.
8. **P2.** §3 (iv) says it needs "a new per-boundary declaration". An empty schema already marks a no-slip wall
   (model.py:80-83, 538), though ISLA holds no wall meaning today ("identity is structural"): smaller cost than stated.

## Questions 1-2: numbers and probes

- Every other cited number matches the JSON (L, first-node t, linear, t < 10 and wall shares, 1e-6 Re_L = 612-1,830,
  2^-24 Re_L = 36-109, the DrivAerML wall gauge 38.3 m and t 1.23, the 38x and 200x ratios, the contingency's GPU-h).
- nwprobe.py is sound: the hook reads qt.seed's input, whose last two columns are `builtin` (model.py:1957-1962).
  The counterfactual `s * L_standing / L_whole` is a fair same-weights d / L_whole for this channel (other seed columns
  stay at standing values). nwprec.py does not fairly measure float32 precision of d (finding 1).

## Question 4: is the recommendation supported?

- Keeping (i) unchanged until AF-WHOLE prices the harm is defensible: (ii) is undefined on 3 of 5 datasets, and (iv)
  needs wall semantics. But "only candidate keeping every invariance" and "every candidate that resolves them does so
  through the frame's origin" rest on findings 1 and 2. The A2 caveats (no retrain, no field accuracy) are honest.
- The docstring should not repeat the 2^-24 L claim: say the channel is linear below 1e-6 L (y+ up to about 25-90 on
  AirfRANS with its far field). The contingency should pair (ii) with the local-frame kernel or a re-measured A3.

Verdict: sound with fixes: rerun A3 and recast F (findings 1-2); correct the physics claim (3); list the local-frame
kernel and the nearest-condition gauge (4); fix the numbers (5-7). Keep-(i) stands, on narrower grounds.
