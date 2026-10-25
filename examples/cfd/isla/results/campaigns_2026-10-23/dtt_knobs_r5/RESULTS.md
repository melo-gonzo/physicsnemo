# DTT-KNOBS stage 2: results (PREREG.md md5 39d8272f)

Read by the coordinator. Every rule is read as registered. Sources: `reduce/out_r5/` (reduce_s2.py md5 e2b605d2) and
`reduce/out_r5/r5_registered_arith.txt` (the registered arithmetic-mean form, recomputed by the coordinator).

## R5: DTT-NOSC-STD against B (read 2026-10-24 22:58 EDT; fix-trained; second-read and corrected 23:30, RESULTS_R5_REVIEW.md)

The runs dttknobs_s2_noscstd_seed42 and seed43 trained 500 epochs. They were scored by job 376686 (array task 6: native and
posed, 200 full_test cases each, float32, eager, code_att090fix + recipe_att090b). Every section-4 check passes: epoch 500,
200 cases with the control's sample ids, the snapshot, the dataset and the extra. B is r090fix_af_dtt 42 / 43. DTT-NOSC is
att2_af_dttnosc. σ_f is ATTRIB 3c's six-arm pool plus DTT-NOSC-STD's two seeds (7 dof): 0.0503 / 0.0866 / 0.0744. The
large velocity value comes from DTT-NOSC-STD's own velocity seeds, which sit 36 % apart (0.00726, 0.00988). The
reducer reproduces ATTRIB's 6-dof σ (0.0422 / 0.0291 / 0.0637) and G_f to the registered three decimals.

| field | DTT-NOSC-STD (s42, s43) | ÷ B, arithmetic (registered) | 2 SE | ÷ DTT-NOSC | A | N | **registered reading** | geometric form (beside) |
|---|---|---|---|---|---|---|---|---|
| pressure | 0.014377, 0.016185 | 1.208 (Δ +0.189) | 0.101 | 0.955 | yes | no | **standardization leaves it (the scales carry it)** | same (Δ +0.187) |
| in-plane velocity | 0.007264, 0.009881 | 1.203 (Δ +0.185) | 0.173 | 0.982 | yes | no | **standardization leaves it (the scales carry it)** | unresolved (Δ +0.173 < 2 SE): **flagged** |
| nut | 0.023150, 0.027435 | 1.177 (Δ +0.163) | 0.149 | 1.018 | yes | no | **standardization leaves it (the scales carry it)** | same (Δ +0.162) |

Per seed, ÷ B: pressure 1.128 / 1.288, velocity 1.022 / 1.382, nut 1.164 / 1.188. Per seed, ÷ DTT-NOSC: 0.916 / 0.992,
0.813 / 1.159, 0.909 / 1.134. In the registered form, φ = Δ / G_f is 0.81 / 0.91 / 1.11 and Δ / D_DTT is
1.83 / 1.18 / 6.26 (geometric form: 0.80 / 0.85 / 1.10 and 1.81 / 1.11 / 6.23). Posed agrees within 0.001 on the ratios
(1.207 / 1.202 / 1.177) and gives the same labels; it is not an independent number.

**Registered readings.**
- R5 per field: standardization leaves DTT-NOSC's loss on pressure, velocity and nut (the scales carry it). On velocity
  the registered arithmetic form clears 2 SE by 7 %; the geometric form would read "unresolved". This is flagged, and
  the registered label stands.
- Bundle thread (Q4), on pressure and nut: C_f, the part of ISLA-DTT's CONT-to-0.9.0 bundle other than the scales,
  measured with the raw speed, **sits in the scales, as far as two seeds can show**. There is no sign that it is an
  interaction with the raw speed: with the speed standardized and no scales, ISLA-DTT is still about 20 % worse than B,
  and not measurably better than DTT-NOSC (pressure ÷ DTT-NOSC 0.955, unresolved).
- Prediction P10 ("removes" on pressure, p 0.60) failed. P12 (nut "unresolved", p 0.45) failed.
- Labels: fix-trained.

**Beside the rule (not a verdict).**
- The two forms read differently. Interacting ISLA's mirror, NOSC-STD ÷ B (ATTRIB 2c, geometric seed means), read
  1.003 / 1.029 / 1.043: no measurable effect, "the input standardization carries the step; the scales do not". No
  registered rule tests the difference between the forms. A plain 2-SE check of Δ_DTT - Δ_ISLA, with
  SE = sqrt(σ_7^2 + σ_2c^2) and 2c's 8-dof 0.063 / 0.045 / 0.059, resolves it on pressure only (0.186 against 2 SE
  0.161). On velocity (0.156 against 0.195; ISLA-DTT's own A is the flagged one and rests on seed 43) and on nut (0.121
  against 0.190), it is within two seeds' noise.
- For the scope freeze: on pressure, two seeds show that ISLA-DTT uses the scales where interacting ISLA does not.
- A reading of why, not a test: ISLA-DTT's passive points never share state with one another. The physical scales may
  give them a frame that interacting ISLA rebuilds from the coupled points.
- Cost: a median step of 0.574 / 0.571 s and 67.1 / 66.2 GPU-h per run (sacct), against B's 0.566 / 0.575 s and
  66.1 / 67.0.
  The arm used 133.3 GPU-h of its registered estimate of 136.
