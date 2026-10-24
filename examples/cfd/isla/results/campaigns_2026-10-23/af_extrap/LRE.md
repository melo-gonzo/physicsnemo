# LRE and LIN readout (PREREG §14, §14b)

ISLA family fix-trained (code_cmpaf_fix / code_afxlin); reference for R2 r090fix_af_isla.

## LRE

- (a) below-band pressure, LRE ÷ afxf ISLA: s42 0.530, s43 0.576 → **the fixed scale helps**
- (b) in-band fit guard (504 training cases; fit inside the band, not generalization): 0.912 / 0.938 / 0.944 → **passes**
- (B) near-band held-out (overlaps the population of (a)): below 0.862 / 0.878 / 0.817, above 1.234 / 1.215 / 5.253
- test error p / v / nut 0.0246 / 0.0111 / 0.0364; g against r090fix_af_isla 2.01 / 1.76 / 2.34; below-band slope per decade 0.156 / 0.063 / 0.201
- (c) gt p: R1 0.998 level, R2 ρ 1.120 (1.203, 1.034) [×1.195] LRE no difference; gt v: R1 4.197 leads, R2 ρ 0.664 (0.641, 0.690) [×1.116] LRE extrapolates worse; gt nut: R1 1.285 level, R2 ρ 1.171 (0.839, 1.766) [×1.556] LRE no difference; t p: R1 1.111 level, R2 ρ 1.246 (1.260, 1.231) [×1.195] LRE extrapolates better; t v: R1 3.430 leads, R2 ρ 0.731 (0.678, 0.790) [×1.116] LRE extrapolates worse; t nut: R1 1.282 level, R2 ρ 1.120 (0.877, 1.553) [×1.556] LRE no difference
- (d) clamp, clamped ÷ native: below 0.876 / 1.627 / 2.130 (pressure s42 0.859, s43 0.894; clamped below-band pressure 0.0263), above 0.991 / 1.241 / 1.440; non-degeneracy: 45 cases inside unchanged (max |Δ| 0.0e+00), 902 outside case-seeds changed

## LIN

- (a) below-band pressure, LIN ÷ afxf ISLA: s42 0.620, s43 0.632 → **helps**
- (b) in-band fit guard (504 training cases; fit inside the band, not generalization): 1.020 / 1.003 / 0.989 → **passes**
- (B) near-band held-out (overlaps the population of (a)): below 0.927 / 0.955 / 0.967, above 1.073 / 1.038 / 1.021
- test error p / v / nut 0.0311 / 0.0161 / 0.0419; g against r090fix_af_isla 2.55 / 2.55 / 2.69; below-band slope per decade 0.230 / 0.134 / 0.504
- (c) gt p: R1 0.790 trails, R2 ρ 0.886 (0.952, 0.819) [×1.195] LIN no difference; gt v: R1 2.891 leads, R2 ρ 0.458 (0.418, 0.507) [×1.197] LIN extrapolates worse; gt nut: R1 1.115 level, R2 ρ 1.016 (0.909, 1.132) [×1.193] LIN no difference; t p: R1 0.879 trails, R2 ρ 0.986 (0.998, 0.974) [×1.195] LIN no difference; t v: R1 2.363 leads, R2 ρ 0.504 (0.442, 0.581) [×1.197] LIN extrapolates worse; t nut: R1 1.113 level, R2 ρ 0.973 (0.950, 0.996) [×1.193] LIN no difference
- (d) clamp, clamped ÷ native: below 0.752 / 1.250 / 1.418 (pressure s42 0.722, s43 0.786; clamped below-band pressure 0.0256), above 0.694 / 0.732 / 1.065; non-degeneracy: 45 cases inside unchanged (max |Δ| 0.0e+00), 902 outside case-seeds changed

**Joint reading (§14b): both help: both.** (LRE (a) the fixed scale helps; LIN (a) helps.)

