# AF-EXTRAP results: ISLA 0.9.0 on AirfRANS's Reynolds and angle-of-attack extrapolation splits (2026-10-22)

*ISLA family fix-trained (code_cmpaf_fix, PREREG §12). Its in-distribution reference is the fix-trained full_test standing
r090fix_af_isla / r090fix_af_dtt (like for like with the afxf arms), not the eager-trained standing r090eag_af_*. REF-090-CLOSE
measured that the fixed compiled path trains the same models on AirfRANS as the bug path: r090fix ÷ r090 = 1.024 / 0.985 /
0.998, inside the seed gaps (scratch/isla-ref090-2026-10-20/r090_final_tables.md, section "REF-090-FIX AirfRANS").*

Registered rules: PREREG.md §6, with §11 (S-RE), §12 (fix relaunch) and §13 (clamp diagnostic, post hoc). Every lane scored
float32 on its test side, 100,000 points, final checkpoint (epoch 500 checked); reynolds 496 cases, aoa 196, full_test 200.
Reductions: `reduce_afx.py` (registered rules → `afx_results.json`, `afx_tables.md`), `posthoc.py` (→ `POSTHOC.md`,
`posthoc.json`), `clamp_reduce.py` (→ `clamp.json`). Score trees: `scores/` (fetched from LHR `$T/evals_afx`,
`$T/evals_afx_clamp`, `$T/evals_r090fix`). Jobs: JOBS.md.

## 1. Test error, in-distribution error and penalty (seed means; per seed in afx_tables.md)

| arm | in distribution (full_test) p / v / nut | reynolds test p / v / nut | g = test ÷ in-dist | aoa test p / v / nut | g |
|---|---|---|---|---|---|
| ISLA 0.9.0 | 0.0122 / 0.0063 / 0.0156 | 0.0389 / 0.0195 / 0.0457 | 3.18 / 3.08 / 2.94 | 0.0485 / 0.0163 / 0.0538 | 3.97 / 2.57 / 3.46 |
| ISLA-DTT 0.9.0 | 0.0127 / 0.0071 / 0.0215 | 0.0335 / 0.0175 / 0.0590 | 2.65 / 2.46 / 2.75 | 0.0528 / 0.0168 / 0.0745 | 4.17 / 2.36 / 3.47 |
| GeoTransolver 256 | 0.0109 / 0.0400 / 0.0171 | 0.0246 / 0.0466 / 0.0467 | 2.26 / 1.17 / 2.74 | 0.1809 / 0.0880 / 0.2313 | 16.6 / 2.20 / 13.5 |
| Transolver 512 × 16 | 0.0109 / 0.0297 / 0.0178 | 0.0274 / 0.0381 / 0.0466 | 2.51 / 1.28 / 2.62 | 0.0856 / 0.0493 / 0.0938 | 7.85 / 1.66 / 5.27 |

## 2. Registered verdicts

**reynolds.** R1 (standing on the test set): ISLA trails both baselines on pressure (B ÷ ISLA 0.632, 0.703), leads on velocity
(2.39, 1.96), is level on nut (1.02, 1.02). R2 (penalty): **ISLA extrapolates worse** on pressure (ρ 0.709 against
GeoTransolver, 0.789 against Transolver) and velocity (0.379, 0.417), both seed pairs agreeing; nut no difference (0.932, 0.892).
Task summary: **"ISLA extrapolates worse than the baselines"** (4 worse, 0 better, 2 no difference).

**aoa.** R1: ISLA leads both baselines on every field (B ÷ ISLA: GeoTransolver 3.73 / 5.41 / 4.30; Transolver 1.77 / 3.03 /
1.74). R2: **ISLA extrapolates better** on pressure (ρ 4.19, 1.98) and nut (3.92, 1.52); velocity no difference against
GeoTransolver (0.856) and **worse** against Transolver (0.646). Task summary: **"mixed / no difference"** (4 better, 1 worse,
1 no difference): the registered "none against" clause fails on the Transolver velocity cell.

**R3, ISLA-DTT ÷ ISLA (beside):** level or ISLA leading on R1 (reynolds 0.861 / 0.900 / 1.292; aoa 1.088 / 1.035 / 1.386); R2
no difference on all 6 cells (reynolds ρ 0.832 / 0.799 / 0.936; aoa 1.051 / 0.919 / 1.004).

Bands (2σ, σ pooled over 8 seed pairs per task and field): reynolds σ 0.069 / 0.082 / 0.071; aoa 0.099 / 0.067 / 0.099. Both
tasks' seed noise is larger than in distribution, so every R2 band is above the 1.10 floor (×1.21-1.32).

## 3. Beside the rules (post hoc; the verdicts above stand)

**3a. Absolute standing next to each penalty.** On reynolds ISLA leads both baselines 2.0-2.4× on velocity, trails on pressure
(0.63-0.70) and is level on nut. On aoa it leads everywhere, 1.7-5.4×.

**3b. Velocity has a floor; pressure does not.** The baselines' in-distribution velocity error (0.030-0.040) is already close to
their extrapolation error, so their velocity penalty (1.17-1.28 on reynolds) is small by construction, while ISLA's in-distribution
velocity is 4.7-6.3× lower. The ratio of ratios ρ then penalizes the low floor. Excess error (POSTHOC.md §1), ISLA ÷ baseline:

| reynolds | additive E − I p / v / nut | quadrature sqrt(E² − I²) | aoa additive | aoa quadrature |
|---|---|---|---|---|
| ISLA ÷ GeoTransolver | 1.95 / 1.98 / 1.02 | 1.68 / **0.77** / 0.99 | 0.21 / 0.21 / 0.18 | 0.26 / 0.19 / 0.22 |
| ISLA ÷ Transolver | 1.62 / 1.56 / 1.05 | 1.47 / **0.77** / 1.00 | 0.49 / 0.51 / 0.50 | 0.55 / 0.38 / 0.56 |
| ISLA-DTT ÷ GeoTransolver | 1.52 / 1.56 / 1.27 | 1.41 / 0.67 / 1.26 | 0.24 / 0.20 / 0.25 | 0.28 / 0.19 / 0.31 |
| ISLA-DTT ÷ Transolver | 1.27 / 1.23 / 1.30 | 1.24 / 0.67 / 1.28 | 0.54 / 0.49 / 0.70 | 0.60 / 0.39 / 0.78 |

So the reynolds pressure "worse" is robust under every measure (additive 1.95 / 1.62, quadrature 1.68 / 1.47). The reynolds
velocity "worse" and the aoa Transolver velocity "worse" are floor-driven: under the quadrature excess both flip to ISLA better
(0.77; 0.38). Pressure is the clean evidence.

**3c. Per-side slopes (reynolds, error per decade of Re outside the band; POSTHOC.md §2).** Below the band (low Re) ISLA's error
grows fastest: p 0.62, v 0.32, nut 0.76 per decade (Spearman 0.84 / 0.90 / 0.88), against GeoTransolver 0.16 / 0.020 / 0.44 and
Transolver 0.32 / 0.062 / 0.45. Above the band the slopes are small for all: ISLA 0.21 / 0.14 / 0.30, GeoTransolver 0.11 / 0.065 /
0.38, Transolver 0.042 / 0.062 / 0.50 (ISLA's nut grows less than both). In the farthest below-band quartile ISLA's pressure error
is 0.100 against 0.045 / 0.054. On aoa the baselines' error climbs steeply with distance on both sides (Spearman 0.82-0.97), ISLA's
much less (0.19-0.66; slope below the band 0.067 / 0.016 / 0.031 per degree against 0.24 / 0.063 / 0.21 for GeoTransolver).

**3d. Matched cases (reynolds).** On the 103 full_test cases outside 3e6-5e6 (all inside reynolds_test), reynolds-trained ÷
full-trained on the same cases: g = ISLA 2.59 / 2.63 / 2.75, ISLA-DTT 2.05 / 2.12 / 2.61, GeoTransolver 1.78 / 1.17 / 2.37,
Transolver 1.81 / 1.26 / 2.35; ρ_matched GeoTransolver ÷ ISLA 0.69 / 0.44 / 0.86, Transolver ÷ ISLA 0.70 / 0.48 / 0.86. The
same picture as the registered ρ.

**3e. S-RE, the data-size control (registered §11; triggered by the four reynolds "worse" cells).** 504-case in-distribution
error (full_test): ISLA 0.0168 / 0.0082 / 0.0203, GeoTransolver 0.0148 / 0.0446 / 0.0206, Transolver 0.0132 / 0.0326 / 0.0203.
I_800 ÷ I_504: ISLA 0.73 / 0.77 / 0.77, GeoTransolver 0.74 / 0.90 / 0.83, Transolver 0.83 / 0.91 / 0.88 (ISLA gains most from
data). With I_504 in place of I_800: GeoTransolver ÷ ISLA ρ_504 0.716 (pressure, worse), 0.440 (velocity, worse), 1.008 (nut);
Transolver ÷ ISLA 0.898 (pressure, **no difference**), 0.493 (velocity, worse), 1.022 (nut). So the 37 % smaller training set
explains the Transolver pressure cell, not the GeoTransolver pressure cell nor the (floor-driven) velocity cells.

**3f. Input-space reach and the clamp test (PREREG §13, registered before running; post hoc diagnostic).** ISLA reads
z = (log10 Re − 6.223) / 0.06224 with its own length: train −2.05 to 1.80, test −4.85 to 3.04. The baselines read the standardized
speed u: train −1.80 to 1.75, test −3.57 to 3.54. Below the band ISLA reaches 2.81 units beyond its edge, the baselines 1.78; above,
1.24 against 1.79 (log against linear). Per-case error tracks each model's own reach below the band (Spearman ISLA 0.86 / 0.92 /
0.88; GeoTransolver 0.42 / 0.00 / 0.82). Clamping each model's input to its training range at inference (LHR job 335143;
`kernel/afxc`), clamped ÷ native error, seed means:

| arm | below the band p / v / nut | above the band p / v / nut | non-degeneracy |
|---|---|---|---|
| ISLA | **0.469** / **0.816** / 1.098 | 0.814 / 0.906 / 1.351 | the 45 cases inside z's range unchanged (Δ 0); all 902 outside case-seeds changed |
| ISLA-DTT | **0.584** / 0.924 / 0.872 | 0.884 / 1.114 / 1.505 | inside unchanged (≤ 1.8e-6); all outside changed |
| GeoTransolver | 0.860 / 1.019 / 2.93 | 0.811 / 1.005 / 2.42 | no case inside; all changed |
| Transolver | 0.654 / 0.994 / 3.25 | 0.927 / 1.013 / 2.17 | no case inside; all changed |

Below the band, clamped, the pressure errors meet: ISLA 0.0265, ISLA-DTT 0.0269, GeoTransolver 0.0261, Transolver 0.0248 (native
0.0564 / 0.0461 / 0.0303 / 0.0379). Reading as registered: **ISLA's extrapolation in z is actively harmful for pressure** (and
velocity below the band; both seeds, beyond 2σ). The baselines' pressure also improves when their speed is clamped (Transolver
0.65 below), less so. Nut needs the true scale: clamping costs every model on nut, the baselines 2-3× (their nut is in m²/s, so
the speed carries its magnitude), ISLA 1.1-1.35× (its nut is in units of U L, so the clamp only removes the shape's Re
dependence). Hypothesis, not a finding: ISLA's reynolds pressure penalty is input-space amplification, from reading log Re
standardized by a narrow train-split scale far outside its training range (the narrow s alone is harmless, since a linear first
layer absorbs any fixed affine scale; what matters is that low-Re test cases sit 2.8 training ranges beyond the edge in z, but
only 1.8 in the baselines' linear speed). A training control (a split-independent log Re scale, about 57 GPU-h) waits for the
supervisor's approval.

**3g. Other checks.** Posed ÷ native: ISLA and ISLA-DTT within 1 ± 0.0008 on both tasks; the baselines 4.5-91 (not equivariant).
nut scoring-space factor (train-split μ against 7.9e-4): reynolds 0.9997-1.0053 (mean 1.0007), aoa 0.957-1.0033 (mean 0.9961).
nut in the 7.9e-4 space moves under 0.5 % (afx_tables.md).

## 4. Predictions (PREREG §7)

P1 holds (ISLA leads on velocity, 4 cells). P2 fails (nut level on reynolds). P3 fails (ISLA trails on reynolds pressure). P4
fails (nut no difference). P5 holds (aoa pressure ρ 4.19). P6 holds. P7 fails on both tasks (reynolds worse; aoa mixed). P8 holds
(every g > 1; smallest 1.17). P9 holds (ISLA-DTT ÷ ISLA 0.86-1.39; R2 no difference on 6 of 6). P10 holds (within 1 ± 0.0008).
P11 fails on one cell (ISLA-DTT nut above the band, Spearman 0.24; every other arm and side > 0.3). P12: the 8 baseline lanes and
the 8 relaunched ISLA-family lanes finished cleanly; the 8 first ISLA-family lanes were cancelled by the supervisor for the
compiled-path defect (not a failure of the lanes).

## 5. What it means

- AoA extrapolation, AirfRANS: ISLA (and ISLA-DTT) extrapolate far better than both baselines on pressure and eddy viscosity
  (penalties 4.0 / 3.5 against 16.6 / 13.5 and 7.9 / 5.3), and the velocity "worse" against Transolver is floor-driven.
- Reynolds extrapolation, AirfRANS: ISLA extrapolates worse on pressure, robustly. Of the four "worse" cells, one (Transolver
  pressure) is data size (S-RE) and two (velocity) are floor-driven. The remaining clean cell (GeoTransolver pressure) and the
  below-band slope point at ISLA's own Re input: clamping z to its training range removes the pressure gap below the band.
- ISLA-DTT behaves like ISLA on both tasks (R2 no difference everywhere), with a smaller reynolds penalty (2.65 / 2.46 / 2.75
  against 3.18 / 3.08 / 2.94) and a lower clamp sensitivity.
- Confidence: the reynolds pressure and aoa pressure / nut results are large against the bands (σ-pooled 2σ) and hold under the
  additive, quadrature and matched-case readings; the velocity cells depend on the measure. Two seeds per arm.
