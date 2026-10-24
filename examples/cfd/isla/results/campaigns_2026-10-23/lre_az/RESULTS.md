# LRE-AZ results: ISLA's fixed Reynolds scale on AeroZoo's reynolds split (read 2026-10-24 12:02 EDT)

PREREG.md (md5 at approval f9964333, approved 2026-10-23 01:37 EDT; erratum to the section 5 noise note, md5 e00823cd, rule
unchanged). Arms: A = `azc_reynolds_isla_seed4x` (fitted z, reused), B = `lreaz_reynolds_isla_seed4x` (z = log10(Re_ISLA /
1e6), recipe_lreaz). Tables: `tables.md` (registered reduction, `reduce_lreaz.py` unchanged since registration), `lreaz.json`;
post hoc beside: `posthoc.md`. Jobs and checks: JOBS.md.

## 1. Checks before reading: all pass

- Every score row: epoch 250, infer_split test, 1,385 cases with the split's case ids, SNAPSHOT_USED code_r090.
- Reynolds constants, all four registered parts, per row: B shows {6.0, 1.0} (row dataset, scorer Hydra config, scorer
  RECIPE recipe_lreaz with file md5 eb37789f, checkpoint args.json). A shows {5.062, 0.8985} (recipe_r090, f7cc0d6d). Smoke
  345967 had reynolds={'mean': 6.0, 'std': 1.0} on all four ranks, and z over one epoch spanned -2.7533 to 1.0402.
- Event guard: no hit. Largest val-loss / running-minimum ratios are B 1.034 and 1.110 (A 1.026 and 1.092).

## 2. Registered verdict (R1; case means, reynolds test, 1,385 cases)

| field | B ÷ A = exp(D) | D | 2 SE | per seed (42, 43) | verdict | medians beside: exp(D), verdict |
|---|---|---|---|---|---|---|
| pressure | 1.067 | +0.065 | 0.100 | 1.020, 1.114 | **no change** | 1.080, no change |
| velocity | 1.126 | +0.118 | 0.046 | 1.139, 1.113 | **harms** | 1.068, no change |
| nut (beside) | 0.833 | -0.183 | 0.123 | 0.827, 0.838 | helps | 1.029, no change |
| tke (beside) | 1.004 | +0.004 | 0.013 | 1.013, 0.996 | no change | 1.049, no change |
| omega (beside) | 1.001 | +0.001 | 0.033 | 1.000, 1.002 | no change | 0.993, no change |

Velocity: D = 0.118 > ln 1.10 = 0.095 and |D| > 2 SE = 0.046 (s_A 0.029, s_B 0.013). Pressure: D = 0.065 is below
ln 1.10, and s_B = 0.071 makes 2 SE = 0.100.

**Consequence, as registered: "harms" on velocity, so the fixed scale is not adopted without Peter.**

## 3. Extrapolation profile (R2, descriptive)

B ÷ A, seed-mean case means, per bin of distance above the training band (label, decades):

| bin | cases | pressure | velocity | nut | tke | omega |
|---|---|---|---|---|---|---|
| [0, 0.5) | 411 | 1.011 | 1.007 | 1.006 | 1.001 | 1.001 |
| [0.5, 1) | 389 | 1.037 | 1.048 | 0.950 | 1.026 | 1.007 |
| [1, 1.5) | 312 | 1.112 | 1.186 | 0.837 | 1.030 | 0.994 |
| [1.5, 2] | 273 | 1.098 | 1.281 | 0.737 | 0.957 | 1.003 |

- Slope per decade of the per-case error (AF-EXTRAP's definition), A against B: pressure +0.228 and +0.274, velocity +0.037
  and +0.072, nut +1.02 and +0.69. Slope of the per-case ln(B ÷ A): pressure +0.123, velocity +0.120, nut -0.077 per decade.
  On AirfRANS below the band, pressure went from 0.62 (fitted) to 0.16 (fixed). Here the fixed scale steepens it.
- By ISLA's own log10 Re: on the 414 test cases inside ISLA's training band, B ÷ A is 1.015 for pressure and 1.007 for
  velocity. On the 971 cases above it, the ratios are 1.081 and 1.169 (nut 0.80).
- Post hoc, per seed (`posthoc.md`): velocity B ÷ A in the far bin is 1.321 (s42) and 1.245 (s43), and the rise with distance
  holds on both seeds. Pressure rises on both seeds too, more on s43. Velocity is harmed on 3 of the 4 families
  (external_box 1.12 and 1.10, opposed_hemisphere 1.23 and 1.25, internal_duct 1.23 and 1.14; opposed_sphere 0.99 and 1.02).

## 4. Predictions (PREREG section 7)

| id | prediction | p | outcome |
|---|---|---|---|
| P1 | smoke passes first time | 0.93 | yes |
| P2 | pressure no change | 0.80 | yes |
| P3 | velocity no change (harms 0.13) | 0.80 | no: harms |
| P4 | no harm on p or v, so the proposal goes ahead | 0.75 | no |
| P5 | nut, tke, omega all no change | 0.55 | no: nut helps on means (medians no change) |
| P6 | pressure B ÷ A within 0.95-1.05 in every bin | 0.55 | no (1.11, 1.10 far out) |
| P7 | abs. slope of ln(B ÷ A) on pressure < 0.05 per decade | 0.70 | no (+0.123) |
| P8 | lanes finish, no guard hit, no failure beyond a resubmit | 0.85 | yes (one node fault, recovered by the chain) |

I expected the first linear layer to absorb the affine reparametrization, so I predicted no change. That held near and
inside the band, but not far beyond it.

## 5. What it means

- The fixed scale is not dataset-neutral. On AirfRANS it halved ISLA's below-band pressure error. On AeroZoo it costs 13 %
  on velocity, concentrated 1-2 decades above the band (up to 1.28), with pressure 7 % worse (not significant by R1). Inside
  ISLA's own training band the two scales agree to about 1 %, so the difference lies in how the network extrapolates in z, not
  in how well it fits.
- The two-dataset condition for adopting it fails. Keeping per-split fitted constants keeps AirfRANS's deficit; adopting the
  fixed scale costs AeroZoo velocity. That choice is Peter's.
- Confidence: velocity harm, moderately high. Both seeds agree (1.14, 1.11), s_B is small, it grows with distance on both
  seeds, and it appears on 3 of 4 families. Medians move less (1.07), so the harm sits in the higher-error, far-out cases.
  Two seeds, one training configuration (bf16, compiled blocks, 250 epochs). The mechanism is untested. On this split the
  fixed z overshoots its training range by up to 2.09 units, against 2.33 for the fitted z, so the reach in z does not
  explain it.
- Not tested, possible next steps (each would need a new pre-registration): a fixed scale whose spread suits every dataset
  (for example log10(Re / 1e6) / s with one program-wide s); or adopting the fixed scale with an inference clamp of z.

## 6. Cost

The smoke used 0.79 GPU-h, training 184.6 (91.9 + 92.7) and scoring 7.6, about 193 GPU-h in all (registered about 188;
budget 300). CPU: gauge 345841 and extract 368223.
