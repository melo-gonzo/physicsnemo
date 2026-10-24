# family_non_box: ISLA's in-model scales against the training range, and per-case test error (post hoc)

Train cases 2390, test cases 1074. Log quantities (U, L, head RMS) in decades.

| quantity | train min | p1 | p50 | p99 | train max | test min | test max | test below train min | above train max |
|---|---|---|---|---|---|---|---|---|---|
| z | -1.88 | -1.77 | 0.0372 | 1.86 | 2.02 | -1.64 | 2.38 | 0 | 24 |
| log10_re | 3.25 | 3.4 | 5.9 | 8.41 | 8.64 | 3.58 | 9.13 | 0 | 24 |
| U | -0.697 | -0.577 | 0.578 | 1.68 | 1.77 | -0.596 | 1.69 | 0 | 0 |
| L | -1.56 | -1.31 | 0.238 | 1.78 | 1.93 | -1.19 | 2.46 | 0 | 54 |
| head_rms.velocity | -0.105 | -0.0752 | -1.26e-05 | 0.493 | 0.632 | -0.541 | 0.326 | 121 | 0 |
| head_rms.pressure | -1.75 | -0.938 | -0.000251 | 1.84 | 2.22 | -1.66 | 1.94 | 0 | 0 |

## isla: per-case error (seed mean) against the distance outside the training range (1074 cases)

| quantity | side | cases outside | median L2 v / p outside | median L2 v / p inside | Spearman v / p (outside cases: distance vs L2) | slope of log10 L2 per unit distance v / p |
|---|---|---|---|---|---|---|
| z | below | 0 | – | – | – | – |
| z | above | 24 | 0.417 / 1.47 | 0.48 / 1.51 | -0.15 / -0.11 | -0.237 / -0.876 |
| log10_re | below | 0 | – | – | – | – |
| log10_re | above | 24 | 0.417 / 1.47 | 0.48 / 1.51 | -0.15 / -0.11 | -0.171 / -0.634 |
| U | below | 0 | – | – | – | – |
| U | above | 0 | – | – | – | – |
| L | below | 0 | – | – | – | – |
| L | above | 54 | 0.596 / 1.32 | 0.471 / 1.51 | +0.31 / -0.23 | +0.984 / -0.482 |
| head_rms.velocity | below | 121 | 1.09 / 1.26 | 0.395 / 1.7 | +0.86 / +0.57 | +1.464 / +0.615 |
| head_rms.velocity | above | 0 | – | – | – | – |
| head_rms.pressure | below | 0 | – | – | – | – |
| head_rms.pressure | above | 0 | – | – | – | – |

- z: no test case below the training range

- head_rms.velocity, all 1074 cases, distance below the range (0 inside): Spearman with L2 v / p +0.20 / -0.12

By family (cases; median L2 v / p; cases below the head-velocity-RMS range):
- external_opposed_hemisphere: 391; 0.668 / 2.59; 0
- external_opposed_sphere: 380; 0.186 / 1.65; 0
- internal_duct: 303; 0.652 / 0.993; 121

## dtt: per-case error (seed mean) against the distance outside the training range (1074 cases)

| quantity | side | cases outside | median L2 v / p outside | median L2 v / p inside | Spearman v / p (outside cases: distance vs L2) | slope of log10 L2 per unit distance v / p |
|---|---|---|---|---|---|---|
| z | below | 0 | – | – | – | – |
| z | above | 24 | 0.42 / 3.18 | 0.437 / 2.36 | -0.06 / -0.18 | -0.253 / -1.641 |
| log10_re | below | 0 | – | – | – | – |
| log10_re | above | 24 | 0.42 / 3.18 | 0.437 / 2.36 | -0.06 / -0.18 | -0.183 / -1.186 |
| U | below | 0 | – | – | – | – |
| U | above | 0 | – | – | – | – |
| L | below | 0 | – | – | – | – |
| L | above | 54 | 0.534 / 1.62 | 0.431 / 2.45 | +0.28 / -0.33 | +0.773 / -0.876 |
| head_rms.velocity | below | 121 | 0.894 / 1.28 | 0.369 / 2.65 | +0.73 / +0.54 | +1.615 / +0.705 |
| head_rms.velocity | above | 0 | – | – | – | – |
| head_rms.pressure | below | 0 | – | – | – | – |
| head_rms.pressure | above | 0 | – | – | – | – |

- z: no test case below the training range

- head_rms.velocity, all 1074 cases, distance below the range (0 inside): Spearman with L2 v / p +0.19 / -0.16

By family (cases; median L2 v / p; cases below the head-velocity-RMS range):
- external_opposed_hemisphere: 391; 0.572 / 2.32; 0
- external_opposed_sphere: 380; 0.211 / 4.23; 0
- internal_duct: 303; 0.618 / 1.02; 121

