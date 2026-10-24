# LRE-AZ tables (reduce_lreaz.py)

## Split facts (reynolds)

- Label log10 Re: train 3.003 to 5.999 (2119 cases), test 6.001 to 8.000 (1385 cases); test cases above the train maximum by 0.002 to 2.000 decades (median 0.836); below: 0.
- ISLA's log10 Re (its own length L): train 3.247 to 7.040 (2119 of 2119; mean 5.0619, population std 0.8985), test 6.233 to 9.133 (1385 of 1385); test cases above ISLA's train maximum 7.040: 971, inside: 414.
- z fitted, (log10 Re - 5.062) / 0.8985: train -2.020 to 2.202, test 1.304 to 4.531.
- z fixed, log10(Re / 1e6): train -2.753 to 1.040, test 0.233 to 3.133.

## Per-seed case-mean [case-median] relative L2, reynolds test

| arm | seed | pressure_l2 | velocity_l2 | nut_l2 | tke_l2 | omega_l2 |
|---|---|---|---|---|---|---|
| A | 42 | 0.5062 [0.5234] | 0.1608 [0.1520] | 0.8189 [0.4974] | 0.6812 [0.6876] | 0.5848 [0.5688] |
| A | 43 | 0.5122 [0.5369] | 0.1676 [0.1519] | 0.8871 [0.4928] | 0.6889 [0.6932] | 0.5705 [0.5657] |
| B | 42 | 0.5162 [0.5589] | 0.1831 [0.1608] | 0.6772 [0.5065] | 0.6900 [0.7241] | 0.5845 [0.5766] |
| B | 43 | 0.5704 [0.5867] | 0.1866 [0.1640] | 0.7437 [0.5120] | 0.6859 [0.7243] | 0.5717 [0.5498] |

A's two-seed std of ln(case-mean): pressure_l2 0.0083, velocity_l2 0.0293, nut_l2 0.0565, tke_l2 0.0080, omega_l2 0.0174

A's two-seed std of ln(case-median): pressure_l2 0.0180, velocity_l2 0.0007, nut_l2 0.0066, tke_l2 0.0057, omega_l2 0.0038

## Registered rule (case means; pressure and velocity ruled, the others beside)

| field | B ÷ A = exp(D) | D | SE | 2 SE | s_A | s_B | per-seed B ÷ A (42, 43) | verdict | medians: exp(D), 2 SE, verdict (beside) |
|---|---|---|---|---|---|---|---|---|---|
| pressure_l2 | 1.0670 | +0.0648 | 0.0502 | 0.1004 | 0.0083 | 0.0705 | 1.0198, 1.1136 | **no change** | 1.0804, 0.0548, no change |
| velocity_l2 | 1.1257 | +0.1184 | 0.0228 | 0.0455 | 0.0293 | 0.0134 | 1.1385, 1.1133 | **harms** | 1.0685, 0.0196, no change |
| nut_l2 | 0.8329 | -0.1829 | 0.0616 | 0.1231 | 0.0565 | 0.0662 | 0.8270, 0.8384 | **(helps, beside)** | 1.0285, 0.0143, no change |
| tke_l2 | 1.0042 | +0.0042 | 0.0064 | 0.0127 | 0.0080 | 0.0042 | 1.0129, 0.9956 | **(no change, beside)** | 1.0489, 0.0081, no change |
| omega_l2 | 1.0007 | +0.0007 | 0.0166 | 0.0331 | 0.0174 | 0.0156 | 0.9995, 1.0020 | **(no change, beside)** | 0.9929, 0.0479, no change |

**Consequence (PREREG section 4):** harms on velocity_l2: the fixed scale is not adopted without Peter.

## Extrapolation profile (descriptive): seed-mean case-mean error per distance bin, decades above the train band (label)

| bin | cases | A pressure_l2 | B pressure_l2 | A velocity_l2 | B velocity_l2 | A nut_l2 | B nut_l2 | B ÷ A pressure_l2 | B ÷ A velocity_l2 | B ÷ A nut_l2 | B ÷ A tke_l2 | B ÷ A omega_l2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [0.0, 0.5) | 411 | 0.3765 | 0.3808 | 0.1435 | 0.1446 | 0.4118 | 0.4144 | 1.0114 | 1.0072 | 1.0064 | 1.0014 | 1.0005 |
| [0.5, 1.0) | 389 | 0.4587 | 0.4758 | 0.1550 | 0.1625 | 0.4970 | 0.4723 | 1.0373 | 1.0482 | 0.9502 | 1.0261 | 1.0067 |
| [1.0, 1.5) | 312 | 0.5661 | 0.6294 | 0.1723 | 0.2042 | 0.8484 | 0.7097 | 1.1118 | 1.1856 | 0.8365 | 1.0296 | 0.9935 |
| [1.5, 2.0] | 273 | 0.7159 | 0.7858 | 0.1994 | 0.2554 | 2.0298 | 1.4964 | 1.0977 | 1.2806 | 0.7372 | 0.9573 | 1.0030 |

Slopes per decade above the band, least squares over the 1,385 test cases of the seed-mean per-case error (AF-EXTRAP's definition, posthoc.py slope); of its log10 (AZ-DECIDE's); and of ln(B ÷ A) per case:

- pressure_l2: A +0.2284 (log10 +0.2234); B +0.2739 (log10 +0.2768); ln(B ÷ A) +0.1231 per decade
- velocity_l2: A +0.0366 (log10 +0.0897); B +0.0724 (log10 +0.1417); ln(B ÷ A) +0.1196 per decade
- nut_l2: A +1.0211 (log10 +0.2801); B +0.6919 (log10 +0.2467); ln(B ÷ A) -0.0770 per decade
- tke_l2: A +0.1320 (log10 +0.0937); B +0.1146 (log10 +0.0940); ln(B ÷ A) +0.0007 per decade
- omega_l2: A +0.3596 (log10 +0.3065); B +0.3608 (log10 +0.3053); ln(B ÷ A) -0.0026 per decade

AF-EXTRAP for comparison (AirfRANS reynolds, below the band, LRE.md): slope per decade of pressure 0.62 (fitted) and 0.16 (fixed).

## Beside: by ISLA's own log10 Re (the input the model reads), inside vs above ISLA's train maximum

| part | cases | B ÷ A pressure_l2 | B ÷ A velocity_l2 | B ÷ A nut_l2 | B ÷ A tke_l2 | B ÷ A omega_l2 |
|---|---|---|---|---|---|---|
| inside | 414 | 1.0153 | 1.0068 | 1.0125 | 0.9998 | 0.9998 |
| above | 971 | 1.0807 | 1.1693 | 0.8007 | 1.0059 | 1.0009 |

## Event guard (validation loss > 2 × its running minimum)

- A s42: 26 val epochs (last training epoch 249), largest ratio 1.026 (epoch 249), hits none
- A s43: 26 val epochs (last training epoch 249), largest ratio 1.092 (epoch 20), hits none
- B s42: 26 val epochs (last training epoch 249), largest ratio 1.034 (epoch 90), hits none
- B s43: 26 val epochs (last training epoch 249), largest ratio 1.110 (epoch 20), hits none

## Checks

- all checks pass (epoch 250, test side, 1,385 cases, code_r090, dataset name per arm, case ids, labels against case_labels.json)
