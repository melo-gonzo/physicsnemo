# SLICES-DRV registered reading (native decides; posed beside)

| protocol | field | B (s42, s43) | S (s42, s43) | S ÷ B | D | s_B, s_S | 2 SE | seed ratios | label | sub-bar flag |
|---|---|---|---|---|---|---|---|---|---|---|
| native | pressure_l2 | 0.0383775 (0.0379543, 0.0388006) | 0.0384658 (0.0384379, 0.0384937) | 1.002 | +0.0023 | 0.0156, 0.0010 | 0.0221 | 1.013, 0.992 | **no change** | none |
| native | velocity_l2 | 0.0577049 (0.0573261, 0.0580836) | 0.0574079 (0.0574013, 0.0574145) | 0.995 | -0.0052 | 0.0093, 0.0002 | 0.0131 | 1.001, 0.988 | **no change** | none |
| native | nut_l2 | 0.0846549 (0.0840118, 0.0852979) | 0.0845489 (0.0838874, 0.0852104) | 0.999 | -0.0013 | 0.0107, 0.0111 | 0.0218 | 0.999, 0.999 | **no change** | none |
| posed | pressure_l2 | 0.0384963 (0.0380574, 0.0389352) | 0.0385521 (0.0385355, 0.0385687) | 1.001 | +0.0014 | 0.0161, 0.0006 | 0.0228 | 1.013, 0.991 | **no change** | none |
| posed | velocity_l2 | 0.0579145 (0.0575557, 0.0582734) | 0.0576 (0.0575937, 0.0576062) | 0.995 | -0.0054 | 0.0088, 0.0002 | 0.0124 | 1.001, 0.989 | **no change** | none |
| posed | nut_l2 | 0.0846503 (0.0840196, 0.085281) | 0.0844531 (0.0837867, 0.0851194) | 0.998 | -0.0023 | 0.0105, 0.0112 | 0.0217 | 0.997, 0.998 | **no change** | none |

Decision (PREREG section 6 with the 00:20 amendment): **candidate on two datasets**

Beside (native, seed-mean per case): pressure_l2: median S ÷ B 0.996, S better on 52 % of cases; velocity_l2: median S ÷ B 0.989, S better on 58 % of cases; nut_l2: median S ÷ B 1.001, S better on 50 % of cases

Guard 3 (S compiled vs eager, max per-case relative difference over fields): {'42': 5.885843682107428e-06, '43': 4.52950036944344e-06} -> pass True
area == native exactly: {'B': False, 'S': False} (degenerate, no claim)
posed ÷ native: B pressure_l2 1.0027 / 1.0035, velocity_l2 1.0040 / 1.0033, nut_l2 1.0001 / 0.9998; S pressure_l2 1.0025 / 1.0019, velocity_l2 1.0034 / 1.0033, nut_l2 0.9988 / 0.9989
