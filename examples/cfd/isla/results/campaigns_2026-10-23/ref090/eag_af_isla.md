# af_isla: eager-trained vs fix-trained vs bug-trained (all eager-scored, native)

| field | eager (s42, s43) | fix (s42, s43) | bug (s42, s43) | eager ÷ fix (|d|/2SE; ranges) | eager ÷ bug (|d|/2SE; ranges) | fix ÷ bug | reading |
|---|---|---|---|---|---|---|---|
| pressure_l2 | 0.0124785 (0.0131631, 0.011794) | 0.0122237 (0.0129875, 0.0114599) | 0.0119399 (0.012644, 0.0112357) | 1.021 (0.12; overlap) | 1.045 (0.27; overlap) | 1.024 (0.14) | (c) |
| velocity_l2 | 0.00635539 (0.0064407, 0.00627008) | 0.00632686 (0.00642882, 0.0062249) | 0.00642287 (0.00652497, 0.00632077) | 1.005 (0.11; overlap) | 0.989 (0.25; overlap) | 0.985 (0.33) | (c) |
| nut_l2 | 0.0153529 (0.0157382, 0.0149675) | 0.0155647 (0.0157396, 0.0153897) | 0.0155887 (0.0160113, 0.0151661) | 0.986 (0.25; overlap) | 0.985 (0.21; overlap) | 0.998 (0.03) | (c) |

Reading: (c)

posed ÷ native (s42, s43): pressure_l2: eager 1.0022 / 1.0025, fix 1.0025 / 1.0029, bug 1.0025 / 1.0027; velocity_l2: eager 1.0016 / 1.0016, fix 1.0015 / 1.0016, bug 1.0015 / 1.0015; nut_l2: eager 1.0000 / 1.0000, fix 1.0000 / 1.0001, bug 1.0000 / 1.0000
posed means: pressure_l2: eager 0.0125082, fix 0.012256, bug 0.0119708; velocity_l2: eager 0.00636557, fix 0.00633686, bug 0.00643263; nut_l2: eager 0.0153524, fix 0.0155651, bug 0.0155885
