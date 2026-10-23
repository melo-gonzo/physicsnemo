# lug: eager-trained vs fix-trained vs bug-trained (all eager-scored, native)

| field | eager (s42, s43) | fix (s42, s43) | bug (s42, s43) | eager ÷ fix (|d|/2SE; ranges) | eager ÷ bug (|d|/2SE; ranges) | fix ÷ bug | reading |
|---|---|---|---|---|---|---|---|
| von_mises_l2 | 0.00157375 (0.0013762, 0.0017713) | 0.00145367 (0.00146246, 0.00144489) | 0.0018933 (0.00178038, 0.00200622) | 1.083 (0.31; overlap) | 0.831 (0.66; below) | 0.768 (2.20) | (c) |

Reading: (c)

posed ÷ native (s42, s43): von_mises_l2: eager 1.2464 / 1.6191, fix 1.3461 / 1.3829, bug 1.1938 / 1.1054
posed means: von_mises_l2: eager 0.0022916, fix 0.00198343, bug 0.00217147
