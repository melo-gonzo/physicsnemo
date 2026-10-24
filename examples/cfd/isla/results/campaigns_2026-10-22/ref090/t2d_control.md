# PREREG §13: Tensile2d training-path control (eager-trained against fix- and bug-trained; all eager-scored, native)

## 0.9.0

| field | eager (seeds; mean; σ log) | fix (mean; σ log) | bug (mean; σ log) | eager ÷ fix (|d| / 2SE; ranges) | eager ÷ bug (|d| / 2SE; ranges) | reading |
|---|---|---|---|---|---|---|
| displacement | 0.00783 / 0.00629 / 0.00891 / 0.01092 (0.00849; 0.232) | 0.00788; 0.158 | 0.00717; 0.025 | 1.077 (0.26; overlap) | 1.183 (0.72; overlap) | (c) |
| stress | 0.00964 / 0.00794 / 0.00984 / 0.01709 (0.01113; 0.330) | 0.00985; 0.144 | 0.00915; 0.040 | 1.130 (0.34; overlap) | 1.215 (0.59; overlap) | (c) |
| plastic strain | 0.41040 / 0.25005 / 0.21776 / 0.21549 (0.27342; 0.303) | 0.20068; 0.149 | 0.25797; 0.237 | 1.362 (0.92; overlap) | 1.060 (0.15; overlap) | (c) |

## A2

| field | eager (seeds; mean; σ log) | fix (mean; σ log) | bug (mean; σ log) | eager ÷ fix (|d| / 2SE; ranges) | eager ÷ bug (|d| / 2SE; ranges) | reading |
|---|---|---|---|---|---|---|
| displacement | 0.00760 / 0.00655 (0.00707; 0.106) | 0.00829; 0.011 | 0.00682; 0.006 | 0.853 (1.06; below) | 1.037 (0.24; overlap) | (b) |
| stress | 0.00849 / 0.00780 (0.00815; 0.059) | 0.00931; 0.019 | 0.00780; 0.129 | 0.874 (1.53; below) | 1.044 (0.21; overlap) | (b) |
| plastic strain | 0.18696 / 0.43064 (0.30880; 0.590) | 0.21350; 0.178 | 0.40657; 0.136 | 1.446 (0.42; overlap) | 0.760 (0.32; overlap) | (c) |

Displacement readings: 0.9.0 (c), A2 (b) -> overall (b) [families disagree; A2 decides, as registered]

eager posed ÷ native (displacement): s42 1.0001, s43 0.9997, s44 1.0023, s45 0.9998
