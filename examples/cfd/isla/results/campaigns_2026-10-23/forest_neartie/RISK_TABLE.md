# FOREST-NEARTIE risk table (task 1)

Draws joined (face and octant on the same tokens, kept-cell sha equal): 1776; sha mismatches: 0.

| dataset | cases | draws | face/octant median | p99 | max | > 2x | > 3x | > 5x | face pairs median / max | octant pairs median / max | max pairs of one point, face / octant |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aerozoo_uvolume | 320 | 1600 | 0.954 | 1.303 | 2.231 | 1 | 0 | 0 | 41.3M / 179.4M | 44.4M / 101.1M | 1391 / 1063 |
| airfrans_uvolume_2d | 24 | 48 | 1.084 | 1.157 | 1.157 | 0 | 0 | 0 | 4.4M / 4.6M | 4.0M / 4.3M | 195 / 78 |
| drivaer_ml_volume_wholeboundary | 8 | 32 | 0.837 | 0.862 | 0.862 | 0 | 0 | 0 | 141.0M / 144.7M | 168.5M / 174.8M | 1226 / 1490 |
| ginot_lug_volume | 24 | 48 | 0.971 | 1.002 | 1.002 | 0 | 0 | 0 | 1.0M / 1.5M | 1.1M / 1.5M | 182 / 193 |
| plaid_tensile2d_2d | 24 | 48 | 1.072 | 1.287 | 1.287 | 0 | 0 | 0 | 0.2M / 0.3M | 0.2M / 0.2M | 41 / 40 |

AeroZoo by split and family (face / octant pairs on the same draws):

| split | family | cases | draws | ratio median | p99 | max | > 2x | face eta max |
|---|---|---|---|---|---|---|---|---|
| full.test | external_box | 15 | 75 | 0.960 | 1.151 | 1.171 | 0 | 3.90e-05 |
| full.test | external_opposed_hemisphere | 15 | 75 | 1.005 | 1.844 | 2.231 | 1 | 1.91e-02 |
| full.test | external_opposed_sphere | 15 | 75 | 1.070 | 1.815 | 1.940 | 0 | 1.03e-02 |
| full.test | internal_duct | 15 | 75 | 0.941 | 1.101 | 1.101 | 0 | 3.89e-04 |
| full.train | external_box | 145 | 725 | 0.934 | 1.127 | 1.163 | 0 | 2.22e-04 |
| full.train | external_opposed_hemisphere | 15 | 75 | 0.935 | 1.111 | 1.211 | 0 | 5.65e-03 |
| full.train | external_opposed_sphere | 25 | 125 | 1.065 | 1.302 | 1.417 | 0 | 6.42e-03 |
| full.train | internal_duct | 15 | 75 | 0.936 | 1.169 | 1.169 | 0 | 2.23e-04 |
| full.val | external_box | 15 | 75 | 0.917 | 1.032 | 1.034 | 0 | 4.72e-05 |
| full.val | external_opposed_hemisphere | 15 | 75 | 0.966 | 1.279 | 1.379 | 0 | 7.69e-03 |
| full.val | external_opposed_sphere | 15 | 75 | 1.049 | 1.248 | 1.306 | 0 | 3.11e-03 |
| full.val | internal_duct | 15 | 75 | 0.930 | 1.164 | 1.169 | 0 | 5.45e-04 |

## Memory (full training steps, bf16, eager blocks, one GB300)

- aerozoo_uvolume: 640 full steps (single chunk 640); peak reserved = 147.2 GiB + 152 B/pair (allocated: 146.4 GiB + 154 B/pair); pairs 27.8M-179.4M, peak reserved max 172.3 GiB; reserved minus the chunker's prediction: median +3.61, max +4.02 GiB; step 2.8-46.3 s; multi-chunk steps 0
- airfrans_uvolume_2d: 48 full steps (single chunk 48); peak reserved = 31.4 GiB + 18 B/pair (allocated: 30.9 GiB + 72 B/pair); pairs 3.8M-4.6M, peak reserved max 31.7 GiB; reserved minus the chunker's prediction: median -1.72, max -1.54 GiB; step 0.7-35.4 s; multi-chunk steps 0
- ginot_lug_volume: 48 full steps (single chunk 48); peak reserved = 1.9 GiB + 4422 B/pair (allocated: 1.8 GiB + 4360 B/pair); pairs 0.7M-1.5M, peak reserved max 8.1 GiB; reserved minus the chunker's prediction: median -6.91, max -6.67 GiB; step 0.4-30.9 s; multi-chunk steps 0
- plaid_tensile2d_2d: 48 full steps (single chunk 48); peak reserved = 0.4 GiB + 16199 B/pair (allocated: 0.3 GiB + 16248 B/pair); pairs 0.2M-0.3M, peak reserved max 4.3 GiB; reserved minus the chunker's prediction: median -6.84, max -6.75 GiB; step 0.4-32.6 s; multi-chunk steps 0

## Worst draws by face pairs

| dataset | split | idx | case | family | draw | face pairs | octant pairs | ratio | face max/pt | eta (tokens) | full-res gap/E | face mode | face peak reserved GiB | octant peak reserved GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aerozoo_uvolume | full.test | 2977 | case-0000000000000003385-342a38f8c05a | external_opposed_hemisphere | 0 | 179.4M | 80.4M | 2.23 | 1391 | 1.91e-02 | 43.9 | step | 172.3 | 158.7 |
| aerozoo_uvolume | full.test | 2941 | case-0000000000000003343-3714c0d43d42 | external_opposed_hemisphere | 0 | 145.9M | 85.4M | 1.71 | 1100 | 1.45e-02 | 30.1 | step | 167.9 | 159.5 |
| drivaer_ml_volume_wholeboundary | val | 12 |  |  | 0 | 144.7M | 169.0M | 0.86 | 1191 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 12 |  |  | 3 | 144.4M | 169.4M | 0.85 | 1201 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 12 |  |  | 2 | 144.3M | 168.4M | 0.86 | 1189 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 12 |  |  | 1 | 143.7M | 169.9M | 0.85 | 1175 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 24 |  |  | 2 | 143.6M | 173.2M | 0.83 | 1166 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 24 |  |  | 1 | 143.6M | 172.8M | 0.83 | 1169 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 24 |  |  | 0 | 143.5M | 174.8M | 0.82 | 1186 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 24 |  |  | 3 | 143.1M | 171.4M | 0.83 | 1170 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 18 |  |  | 0 | 142.9M | 169.1M | 0.85 | 1199 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 18 |  |  | 1 | 142.8M | 169.7M | 0.84 | 1226 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 18 |  |  | 3 | 142.5M | 170.0M | 0.84 | 1192 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 18 |  |  | 2 | 142.4M | 170.6M | 0.83 | 1211 | 2.39e-06 | 4.19e+05 | None | nan | nan |
| drivaer_ml_volume_wholeboundary | val | 6 |  |  | 2 | 142.3M | 172.3M | 0.83 | 1190 | 2.39e-06 | 4.19e+05 | None | nan | nan |

## Near-tie Monte Carlo (job 429698): 60 cases x 60 draws, pairs only, same draws (kept-cell sha equal on all 3,600)

The 59 cases with a face-forest eta above 1e-3 on some census draw (40 opposed-hemisphere, 19 opposed-sphere; all
splits) and AF-CONFIRM's full.train case 0 (case-0000000000000000001-ede2dfcec3ed, an opposed-hemisphere case).

| forest | pairs / octant median | p99 | max | > 2x | > 3x | > 5x |
|---|---|---|---|---|---|---|
| face (8097ed00f) | 1.034 | 2.047 | 8.399 | 40 (1.1%) | 15 (0.42%) | 3 (0.08%) |

Worst draw: case-...0001 draw 22, eta 0.053, 742M face pairs (88M octant), up to 5,000+ per point.

## Pairs against a forced eta (job 430227, probe_etasweep.py, 7 AeroZoo draws, median of pairs(eta) / pairs(0))

| eta | 0.002 | 0.005 | 0.01 | 0.02 | 0.035 | 0.05 | 0.07 | 0.1 | 0.15 | 0.2 | 0.3 | 0.45+ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| face forest | 1.1 | 1.2 | 1.6 | 2.5 | 5.0 | 8.7 | 15.9 | 32.7 | 80 | 162 | 409 | ~540 (every pair) |

DrivAerML's car (tiny nodes far from the centroid) is the most sensitive: x3.1 at eta 0.002, x119 at 0.02 (its own
eta is 2.4e-6).

## The chunker (job 429698, forced capacity_gib; chunk_sim.py)

0.14.0's recipe splits ISLA-DTT's step into chunks of points by a predicted peak (train._chunk_spans, bf16: 0.136 GiB per
1M pairs). The measured peak reserved exceeds its prediction by +3.6 to +4.0 GiB at 200k points in one chunk (all 640
census steps) and by +1.2 GiB at 100k points per chunk. Forced capacity 150 GiB on a 39.8M-pair draw: predicted 149.1,
one chunk, measured 152.7 (would OOM). On a GB300 (capacity about 275.5 GiB) chunk_sim.py puts the OOM windows at
942-968M pairs (one chunk) and about 2% of each multi-chunk band (2.25-2.28B, 3.56-3.59B, ...).

## OOM chance of a 250-epoch AeroZoo ISLA-DTT lane (risk_model.py)

Expected OOM draws per lane 0.145, P(lane OOM) about 13% (Rice law of each train case's token gap g/E about its
full-resolution value with its fitted spread; pairs = eta-free pairs x m(eta); 2,800 draws per epoch). Beside: 2.4
draws per lane above 1e9 pairs (two-pass chunked steps) and 0.43 above 1e10 (eta >= ~0.3: every pair read, ~4e10).
Uncertain by about a factor 3 (the Rice tail and m(eta) extrapolated beyond the observed draws).

## Resume replay (job 429718, replay_wrap.py; data/replay_429718)

The recipe's checkpoint holds no RNG state; train.py seeds every process with set_seed(seed + rank). Two resumes (B, C)
from copies of one checkpoint drew bitwise the same starts and kept cells at every step; an uninterrupted run (D) drew
different ones in the same epoch; the resumed epoch 1 reused the fresh run's epoch-0 starts (0.4686, 0.3271, 0.6583).
So a crash inside a resumed segment recurs at the same step on every later resume from that checkpoint (a crash loop),
and every chained link restarts the same sequence of starts.
