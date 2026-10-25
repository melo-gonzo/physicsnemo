# AZ-RESCORE results: the AeroZoo standing under AeroZoo's own interior metrics

Written 2026-10-25 06:40 EDT. Lane AZ-RESCORE (supervisor 01a13c2c), after AZ-SPEC-AUDIT's M1 (`scratch/isla-az-spec-audit-2026-10-24/AUDIT.md`,
md5 448bd6f1). Spec: aerozoo `docs/book/06-splits.qmd` @ 96bd2b62 (md5 52118947), lines 282-298 and 429-472. Inference only:
no training code, checkpoint or dataset was changed. Every registered verdict stands as written; everything here is beside it.

## 1. What ran

- **Scorer** `lhr/specscore.py` (on LHR `$T/kernel/azrescore/`; md5 1edc8f04 for jobs 376802-376818, 4577daff for 378817 and
  378820, which differ only in when the verbatim near-field self-check runs). Per case it reads the scoring-only data from the raw
  case file (`_target_quadrature_measure`, `global_data/reference`, `annotation/body_id`, the `release-v1.2.json` stability label),
  builds each lane exactly as the recipe's `infer.py` does on the lane's own snapshot (code_r090 + recipe_r090 for ISLA and ISLA-DTT,
  code_azv2 + recipe_az_c for GeoTransolver and Transolver, code_hn090 + recipe_r090 for E1-AZ, code_r090 + recipe_lreaz for LRE-AZ;
  the registered scorers' Hydra overrides, float32, the model configs' own compile settings), runs the whole interior through the model
  in chunks, and evaluates AeroZoo's `rms`, `on_body` and `near_field` verbatim (`lhr/reference_metrics.py.txt`, md5 b08f1f4a, identical
  to the 06-splits listing) in float64. Beside them: the full-mesh unweighted relative L2 (float64) of all five fields, the turbulence
  fields in the spec's form, and the zero and free-stream predictors.
- **Chunks.** Exactly 200,000 points per forward (every model's training count), a seeded random partition of the cells (pseed 0,
  CRC32 of the case id), the last part filled up to 200,000 with cells of the first part whose predictions there are discarded.
  ISLA-DTT, exact under chunking, at 1,000,000 points per forward. The baselines' 50,000-face draw is seeded per case.
- **Jobs (LHR GB300):** smokes 376168, 376237, 376779; scoring 376802 (baselines, 2 nodes), 376803 (E1-AZ), 376804 (LRE-AZ, 2 nodes),
  376818 (ISLA + ISLA-DTT, 10 nodes), 378817 (baselines' remainder), 378820 (ISLA + ISLA-DTT remainder, 3 nodes). Per-case rows:
  `$T/azrescore_out/score/<run>/test/*.jsonl`, local copy `data/score/`. Commands: `lhr/score.sbatch` (FAMILY=base|isla|hn|lre,
  plans in `lhr/plans/`), details and timings in JOBS.md.
- **Coverage:** all 48 lanes, every test case of their splits (5,969 case-splits per seed for each 7-split architecture; 0 missing,
  0 duplicated, every row epoch 250). The validation re-read (iii) was dropped by the supervisor (23:02 EDT 10-24); no PREREG_ADDENDUM.
- **Cost:** 258.2 GPU-h by sacct (cap 265 after two raises: 150 → 220 at 23:02, → 265 at 02:58): smokes 9.0, baselines 32.1, E1-AZ 10.6,
  LRE-AZ 23.7, ISLA + ISLA-DTT 182.8. Model forward time was 92 % of the node time.

## 2. Scorer validation

| check | result |
|---|---|
| Model path: this scorer's loading and forward on the registered scorer's 200k sample vs the registered per-case relL2 | 0.0 difference for GeoTransolver, Transolver, ISLA and E1-AZ (3 + 2 cases per lane); LRE-AZ 0.0 and 5.8e-6; ISLA-DTT ≤ 2.1e-7 (its float32 run-to-run noise) |
| Independent float64 implementation from the spec's text (`lhr/indep_check.py`, md5 46c286b2: raw memmaps, vertex-mean face centroids, brute-force GPU nearest distance) on 6 `full` test cases × GeoTransolver, Transolver, ISLA, ISLA-DTT (seed 42) + zero + free stream | worst relative difference 2.4e-8 (one case's near field: a few cells at the L_ref threshold, centroids differing in the last float32 bit); ≤ 1e-13 elsewhere (`data/smoke/INDEP_CHECK.txt`) |
| Zero predictor | matches the independent implementation exactly (0.0 to 2e-16) on the 6 cases |
| Free stream vs AeroZoo's own `paper/data/freestream-v1.2.csv` (ε_U, ε_p, ε^N_U, ε^N_p, near share) | all 5,969 test case-splits, worst 4.8e-9 (the CSV's printed precision) |
| Threaded `near_field` (cKDTree `workers`) vs the verbatim function | equal on all 75 cases where both ran |
| Interior row order and frame: the chain's interior points vs the raw file's | asserted equal (`torch.equal`) on every scored case |
| Full-mesh relL2 vs registered 200k relL2 | case means agree to ≤ 0.5 % per lane (worst GeoTransolver reynolds pressure, also a different face draw): the 200k sample was not the issue; the metric was |

## 3. Smokes and prices

Measured in the smokes, per 200,000-point chunk on one GB300 (float32): GeoTransolver 0.22 s, Transolver 0.21 s, interacting ISLA
0.86 s, ISLA-DTT 1.24 s at 200k chunks and 0.75 s per 200k at 1M chunks (2M chunks: no further gain, 162 GiB peak). Priced
before spending (cap 150): (i) baselines 28 GPU-h, (ii) ISLA + ISLA-DTT 163, (iii) validation 85, (iv) E1-AZ + LRE-AZ 15 (sent to the
supervisor; cap raised to 220, (iii) dropped, order (i), (iv), (ii)). Actual, from the production rows: GeoTransolver 0.222 s, Transolver
0.216 s, ISLA 0.884 s, ISLA-DTT 0.80 s per 200k cells; actual GPU-h (i) 32.1 against 28, (iv) 34.3 against 15 (I had counted one lane
each; it is two), (ii) 182.8 against 163 (start-up, compiles and case overheads).

## 4. Partition, chunk and draw dependence (`PARTITION.md`)

- **Interacting models, repartitioned with pseed 1** (2 `full` test cases): interacting ISLA moves ε and relL2 by at most 4.5e-4 relative
  (ε^N_U, the 2.2 M-cell case; 1e-5 to 1e-4 on the 15 M-cell case); GeoTransolver ≤ 4.3e-6; Transolver ≤ 3e-8.
- **ISLA-DTT, exact under chunking:** 200k → 1M chunks with another partition moves it by ≤ 9.8e-5, and 200k → 2M by ≤ 1.9e-4, which is
  its float32 roundoff under a different batch shape, not a dependence. So interacting ISLA's partition dependence is at most about twice
  ISLA-DTT's float32 noise on these cases.
- **Run to run** (same partition, another process): ISLA and ISLA-DTT ≤ 2.6e-6. **The baselines' 50,000-face draw** (per-case seed vs the
  registered sequence): GeoTransolver ≤ 1.8e-3, Transolver ≤ 2.2e-6.
- All of these are 2 to 4 orders of magnitude below every gap read below.

## 5. The standing under AeroZoo's metric

Per split and model: the mean over cases [the median over cases], each the mean of the two seeds' statistics (AeroZoo's paper
convention); the book's median is the median of per-case seed means, which `AEROZOO_SPECMETRIC.json` also holds
(`median_of_case_seed_mean`). The relL2 columns give the registered 200k float32 number and [the full-mesh float64 one].
Full tables with every seed, mean_stable and the orders: `TABLES_FULL.md`; machine-readable, with the turbulence fields (ε_nut, ε_k, ε_ω and their relL2): `AEROZOO_SPECMETRIC.json`
(md5 c8fb1c82).

**All test cases: mean [median] over cases, seed means.**

| split (n) | model | ε_U | ε_p | ε^N_U | ε^N_p | relL2 u, registered 200k [full mesh] | relL2 p, registered 200k [full mesh] |
|---|---|---|---|---|---|---|---|
| full (716) | ISLA (interacting) | 0.0791 [0.0586] | 0.169 [0.0279] | 0.114 [0.0969] | 0.185 [0.0491] | 0.118 [0.118] | 0.304 [0.304] |
| full (716) | ISLA-DTT | 0.0773 [0.057] | 0.162 [0.0256] | 0.109 [0.0933] | 0.171 [0.0432] | 0.11 [0.11] | 0.256 [0.257] |
| full (716) | GeoTransolver | 0.695 [0.568] | 1.69 [0.147] | 0.677 [0.553] | 1.89 [0.161] | 0.609 [0.609] | 1.25 [1.25] |
| full (716) | Transolver | 1.15 [1.01] | 1.69 [0.149] | 1.14 [1.01] | 1.88 [0.153] | 0.998 [0.998] | 1.17 [1.17] |
| full (716) | zero predictor | 1.16 [1] | 1.68 [0.11] | 1.15 [1.01] | 1.88 [0.136] | – [1] | – [1] |
| full (716) | free stream | 0.47 [0.184] | 1.68 [0.11] | 0.459 [0.183] | 1.88 [0.136] | – [0.469] | – [1] |
| bank_zero_to_cad (333) | ISLA (interacting) | 0.115 [0.0869] | 0.224 [0.0437] | 0.178 [0.149] | 0.265 [0.0827] | 0.175 [0.175] | 0.377 [0.378] |
| bank_zero_to_cad (333) | ISLA-DTT | 0.114 [0.0855] | 0.216 [0.0445] | 0.173 [0.146] | 0.254 [0.0841] | 0.167 [0.167] | 0.361 [0.361] |
| bank_zero_to_cad (333) | GeoTransolver | 0.525 [0.296] | 1.81 [0.176] | 0.541 [0.347] | 1.93 [0.241] | 0.506 [0.506] | 1.12 [1.12] |
| bank_zero_to_cad (333) | zero predictor | 1.16 [1.01] | 1.78 [0.145] | 1.15 [1.01] | 1.92 [0.203] | – [1] | – [1] |
| bank_zero_to_cad (333) | free stream | 0.486 [0.206] | 1.78 [0.145] | 0.512 [0.296] | 1.92 [0.203] | – [0.504] | – [1] |
| blobs (1367) | ISLA (interacting) | 0.0963 [0.0728] | 0.195 [0.0343] | 0.139 [0.117] | 0.207 [0.0578] | 0.145 [0.145] | 0.388 [0.388] |
| blobs (1367) | ISLA-DTT | 0.0878 [0.0665] | 0.183 [0.0305] | 0.125 [0.106] | 0.183 [0.0511] | 0.13 [0.13] | 0.307 [0.307] |
| blobs (1367) | GeoTransolver | 0.836 [0.672] | 1.8 [0.166] | 0.822 [0.663] | 2.01 [0.164] | 0.718 [0.718] | 1.11 [1.11] |
| blobs (1367) | zero predictor | 1.17 [1.01] | 1.76 [0.109] | 1.16 [1.01] | 1.98 [0.136] | – [1] | – [1] |
| blobs (1367) | free stream | 0.476 [0.17] | 1.76 [0.109] | 0.464 [0.184] | 1.98 [0.136] | – [0.468] | – [1] |
| nonbox (1074) | ISLA (interacting) | 0.418 [0.334] | 0.72 [0.176] | 0.363 [0.312] | 0.585 [0.172] | 0.533 [0.533] | 2.77 [2.77] |
| nonbox (1074) | ISLA-DTT | 0.404 [0.324] | 0.706 [0.293] | 0.331 [0.273] | 0.561 [0.287] | 0.491 [0.491] | 3.53 [3.53] |
| nonbox (1074) | GeoTransolver | 0.396 [0.264] | 0.479 [0.114] | 0.312 [0.217] | 0.357 [0.117] | 0.591 [0.59] | 1.17 [1.17] |
| nonbox (1074) | E1-AZ HEADNORM | 0.45 [0.399] | 0.766 [0.228] | 0.387 [0.382] | 0.589 [0.216] | 0.558 [0.558] | 3.57 [3.57] |
| nonbox (1074) | zero predictor | 1.01 [1] | 0.467 [0.0924] | 1.03 [1.01] | 0.347 [0.109] | – [1] | – [1] |
| nonbox (1074) | free stream | 0.27 [0.179] | 0.467 [0.0924] | 0.163 [0.15] | 0.347 [0.109] | – [0.461] | – [1] |
| profile_crossflow (578) | ISLA (interacting) | 1.29 [1.31] | 5.01 [5.1] | 1.55 [1.56] | 5.15 [5.22] | 1.42 [1.42] | 2.66 [2.66] |
| profile_crossflow (578) | ISLA-DTT | 1.26 [1.35] | 4.74 [4.79] | 1.52 [1.59] | 5.05 [5.09] | 1.34 [1.34] | 2.19 [2.19] |
| profile_crossflow (578) | GeoTransolver | 0.827 [0.777] | 2.59 [1.97] | 0.754 [0.725] | 2.92 [2.2] | 0.831 [0.831] | 1 [1] |
| profile_crossflow (578) | zero predictor | 1.04 [0.99] | 2.59 [1.97] | 0.929 [0.897] | 2.92 [2.19] | – [1] | – [1] |
| profile_crossflow (578) | free stream | 0.855 [0.804] | 2.59 [1.97] | 0.799 [0.779] | 2.92 [2.19] | – [0.867] | – [1] |
| profile_dual_inlet (516) | ISLA (interacting) | 1.39 [1.37] | 7.04 [5.88] | 1.49 [1.45] | 7.44 [6.12] | 0.652 [0.652] | 0.799 [0.799] |
| profile_dual_inlet (516) | ISLA-DTT | 1.28 [1.29] | 7.22 [6.17] | 1.36 [1.35] | 7.68 [6.48] | 0.606 [0.606] | 0.826 [0.826] |
| profile_dual_inlet (516) | GeoTransolver | 1.67 [1.6] | 9.34 [7.37] | 1.62 [1.59] | 10.1 [7.8] | 0.788 [0.788] | 1 [1] |
| profile_dual_inlet (516) | zero predictor | 2.14 [2.06] | 9.34 [7.34] | 2.12 [2.1] | 10.1 [7.78] | – [1] | – [1] |
| profile_dual_inlet (516) | free stream | 1.68 [1.59] | 9.34 [7.34] | 1.63 [1.59] | 10.1 [7.78] | – [0.793] | – [1] |
| reynolds (1385) | ISLA (interacting) | 0.115 [0.0709] | 0.371 [0.0375] | 0.145 [0.107] | 0.366 [0.0629] | 0.164 [0.164] | 0.509 [0.51] |
| reynolds (1385) | ISLA-DTT | 0.114 [0.0699] | 0.401 [0.0463] | 0.142 [0.104] | 0.41 [0.0632] | 0.152 [0.152] | 0.473 [0.473] |
| reynolds (1385) | GeoTransolver | 0.82 [0.624] | 1.75 [0.0753] | 0.805 [0.628] | 1.9 [0.115] | 0.71 [0.709] | 1.04 [1.04] |
| reynolds (1385) | LRE-AZ B | 0.14 [0.0768] | 0.531 [0.0419] | 0.167 [0.11] | 0.544 [0.0684] | 0.185 [0.185] | 0.543 [0.544] |
| reynolds (1385) | zero predictor | 1.17 [1] | 1.75 [0.0688] | 1.15 [1.01] | 1.9 [0.113] | – [1] | – [1] |
| reynolds (1385) | free stream | 0.431 [0.119] | 1.75 [0.0688] | 0.427 [0.146] | 1.9 [0.113] | – [0.429] | – [1] |

**Mean_stable test cases: mean [median] over cases, seed means.**

| split (n) | model | ε_U | ε_p | ε^N_U | ε^N_p | relL2 u, registered 200k [full mesh] | relL2 p, registered 200k [full mesh] |
|---|---|---|---|---|---|---|---|
| full (515) | ISLA (interacting) | 0.0761 [0.0492] | 0.207 [0.0217] | 0.109 [0.0908] | 0.22 [0.0414] | 0.11 [0.11] | 0.336 [0.337] |
| full (515) | ISLA-DTT | 0.0732 [0.0492] | 0.198 [0.0193] | 0.103 [0.0845] | 0.203 [0.0372] | 0.102 [0.102] | 0.28 [0.281] |
| full (515) | GeoTransolver | 0.704 [0.538] | 1.82 [0.0922] | 0.692 [0.518] | 2.03 [0.129] | 0.571 [0.571] | 1.33 [1.33] |
| full (515) | Transolver | 1.2 [1.01] | 1.82 [0.0928] | 1.19 [1.01] | 2.02 [0.124] | 0.998 [0.998] | 1.23 [1.23] |
| full (515) | zero predictor | 1.2 [1.01] | 1.81 [0.0587] | 1.19 [1.01] | 2.01 [0.108] | – [1] | – [1] |
| full (515) | free stream | 0.44 [0.11] | 1.81 [0.0587] | 0.442 [0.16] | 2.01 [0.108] | – [0.389] | – [1] |
| bank_zero_to_cad (234) | ISLA (interacting) | 0.112 [0.0786] | 0.28 [0.0391] | 0.17 [0.136] | 0.315 [0.0708] | 0.16 [0.16] | 0.4 [0.4] |
| bank_zero_to_cad (234) | ISLA-DTT | 0.111 [0.0786] | 0.268 [0.0381] | 0.165 [0.132] | 0.298 [0.0726] | 0.152 [0.152] | 0.379 [0.379] |
| bank_zero_to_cad (234) | GeoTransolver | 0.531 [0.248] | 2.12 [0.122] | 0.544 [0.296] | 2.25 [0.176] | 0.451 [0.451] | 1.14 [1.14] |
| bank_zero_to_cad (234) | zero predictor | 1.23 [1.01] | 2.09 [0.0935] | 1.2 [1.01] | 2.23 [0.162] | – [1] | – [1] |
| bank_zero_to_cad (234) | free stream | 0.482 [0.15] | 2.09 [0.0935] | 0.507 [0.246] | 2.23 [0.162] | – [0.435] | – [1] |
| blobs (917) | ISLA (interacting) | 0.0936 [0.0651] | 0.194 [0.0284] | 0.137 [0.111] | 0.197 [0.0523] | 0.139 [0.139] | 0.439 [0.439] |
| blobs (917) | ISLA-DTT | 0.0847 [0.0593] | 0.182 [0.024] | 0.123 [0.0995] | 0.169 [0.0445] | 0.122 [0.122] | 0.34 [0.341] |
| blobs (917) | GeoTransolver | 0.831 [0.638] | 1.7 [0.102] | 0.822 [0.633] | 1.86 [0.134] | 0.686 [0.686] | 1.14 [1.14] |
| blobs (917) | zero predictor | 1.2 [1.01] | 1.66 [0.0656] | 1.19 [1.01] | 1.84 [0.113] | – [1] | – [1] |
| blobs (917) | free stream | 0.421 [0.105] | 1.66 [0.0656] | 0.426 [0.164] | 1.84 [0.113] | – [0.381] | – [1] |
| nonbox (692) | ISLA (interacting) | 0.371 [0.237] | 0.792 [0.109] | 0.301 [0.283] | 0.595 [0.142] | 0.407 [0.407] | 2.66 [2.66] |
| nonbox (692) | ISLA-DTT | 0.362 [0.218] | 0.836 [0.273] | 0.283 [0.242] | 0.64 [0.268] | 0.388 [0.388] | 4.11 [4.12] |
| nonbox (692) | GeoTransolver | 0.343 [0.203] | 0.609 [0.0574] | 0.27 [0.191] | 0.442 [0.0958] | 0.431 [0.431] | 1.22 [1.22] |
| nonbox (692) | E1-AZ HEADNORM | 0.42 [0.309] | 0.85 [0.178] | 0.352 [0.353] | 0.595 [0.156] | 0.44 [0.44] | 3.81 [3.82] |
| nonbox (692) | zero predictor | 1.06 [1] | 0.595 [0.0391] | 1.02 [1] | 0.43 [0.087] | – [1] | – [1] |
| nonbox (692) | free stream | 0.238 [0.075] | 0.595 [0.0391] | 0.147 [0.136] | 0.43 [0.087] | – [0.339] | – [1] |
| profile_crossflow (173) | ISLA (interacting) | 1.36 [1.38] | 5.4 [5.55] | 1.65 [1.68] | 5.53 [5.64] | 1.38 [1.39] | 2.3 [2.3] |
| profile_crossflow (173) | ISLA-DTT | 1.34 [1.42] | 5.34 [5.58] | 1.64 [1.68] | 5.62 [5.76] | 1.32 [1.32] | 2.01 [2.01] |
| profile_crossflow (173) | GeoTransolver | 0.922 [0.86] | 3.49 [2.62] | 0.865 [0.845] | 3.86 [2.79] | 0.865 [0.865] | 1 [1] |
| profile_crossflow (173) | zero predictor | 1.12 [1.05] | 3.49 [2.62] | 1.01 [0.968] | 3.86 [2.78] | – [1] | – [1] |
| profile_crossflow (173) | free stream | 0.947 [0.886] | 3.49 [2.62] | 0.907 [0.883] | 3.86 [2.78] | – [0.895] | – [1] |
| profile_dual_inlet (459) | ISLA (interacting) | 1.38 [1.36] | 6.91 [5.78] | 1.47 [1.43] | 7.31 [5.96] | 0.652 [0.652] | 0.803 [0.803] |
| profile_dual_inlet (459) | ISLA-DTT | 1.27 [1.27] | 7.09 [6.06] | 1.34 [1.34] | 7.55 [6.32] | 0.607 [0.607] | 0.831 [0.831] |
| profile_dual_inlet (459) | GeoTransolver | 1.65 [1.57] | 9.13 [7.14] | 1.6 [1.57] | 9.89 [7.62] | 0.787 [0.787] | 1 [1] |
| profile_dual_inlet (459) | zero predictor | 2.12 [2.05] | 9.13 [7.15] | 2.1 [2.07] | 9.89 [7.62] | – [1] | – [1] |
| profile_dual_inlet (459) | free stream | 1.66 [1.56] | 9.13 [7.15] | 1.61 [1.57] | 9.89 [7.62] | – [0.793] | – [1] |
| reynolds (1001) | ISLA (interacting) | 0.106 [0.0576] | 0.37 [0.0295] | 0.134 [0.0968] | 0.357 [0.0529] | 0.151 [0.151] | 0.554 [0.555] |
| reynolds (1001) | ISLA-DTT | 0.102 [0.0592] | 0.387 [0.0378] | 0.13 [0.0945] | 0.39 [0.0549] | 0.138 [0.138] | 0.511 [0.512] |
| reynolds (1001) | GeoTransolver | 0.807 [0.601] | 1.64 [0.0518] | 0.8 [0.607] | 1.77 [0.0928] | 0.679 [0.679] | 1.05 [1.05] |
| reynolds (1001) | LRE-AZ B | 0.128 [0.0617] | 0.516 [0.0333] | 0.153 [0.099] | 0.52 [0.0581] | 0.169 [0.169] | 0.583 [0.584] |
| reynolds (1001) | zero predictor | 1.19 [1] | 1.64 [0.0457] | 1.17 [1.01] | 1.77 [0.0894] | – [1] | – [1] |
| reynolds (1001) | free stream | 0.379 [0.076] | 1.64 [0.0457] | 0.382 [0.128] | 1.77 [0.0894] | – [0.359] | – [1] |

## 6. The order of the architectures under each metric

The per-split order under every metric, statistic and case set is in `TABLES_FULL.md` ("Order (best first)"); the rows where it
differs from the registered relL2 are in `ORDERS.md` (21 of the 96 split × pair × case-set × statistic rows change on at least one
field). On case means over all test cases:

| split | ε_U | ε_p | under registered relL2 (u; p) |
|---|---|---|---|
| full | DTT < ISLA < GT < T | DTT < ISLA < T ≈ GT | DTT < ISLA < GT < T; DTT < ISLA < T < GT |
| bank_zero_to_cad | DTT < ISLA < GT | DTT < ISLA < GT | same |
| blobs | DTT < ISLA < GT | DTT < ISLA < GT | same |
| nonbox | **GT < DTT < ISLA** | GT < **DTT < ISLA** | DTT < ISLA < GT; GT < ISLA < DTT |
| profile_crossflow | GT < DTT < ISLA | GT < DTT < ISLA | same |
| profile_dual_inlet | DTT < ISLA < GT | ISLA < DTT < GT | same |
| reynolds | DTT < ISLA < GT | **ISLA < DTT** < GT | DTT < ISLA < GT; DTT < ISLA < GT |

(DTT = ISLA-DTT, ISLA = interacting ISLA, GT = GeoTransolver, T = Transolver; bold = a change.) What changes:
- **nonbox velocity:** GeoTransolver moves ahead of both forms of ISLA on ε_U (0.396 against 0.404 and 0.418; medians 0.264 against
  0.324 and 0.334) and on ε^N_U, on all cases and on mean_stable. The audit's records-based screen flagged exactly this pair (K* 1.54).
- **ISLA against ISLA-DTT on pressure:** on nonbox ISLA-DTT moves ahead (0.706 against 0.720; R 0.981), on reynolds interacting ISLA moves
  ahead (0.371 against 0.401; R 1.082), as the audit's estimate predicted; on medians and mean_stable a few more ISLA/ISLA-DTT pairs swap
  (bank_zero_to_cad and profile_crossflow; `ORDERS.md`).
- Everything else keeps its order. GeoTransolver against ISLA keeps its order on every other split and field, with the gaps larger on
  pressure (for example `full` ε_p 1.69 against 0.169).

**Against the references** (the ratio of each model's mean ε_U to the free stream's, and of its mean ε_p to the zero predictor's; below 1
is skill): interacting ISLA / ISLA-DTT 0.17 / 0.16 and 0.10 / 0.10 on `full`, 0.18-0.27 and 0.10-0.23 on bank_zero_to_cad, blobs and
reynolds, 0.77-0.83 and 0.75-0.77 on profile_dual_inlet, but **1.50-1.55 and 1.51-1.54 on nonbox and 1.47-1.51 and 1.83-1.94 on
profile_crossflow**: on those two held-out families both forms of ISLA are worse than AeroZoo's free stream on velocity and worse than
predicting zero pressure, on ε as on relL2 (on crossflow ISLA's ε_U, 1.29, is above even the zero predictor's 1.04). GeoTransolver:
0.97-1.90 on velocity, 1.00-1.03 on pressure, so no skill beyond the free stream anywhere (its best is crossflow velocity, 0.97).
Transolver: 2.45 and 1.00.

## 7. §12(f): ISLA against ISLA-DTT

The registered read (PREREG_v2 §12(f), on test sides, registered relL2) stands as written: a close call, so ISLA-DTT is the default
interior form, on principle (Peter, 10-23); and since 10-24, "interior means "use DTT"". The validation-side re-read was dropped
(supervisor, 23:02 EDT 10-24): under those rulings it decides nothing. Recomputed here from the registered files, the registered read
reproduces: six-split mean R = 0.928 (velocity), 0.968 (pressure), nonbox pressure R 1.274 > 1.05, so "close call → ISLA-DTT".

The same bands on the same test sides under the spec metric, descriptive only (`READINGS.md`): R(ε_U) 0.912-0.990 on the six held-out
splits, mean 0.958; R(ε_p) 0.940-1.082, mean 0.990, above 1.05 on reynolds (1.082; seeds 1.516 and 0.753) → also **a close call**.
Under ε^N: mean R 0.944 and 0.989, the same band. On `full` (reported) R = 0.978 and 0.959. The seed-matched ratios straddle 1 on
nonbox, crossflow, dual_inlet and reynolds pressure, so most per-split pressure differences between the two forms are within seed
spread. S1 is handled as text (NOTEBOOK_DRAFT.md): the default rests on principle, not on the test-side read.

## 8. E1-AZ (HEAD-NORM) beside its registered REJECT

The registered verdict, REJECT by branch 1 (pressure relL2 ratio HEADNORM ÷ CONTROL 1.287, seeds 1.339 and 1.178, all ≥ 1.10),
**stands as written**; reproduced exactly here from the registered files. Read on the spec metric in the rule's own form (seed-mean
ratio of case means, family_non_box test, 1,074 cases; CONTROL = azc_nonbox_isla):

| metric | velocity ratio (s42, s43) | pressure ratio (s42, s43) | branch 1 | G14 |
|---|---|---|---|---|
| registered relL2 (the verdict) | 1.047 (1.022, 1.081) | **1.287 (1.339, 1.178)** | fires | 1.004 |
| ε_U, ε_p, case means | 1.075 (1.080, 1.066) | 1.064 (1.002, 1.157) | does not fire | 0.992 |
| ε_U, ε_p, case medians | 1.192 (1.276, 1.046) | 1.298 (1.412, 1.103) | fires (pressure) | 1.014 |
| ε, mean_stable means | 1.131 (1.180, 1.053) | 1.073 (1.065, 1.083) | does not fire | 0.996 |
| ε^N, case means | 1.066 (1.127, 0.942) | 1.007 (0.971, 1.065) | does not fire | 0.988 |

So on the spec metric in the rule's form, branch 1 does not fire and G14 = 0.992 ≤ 1.01 with no seed-mean ratio ≥ 1.10, so the adopt
conditions (a') would hold, with (b1)-(b3) as recorded; on case medians branch 1 fires on pressure. HEADNORM is worse than CONTROL on
every row (every seed-mean ratio > 1). The reading therefore depends on the statistic, and the registered REJECT is not reversed by this table;
whether HEAD-NORM's status changes is Peter's call (§11). Full table: `READINGS.md`.

## 9. LRE-AZ beside its registered verdict

Registered R1 (relL2, reynolds test): velocity 1.126 "harms", pressure 1.067 "no change" → not adopted without Peter; reproduced. Under
the spec metric: ε_U 1.218 (seeds 1.249, 1.190; D 0.197 > 2 SE 0.154) "harms", ε_p 1.433 "no change" (seed spread large, 2 SE 0.404),
ε^N_U 1.155 "harms"; case medians beside: 1.083 and 1.116, "no change". The consequence (not adopted) holds and is stronger on pressure.

## 10. What this means, and how sure

- The AeroZoo standing now has AeroZoo's own numbers, scored exactly as the spec's reference functions define them over every cell, with
  the two independent checks and AeroZoo's own free-stream file agreeing to 1e-8. Confidence in the numbers: high.
- Moving from relL2 to ε changes three readings: GeoTransolver passes ISLA on nonbox velocity; ISLA and ISLA-DTT swap on nonbox and reynolds
  pressure. The headline does not change: both forms of ISLA lead the baselines by 4-10× on velocity and on pressure on four of the
  seven splits, and GeoTransolver has no skill beyond the free stream on any split.
- The failures are sharper under ε: on nonbox and profile_crossflow both forms of ISLA are worse than the free stream on velocity and
  worse than zero on pressure. These two families are where the book's failure section already points.
- E1-AZ: the REJECT rests on the mean of a per-case relative error; on the spec's mean ε it would not fire. Peter may want to see this
  before the architecture freeze (~10-27); the registered verdict stands until he decides.

## 11. Rulings relied on, and what is for Peter

- Relied on: Peter's "ISLA-DTT, on principle" (10-23) and "interior means "use DTT"" (10-24) for §7; the supervisor's cap and order
  (23:02 EDT 10-24: cap 220, (iii) dropped; 02:58 EDT 10-25: cap 265, finish (ii)); `global_data/reference` and the cell volume used for
  scoring only (the brief).
- For Peter: (1) whether the spec metric becomes the primary AeroZoo number in the book (NOTEBOOK_DRAFT.md proposes it, with relL2
  beside); (2) E1-AZ: the REJECT stands as registered, but on the spec's mean ε it would not fire (§8); any change to HEAD-NORM's status
  is his; (3) the baselines have no skill beyond AeroZoo's free stream under its metric, which BASELINES_FOR_AEROZOO.md (md5 ced2ee93,
  sent by the supervisor at 04:16 EDT) says plainly.

## 12. Files

- `RESULTS.md` (this file); `TABLES_FULL.md` (every lane and seed, all and mean_stable, orders); `READINGS.md` (E1-AZ, LRE-AZ, §12(f)
  bands, all statistics); `ORDERS.md` (order changes, models above the zero predictor); `PARTITION.md` (partition and chunk probes);
  `AEROZOO_SPECMETRIC.json` (the book's input); `BASELINES_FOR_AEROZOO.md`; `NOTEBOOK_DRAFT.md`; `JOBS.md`.
- Code: `lhr/specscore.py` (scorer; `lhr/specscore_1edc8f04.py` the earlier version), `lhr/indep_check.py`, `lhr/score.sbatch`,
  `lhr/smoke*.sbatch`, `lhr/plans/`, `lhr/watch.sh`; reductions `reduce.py`, `analysis.py`, `orders.py`, `partition.py`,
  `build_json.py`, `results_tables.py`, `make_baselines.py` (run: `reduce.py data/score data/final; analysis.py data/final;
  orders.py data/final; build_json.py data/score data/final AEROZOO_SPECMETRIC.json; results_tables.py`).
- Data: `data/score/` (63 MB, the per-case rows), `data/smoke*/`; on LHR `$T/azrescore_out/{score,smoke,smoke3} (smoke 2 wrote into smoke/)` (81 MB of rows
  and logs). The smoke's prediction dumps (11 GB) and the per-process compile caches (1.8 GB) were deleted after the checks.
