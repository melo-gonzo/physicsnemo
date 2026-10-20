# HILIFT-090 Part A: per-case error analysis of A2 ISLA on HiLift

Written 2026-10-20 (EDT). Data: `highlift_surface_medium` val split (180 cases), 200k-cell score set, float32, epoch 500.
Every number below comes from `reduce_a.py`, which writes the full tables to `tables_a.md` and one row per case to
`percase.csv`. Run it with `nice -n 19 python3 reduce_a.py`. It takes the last `infer_step` row per case and matches
cases on `geo_LHCxxx_AoA_yy`. Unless a row says otherwise, values are seed means (seeds 42 and 43) of per-case
relative L2, native protocol. Arms:
ISLA = a2r_hl_isla (A2); GT = udrv_hl_gt (GeoTransolver 256, unit drive); T = udrv_hl_t (Transolver 512x16, unit
drive); old = CHAR r80_hl_isla_ref (pre-A2 ISLA, AGA 200k subsample); GT-phys = r80_hl_gt_h256, both as the CHAR
record and as its LHR re-score.

## Premise check: the "6% A2 regression against old ISLA" does not reproduce

| | pressure | wall shear |
|---|---|---|
| A2 ISLA, case mean (s42 / s43 infer_summary) | 0.0432 (0.0435 / 0.0430) | 0.0943 |
| old r80_hl_isla_ref (s42 / s43 infer_summary) | 0.0450 (0.0442 / 0.0459) | 0.0984 |
| A2 ÷ old, ratio of means: seed 42 / seed 43 / seed mean | 0.986 / 0.933 / **0.959** | 0.976 / 0.941 / **0.958** |
| A2 ÷ old, median per-case ratio | 0.900 | 0.953 |
| cases where A2 is worse | 21% | 16% |

Against the reference run named in the brief, A2 is about 4% better, not 6.2% worse. None of the CHAR ISLA variants
gives 1.062 with A2 in the numerator either (w512_s512: about 1.07; w512: about 1.03; s512: about 1.02). The 1.062 /
1.010 figures must come from another reference or another table. The supervisor has been told
(progress_update). Q3 below answers the question for r80_hl_isla_ref.

## Findings, ranked by how much of the gap they explain

**F1. The ISLA-vs-GeoTransolver pressure gap is a handful of high-AoA post-stall cases.** ISLA/GT is 1.107 for pressure
and 1.059 for wall shear. In the AoA ≥ 16 bins (45% of cases) the pressure gap is 101% of the total. The AoA 16, 18
and 22 bins give 41%, 37% and 27%; AoA 20 gives −4% and AoA 14 gives −7%.
- 5 cases carry 76% of the gap and 10 cases carry 101%. Dropping those 10 makes ISLA/GT 0.999 for pressure and 1.017
  for wall shear.
- Seventeen "branch cases" (AoA 14-22) are cases where some run's error is above 0.08 and more than twice the best
  run's. Together they carry 81% of the ISLA-GT gap. Without them, ISLA/GT is 1.025 and ISLA/T is 1.084.
- Inside the AoA ≥ 16 cases, the median per-case ISLA/GT is only 1.047, and ISLA is worse than GT on 60% of them, but
  by more than 0.02 absolute on only 10%. Five geometries carry 75% of the AoA ≥ 16 gap, and ten carry 96%. The high-AoA
  badness is therefore a few cases, not all high-AoA cases.
- These cases behave like a branch choice: the predicted flow is either separated or attached, and the error is close
  to all or nothing. Examples (s42/s43):

| case | A2 ISLA | old ISLA | GT unit | T unit | GT phys | share of ISLA-GT gap |
|---|---|---|---|---|---|---|
| LHC002_AoA_18 | 0.214/0.223 | 0.252/0.200 | 0.054/0.043 | 0.033/0.146 | 0.062/0.062 | 22.8% |
| LHC132_AoA_22 | 0.203/0.177 | 0.227/0.175 | 0.066/0.049 | 0.150/0.130 | 0.164/0.155 | 17.8% |
| LHC153_AoA_16 | 0.269/0.288 | 0.045/0.268 | 0.251/0.087 | 0.286/0.130 | 0.239/0.239 | 14.6% |
| LHC162_AoA_16 | 0.255/0.241 | 0.168/0.232 | 0.124/0.192 | 0.250/0.165 | 0.192/0.196 | 12.0% |
| LHC119_AoA_16 | 0.243/0.173 | 0.204/0.216 | 0.074/0.204 | 0.131/0.066 | 0.057/0.182 | 9.3% |
| LHC131_AoA_18 | 0.109/0.107 | 0.119/0.096 | 0.066/0.030 | 0.032/0.076 | 0.039/0.043 | 8.0% |

  ISLA, both A2 and old and on all 4 seed runs, takes the wrong branch on LHC002_18 and LHC132_22, where GT-unit is
  right on both seeds. Those two cases alone are 41% of the ISLA-GT gap. In the other branch cases, every model flips
  from seed to seed (for example LHC153_16 and LHC119_16 for GT and for old ISLA), so their share of the gap in any one
  2-seed comparison is close to a coin toss.
- Seed noise in the gap: the per-case ISLA−GT pressure gap has a Spearman correlation of only 0.36 between seeds (0.44
  for wall shear). The per-seed ISLA/GT ratio of means is 1.115 / 1.099.

**F2. Against Transolver, ISLA has a broad offset, not outliers.** ISLA/T is 1.096 for pressure, but the median per-case
ratio is 1.125 and ISLA is worse on 85% of cases. In every bin from AoA 4 to 12, ISLA/T is 1.12-1.20 for pressure and
1.07-1.08 for wall shear. AoA ≥ 16 gives 75% of the ISLA-T pressure gap, close to its 45% share of cases plus its larger
errors. The branch cases give 34%. For wall shear, ISLA/GT also has a broad offset of about 2% (median 1.022, worse on
77%, AoA 4-12 bins 1.015-1.026), and on top of it AoA 16-22 gives 92% of the gap.

**F3. A2 versus old ISLA (Q3): A2 is better everywhere except the noisy AoA 16-20 bins.** For pressure (seed mean),
A2/old is 0.89-0.93 in every bin from AoA 4 to 14, and per-case medians are 0.87-0.90. AoA 16 is 1.047, AoA 18 0.983,
AoA 20 1.026 and AoA 22 0.916. The ratio-vs-AoA Spearman is 0.28, so the improvement shrinks at high AoA.
- The AoA 16 bin swings with the seed: 1.350 on s42 and 0.843 on s43. This is the branch-case coin toss (for example,
  old s42 got LHC153_16 right at 0.045).
- Per-seed shares of the A2−old gap therefore change sign from bin to bin. Only the low-AoA improvement, about 10%, is
  stable on both seeds. Wall shear is a uniform 0.94-0.96 in every bin except AoA 16 (1.002) and AoA 20 (0.984).
- Noise floor: the same GT-phys model on the two 200k subsamples (CHAR vs LHR) gives per-case ratio sd 0.034 (seed
  mean; single seed 0.037-0.046), |r−1| p90 of 0.05, and bin ratio of means within 0.99-1.01, except AoA 16 at 0.970.
  For wall shear, sd is 0.024 and p90 is 0.038. So per-case A2/old ratios within ±5% are noise, but the bin-level ~10%
  improvement at AoA ≤ 14 is real.

**F4. AoA, training exposure and geometry do not separate bad high-AoA cases from good ones (Q1/Q2).** Within the 81
AoA ≥ 16 cases, the ISLA/GT ratio has |Spearman| ≤ 0.11 with each of: AoA, post-stall flag, high-deflection flag, number
of same-geometry train cases, number of same-geometry AoA ≥ 16 train cases, |ΔAoA| to the nearest same-geometry train
case, slat/flap area share, and n_cells. Case difficulty predicts ISLA's error: the Spearman with GT error is 0.93 and
with T error 0.90.
- Cases where ISLA/GT is above the median have somewhat harder GT errors (0.060 vs 0.050), slightly fewer train cases
  of the same geometry (2.25 vs 2.73), and a slightly higher high-deflection share (25% vs 17%).
- High-deflection geometries at AoA ≥ 16: ISLA/GT is 1.288 (n=17, 25% of the gap). Other geometries at AoA ≥ 16: 1.152
  (n=64, 76%).
- No flap/slat deflection table exists in the dataset root (README, splits.md, units.json, manifest.json), in
  hl_medium_manifest.json, or in examples/cfd/{isla,external_aerodynamics}. The per-case `element_labels.json` files
  hold only area shares. Proxies used instead: `deflection_test` from manifest.json (the 36 geometries with the highest
  mean deflection) and `stall_test` (post-stall cases).

**F5. Training distribution (Q2).** `hl_medium_manifest.json` equals `medium_train`/`medium_val` of the dataset
manifest, which is a random case split: train 510, val 180.
- 173 geometries appear in train and 118 in val. 113 of the 118 val geometries also appear in train at other AoAs;
  only 5 do not.
- Train cases per AoA are flat, 45-60 per value. AoA 16, 18, 20 and 22 have 49, 45, 50 and 48 train cases (192 in
  total, 37.6%), and 123 of the 173 train geometries have at least one AoA ≥ 16 train case.
- Val cases per AoA (4→22): 17, 18, 19, 13, 17, 15, 15, 27, 15, 24. AoA 18 and 22 are over-represented in val.
- Post-stall (stall_test): 84 of 180 val cases and 191 of 510 train cases. By AoA in val: 14 → 5/15, 16 → 14/15, and
  18-22 → all.
- High AoA is not under-trained in count. The failures are cases where the flow regime is ambiguous.

**F6. Seed consistency (Q4).** Every model ranks cases almost identically on both seeds:
- seed-to-seed Spearman: ISLA 0.983 for pressure and 0.984 for wall shear, GT 0.979 / 0.960, T 0.979 / 0.952, old
  0.964 / 0.942;
- overlap of the 18 worst cases: 14-16 of 18 for every model.
- The worst cases are shared across models (LHC040_16, LHC087_18, LHC153_16, LHC060_14, LHC118_14, LHC162_16). Across
  models, the Spearman of seed-mean pressure is 0.965 for ISLA vs GT and 0.979 for ISLA vs old.
- What is not consistent is the difference between models on a case. The A2−old per-case gap has a seed-to-seed
  Spearman of only 0.08 for pressure and 0.13 for wall shear.

**F7. Protocol ratios for A2 ISLA (Q5).**
- Posed/native: 1.009 for pressure (per-case median 1.012, sd 0.017, range 0.993-1.082) and 1.006 for wall shear
  (sd 0.007). The ratio falls with native error (Spearman −0.80) and with AoA (−0.45), and does not depend on case size
  (n_cells, +0.11).
- In absolute terms, posed adds an almost constant +2 to +6 ×1e-4 to every case (pressure: 6.3e-4 at AoA 12, 2-3e-4 at
  AoA 18-22). The ratio is largest where the error is small (1.034 at AoA 12, 1.004 at AoA 16-22).
- This is a small, uniform additive shift, not a high-AoA effect. old ISLA shows the same: 1.008 / 1.005.
- Area/native: 0.995 for pressure and 1.002 for wall shear overall. Per case the spread is large (pressure sd 0.053,
  range 0.76-1.21), but it does not correlate with AoA (−0.04), n_cells (−0.04) or error (−0.30). The AoA 16 and 18
  bins (0.963, 0.981) push the mean down.
- Side note: GT and T unit-drive posed/native is 24× / 22× (area 4.3× / 3.5×). This matches the known non-invariance of
  the baselines in `udrv_hl_table.md`, where posed pressure is 0.95 and 0.87.

## What it means

- The headline ISLA-GT pressure gap of 10.7% comes from about 10 post-stall branch cases. Two of them, LHC002_18 and
  LHC132_22, are systematic for ISLA. Most of the rest are seed coin tosses.
- Without those cases, ISLA is level with GeoTransolver (about 1.00-1.03).
- Against Transolver the deficit is broad: about 8-12% on ordinary cases.
- A2 improves on old ISLA by about 10% at AoA ≤ 14 and by about 4% overall.
- Confidence: high for the ranking and concentration statements, which agree on both seeds and lie above the measured
  subsample noise. Low for any single high-AoA bin's share in a 2-seed comparison.

Missing input: one val case has no `element_labels.json`, so `case_meta.json` covers 179 cases and the n_cells /
area-share correlations use those 179.
