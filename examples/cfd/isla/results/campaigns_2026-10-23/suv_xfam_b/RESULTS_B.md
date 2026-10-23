# SUV-XFAM-B stage 1: fastback-trained ISLA 0.9.0 and Transolver (read 2026-10-23 07:05 UTC)

**Stage 2 added 2026-10-23 21:32 UTC (sections 7-10 below): against fastback-trained GeoTransolver too, ISLA transfers
better on wall shear** (forward gap 1.78 against 1.92, ratio 1.078 at 2.7 SE; replicate 1.086 at 2.9 SE). Hardness and
the reverse direction read no difference, and pressure reads no difference in every comparison.

Registered rules: PREREG_B.md (approved on 6e77d876; amendments in section 8; now 57d188c3). Jobs, md5s and paths:
JOBS.md, part B. Raw numbers: reduce_b_out.json (reduce_b.py 25a1a87c, as registered) and identity_b_out.json
(identity_b.py 4b03096a, amendment 2). Scores: LHR $T/evals_suvxb/<run>/<split>/<run>/metrics.jsonl. Notation: E(x→y) is
the error of a model trained on family x and scored on family y; fb = fastback, est = estate; seed means are geometric.

**Headline.** ISLA's smaller wall-shear penalty from part A is mostly a **transfer advantage**. Against the in-family
fastback reference, ISLA's estate-trained model loses 1.78× on fastback wall shear and Transolver's loses 1.93×
(D_G ratio 1.085 at 5.8 SE; the replicate gives 1.098 at 8.7 SE). Intrinsic hardness carries the rest of part A's
11 percent: fastback wall shear is 2.3 percent relatively harder for Transolver in family (3.8 SE, but under the
registered 3-percent floor, so "no difference"). The registered split of part A's D_P is 78 percent G and 22 percent H.
The reverse direction (fastback-trained models on estates) shows **no difference** on either field: the fastback-trained
seeds differ by 7-9 percent on estates, too much for a 3-percent call. Pressure shows no difference anywhere. The
stage-2 trigger fires.

## 0. Validity

- Lanes (smoke_check.py on the 4 real runs): each resolved config differs from its estate reference run's only in
  `dataset` and `run_id`; Transolver `forward_kwargs.fx.source: global_data.U_inf_dir`; 199 train / 25 val per rank,
  dataset_size 995; all losses finite; norm_stats.pt = the declared constants; Training completed at epoch 500;
  validation epochs 0-499 without gaps (the quota event of 10-23 00:38 UTC hung three links, and each successor resumed
  from its newest checkpoint; JOBS.md).
- Training splits (splitcheck.py, job 334400): both lanes read exactly fastback/train (796) and fastback/val (99) under
  the fastback root, disjoint.
- 16 of 16 scorings: `"epoch": 500`, n 99 / 99 / 100 / 99, the requested split and dataset, float32, one START, no FAILED.
- Scored-set identity. (a) Loader digests equal part A's for both loaders on every split (fb_test 4241f134, fb_val
  1bb14edd, est_test 24460c3d, est_val 983a2c64). (b) **As registered: fails on est_val** (1.28e-4 > 1e-4; one near-zero
  roll moment, case 00888 CMR, ref -4.36e-6, absolute difference 5.6e-10, the same in all four Transolver est_val
  scorings, estate-trained ones included); passes on fb_test 1.1e-5, fb_val 4.2e-5, est_test 1.2e-5. **Amended**
  (section 8, approved before results; the denominator is max(|ref|, the component's median)): passes on all four
  splits, largest 6.1e-7.

## 1. In-family fastback error E(fb→fb), fastback test (B1; seed 42 · 43)

| arm | pressure | wss |
|---|---|---|
| ISLA 0.9.0 | 0.0425 (0.0426 · 0.0424) | 0.0656 (0.0658 · 0.0654) |
| Transolver | **0.0405** (0.0405 · 0.0405) | 0.0643 (0.0644 · 0.0642) |

| cell | T ÷ ISLA | b1 | verdict | fastback val (replicate) | for comparison: estate-trained, estate test (part A) |
|---|--:|--:|---|---|---|
| pressure | 0.953 | 0.030 | **Transolver leads** | 0.948, leads | 0.951, leads |
| wss | 0.980 | 0.030 | level | 0.968, leads | 0.958, leads |

In family, the fastback ordering repeats the estate one: Transolver leads on pressure by 5 percent and is 2-3 percent
ahead on wall shear.

## 2. G, R and H per arm (lead sets; SE of each ln in brackets)

| quantity | ISLA pressure | T pressure | ISLA wss | T wss |
|---|--:|--:|--:|--:|
| G = E(est→fb) ÷ E(fb→fb), fastback test | 2.629 (0.016) | 2.625 (0.003) | **1.780** (0.012) | **1.931** (0.007) |
| R = E(fb→est) ÷ E(est→est), estate test | 3.144 (0.044) | 3.431 (0.034) | 1.976 (0.037) | 2.042 (0.044) |
| H = E(fb→fb) ÷ E(est→est) | 0.972 (0.018) | 0.974 (0.005) | 0.999 (0.005) | 1.022 (0.003) |
| P = G × H (part A) | 2.557 | 2.557 | 1.778 | 1.972 |

E(fb→est), estate test, seed 42 · 43: ISLA pressure 0.1320 · 0.1430, wss 0.1347 · 0.1251; T pressure 0.1476 · 0.1379,
wss 0.1343 · 0.1230. Fastbacks are not harder than estates in themselves (H within 3 percent of 1 for both arms on both
fields). Part A's penalty is therefore almost entirely transfer, and the reverse transfer costs more than the forward one
(R > G for both arms on both fields).

## 3. Comparisons D = ln(X_T ÷ X_ISLA); a cell needs |D| > 2 SE and |D| ≥ ln 1.03

| cell | lead: T ÷ ISLA | D / SE | verdict | replicate | verdict |
|---|--:|--:|---|--:|---|
| **D_G wss** (B2, decision cell) | **1.085** | **5.8** | **ISLA transfers better** | 1.098 (8.7 SE) | better |
| D_G pressure | 0.999 | -0.1 | no difference | 1.010 (0.6) | no difference |
| D_R wss (B3) | 1.034 | 0.6 | no difference | 1.033 (0.6) | no difference |
| D_R pressure | 1.091 | 1.6 | no difference | 1.104 (1.8) | no difference |
| D_H wss (B4) | 1.023 | 3.8 | no difference (below the 3-percent floor) | 1.007 (0.7) | no difference |
| D_H pressure | 1.002 | 0.1 | no difference | 1.006 (0.2) | no difference |

Registered reading: **wss, "transfer advantage, estate → fastback only"** (B2 better, B3 no difference); the replicate
gives the same. Pressure: B2 and B4 both no difference, "split unresolved", with nothing to split (part A's D_P is 0.000).
Shares of part A's wss D_P = 0.104: D_G 0.081 (78 percent), D_H 0.023 (22 percent). The replicate's shares do not sum to
1, because part A's replicate P divides by estate test while the G and H replicates divide by estate val.

D_R reads "no difference" because of the seed spread, not because the effect is small. On pressure, ISLA's reverse gap
is 9 percent smaller, but each arm's two fastback-trained seeds differ by 7-9 percent on estates (ISLA 8.3 / 7.7,
T 7.0 / 9.2 percent, pressure / wss), which sets SE ≈ 0.056. The same models' seeds agree to within 0.5 percent on
fastbacks. Part A saw the same with GeoTransolver: transfer amplifies seed spread.
*Erratum (supervisor, 2026-10-23 07:05 EDT, found by BOOK-AM's second read):* the fastback-trained seeds agree within 0.7 percent
on fastback test and within 0.9 percent on fastback val (reduce_b_out.json), not 0.5 percent. ISLA's estate-test pressure
spread is 8.4 percent at the json's full precision (8.3 from the four-digit seed values). No verdict changes.

## 4. Stage-2 trigger: FIRES (B2 wss reads "ISLA transfers better")

The lane proposes stage 2 (GeoTransolver seeds 42-43 on fastback train, copies of clud_suv_gt_seed42/43, about 106 GPU-h)
to the supervisor; it is not launched without approval.

## 5. Predictions against outcomes

| # | prediction | p | outcome |
|---|---|--:|---|
| Q1 | B2 wss "ISLA transfers better" (trigger) | 0.65 | held (1.085, 5.8 SE) |
| Q2 | B2 pressure no difference | 0.65 | held |
| Q3 | B4 wss no difference | 0.55 | held (1.023 at 3.8 SE, under the floor) |
| Q4 | B3 wss "ISLA transfers better" | 0.45 | failed (1.034, 0.6 SE) |
| Q5 | B3 pressure no difference | 0.60 | held (1.091, 1.6 SE) |
| Q6 | B1 T leads ISLA on fb→fb pressure | 0.65 | held (0.953) |
| Q7 | B1 fb→fb wss T ÷ ISLA < 1 | 0.60 | held (0.980) |
| Q8 | G > 1.5 for both arms, both fields | 0.80 | held (lowest 1.78) |
| Q9 | 0.8 < H < 1.25 for both arms, both fields | 0.60 | held (0.97-1.02) |
| Q10 | R > 1.5 for both arms, both fields | 0.70 | held (lowest 1.98) |
| Q11 | replicates give the same B2 and B4 verdicts | 0.80 | held |

## 6. Cost

147.5 GPU-h (sacct, 10-23 07:05 UTC): training ISLA 30.1 + 32.7, Transolver 41.6 + 41.4; smoke 0.5; scoring 1.2; the
cpu-short checks ~0. Registered estimate 144; the excess is mostly the three links hung by the quota event (about
1 h × 4 GPUs each).

# Stage 2: fastback-trained GeoTransolver (read 2026-10-23 21:32 UTC)

Registered rules: PREREG_B.md section 9 (118f36a0, approved as written 2026-10-23 07:15 UTC). Reducer reduce_b2.py
f7c66a5d (as registered) -> reduce_b2_out.json. Runs sxb_gt_seed42/43 (kernel/suvxb/gt; cl_ud rows 4-5 with dataset and
run_id the only changes). D_X = ln(X_GT ÷ X_ISLA), against stage 1's ISLA runs; the rules, SE and floors are stage 1's.

## 7. Validity

- Lanes (smoke_check.py on the real runs, 21:24 UTC): each config differs from clud_suv_gt_seed4x's only in `dataset`
  and `run_id`; `forward_kwargs.global_embedding: global_data.U_inf_dir`; 199 / 25 per rank, dataset_size 995; finite;
  norm_stats.pt = the declared constants; Training completed; validation epochs 0-499 without gaps.
- 8 of 8 scorings (job 358922): epoch 500, n 99 / 99 / 100 / 99, the requested split and dataset, float32, one START,
  no FAILED.
- Identity (b), amended (the check): passes on all four splits (largest 6.1e-7). Registered statistic beside it: passes
  on fb_test, fb_val and est_test; fails on est_val at 1.28e-4, the same near-zero roll moment (case 00888 CMR) as in
  stage 1. GeoTransolver's recipe_v8 force integration gives the same value there as Transolver's.

## 8. In-family fastback error and the quantities (lead sets; seed 42 · 43; SE of each ln in brackets)

| | GeoTransolver pressure | GeoTransolver wss | for comparison: ISLA pressure / wss |
|---|---|---|---|
| E(fb→fb), fastback test | 0.0439 (0.0437 · 0.0441) | 0.0660 (0.0658 · 0.0662) | 0.0425 / 0.0656 |
| E(fb→est), estate test | 0.1364 (0.1415 · 0.1314) | 0.1237 (0.1277 · 0.1199) | 0.1374 / 0.1298 |
| G, fastback test | 2.627 (0.068) | **1.920** (0.025) | 2.629 / **1.780** |
| R, estate test | 3.031 (0.037) | 1.887 (0.031) | 3.144 / 1.976 |
| H | 0.975 (0.004) | 1.007 (0.004) | 0.972 / 0.999 |
| P = G × H (part A) | 2.561 | 1.933 | 2.557 / 1.778 |

B1 (part A's R1 rule): fastback-test pressure GT ÷ ISLA 1.032, b1 0.030, the seeds separate: **GeoTransolver trails**
(fastback val 1.033, trails). Wall shear 1.006, level (val 1.001, level). The estate-trained models were level on estate
test pressure (1.029, inside part A's wider band).

## 9. Comparisons and reading (|D| > 2 SE and |D| ≥ ln 1.03)

| cell | lead: GT ÷ ISLA | D / SE | verdict | replicate | verdict |
|---|--:|--:|---|--:|---|
| **D_G wss** | **1.078** | **2.7** | **ISLA transfers better** | 1.086 (2.9 SE) | better |
| D_G pressure | 0.999 | -0.0 | no difference | 1.002 (0.0) | no difference |
| D_R wss | 0.955 | -0.9 | no difference | 0.954 (-0.9) | no difference |
| D_R pressure | 0.964 | -0.6 | no difference | 0.969 (-0.5) | no difference |
| D_H wss | 1.008 | 1.3 | no difference | 1.001 (0.1) | no difference |
| D_H pressure | 1.003 | 0.2 | no difference | 1.009 (0.4) | no difference |

Registered reading, wss: **"transfer advantage, estate → fastback only"**; the replicate gives the same. Of part A's
wss D_P against GeoTransolver (0.084), 90 percent is the forward gap and 10 percent hardness. Pressure: "split
unresolved" by the registered words, with nothing to split (part A's D_P is 0.002), so the two lose alike.

The D_G SE (0.028) is twice stage 1's against Transolver because GeoTransolver's estate-trained seeds differ by 5
percent on fastback wall shear (part A). The verdict clears 2 SE by 0.019 and the floor by 0.046.

Across both baselines (beside the rules): the forward wall-shear gap is 1.78 for ISLA against 1.92-1.93 for
both baselines, so ISLA's advantage is about 8 percent against each, at 5.8 and 2.7 SE. The forward pressure gap is
2.63 for all three architectures (2.629, 2.625, 2.627). The reverse direction has no consistent sign: Transolver's
wall-shear R is 3 percent above ISLA's, GeoTransolver's 4.5 percent below. Fastback-trained GeoTransolver seeds differ
by 7.4 / 6.3 percent on estate test (pressure / wss) against 0.8 / 0.7 percent on fastback test: the out-of-family
seed spread seen for ISLA and Transolver holds for the third architecture too.

## 10. Predictions S1-S9, cost

| # | prediction | p | outcome |
|---|---|--:|---|
| S1 | D_G wss "ISLA transfers better" | 0.65 | held (1.078, 2.7 SE) |
| S2 | D_G pressure no difference | 0.75 | held |
| S3 | D_H wss no difference | 0.65 | held (1.008) |
| S4 | D_R wss no difference | 0.65 | held (0.955) |
| S5 | D_R pressure no difference | 0.70 | held (0.964) |
| S6 | B1 fastback test: both GT cells level | 0.55 | failed (pressure: GeoTransolver trails by 3.2 percent) |
| S7 | 0.95 < H_GT < 1.05, both fields | 0.70 | held (0.975, 1.007) |
| S8 | GT seeds differ > 5 percent on estate test, some field | 0.60 | held (7.4, 6.3) |
| S9 | replicates give the same D_G and D_H verdicts | 0.75 | held |

Cost (sacct, 21:32 UTC): 105.0 GPU-h, with training 52.2 + 52.1, smoke 0.3 and scoring 0.4. The registered estimate
was 108 and the cap 125. Parts B1 and B2 together: 252.5 GPU-h.
