# HL-AREA: the uniform point cloud and the training cost of HL-DUP stage 2's deduplicated-wing lanes

Written 2026-10-24 at about 21:25 EDT. Brief: the supervisor's HL-AREA task (20:57 EDT). The checkpoints are HL-DUP
stage 2's six lanes, `$T/runs/hldupx_hl_{isla,gt,t}_seed{42,43}`, all at epoch 500 (`scratch/isla-hl-dup-2026-10-24/RESULTS_STAGE2.md`,
md5 4d9fe204). No pre-registration was needed: this lane only scores and prices, and it reads no rule.

## What ran

- **Scorer.** `$T/kernel/hlarea_score/score.sbatch` (7f7176ca) with `score_rows.tsv` (e0575c0a). It is HL-DUP stage 2's registered
  scorer `kernel/hldupx_score/score.sbatch` with these changes only: job name `isla.hlarea.score`; time limit 01:30:00; the five bad
  nodes excluded; the run directory (`A2R_ROOT kernel/hlarea_score`, `EVAL_ROOT evals_hlarea`); and the protocol (rows: `area`
  instead of `native`, on `recipe_hlarea_{isla,base}`). There is one more change, in the dispatch only, to stay within the 3 GPU-h
  cap: array task 0 runs rows 0-3 on GPUs 0-3, and rows 4-5 (Transolver) at the same time on GPUs 0-1. The inference command is
  unchanged: recipe `src/infer.py`, float32 (the log says `precision: float32`), `infer_split=val`, 200,000 cells, the final
  checkpoint, and the same model overrides as the native and posed rows.
- **Protocol twin.** `recipe_hlarea_{isla,base}` (read-only; SNAPSHOT_HEAD; MD5SUMS e5ad6722 / 871ce6a7) are byte copies of
  `recipe_hldupx_{isla,base}` plus one file, `datasets/highlift_surface_medium_dedup_area.yaml` (99d7f52d, the same in both). That
  file is the deduplicated dataset file with CHAR-PROTOCOLS' area diff applied by `patch`. The diff is
  `highlift_surface_medium.yaml` → `highlift_surface_medium_area.yaml`: a 1,000,000-cell reader pool, then
  `PoissonBiasedSubsampleMesh(weight_mode=area)` to 200,000 expected cells with exact Horvitz-Thompson weights. Its content lines
  equal the stored-wing twin's diff. The deduplicating reader is unchanged and reads its masks from `$T/hldup/masks` (read-only).
  Every log says `fingerprint checked`. Per case, the reader draws the cyclic 1,000,000-cell block, drops about 66,000 extra copies
  (about 934,000 kept, with the measure factor N / 1,000,000), and the area draw then takes about 200,000 cells.
- **Job.** LHR 375068 (array 0, nvl72d022-T18). It started at 21:02 EDT and took 16 min 22 s: **1.09 GPU-h** (cap 3).
- **Cost.** Child HL-AREA-COST (`COST_REPORT.md`, `cost/`) ran the book's own `extract_cost.py` (copied unchanged, md5 9b87878e) on
  the six lanes' training logs, and on the six stored-wing lanes as a check: it reproduces cost_090.json's per-seed values exactly. It
  then took the GPU-h of every chain link from sacct (`cost/sacct_links_D.psv`).
- **Reduction.** `area_reduce.py` (7e6c53ec) → `area_table.md` (784802a1) and `area_summary.json` (feb1aff2). It reads the area
  scores here (`scores_area/`, rsynced from `$T/evals_hlarea/`) and the native and posed scores from HL-DUP's `scores_stage2/`. Score
  md5s: ISLA 8ef1974e / 3e7bf1a2, GT 612ad290 / d6dfaa8b, T 52a6bb4b / 2eb33a9b (s42 / s43).

## Checks before reading

- All six area rows: the `config` row says `"epoch": 500` and dataset `highlift_surface_medium_dedup_area`, with 180 `infer_step` rows.
  These are **the same 180 sample ids as each run's native scoring** (asserted).
- **Not degenerate.** The area draw changes the scored cells and their weights for every arm. Per-case pressure differs from native
  on 180 of 180 cases for each of the six runs (0 identical). HL-DUP's RESULTS_STAGE2 ("Not done here") calls the area protocol
  "degenerate for ISLA, as before". That does not hold on HiLift: the CHAR-PROTOCOLS audit (2026-10-16) found that the point cloud
  redraws HiLift's scored set, and the counts here confirm it.
- Independent numbers: 2 seeds per architecture. The area rows score the same six models as the native and posed rows, so the three
  protocols are not independent trainings.

## Area-protocol results (seed mean, seeds 42 / 43 in parentheses; native from HL-DUP stage 2)

| arm | native pressure | **area pressure** | native wall shear | **area wall shear** | area / native, p (s42, s43) |
|---|---|---|---|---|---|
| ISLA 0.9.0 | 0.05458 | **0.05426** (0.05245, 0.05606) | 0.09633 | **0.09612** (0.09417, 0.09807) | 0.989, 0.999 |
| GeoTransolver, unit drive | 0.05109 | **0.26088** (0.23886, 0.28289) | 0.09214 | **0.35275** (0.33094, 0.37455) | 4.72, 5.48 |
| Transolver, unit drive | 0.05274 | **0.18938** (0.19495, 0.18381) | 0.09189 | **0.29541** (0.31049, 0.28032) | 3.60, 3.58 |

The other fields under area are velocity / density / temperature: ISLA 0.0709 / 0.0537 / 0.0634; GT 0.239 / 0.243 / 0.232; T 0.201 /
0.176 / 0.188. `area_table.md` has every field, protocol and seed.

**Baseline ÷ ISLA** (seed-mean ratio; per-seed ratios s42, s43):

| baseline | protocol | pressure | wall shear |
|---|---|---|---|
| GeoTransolver | native | 0.936 (0.953, 0.920) | 0.957 (0.974, 0.939) |
| GeoTransolver | **area** | **4.81** (4.55, 5.05) | **3.67** (3.51, 3.82) |
| GeoTransolver | posed | 18.5 (19.2, 17.8) | 13.0 (13.5, 12.6) |
| Transolver | native | 0.966 (1.022, 0.914) | 0.954 (0.988, 0.921) |
| Transolver | **area** | **3.49** (3.72, 3.28) | **3.07** (3.30, 2.86) |
| Transolver | posed | 18.0 (18.9, 17.3) | 13.2 (13.9, 12.6) |

- ISLA's area error is 0.1-1.1 % *below* its native error on each field and seed (area / native 0.989-0.999). Its predictions do
  not depend on how the surface was sampled.
- Both baselines' errors rise 2.8-5.5 times (pressure 3.6-5.5) on every field when the cells are drawn by area rather than uniformly over the mesher's
  cells. GeoTransolver loses more than Transolver.
- Beside the stored-wing standing (`results/char/hilift_surface_090.json`, area / native on pressure: ISLA 1.002, GT 4.31, T 3.47),
  the picture is the same. GeoTransolver's loss is somewhat larger on the deduplicated wing (5.1 against 4.3 in the seed mean).
- How sure: firm on direction and size. The two seeds agree within about 15 % on every baseline ratio, and every baseline ratio is
  at least 2.86.

## Cost (the cost_090.json rule; seed means, per seed in `COST_REPORT.md`)

| architecture | parameters | step [s], dedup (stored) | peak [GiB/GPU], dedup (stored) | GPU-h per lane, dedup (stored) | GPU-h, 2 seeds |
|---|---|---|---|---|---|
| ISLA 0.9.0 | 68.8 M | **0.323** (0.341), 0.946× | **64.6** (65.6) | **32.8** (32.5) | 65.5 |
| GeoTransolver 256 | 9.05 M | **0.431** (0.457), 0.942× | **77.9** (77.9) | **38.3** (39.2) | 76.6 |
| Transolver 512 × 16 | 47.2 M | **0.323** (0.342), 0.944× | **62.3** (62.3) | **32.3** (31.9) | 64.6 |

- Step: the median `train_step` with `step_time_s < 10 s`. Peak: the maximum over epochs of `train_summary` `peak_memory_gib`.
  Parameters come from the config row and are unchanged by deduplication. Every lane on both sides ran on GB300, 4 GPUs and 4 ranks,
  bfloat16, one lane per node. ISLA ran on code_h090 with its blocks compiled (before the compile fix, as the stored-wing lanes did).
  The baselines ran uncompiled on code_a2r, as released. GPU-h is the sum over a lane's chain links of ElapsedRaw × 4 GPUs, failed
  links included (dedup ISLA s43: 0.33 GPU-h).
- The step is 5-6 % shorter on every architecture because each draw keeps about 7 % fewer cells: about 186,000 of 200,000 for ISLA
  and GT, and about 93,000 of 100,000 for T. This is a cost of the data, not of the method. GPU-h per lane moves only 0.98-1.01×,
  since loading and validation fill the rest of each epoch on a shared cluster. Read differences of ±3 % in GPU-h as noise.
- Total for the six dedup lanes: 206.8 GPU-h (the six stored-wing lanes: 207.3). Not counted there: smokes 1.49 GPU-h; stage-2
  scoring 2.79 GPU-h; this lane's area scoring 1.09 GPU-h.

## What it means for the program

- HiLiftAeroML gets its third protocol back on the deduplicated wing. On the uniform point cloud ISLA leads both baselines by 3.1-4.8
  times on pressure and wall shear, while on the native mesh the baselines lead by 3-6 %. The book's surface-task statement
  ("ISLA is unchanged by the point cloud, the baselines are not") now holds on all three surface tasks again.
- The cost rows can move to the deduplicated lanes, and the stored-wing caveat in 08 and 19 can go. With the copies dropped, every
  step is 5-6 % cheaper for all three architectures alike, peak memory is unchanged within 1.5 %, and so the cross-architecture
  ratios of the cost table do not move.
