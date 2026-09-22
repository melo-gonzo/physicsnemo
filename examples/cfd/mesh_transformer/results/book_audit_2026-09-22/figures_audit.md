# Figures, tables and data-provenance audit (fresh-context reviewer, read-only, 2026-09-22)

Scope: STYLE.md, index, 01-09, 12, 13, 17; every fig-*/tbl-* cell (55 labels), the JSON each opens, the rendered
PNGs. Yardstick = the configuration the book itself states as current on 2026-09-22 (ISLA 192x12x256, cosine,
rms_distance, 200k cells, 1e-3, two seeds, float32; GeoTransolver/Transolver 256 at 3e-3, unit direction). Note: the REF
campaign (#sec-nb-ref-prereg) will move the ISLA reference to 384 wide, so "cos-192" below reads "the reference
configuration".

**Headline finding.** No accuracy figure or table is drawn from lanes matching the current configuration on every
axis. Every 200k surface lane (trio, ladder, fleet, pose, wing training law) is linear-routed with the total-measure
unit; every cosine lane is at 100k (cap/routenorm/scalelaw) or on the interior/AirfRANS task; the two cosine 200k
lanes that exist (routenorm_p200k_2026-09-22.json: cosine 192/384 on the 35-wing rung; cosine 384x24 on the cars)
are read by no chapter cell.

## 1. Ledger (configuration, gaps, routing) — condensed
- index tbl-thesis / fig-headline: rows 1,2,4,5,6,7 linear (200k/100k mix); row 3 cosine at 100k; no seed dots on
  fig-headline; budgets mixed on one axis. fig-key omits styles used elsewhere (cosine vs linear, fused vs fallback,
  area-law marker, width shades).
- 01 fig-metric-calibration: constants only, bf16, p-p_inf scale; the artifact's trained-model value not drawn.
- 02 concept figures fine; fig-isla-slices uses architecture hues for slices.
- 03 tbl-scoreboard-200k (linear; Transolver has no 1e-3 control anywhere); tbl-scoreboard-200k-wings (linear; 510/1,260
  absent; GeoTransolver rate undocumented on 210 and 21-geo rungs, only a single gt_unit arm in the artifact; 35-rung
  cosine 0.1408/0.1361 exist unread); tbl-scoreboard-200k-interior (linear row beside cosine; no source_total arm);
  fig-scoreboard-density + tbl (100k only; Transolver seeds 3.4x/7.3x behind a 5.3x bar; wingbench_200k exists
  unread); tbl-scoreboard-training-law (cars at 100k); tbl-scoreboard-cells (GT/T 600k not run; nothing below 100k);
  fig-cost-sweep: step/memory panels end at 200k while x runs to 600k (isocost has peak_mem_gb at 400k/600k: GT-256 158
  GB, ISLA-192 54/80 GB, T 106 GB, unused); accuracy panel empty below 100k though 10k/40k/80k standings exist; the
  "default" fused arm is a path no lane trained on; GT 49 GiB (sweep) vs 80 GB (training log) unreconciled.
- 04 tbl-architectures (Transolver 200k memory missing); tbl-unit-vs-physical (10k only); tbl-cells-ladder (as above);
  fig-scaling-cases/tbl-scaling-* (cosine, 100k; lr field null in artifact; hilift half pending; shades not in key);
  fig-scaling-airfrans (eddy viscosity omitted; failed GT fit drawn like an accepted one); tbl-pose (GT only, 10k, bf16).
- 05 fig-hilift-data-efficiency/tbl-hilift-ladder: linear; 510/1,260 absent; rate-label bug (gt_best prints "3e-3"
  whenever only one GT arm exists); bar labels collide with seed dots; five unshared y-axes.
- 06 tbl-density (100k); fig-consistency-curves/tbl-consistency-40k (10k-trained checkpoints; DrivAerML GT at 1e-3;
  ISLA curves start at 10k; nothing above 40k); fig-training-law (two budgets; categorical dodge); territory figures
  (100k linear; wings twin unread); tbl-consistency-matrix (two budgets); tbl-floor (cosine rows without memory;
  cosine 384x24 at 200k unread; caption still "default candidate"); fig-equal-budget (all linear; 30+ overlapping
  labels; cosine excluded for lack of cost logs); fig-capacity (100k only; hand-set floor band; cosine/linear only in
  legend text); tbl-hilift-ood (removed variant + physical-velocity GT, 10k).
- 07 tbl-isla-params instantiates the class default (now cosine) while every lane is linear: check the total.
- 08 tbl-interior-200k (no source_total; no cosine rate control; no 54/109 rungs); tbl-interior-airfrans (seed spread
  22-34%).
- 09 tbl-using-rate-ceiling (linear ceiling; hard-coded cells); tbl-isla-cost (eager path; Transolver row missing).
- 13/17 static tables: hand-quoted numbers (4-geo 0.76 vs chapters' 0.68); cosine row stale.

## 2. Contradictions (12)
Parity prose vs fig-cost-sweep/fig-scaling-cases/fig-capacity where GeoTransolver sits below ISLA at every point
(0.3-1% gaps on a 0.040-0.050 axis); 4-geo rung 24% vs 32% vs 0.76 (three numbers); GeoTransolver rate on 210/21-geo
undocumented yet headed "3e-3"; cosine default vs candidate in ten places; 0.1408 quoted from memory (no cell reads
routenorm_p200k); interior lead led with the linear 1.42 while cosine 1.56 is the default configuration; ratevol
pressure bar "budget effect" reframed as noise in a caption; Transolver density 5.3x from 3.4x/7.3x; cost 27 vs 80 GB
and 49 GiB vs 80 GB from two instruments; 21-geo Transolver ahead of GeoTransolver hidden by ordering; width-192 rate
under cosine stated as the linear rule; tbl-thesis row 3 "same floor" while GT-2048 sits below every ISLA point.

## 3. Figure standards
Seeds missing on fig-headline and fig-equal-budget; fig-key incomplete; unshared y-axes and label/dot collisions on
fig-hilift-data-efficiency; hard-coded hex and shades on fig-scaling-cases; failed fits drawn as fits; y-ticks missing
above 1 on fig-consistency-curves; categorical dodge on fig-training-law; tab20 over 256 slices on territory maps;
hand-drawn floor band on fig-capacity; slice colors clash with architecture hues.

## 4. Re-run list (under the reference configuration once chosen; baselines at 3e-3 unless noted)
A. DrivAerML Surface, 200k: (1) reference at 435, (2) reference at 27/54/109/218 + Transolver 1e-3 control per rung,
(3) pose native + posed, (4) density/stratified/biased draws at 200k for all three, (5) training-law twin at 200k for
all three, (6) cells ladder under the reference at 10k/40k/80k/100k/400k/600k + GT/T at 600k, (7) cost sweep at
400k/600k tokens for every arm incl. cosine/fused, Transolver 200k memory, (8) consistency benchmark on the 200k
reference checkpoints at 2.5k-200k, (9) territory probe on the reference checkpoints, (10) GT canonical-frame pose
control at 200k float32.
B. 100k scaling budget: (11) cost logging for the cosine width/depth ladders, (12) cosine rate controls per width/depth,
(13) record lr in the scalelaw artifact, (14) 200k rung of the parameter-matched pair.
C. HiLiftAeroML 200k: (15) reference on 35/210/510/1,260/4-geo/21-geo/fixed-angle, (16) GT 3e-3 on 210 and 21-geo,
(17) training-law twin on the 35-rung, (18) wing scaling half, (19) unit-vs-physical at 200k, (20) regime-extrapolation
splits with the reference and unit-direction GT, (21) consistency + territory on the 200k wing checkpoints.
D. DrivAerML Volume: (22) reference with source_total, 1e-3, (23) 54/109-car interior rungs, (24) request-size probe.
E. AirfRANS: (25) reference under source_total; eddy-viscosity panel; extrapolation splits (running).

## 5. Artifact hygiene
No chapter reads a missing file. Unread current-protocol artifacts to wire in: routenorm_p200k_2026-09-22,
wingbench_200k_reduction_2026-09-19, territory_wings_2026-09-21, tokproto_reduction_2026-09-18 (40k/80k rungs), others
listed by the reviewer. 265 of 319 top-level JSONs are read by no chapter (pre-ISLA program, preregistration JSONs,
superseded dated duplicates, 10k studies): candidates for results/archive/. Chapter-read artifacts carrying removed or
non-protocol configurations (to retire once re-runs land): hilift_ladder_reduction_2026-09-03 + hilift_sealed_test
(legacy variant, physical-velocity GT), p1 canonicalization controls (bf16, 10k), the 10k rows' sources
(fp32_shift_35, relframe_reduction, udrv_reduction, frame_reduction, fp32_shift), consistency_bench/* (10k),
quadvar_reduction (10k, legacy), hilift_reference_predictors_35 (bf16).
