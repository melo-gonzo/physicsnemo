# SURF-QPOP pre-registration: does a DrivAerML Surface prediction depend on the other requested cells?

Lane SURF-QPOP of the ISLA program, written 2026-10-23 23:11 EDT, before any job. Inference only; nothing is trained.
Supervisor 01a13c2c. Context: scratch/isla-surface-bc-2026-10-23/DESIGN.md option (A), where the requested cells of the
body are also the model's view of the body (every requested cell is a boundary token; slice states are measure-weighted
means over the requested cells, Horvitz-Thompson measures).

No number of any arm below has been seen by this lane. Known before writing: the standing's scorer numbers (0.04113 /
0.06622, RESULTS_partial.md), PROBES-090 (f) on AirfRANS (chunks4 raises ISLA's interior error 1.31 / 1.36 / 1.40x), and
one data fact checked on LHR at 23:08 (run_1): the stored vehicle cells (17,684,913) are row-shuffled, so a contiguous
200k block has the same mean, spread and x-percentiles as a uniform draw (block median x 1.808-1.848 m against 1.814).

## 1. What is measured

Checkpoints: the standing r090_dr_isla_seed42 and r090_dr_isla_seed43 ($T/runs/<run>, epoch 500), on code_r090 /
recipe_r090, dataset drivaer_ml_surface, model isla_surface with the scorer's extra (model.hidden=512
model.n_slices=512 model.compile_blocks=true), infer_split val (48 cars), float32, **eager** (overrides compile=false
model.compile_blocks=false). All 48 validation cars, both seeds.

A request = a set of vehicle cells given to the model as `boundaries.vehicle`, built exactly as the reader builds the
scored sample: the case's full stored vehicle mesh (M ~ 17.7M cells), `slice_cells(idx)`, vertex compaction,
`scale_measures(M / n_density)` (the Horvitz-Thompson factor; n_density = the request size for a covering request),
then the dataset's transform chain in order with `SubsampleMesh` skipped (equivalent to sampling_resolution = N; at
N = 200k it is a no-op anyway). CenterMesh centres each request on its own mean, as the scorer does; ISLA is exactly
translation invariant.

Laws:
- **uniform** (primary): idx uniform without replacement from all M cells (the law of SubsampleMesh's exact draw).
- **block** (secondary, Q1(b) only): the reader's own law, a cyclic contiguous block of N stored rows with a uniform start.

Per case, one CPU generator seeded 2026 + dataset index draws, in this order: the uniform requests (N ascending, k
ascending), the block starts, the Q2 partition. Readouts are in the metric's space (training space, as infer.py's
metrics), per field: pressure (`pressure_l2`) and wall shear (`wss_l2`, vector).

### Validity (checked before reading Q1-Q4)
- V1, builder: the native sample's own reader block, replicated from the reader's per-index generator and rebuilt by the
  request builder, gives the identical DomainMesh (points, cells, measures, targets: torch.equal) and the identical
  forward output, on every case.
- V2, native vs scorer: eager metrics of the native sample against the scorer's record (compiled path, job 301151);
  reported per case (max and median relative difference). Not a gate: the scorer ran whole-model compile.
- Non-degeneracy: token counts, shared-cell counts per pair, region cell counts are written per case.

### Q1, request size
N in {50k, 100k, 200k, 400k, 800k}; K = 6 uniform requests at 50k and 100k (15 pairs), K = 4 at 200k, 400k, 800k (6
pairs); 2 block requests per N.
- (a) spread: for each pair of uniform requests at the same N, over their shared cells S (expected |S| ~ N^2 / M: about
  140 at 50k, 2,300 at 200k, 36,000 at 800k), s = ||pred_a - pred_b||_S / ||target||_S (vector norm for wss). Per case
  the mean over pairs (pairs with |S| < 20 dropped); per N the mean over cases. Request noise delta(N) = s / sqrt(2).
- (b) scored error at each N: the metric on each request's own cells, mean over its K requests, then over cases (the
  standing's case-mean). Both laws.
- Slope: least-squares slope of log s(N) against log N over the five N (case-mean s), with a 95% interval from 2,000
  case-bootstrap resamples. Per field and seed.
- Plateau: the smallest N whose case-mean error is within 1% of the 800k error.

### Q2, chunking
R = the first uniform 200k request. Its cells split by a random partition into 4 disjoint 50k chunks, each built as its
own covering request (n_density 50k, measure factor M / 50k). Readout: the metric of the reassembled chunk predictions on
R's cells over the metric of R's own predictions; ratio of case means, and the median per-case ratio.

### Q3, a request that does not cover the body
From R: the cells whose centroid x (stored frame) is below R's median x (the low-x half; DrivAerML's nose is at low x),
and separately the rest (the high-x half). Each half is its own request at R's density (measure factor M / 200k), ~100k
cells. Readout: the metric of the half request on its cells over the metric of R's predictions on the same cells; ratio of
case means and median per-case ratio.

### Q4, the floor
R forwarded twice, and the native sample forwarded twice: d = ||pred_1 - pred_2|| / ||target|| and max |pred_1 - pred_2|
over RMS target, per case; max and median over cases.

## 2. Predictions (probabilities)

- P1. Q1(a) slope within [-0.6, -0.4] for both fields and both seeds (quadrature-like noise): 0.70. Slope's 95% interval
  excludes 0 for every field and seed: 0.95.
- P2. delta(200k) below 0.25x the 200k scored error on both fields (request noise is a small part of the error): 0.70.
- P3. Q1(b): error at 50k above error at 200k by more than 2% on both fields: 0.70; error at 800k within 2% of error at
  200k on both fields: 0.75; plateau at N <= 200k: 0.55.
- P4. Consistency: e(50k)^2 - e(800k)^2 within a factor 2 of delta(50k)^2 - delta(800k)^2 (per field, pooled seeds):
  0.55.
- P5. Q2 ratio in (1.02, 1.40) for both fields and seeds: 0.65; below the AirfRANS interior's 1.31: 0.70.
- P6. Q3 ratio above 1.5 on both fields and both halves: 0.75; above 3: 0.35.
- P7. Q4: d below 1e-5 on every case: 0.75 (eager float32; scatter atomics may break bit equality).
- P8. Block law equals uniform law at 200k within 1% of error: 0.85.
- V2: native eager within 1e-3 relative of the scorer on every case: 0.55 (the compiled AirfRANS path differed by 4-88%).

## 3. What it means for option (A) (decided now)
- "Sound quadrature": P1's slope interval contains -0.5 and P2 holds. Then a cell's prediction depends on the other cells
  only through a Monte Carlo error that shrinks with the request, and option (A) is defensible as is.
- "Request-coupled beyond quadrature": slope interval excludes -0.5 and is above -0.4 (spread falls slower), or the
  spread at 800k is not below half the spread at 200k. Then predictions carry a request dependence that more cells do
  not remove; report it as a point against (A) and for declaring prediction cells apart from the boundary tokens.
- Q3's ratio is reported as the cost of a non-covering request; it is not a verdict on (A), which always gives the
  model the whole body.

## 4. Rules and cost
- LHR only, short QoS, 4-GPU jobs, one unit per GPU, names isla.surfqpop.*, the five bad nodes excluded, registry row
  ^isla\.surfqpop\. added at the first job and removed after the last. Cap 6 GPU-h allocated.
- Smoke (blind: writes only validity, counts, timings, memory): 2 cases per seed, ~15 min, ~1 GPU-h.
- Production: 2 seeds x 2 shards of 24 cars, one 4-GPU job; projected about 1 h, ~4 GPU-h allocated. If the smoke projects
  above the cap, K is cut to 4 at all N and then the 800k arm dropped, in that order, before submission.
- Code: new launcher $T/kernel/surfqpop/ (surfqpop.py, surfqpop.sbatch, rows, units), md5s recorded in JOBS.md;
  code_r090 / recipe_r090 snapshots unchanged. Local work CPU only.
