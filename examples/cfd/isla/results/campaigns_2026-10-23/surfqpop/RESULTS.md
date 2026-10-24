# SURF-QPOP results: request dependence of ISLA 0.9.0 on DrivAerML Surface

Written 2026-10-23 23:46 EDT. Pre-registration: PREREG.md (md5 aadf3179, sent 23:11 EDT, no amendment). The
supervisor's steer of 23:33 EDT: Peter ruled at 23:26 on 2026-10-23 that **option (A) is the surface task** for now
(the requested cells of the body are also the model's view of the body), and ISLA-DTT is revisited once its speed and
memory work lands. So this probe no longer decides between (A) and (B). It measures the request dependence of the
adopted (A), for the book and for that revisit.

Checkpoints: r090_dr_isla_seed42/43 (epoch 500), code_r090 / recipe_r090, drivaer_ml_surface val, **all 48 cars, both
seeds**, float32, eager. Job 362779 (LHR, 17 min, 1.13 GPU-h); smokes 362718 (failed at Q3, harness bug) and 362751.
Total 1.47 GPU-h. Data: `results/out/reduced.json` (md5 8ff273b3), `results/out/tables.md` (42ab3103), per-case rows
`results/out/<run>/shard_*/cases.jsonl`; reducer `lhr/reduce.py` (ee11b119); harness `lhr/surfqpop.py` (1e9b746d).
Errors are the recipe's area-weighted (Horvitz-Thompson) relative L2, in the metric's space, means over the 48 cars.
The spread s and the floor d use the same weighted norm (checked: 1 +- 1e-7 against the metric on every case).

## Validity

- V1: the native scored sample, rebuilt by the request builder from the reader's own block, is identical
  (torch.equal on every tensor) on 96/96 car-seed pairs, and gives the identical prediction (max change 0). So every
  request below is built exactly as the scored sample is.
- V2: eager native metrics against the scorer's compiled record (job 301151): max 3.9e-6 relative, median 2.6e-7.
  Native means 0.04116 / 0.06619 (seed 42) and 0.04111 / 0.06625 (seed 43), as in RESULTS_partial.md.
- The targets of a cell agree exactly between any two requests that share it (max difference 0).
- Non-degeneracy: every request has its registered size; pairs share about N^2 / M cells (120-200 at 50k, about
  2,300-2,800 at 200k, 37,000-44,000 at 800k; no pair below the 20-cell minimum); the Q3 halves hold 100,000 cells each.

## Q1: request size (uniform draws; K = 6, 6, 4, 4, 4; 2 block draws per N)

s = relative L2, over the shared cells, of the difference between two independent requests' predictions (mean over
pairs, then cars); delta = s / sqrt(2) is one request's noise. e = error on the request's own cells.

| field, seed | N | s | delta | delta / e(200k) | s x sqrt(N / 200k) | e uniform | e block | e / e(200k) [95%] |
|---|---|---|---|---|---|---|---|---|
| p, 42 | 50k | 0.02484 | 0.01757 | 0.424 | 0.01242 | 0.04449 | 0.04415 | 1.0751 [1.0646, 1.0863] |
| | 100k | 0.01738 | 0.01229 | 0.297 | 0.01229 | 0.04229 | 0.04232 | 1.0218 [1.0153, 1.0279] |
| | 200k | 0.01233 | 0.00872 | 0.211 | 0.01233 | 0.04138 | 0.04122 | 1 |
| | 400k | 0.00857 | 0.00606 | 0.146 | 0.01212 | 0.04080 | 0.04081 | 0.9860 [0.9826, 0.9895] |
| | 800k | 0.00606 | 0.00429 | 0.104 | 0.01212 | 0.04058 | 0.04067 | 0.9807 [0.9771, 0.9842] |
| p, 43 | 50k | 0.02444 | 0.01728 | 0.417 | 0.01222 | 0.04450 | 0.04442 | 1.0742 [1.0632, 1.0857] |
| | 100k | 0.01727 | 0.01221 | 0.295 | 0.01221 | 0.04241 | 0.04248 | 1.0238 [1.0191, 1.0288] |
| | 200k | 0.01214 | 0.00858 | 0.207 | 0.01214 | 0.04142 | 0.04142 | 1 |
| | 400k | 0.00861 | 0.00609 | 0.147 | 0.01218 | 0.04095 | 0.04096 | 0.9886 [0.9853, 0.9917] |
| | 800k | 0.00598 | 0.00422 | 0.102 | 0.01195 | 0.04071 | 0.04078 | 0.9827 [0.9791, 0.9864] |
| wss, 42 | 50k | 0.03596 | 0.02543 | 0.382 | 0.01798 | 0.07094 | 0.07008 | 1.0657 [1.0568, 1.0743] |
| | 100k | 0.02508 | 0.01773 | 0.266 | 0.01773 | 0.06785 | 0.06783 | 1.0193 [1.0149, 1.0236] |
| | 200k | 0.01799 | 0.01272 | 0.191 | 0.01799 | 0.06656 | 0.06635 | 1 |
| | 400k | 0.01271 | 0.00899 | 0.135 | 0.01797 | 0.06581 | 0.06571 | 0.9887 [0.9860, 0.9912] |
| | 800k | 0.00932 | 0.00659 | 0.099 | 0.01864 | 0.06557 | 0.06553 | 0.9851 [0.9822, 0.9880] |
| wss, 43 | 50k | 0.03545 | 0.02506 | 0.376 | 0.01772 | 0.07099 | 0.07040 | 1.0639 [1.0539, 1.0735] |
| | 100k | 0.02469 | 0.01746 | 0.262 | 0.01746 | 0.06813 | 0.06812 | 1.0210 [1.0178, 1.0242] |
| | 200k | 0.01798 | 0.01271 | 0.190 | 0.01798 | 0.06673 | 0.06667 | 1 |
| | 400k | 0.01293 | 0.00914 | 0.137 | 0.01829 | 0.06607 | 0.06594 | 0.9901 [0.9878, 0.9924] |
| | 800k | 0.00913 | 0.00645 | 0.097 | 0.01826 | 0.06579 | 0.06573 | 0.9860 [0.9832, 0.9889] |

**Convergence slope** (least squares of log s on log N over the five N; 95% interval from 2,000 car-bootstrap
resamples):

| | pressure | wall shear |
|---|---|---|
| seed 42 | -0.509 [-0.517, -0.501] | -0.488 [-0.500, -0.475] |
| seed 43 | -0.507 [-0.514, -0.501] | -0.485 [-0.495, -0.474] |
| s(800k) / s(200k) (pure law: 0.500) | 0.492, 0.492 | 0.518, 0.508 |

The spread falls as 1/sqrt(N): s x sqrt(N) is constant to within 4% over the 16-fold range of N, on both fields and
seeds (unweighted norm: slopes -0.514, -0.510, -0.501, -0.509). The error follows e(N)^2 ~ e_inf^2 + k delta(N)^2 with
k = 1.15 (pressure) and 1.21-1.22 (wall shear), inside P4's factor 2. Extrapolated, e_inf = 0.0403 / 0.0404 (pressure)
and 0.0652 / 0.0654 (wall shear), so the scored 200k request carries 2.0-2.6% of error from request noise. The error
plateaus at 400k (within 1% of 800k), not at 200k. The block law equals the uniform law (0.4% or less at 200k).

## Q2: chunking (R = a 200k request, against its cells in 4 disjoint covering 50k requests)

| | e(R) | e(4 chunks) | ratio [95%] | median per car | e(50k) / e(200k), from Q1 |
|---|---|---|---|---|---|
| p, 42 | 0.04136 | 0.04446 | 1.0749 [1.0669, 1.0833] | 1.0809 | 1.0751 |
| p, 43 | 0.04152 | 0.04457 | 1.0735 [1.0661, 1.0816] | 1.0745 | 1.0742 |
| wss, 42 | 0.06653 | 0.07074 | 1.0632 [1.0568, 1.0700] | 1.0667 | 1.0657 |
| wss, 43 | 0.06678 | 0.07098 | 1.0628 [1.0568, 1.0695] | 1.0639 | 1.0639 |

Splitting one request in four costs exactly what a four-times-smaller request costs (to within 0.003), so the coupling
runs through the quadrature's sample size alone. For scale: on AirfRANS, interacting ISLA's interior error rises 1.31 /
1.36 / 1.40 times under the same split (PROBES-090 (f)); the surface task's 1.06-1.07 is a fifth of that.

## Q3: a request that does not cover the body (R's cells split at R's median x, each half at R's density)

| | half alone | the same cells in R | ratio [95%] | median per car |
|---|---|---|---|---|
| p, 42, low x | 0.6693 | 0.0365 | 18.4 [17.4, 19.4] | 19.1 |
| p, 42, high x | 0.8664 | 0.0503 | 17.2 [15.6, 19.0] | 19.6 |
| p, 43, low x | 0.6399 | 0.0367 | 17.4 [16.5, 18.4] | 18.4 |
| p, 43, high x | 0.7697 | 0.0503 | 15.3 [13.9, 16.8] | 17.4 |
| wss, 42, low x | 0.6579 | 0.0635 | 10.4 [9.8, 10.9] | 10.9 |
| wss, 42, high x | 0.7699 | 0.0694 | 11.1 [10.2, 12.1] | 12.7 |
| wss, 43, low x | 0.6658 | 0.0642 | 10.4 [9.9, 10.9] | 10.8 |
| wss, 43, high x | 0.7186 | 0.0690 | 10.4 [9.5, 11.4] | 12.0 |

Given half a car, the model has no useful skill on that half: errors of 0.64-0.87, ten to eighteen times the covering
request's on the same cells. Under (A) the request is the model's whole view of the body, and the measures say half a
body, so a prediction is defined only for a request that covers the body.

## Q4: the floor

R forwarded twice, and the native sample forwarded twice: bit-identical on every car and seed (d = 0, max change 0).
Eager float32 on one GB300 is deterministic here, so every spread above is request dependence, not run-to-run noise.

## Registered predictions

| | prediction (p) | outcome |
|---|---|---|
| P1 | slope in [-0.6, -0.4] on every field and seed (0.70); interval excludes 0 (0.95) | held; held |
| P2 | delta(200k) < 0.25 e(200k) on both fields (0.70) | held (0.190-0.211) |
| P3 | e(50k) > 1.02 e(200k) (0.70); e(800k) within 2% of e(200k) (0.75); plateau at N <= 200k (0.55) | held (1.064-1.075); held, narrowly on pressure seed 42 (1.93%); failed (400k) |
| P4 | excess e^2 within 2x of excess delta^2 (0.55) | held (1.15-1.22) |
| P5 | Q2 ratio in (1.02, 1.40) (0.65); below 1.31 (0.70) | held (1.063-1.075); held |
| P6 | Q3 ratio > 1.5 (0.75); > 3 (0.35) | held (10.4-18.4); held |
| P7 | Q4 d < 1e-5 on every case (0.75) | held (exactly 0) |
| P8 | block = uniform at 200k within 1% (0.85) | held (at most 0.4%) |
| V2 | eager within 1e-3 of the scorer on every case (0.55) | held (3.9e-6) |

## The registered reading (PREREG section 3), as written

- "Sound quadrature" needs the slope's interval to contain -0.5 and P2 to hold. P2 holds. The interval contains -0.5
  only on wall shear, seed 42 ([-0.500, -0.475]). On pressure it misses -0.5 on the steep side by 0.0006 and 0.0005,
  and on wall shear, seed 43, on the shallow side by 0.005.
- "Request-coupled beyond quadrature" needs the interval above -0.4 (no field or seed), or s(800k) not below half of
  s(200k): wall shear 0.518 and 0.508 on both seeds, pressure 0.492 and 0.492. So on wall shear the registered label is
  "request-coupled beyond quadrature" (seed 42 satisfies both labels). Pressure takes neither label: its spread falls
  slightly *faster* than 1/sqrt(N), a case the rule did not foresee.
- Since Peter's ruling, neither label decides anything about (A).

**Beside the rule (not registered).** The second clause tests the pure law at its own value: an exact 1/sqrt(N) law
gives s(800k) / s(200k) = 0.500, so it fires on about half of all draws from a sound quadrature. And bootstrap intervals
of width about 0.015 resolve departures from -0.5 far smaller than any that would matter. What the data show is the
law to within 3% in the exponent and 4% in s x sqrt(N), over 16 times in N, on both fields and seeds. The wall-shear
exponent (-0.487 on average) is slightly shallower than -0.5 and the pressure exponent (-0.508) slightly steeper. The
shallow wall-shear value comes mostly from 800k on seed 42 (s x sqrt(N) rises 3.7% there, flat on seed 43), so this
probe does not resolve it. The reading I would give the book: the request dependence of (A) is a Monte Carlo
quadrature error that falls as 1/sqrt(N).

**Beside the rule: the scored draw reads slightly low.** On the same cars, the native scored request's error is 0.4%
(pressure) and 0.5% (wall shear) below the mean of six fresh 200k requests on seed 42, and 0.8% and 0.7% on seed 43
(mean z = -0.17 / -0.32 / -0.41 / -0.53 against standard errors 0.15-0.20). Both seeds score the same native cells, so
these are not two independent tests. One car's 200k score varies by 1.6-1.8% (SD) between requests. Whether the
scored draw is merely a lucky draw, or a mild program-level selection on that draw (every ISLA choice was read on these
exact 48 x 200k cells), this probe cannot tell. It is below the 1-16% two-seed gaps that rulings use.

## What it means for option (A), adopted by Peter's ruling of 2026-10-23

- A cell's prediction depends on its co-requested cells only through the quadrature of the slice states. The spread
  between two covering requests falls as 1/sqrt(N), chunking costs exactly a smaller request, and the floor is zero.
  This is the behaviour the brief asked of a sound construction; I am confident of it (both seeds, 48 cars, tight
  intervals).
- At the scored 200k, one request's noise is about 0.2 times the error, and it adds 2.0-2.6% to the error over an
  infinite request. Going to 400k recovers 1.0-1.4% of the error, and 800k 1.4-1.9%. For inference this is a knob: more cells,
  or the mean of several requests, lowers the error at the cost of a linear forward (0.48 s at 200k, 1.98 s at 800k,
  7.4 GB and 28.7 GB peak).
- A request must cover the body: a regional request fails (10-18 times the error). That is a usage contract of (A).
  A user who wants one region asks for a covering sample and reads the region from it. An ISLA-DTT-style surface
  readout, whose prediction cells would be separate from the boundary tokens, would remove the contract. It is the
  point to measure when ISLA-DTT is revisited.
- Unlike the interior task, where AirfRANS's dependence has "no knob that converges" (book, limits chapter), the
  surface task's dependence converges with the request size.
