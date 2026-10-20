# E3b ANCHOR pilot on AirfRANS: results (readout 2026-10-19 20:15 EDT)

**Verdict (registered rule): DOES NOT REPLACE.** C passes the validity check. On pressure and on velocity, C/A is
≥ 1.10 for both seeds (p 1.640 / 1.201, v 1.753 / 1.226; seed means 1.417 / 1.481, geometric mean 1.449). The proceed
rule fails (it needs a geometric mean ≤ 1.05 and nut ≤ 1.10; nut is 1.104). C does not go on to the AeroZoo stage.

All ten lanes finished 500 epochs (arm E included, both seeds). Full tables: `readout/tables.md`. Machine-readable
results: `readout/verdict.json`, `readout/scores.json`, `readout/qpop_f32_reduced.{json,md}`,
`readout/qpop_f64_reduced.{json,md}`, `readout/cost_*.json`. Scripts: `readout/{scores,cost,verdict}.py`.

## AF-UNIFORM native (float32, full_test, 200 cases, 100,000 points; seed mean, then seed 42 / 43)

| arm | p | v | nut |
|---|---|---|---|
| A isla | 0.01835 (0.01808 / 0.01861) | 0.00971 (0.00940 / 0.01002) | 0.02257 (0.02112 / 0.02402) |
| B passive | 0.01569 (0.01685 / 0.01452) | 0.01025 (0.01031 / 0.01018) | 0.02417 (0.02372 / 0.02461) |
| C anchor | 0.02601 (0.02967 / 0.02234) | 0.01438 (0.01648 / 0.01228) | 0.02493 (0.02873 / 0.02113) |
| D dtt | 0.01361 (0.01356 / 0.01367) | 0.00861 (0.00865 / 0.00857) | 0.02231 (0.02137 / 0.02324) |
| E mesh | 0.02068 (0.02146 / 0.01990) | 0.01054 (0.01052 / 0.01056) | 0.02363 (0.02388 / 0.02337) |

Ratios of seed means, native (per-seed ratios paired by seed in brackets):

| ratio | p | v | nut | geomean(p, v) |
|---|---|---|---|---|
| C/A | 1.417 (1.640 / 1.201) | 1.481 (1.753 / 1.226) | 1.104 (1.360 / 0.879) | 1.449 |
| B/A | 0.855 (0.932 / 0.780) | 1.055 (1.097 / 1.016) | 1.071 (1.123 / 1.024) | 0.950 |
| D/A | 0.742 (0.750 / 0.734) | 0.887 (0.920 / 0.856) | 0.988 (1.012 / 0.968) | 0.811 |
| E/A | 1.127 (1.186 / 1.069) | 1.086 (1.119 / 1.054) | 1.047 (1.131 / 0.973) | 1.106 |
| C/D | 1.910 | 1.670 | 1.117 | 1.786 |
| C/B | 1.658 | 1.404 | 1.031 | 1.525 |
| C/E | 1.258 | 1.364 | 1.055 | 1.310 |

Posed: the seed means are the same as native to within 0.3 % for A, B, D and E. C's posed C/A is 1.441 / 1.493 / 1.172.
C does not keep posed equal to native: seed 43's posed/native is 1.026 (p), 1.020 (v) and 1.185 (nut); seed 42's is
1.008 / 0.999 / 0.972. Every other run is within 0.23 %. Sanity check: A reproduces the earlier a2s lanes
(`evals_afu/afu_a2s_isla_*`, seed means 0.0180 / 0.00946 / 0.0223) to within 2-3 %.

## QUERY-POP (`$T/kernel/anchor_af/qpop/out`, float32, 200 cases; `out_f64`, float64 with eager blocks, first 20 cases)

| arm (both seeds) | f32 max abs(ratio - 1) | f32 max change / RMS | f64 max change / RMS | validity (f32 ratio ≤ 1e-4 and f64 change ≤ 1e-9) |
|---|---|---|---|---|
| A isla | 49-62 (far) | 15-23 | 14-23 | not query-independent, as expected |
| B passive | ≤ 2.9e-6 | ≤ 3.7e-3 | ≤ 2.2e-13 | pass |
| C anchor | ≤ 3.4e-5 | ≤ 2.1e-2 | ≤ 6.9e-12 | **pass** |
| D dtt | ≤ 2.6e-6 | ≤ 1.4e-3 | **1.6e-6 / 2.0e-6** | **fails the 1e-9 line** (ratios 1 to 9e-8) |
| E mesh | ≤ 3.9e-6 | ≤ 3.9e-3 | ≤ 4.9e-12 | pass |

- A's error goes up 26-63× when far-field companion points are added, 7-18× with wall-heavy companions, 1.25-1.6× with
  4 chunks and 1.08-1.37× with half the points (float32 and float64 agree on this).
- C's float32 changes, up to 2 % of the RMS target (seed 43), are the largest of the independent arms. The same seed's
  `repeat` floor (the identical forward pass run again) is 4e-4. In float64 the changes are 1e-12, so they are roundoff,
  but C is the arm most sensitive to roundoff.
- D's float64 change of about 2e-6 is not at float64 roundoff (B, C and E are at 1e-12). It may be a float32 operation
  that stays float32 inside the boundary read when the probe runs in float64, or a real dependence of 2e-6 on the
  requested points. This readout cannot tell which. The rule gates only C's accuracy, so the verdict does not change.
- Checking the instrument: `full` against infer.py was within 1e-4 on 7 of 10 runs. Three runs were just over:
  anchor42 1.7e-4, mesh43 1.2e-4 and passive42 1.0e-4. That is consistent with the float32 run-to-run floor
  (`repeat` up to 4e-4). The float64 predictions differ from the scored float32 ones by up to 30 % on single cases.
  The median difference is ≤ 2 % for A, B, D and E and 6-8 % for C. The float64 probe therefore tests query
  independence only; it is not an accuracy check. On its 20 cases C is no better in float64 than in float32
  (seed 42 p 0.0307 against A's 0.0206), so C's shortfall is not a float32 artifact.

## Cost (train.log, rank 0; GB300)

| arm | median step, epoch 0, steps 10-200 | × A | median step, epoch 250 | × A | peak Mem |
|---|---|---|---|---|---|
| A | 0.455 s | 1.00 | 0.461 s | 1.00 | 38.0 GB |
| B | 0.456 s | 1.00 | 0.462 s | 1.00 | 38.1 GB |
| C | 0.661 s | 1.45 | 0.553 s | 1.20 | 55.5 GB |
| D | 0.640 s | 1.41 | 0.564 s | 1.22 | 42.0 GB |
| E | 0.659 s | 1.45 | 0.556 s | 1.21 | 55.7 GB |

Training took about 650 GPU-h, estimated from the step times; I did not sum sacct over the chains. Scoring used
10 jobs and 3.4 GPU-h; the probe used 6 jobs and 11.1 GPU-h.

## Predictions against outcomes

| # | prediction (p) | outcome |
|---|---|---|
| P1 | C, B, D, E valid; A's ratios ≠ 1 (0.97) | **false**: D fails the float64 line (2e-6); C, B and E pass; A is far from 1 |
| P2 | proceed (0.45) | false |
| P2b | does not replace (0.25) | **true** |
| P3 | C ≤ D on all fields (0.65) | false: C/D 1.91 / 1.67 / 1.12 |
| P4 | B worst on v, B/A ≥ 1.10 (0.70) | false: C is the worst on v; B/A v 1.055; B beats A on p (0.855) |
| P5 | E/A ≤ 1.05 on p and v (0.55) | false: 1.127 / 1.086 |
| P6 | C closes ≥ 50 % of the B−A gap on every field (0.55) | false: C is worse than both A and B on p and v |
| P7 | diagnostic: law vs role | E ≉ A, so the condition does not hold. C/E 1.26 / 1.36: the Lambert law costs about 30 % on top of the writers' role, but E is itself behind B on p (E/B 1.32) |
| P8 | C, E ≤ 1.6× A; B ≤ 1.05× A (0.80) | true: 1.45× at epoch 0 and 1.20× mid-run; B 1.00× |
| P9 | posed/native within 0.5 % for every arm (0.95) | false for C only (up to +18.5 % on nut, seed 43) |

## What it means

- Boundary-drawn anchors as interior writers lose to interacting ISLA by about 45 % on p and v. Both seeds agree,
  although C's own seeds differ by 33 %. They also lose to the two cheaper query-independent designs: passive (B, no
  interior writers) and ISLA-DTT (D). Interior writers of any law hurt pressure on AirfRANS here: E/B is 1.32 on p,
  C/B 1.66.
- **A surprise beyond the question asked:** ISLA-DTT (D) beats interacting ISLA by 26 % on p and 11 % on v (nut
  equal), with both seeds agreeing. The PREREG assumed the opposite (the pre-A2 DTT trailed ISLA by 11 / 26 / 21 %).
  The passive arm also beats A on pressure. On this benchmark, the query-population coupling of interacting ISLA seems
  to cost accuracy, not add it.
- C's accuracy depends on its anchor realization. The posed protocol changes C's prediction by up to 18 % (nut, seed 43)
  and no other arm's by more than 0.23 %. The likely cause is that a pose changes which anchors the fixed seed-0 eval
  stream draws (the random directions s are drawn in the global frame). C's float32 sensitivity is a possible second
  cause. Either way, C is not pose-exact as built.

## Operating notes

- Jobs (LHR, QOS normal, none in short): scoring 290707-290715 and 290847 (all 3/3 protocols `.done`, into
  `$T/evals_anc`); probe float32 290727, 290728, 290848 (`qpop/out`); probe float64 290729, 290730, 290849
  (`qpop/out_f64`, `QPOP_CASES=20`). The probe was split so that each run had its own GPU, instead of one job of
  10 runs on 4 GPUs. The float64 jobs took 31-49 min each.
- Removed `$T/kernel/anchor_af/smoke_runs` (2.6 GB, five `smk_*` dirs), as JOBS.md says. `smoke_evals` is kept.
- No anc chain links are left in the queue. No repo edits, no pushes.
- Open items for Peter: (1) adoption of the verdict, which stops the ANCHOR line before AeroZoo; (2) whether to find
  the source of D's 2e-6 float64 change before D is used as the query-independent reference; (3) the D > A result,
  which changes the baseline that later ISLA work compares against.
