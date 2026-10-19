Status: DONE

# AF-PLANAR (DRIVE-FIX2 Part C): in-plane velocity error of the AirfRANS baselines, 2026-10-19

Started 12:05 EDT; all jobs finished by 12:29 EDT. No local GPU was used, nothing was committed, and no shared script,
table or eval tree was edited.

## Result

The out-of-plane component does not change the baselines' velocity error to the precision the book reports. The
in-plane velocity_xy_l2 equals the book's 3-D velocity_l2 to within 4e-6 relative on native (1.5e-4 on posed), so every
baseline/ISLA velocity ratio is the same under both metrics. RESULTS.md's rough estimate, that z could account for
about a quarter of the baselines' velocity error, is wrong. The z component carries at most 1.2e-5 of the squared error
on native.

## How velocity_l2 is computed (recipe_v8 src/infer.py + src/metrics.py, code_v20)

- Units: training space, U / |U_inf| with |U_inf| = sqrt(sum U_inf^2) per case (NonDimensionalizeByMetadata; velocity
  has no NormalizeMeshFields entry).
- Points: the 100,000 interior points drawn uniformly (SubsampleMesh), unweighted, because the interior carries no measure.
- Formula: the library relative_l2 over the flattened (N, C) field, sqrt(sum diff^2 / max(sum t^2, 1e-8)).
  velocity_{x,y,z}_l2 apply the same formula to one column each.
- Averaging: the summary is the plain mean of the 200 per-case values.
- Why velocity_z_l2 reads 20-4,000: the truth z is float noise (|t_z| <= 6.8e-13, RMS 3e-17). The 1e-8 floor
  therefore takes over, and velocity_z_l2 = ||e_z|| / 1e-4 is a scaled absolute norm, not a relative error. The
  predicted z RMS is 6e-6 to 1.2e-4 of |U_inf|.

## Method: option (a)

- New scorer `$T/kernel/afplanar/score.sbatch` (md5 40898fab). It is a copy of the unit() in `kernel/afu_ud/score.sbatch`
  (b23f4507), with the same infer.py call, overrides, code_v20 + recipe_v8ud (afu_ud_*) or recipe_v8 (afu_r_*), 100,000
  cells, float32 and final checkpoints. Two things differ: it passes `+save_predictions=true`, and it writes to
  `$T/evals_afplanar/<run>/<protocol>/`.
- `planar.py` (16a25fd5) then:
  - loads each saved .pdmsh and divides pred_velocity and true_velocity by the case's |U_inf|;
  - recomputes the metrics in float32 with the same relative_l2 and eps;
  - checks the recomputed velocity_l2, _x and _y per case against both this run's rows and the shared scorer's
    metrics.jsonl.
- Sanity check: all 200 cases match in every lane and protocol. The largest per-case relative difference is 2.4e-7.
- The prediction dumps were deleted after a passing check (0 `predictions` dirs left; 163 files in evals_afplanar).

Commands, from `$T/kernel/afplanar`, on the `normal` QoS:

```
sbatch -J isla.afplanar.<tag> --export=ALL,AFP_RUNS='<runs>',AFP_PROTO=<native|posed> score.sbatch
```

Each job took one LHR node with 4 GB300 GPUs and about 8-14 min.

| job | tag | runs | protocol |
|---|---|---|---|
| 283963 | ud-native | afu_ud_{gt,t}_seed{42,43} | native |
| 283964 | r-native | afu_r_{gt,t}_seed{42,43} + afu_r_isla_seed42 (ISLA check) | native |
| 283965 | ud-posed | afu_ud_{gt,t}_seed{42,43} | posed |
| 283966 | r-posed | afu_r_{gt,t}_seed{42,43} | posed |

All four jobs exited 0:0.

## Native: seed values (42 / 43) and seed means

| arm | velocity_l2 (3-D, as in the book) | velocity_xy_l2 (in-plane) | xy / 3-D | z share of squared error |
|---|---|---|---|---|
| GeoTransolver afu_ud_gt | 0.03973 / 0.04022 / **0.03998** | 0.03973 / 0.04022 / **0.03998** | 0.999998 | <= 6.3e-6 |
| Transolver afu_ud_t | 0.02976 / 0.02958 / **0.02967** | 0.02976 / 0.02958 / **0.02967** | 1.000000 | <= 3.4e-7 |
| GeoTransolver afu_r_gt (raw drive) | 0.03801 / 0.03499 / **0.03650** | 0.03801 / 0.03499 / **0.03650** | 0.999996 | <= 1.2e-5 |
| Transolver afu_r_t (raw drive) | 0.06517 / 0.06871 / **0.06694** | 0.06517 / 0.06871 / **0.06694** | 1.000000 | <= 8.2e-8 |
| ISLA A2 afu_a2s (airfrans_a2.json) | 0.00970 / 0.00921 / **0.00946** | same (2-D: in-plane = velocity_l2) | 1 | 0 |
| ISLA afu_r_isla | 0.01161 / 0.01236 / **0.01198** | same; verified on seed 42: 0.011607 = 0.011607 | 1 | 0 |

## Baseline / ISLA ratios, native seed means (identical under both metrics to 3 decimals)

| arm | / A2, 3-D | / A2, in-plane | / afu_r_isla, 3-D | / afu_r_isla, in-plane |
|---|---|---|---|---|
| afu_ud_gt | 4.228 | 4.228 | 3.336 | 3.336 |
| afu_ud_t | 3.138 | 3.138 | 2.476 | 2.476 |
| afu_r_gt | 3.860 | 3.860 | 3.046 | 3.046 |
| afu_r_t | 7.079 | 7.079 | 5.587 | 5.587 |

## Posed (optional, in-plane SO(2) pose), seed means

| arm | 3-D | in-plane |
|---|---|---|
| afu_ud_gt | 1.16699 | 1.16699 |
| afu_ud_t | 1.23054 | 1.23054 |
| afu_r_gt | 1.09442 | 1.09426 |
| afu_r_t | 1.12396 | 1.12396 |

The largest z share is 6e-4, on afu_r_gt posed.

## Recommendation

Define the velocity comparison as the in-plane relative L2 for every arm. AirfRANS is a 2-D problem. The z column of
the 3-D embedding is not a physical quantity, and ISLA's metric has no z column, so in-plane is the only definition
that is the same for every arm. The book's numbers need no rescoring: in-plane and 3-D agree to 4-5 significant figures
on native.

What should change in the book:

- Label the velocity metric "in-plane (x, y)".
- Drop velocity_z_l2 from every table, or footnote it. It is eps-floored, an absolute error scaled by 1e4 rather than a
  relative error, and it is what made z look large.
- Strike the "about a quarter" note in RESULTS.md.

The headline ratios 3.05x and 5.59x (vs afu_r_isla) and 4.23x and 3.14x (vs A2) stand. Confidence is high: the result
is an exact recomputation, matched to 2e-7 per case.

## Rulings relied on and memory notes

- Rulings relied on: LHR `normal` QoS only; scoring only, with the shared trees read-only; the AF-UNIFORM reduction
  (the mean of the per-case values, seed means over 42 and 43).
- New for project memory: AirfRANS baseline velocity_l2 equals the in-plane metric to within 1e-5. velocity_z_l2 on
  the 3-D-embedded set is meaningless, because it is eps-floored.

## Files

Local, in `partC/`:

- `planar_table.md` and `planar_scores.json`, both written by `reduce_planar.py` (81299658).
- `results/<run>/<protocol>/`: planar_cases.csv (per case), planar_summary.json and the logs.
- `cluster/afplanar/`: score.sbatch, planar.py, the four sbatch logs, and md5_and_jobs.txt.

Cluster: `$T/kernel/afplanar/` and `$T/evals_afplanar/`.

## What is left

Nothing is running. The main session should decide whether to relabel the book's velocity column as in-plane and drop
velocity_z_l2.
