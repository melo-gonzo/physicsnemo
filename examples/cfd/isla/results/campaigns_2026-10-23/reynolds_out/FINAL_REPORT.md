Status: DONE

The full design is in `DESIGN.md`. The bitwise check is in `bitwise/RESULTS_BITWISE.md` (child REYNOLDS-OUT-BITWISE) and its jobs in `bitwise/JOBS.md`. I wrote no code on any branch and made no commits.

**What ran.** Two LHR GB300 short-QoS jobs, with the five bad nodes excluded: 376561 (0.245 GPU-h) and repeat 376855 (0.161 GPU-h), 0.406 of the 1 GPU-h cap. Seven cpu-short debug jobs. Nothing is running. I added the registry row `^isla\.reynoldsout\.` and have removed it again.

**Survey** (`DESIGN.md` §1). At 5b70c8686 the application-specific pieces are:
- the `reynolds={mean, std}` argument;
- `_init_scales`, with the rule "viscosity role iff reynolds" and the check "Re = U L / nu";
- `_case_scales`, which computes z = (log10(U L / nu) − m) / s in float64 and appends it as the last global scalar;
- the vocabulary in scales.py: `BASE_SCALES = (L, U, rho)`, `ROLES` density/viscosity, `velocity_scale`;
- the docstrings ("Pa", "density", "Reynolds").

Only two configs use any of this: `isla_volume_airfrans.yaml` and `isla_volume_aerozoo.yaml`. They take m and s from the per-split `isla_constants.reynolds` in 7 AeroZoo and 12 AirfRANS dataset files. The arithmetic itself is generic: L (the boundary's RMS pairwise distance), units, `scale` and `output_stats`. The recipe already has `ComputeLogReynolds`, but it computes in float32 and no dataset uses it.

**Evidence** (`DESIGN.md` §2):
- **AirfRANS, interacting ISLA.** The 0.9.0 gain comes entirely from standardizing the one global scalar. NOSC-STD reads 1.003 / 1.029 / 1.043 and STDU 1.008 / 1.015 / 0.995 (both measured): log Re and standardized speed serve equally well.
- **AirfRANS, ISLA-DTT** (DTT-KNOBS R5, 22:58 EDT). Without the scales it is 1.208 / 1.203 / 1.177 worse ("the scales carry it"; velocity flagged). So the scales' function must be kept.
- **How z is scaled decides extrapolation.** On AirfRANS (AF-EXTRAP) the fixed log10(Re/1e6) halves the below-band pressure error. On AeroZoo (LRE-AZ) it harms velocity by 1.126. So the per-split fitted constants stay, and the choice remains Peter's.
- **AeroZoo without z (not measured).** No arm has dropped z. With the scales on, z is ISLA's only channel for the Reynolds number, so without it ISLA's dimensionless outputs could not depend on Re.
- **AeroZoo spec** (#sec-allowed). It allows a Reynolds number that a model computes from the inputs "in its architecture or its own preprocessing". Its `derive_globals` uses ISLA's length.

**Options** (`DESIGN.md` §3):
- **(i) Exact relocation.** The recipe computes z and passes it as an ordinary global `re_z: {rank: 0}`. This needs no retrain. Measured: z is bitwise the same on 160 of 160 cases over three real checkpoints, and the network's input is bitwise the same in every forward. Outputs are bitwise wherever the GPU forward repeats itself. Where it does not, the differences between the old and new form stay inside the old model's own run-to-run spread (AirfRANS DTT 1.8–2.6e-6 against 1.9–2.4e-6). On AeroZoo case 0 one new-form forward matched the old model at all 24 block tensors; the other two differed from it and from each other by up to 8.8e-6.
- **(ii) Remove the feature.** This makes the model blind to Re. Its no-harm check would cost about 950–1,300 GPU-h (estimated) and would likely fail on AeroZoo.
- **(iii) Generic transforms** that implement (i).
- **(iv) A generic dimensionless group inside the model.** It works, but it puts a log and fitted constants back into the constructor.
- Moving the output scaling to the recipe: not recommended. It is dimensional analysis, it needs L, and only its names are about fluids.

**Recommendation for 0.13.0:** (i), built as (iii), plus a renaming step. Neither step changes any arithmetic, so no retrain check is needed.
- **Step 1, the relocation.**
  - model.py: remove `reynolds`, the viscosity role, z and its column term.
  - Recipe: a new `ComputeBoundaryScales` (L and the source's RMS speed, computed the way ISLA does) and `ComputeLogReynolds` changed to float64. Both are appended in ISLA's two model configs.
  - `_apply_dataset_transform_appends` fills mean and std from `isla_constants.reynolds`; `model_dataset_constants` stops injecting `reynolds`.
  - Dataset files and baselines unchanged.
- **Step 2, generic vocabulary.** A new key `defines: <name>` replaces `velocity_scale` and the roles: on a rank-1 field it sets an RMS magnitude (plus the direction g), on a rank-0 global its value. `units` may name L and any declared scale. After this, model.py names no fluid quantity.
- **Tests.** test_isla_scales.py and the configs that use it; recipe transform tests (hand values, bitwise equal to `_gauge` and `_case_scales`, nothing read from `reference.*`); config tests; one datapipe check per dataset in ISLA-API's GB300 job that `re_z` equals the old z.

**Baselines and AeroZoo's input rule.** The baselines get no recipe Re. Peter ruled on 10-19 that ISLA's RMS length must not reach other models, and their dimensional inputs already determine Re. z reads only the posed problem plus training-split constants. It must run in ISLA's model config, never in a dataset chain that every model shares.

**For Peter or the supervisor:**
1. Keep the Reynolds feature as ISLA's recipe-side preprocessing (recommended).
2. The name of the key `defines` (step 2).
3. Fitted or fixed z scaling is still open.
4. For memory: the Reynolds input is ISLA's own preprocessing, in its model config only.
5. Side finding, not about Reynolds: 0.9.0's eager AeroZoo GPU forward does not always repeat before block 0 (case 0; the child's untested guess is token order). Any bitwise A/B test on AeroZoo needs repeats.
6. The brief's AirfRANS constants {9.065, 0.1338} are 5b70c8686's whole-boundary values. The REF-090 checkpoints carry {6.205, 0.1347}, from the airfoil-only L. Both are correct for their code.

**On disk:** this folder (under 1 MB) and `$T/kernel/reynoldsout/` (small). Both can be pruned after review.
