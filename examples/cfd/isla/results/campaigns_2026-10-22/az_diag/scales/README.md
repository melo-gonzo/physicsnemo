# AZ-DECIDE scales sub-lane: ISLA 0.9.0's case scales on the 40 worst AeroZoo test cases (2026-10-22)

Scoring-side analysis. CPU only (LHR cpu-short). No score file and no ISLA-DTT (`azc_*_dtt_*`) file was read. No
snapshot, launcher or dataset was edited.

## Question

Do ISLA's worst test cases on `family_non_box` (20) and `profile_crossflow` (20) (`cases.tsv`, from `../worst/`) sit
outside the training range of ISLA's own in-model case scales U, L and the Reynolds input z?

## Method: the constants-fit path, reused unchanged

- `isla_constants_fit.py` is a verbatim copy of `examples/cfd/isla/results/aerozoo_dataset/isla_constants_fit.py`
  (origin/psharpe/isla 43e5b6f76, md5 41e26ee0). This script fitted the splits' `isla_constants` (jobs 295774-295776).
  `scales_cases.py` imports its `aerozoo_case` and adds only the case's density and viscosity (`global_data physics.*`).
- `aerozoo_case` (isla_constants_fit.py:110) builds ISLA from `recipe_r090/conf/model/isla_volume_aerozoo.yaml`
  (`_model`, :62; `scale` keys stripped, `reynolds={mean 0, std 1}`, so `s["z"]` is the raw log10 Re). It loads
  `domain.pdmsh` and passes the four boundaries **at full resolution, in float64**, with the inlet's
  `prescribed.dirichlet.velocity` renamed `bc_velocity` (:124, the pipeline's RenameMeshFields). Then `_scales` (:88-95)
  calls `ISLA._gauge` and `ISLA._case_scales`.
- The model code is code_r090 = ISLA 0.9.0, psharpe/isla 92d9d7800, `physicsnemo/experimental/models/isla/model.py`:
  `_gauge` :1306 (L = `rms_pairwise_distance`, geometry.py:71, of all boundary cells with their own second moments,
  `cell_second_moments` geometry.py:106). `_case_scales` :1317: U = sqrt(Σ a|v|² / Σ a) over the inlet's `bc_velocity`
  (`velocity_moments`, scales.py:200), and log10 Re = log10(U L / ν) with ν = `physics.kinematic_viscosity`. The forward
  pass calls the same two methods (model.py:1600-1601) on the boundaries it is given. The ISLA files are identical between
  the fit's code 6c3aac5a8 and 92d9d7800 (`git diff --stat` is empty).
- **Which boundaries.** For ISLA the recipe gives the model the full-resolution boundaries:
  `isla_volume_aerozoo.yaml:48` drops SubsampleMesh, ComputeSDFFromBoundary, HarmonizeBoundaryFields and MergeBoundaries,
  and `forward_kwargs.boundary_meshes` (:83) passes the whole meshes. The transforms that remain do not change U, L or ν:
  the rename, StandardizeGlobalLeaf (writes a separate `nu_std` leaf), FlipBoundaryOrientation (U is an RMS, L depends
  only on geometry), and the interior target normalization. So the fit path equals the inference path, except that it
  runs in float64 throughout (the forward pass computes the gauge in the points' dtype).
- **Derived columns** (`reduce_scales.py`): z = (log10 Re − mean) / std with the split's `isla_constants.reynolds`
  (nonbox 5.846/1.383, crossflow 5.969/1.408). The output scales are per case, so two columns: `out_factor.<o>` =
  c·U^a·L^b·ρ^e, the physical size of a head output of 1, and `head_rms.<o>` = (per-case RMS of the dimensionless target)
  / c, the RMS of the head's target. The constant c is the train median of that RMS, so the typical train case has 1.
  `in_rms.<i>` = per-case input RMS / c_in. Velocity is the isotropic RMS, on a seeded 20,000-point interior sample.
- **Train ranges.** As asked: a seeded uniform sample of 300 train cases per split (`random.Random(0).sample`,
  `scales_train_sample.tsv`). The fit took about 1 s of wall time per case on 60 workers, so I also ran **every** train case (3,332
  unique cases; 2,390 nonbox and 2,820 crossflow rows, `scales_train_full.tsv`) and every test case of both splits
  (`scales_test_all.tsv`). The flags use the full train range; the 300-sample flags are in `flag_sample.*` and agree.
- **Check.** The full train side reproduces the fitted constants exactly: mean and population std of log10 Re are
  5.846/1.383 (nonbox) and 5.969/1.408 (crossflow), and the median head RMS is 1.000 for every output. Cases computed in
  two jobs agree to 0 in log10 Re.

## Jobs (LHR cpu-short, 60 workers each; cluster files in `$T/kernel/azdecide/scales/`)

| job | what | elapsed |
|---|---|---|
| 326388 | 40 worst + 300 nonbox train sample (`scales_a.jsonl`) | 6:40 |
| 326389 | 300 crossflow train sample (`scales_b.jsonl`) | 6:33 |
| 326497, 326498 | all train cases of both splits, 2 shards (`scales_full{0,1}.jsonl`) | 34:53, 34:58 |
| 326982, 326983 | all test cases of both splits, 2 shards (`scales_test{0,1}.jsonl`) | 12:54, 11:42 |

Command: `sbatch -J isla.azdecide.scales-<x> scales.sbatch <select> <out.jsonl> [n_per_split [shard n_shards]]`,
then `python reduce_scales.py > report.md` locally (raw jsonl and logs in `raw/`). 0 errors in 5,620 case runs.

## Results (full tables: `report.md`; per case: `scales_worst.tsv`)

Worst cases against the full train range: outside [min, max] / beyond the 1st or 99th percentile (outside included):

| split (20 worst each) | U | L | log10 Re = z | head RMS velocity | head RMS pressure | head RMS omega |
|---|---|---|---|---|---|---|
| nonbox | 0 / 0 | 2 / 4 | 2 / 4 | **9 / 9** | 0 / 2 | 1 / 6 |
| profile_crossflow | 0 / 3 | 0 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

- **nonbox (train: external_box only; test: other families).** The 10 velocity-worst cases are all `internal_duct`.
  Their U, L and z are typical (z between −0.9 and +0.8, except one at +2.08), but 9 of the 10 have head RMS velocity
  0.29-0.38, below the train minimum of 0.785. Their isotropic interior velocity RMS is about 0.17 U (|v| RMS about 0.3 U), against at
  least 0.45 U (|v| 0.78 U) in every training case: ISLA's velocity unit (the inlet RMS speed) is about 3 times too
  large for these ducts. In the whole
  duct test set, 40% are below the train minimum, and the worst are its smallest (percentile 0 among all test cases). The
  pressure-worst cases are 6 `external_opposed_hemisphere` cases at high Re (median log10 Re 8.39; 3 beyond p99, 1 above
  the train maximum of 8.638) and 2 `external_opposed_sphere` cases with a tiny head pressure RMS (0.03-0.04, below the
  train p1 of 0.115).
- **profile_crossflow.** Every worst case lies inside the train range of U, L, z and all head RMS. Three are beyond
  p1/p99 in U (48.4, 49.6 and 0.265 m/s) and one in L. The pressure-worst cases are the test cases with the smallest
  dimensionless pressure (head RMS 2.3-4.0 against a test median of 11.5; percentile 2 among the 578 test cases). Their
  z is typical (−1.7 to +1.05).

## Conclusion

1. U, L and z do not explain the failures. No worst case on either split falls outside the training U range, and only
   2/20 on nonbox (opposed-hemisphere and duct cases at log10 Re 8.65-8.72) and 0/20 on crossflow fall outside the
   training Re/z range.
2. The out-of-range quantity is the in-model output scale. On nonbox, 9 of the 10 velocity-worst ducts have interior
   speeds of about 0.3 of ISLA's velocity unit, below every training case. ISLA's U (the inlet RMS speed) is the wrong
   unit for these internal flows.
3. Crossflow's worst cases sit inside every training range, with z typical. They are the test cases with the smallest
   pressure relative to ρU², so their failure is not scale extrapolation. Likely causes are the new inflow profile and a
   small target under a relative-L2 metric (inference, not tested here).
