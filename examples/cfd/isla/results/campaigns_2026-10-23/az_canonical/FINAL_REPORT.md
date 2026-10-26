Status: DONE

# AZ-CANONICAL: AeroZoo's canonical benchmark (mean_stable only) in the rerun's configs (14:10-15:25 EDT 10-26)

## Branch
`psharpe/isla-az-canonical` (worktree /home/psharpe/gh/physicsnemo-isla-az-canonical, from d8a0ca6a0). Pushed, not merged.
Head **06ba4e00b**. Both commits are `-s`, author Peter Sharpe, all hooks passed, no trailers:
- def1dcf46 "AeroZoo canonical benchmark: mean_stable samples only, with constants refit on them".
  - `src/datasets.py`: a dataset file's `sample_filter` block (`global_data: <leaf>`, `keep: [...]`) filters every
    manifest split part. The label is read from each file outside the transform chain, so it never reaches a model.
    `++apply_sample_filter=false` turns the filter off. A filter that keeps nothing, a malformed block, or a filter in
    directory mode raises an error.
  - The seven `datasets/aerozoo_uvolume*.yaml` files keep `diagnostics.solve.solution_quality_id` = 4 (mean_stable).
    All their constants are refit on the mean_stable train cases (see Constants).
  - New `results/aerozoo_dataset/aerozoo_mean_stable_manifest.py` and `aerozoo_mean_stable_constants_fits.json`
    (md5 3f01b074). `isla_constants_fit.py summarize` accepts `-` for the AirfRANS manifest.
  - 8 new tests. The AeroZoo config test now expects the new keys and the new full-split pressure std.
- 06ba4e00b: reads the label leaf through its `meta.json` and memmap. The old path ran `TensorDict.load_memmap`
  on the whole group, and the filter took 244 s cold on LHR with 7 processes reading at once. It now takes 3-9 s.

No change under physicsnemo/ or to any model config. The dataset is not modified.

## Counts: spec vs configs (train / val / test; all 21 split parts equal)
| split | spec (06-splits.qmd code, ba5b38e8) | configs (loader smoke, side B) | released (side A) |
|---|---|---|---|
| full | 2,011 / 290 / 515 | 2,011 / 290 / 515 | 2,713 / 386 / 716 |
| family_non_box | 1,850 / 274 / 692 | same | 2,390 / 351 / 1,074 |
| reynolds | 1,577 / 238 / 1,001 | same | 2,119 / 311 / 1,385 |
| bank_blobs | 1,662 / 237 / 917 | same | 2,156 / 292 / 1,367 |
| bank_zero_to_cad | 2,250 / 332 / 234 | same | 3,040 / 442 / 333 |
| profile_crossflow | 2,298 / 345 / 173 | same | 2,820 / 417 / 578 |
| profile_dual_inlet | 2,044 / 313 / 459 | same | 2,875 / 424 / 516 |

- The stored labels equal release-v1.2.json's `solution_quality` on all 3,815 complete cases: 2,816 mean_stable
  (job 407928).
- The kept lists equal, case by case, the lists the spec's own code computes from splits.json and the release.
- All 21 configs (ISLA, GeoTransolver, Transolver × 7 splits) give these counts. Full table: AB_RESULTS.md.

## Constants (supervisor's REFIT ruling, 10-26)
- Every fixed constant is refit on each split's mean_stable train list, with the same tools and seeds as before:
  NormalizeMeshFields stds, nu_std, the MergeBoundaries token stds, and ISLA's reynolds, output_scale and input_scale.
  The ISLA scale rows come from jobs 407910-407911: all 3,656 train cases, code_r090, 0 errors.
- The same tools run on every train case reproduce the old values: pstats bitwise, AZ7's rms fit bitwise, and every
  `isla_constants` line of all 7 files exactly.
- Largest relative change of each constant over the splits (all values in CONSTANT_CHANGES.md):
  - NormalizeMeshFields stds: tke 38 %, nut 28 %, omega 15 %, pressure 10 %, velocity 6.0 %, retheta 1.8 %,
    intermittency 0.2 %.
  - ISLA output_scale: pressure 26 %, tke 15 %, nut 10 %, omega 8.7 %, retheta 1.8 %, velocity 0.3 %.
  - ISLA input_scale: bc_tke 13 %.
  - MergeBoundaries token stds: bc_omega 12 %, bc_tke 11 %.
  - reynolds std 2.9 %, nu_std mean 2.2 %.

## A/B (LHR cpu-short 408024 = side A at d8a0ca6a0; 408099 = side B at 06ba4e00b, plus B' = B's code on A's yaml files)
- Each config reads 1 train, 1 val and 1 test case, each on its own seeded generator: 63 case reads per side.
- **A = B': 3,434 of 3,434 leaves are bitwise**, for ISLA, GeoTransolver and Transolver on all 7 splits.
- **B vs B'**: every one of the 378 differing leaf columns equals c·B' + d, where (c, d) is exactly one refitted
  constant pair. The max relative residual is 2.1e-7 (float32 rounding).
  - 252 columns are targets and ISLA's prediction-point targets (NMF), 98 are GeoTransolver token columns (MB),
    21 are the baselines' nu_std, and 7 are ISLA's re_z (reynolds).
  - Every other leaf is bitwise.
- **Baselines**: their inputs per sample change only through their own refitted constants. Their model configs and
  code are untouched.
- Recipe tests: **639 passed, 24 skipped** (local CPU, td0153; logs/recipe_tests_06ba.log). I did not run the ISLA
  test directory: there is no model or physicsnemo change.

## Scorer (child AZ-CANONICAL/SCORER; scorer/REPORT_SCORER.md)
- Files in scratch/isla-final-rerun-2026-10-26/lhr/:
  - specscore.py ad9d130c: AZ-RESCORE 4577daff + AZ7's diff + the canonical handling.
  - build_json.py d8be966c, which writes the `canonical` and `all_complete` subsets.
- One pass with `++apply_sample_filter=false` scores every complete test case. The scorer asserts that the
  canonical list equals the release's mean_stable cases.
- **Reproduces AZ-RESCORE bitwise**:
  - 240 of 240 per-case eps and rel values (relL2 included) on 10 full.test cases (5 stable, 5 unstable) of
    azc_full_gt and azc_full_t seed 42. Jobs 407637 and 407728, 0.33 GPU-h.
  - Its reduction gives 5,376 of 5,376 numbers in AEROZOO_SPECMETRIC.json exactly: stable and all, mean and median.
- On the canonical recipe the assertion passes: "selects exactly the release's 515 mean_stable cases" (job 408106,
  CPU). The negative check stops as it should.
- Untested: the eps_gamma and eps_retheta model path, until a seven-field checkpoint exists.

## SCORING.tsv and price
- SCORING.tsv gains 60 rows, 2 per AeroZoo lane; the 121 old rows are intact:
  - the primary row: "canonical (spec, mean_stable) PRIMARY";
  - the second row: "all complete (spec), labelled second score".
  Each gives ε with r_U and r_p (per-sample relL2 over every interior point, float64, mean and median) beside it.
- price/PRICE.md gains a new section (price_canonical.py and .tsv) with 250 epochs:
  - **30 AeroZoo lanes: 4,732 → 3,598 GPU-h (−1,134, −24 %).**
  - **Rerun total, 48 lanes: 5,658 → 4,524 GPU-h.**
  - Longest lane 48 → 35 h. 10 nodes now finish in 5.5 days (13 before).

## For the main session
- Rulings relied on: Peter's mean_stable and relL2 rulings (via the brief); the supervisor's 250 epochs; REFIT (10-26).
- Suggested memory entry: "AeroZoo canonical: every split part keeps only its mean_stable cases (`sample_filter`); all
  dataset constants are fitted on the mean_stable train cases; `++apply_sample_filter=false` gives the labelled
  all-complete score."
- Before launch:
  - Snapshot 06ba4e00b (after the merge).
  - LANES.tsv notes still cite the AZ7 commits.
  - The 60 old "native (spec)" rows are superseded for reading, not deleted.
  - Each link start reads about 3,100 labels, 3-9 s.
- For the book: every AeroZoo number from the rerun is canonical (mean_stable), and the all-complete score is
  labelled. Old standings were trained on all complete cases.
- Nothing is running, and the registry row is removed.
- On disk: the worktree, scratch (20 MB), LHR $T/kernel/azcanon (86 MB: snapshots and probes) and
  $T/kernel/azcanon_score (2.4 MB).
