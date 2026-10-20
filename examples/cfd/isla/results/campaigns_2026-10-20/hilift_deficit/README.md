# HILIFT-090 records: where ISLA trails on HiLiftAeroML (analysed 2026-10-20)

The HiLiftAeroML standing's checkpoints, read case by case and cell by cell: `highlift_surface_medium` (510 training cases,
180 validation cases), ISLA's 200,000-cell set, float32, epoch-500 checkpoints, native mesh.

| arm | runs (seeds 42, 43) |
|---|---|
| ISLA, normalized slice reads (the standing's ISLA) | `a2r_hl_isla_seed42/43` |
| GeoTransolver 256, unit direction | `udrv_hl_gt_seed42/43` |
| Transolver 512 x 16, unit direction | `udrv_hl_t_seed42/43` |
| ISLA, raw slice reads (the CHAR record's ISLA, scored on another 200,000-cell draw) | `r80_hl_isla_w512_s512_seed42/43` |

Copied unmodified from `scratch/isla-hilift090-2026-10-20/` of the main checkout:

- `RESULTS.md` (the readout), `PREREG.md` (the pre-registration of the class's HiLift runs, HL090, for the record).
- `partA/`: `reduce_a_w512s512.py` and its outputs `tables_a_w512s512.md` and `percase_w512s512.csv` (one row per validation case,
  seed means); `RESULTS_A.md`, the first per-case readout, which compares against `r80_hl_isla_ref`, a smaller run that is
  not the standing's raw-read partner (its "Premise check" section does not apply to the book); the inputs
  `case_meta.json`, `hl_medium_manifest.json` (train 510 / val 180) and `splits_manifest.json` (the dataset's split lists).
- `partA_spatial/`: `SPATIAL.md`, the squared error split over surface regions, from `summarize_spatial.py` over
  `percase_region.csv` and `checks.csv`; `cluster_spatial/` holds the cluster scripts that saved the predictions and wrote those
  two files (`savepred.sbatch`, `infer_wrap.py`, `spatial_reduce.py`, `spatial_reduce.sbatch`). The predictions themselves
  (27 GB) were deleted after the reduction. In the spatial split ISLA is the mean of its two seeds and GeoTransolver and
  Transolver are seed 42 only.
- `partB/PARTB.md`: the class's HiLift surface task needs no decimated geometry asset (memory and step time measured).

Written here: `reduce_deficit.py`, which reads the copied files and writes `hilift_deficit_2026-10-20.json`, the numbers
chapter 10 prints (`python3 reduce_deficit.py`).

Readings of `RESULTS.md` that the files do not carry, or carry differently:

- The lift coefficients ("CL 4-8 % low" on the wrong branch, "CL -0.5 to -2.4 %" for the raw-read variant) appear in no file
  of the record; the book does not print them.
- The ten cases with the largest ISLA - GeoTransolver gap include one pre-stall case at 4 degrees (`geo_LHC173_AoA_4`); the other
  nine are post-stall at 16 to 22 degrees.
- `geo_LHC002_AoA_18` and `geo_LHC132_AoA_22` are wrong on both seeds of the standing's ISLA; the raw-read variant
  `r80_hl_isla_w512_s512` reads 0.099 and 0.098 on its seed 42 there (the "every ISLA run" of `RESULTS.md` counts
  `r80_hl_isla_ref`).
- Outside the 16-degree bin the normalized reads' ratio to the raw-read variant is 0.94 to 1.02 per angle (22 degrees: 1.019),
  not "1-6 % better" in every bin.
