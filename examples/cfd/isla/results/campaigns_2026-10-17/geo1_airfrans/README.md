# GEO1: exact full-resolution query geometry on AirfRANS, against the standing's ISLA arm

Pre-registration `PREREG.md` (approved before any lane was submitted), launch report `REPORT.md`, jobs `JOBS.md`, lane
rows `rows_10_13.tsv`; copied from `~/gh/physicsnemo/scratch/isla-geo1-airfrans-2026-10-17/`.

The score records are in `../../campaigns_2026-10-14/afu/`: `protocols/afu_{geo1,a2s}_isla_seed4{2,3}/{native,area,posed}/`
and `ca2_400/afu_{geo1,a2s}_isla_seed4{2,3}/{ca2u,ca2a,ca2v}/` (LHR job 253816 for the 400-segment protocol). The book reads
them through `../../campaigns_2026-10-19/geo_evidence/geo_evidence.json`.

Arms: `afu_geo1_isla`, each prediction point's distance and direction from the exact closest point on the full-resolution
airfoil (snapshot code_geo1, encoding [s, sign(s) log(|s| + 1e-12)]); `afu_a2s_isla`, the nearest boundary token (snapshot
code_a2s, the standing's ISLA arm). Both: normalized slice reads, width 512, 12 layers, 512 slices, 3.75e-4, 500 epochs,
uniform-interior AirfRANS files, recipe_v11, seeds 42 and 43.
