# GEO0: ISLA checkpoints whose query geometry is computed against the full mesh, under boundary redraws

Copied from `~/gh/physicsnemo/scratch/isla-geo0-2026-10-16/`. Pre-registration `PREREG.md`, report `REPORT.md`, jobs
`JOBS.md`; per-unit numbers `RESULTS_TABLE.md` and `results.json`, both from `reduce.py` (its verdicts in `reduce.log`);
`diag_af_percase.{py,txt}` the unregistered per-case diagnostic; `evals_lhr/<run>/<protocol>/<run>/metrics.jsonl` the
score records (LHR jobs 241664, 241665, 241802).

Instrument: ISLA at width 384, 12 layers, 256 slices (an older, smaller class than the reference configuration), code_v3 +
recipe_v3, raw slice reads. Each `sg_*_provided` checkpoint (query distance and direction from the stored, full-mesh
geometry) is compared with its `sg_*_internal` twin (the same training with the nearest-boundary-token geometry). Protocols:
native; `g0u`/`g1u` a uniform redraw of the boundary; `g0a`/`g1a` an area-law redraw; `g0v`/`g1v` a second uniform redraw
(the noise floor). DrivAerML Volume: 48 validation cars, 200,000 vehicle tokens drawn from 1,000,000. AirfRANS: the 3-D
`airfrans_volume` set at 400 airfoil segments.
