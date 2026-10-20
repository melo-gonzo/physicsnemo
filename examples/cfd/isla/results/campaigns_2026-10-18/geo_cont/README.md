# GEO-CONT: a continuous query-geometry encoding on exact geometry

Copied from `~/gh/physicsnemo/scratch/isla-geo-cont-2026-10-18/`: pre-registration `PREREG.md` (with its amendment), jobs
`JOBS.md`, the real-data probe `MEASUREMENTS.md`, `DESIGN_NOTE_medial_ties.md` (the direction ties on a part's medial
surface), the Tensile2d reduction (`reduce_t2d.py` -> `t2d_scores.json`, `t2d_table.md`) and the lug table
(`lug_table_2026-10-19.md`, scoring job 285539). Score records: `scores/gcont_{ctl,cont}_t2dz_isla_seed4{2,3}/<protocol>/`
and `scores/evals_gcont/gcont_{ctl,cont}_{af,lug}_isla_seed4{2,3}/<protocol>/` (native, area, posed; float32, final
checkpoints).

Arms: CONTROL (`gcont_ctl_*`), exact query geometry against the full-resolution surface with the encoding
[s, sign(s) log(|s| + 1e-12)] and direction u; CONT (`gcont_cont_*`), the same with [s, asinh(s / 1e-6)] and direction
sign(d) u, sign(0) = +1. Both: normalized slice reads, the reference configuration, two seeds, on AirfRANS uniform, the GINOT
lug and PLAID Tensile2d. The book reads them through `../../campaigns_2026-10-19/geo_evidence/geo_evidence.json`.
