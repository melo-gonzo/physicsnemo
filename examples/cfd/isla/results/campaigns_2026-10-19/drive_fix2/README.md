# DRIVE-FIX2 score records (scored 2026-10-19)

Two scoring-only checks on existing checkpoints, run on the LHR site. No model was trained and no shared scorer, table or
eval tree was edited. Both folders are copied unmodified from `scratch/isla-drive-fix2-2026-10-19/` of the main checkout
(`partB/` → `scoreset/`, `partC/` → `planar/`); the readout of each is its `REPORT.md`.

| check | folder | what it scores | record | reader |
|---|---|---|---|---|
| T-SCORESET | `scoreset/` | Transolver 512 x 16 on the DrivAerML Surface task (`r80_dr_t_h512_l16_c100k`, the reference arm, and `r80ud_dr_t_h512_l16_c100k`, its unit-direction control), seeds 42 and 43, native / area / posed, re-scored at 200,000 cells on ISLA's scored set (hash-identical per car, `scoredset_compare.json`) | `tset200k_scores.json` (`rows`: 100k and 200k seed values per lane, protocol and field; `files`: every record reduced to its infer_summary row) | `results/char/a2r_protocol_artifacts_2026-10-17.py` (SCORESET: the 200k rows become the standing's Transolver arm in `char/drivaerml_surface_a2.json`, the 100k arm kept under `transolver_100k_scoring`); `book/standing.py`, `unit_drive_control()` |
| AF-PLANAR | `planar/` | the AirfRANS baselines (`afu_ud_{gt,t}`, the standing's arms, and `afu_r_{gt,t}`, the raw-drive arms), seeds 42 and 43, native and posed: the in-plane velocity error `velocity_xy_l2` beside the scored 3-D `velocity_l2` | `planar_scores.json`, `planar_table.md` (per-case values in `results/<run>/<protocol>/planar_cases.csv`) | the book's AirfRANS chapter; `book/standing.py` draws the AirfRANS velocity verdict (`provisional={}`) |

- `scoreset/reduce_tset200k.py` reads `scoreset/scores/evals_{char,r80ud,tset200k}/<run>/<protocol>/<run>/metrics.jsonl` and
  writes `tset200k_scores.json` and `reduce_tset200k.out`; run here, it reproduces both byte for byte.
- `planar/reduce_planar.py` reads `planar/results/<run>/<protocol>/planar_summary.json` and writes `planar_scores.json`
  (stdout: `planar_table.md`); run here, it reproduces both byte for byte.
- Scorers and job lists: `scoreset/cluster/tset200k/` and `planar/cluster/afplanar/` (with `md5_and_jobs.txt`).
- The `velocity_z_l2` column of the planar records is the relative L2 of a component whose true value is float noise; the
  1e-8 floor of the denominator makes it a scaled absolute norm, not a relative error, and the book does not print it.
