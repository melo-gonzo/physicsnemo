# UDRV-HL score records: the HiLift standing with unit-direction baselines (scored 2026-10-19/20)

The HiLiftAeroML 210-wing rung, 180 validation cases, float32, epoch-500 checkpoints, every arm scored on ISLA's
200,000-cell set under the three CHAR protocols (native mesh, uniform point cloud, rotated + translated):

| arm | runs (seeds 42, 43) | reads |
|---|---|---|
| ISLA, normalized slice reads (A2-RERUN rows 8-9) | `a2r_hl_isla_seed42/43` | `U_inf_dir` |
| GeoTransolver 256, 200,000 training cells | `udrv_hl_gt_seed42/43` | `U_inf_dir` |
| Transolver 512 x 16, 100,000 training cells | `udrv_hl_t_seed42/43` | `U_inf_dir` |
| GeoTransolver 256 (REF-80, the CHAR record's arm), re-scored on ISLA's set | `r80_hl_gt_h256_seed42/43` | `U_inf` in inch/s |
| Transolver 512 x 16 (REF-80, the CHAR record's arm), re-scored on ISLA's set | `r80_hl_t_h512_l16_c100k_seed42/43` | `U_inf` in inch/s |

Copied unmodified from `scratch/isla-udrv-hl-2026-10-18/` of the main checkout:

- `PREREG.md` (written before any UDRV-HL lane ran: predictions P1-P3 and the P2 bands), `RESULTS.md` (the readout, with
  verdicts and caveats), `JOBS.md` (jobs, md5s, scoring commands).
- `udrv_table.py`, the reducer, and its outputs `udrv_hl_table.md` and `udrv_hl_scores.json` (seed means and per-seed values
  of five fields, three protocols, every arm; `verdict_inputs`: the P1 and P2 outcomes). Its `CHARP` path is written for the
  scratch folder; from here the CHAR protocol files it reads for its record rows are `../../campaigns_2026-10-02/char/protocols/`.
- `scores/<run>[_s200k]/<protocol>/<run>/metrics.jsonl`: the scorer output of every arm and protocol (per-case `infer_step`
  rows and the `infer_summary` row); the scorer logs and markers are not copied.
- `phys/*.md`: the checks behind the physical-drive re-scores (set identity, leaf diff, scorer check).

Reader: `results/char/hilift_a2_2026-10-20.py` writes `results/char/hilift_surface_a2.json`, the standing artifact of
`book/standing.py` (`DATASETS["hilift"]`), from `udrv_hl_scores.json` and the per-case rows here. It keeps the CHAR record's
raw-read ISLA arm (`raw_read_isla`), the physical-drive arms on ISLA's set (`raw_drive_geotransolver`, `raw_drive_transolver`)
and the CHAR record's baseline arms (`char_baselines`), and records the registered outcomes (`registered`: A2-RERUN R-A2 on
HiLift, UDRV-HL P1, P2, P3) and the per-case readouts (`case_level`).

Caveats the readout states: the arms trained on different clusters with different code and recipe snapshots (the
pre-registration accepted this; the GeoTransolver and Transolver model sources are md5-identical); the CHAR record's arms were
scored on another copy of the dataset whose dataset index, and so whose 200,000-cell draw, differs from ISLA's set.
