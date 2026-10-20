# UDRV-HL preregistration: the HiLift reference baselines with the unit freestream direction (2026-10-18)

Written 2026-10-18 before any UDRV-HL lane was submitted (smoke job 1097163 only). Lane files: JHB
`$J/kernel/udrv_hl/` (`J=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-17-isla-a2r`);
local copies in `cluster/udrv_hl/`. Jobs, md5s and the scoring commands: `JOBS.md`.

## Why

The REF-80 HiLift baseline lanes, whose scores are the HiLift standing of GeoTransolver and Transolver in the CHAR record
(`examples/cfd/isla/results/char/hilift_surface.json`), fed the models the physical freestream velocity: GeoTransolver
`forward_kwargs.global_embedding: global_data.U_inf`, Transolver `forward_kwargs.fx.source: global_data.U_inf` (verified in the
lanes' own `resolved_config.yaml` on AGA, `$T/runs/r80_hl_gt_h256_seed42`, `r80_hl_t_h512_l16_c100k_seed42`). On HiLift
`U_inf` is in inch/s: the smoke probe (job 1097163) read U_inf = (2664.83, 0, 280.08) and (2620.95, 0, 557.10) for two
training cases, |U| = 2679.51 in both, so the magnitude is fixed and the direction turns with angle of attack (6.0 and
12.0 degrees); `NonDimensionalizeByMetadata` touches only cell_data. ISLA reads the unit direction `U_inf_dir`
((0.99452, 0, 0.10453) and (0.97815, 0, 0.20791) for those cases, norm 1.000000). The September unit-drive ladder (#sec-nb-udrvl-210-verdict,
#sec-nb-udrvl-510-verdict, #sec-nb-udrv-1260-verdict) found that this input scale alone inflated the baselines' HiLift error
(physical-drive GeoTransolver 3.4x its unit-drive self at 210 cases, Transolver 1.5x; at 510 cases the unit drive improved
them by a median 0.62x and 0.67x per case). REF-80 and the CHAR record were set after the ladder, with the physical drive.

## Question

At the reference configuration, with the input ISLA reads, where do GeoTransolver and Transolver stand against the HiLift A2
ISLA lanes `a2r_hl_isla_seed42/43`?

## Arms (lanes.tsv rows 0-3; group udrv; 500 epochs; JHB, normal QoS, 03:59:00 singleton chains as a2r_hl)

| row | run id | model | reference REF-80 row (LHR/AGA `$T/kernel/char/r80_lanes.tsv`, md5 a9f5c6f9) | extra | lr | cells |
|---|---|---|---|---|---|---|
| 0, 1 | udrv_hl_gt_seed42/43 | geotransolver_surface (width 256, 12 layers) | idx 32 / 46 `r80_hl_gt_h256_seed42/43` | `+training.val_interval=25 forward_kwargs.global_embedding=global_data.U_inf_dir` | 1e-3 | 200,000 |
| 2, 3 | udrv_hl_t_seed42/43 | transolver_surface (width 512, 16 layers) | idx 38 / 52 `r80_hl_t_h512_l16_c100k_seed42/43` | `+training.val_interval=25 model.n_hidden=512 model.n_layers=16 forward_kwargs.fx.source=global_data.U_inf_dir` | 1e-3 | 100,000 |

- These are the REF-80 reference configurations (#sec-nb-ref80-verdict): GeoTransolver width 256 at 200,000 cells,
  Transolver width 512 x 16 at 100,000 cells. Model, width, depth, lr, seed, cells, epochs and extras are the REF-80 rows'
  exactly; the fixed arguments of `lanes.sbatch` equal `r80_lanes.sbatch`'s (bfloat16, augment=false, compile=false,
  save_interval 2, batch 1, 4 GPUs). The only added override is the drive.
- The override is spelled `forward_kwargs.*`, not `model.*`: the model yamls are `@package _global_`, so the drive keys sit at
  `forward_kwargs.global_embedding` and `forward_kwargs.fx.source` (the September UDRV lanes used the same keys).
- Changed besides the drive, by necessity: dataset `highlift_surface_medium` from `recipe_a2r_hl` (the dataset yaml and
  its `_area` / `_posed` twins differ from recipe_v8's only in the manifest line, and the HiLift root points at the JHB
  subset copy; the same 510 train / 180 val cases, manifest md5 314bd9d2; the two model yamls are identical to
  recipe_v8's), the recipe source of recipe_a2r (train.py, loss.py, nondim.py, domain_transforms.py and others refactored
  since recipe_v8; the same recipe the ISLA comparator trains with), and code `code_a2r` (7e167820a) instead of code_v18
  (7ef7c34fa). The GeoTransolver and Transolver model
  sources and their building blocks are md5-identical between the two (JHB `verify/` listing in JOBS.md); the differing files
  are ISLA, the mesh library, datapipes refactors and a radius-search padding fix that cannot trigger at these sizes.
- Cluster: JHB GB300 instead of AGA GB300.
- Comparator: `a2r_hl_isla_seed42/43` (JHB `$J/runs`, A2-RERUN rows 8-9; still training, epoch ~285 of 500 on 2026-10-18).
  Record arms: the physical-drive REF-80 baselines (CHAR record, below).

## Metric

Relative L2 of pressure (primary) and wall shear on the 180 validation cases, seed means and per seed, under the three
CHAR protocols native / area (uniform point cloud) / posed (rotated + translated): final checkpoint, float32,
`infer_split=val`, recipe `src/infer.py`. Scorer: `$J/kernel/udrv_hl/score.sbatch`, a byte copy of
`$J/kernel/a2r_hl/score.sbatch` (md5 8e238045, the A2-RERUN copy of the CHAR-PROTOCOLS scorer), rows in
`$J/kernel/udrv_hl/score_rows.tsv`, output `$J/evals_udrv/`. The score rows carry each lane's own extra, so inference reads
`U_inf_dir` as training did. Cells: 200,000 for GeoTransolver and ISLA; 100,000 for Transolver, its training resolution, as
the CHAR fleet rule scored the `_c100k_` REF-80 lanes. Reduction: `udrv_table.py` here, which reduces like
`char_a2_table.py` (infer_summary, else the mean of the per-sample rows, last row per sample id).

The ISLA comparator is scored through the same scorer copy with the same rows format (rows 16-22 of score_rows.tsv), unless
the A2-RERUN program has already scored those runs with identical rows, in which case its scores are used.

## Record (physical drive, CHAR, seed means; per seed 42 / 43)

| arm | native p | native tau | area p | posed p |
|---|---|---|---|---|
| ISLA 512 x 512 slices (old arch) | 0.0407 (0.0402 / 0.0411) | 0.0933 | 0.0412 | 0.0411 |
| GeoTransolver 256, U_inf | 0.0522 (0.0552 / 0.0492) | 0.1233 | 0.2215 | 1.7095 |
| Transolver 512 x 16, U_inf | 0.0434 (0.0435 / 0.0433) | 0.1009 | 0.3956 | 1.0144 |

## Predictions (from the UDRV entries)

- P1, native pressure: both unit-drive baselines improve on their physical-drive selves on most cases (UDRV: 510 cases,
  median per-case factor 0.62 GeoTransolver, 0.67 Transolver). Here: GeoTransolver below 0.0522 by at least 10 % (expected
  0.035-0.045); Transolver below 0.0434 (expected 0.033-0.041). The physical drive cost REF-80 GeoTransolver less than it cost
  the ladder's 10,000-cell lanes (0.0522 is 1.28x ISLA, against 2.2x at 210 cases), so the gain may be smaller than the
  ladder's.
- P2, against A2 ISLA, native pressure: each baseline ÷ A2 ISLA (seed means) at or below 1.00; the UDRV ladder put the
  ratio at 0.84 (GeoTransolver) and 0.90 (Transolver) at 510 cases against the then-ISLA. "Baseline ahead" if the ratio is
  ≤ 0.95 with the baseline lower on at least 2/3 of the 180 cases (paired on seed means); "parity" within ±5 %; "ISLA ahead"
  if ≥ 1.05. Wall shear is expected to follow pressure.
- P3, area and posed: the unit drive does not make the baselines discretization- or pose-invariant. Their area and posed
  errors stay several times their native error (record: area 4-9x, posed 23-33x); ISLA's posed ÷ native stays within
  0.002 of 1 (R-POSE of A2-RERUN).

## Decision use

No adoption rule. This corrects the HiLift standing of the two baselines in the CHAR record: once scored, the unit-drive
rows replace the physical-drive rows as the HiLift baseline standing, with the physical-drive rows kept as the record of
what REF-80 measured. REF-80's selection of the baseline configurations is not reopened here (the drive bias applied to
every HiLift baseline candidate, so it may have affected the selection too; that is a separate question for the program).

## Acceptance of each lane at its first link

At least 100 training steps, finite loss, no fatal traceback, and the override in the logged config
(`global_embedding: global_data.U_inf_dir` or `source: global_data.U_inf_dir` in the train.log Config block).
