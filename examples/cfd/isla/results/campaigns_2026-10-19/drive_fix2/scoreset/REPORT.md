Status: DONE

# T-SCORESET (DRIVE-FIX2 Part B): Transolver 512x16 on DrivAerML Surface re-scored at 200,000 cells (2026-10-19, 12:05–13:20 EDT)

**Readout.** At 200,000 cells, Transolver 512x16 scores on exactly ISLA A2's set (hash-checked, 48/48 cars). Its native
errors are within 0.4 % of the 100k scores: the book's reference gives pressure **0.04352**, against 0.04368 at 100k,
and wall shear **0.06961**, against 0.06961. Posed changes by at most 0.14 % per seed (0.06 % on the seed means).
**Area** falls by 12–21 %, because the area protocol depends on how many cells are scored. Against ISLA A2 at 200k
(0.0445 / 0.0700), Transolver native is 0.978 / 0.994 (reference) and 0.951 / 0.969 (unit drive, r80ud).

## What ran (LHR, GB300)

| job | what | QoS | result |
|---|---|---|---|
| 283989 | setcheck GROUP=dr (DrivAerML Surface Transolver, 200k, with and without fx.source=U_inf_dir) | cpu / cpu-short | 6/6 rc 0 |
| 283990 | setcheck GROUP=suv (SHIFT-SUV Transolver vs ISLA A2) | cpu / cpu-short | 6/6 rc 0 |
| 284005 | setcheck GROUP=vol (DrivAerML Volume Transolver vs ISLA A2) | cpu / **cpu-normal** (cpu-short is capped at 4 jobs per user) | 4/4 rc 0 |
| 284205_[0-1] | score.sbatch, rows 0–7 | batch / normal | COMPLETED, 2:39–3:05 per block |
| 284387_[2] | score.sbatch, rows 8–11, after the seed-43 copy finished | batch / normal | COMPLETED, 2:33 |

- Submitted with `cd $T/kernel/tset200k && sbatch -q normal --array=<b> score.sbatch`, and the setchecks with
  `sbatch [-q cpu-normal] --export=ALL,GROUP=<dr|suv|vol> setcheck.sbatch`. All 12 rows have a `.done` file, 48 cars
  each, epoch 500, and each START line says `cells=200000`.
- Each slurm log has a harmless `[: : integer expression expected` line. It comes from the inherited 3-attempt check
  on a log that does not exist yet, and the same line is in r80_ud's scorer.
- No git commits. No existing shared script or table was touched.

## New files on LHR (`T=/scratch/fsw/.../agents/2026-08-09-mt2-stage0`); local copies are in `partB/cluster/tset200k/`

| file | md5 | notes |
|---|---|---|
| kernel/tset200k/score.sbatch | 5eee9ca4 | copy of r80_ud/score.sbatch d5d16eef |
| kernel/tset200k/score_rows.tsv | a1e912f3 | rows 0–11 |
| kernel/tset200k/scoredset_check.py | e611a919 | byte copy of the globe_verify script |
| kernel/tset200k/setcheck.sbatch | e37f83c6 | adapted from globe_verify's dd166f0a |
| evals_tset200k/<run>/<p>/ | | scorer output |
| kernel/tset200k/scoredset/*.jsonl | | per-case hashes |
| runs_from_aga/r80_dr_t_h512_l16_c100k_seed{42,43}/ | | the 3 checkpoint files, train.log, resolved_config.yaml |

- **Changes in score.sbatch:**
  - job name `isla.tset200k.score`;
  - A2R_ROOT default `kernel/tset200k`;
  - EVAL_ROOT default `evals_tset200k`;
  - a header comment;
  - an optional 11th rows column `runs_root` (relative to TASK, default `runs`). It is used for the train.log check and
    for `checkpoint_dir`. The diff is in `cluster/tset200k/`.
- **Rows:**
  - 0–5 are r80_ud's rows with cells=200000;
  - 6–11 are r80_dr_t seeds 42/43, native/area/posed, with the same model, code_v20 and recipe_v8, the reference extra
    `model.n_hidden=512 model.n_layers=16`, and runs_root=runs_from_aga.
- **Copy of the checkpoints (task 2):**
  - Relayed through this machine via the datamovers aga1dc and lhr1dc. The aga1 login node ran at 90 KB/s, so the
    first attempt was restarted.
  - md5s agree for all 10 files across AGA, local and LHR: seed 42 mdlus f553225d, pt 61f4784c; seed 43 mdlus
    0b0518e1, pt 8506e86d; norm_stats 1f6c7280 (both seeds). Listings are in `md5_local.txt` and `md5_lhr_r80dr.txt`.
  - The local temp copy has been deleted.
  - The infer log confirms it loaded `runs_from_aga/.../Transolver.0.500.mdlus` (epoch 500).

## Task 1: inference fits at 200k

- AGA (GB300) already scored cl_r_suv_t and cl_r_vol_t (512x16, float32) at 200k on all three protocols.
- My 12 rows ran on GB300 nodes nvl72d048, d050 and d031, with rc 0 and about 1 min per row.
- infer.py does not log GPU memory, and sacct records only host RSS (a maximum of about 28 GB per 4-row task).
- Estimate: 200k × 512 × 4 B is about 0.4 GB per activation, against about 277 GiB per GPU. The fit is not in doubt.

## Cell counts: every Transolver lane the book uses, against ISLA A2 (from START lines and score rows)

| dataset | Transolver lane | scored cells | ISLA A2 lane | A2 cells | same? |
|---|---|---|---|---|---|
| DrivAerML Surface | r80_dr_t_h512_l16_c100k (book reference), r80ud_dr_t_h512_l16_c100k | 100k | a2r_dr_isla (JHB) | 200k | **no**, fixed here (200k rows) |
| DrivAerML Surface | r80_dr_t_h1024_c100k; r80_dr_t_h512, _h256 | 100k; 200k | a2r_dr_isla | 200k | h1024_c100k: **no**; the others yes |
| HiLift | r80_hl_t_h512_l16_c100k, r80_hl_t_h1024_c100k; r80_hl_t_h512, _h256 | 100k; 200k | a2r_hl_isla | 200k | the c100k lanes: **no** (listed only, not scored) |
| SHIFT-SUV | cl_r_suv_t (trained at 100k) | 200k | a2r_suv_isla | 200k | yes, confirmed |
| DrivAerML Volume | cl_r_vol_t (trained at 100k) | 200k | a2r_vol_isla (JHB) | 200k | yes, confirmed |
| AirfRANS AF-UNIFORM | afu_r_t, afu_ud_t | 100k | afu_a2s_isla | 100k | yes |
| AirfRANS CHAR | cl_r_af_t | 100k | (no A2 lane; cl_r_af_isla) | 100k | yes |
| lug | cl_r_lug_t | 200k (whole part) | a2r_lug_isla | 200k | yes |
| Tensile2D | cl_r_t2dz_t | 200k | a2r_t2dz_isla | 200k | yes |
| DeepJEB++ | cl_r_jeb_t | 200k | jbn2_isla | 200k | yes |
| older CHAR (n_hidden 1024) | cl_{af,vol,suv,t2d,t2dz,lug,jeb}_t | 200k | cl_*_isla (not A2) | 200k | yes |

For lug, Tensile2D and DeepJEB++ only the cell count was checked, not the hashes.

## Scored-set identity (task 4): `scoredset_compare.json`, from `compare_sets.py`

| set | Transolver digest | ISLA A2 digest | points + targets match | boundaries match |
|---|---|---|---|---|
| DrivAerML Surface native, 200k (with and without U_inf_dir) | 52c7f2b8 | 52c7f2b8 | 48/48 | 48/48 |
| DrivAerML Surface area, first 12 (both arms) | ebe7c219 | ebe7c219 | 12/12 | 12/12 |
| DrivAerML Surface posed, first 12 (both arms) | a55c3bf0 | a55c3bf0 | 12/12 | 12/12 |
| DrivAerML Surface native at 100k (the old CHAR set) | 3d60cb6f | 52c7f2b8 | 0/48 | 0/48 |
| SHIFT-SUV native (all 99) / area (12) / posed (12) | 983a2c64 / 8df5b101 / 9cf4b6a6 | the same | 99/99, 12/12, 12/12 | all |
| DrivAerML Volume native (48) / posed (12) | 87c9e0b4 / 9afa2b2a | 56444d2f / 7e15aa46 | **48/48, 12/12** | 0 (see below) |

- The Volume digests differ only because ISLA's loader also carries five boundary meshes (inlet, no_slip, outlet, slip,
  vehicle), which Transolver's does not read. The scored points and targets are identical.
- The A2 Volume arm used LHR's code_a2r and recipe_a2r. They are md5-identical to JHB's, where the A2 lanes ran:
  code_a2r/MD5SUMS 0c4445ea, the vol yamls 538e2f57 / 6a017b48 / 72a81bf7, datasets.py 48a06ae5.

## 100k vs 200k (task 5): `reduce_tset200k.py`, giving `tset200k_scores.json` and `reduce_tset200k.out`

All 24 files have an infer_summary row.

| lane | protocol | field | 100k s42 / s43 / mean | 200k s42 / s43 / mean | 200k/100k mean (s42, s43) |
|---|---|---|---|---|---|
| r80_dr_t (reference) | native | pressure | 0.04384 / 0.04351 / **0.04368** | 0.04366 / 0.04338 / **0.04352** | 0.9964 (0.9957, 0.9970) |
| r80_dr_t | native | wss | 0.06994 / 0.06929 / **0.06961** | 0.06993 / 0.06930 / **0.06961** | 1.0000 (0.9999, 1.0001) |
| r80_dr_t | area | pressure | 0.28162 / 0.27588 / 0.27875 | 0.22196 / 0.22950 / 0.22573 | 0.8098 (0.7882, 0.8319) |
| r80_dr_t | area | wss | 0.36528 / 0.34828 / 0.35678 | 0.30687 / 0.30102 / 0.30395 | 0.8519 (0.8401, 0.8643) |
| r80_dr_t | posed | pressure | 0.99420 / 0.97403 / 0.98412 | 0.99375 / 0.97535 / 0.98455 | 1.0004 (0.9995, 1.0014) |
| r80_dr_t | posed | wss | 1.14028 / 1.18855 / 1.16442 | 1.14018 / 1.19005 / 1.16511 | 1.0006 (0.9999, 1.0013) |
| r80ud_dr_t (unit drive) | native | pressure | 0.04225 / 0.04226 / **0.04226** | 0.04231 / 0.04236 / **0.04234** | 1.0019 (1.0014, 1.0025) |
| r80ud_dr_t | native | wss | 0.06741 / 0.06779 / **0.06760** | 0.06759 / 0.06806 / **0.06782** | 1.0033 (1.0027, 1.0039) |
| r80ud_dr_t | area | pressure | 0.34787 / 0.34186 / 0.34486 | 0.29397 / 0.28423 / 0.28910 | 0.8383 (0.8450, 0.8314) |
| r80ud_dr_t | area | wss | 0.41309 / 0.43171 / 0.42240 | 0.36524 / 0.37408 / 0.36966 | 0.8751 (0.8842, 0.8665) |
| r80ud_dr_t | posed | pressure | 1.29998 / 1.28659 / 1.29328 | 1.30029 / 1.28529 / 1.29279 | 0.9996 (1.0002, 0.9990) |
| r80ud_dr_t | posed | wss | 1.40114 / 1.30696 / 1.35405 | 1.40212 / 1.30692 / 1.35452 | 1.0004 (1.0007, 1.0000) |

## What it means, and how sure I am

- **Native is unchanged (high confidence).** Moving to ISLA's scored set changes native by −0.4 % to +0.3 %, which is
  less than the seed spread. The book's Transolver native numbers, and the DRIVE-FIX R2 rule result (pressure 0.967,
  outside the band), stand.
  - On the 200k set, the R2 ratios r80ud ÷ reference are 0.9729 (pressure) and 0.9743 (wss), both inside
    [0.97, 1.03]. At 100k the pressure ratio was 0.9674.
  - So the R2 outcome depends on which scored set is used. The rule was registered at 100k (PREREG). Which set
    applies is the supervisor's call, so I flag it and do not decide it.
- **Area depends on the cell count,** at about 0.81–0.88× for 200k against 100k. Any Transolver area figure in the
  book should be taken from the 200k rows.
- **Posed** is broken (about 1.0–1.3) at either count.

## Rulings relied on, and items for project memory

- Relied on: the brief's approval of the small checkpoint transfer; normal QoS for GPU work, no short QoS; CPU for
  the setchecks.
- To record:
  - Transolver c100k lanes on DrivAerML Surface and HiLift were scored at 100k, while ISLA A2 was scored at 200k. The
    200k DrivAerML Surface scores of these two lanes (r80_dr_t_h512_l16_c100k, r80ud_dr_t_h512_l16_c100k) are now in
    `$T/evals_tset200k`.
  - SHIFT-SUV, DrivAerML Volume, AirfRANS, lug, Tensile2D and DeepJEB++ are each scored at the same cell count as
    ISLA A2. Hash-identical sets were confirmed for SHIFT-SUV and Volume.

## Left

- HiLift r80_hl_t_h512_l16_c100k (and h1024_c100k) at 200k. It is not scored, by the brief: the checkpoints are on AGA
  and the data reaches LHR this evening.
- r80_dr_t_h1024_c100k is also a 100k DrivAerML lane that the book cites. It is not re-scored here.
- Hash checks for lug, Tensile2D and DeepJEB++, if wanted.
- Nothing is still running.
