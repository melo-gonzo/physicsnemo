# HL090 pre-registration (2026-10-20, written before submission): ISLA 0.9.0 on HiLift, checked against A2

**Scope.** This is a separate HiLift check. REF-090's pooled R1 (scratch/isla-ref090-2026-10-20/PREREG.md §§6, 11) was
registered without HiLift, and one of its datasets is already scored, so R1 stays as it is and HiLift does not enter it.

## 1. Arms

| arm | runs | code / recipe | where |
|---|---|---|---|
| ISLA 0.9.0 | h090_hl_isla_seed42, h090_hl_isla_seed43 (`$T/kernel/h090/lanes.tsv` rows 0-1) | `$T/code_h090` (psharpe/isla 75ac723d8 + td0152 tensordict 0.15.2; MD5SUMS md5 b12efc3f) / `$T/recipe_h090` (MD5SUMS md5 3751a23c) | LHR, GB300 x 4, bf16 |
| A2 partner | a2r_hl_isla_seed42, a2r_hl_isla_seed43 (kernel/a2r_hl rows 8-9) | code_a2r / recipe_a2r_hl | JHB epochs 0-285, then LHR |

The rows are the partner's rows with only `run_id` and `code` changed (recipe through site.env). The fixed settings:
`highlift_surface_medium`, `isla_surface_highlift`, `+training.val_interval=25 model.hidden=512 model.n_slices=512
model.compile_blocks=true`, lr 3.75e-4, 200,000 cells, 500 epochs, `precision=bfloat16 augment=false compile=false
training.save_interval=2`. Evidence that the arm runs: scratch/isla-hilift090-2026-10-20/partB/PARTB.md (jobs 299686,
299755: 64.5 GiB, warm step 0.34 s, the same as A2).

## 2. Scoring

The same scorer, protocols and set as a2r_hl:
- `infer.py`, float32, `infer_split=val` (180 cases), ISLA's 200,000-cell set, the epoch-500 checkpoint;
- native, area (`highlift_surface_medium_area`) and posed (`highlift_surface_medium_posed`);
- `$T/kernel/h090/score.sbatch` (kernel/r090's, with defaults kernel/h090 and evals_h090), rows `score_rows.tsv`
  (the a2r_hl rows with run_id, code and recipe replaced). The command is in JOBS.md.

Before reading, check that the `config` row says `"epoch": 500` and that 180 `infer_step` rows exist per protocol.
Reduction: the per-field mean over cases of the per-case relative L2 (last row per sample id), then seed means over 42
and 43, as in `scratch/isla-udrv-hl-2026-10-18/udrv_table.py`.

Partner values (native; seed mean, then seeds 42 and 43): pressure 0.0432 (0.0435, 0.0428), wall shear 0.0943 (0.0944,
0.0942).

## 3. Rule R-HL090 (the verdict)

On pressure and wall shear, with native seed means:
- G = (0.9.0 seed mean) ÷ (A2 seed mean);
- per seed, r_s = 0.9.0 seed s ÷ A2 seed s, for s = 42, 43;
- "both seeds agreeing" means r_42 and r_43 are both on the stated side of the stated bound.

| verdict | condition |
|---|---|
| **level** | G within 0.95-1.05 on both fields |
| **better** | G ≤ 0.95 on both fields, with both seeds agreeing on each (r_s ≤ 0.95) |
| **worse** | G ≥ 1.05 on either field, with both seeds agreeing on that field (r_s ≥ 1.05) |
| **within seed spread** | anything else |

The rules are checked in the order worse, better, level. "Within seed spread" also covers a mixed outcome, for example
pressure 0.94 with wall shear 0.99. That is the likely result if only the three AoA-16 cases flip, since they weigh
about 6 % on pressure and about 1 % on wall shear.

**R2-style.** The h090 rows become HiLift's ISLA standing in the book, whatever R-HL090 gives. The A2 rows stay in the
record as the partner.

## 4. Predictions (probabilities registered now)

Case thresholds, on seed-mean pressure relative L2: **wrong** if ≥ 0.15, **right** if ≤ 0.10, **between** otherwise.
On these cases A2's two seeds lie at 0.17-0.29. When the old 512/512 reference (r80_hl_isla_w512_s512) was right, it lay
at 0.04-0.14, and GeoTransolver at 0.04-0.07.

| # | prediction | probability |
|---|---|---|
| P1 | R-HL090 = **level** | 0.60 |
| P1' | the other verdicts: within seed spread 0.20, worse 0.10, better 0.10 | — |
| P2 | all three AoA-16 cases LHC153_16, LHC162_16, LHC119_16 stay **wrong** (A2: 0.278, 0.248, 0.208) | 0.55 |
| P2' | each one stays wrong | 0.75 |
| P3 | LHC002_AoA_18 and LHC132_AoA_22 both stay **wrong** (wrong on all 4 ISLA runs so far, A2 and old; A2 0.219, 0.190) | 0.75 |
| P4 | posed ÷ native **persists**: pressure in 1.003-1.012 on both seeds and wall shear in 1.002-1.010 on both seeds (A2: 1.0090 / 1.0089 and 1.0055 / 1.0061) | 0.70 |
| P4' | it **disappears** (\|posed/native − 1\| ≤ 0.002 on both fields and seeds) 0.15; it **changes otherwise** (grows, or moves outside the bands) 0.15 | — |
| P5 | the unit-drive baselines stay ahead of 0.9.0 on native pressure: 0.9.0 seed mean > 0.0394 (Transolver) and > 0.0390 (GeoTransolver) | 0.80 |
| P6 | the unit-drive baselines stay ahead of 0.9.0 on native wall shear: 0.9.0 seed mean > 0.0890 (GeoTransolver) and > 0.0878 (Transolver) | 0.80 |
| P7 | under area and posed, ISLA 0.9.0 stays ahead of both baselines by ≥ 3x on pressure (baselines: area 0.137-0.168, posed 0.87-0.95) | 0.95 |

Reasoning:
- On HiLift's surface path, 0.9.0 is A2 on the same 200k tokens plus inputs by name, the in-model unit direction and
  the length unit with per-cell second moments.
- The H-RESID A2 reads are in 0.9.0 (7e167820a is an ancestor of 75ac723d8).
- HiLift's means are set by about 10 post-stall branch cases (scratch/isla-hilift090-2026-10-20/RESULTS.md, Part A).
  Some of them flip from seed to seed, so a level verdict is likely but not certain.

## 5. What would change the program

- Level or within seed spread, with P2 holding: 0.9.0 replaces A2 as the standing (R2), and the HiLift gap to the
  baselines stays a stall-onset problem to study separately.
- The three AoA-16 cases right on both seeds, with the overall mean near the old reference's 0.0407: points to the gauge
  or the input changes. Not the reads, which are unchanged.
- Worse: stop and report before anything is written into the book.
