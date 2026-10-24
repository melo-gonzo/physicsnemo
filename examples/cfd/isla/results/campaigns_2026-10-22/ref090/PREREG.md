# REF-090 pre-registration: ISLA 0.9.0 against its A2 partners, and the compliant AeroZoo ISLA / ISLA-DTT lanes

Written 2026-10-20 from 02:09 EDT by research-lane REF-090, **before any training lane was submitted** (the only jobs before
approval are the 50-step smokes of §8, which produce no scored number). **Approved by the supervisor at 02:58 EDT, before
any training lane was submitted, with the changes recorded in §11 (Amendment before submission), which governs where the text
above it says otherwise.**

`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0` (LHR).
Evidence behind every number quoted from earlier work: `evidence/REPORT.md` and `evidence/partners.json` beside this file.

## 1. Question

ISLA 0.9.0 (psharpe/isla 92d9d7800) is the adopted reference: the exact full-mesh gauge, boundary conditions by name, exact
query geometry with the CONT encoding, ISLA's own scales (declared on AeroZoo and AirfRANS, per-split constants in the dataset
files under `isla_constants`, injected by train.py and infer.py), and upstream/main merged. (a) Does it confirm its adoption
against each A2-RERUN partner on that partner's protocol? (b) Its rows become the book's ISLA standings. (c) How do its posed
scores compare with native? (d) On AeroZoo, under the compliant protocol, where do ISLA and ISLA-DTT stand (PREREG_v2).

## 2. Code and launcher (all new; no shared file edited; nothing running was touched)

| item | what | md5 |
|---|---|---|
| `$T/code_r090` | `git archive 92d9d7800 physicsnemo pyproject.toml test` (tar md5 47bbc187) + `tensordict/` and `tensordict-0.15.2.dist-info/` from `$T/../2026-10-15-tensordict-015/td0152` (the unzipped 0.15.2 wheel, sha256 de06ae27…b0d8; no `__pycache__`). 3,248 files; `md5sum -c` passes; `diff -r` against the extracted archive: only the two tensordict dirs, MD5SUMS and SNAPSHOT_HEAD. chmod a-w. | MD5SUMS c1227687, SNAPSHOT_HEAD ccdcc32d |
| `$T/recipe_r090` | `git archive 92d9d7800:examples/cfd/external_aerodynamics/unified_external_aero_recipe` (tar md5 877445e7) + the 11 cluster-only files of recipe_az_c (found by diffing recipe_az_c against `git archive caac441d7` of the recipe: `datasets/dataset_paths.yaml`, the only differing file, and `datasets/{ginot_lug_volume,plaid_tensile2d_2d,highlift_surface_medium}{,_area,_posed}.yaml`, `drivaer_ml_surface_area.yaml`) + **`datasets/deepjebpp_surface{,_area,_posed}.yaml`** (not in recipe_az_c; byte-identical in recipe_v8, the jbn2 partner's recipe, and recipe_hn; needed for the DeepJEB++ arm). 133 files; `md5sum -c` passes; `diff -r` against the archive: exactly these 14 files plus MD5SUMS and SNAPSHOT_HEAD. chmod a-w. | MD5SUMS bb4ee858, SNAPSHOT_HEAD 1d4764ef |
| `$T/kernel/r090/` | REF-090 lanes `r090_*`, chains `isla.r090.<slug>`. `lanes.sbatch` and `ckpt_preflight.py` byte copies of kernel/azp (8eecec8f, e34bd611: the a2r launcher with `precision="${LANE_PRECISION:-bfloat16}"`); `site.env` = azp's with RECIPE → recipe_r090, LANE_PRECISION=bfloat16; `submit.sh` = azp's with the prefix isla.r090.; `score.sbatch` = kernel/a2r/score.sbatch (8e238045) with job name isla.r090.score and default A2R_ROOT / EVAL_ROOT kernel/r090 / evals_r090 | lanes.tsv 81d34e1d, site.env 4a6ed5fb, submit.sh 7f559159, score.sbatch a4a24f5e, score_rows.tsv 2104fd1e |
| `$T/kernel/azi/` | AeroZoo ISLA / ISLA-DTT lanes `azc_<split>_{isla,dtt}_seed4x`, chains `isla.azc.<slug>`. lanes.sbatch, ckpt_preflight.py byte copies of kernel/azp; `site.env` as kernel/r090's; `submit.sh` = azp's (same prefix isla.azc.; header line only); `eval_test.sbatch` = kernel/azp's (08ca3414) with CODE/RECIPE code_r090/recipe_r090, LANES kernel/azi/lanes.tsv, job name isla.azc.test-eval-isla, the same ROOT p200k_fp32_test_azc | lanes.tsv 11d10f9b, site.env c9006c02, submit.sh 82423cee, eval_test.sbatch d209c3de |

Venv `physicsnemo-mesh-transformer/.venv-recipe` (torch 2.12.0+cu130), tensordict 0.15.2 from the snapshot (first on
PYTHONPATH). Training command (lanes.sbatch, unchanged): `torch.distributed.run --nproc-per-node 4 train.py model=<m> <extra>
dataset=<d> sampling_resolution=<cells> run_id=<run> training.seed=<s> training.num_epochs=<E> training.optimizer.lr=3.75e-4
training.save_interval=2 logging.log_every_n_steps=50 precision=bfloat16 augment=false compile=false`.

## 3. REF-090 arms and lanes (`kernel/r090/lanes.tsv`; idx 2k seed 42, 2k+1 seed 43)

Each row is its A2 partner's row with only `code` (→ code_r090) and the recipe (→ recipe_r090) replaced, checked
field by field against the partner tables on LHR (dataset, model, extra, lr, seed, cells, epochs: no difference except as noted).

| idx | run (seed 42 / 43) | dataset (config) | model | cells (train) | epochs | A2 partner (table) | differs from the partner row |
|---|---|---|---|---|---|---|---|
| 0-1 | r090_suv_isla | SHIFT-SUV (shift_suv_surface) | isla_surface | 200k | 500 | a2r_suv_isla (kernel/a2r rows 0-1) | – |
| 2-3 | r090_t2dz_isla | Tensile2d (plaid_tensile2d_2d, train_400 / val_100) | isla_tensile2d | 20k | 500 | a2r_t2dz_isla (2-3) | – |
| 4-5 | r090_lug_isla | GINOT lug (ginot_lug_volume, train / test) | isla_lug | 20k | 500 | a2r_lug_isla (4-5) | – |
| 6-7 | r090_dr_isla | DrivAerML Surface (drivaer_ml_surface) | isla_surface | 200k | 500 | a2r_dr_isla (10-11; trained on JHB) | runs on LHR (same data, manifest md5 4f49e562 on both; both GB300) |
| 8-9 | r090_vol_isla | DrivAerML Volume (drivaer_ml_volume_reference) | isla_volume | 200k | 500 | a2r_vol_isla (12-13; JHB) | runs on LHR |
| 10-11 | r090_jb_isla | DeepJEB++ (deepjebpp_surface, train_clean / test_clean) | isla_surface_deepjebpp | 200k | 40 | jbn2_isla (kernel/deepjeb_bn rows 0-1; A2-RERUN scored 0.3919 / 0.1653) | **+ `model.compile_blocks=true`** (below) |
| 12-13 | r090_af_isla | AirfRANS (airfrans_uvolume_2d, full_train / full_test) | isla_volume_airfrans | 100k | 500 | afu_a2s_isla (kernel/airfrans_u rows 10-11) | – ; **not launched until the supervisor says** |
| 14-15 | r090_af_dtt | AirfRANS, ISLA-DTT | isla_volume_airfrans | 100k | 500 | afu_a2s_isla row + `++model.query_independent=true` | secondary arm; **held as 12-13** |
| 16-17 | r090_lug_dtt | lug, ISLA-DTT | isla_lug | 20k | 500 | a2r_lug_isla row + `++model.query_independent=true` | secondary arm |

Not run: HiLift (needs a decimated geometry asset). Common to every row: width 512, 512 slices, lr 3.75e-4, bf16 (ISLA's head
and redimensionalization in fp32 where its scales are on), augment=false, save_interval=2, the partner's splits and val_interval.

**Compile.** Every lane passes `compile=false` (the launcher) and `model.compile_blocks=true`: per-block `torch.compile` of
the twelve slice blocks, in torch's default mode, as every A2 partner ran. The configs' whole-model `compile_options`
(`max-autotune-no-cudagraphs`) are not used by any lane, nor were they by the partners.
- DeepJEB++: jbn2 ran on recipe_v8, whose `isla_surface_deepjebpp.yaml` set `compile_blocks: true`. At 92d9d7800 the
  config sets whole-model compile and `compile_blocks: false`, and the launcher passes `compile=false`. Without the flag the
  lane would run uncompiled. Adding it reproduces the partner's effective setting, as A2-RERUN did for its rows.
- AirfRANS: the Inductor backward failure under `max-autotune-no-cudagraphs` (lane 7d131d48) is in the whole-model compile.
  The partner-copy rows never enter it, so they may be launchable as they stand. They are held anyway, per the brief, until
  the supervisor names the code or mode. They are smoked as written, which is per-block compile in default mode.

**What R1 measures.** 0.9.0 against A2 is a bundle: the full-mesh gauge, conditions by name, the exact query geometry with
CONT, the scales (AeroZoo, AirfRANS only), upstream/main and the code drift since each partner's snapshot (code_a2r,
code_a2s, code_v20a2), and the recipe drift. It is read as the reference's standing move, not as one change's effect.

## 4. AeroZoo compliant ISLA and ISLA-DTT lanes (`kernel/azi/lanes.tsv`; PREREG_v2 §4, amended there 2026-10-20)

28 rows: idx 0-13 ISLA, 14-27 ISLA-DTT (`++model.query_independent=true`); within each arm the splits full, nonbox,
reynolds, blobs, bank_zero_to_cad, profile_crossflow, profile_dual_inlet, seeds 42 / 43. Dataset `aerozoo_uvolume{,_nonbox,
_reynolds,_blobs,_bank_zero_to_cad,_profile_crossflow,_profile_dual_inlet}`, model `isla_volume_aerozoo` (0.9.0: ISLA's
own scales, the split's `isla_constants` injected; case_relative_l2 with velocity weight 3; `compile: false`,
`compile_blocks: true`), extra the azc baseline rows' (`train_split=<s>.train val_split=<s>.val +training.val_interval=10
dataloader.num_workers=12 dataloader.prefetch_factor=12`), lr 3.75e-4, 250 epochs, 200,000 cells, code_r090, bf16.
The old ISLA-DTT knobs `kernel_read*` no longer exist at 0.9.0; ISLA-DTT is `query_independent=true` alone.
Readouts, predictions and the ISLA-vs-ISLA-DTT rule: PREREG_v2 §§7-8 as amended in §11 and §12.

## 5. Scoring (float32, final checkpoint, each lane on its own snapshot code_r090 / recipe_r090)

Same scored sets and protocols as each partner: infer.py, `infer_split=val` (the partner's validation split), native /
area / posed (the dataset's `_area` / `_posed` twins), 200,000 cells on DrivAerML Surface and Volume, SHIFT-SUV, DeepJEB++,
and on Tensile2d and the lug (the partner's: A2-RERUN scored both at 200,000, the CHAR fleet rule), 100,000 on AirfRANS
uniform. Score trees and commands (run on lhr1 after the lane's "Training completed"; idempotent; one run per array task,
GPUs 0-2 for its three protocols):
```bash
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
cd $T/kernel/r090 && sbatch --array=<k,...> score.sbatch        # k = the lane idx (rows 4k..4k+2 of score_rows.tsv)
# -> $T/evals_r090/<run>/<native|area|posed>/<run>/metrics.jsonl, SNAPSHOT_USED, DATASET_USED, .done
cd $T/kernel/azi && sbatch --array=0-6 eval_test.sbatch          # AeroZoo: test side, 4 lanes per task, one GPU each
# -> $T/p200k_fp32_test_azc/azc_<split>_{isla,dtt}_seed4x/<run>/metrics.jsonl (infer_summary row), SNAPSHOT_USED, .done
```
AeroZoo: `<split>.test`, 200,000 interior points, the recipe's default score (the physical relative L2 per field and case,
averaged over cases), epoch 250; check the `config` row's `"epoch": 250` (and 500 / 40 for REF-090) before reading.
Reduction: seed means over 42 and 43 of the per-field case-mean relative L2 that the partner's table reports
(`scratch/isla-a2-rerun-2026-10-17/char_a2_table.py`, run unchanged on the fetched trees).

## 6. Rules (registered before any result)

**R1, adoption confirmation of 0.9.0 against each A2 partner (native protocol, seed means).** *Amended in §11: the verdict is
the pooled one; the per-dataset G_d are descriptive.*
- Fields: the low-noise headline fields per dataset (table in §6a, from CHAR-AREA2 and the A2-RERUN two-seed gaps). Known
  noisy and excluded from the verdict, reported beside it: Tensile2d stress and plastic strain, the lug's von Mises.
- Per dataset: ratio r_f = new seed mean ÷ A2 seed mean per low-noise headline field; per-seed ratios q_f,s = new seed s ÷
  A2 seed mean.
- **Confirmed** if the geometric mean of r_f over the dataset's low-noise headline fields is ≤ 1.03 **and** no low-noise
  field has r_f ≥ 1.10 with both seeds agreeing (q_f,42 ≥ 1.10 and q_f,43 ≥ 1.10). Otherwise **flagged for Peter**, with the
  table. A dataset whose only headline field is noisy (the lug) has no R1 verdict; its ratio is reported.
- Also reported: the geometric mean over every dataset's low-noise fields (one pooled number).
- **R1 cannot un-adopt 0.9.0.** The reference stays adopted on principle unless Peter vetoes; a flag only puts the table in
  front of him.
- False-alarm rate under the null (0.9.0 ≡ A2 in distribution), from the observed two-seed gaps: §6a.

**R2.** The new rows (r090_*, and for AeroZoo azc_*_isla / azc_*_dtt) become the book's ISLA standings on their datasets,
whatever R1 gives. ISLA-DTT on AirfRANS and the lug is a secondary arm, recorded beside ISLA, not a standing.

**R3, posed ÷ native per dataset** (seed means and per seed, every headline field). The interior task's boundary-token draw
follows the lab-axis Morton order of the cell centroids, so where the in-model draw subsamples (a condition with more cells
than its share of the 200,000-token budget) a pose changes the tokens drawn; the lug has medial ties (equidistant nearest
surfaces), where the exact direction flips under a pose. Predictions in §7.

### 6a. R1 fields, partner values and the false-alarm rate (from `evidence/REPORT.md` §§1-3; values per-case relative L2, native)

| dataset | R1 low-noise headline fields: A2 seed 42 / 43 (mean; two-seed gap) | excluded (noisy), reported | A2 posed ÷ native (s42 / s43) |
|---|---|---|---|
| SHIFT-SUV | pressure 0.04580 / 0.04545 (0.04562; 0.8 %), wall shear 0.06802 / 0.06729 (0.06766; 1.1 %) | – | 1.0000 / 1.0000 |
| Tensile2d | displacement 0.006794 / 0.006964 (0.006879; 2.5 %) | stress 0.00781 (20 %), plastic strain 0.4145 (22 %) | 1.06-1.12 on stress and displacement; q 1.01 / 1.00 |
| lug | none: no R1 verdict | von Mises 0.001602 / 0.001850 (0.001726; 14 %) | 4.14 / 2.87 |
| DrivAerML Surface | pressure 0.04358 / 0.04545 (0.04452; 4.2 %), wall shear 0.06835 / 0.07172 (0.07004; 4.8 %) | – | 1.0000 / 1.0000 |
| DrivAerML Volume | pressure 0.04132 / 0.04040 (0.04086; 2.2 %), velocity 0.06622 / 0.06570 (0.06596; 0.8 %), eddy viscosity 0.08947 / 0.08666 (0.08806; 3.2 %) | – | ≤ 1.0005 |
| AirfRANS (afu_a2s) | pressure 0.01854 / 0.01745 (0.01799; 6.1 %), velocity 0.009698 / 0.009214 (0.009456; 5.1 %), eddy viscosity 0.02326 / 0.02134 (0.02230; 8.6 %) | – | 0.999-1.000 |
| DeepJEB++ (jbn2) | von Mises 0.3942 / 0.3896 (0.3919; 1.2 %), displacement 0.1688 / 0.1619 (0.1653; 4.2 %) | – | 1.0000 / 1.0000 |

- Tensile2d displacement is low-noise on the A2 pair (2.5 %) but its other recorded pairs gap 5.6-21 % (evidence §2): its
  verdict is marked "contested field" when read.
- **False-alarm rate under the null** (0.9.0 ≡ A2 in distribution; `evidence/far.py`, 400,000 draws; per-seed log error
  N(μ_f, σ_f), σ_f = |log s42 − log s43| / √2 from the A2 pair; both the A2 and the new seed means are two-seed samples;
  datasets independent, ρ the within-dataset correlation of seed effects):

| σ source | ρ | SUV | T2D | DrvS | DrvV | AirfRANS | DeepJEB++ | P(≥ 1 of 6 flagged) | one pooled G over all 13 fields |
|---|---|---|---|---|---|---|---|---|---|
| A2 pair per field | 0 | 0.000 | 0.045 | 0.095 | 0.001 | 0.148 | 0.028 | **0.29** | **0.029** |
| A2 pair per field | 0.5 | 0.000 | 0.046 | 0.142 | 0.008 | 0.221 | 0.044 | 0.40 | 0.029 |
| pooled σ 0.029 | 0 | 0.075 | 0.155 | 0.075 | 0.040 | 0.039 | 0.076 | 0.38 | 0.002 |
| all recorded pairs | 0 | 0.000 | 0.377 | 0.050 | 0.009 | 0.127 | 0.027 | 0.50 | 0.083 |

  The observed A2 gaps have one sign within every dataset (P = 2^-9 under independence), so ρ > 0 is likely; the per-dataset
  rates rise with ρ, the pooled rate does not. Almost every flag comes from G > 1.03; the 1.10 clause adds ≤ 0.007.
  **Reading:** per dataset, the rule flags at least one of six datasets with probability about 0.3-0.5 when nothing changed
  (AirfRANS 0.15, DrivAerML Surface 0.10). A flag is therefore weak evidence by itself; it puts the table in front of
  Peter and cannot un-adopt 0.9.0. The pooled G (false-alarm rate about 0.03) is reported beside the per-dataset verdicts.

## 7. Predictions (seed means; probabilities are mine, from GEO1, GEO-CONT, the AeroZoo 5-epoch smoke; evidence §4)

Evidence in short: CONT ÷ A2 on AirfRANS p 0.875, v 0.913, nut 0.922; lug CONT ÷ A2 1.011; Tensile2d CONT ÷ CONTROL
displacement 0.956 (CONT ÷ A2: displacement 0.934, stress 1.076, q 0.887); GEO1 ÷ A2 AirfRANS p 0.87, v 0.94, nut 0.94.
Under CONT, posed ÷ native: Tensile2d 0.9999-1.0002, AirfRANS 1.000-1.0014, lug 1.05 / 1.06 (A2 4.14 / 2.87). The surface
tasks (SHIFT-SUV, DrivAerML Surface, DeepJEB++) have no query geometry and no in-model draw; for them 0.9.0 changes little
beyond code drift. DrivAerML Volume is an interior task whose 753,234-triangle body is drawn to 200,000 tokens in the lab-axis
Morton order (a pose changes the draw); AirfRANS (≈1,000 segments), Tensile2d and the lug keep their boundaries whole.

| id | prediction | p |
|---|---|---|
| P1 | R1 confirmed on SHIFT-SUV (G ≤ 1.03) | 0.92 |
| P2 | R1 confirmed on DrivAerML Surface | 0.85 |
| P3 | R1 confirmed on DrivAerML Volume; G ≤ 1.00 | 0.85; 0.50 |
| P4 | R1 confirmed on DeepJEB++ | 0.88 |
| P5 | Tensile2d displacement r ≤ 1.03 (R1 confirmed); r ≤ 0.97 | 0.75; 0.50 |
| P6 | AirfRANS (when launched): G(p, v, nut) ≤ 0.97; R1 confirmed | 0.70; 0.92 |
| P7 | lug von Mises (no verdict): new ÷ A2 in [0.85, 1.20] | 0.65 |
| P8 | Pooled G over all 13 low-noise fields ≤ 1.00 | 0.60 |
| P9 | R3: posed ÷ native within 1 ± 0.002 on every headline field and seed: SHIFT-SUV, DrivAerML Surface, DeepJEB++ | 0.90 each |
| P10 | R3: AirfRANS and Tensile2d posed ÷ native within 1 ± 0.005 (boundary kept whole) | 0.85 |
| P11 | R3: DrivAerML Volume posed ÷ native within 1 ± 0.01 on every field and seed; within 1 ± 0.002 | 0.85; 0.40 |
| P12 | R3: lug posed ÷ native ≤ 1.15 on both seeds (medial ties, 0.8 % of nodes flip under the pose) | 0.75 |
| P13 | Secondary: ISLA-DTT ÷ ISLA on the lug von Mises in [0.8, 1.25]; on AirfRANS G ≤ 1.05 | 0.6; 0.6 |
| P14 | All 14 REF-090 wave-1 lanes and all 28 AeroZoo lanes finish with finite loss and no BLOCKED marker | 0.85 |

## 8. Smokes (before launch; LHR short QoS, 1-2 jobs at a time)

Every distinct config (dataset × arm: 9 REF-090, 14 AeroZoo) runs 50 steps with its lane's exact training command, through
the lane launcher (`kernel/r090/smoke/smoke.sbatch`: the kernel's own site.env and lanes.sbatch, with `"${SMOKE_WRAP}"`
inserted before train.py, NO_CHAIN=1, smoke run dirs). Reported per config: first step, median step (after step 5), peak
memory, torch.compile graph and recompile counts, finite loss, the `SMOKE ISLA 0.9.0` line (scales on / off, fp32 head,
reynolds and output_stats injected), and the resolved config. AirfRANS rows are smoked in their partner setting (per-block
compile, default mode) for timing only.

### 8a. Smoke results (LHR short QoS jobs 297254 = REF-090, 297255 = AeroZoo; 23 units, every rc=0; seed-42 rows)

Steps are wall times between consecutive training forwards on rank 0 (forward, backward, optimizer and data); "first" includes
the per-block compile. Peak = max over the 4 ranks of torch.cuda.max_memory_allocated. Graphs / recompiles = torch.compile
unique graphs and rank 0's "Recompiling" lines (the slice-block forward: a static graph at the first token count, then one
dynamic-shape graph; DeepJEB++ and Tensile2d add one for an absent optional input). No graph breaks. Every unit printed
`ISLA 0.9.0`; finite outputs at calls 1-3 and finite per-field losses through step 50. The scaled models (AeroZoo, AirfRANS)
report scaled=True, fp32_head=True and their split's injected reynolds (e.g. AirfRANS {mean 6.205, std 0.1347}, z of case 1
1.298, L 0.4233, U 88.4 m/s); output dtypes float32. Logs: `$T/kernel/r090/smoke/logs/`, local copy `launcher/smoke/logs/`;
table `launcher/smoke/smoke_summary.json` (made by `launcher/smoke/summarize.py`).

| unit | first (s) | median / mean step after 5 (s) | peak (GiB) | graphs / recompiles | scales | loss total, mean of steps 1-10 → 41-50 | steps/epoch × epochs | wall h | GPU-h per lane |
|---|---|---|---|---|---|---|---|---|---|
| r090_suv_isla | 58 | 0.289 / 0.297 | 64.5 | 1 / 0 | off | 0.101 → 0.0631 | 198 × 500 | 9.0 | 36 |
| r090_t2dz_isla | 59 | 0.165 / 0.167 | 5.2 | 3 / 2 | off | 0.337 → 0.274 | 100 × 500 | 2.6 | 10 |
| r090_lug_isla | 76 | 0.171 / 0.172 | 8.5 | 2 / 1 | off | 0.433 → 0.303 | 600 × 500 | 15.8 | 63 |
| r090_dr_isla | 37 | 0.339 / 0.344 | 64.5 | 1 / 0 | off | 0.0913 → 0.0486 | 108 × 500 | 5.7 | 23 |
| r090_vol_isla | 56 | 0.664 / 0.700 | 128.7 | 1 / 0 | off | 0.0919 → 0.0627 | 108 × 500 | 11.6 | 46 |
| r090_jb_isla | 30 | 0.167 / 0.228 | 17.3 | 3 / 2 | off | 0.148 → 0.0294 | 12451 × 40 | 34.7 | 139 |
| r090_af_isla | 86 | 0.393 / 0.417 | 35.1 | 2 / 1 | on | 0.227 → 0.182 | 200 × 500 | 12.7 | 51 |
| r090_af_dtt | 41 | 0.580 / 0.580 | 39.4 | 2 / 1 | on | 0.292 → 0.184 | 200 × 500 | 17.7 | 71 |
| r090_lug_dtt | 40 | 0.452 / 0.454 | 9.9 | 2 / 1 | off | 0.437 → 0.341 | 600 × 500 | 41.6 | 166 |
| azc_full_isla | 59 | 0.765 / 0.873 | 126.6 | 1 / 0 | on | 1.19 → 1.47 | 678 × 250 | 45.2 | 181 |
| azc_nonbox_isla | 47 | 0.693 / 0.795 | 126.4 | 1 / 0 | on | 1.39 → 2.05 | 597 × 250 | 36.3 | 145 |
| azc_reynolds_isla | 50 | 0.705 / 0.808 | 126.6 | 1 / 0 | on | 1.31 → 1.33 | 529 × 250 | 32.7 | 131 |
| azc_blobs_isla | 37 | 0.704 / 0.807 | 126.6 | 1 / 0 | on | 1.13 → 0.97 | 539 × 250 | 33.2 | 133 |
| azc_bank_zero_to_cad_isla | 38 | 0.641 / 0.712 | 126.6 | 1 / 0 | on | 1.08 → 0.777 | 760 × 250 | 41.3 | 165 |
| azc_profile_crossflow_isla | 30 | 0.707 / 0.729 | 126.6 | 1 / 0 | on | 1.1 → 0.866 | 705 × 250 | 39.3 | 157 |
| azc_profile_dual_inlet_isla | 33 | 0.687 / 0.728 | 126.6 | 1 / 0 | on | 7.1 → 2.26 | 718 × 250 | 39.9 | 160 |
| azc_full_dtt | 44 | 1.320 / 1.350 | 171.4 | 1 / 0 | on | 1.3 → 1.32 | 678 × 250 | 69.9 | 280 |
| azc_nonbox_dtt | 51 | 1.212 / 1.208 | 162.7 | 1 / 0 | on | 1.47 → 3.61 | 597 × 250 | 55.1 | 220 |
| azc_reynolds_dtt | 48 | 1.366 / 1.377 | 171.2 | 1 / 0 | on | 1.48 → 1.62 | 529 × 250 | 55.6 | 223 |
| azc_blobs_dtt | 51 | 1.346 / 1.338 | 170.9 | 1 / 0 | on | 1.18 → 1.11 | 539 × 250 | 55.1 | 220 |
| azc_bank_zero_to_cad_dtt | 54 | 1.286 / 1.316 | 170.7 | 1 / 0 | on | 1.16 → 1.11 | 760 × 250 | 76.4 | 306 |
| azc_profile_crossflow_dtt | 55 | 1.335 / 1.357 | 170.6 | 1 / 0 | on | 1.17 → 1.01 | 705 × 250 | 73.1 | 292 |
| azc_profile_dual_inlet_dtt | 56 | 1.297 / 1.351 | 170.9 | 1 / 0 | on | 9.47 → 1.82 | 718 × 250 | 74.1 | 296 |

AeroZoo losses are the per-case relative L2 (velocity weight 3), about 1 per field at the start, with spikes on cases with
tiny targets (tke; known, shared by every arm). AirfRANS ran in its partner setting (per-block compile, default mode).

## 9. Cost (filled from the smokes in §9a)

GPU-h per lane = 4 × epochs × steps per epoch × smoke median step × 1.1 (validation, link restarts). Earlier estimates:
AeroZoo ISLA about 110 GPU-h per lane, ISLA-DTT about 230.

### 9a. Cost from the smokes (GPU-h per lane = 4 × epochs × steps/epoch × mean step × 1.1)

| group | lanes | GPU-h |
|---|---|---|
| REF-090 wave 1 (SUV 36, T2D 10, lug 63, DrivAerML S 23, V 46, DeepJEB++ 139, lug ISLA-DTT 167 per lane) | 14 | about 970 |
| REF-090 AirfRANS ISLA (51) and ISLA-DTT (71), held | 4 | about 240 |
| AeroZoo ISLA (131-181 per lane) | 14 | about 2,140 |
| AeroZoo ISLA-DTT (220-306 per lane) | 14 | about 3,680 |
| scoring | – | about 60 |
| **total** | 46 | **about 7,100** (±20 %) |

The AeroZoo lanes cost more than the earlier estimates (ISLA about 1.6×, 180 against 110 GPU-h on `full`; ISLA-DTT about
1.2×, 280 against 230): the 4-rank step is 0.71-0.87 s (ISLA) and 1.21-1.38 s (ISLA-DTT). Longest wall time per lane:
AeroZoo ISLA-DTT bank_zero_to_cad 76 h (about 20 links), ISLA `full` 45 h, DeepJEB++ 35 h, lug ISLA-DTT 42 h. All fit the
MARS budget (2,048 GPUs, 2 weeks) with 42 concurrent nodes.

## 10. Launch (after approval)

Normal QoS, 03:59:00 links, `--segment=1`, singleton chains as azp. Wave 1: the 14 REF-090 lanes minus AirfRANS (rows 0-11)
plus the lug ISLA-DTT (16-17), `cd $T/kernel/r090 && bash submit.sh lanes 0 11 && bash submit.sh lanes 16 17`. Wave 2, about
13 minutes later: `cd $T/kernel/azi && bash submit.sh lanes 0 27`. AirfRANS (r090 rows 12-15) on the supervisor's word.
Acceptance per lane after ≥ 15 minutes: finite loss; the resolved config (model, dataset, forward_kwargs, precision
bfloat16, lr, epochs, cells, seed); constants injected (the AeroZoo / AirfRANS models built with their split's
isla_constants); ISLA 0.9.0 in the log; step time near the smoke's.

## 11. Amendment before submission (2026-10-20, 02:58 EDT; the supervisor's approval, before any training lane was submitted)

- **Design deviations** §3 (a) DeepJEB++ `model.compile_blocks=true`, (b) the three deepjebpp dataset files in
  recipe_r090, (c) DrivAerML on LHR, (d) ISLA-DTT = `query_independent=true` alone: approved as written.
- **R1 is the pooled verdict** (the brief's wording: the geometric mean of new ÷ A2 over all low-noise headline fields).
  - **Confirmed** if the pooled G over all 13 low-noise headline fields (§6a: SHIFT-SUV 2, Tensile2d 1, DrivAerML Surface 2,
    DrivAerML Volume 3, AirfRANS 3, DeepJEB++ 2) is ≤ 1.03 **and** no low-noise field has r_f ≥ 1.10 with both seeds agreeing
    (q_f,42 ≥ 1.10 and q_f,43 ≥ 1.10). Otherwise **flagged for Peter**.
  - Each dataset's G_d is reported beside it, descriptively. Any dataset with G_d > 1.05 is listed for Peter with its
    per-dataset false-alarm rate from §6a (A2-pair σ, ρ 0 / 0.5: SUV 0.000 / 0.000, T2D 0.045 / 0.046, DrivAerML S 0.095 /
    0.142, V 0.001 / 0.008, AirfRANS 0.148 / 0.221, DeepJEB++ 0.028 / 0.044; these rates are for G_d > 1.03, an upper
    bound for G_d > 1.05).
  - The pooled false-alarm rate under the null: about 0.03 at the A2-pair σ (0.029; 0.002 at the pooled σ, up to 0.08 with
    every recorded pair's σ); the 1.10 clause contributes ≤ 0.007 (evidence §3).
  - R1 still cannot un-adopt 0.9.0.
  - With AirfRANS now in wave 1, the pooled verdict needs all 13 fields; it is read once AirfRANS is scored.
- **AirfRANS (r090 rows 12-15, ISLA and ISLA-DTT) launch with wave 1, as they stand:** per-block compile in default mode
  (compile=false, compile_blocks=true) is the partner's setting; the whole-model max-autotune failure is on a path these rows
  never take. The wave-1 command becomes `bash submit.sh lanes 0 17` (all 18 rows).
- **Known issue: torch 2.12's Inductor mix-order reduction** has a silent bug (pytorch#195845: padded rows added after exp in
  fused column sums, about 1e-5 relative on an ISLA-shaped softmax sum), and it can fire in the per-block default-mode
  compiles these lanes use. The A2 partners and the new lanes train under the same compiler (torch 2.12.0+cu130) and settings,
  so it does not bias new ÷ A2; its size is about 400× below bf16 resolution. Scoring runs uncompiled (infer.py compile=false).
- PREREG_v2 §12(f) (the ISLA against ISLA-DTT rule): approved with two edits, written there.
- Cost about 7,100 GPU-h, the longest lane about 76 h: approved.

## 12. Amendment before results of the retrain (2026-10-21 03:15 EDT; the supervisor's approval of 03:10 EDT; no retrained lane had been submitted)

**Why.** COMPILE-AF (scratch/isla-compile-af-2026-10-21/RESULTS.md §§1-4) found that ISLA's per-block compiled path
(`model.compile_blocks=true`, torch 2.12) hits pytorch#195845 (Inductor mix-order reduction) on AirfRANS, Tensile2d and
the lug: per case, the compiled function differs from the eager one (median 9-10 % on AirfRANS, up to 63 % on the lug), and
ISLA-DTT is not query-independent in it (probe up to 5e-2). The r090 lanes of those datasets, and their A2 partners, trained
and were scored in that path. SHIFT-SUV, DrivAerML Surface and Volume, and DeepJEB++ are unaffected (fused kernel). The fix
(blocks compiled with mix-order reduction off) is certified compiled = eager per case on code_cmpaf_fix (code_r090 with only
model.py replaced, md5 45b24fb8; merged into psharpe/isla as 6e2f247e9).

(a) **R2 on the affected datasets.** On AirfRANS, Tensile2d and the lug, the standings become the retrained rows
    `r090fix_*` (below), not the original r090 rows. Elsewhere R2 is unchanged.
(b) **R1 is read as registered,** on the original rows' scorer numbers (the compiled path), with the verdict of §11. Beside
    it, and descriptive only: R1 with the eager re-scores of both sides on the affected datasets ($T/evals_cmpaf_eager for
    r090; COMPILE-AF impact/ for the A2 partners), and R1 with the retrained rows (eager-scored) against the eager A2
    partners.
(c) **New measurement: the training-path effect** T = retrained ÷ original, both scored fully eager (infer.py
    `compile=false model.compile_blocks=false`), per dataset, field and seed (seed-paired: r090fix seed s ÷ r090 seed s),
    and per seed mean. It is read against each field's A2-pair two-seed gap (§6a: AirfRANS p 6.1 %, v 5.1 %, nut 8.6 %;
    Tensile2d displacement 2.5 %; lug von Mises 14 %, noisy). **If |T − 1| on the seed means exceeds that gap on any low-noise
    field of an affected dataset (AirfRANS p / v / nut, Tensile2d displacement), the supervisor is told before anything
    else**, since the A2 partners (afu_a2s, a2r_t2dz, a2r_lug; about 210 GPU-h) would then be retrained too, so that R1
    reads under one function.
(d) **The lug ISLA-DTT secondary arm** (rows 16-17, r090_lug_dtt, cancelled before epoch 500 (epoch 235-237 at 21:20 EDT on 10-20) by the supervisor at 01:09
    EDT because of the compiled-path defect; checkpoints kept, never read) is recorded "incomplete as registered" and is
    completed by r090fix_lug_dtt. P13's lug part is read on r090fix_lug_dtt ÷ r090fix_lug_isla.

**Retrain lanes** (`$T/kernel/r090fix`, new directory; job prefix `isla.r090fix.`; registry row added): ten rows, each the
r090 row with only the run id (`r090fix_*`, fresh runs, never resumed from a bug-trained checkpoint), group and code
(`code_cmpaf_fix`) changed; dataset, model, extra (including `compile=false model.compile_blocks=true`), lr, seed, cells,
epochs, recipe_r090, bf16 byte-identical (checked field by field):

| idx | run | from r090 row | idx | run | from r090 row |
|---|---|---|---|---|---|
| 0 | r090fix_lug_dtt_seed42 | 16 | 1 | r090fix_lug_dtt_seed43 | 17 |
| 2 | r090fix_af_isla_seed42 | 12 | 3 | r090fix_af_isla_seed43 | 13 |
| 4 | r090fix_af_dtt_seed42 | 14 | 5 | r090fix_af_dtt_seed43 | 15 |
| 6 | r090fix_t2dz_isla_seed42 | 2 | 7 | r090fix_t2dz_isla_seed43 | 3 |
| 8 | r090fix_lug_isla_seed42 | 4 | 9 | r090fix_lug_isla_seed43 | 5 |

Scoring: `kernel/r090fix/score.sbatch` (kernel/r090's with job name and roots changed) on score_rows.tsv rows 4k..4k+2 =
the r090 source row's three protocol rows with the run id renamed, code_cmpaf_fix, and ` compile=false
model.compile_blocks=false` appended to extra (fully eager, as kernel/cmpaf/score_rows.tsv), output `$T/evals_r090fix`.
Check `"epoch": 500` and the case counts (AirfRANS 200, Tensile2d 100, lug 600). Cost about 647 GPU-h (COMPILE-AF
RETRAIN_PLAN.md). Smoke first (one AirfRANS row, short QoS, step about 0.39 s, finite losses).

## 13. Amendment: the training-path control and the R2 hold (2026-10-21 09:03 EDT; the supervisor's decision of ~09:00 EDT)

**Status.** For AirfRANS and the lug this is an amendment before results: no retrained (r090fix) AirfRANS or lug lane has
been scored. **For Tensile2d it is a deviation made after the retrained result was seen** (t2d_fourseed.md, 08:55 EDT:
fix-trained ÷ bug-trained displacement 1.099 on four seeds against four, both eager; A2 a2rfix ÷ a2r 1.216). The criterion
below is fixed before any eager-trained (control) result exists. The registered R1 (0.891, confirmed) and §12(c)'s
registered reading (it fired on Tensile2d displacement) stand as written.

**Question.** On Tensile2d, the fixed compiled path (mix-order reduction off) trains models that differ from the
bug-trained ones, field by field (worse displacement and stress, better plastic strain). Is that because (a) the bug's
perturbation changed training and the fixed path trains the model's own (eager) function, or (b) the fixed path trains
differently from eager?

**Controls (eager-trained; no result exists).** Each is its row with ` compile=false model.compile_blocks=false` appended
to extra (fully eager blocks; Hydra's last value wins), everything else byte-identical:
- `r090eag_t2dz_isla_seed42..45`: the r090 Tensile2d row on code_cmpaf_fix (launcher `$T/kernel/r090eag`, prefix
  `isla.r090eag.`; about 65 GPU-h);
- `a2reag_t2dz_isla_seed42/43`: the a2r Tensile2d row on code_a2r, recipe_a2r (launcher `$T/kernel/a2reag`, prefix
  `isla.a2reag.`; about 30 GPU-h).
Scored fully eager like every other row here (epoch 500, 100 cases, float32, native).

**Rule, per field (displacement, stress, plastic strain), per model family (0.9.0: eager 42-45 against fix 42-45 and bug
r090_t2dz 42/43 + att_t2dz_base 44/45; A2: eager 42/43 against a2rfix 42/43 and a2r 42/43).** For two sets X, Y: d = log(mean X
÷ mean Y), SE = sqrt(σ_X²/n_X + σ_Y²/n_Y) with each set's own σ(log) over its seeds; X *matches* Y if |d| ≤ 2 SE, *differs*
if |d| > 2 SE. Seed ranges reported beside.
- (a) eager matches fix and differs from bug: the bug's perturbation changed training; the fixed path is the model's function.
- (b) eager matches bug and differs from fix: the fixed path trains differently from eager.
- (c) otherwise: reported, no reading.
**Weighing, fixed now:** the overall reading is displacement's (the R1 low-noise field); stress and plastic strain are
reported beside. Between the families, the A2 pair is weighed first: its bug-trained and fix-trained pairs already separate
cleanly at two seeds (displacement 0.0068 against 0.0083, plastic strain 0.37-0.45 against 0.19-0.24), so its reading
decides when 0.9.0's and A2's readings disagree, and the disagreement is reported.

**R2 hold.** On AirfRANS, Tensile2d and the lug the standings become the rows trained by the model's own (eager) function:
under (a), the r090fix rows (§12a); under (b), eager-trained rows, with AirfRANS and lug eager retrains planned then; under
(c), the supervisor decides. Until then those three standings are "pending the training-path control". §12(c) on AirfRANS
and the lug is still read as registered when their r090fix rows are scored; their partner retrains (afu_a2s, a2r_lug) wait
for this control, since the training path is the question.

**Lockstep check beside it (job 317142, `kernel/r090fix/lockstep/lockstep.py`; no training).** One-step gradients per
parameter group (eager, eager again as the floor, fix, bug) at r090_t2dz_isla_seed42 epochs 51, 251, 500 and
r090fix_t2dz_isla_seed44 epoch 500, 8 training batches each; and a 300-step lockstep from r090_t2dz_isla_seed42 epoch 251
(model, optimizer, scheduler; same batches, order and RNG) under eager, eager again, fix and bug. Fix within the eager-repeat
floor points to (a); clearly above it points to (b), and the supervisor is told at once.

### 13a. §13 as read (2026-10-21 22:41 EDT; t2d_control.md, scorers 321760 a2reag, 322902 r090eag)

**Overall (b)**, by the registered A2 weighing. A2, displacement: eager 0.00707 (0.00760, 0.00655) against fix 0.00829
→ 0.853 (|d| = 1.06 × 2SE, both eager seeds below both fix seeds: differs), against bug 0.00682 → 1.037 (0.24 × 2SE:
matches) → (b); stress (b) (0.874, 1.53 × 2SE; 1.044); plastic strain (c). 0.9.0 reads (c) on every field (displacement
eager 0.00849, σlog 0.23, ÷ fix 1.077 at 0.26 × 2SE, ÷ bug 1.183 at 0.72 × 2SE), its point estimates leaning (a).
Caveats beside the verdict: A2's (b) rests on two seeds at a 1.06 × 2SE margin; a2rfix used code_a2r with
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 (both recipes build the optimizer with compile_optimizer=cfg.compile and every lane runs
compile=false, so the variable reaches only the compiled blocks, the scope of the model.py fix; checked in recipe_a2r and
recipe_r090 train.py); the 0.9.0 fixed path's fp32 one-step gradients equal eager's to about 2e-5 (lockstep v2, 317294);
r090eag seed 45 ended in a late spike (final validation 0.01196 against its best 0.01012). Consequence (§13 R2): the Tensile2d
standing is r090eag_t2dz 42-45.

## 14. Amendment before results: eager-trained AirfRANS and lug standings, and the AirfRANS training-path read (2026-10-21 22:41 EDT; the supervisor's decision of ~22:40 EDT; no eager-trained AirfRANS or lug lane exists)

**Lanes** (`$T/kernel/r090eagaf`, prefix `isla.r090eag.`, output evals_r090eag): each the r090fix row (code_cmpaf_fix,
recipe_r090, extra, lr, seed, cells, epochs, bf16 byte-identical) with run id `r090eag_*` and ` compile=false
model.compile_blocks=false` appended (fully eager blocks): idx 0-1 r090eag_af_isla_seed42/43, 2-3 r090eag_af_dtt_seed42/43,
4-5 r090eag_lug_isla_seed42/43. About 700 GPU-h. Scored fully eager (score_rows from kernel/r090fix with the run id renamed).
**Held:** the lug ISLA-DTT eager pair (secondary arm, about 480 GPU-h, about 60 h): launched only if AirfRANS ISLA-DTT
reads (b) below; otherwise r090fix_lug_dtt fills the secondary arm, labelled fix-trained.

**Standings (R2, under §13's (b)):** AirfRANS ISLA → r090eag_af_isla, the lug ISLA → r090eag_lug_isla, whatever the read
below gives. ISLA-DTT on AirfRANS (secondary arm) → r090eag_af_dtt.

**AirfRANS training-path read**, the §13 rule per field (pressure, velocity, eddy viscosity), separately for ISLA and
ISLA-DTT: eager-trained r090eag_af_* 42/43 against fix-trained r090fix_af_* 42/43 and bug-trained r090_af_* 42/43, all scored
fully eager (r090 rows: COMPILE-AF's $T/evals_cmpaf_eager). d = log(mean X ÷ mean Y), SE = sqrt(sX²/nX + sY²/nY) with each
set's σ(log); match if |d| ≤ 2 SE, differ if > 2 SE; (a) eager matches fix and differs from bug; (b) eager matches bug and
differs from fix; (c) otherwise. Per model, the reading is (a) or (b) when at least two of the three fields read it and no
field reads the other; otherwise (c). **Consequences:** if either model reads (b), the supervisor is told at once
(AF-EXTRAP's ISLA family and ATTRIB-090's AirfRANS and lug arms trained on the fix path); under (a) or (c) those stand,
labelled fix-trained. If ISLA-DTT reads (b), the held lug ISLA-DTT eager pair is launched.

§12(c) on AirfRANS and the lug is still read when r090fix finishes (T = r090fix ÷ r090, both eager); no A2 partner retrain
follows from it without the supervisor (decision of 22:40 EDT: the AirfRANS read is the cleaner test).

### 14a. §12(c) as read on AirfRANS ISLA and the lug ISLA (2026-10-22)
- AirfRANS ISLA (scorer 325815): T = r090fix ÷ r090 (both eager) pressure 1.024 (1.027, 1.020), velocity 0.985 (0.985, 0.985),
  eddy viscosity 0.998 (0.983, 1.015), all inside the A2-pair gaps (0.061 / 0.051 / 0.086): does not fire; no afu_a2s retrain.
- Lug ISLA (scorer 326815): von Mises T = 0.768 (seed-paired 0.821, 0.720); both fix seeds (0.001463, 0.001445) below both
  bug seeds (0.001780, 0.002006). The lug's von Mises is its noisy field, so the trigger, which reads low-noise fields, **does
  not formally fire**. Beside it: |T−1| = 0.232 lies far outside the A2-pair gap (0.144) and the bug-trained pair's own gap
  (about 12 %), so the effect is real: on the lug the compiled-path bug changed training, by about 30 % in von Mises.
  Posed: fix-trained ÷ bug-trained 0.926 / 0.901, so posed ÷ native rose to 1.35 / 1.38 (bug-trained 1.14 / 1.11). The
  lug's ratio beside R1 (no verdict): fix-trained 0.9.0 ÷ bug-trained A2 (eager) 0.843, labelled "mixed training paths",
  until any A2 retrain.

## 15. Amendment before results: the lug training-path read, decisive for the fixed path (2026-10-22 05:55 EDT; the supervisor's decision of 03:25 EDT; no eager-trained lug lane has finished or been scored)

**Read.** Von Mises, native, all scored fully eager (600 test cases, epoch 500, float32): eager-trained r090eag_lug_isla 42/43
against fix-trained r090fix_lug_isla 42/43 and bug-trained r090_lug_isla 42/43 (COMPILE-AF eager re-scores). The §13 rule:
d = log(mean X ÷ mean Y), SE = sqrt(sX²/nX + sY²/nY) with each set's σ(log); match if |d| ≤ 2 SE, differ if > 2 SE; seed
ranges reported beside. (a) eager matches fix and differs from bug; (b) eager matches bug and differs from fix; (c) otherwise.
On the lug fix and bug separate cleanly (d = log 0.768, both fix seeds below both bug seeds), so **this read is the decisive
test of the fixed path** (on AirfRANS fix and bug agree within 2.4 %, and the §14 read there cannot discriminate).

**Consequences.**
- (a): the fixed path is the model's function and the bug changed lug training. Every fix-path lane (r090fix, AF-EXTRAP's
  afxf, ATTRIB-090's att2) stands, labelled fix-trained. The lane then proposes the a2r_lug retrain on the fixed path
  (code_a2r + TORCHINDUCTOR_MIX_ORDER_REDUCTION=0, about 105 GPU-h) for a like-for-like lug ratio beside the record; the
  supervisor confirms it then.
- (b): the supervisor is told at once; every fix-path lane is then in question. The held lug ISLA-DTT eager pair is launched.
- (c): reported, no reading.
- **The trigger for the held lug ISLA-DTT eager pair moves here** (replacing §14's AirfRANS ISLA-DTT trigger): it is launched
  only under (b). Otherwise r090fix_lug_dtt fills the secondary arm, labelled fix-trained.
- The lug standing is r090eag_lug_isla whatever this reads (§13's (b), as registered).

**Predictions** (mine; lockstep v2/v3: the fixed path's fp32 gradients equal eager's to about 2e-5 and its bf16 gradients sit
at the eager-bf16 floor, so a systematic difference between fix and eager training has no known mechanism):
| id | prediction | p |
|---|---|---|
| L1 | the lug read is (a) | 0.75 |
| L2 | it is (b) | 0.07 |
| L3 | eager ÷ fix within [0.90, 1.10] | 0.80 |
| L4 | both eager seeds below both bug seeds (0.001780, 0.002006) | 0.80 |
| L5 | eager posed ÷ native within [1.25, 1.50] on both seeds | 0.60 |
