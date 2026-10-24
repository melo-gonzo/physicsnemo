# LRE-AZ: ISLA's fixed Reynolds scale on AeroZoo's reynolds split (pre-registration, 2026-10-23)

Lane LRE-AZ of ISLA (research-lane, launched 2026-10-23 about 01:22 EDT by the supervisor 01a13c2c). Written 01:33 EDT,
2026-10-23, **before any B job: no B job has been submitted, and no B checkpoint or score exists.** The only cluster job
so far is a CPU job (345841, below) that computes ISLA's own Reynolds number for 285 val and test cases.

## 1. Question, and what kind of test this is

Should ISLA's Reynolds input become the fixed, dataset-independent scale z = log10(Re_ISLA / 1e6), in place of the
per-split fitted z = (log10 Re_ISLA - mean) / std? This is the second-dataset test of AF-EXTRAP's LRE finding (AirfRANS
reynolds: below-band pressure 0.530 / 0.576 of the fitted-scale model; `../isla-af-extrap-2026-10-21/LRE.md`, RESULTS.md
section 6). That effect came from a fitted spread of 0.062 decades, which stretched a narrow Re band to about ±2 units of z.

On AeroZoo the fitted spread is 0.8985 decades, so the fixed scale only rescales z by 0.8985 and shifts it by -0.938
(z_fixed = 0.8985 z_fitted - 0.938). z enters ISLA 0.9.0 only as one more global scalar, concatenated to the inputs of
linear layers (code_r090 `model.py` 1300-1301, 810, 902), so a fixed affine map of z can be absorbed by the first layer.
It can change only what training finds. **This is therefore registered as a check for harm** on a dataset whose
Reynolds number spans several decades, not as a test for a gain like AirfRANS's.

## 2. The split's facts (step 1)

Sources: labels `examples/cfd/isla/results/aerozoo_dataset/case_labels.json` (`log10_re`) and the split lists
(`manifest.json` keys `reynolds.{train,val,test}`), as tabulated in `../isla-az-decide-2026-10-21/reynolds/reynolds_re.tsv`
(every value matches case_labels.json to 1e-5, checked by `reduce_lreaz.py`). ISLA's own log10 Re (U L / nu with its
gauge length L, the quantity z is made from): AZ-DECIDE's `scales/scales_{train_full,test_all}.tsv` (all 2,119 train and
1,287 test cases), plus job 345841 (LHR cpu-short, 3 min, 285 cases, 0 errors; `cluster/gauge/`, the same fit path
imported unchanged) for the other 98 test and 187 val cases. The train side reproduces the fitted constants: mean 5.0619,
population std 0.8985.

| quantity | train (2,119) | test (1,385) |
|---|---|---|
| label log10 Re | 3.003 to 5.999 | 6.001 to 8.000 |
| test cases' distance above the train maximum (label) | - | 0.002 to 2.000 decades (q25 0.432, median 0.836, q75 1.369); none below; bins [0, 0.5) 411, [0.5, 1) 389, [1, 1.5) 312, [1.5, 2] 273 |
| ISLA's log10 Re | 3.247 to 7.040 | 6.233 to 9.133; 414 inside ISLA's train band, 971 above it (by up to 2.09 decades) |
| fitted z = (log10 Re_ISLA - 5.062) / 0.8985 (A) | -2.020 to 2.202 | 1.304 to 4.531 |
| fixed z = log10(Re_ISLA / 1e6) (B) | -2.753 to 1.040 | 0.233 to 3.133 |

By the label the split is pure extrapolation upward. By ISLA's own Re (label + 0.55 ± 0.2 decades, case by case), 30 % of
the test cases sit inside the training band. Full tables: `tables_A.md`.

## 3. Arms

`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0` (LHR).

- **A (reused, not retrained):** `azc_reynolds_isla_seed42` and `_seed43`, rows 4 and 5 of `$T/kernel/azi/lanes.tsv`
  (md5 11d10f9b). Snapshot `$T/code_r090` (psharpe/isla 92d9d7800 + tensordict 0.15.2; MD5SUMS c1227687, SNAPSHOT_HEAD
  ccdcc32d) and `$T/recipe_r090` (MD5SUMS bb4ee858). Launcher `$T/kernel/azi/` (lanes.sbatch 8eecec8f, site.env c9006c02:
  bf16, normal QoS, 03:59:00 links, `--segment=1`). Resolved config: `model=isla_volume_aerozoo`,
  `dataset=aerozoo_uvolume_reynolds`, `train_split=reynolds.train val_split=reynolds.val +training.val_interval=10
  dataloader.num_workers=12 dataloader.prefetch_factor=12`, lr 3.75e-4, 250 epochs, 200,000 cells, `compile=false` (the
  model's compiled blocks, `compile_blocks` default), `augment=false`, `training.save_interval=2`. Its checkpoint's
  args.json records `reynolds: {mean: 5.062, std: 0.8985}`. Scores `$T/p200k_fp32_test_azc/azc_reynolds_isla_seed4x/`
  (AZ-DECIDE's fixed scorer `kernel/azdecide/eval_azi/eval_test.sbatch`, md5 ca78c254; local copies in
  `../isla-az-decide-2026-10-21/scores/p200k_fp32_test_azc/`).
- **B:** A's exact snapshot and resolved config with one change: `dataset=aerozoo_uvolume_reynolds_lre`.
  - The file is `aerozoo_uvolume_reynolds.yaml` (md5 f7cc0d6d, the same in recipe_r090, 92d9d7800 and psharpe/isla) with
    `isla_constants.reynolds {mean: 6.0, std: 1.0}` and a header, as AirfRANS's `airfrans_uvolume_2d_reynolds_lre.yaml`.
    md5 eb37789f, committed with its tests on psharpe/isla-lre-az 4781af873 (pushed; base psharpe/isla aa3ab2df5). The tests:
    the file equals its source but for those constants, the AeroZoo dataset-file checks, and ISLA built from it takes
    {6, 1}; 35 passed on CPU.
  - The recipe snapshot is `$T/recipe_lreaz`: `cp -a recipe_r090`, plus this one file, a regenerated MD5SUMS, and a
    SNAPSHOT_HEAD line; read-only. The overlay is proved by `diff -rq recipe_r090 recipe_lreaz`, which must list only
    the new file and the two metadata files (MD5SUMS, SNAPSHOT_HEAD), and by MD5SUMS agreeing on every other file. Code
    `$T/code_r090` is unchanged.
  - The launcher is `$T/kernel/lreaz/`, with byte copies of kernel/azi's lanes.sbatch, ckpt_preflight.py and submit.sh
    (submit.sh with only the job-name prefix `isla.lreaz.`). site.env is A's with `RECIPE=$T/recipe_lreaz` and
    `LANE_SBATCH_EXTRA="--segment=1 --exclude=$T/kernel/lreaz/exclude_nodes.txt"`: a node-list file, because a comma
    list would break the chain-ahead `--export`, as in ATTRIB's kernel/att2m. It excludes nvl72d015-T12,
    nvl72d058-T08, nvl72d052-T01 and nvl72d057-T13. lanes.tsv holds A's rows 4 and 5 with run ids
    `lreaz_reynolds_isla_seed4x`, group `lreaz`, and the dataset changed. Everything else is equal: seeds 42 and 43,
    epochs, rate, cells, bf16, QoS, link length, and the training path (`compile=false` with compile_blocks; AeroZoo is
    not exposed to the compiled-path fault, CMPDIAG).

## 4. Training, smoke, scoring and checks

- **Smoke** (`$T/kernel/lreaz/smoke/smoke.sbatch`, short QoS, one 4-GPU node, 00:15:00 limit, so at most 1 GPU-h). It is
  kernel/afxl/smoke's smoke (lane 0 through the lane launcher, NO_CHAIN, smoke run dir). Its wrapper is
  `smoke_wrap_afxl.py` with a ZRANGE line also printed every 100 training forwards, so a cut run still reports.
  SMOKE_STEPS=529 makes the run one epoch (A's reynolds epoch is 529 steps per rank). Passes if:
  - every rank's `SMOKE ISLA` line shows `reynolds={'mean': 6.0, 'std': 1.0}`;
  - z over all four ranks' training batches lies within [-2.753, 1.040] and spans at least [-2.70, 1.00];
  - the smoke's resolved_config.yaml equals A's (`runs/azc_reynolds_isla_seed42/resolved_config.yaml`) except run_id,
    dataset and the output, checkpoint and runs paths;
  - there is no traceback and no non-finite output.

  The median step against A's 0.528 s is reported beside. The lanes are submitted only if the smoke passes.
- **Lanes:** chained singleton links `isla.lreaz.reynolds-isla-seed4x`, normal QoS, 03:59:00, as A's. A final link that
  fits under 2 h is moved to short QoS with `scontrol update` (at most 12 short-QoS jobs user-wide). NODE_FAIL
  duplicates are cancelled (the oldest pending link is kept).
- **Scoring exactly as A:** `$T/kernel/lreaz/eval_test.sbatch` is a copy of AZ-DECIDE's ca78c254 with only these
  changed: RECIPE (recipe_lreaz), LANES (kernel/lreaz/lanes.tsv), ROOT (`$T/p200k_fp32_test_lreaz`), KDIR
  (kernel/lreaz/eval), the job name (`isla.lreaz.test-eval`) and the excluded nodes. The rest is unchanged: float32,
  200,000 interior points, `val_split=reynolds.test`, `infer_split=test`, the native protocol (A was scored native only),
  and the final checkpoint.
- **Checks before reading** (in `reduce_lreaz.py` and `cluster/check_re.sh`):
  - each score row has `"epoch": 250` and `infer_split` test, 1,385 cases with the same case ids as A's and the
    split's list, and SNAPSHOT_USED code_r090.
  - **reynolds {6, 1} for B and not A.** The recipe's resolved config never holds isla_constants: it names the dataset
    file, keeps `model.reynolds: null`, and train.py and infer.py inject the constants from the file. So the check is,
    per row:
    1. the score row's config line has `dataset` `aerozoo_uvolume_reynolds_lre` (B) or `aerozoo_uvolume_reynolds` (A);
    2. the scorer's saved Hydra config (`<KDIR>/hydra_cwd/<run>/outputs/*/*/.hydra/config.yaml`) has the same dataset,
       and its log's RECIPE is recipe_lreaz (B) or recipe_r090 (A);
    3. that recipe's dataset file has md5 eb37789f, which holds {6.0, 1.0} (B), or f7cc0d6d, which holds
       {5.062, 0.8985} (A);
    4. the epoch-250 checkpoint's args.json records `reynolds: {mean: 6.0, std: 1.0}` (B) or `{5.062, 0.8985}` (A).

    reynolds is a plain attribute, not a buffer, so a score of B built with A's file would silently use A's z. Any
    failure makes that row invalid, and the read **incomplete as registered**.
- **Log extract for the guard:** `$T/kernel/lreaz/extract_runs.py`, a byte copy of AZ-DECIDE's (cpu-short), on B's run
  dirs: validation loss per epoch and step times.

## 5. Rules (registered)

**R1, reading rule, per field, on the reynolds test set (1,385 cases).** Pressure and velocity are ruled; nut, tke and
omega are computed the same way and reported beside.
- `D = ln( mean_seeds CM(B) / mean_seeds CM(A) )`, where CM is a run's case-mean relative L2 over the test cases.
- `SE = sqrt((s_A^2 + s_B^2) / 2)`, where s is an arm's two-seed sample standard deviation (n - 1) of ln CM.
- **Helps** if D < ln 0.90 and |D| > 2 SE. **Harms** if D > ln 1.10 and |D| > 2 SE. **Otherwise no change.**
- Beside it, not a verdict: the same D, SE and label computed from case medians (AeroZoo's errors are heavy-tailed),
  and the per-seed ratios B / A (seed-matched).
- What A's noise implies (from A's scores): pressure s_A = 0.0083, so 2 SE >= 0.012 and the ln 1.10 = 0.095 bar binds
  unless s_B > 0.13. Velocity s_A = 0.0293, so 2 SE >= 0.041, and the bar binds unless s_B > 0.13.
  - **Erratum (2026-10-23 14:56 EDT, from the book's second read via the supervisor; the rule is unchanged):** with the
    registered SE = sqrt((s_A^2 + s_B^2) / 2), the harms bar ln 1.10 binds only while s_B < 0.067 on pressure and
    s_B < 0.061 on velocity, and the helps bar ln 0.90 while s_B < 0.074 on pressure and s_B < 0.069 on velocity; the
    "s_B > 0.13" values above are wrong. R1 is read as registered, with both of its conditions.
- Slots: 2 arms x 2 seeds = 4 independent scores, one protocol. Pressure and velocity come from the same runs, so they
  are two readings of 4 numbers each, not 8 independent ones.

**R2, descriptive extrapolation profile (no verdict).**
- Per distance bin of the label (decades above the train maximum: [0, 0.5), [0.5, 1), [1, 1.5), [1.5, 2]), the seed-mean
  case-mean error of A and B and the ratio B / A, for all five fields.
- Slopes per decade, least squares over the 1,385 test cases. On the seed-mean per-case error: AF-EXTRAP's definition
  (`posthoc.py` slope), which gave pressure 0.62 (fitted) and 0.16 (fixed) below AirfRANS's band. On its log10:
  AZ-DECIDE's definition. And on the per-case ln(B / A).
- Beside: B / A on the 414 test cases inside ISLA's own training band and on the 971 above it.

**Consequence (registered).**
- 'Harms' on pressure or on velocity: the fixed scale is **not adopted without Peter**.
- Otherwise (no change or helps on both): the supervisor proposes adopting z = log10(Re / 1e6) as ISLA's Reynolds input
  contract at the scope freeze (about 10-27), on principle. It is dataset-independent, it fixes AirfRANS, and it costs
  nothing here. The decision and the merge are the supervisor's and Peter's; the lane only reports.
- 'Harms' on nut, tke or omega alone is reported beside it and does not block.

**Event guard (program standard):** if a B lane's validation loss (val_summary every 10 epochs) passes twice its running
minimum, the read stops and goes to the supervisor. A's largest ratios on this split: 1.026 (s42, epoch 249) and 1.092
(s43, epoch 20).

**Incomplete.** A rule that cannot complete as registered (a lane lost, a scorer that cannot finish, an invalid row) is
reported as "incomplete as registered", with what the available slots show, and escalated.

## 6. What is already known

Known to this lane and the supervisor before registration: all of A's test-side scores, from the scorer's rows
(`tables_A.md`). Case mean [case median]:

| seed | pressure | velocity | nut | tke | omega |
|---|---|---|---|---|---|
| 42 | 0.5062 [0.5234] | 0.1608 [0.1520] | 0.8189 [0.4974] | 0.6812 [0.6876] | 0.5848 [0.5688] |
| 43 | 0.5122 [0.5369] | 0.1676 [0.1519] | 0.8871 [0.4928] | 0.6889 [0.6932] | 0.5705 [0.5657] |

A's profile, seed-mean case means by bin: pressure 0.377 / 0.459 / 0.566 / 0.716, velocity 0.144 / 0.155 / 0.172 / 0.199.
Slopes per decade: pressure +0.228 (log10 +0.223), velocity +0.037 (log10 +0.090), nut +1.02. Also known: GeoTransolver's
scores on this split (pressure about 2.0x ISLA's, flat in distance; AZ-DECIDE `reynolds/RESULTS_reynolds_partial.md`), and
AF-EXTRAP's LRE and LIN results on AirfRANS. Nothing of B exists.

## 7. Predictions (mine)

| id | prediction | p |
|---|---|---|
| P1 | The smoke passes on its first run (reynolds {6, 1} on every rank, z in range) | 0.93 |
| P2 | R1 pressure: no change / helps / harms | 0.80 / 0.08 / 0.12 |
| P3 | R1 velocity: no change / helps / harms | 0.80 / 0.07 / 0.13 |
| P4 | Consequence: no harm on pressure or velocity, so the proposal goes to the supervisor | 0.75 |
| P5 | nut, tke and omega all read no change (beside) | 0.55 |
| P6 | R2: B / A on pressure within 0.95-1.05 in every distance bin | 0.55 |
| P7 | R2: the slope of per-case ln(B / A) on pressure has magnitude below 0.05 per decade | 0.70 |
| P8 | Both B lanes finish with no event-guard hit and no failure needing more than a resubmit | 0.85 |

Why: the change is an affine reparametrization that the first linear layer can absorb. Its fitted counterpart is already
well conditioned (a spread of 0.9 decades, not 0.06), so only optimization (initialization, AdamW's weight decay on the
z weights against the bias) can differ. On AirfRANS the fitted z spread 0.06 decades over about ±2 units, which these
AeroZoo numbers do not reproduce. A 10 % move on either field would be a surprise.

## 8. Cost and schedule

- B lanes: A allocated 91 GPU-h per lane (sacct over all links; AZ-DECIDE `tables_partial.md`, Cost), so 2 x 91 = 182
  GPU-h.
- Smoke: at most 1 GPU-h (00:15:00 x 4 GPUs). Scoring: one 4-GPU job, about 1.1 h (1,385 cases at about 2.8 s on each of
  2 GPUs), about 4.5 GPU-h. CPU: the gauge job (done) and the log extract.
- **Total about 188 GPU-h** against the brief's 300. If failures or reruns project the total above 300, I ask first.
- Wall: A trained about 22 h per lane (6 links). If the lanes start by 10-23 06:00 EDT, the read lands about 10-24
  midday. If the arms are not read by 10-25 09:00 EDT, I write HANDOFF_LREAZ.md and report.
