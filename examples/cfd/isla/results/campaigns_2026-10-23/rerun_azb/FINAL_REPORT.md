# RERUN-AZB: final report (written 09:45 EDT 10-27)

Status: DONE. All 16 AeroZoo baseline lanes (14 GeoTransolver, 2 Transolver) train on JHB. Every lane passed the first-epoch check,
and the scoring is staged but not run. Commands and job ids: `JOBS.md`. Handoff (watch line, resume steps, scoring): `HANDOFF_AZB.md`.

## Snapshot (psharpe/isla 6f1e527f1, on JHB under the mirrored task root $T)
- **Code:** `$T/code_fin6f1e` holds `git archive 6f1e527f1 physicsnemo pyproject.toml test` (tar md5 50e565e1).
- **tensordict 0.15.3:** unzipped from the premerge wheel. Its sha256 a437693b matches uv.lock at 6f1e527f1. Its 79 files equal LHR `$T/td0153` by md5 (pyc files excluded).
- **Recipe:** `$T/recipe_fin6f1e` holds `git archive 6f1e527f1:<recipe>` (tar md5 107ebc2f). Its dataset_paths.yaml is as in git, and that names the JHB AeroZoo copy.
- **md5 check on JHB:** code 3,229/3,229, td 79/79 and recipe 158/158 match.
- SNAPSHOT_HEAD and MD5SUMS are written (MD5SUMS md5 078b6ff4 for the code, e6f4235c for the recipe), and both trees are `chmod -R a-w`, with 0 writable entries left.

## Smoke: JHB 1325301, 0.94 GPU-h
- It ran the lanes' exact command on lanes 0 (GeoTransolver full, seed 42) and 14 (Transolver full, seed 42).
- smoke_wrap's step cap does not work with 6f1e527f1's loop, so each lane ran one whole epoch and wrote a checkpoint.
- GeoTransolver: iter_time_s median 0.547 s (AZ-COPY-JHB measured 0.554); epoch 356 s against a priced 388 s.
- Transolver: 0.490 s; epoch 310 s against a priced 369 s.
- The configs resolved as intended: lr 1e-3, out_dim 9, bf16, eager, 12/12 loader.
- Losses were finite and fell. 0 non-finite numbers.

## The 16 lanes (one batch_long job each, plus a pending chain-ahead link as the fallback)
- **Job ids:** 1325667, 1325668, 1325669, 1325670, 1325716, 1325717, 1325718, 1325719, 1325759, 1325828, 1325761, 1325763, 1325806, 1325807, 1325808 and 1325809 (lanes 0-15, in that order).
- **Lane 9:** its first job, 1325760, died in 24 s on nvl72d421-T14, a node that failed other users' jobs the same way. It was resubmitted with a JHB exclude list.
- **First epochs** (`checks/chk_latest.txt`, `checks/projection_0940.md`):
  - Epoch 1 took 0.73-0.82 of PRICE's epoch, for example GT full 313 s against 388 s, GT profile_crossflow 362 s against 443 s, and T full 270-285 s against 369 s.
  - Epoch 0, with cold reads, took 0.78-1.05.
  - No NaN and no loader stall: mean/median iter_time_s is 0.99-1.16, and 0 % of steps ran over 2x the median.
- **Expected ends:** 10-28 between 02:15 (blobs s43) and 11:30 EDT (profile_crossflow s42).
- **Projected GPU-h:** 1,344 for training, against PRICE's 1,858 (0.72) and the 1.25x cap of 2,322.
- **GPU-h used so far** (sacct, 09:44): 26.1, of which the smoke took 0.94.

## What it means
- JHB does not slow these lanes the way it slowed HiLift: they run about 20 % faster than PRICE's LHR lane walls.
- Those walls came from the older recipe, and #2171's loop is in this code.
- So placement C holds for the baselines, and AGA's closing on 10-30 does not threaten them.
- This is my confidence in the numbers, not a speed claim: speed figures still come from LHR (SUPERVISOR_DECISIONS item 8).

## A discovery the main session should know
- **JHB's Lustre MDT0009 is read-only** (`lfs df` flag R). New directories go round-robin over 16 MDTs down to depth 12, so about 1 mkdir in 16 fails with EROFS (my snapshot tar and one mkdir hit it).
- Fix: I pinned the default directory layout of `$T/runs`, `$K`, `$T/kernel/finazb_score` and `$T/finazb_score_out` to their own healthy MDTs (`lfs setdirstripe -D -i <mdt> -c 1`).
- The ISLA AeroZoo launcher on JHB needs the same: a new top-level dir under `$T` should get the pin, or a retried mkdir.
- Suggested LANE_RULES note under "JHB": "MDT0009 read-only (10-27): pin new task dirs with `lfs setdirstripe -D -i <own MDT> -c 1`; JHB exclude list `$T/kernel/finazb/exclude_nodes_jhb.txt` (nvl72d421-T14)."

## Rulings relied on
- **Placement C** (SUPERVISOR_DECISIONS item 8).
- **LANE_RULES "JHB":** no LHR exclude list, and Slurm added to the PATH. I chose the mirrored task root of RERUN-LAUNCH-PLAN (M), with the existing JHB venv.
- **The brief:** batch_long, one job per lane; the scorer of SUPERVISOR_DECISIONS item 4. No code, config or LANES.tsv change.

## What is left
- Everything above is still running.
- After lanes 0 and 14 finish: `check.sbatch` (about 0.3 GPU-h).
- After all 16 finish: `sbatch --array=0-2 score.sbatch` (about 32 GPU-h), then `build_json.py` locally.
- Then remove the registry row.

## On disk
- **JHB:**
  - `$T/code_fin6f1e` and `$T/recipe_fin6f1e` (about 60 MB, read-only);
  - `$K` (launcher, smoke runs about 150 MB, snap tgz 35 MB);
  - `$T/kernel/finazb_score`;
  - `$T/runs/fin_az_*`, about 9 GB per lane at the end (every second epoch kept), about 146 GB in all.
- **Here:** this folder (3 MB). No worktree, no venv.
