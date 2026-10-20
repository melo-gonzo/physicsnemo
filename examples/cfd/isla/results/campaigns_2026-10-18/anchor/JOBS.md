# E3b ANCHOR pilot jobs (LHR, `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`)

Launcher: `$T/kernel/anchor_af/` (copies of `kernel/airfrans_u`; `SOURCE_MD5SUMS` has the sources' md5s).
`lanes.sbatch` differs from the source only in its default job name (`isla.anc`); `submit.sh`, `score.sbatch`,
`score_submit.sh`, `score_when_done.sh` also point at `kernel/anchor_af`, `recipe_anc` / `code_anc` and `$T/evals_anc`
(diffs in this lane's report). Lanes table `lanes.tsv` (rows 0-9), smoke table `lanes_smoke.tsv`, `smoke.sbatch`.
Code: `$T/code_anc` + `$T/recipe_anc`, `psharpe/isla-anchor` abf8e1d5f (SNAPSHOT_HEAD, MD5SUMS md5 de1affeb / 4c6beb38).
Runs: `$T/runs/anc_<arm>_seed<s>/train.log` and `checkpoints/`.

| job | what | QOS | outcome | log |
|---|---|---|---|---|
| 275224 | smoke rows 2 4 3 + GPU pytest (short) | short | cancelled by me as it started (after a QOSGrpNodeLimit wait) | `sbatch_logs/isla.anc.smoke-275224.log` |
| 275344 | smoke rows 2 (anchor), 4 (mesh), 3 (dtt) + GPU pytest | normal | pytest 25 passed; anchor, dtt ok; mesh failed (CUDA generator in SplitAnchorNodes) | `sbatch_logs/isla.anc.smoke-275344.log` |
| 275345 | smoke rows 1 (passive), 0 (isla) | normal | ok | `sbatch_logs/isla.anc.smoke-275345.log` |
| 275469 | smoke row 4 (mesh) after fix abf8e1d5f | normal | ok | `sbatch_logs/isla.anc.smoke-275469.log` |
| 275514 / 275515 | score.sbatch smoke mode on the anchor / mesh smoke checkpoints (`AFU_SCORE_CKPT`, root `kernel/anchor_af/smoke_evals`) | normal | 3/3 protocols .done for both (untrained: rel-L2 ≈ 1; posed = native to 0.06 %) | `sbatch_logs/isla.anc.score.smoke-*-<job>.log` |
| 275547-275552 | wave 1 lanes 0,1 (isla), 4,5 (anchor), 8,9 (mesh), submitted 2026-10-18 23:59 EDT | normal 03:59 chains | queued | `$T/runs/anc_*/train.log` |
| 275648-275651 | wave 2 lanes 2,3 (passive), 6,7 (dtt), submitted 2026-10-19 00:12 EDT | normal 03:59 chains | queued | `$T/runs/anc_*/train.log` |

Smoke runs: `$T/kernel/anchor_af/smoke_runs/smk_<arm>_seed42/` (remove after the pilot).

## Scoring (after "Training completed" in each train.log)

    ssh lhr2 'bash /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/anchor_af/score_when_done.sh'

Output `$T/evals_anc/<run>/{native,area,posed}/<run>/metrics.jsonl` (mean of the per-case `infer_step` rows =
the summary). Seed means and C/A ratios per the PREREG rules.

| 275463 / 275575 / 275632 | QUERY-POP probe smokes (subagent), float32, smoke checkpoints | normal | ratios 1 for passive/anchor/dtt/mesh; changes ≤ 2.1e-4 (anchor), ≤ 6e-5 others; isla 1.5; validity vs infer.py ≤ 2e-7 | `$T/kernel/anchor_af/qpop/smoke/` |
| 276051 | QUERY-POP float64 smoke (smk_anchor, smk_passive, smk_isla; 12 cases) | normal | anchor 1.4e-13, passive 5e-14, isla 1.8 | `$T/kernel/anchor_af/qpop/smoke_f64/reduced.md` |

## QUERY-POP probe

`$T/kernel/anchor_af/qpop/` (af_qpop.py, af_qpop.sbatch, reduce.py, README.md; copy in `qpop/` here). After the
lanes finish (it skips unfinished runs):

    ssh lhr2; cd $T/kernel/anchor_af/qpop
    sbatch -q normal -t 01:00:00 -J isla.anc.qpop.final af_qpop.sbatch            # float32, 200 cases, 10 runs -> out/
    sbatch -q normal -t 01:00:00 -J isla.anc.qpop.f64 --export=ALL,QPOP_FLOAT64=1,QPOP_CASES=20,QPOP_OUT=$PWD/out_f64 af_qpop.sbatch
    /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python reduce.py $PWD/out      # out/reduced.md, .json
    (same for out_f64)

Add `--export=ALL,VALIDATE=1` to the float32 call if `$T/evals_anc` is not scored yet (it reruns infer.py there).
