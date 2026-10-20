# GEO-CONT jobs (LHR; T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0)

## Probes and smokes (finished)

| job | what | result |
|---|---|---|
| 273927 | `smoke_gcont/probe.sbatch` (cpu-short): geo_probe.py, 6 cases per dataset, native vs posed, current encoding | MEASUREMENTS.md; probe/probe_273927.json |
| 274801 | `smoke_gcont/smoke.sbatch` ARM=ctl: 3 x 20-step smokes, ISLA suite on GPU, dirflip instrument | rc 0; 269 passed; smoke/dirflip.json |
| 274802 | ARM=cont (afterany:274801): 3 x 20-step smokes, ISLA suite on GPU, recipe suite | rc 0; 271 passed; recipe 554 passed / 2 Slurm-env failures |

## Lanes (kernel/gcont/lanes.tsv; chains `isla.gcont.<slug>`, singleton, normal QoS, 03:59:00 links)

Wave 1 (seed 42), submitted 2026-10-18 23:33 EDT (20:33 PDT); first-link job ids:

| idx | run id | first job |
|---|---|---|
| 0 | gcont_ctl_t2dz_isla_seed42 | 275257 |
| 2 | gcont_ctl_lug_isla_seed42 | 275258 |
| 4 | gcont_ctl_af_isla_seed42 | 275259 |
| 6 | gcont_cont_t2dz_isla_seed42 | 275260 |
| 8 | gcont_cont_lug_isla_seed42 | 275261 |
| 10 | gcont_cont_af_isla_seed42 | 275262 |

Wave 2 (seed 43), submitted 23:46 EDT: idx 1 gcont_ctl_t2dz_isla_seed43 275409; 3 gcont_ctl_lug_isla_seed43 275410;
5 gcont_ctl_af_isla_seed43 275411; 7 gcont_cont_t2dz_isla_seed43 275412; 9 gcont_cont_lug_isla_seed43 275413;
11 gcont_cont_af_isla_seed43 275414.

Acceptance (00:12 EDT Oct 19): 12/12 (`bash $T/smoke_gcont/accept.sh`).

Every link appends to `$T/runs/<run>/train.log` (link-start line with code / recipe), and chain-ahead links submit
themselves (`kernel/gcont/sbatch_logs/resubmit.log`). Markers: `kernel/gcont/STATUS_a2r<job>_<idx>`, `RUN_*` (DONE),
`BLOCKED_<run>` (two identical failures).

Monitor: `ssh lhr2 "squeue -u \$USER -h -o '%i %j %T %M %r' | grep gcont"`;
`ssh lhr2 "grep -h 'Training completed' $T/runs/gcont_*/train.log | wc -l"`.

## Scoring (after all 12 lanes print "Training completed")

36 rows = 12 runs x {native, area, posed}, 4 rows per array task (9 tasks), scorer `kernel/gcont/score.sbatch`
(a byte copy of kernel/a2r/score.sbatch: infer.py, float32, infer_split=val, the final checkpoint, sampling_resolution =
the lane's cells). Build the rows table and submit:

```bash
ssh lhr2
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
K=$T/kernel/gcont; V=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
cd $K
{ printf 'idx\trun_id\tprotocol\tdataset\tmodel\textra\tcode\trecipe\tcells\tvenv\n'; n=0
  awk -F'\t' 'NR>1' lanes.tsv | while IFS=$'\t' read -r i g run ds model extra lr seed cells code ep; do
    for p in native area posed; do printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $n "$run" $p "$ds" "$model" "$extra" "$code" recipe_gcont "$cells" "$V"; n=$((n+1)); done; done; } > score_rows.tsv
sbatch -J isla.gcont.score --array=0-8 --export=ALL,A2R_ROOT=$K,SCORE_ROWS=$K/score_rows.tsv,EVAL_ROOT=$T/evals_gcont score.sbatch
```

Outputs: `$T/evals_gcont/<run>/<protocol>/<run>/metrics.jsonl` (the `infer_summary` row; per-sample `infer_step` rows
when a 2-D summary is refused), `.done`, `SNAPSHOT_USED`, `DATASET_USED`, `<protocol>.log`. Reduce exactly as
`examples/cfd/isla/results/char/a2r_protocol_artifacts_2026-10-17.py` does (infer_summary, else the mean of the per-sample
rows, last row per sample id), then apply the PREREG rule (amendment section).
