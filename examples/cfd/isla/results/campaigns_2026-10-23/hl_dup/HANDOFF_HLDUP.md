# HL-DUP handoff (2026-10-24, about 08:35 EDT)

Stage 1 is done (`RESULTS.md`, revised after the second read, see `REVISION_NOTE.md`). Stage 2 lanes are running, as
registered in `PREREG.md` (md5 072b6b1c3ac1b59df729eadda9f04e40, with the supervisor's amendment of 06:44 EDT and the
smoke checks). LHR $T = `/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`.
Details and md5s: `JOBS.md`, "Stage 2".

## Lanes (normal QoS, 03:59:00 links, singleton chains; first submitted 08:23-08:24 EDT)

| run | kernel / index | first link | chain-ahead link at handoff |
|---|---|---|---|
| hldupx_hl_isla_seed42 | hldupx_isla 0 | 367678 (running) | 367684 |
| hldupx_hl_isla_seed43 | hldupx_isla 1 | 367679 (FAILED 4:53, Errno 521 in the shared Triton cache) | 367683 |
| hldupx_hl_gt_seed42 | hldupx_base 0 | 367681 (running) | 367709 |
| hldupx_hl_gt_seed43 | hldupx_base 1 | 367682 (running) | 367710 |
| hldupx_hl_t_seed42 | hldupx_base 2 | 367686 (running) | 367711 |
| hldupx_hl_t_seed43 | hldupx_base 3 | 367687 (running) | 367723 |

- Run dirs: `$T/runs/hldupx_hl_*/train.log`.
- Code and recipes: code_h090 + `$T/recipe_hldupx_isla`, and code_a2r + `$T/recipe_hldupx_base`, each with the
  dataset `highlift_surface_medium_dedup`.
- Masks: `$T/hldup/masks` (690 cases, MANIFEST.tsv, 23 GB, read-only; never delete while the lanes or the scorer run).
- Registry row: `^isla\.hldupx\.`, expected_end 2026-10-26.
- Seed 43's first link failed on the NFS cache race that LANE_RULES describes. It is not BLOCKED; 367683 restarts it.
  If an identical failure follows, the launcher writes `kernel/hldupx_isla/BLOCKED_hldupx_hl_isla_seed43`: archive the
  marker and resubmit with `cd $T/kernel/hldupx_isla && bash submit.sh lanes 1 1`.

## Acceptance at the first link (check once each lane is past epoch 2)

- the `=== link start` line reads recipe_hldupx_*;
- `drop_duplicate_cells: ... fingerprint checked` in train.log;
- losses finite;
- step time from epoch 1: ISLA about 0.31-0.52 s (seed 43's epoch 0 already ran at 0.31 s), GT about 0.46-0.62 s,
  T about 0.33 s;
- no traceback other than a link's SIGTERM at its time limit.

Cost projection (standing spans x 4 GPUs x smoke step ratio): about 270 GPU-h of the 320 cap. If ISLA's epochs run
faster than the standing's, as epoch 0 suggests, it will be less. Re-price after the first link; if the projection
passes 320, tell the supervisor.

## Scoring (after "Training completed" in all six train.logs)

```bash
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
grep -c "Training completed" $T/runs/hldupx_hl_*/train.log        # each 1
ssh lhr2 "cd $T/kernel/hldupx_score && sbatch --array=0-2 score.sbatch"
# 12 rows: native + posed per run, dataset highlift_surface_medium_dedup, float32, val, 200,000 drawn cells.
# Output: $T/evals_hldupx/<run>/<protocol>/<run>/metrics.jsonl
```

Before reading:

- every `config` row says `"epoch": 500`, with 180 `infer_step` rows per protocol;
- the score logs show `drop_duplicate_cells ... fingerprint checked`;
- each baseline's saved resolved_config shows U_inf_dir;
- protocol non-degeneracy: posed changes the input for every arm (it did on the standing);
- if short QoS is preferred for scoring, the standing scorer took 3 h normal QoS for 6 rows; split the blocks.

Comparators are the stage-1 (b) and (c) rows in `score_summary.json`. The rule, read as registered (PREREG section
"Rule"): G = seed-mean baseline / ISLA, against the references GT 0.917 / 0.922 and T 0.924 / 0.909
(pressure / wall shear). Predictions P1-P5 are in the PREREG.

## Open items

- The hole (RESULTS section 4): the files hold the root outline as vertices (about 2,950 orphan on-plane vertices per
  broken root); the root triangles are missing. A robust rebuild (chaining orphans into the open loops) is about 2-4 h
  of future work and has not been started.
- The NOTEBOOK_DRAFT for stage 1 was second-read (`REVIEW_STAGE1.md`); its fixes are applied.
- UPSTREAM_DRAFT.md stays a draft for Peter; it has not been sent.
