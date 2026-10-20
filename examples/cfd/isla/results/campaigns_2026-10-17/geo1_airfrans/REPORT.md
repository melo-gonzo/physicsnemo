# GEO1-AF launch report (2026-10-17 lane; local clock 2026-10-16 23:10 to 2026-10-17 00:10 EDT)

Pre-registration: `PREREG.md` (approved 2026-10-17T03:44Z, before any lane was submitted). Jobs: `JOBS.md`.

## STEP A (git)
* `git merge --no-ff --no-verify psharpe/isla` (e6cec857e) into both worktrees, no conflicts:
  psharpe/isla-a2 89af778ee -> **ebf841578**, psharpe/isla-geo-exact 0a0b6a176 -> **c7adf4d7a**.
* Tests (CPU, nice 19, `PYTHONPATH=<worktree>`, `.venv` python; `tests.log`):
  a2 ISLA 219 passed / 65 skipped, a2 recipe 483 passed / 12 skipped; geo-exact ISLA 231 passed / 52 skipped,
  geo-exact recipe 487 passed / 12 skipped.
* Pushed both (plain push) to origin (peterdsharpe/physicsnemo-private).

## STEP B (LHR snapshots, `T=.../agents/2026-08-09-mt2-stage0`)
| dir | source | files | md5 of MD5SUMS |
|---|---|---|---|
| `$T/code_a2s` | git archive ebf841578 (physicsnemo pyproject.toml test) + tensordict 0.15.0 | 3,237 | 13c2c18f17e1ecdcff89582601d73183 |
| `$T/code_geo1` | git archive c7adf4d7a (same paths) + tensordict 0.15.0 | 3,237 | d35b1cb9b4aa6f8d7bedff9e38ced0ff |
| `$T/recipe_v11` | git archive c7adf4d7a:examples/cfd/external_aerodynamics/unified_external_aero_recipe, unmodified | 161 | 38ed2a7951f8d4ca30b9654a97d9f835 |

Archive md5s (local = LHR): code_a2s.tar 330cbc78, code_geo1.tar a99802ad, recipe_v11.tar 36cdd1d5 (`snapshot_md5s.txt`).
TensorDict 0.15: `lanes.sbatch` overwrites PYTHONPATH with `<code>:<recipe>/src` and may not be edited, so td015's
`tensordict/` and `tensordict-0.15.0.dist-info/` are copied into each code snapshot (noted in SNAPSHOT_HEAD). Both
`.venv-recipe` and `.venv-recipe-td14` are torch 2.12.0+cu130 and warp 1.14.0; Warp imports, and
`physicsnemo.mesh.spatial.signed_distance_field` imports.

## STEP C (LHR short QOS, jobs 243376 and 243405; `check_geo1.py`, `check.sbatch`, `check_dv.sbatch`, `check_logs/`)
* Eager fwd+bwd, one airfrans_uvolume_2d case (100,000 queries, 1,007 segments), bf16: finite for both arms; 306
  parameters, all with nonzero finite gradients (grad norm 10.7 a2s, 11.1 geo1).
* geo1 geometry against a float64 brute force over the airfoil's segments: max abs error 3.1e-7, median rel 6.5e-8,
  100 % within 1 %, d >= 0 everywhere (wall nodes d = 0), |u| = 1 to 1.2e-7. a2s on the same case: 0.27 % d <= 0
  (min -0.0041), 65 % within 1 % of the exact distance.
* geo1 on `.venv-recipe` and on `.venv-recipe-td14`: the same loss to every digit (0.38506099581718445).
* `model.compile_blocks=true`, two cases: finite, both arms.
* Whole-model `torch.compile(fullgraph=True, dynamic=True)`, mode default, fwd+bwd on two cases: `unique_graphs = 1`
  after both, no graph break, no recompile, both arms (a2s 2,726 / geo1 2,990 captured calls). The
  `rel_diff_vs_eager_bf16` field in the log compares different random interior draws, so ignore it.
* DrivAerML Volume, code_geo1, `model=isla_volume dataset=drivaer_ml_volume_reference` (recipe copy
  `kernel/geo1_af/recipe_v11dv`: recipe_v11 + recipe_v8's LHR dataset_paths.yaml; recipe_v11 has `???` there): full STL
  753,234 triangles / 376,531 points, 200,000 vehicle tokens, 200,000 queries, 7.8 s eager forward, finite. The vote
  flipped the STL; afterwards it agrees with 99.995 % of the token measure (weighted cos 0.9989). |d| = |stored sdf|
  (median difference 1.9e-9), **but d < 0 in 98.5 % of the fluid (d = -sdf)**: the vehicle tokens' normals point into
  the car and the sign follows the tokens, as today's nearest form does. This is a finding for the DrivAerML half of
  Step 1. It does not affect these lanes.

## STEP E (lanes)
* Backups: `kernel/airfrans_u/lanes.tsv.bak-2026-10-17-pregeo1` (md5 1eb95630), `score.sbatch.bak-2026-10-17-pregeo1`
  (0136010b). lanes.tsv: rows 10-13 appended (md5 030c8422). score.sbatch (md5 f629ffc5): afu_a2s_* -> code_a2s and
  afu_geo1_* -> code_geo1, both on recipe_v11 (`RECIPE_DIR` is now per run) and `.venv-recipe`. The recipe also goes in
  the START log line.
* Submitted 20:46 PT with submit.sh's sbatch line plus `RECIPE=$T/recipe_v11`. These are the first links; each started
  its singleton next link:

  | lane | first link | next link (pending) |
  |---|---|---|
  | afu_a2s_isla_seed42 | 243455_10 | 243462_[10] |
  | afu_a2s_isla_seed43 | 243456_11 | 243461_[11] |
  | afu_geo1_isla_seed42 | 243457_12 | 243463_[12] |
  | afu_geo1_isla_seed43 | 243458_13 | 243468_[13] |
* Accepted: by 21:03 PT every lane was at epoch 7-8, at least 1,400 steps, with no traceback. Steady step time is
  a2s 0.388-0.390 s, geo1 0.402 s (afu_bn2: 0.386 s). Memory is 37.7-38.0 GB per GPU. The resolved config has
  compile_blocks true, hidden 512, n_slices 512, and the right seed.
* Expected finish, all four lanes: about 81 s per epoch, so 11 h of training. With gapless 3:59 links that is about
  08:00 PT (11:00 EDT) on 2026-10-17; with ordinary queue gaps (afu_bn2 took 13 h), about 08:30-11:00 PT. geo1 runs
  about 3 % slower than a2s.

## Post-training scoring: exact commands (LHR)
```
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
# native / area / posed, once all four train.logs say "Training completed" (3 GPUs per run, ~10 min):
bash $T/kernel/airfrans_u/score_submit.sh afu_a2s_isla_seed42 afu_a2s_isla_seed43 afu_geo1_isla_seed42 afu_geo1_isla_seed43
#   -> $T/evals_afu/<run>/{native,area,posed}/<run>/metrics.jsonl
# 400-segment CHAR-AREA2 ca2a / ca2u / ca2v, 12 units, 3 per GPU in one 4-GPU task (~15 min):
cd $T/kernel/geo1_ca2 && sbatch --array=0 --qos=short --time=01:59:00 --export=ALL,CA2_U=3 score.sbatch
#   -> $T/kernel/geo1_ca2/evals/<run>/<ca2a|ca2u|ca2v>/<run>/metrics.jsonl (+ <protocol>.log with 201 guard lines)
```
How geo1 receives the whole airfoil: `kernel/geo1_ca2/recipe_v11a` is recipe_v11 plus two changes:
* the CHAR-AREA2 `GuardedPoissonSubsampleMesh`, appended to `src/domain_transforms.py`, with one addition,
  `keep_whole: {airfoil: airfoil_surface}`. Before the draw it copies the unsampled airfoil to
  `boundaries.airfoil_surface`, and the draw never touches that copy. It consumes no random numbers, so the draws are
  the CHAR-AREA2 ones;
* the three `airfrans_uvolume_2d_ca2{a,u,v}.yaml` twins, which are the char_area2 files plus that `keep_whole` line.

The geo1 rows of `rows.tsv` add `+forward_kwargs.boundary_geometry.airfoil=boundaries.airfoil_surface`. The a2s rows
read the same files without that override.

Smoke test, job 243551, on epoch-3 checkpoint copies (since deleted):
* geo1 ca2u with the override and a2s ca2u each scored 200 cases with 201 guard lines. Both arms kept identical cells
  (same idx_crc), so they stay in lockstep.
* geo1 without the override refused, as designed: "boundary_meshes['airfoil'] is a sample of its boundary ... pass the
  whole boundary as boundary_geometry".

Then c = ca2u / native (native from `evals_afu`), r = ca2a / ca2u and d = |ca2v / ca2u - 1|, per field and seed, read
against PREREG.md.

## Left on disk
* LHR: `code_a2s` and `code_geo1` (64 MB each), `recipe_v11`, `kernel/geo1_af` (114 MB: the archives, check logs and
  recipe_v11dv), `kernel/geo1_ca2` (2.4 MB). Runs go to `$T/runs/afu_{a2s,geo1}_isla_seed4{2,3}`. The lane mechanism
  keeps every 2-epoch checkpoint, as afu_bn2 did (130 GB per run, 503 files), so about 520 GB for the four lanes. The
  account share has 24 of 110 TB used and 9.1 of 52 M files.
* Local, this directory: about 1 MB (the archive copies were deleted; LHR keeps them in kernel/geo1_af).
