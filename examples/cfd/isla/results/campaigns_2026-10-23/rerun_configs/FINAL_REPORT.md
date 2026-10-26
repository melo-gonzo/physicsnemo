Status: DONE (RERUN-CONFIGS; written 02:45 EDT 10-26)

## Branch
`psharpe/isla-rerun-configs`, head **b7196f0d2**, off psharpe/isla bf71190af, worktree /home/psharpe/gh/physicsnemo-isla-rerun-configs.
Pushed to origin with a plain push; `git ls-remote` showed no such branch before the push and shows b7196f0d2 after it. Not merged.
Every commit was made with `-s`, author Peter Sharpe, no trailers, and all hooks passing.

| commit | adds | lines |
|---|---|---|
| 4486b32ab | The 14 cluster-only dataset files the reference runs read, committed as they ran, with only the license header prepended: ginot_lug_volume, plaid_tensile2d_2d and deepjebpp_surface, each with its _posed and _area twins; highlift_surface_medium_dedup{,_posed} (HL-DUP), _area (HL-AREA); drivaer_ml_surface_area and shift_suv_surface_fastback (still stale here) | +1,343 |
| 28652b50a | The whole boundary. drivaer_ml_surface_area gets the tunnel inlet, no_slip, outlet and slip, plus PlaceGroundUnderBody in the reader, copied verbatim from drivaer_ml_surface.yaml. shift_suv_surface_fastback gets AddBoxBoundaries, verbatim from shift_suv_surface.yaml. highlift_surface_medium_dedup, _posed and _area get AddSymmetryPlane and AddHemisphereBoundaries (HL-SYMPLANE's and BC-SUPPLY's blocks). All end in SurfaceMeshToDomainMesh. dataset_paths.yaml gains `shift_suv_fastback_root: ???` | +101 / -7 (6 files) |
| 944fd6c7c (AZ7 child) | AeroZoo's seven fields: intermittency and retheta become targets in the 7 dataset files, with per-split scale-only std and ISLA output_scale; the ISLA model config gets two outputs; GeoTransolver comment only (out_dim 9 comes from `targets:`); Transolver unchanged; recipe tests pin seven fields; fit tools and fits JSON (md5 5e78c599) | +780 / -38 (14 files) |
| b7196f0d2 | merge of psharpe/isla-rerun-configs-az | — |

No physicsnemo/ and no recipe src/ changes; the baselines' code is untouched.

## Checks
- Recipe tests, local CPU, td0153: 631 passed, 24 skipped, on the files of 28652b50a (`logs/recipe_tests_1.log`), on 944fd6c7c (AZ7) and on the head (`logs/recipe_tests_head.log`).
- LHR cpu-short loader smokes and leaf A/B: jobs 397603, 397604 and 397696 (mine; `JOBS.md`) and 397598/397599/397691 (AZ7). Code snapshot code_bf71 was built by the supervisor's rsync method.
  - The smokes list the boundary pieces and every leaf's shape: `logs/leaf/leafshapes_B_397603_397604.txt`, `az7/SMOKE_AB.md`.
  - ISLA DrivAerML area reads vehicle, inlet, no_slip, outlet, slip. SUV fastback: vehicle, inlet, outlet, floor, side_walls, roof. HiLift native/posed/area: vehicle, inlet, outlet (1,952 cells each), symmetry (1,032).
  - lug/Tensile2d/DeepJEB++ posed, read from git: fixed, hole, loaded, free / pinned, loaded, fixed, free / fixed, free, loaded.
  - ISLA on today's files fails on all five, as before.
- A/B against today's files (3 train + 2 val batches):
  - GT/T on DrivAerML area, SUV fastback, HiLift native and HiLift posed: BITWISE.
  - **GT/T on HiLift area: the leaves MOVE.** The cause: the dataset forks one generator per transform *by position* (mesh_dataset.set_generator). AddHemisphereBoundaries at transforms[0] shifts the Poisson draw's stream (195,353 against 195,660 cells). ISLA and the baselines share the new draw.
  - **The same is already in git for SHIFT-SUV area since BC-SUPPLY's box (456b19260).** Job 397696 compared the pre-box file with today's: 3/21 train and 2/14 val leaves bitwise. So the SUV area standing's baselines were scored on a different draw from the one ISLA's rerun score will use. No config-only fix exists (the reader's whole-surface domain has no global_data).
  - AeroZoo seven fields: every common leaf is bitwise for ISLA, GT and T on all 7 splits; only targets/{intermittency,retheta} were added. A one-batch forward and loss ran for all three models (GT/T out [1, 20000, 9]).

## Registered lists
- `LANES.tsv` (md5 acf31b08): 52 lanes, 32 ISLA and 20 baselines (HiLift GT/T; AeroZoo GT on 7 splits; T on full).
  - Every lane is bf16 with `compile=false` from the launcher. ISLA lanes add model.compile_blocks=true, except Tensile2d, which is eager.
  - num_workers=12 and prefetch_factor=12 on HiLift (all three architectures) and on DeepJEB++; AeroZoo already had them; the rest as registered.
  - Each row names its registration (REF-090, AZ PREREG_v2, HL-DUP, BC-SUPPLY, HL-SYMPLANE). Script: `mk_tables.py`.
- `SCORING.tsv` (md5 191f6fc1): 120 jobs.
  - Protocols per standing.py; area and posed files are `<dataset>_<protocol>`, as the REF-090 scorer requires.
  - AeroZoo: specscore.py on every cell (ε_U, ε_p, ε^N, mean and median, all cases and mean_stable), plus relL2 of all seven fields at 200k on `<split>.test`.
  - Also 4 recommended SUV GT/T area rescoring rows, and the suv_xfam extra.

## Re-pricing (`price/PRICE.md`, child PRICE; from iter_time_s or metrics.jsonl wall-time steps, never the host step)
- ISLA's 32 lanes: **3,160 GPU-h** (±10 %), against FREEZE-PLAN's S2 of 3,380. Only the lug is measured on code at or after 9f279664c (390719, 268 ms); every other step is an estimate, marked with its source.
- The baseline retrains add **2,641 GPU-h**: AeroZoo GT 2,185, T 313, HiLift 140. **Total about 5,800 GPU-h**, above the 4,000 Peter was told. 13 nodes for 5.5 days, or 8.5 days at 8.

## Open questions for the supervisor
1. HiLift and SHIFT-SUV area draws: accept the new draws and rescore the non-retraining SUV GT/T standing checkpoints on area (about 1 GPU-h; already in SCORING.tsv as PENDING)? The alternative is a recipe change that keys the forks by transform identity. I recommend the rescore.
2. Cost: about 5,800 GPU-h with the seven-field baseline retrains. Do you tell Peter, or cut AeroZoo GT seeds or splits?
3. HiLift ISLA training file: I registered highlift_surface_medium_dedup, the same file and manifest as the baselines and the HL-DUP standing. Git's highlift_surface.yaml (the root manifest's medium_* keys, no DropDegenerateCells) is the other choice; premerge's leafcfgs used it.
4. specscore.py hard-codes five fields and would silently skip γ and Re_θt. Apply `az7/SPECSCORE_PROPOSAL.diff` before scoring. The spec defines ε only for U and p; the γ and Re_θt ε rows (scale 1) would be our extension, with relL2 the main number for them.
5. HL-DUP stage 2 lanes: if you accept them as the HiLift baseline standing, drop fin_hl_gt/t (140 GPU-h). Note, though, that HL area then needs the rescore of question 1.
6. DeepJEB++ splits are train_clean/test_clean, following REF-090; ISLA-PREMERGE's leafcfgs used train/test.
7. `$T/kernel/premerge` (premerge's snapshots and 36 GB of A/B files) disappeared from LHR around 01:40 EDT, which I did not do. Say whether that was expected.

## Rulings relied on
Peter's whole-boundary ruling (10-23) and his approvals of the box, far field and plane (10-24); the supervisor's seven-field decision and the num_workers/prefetch 12 approval (this brief); LANE_RULES (CERT reading: Tensile2d eager); the supervisor's 02:10 snapshot tip.
For memory: "the pipeline forks transform generators by list position: inserting a transform before a stochastic one changes its draw."

## Left on disk; nothing running
- Worktree /home/psharpe/gh/physicsnemo-isla-rerun-configs (1.1 GB). Remove it after the merge. AZ7's worktree is removed; its branch psharpe/isla-rerun-configs-az stays local, merged.
- LHR $T/kernel/rerunconfigs (782 MB: snapshots and probe records), $T/kernel/az7_ab (3.5 GB, AZ7's probe records) and $T/kernel/az7_fit (4 MB). These can go after the merge.
- The registry row has been removed. GPU used: 0 GPU-h.
