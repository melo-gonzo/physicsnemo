# BC-SUPPLY: SHIFT-SUV's box, HiLift's far field and the deduplicated wing in canonical ISLA

This is a research lane of the ISLA program. The supervisor (01a13c2c) launched it on 2026-10-24 at about 10:30 EDT. I
wrote this report at 12:25 EDT and updated it at 12:50 EDT, after merging psharpe/isla eba038f08 and rerunning the final
suites.

- Branch: `psharpe/isla-bc-supply`, head **0a706b148** (ee9237933 merged with psharpe/isla eba038f08), pushed. Worktree: `/home/psharpe/gh/physicsnemo-isla-bc-sidecar`.
- On LHR, `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`.
- `JOBS.md` lists every job, harness md5 and snapshot, and `MD5S.txt` the md5s of this folder's files.

## 0. What changed from the brief

- **HiLift's far field is a recipe transform, not 690 sidecars.** The supervisor approved this at 10:42 EDT on my
  need_decision, with 8 conditions; every condition is met below.
  - Curator ee6aeda's `HemisphereBoundaries.generate` builds the inlet and the outlet from four inputs only: the radius,
    `n_theta`, `n_phi` and `U_inf`. The wing enters only the symmetry plane, which is dropped.
  - The census found 10 distinct `U_inf` vectors, bit for bit, over all 1,800 cases, one per angle of attack.
  - Per-case files would therefore have been 10 distinct far fields copied 690 times.
- **What the branch builds.** `AddHemisphereBoundaries` (in `src/domain_transforms.py`) vendors curator's `build_hemisphere`
  and `split_by_freestream` with their arithmetic and tie rule unchanged. Its comment names the source file and the commit,
  and curator is not imported.
  - It is the first transform of `highlift_surface.yaml`, in raw inches: radius 24533.41796875 in, `n_theta` 64, `n_phi` 32,
    split by each case's own stored `U_inf`.
  - Its cost is 3.5 ms per load. It needs no file, no fingerprint and no manifest, and it works on every copy and every case.
- **Condition 6: the scaffolds are removed.** BC-SIDECAR's `domain_boundaries_root` reader option,
  `datasets/highlift_surface_whole.yaml`, `conf/model/isla_surface_highlift_whole.yaml`,
  `dataset_paths.highlift_aero_ml_boundaries`, their tests and their README entry are gone. BC-SIDECAR's `walk_path` fix
  stays. Its 7 sidecars stay on disk under `$T/bcsidecar/hilift` as its record.
- **Canonical `isla_surface_highlift.yaml`** reads the deduplicated wing, `inlet` and `outlet`, and predicts at the wing.
  - It reverses the wing's winding for ISLA only (`dataset_append_transforms: FlipBoundaryOrientation`). The wing is wound into
    the body: its signed volume is negative on 6 of 6 deduplicated -medium cases (`o` checks).
  - Its comment says that the symmetry plane is absent, and why: the stored roots are open on every geometry but LHC001, and a
    rebuild of the root outline from the stored vertices is being prototyped. No case reads a symmetry plane.

## 1. Merges and their per-path checks (`merges/`)

| merge | commit | result |
|---|---|---|
| origin/psharpe/isla-bc-sidecar 12fce7bfc into 921472488 | 6707e866a | clean; every one of the 186 paths equals a parent and the fresh `git merge-tree` result (170 from psharpe/isla, 14 from bc-sidecar, 2 clean line-level combinations: `domain_transforms.py`, `test_isla_configs.py`) |
| origin/psharpe/isla-hl-dedup add3da918 | 63d6a2ec0 | run with rerere off; conflicts in `src/merge_global_data.py` (5 hunks) and `datasets/dataset_paths.yaml` (1); the other 109 paths equal a parent and the fresh merge |

**How the conflicts were resolved.** Every line of both sides is kept (`merges/merge2_lines_dropped.txt`). The file lists only
the lines that were replaced: two import lines widened, a docstring word made plural, and one call renamed.

- The rename: hl-dedup's `__getitem__` (the deduplicated surface read) becomes `_read_surface`, and bc-sidecar's `__getitem__`
  calls it in place of `super().__getitem__`.
- So with `drop_duplicate_cells` off, the path is exactly bc-sidecar's. With `domain_boundaries` empty, it is exactly
  hl-dedup's.
- Neither side's behaviour changes. HL-DUP's test file and BC-SIDECAR's tests pass on the merge.

After the merges come two commits of my own:

- e7667d24b: the transform, the canonical config, and the removal of the scaffolds;
- ee9237933: the radius evidence written into the dataset file's comment.

- 0a706b148: merge of origin/psharpe/isla eba038f08 (the book batch 2c908851a and KERNEL-TESTS' fix of
  `test_isla_kernels.py`), at the supervisor's request. It ran with rerere off and was clean: all 42 paths equal a parent and
  the fresh merge (`merges/merge3_isla_eba0.tsv`). In code it changes only `test_isla_kernels.py`.

The branch does not touch the files that lane DEFAULT-DTT is changing (`model.py`, `conf/model/isla_volume*.yaml`,
`isla_lug.yaml`, `isla_tensile2d.yaml`).

## 2. SHIFT-SUV

- **Composes and trains on the merged head.** The model code is 921472488's, with `prediction_boundaries` required and the
  token axis padded; the branch changes neither `physicsnemo/` nor `test/`. `isla_surface_shift_suv` reads the vehicle and
  the five box pieces and predicts at the vehicle (`test_isla_configs`). Its 20-step GB300 smoke is finite (section 5).
- **The box moves with the vehicle.** I checked this on real cases in job 369235 (`out/move_369235.jsonl`, 0 failures):
  - all 5 dataset files, 2 cases each, augment off and on, so `shift_suv_surface_posed`'s FixedRandomPose and every file's
    CenterMesh and L_ref scaling are covered;
  - method: one similarity is fitted from the raw box onto the output box. It maps the box to within 2.9e-6, and the raw
    vehicle onto the output vehicle to within 3.0e-7, in output units where the box spans about 16. Rotations go up to 170°,
    and the scale is 1/L_ref.
  - The recipe test `test_shift_suv_pipeline_moves_the_box_with_the_vehicle` checks the same on synthetic data for each file.
- **The baselines are untouched.** Job 368882 compared GeoTransolver's and Transolver's collated `forward_kwargs` and targets
  before the box (921472488) and after it (the head e7667d24b):
  - 5 dataset files × 3 cases (0, 500, 990) × both models, augment off and on (the global seed fixed per case);
  - **60 of 60 items are bitwise equal** (`out/slurm_isla.bcsupply.base_368882.log`).
- **Cost.** `AddBoxBoundaries` takes 1.93-2.05 ms per load (median of 50 calls, 8 threads, LHR CPU) on all 5 files, whatever
  the surface's size (0.2-5.0 M cells). It adds 4,332 cells: 722 for each of inlet, outlet, floor and roof, and 1,444 for
  side_walls.

## 3. HiLift far field: validation (`out/hl_val_368914.jsonl`, `gb/hl_val_gb.jsonl`; 0 failures)

The checks ran in job 368914 on an LHR CPU node (aarch64), and (a) and (b) ran again on the GB300 training node in job 368918.

- **(a) The stored far fields.** On LHC001_AoA_10 and _12, in both the full dataset and the -medium copy (8 meshes), the
  transform's inlet and outlet equal the stored ones bit for bit: 1,024 points and 1,952 cells each, as sorted points and as
  oriented cells. The stored *order* differs, because it is curator's RandomPermutationFilter, which was run on the files. So
  equality is checked order-free, and the stored points are an exact permutation of the built ones.
  - The -medium copies of LHC001_AoA_10 and _12 store their own far field. The transform's pieces are the ones read, and
    they are equal to the stored ones.
- **(b) Curator's own code.** For each of the 10 distinct `U_inf` vectors, the transform's inlet and outlet equal those of
  curator's `build_hemisphere` and `split_by_freestream`, run on the same node, bit for bit and in order (20 of 20).
- **(c) The census** (`out/uinf_census.json`, md5 9d1aa432): every one of the 1,800 full-dataset cases and of the 690
  -medium cases stores one of those 10 vectors, bit for bit (180 cases per angle in the full dataset). So (b) covers every
  case.
- **The radius against the stored interiors.** 31 full-dataset cases list an interior (`out/stored_pieces.txt`, md5 9ae83e4f).
  Of those, 15 are a bare `meta.json` stub and 3 have truncated memmaps (LHC002_AoA_10 in both copies, and LHC002_AoA_20).
  - On all **13 that hold points** (LHC001_AoA_4/8/14/20/22, LHC002_AoA_12/14/16/18, and the -medium copies of 4 of them), the
    extent along -x, +x, -z and +z equals 24533.41796875 to the bit (0 ulp).
  - Between those axes the CFD far field's own nodes reach 1.0188 R (90.6 MAC), with 9,284 nodes beyond R on every case
    (`out/iprobe_LHC001_AoA_14.json`). The stored interior is therefore a polyhedral far field whose axis vertices lie at R.
    The radius is the one curator used for the stored pieces, not a fit to the CFD mesh.
- **(d) Recipe tests.**
  - `test_add_hemisphere_boundaries_reproduces_the_stored_far_field` runs on real files (LHC001_AoA_10, LHC001_AoA_12, and
    LHC002_AoA_10 against LHC001_AoA_10's stored far field) when `HIGHLIFT_AERO_ML_ROOT` is set. It ran and passed in 369258.
  - Synthetic tests check the far field's geometry, its normals into the fluid, the upstream/downstream split, that the wing
    does not enter, and what the transform refuses.
  - A pipeline test moves the far field with the wing, native and posed.
- **The far field moves with the wing** on real cases (369235): `highlift_surface`, augment off and on, and HL-DUP's cluster
  files `highlift_surface_medium_dedup` and `_posed` (FixedRandomPose). The similarity residual is ≤1.4e-5 on the far field
  and ≤3.9e-6 on the wing, where the far field spans about 94.
- **Baselines' HiLift inputs.** Job 368882 compared before (63d6a2ec0) and after (e7667d24b) on 3 dataset files × 3 cases ×
  both models: **24 of 24 bitwise**. HL-DUP's two cluster files were run with augment off only, because their
  `RandomTranslateMesh` bounds are plain lists, which torch's Uniform refuses under augment, in both trees alike. HL-DUP's
  lanes run with augment off.
- **Which cases a configuration reads.** The canonical configurations read the -medium copy's 690 cases (510 train, 180 val)
  through the cluster `dataset_paths`. No configuration reads the other 1,110 cases, and the transform would treat them
  identically.
- **Cost.** 3.5 ms per load (median; at most 4.2 ms). It adds 3,904 cells (inlet 1,952, outlet 1,952). BC-SIDECAR's per-case
  generation cost 7-40 s and 17-19 GB; that cost is now gone.
- **Cluster dataset files.** HL-DUP's `highlift_surface_medium_dedup{,_posed}.yaml` exist only on LHR. In my snapshot
  `$T/recipe_bcsupply_ee92`, I add the transform first to each (`lhr/make_cluster_files.py`) and use
  `SurfaceMeshToDomainMesh` as the terminal. HL-DUP's own snapshots are untouched. The supervisor should apply the same
  edit when the canonical runs start.

## 4. Tests on the head

The final suites ran on the merged head 0a706b148, with code snapshot `$T/code_bcsupply_0a70` and recipe snapshot
`$T/recipe_bcsupply_ee92`; the recipe tree is identical at both commits.

| suite | where | result |
|---|---|---|
| whole ISLA test directory, CPU | LHR cpu-short 369519 | 397 passed, 122 skipped, **0 failed** |
| recipe tests | LHR cpu-short 369520 | 579 passed, 12 skipped (all CUDA), 2 failed |
| whole ISLA test directory, every CUDA test | GB300 369518, short QoS | 527 collected: 526 passed, 1 skipped, **0 failed** |
| hooks on the 19 files the head changes against psharpe/isla | local | all passed |

- **How the GB300 run fit the cap.** Only 0.91 GPU-h were left, and the directory takes 29 min on one GPU. So job 369518 split
  the collected tests into 8 time-balanced shards (`lhr/shard.py`: weights from run 368918; every test in exactly one shard),
  ran 2 pytest processes per GPU with their own compile caches, and finished in 6:55 (`gb2/`). Its count matches
  KERNEL-TESTS' run of the fix (526 passed, 1 skipped). The one skip is the posed-sample test that needs
  `ISLA_AF_POSED_SAMPLE`.
- **Before the merge** (ee9237933 on 921472488's model code), the same suites gave: CPU 395 passed and 0 failed (369270);
  recipe tests the same 2 failures (369258); GB300 517 passed and 5 failed (368918). Those 5 were the `test_isla_kernels.py`
  failures that KERNEL-TESTS fixed, with its assertion messages.

- **The 2 recipe failures** are `test_campaign_c_hooks` `initialize_slurm` TypeErrors, which occur only inside a Slurm job.
  They fail with the same message as BC-SIDECAR's 363575 and the base's 362118.
- **The real-data tests ran.** `HIGHLIFT_AERO_ML_ROOT` was set, and all 12 recipe skips are CUDA-only, so the three
  real-data far-field tests ran and passed.

## 5. GB300 smokes (job 368918, 20 steps, bfloat16, eager with compile_blocks, seed 42; `gb/smoke_summary.txt`)

| arm | losses finite | loss, step 1 to 20 | median step, steps 6-20 | peak memory | tokens |
|---|---|---|---|---|---|
| SHIFT-SUV body (box built, not read) | yes | 0.151 to 0.068 | 168.7 ms | 64.85 GB | 200,000 |
| SHIFT-SUV whole (vehicle + 5 box pieces) | yes | 0.138 to 0.071 | 217.4 ms (1.29x) | 67.59 GB (1.04x) | 204,332 |
| HiLift body (deduplicated wing) | yes | 0.602 to 0.231 | 782 ms | 30.98 GB | 93,364 |
| HiLift whole (wing + inlet + outlet) | yes | 0.640 to 0.243 | 160 ms | 32.23 GB (1.04x) | 97,268 |

- **SHIFT-SUV** ran on `shift_suv_surface` at 200,000 cells with augment on.
- **HiLift** ran on one case, geo_LHC002_AoA_18 of -medium, at 100,000 cells drawn, with augment off. The deduplicating reader
  logged a fingerprint check: 18,787,893 of the 278,199,578 stored cells are extra copies.
- **The HiLift step times measure data loading, not the model.** hl_body ran beside the two SHIFT-SUV arms, which loaded data
  with 12 workers each. hl_whole ran alone after them, on a 278 M-cell wing. So the 0.21x does not measure the far field's
  cost.
- **The SHIFT-SUV 1.29x** comes from 2.2 % more tokens. BC-SIDECAR measured 1.19x on the same pair.

## 6. The book (`book/BOOK_CHANGES.md`; I did not edit the book)

I ran chapters 05, 06, 07, 10, 11, 20 and index on the base eba038f08 and on the branch merged with it.

- **Failing cells.** Chapter 20 fails 3 of its 36 cells, all from one cause. `para-limits-whole-boundary` asserts that one
  configuration reads the body alone (HiLift), which is no longer true. `para-oq-far-field` and `para-oq-whole-boundary` then
  fail on its undefined `BA`.
- **Printed values that change.** 05's and 06's boundary tables gain HiLift's `inlet` and `outlet` and the
  `isla_surface_shift_suv` rows, and 06's "8 ISLA configurations" becomes 9. These are generated, so they need no edit.
- **Static prose to edit** is in 20 (two places), 07 line 649, 10 line 289, 11 lines 172 and 175, and 21 line 283. The text for
  each is in the file.

## 7. Compute

- **GPU: 2.55 GPU-h**, under the cap of 3: smokes and tests 368918 (31 min) and the final tests 369518 (7 min).
- **CPU: 104.4 core-h allocated** (`out/sacct_bcsupply.txt`). Of that, 91.8 core-h are the two GB300 nodes' CPUs and 12.6 the
  cpu-short jobs (at most 2 at once).
- One cpu-short submission was refused on the user limit and retried (`lhr/submit_tI.sh`).
- I cancelled one job of my own, 368927 (a probe that was too slow).

## 8. What stays on disk

- The worktree `/home/psharpe/gh/physicsnemo-isla-bc-sidecar`, on branch psharpe/isla-bc-supply, clean.
- This folder (2.2 MB).
- On LHR: `$T/bcsupply` (harness and outputs, small), `$T/kernel/bcsupply` (small), the read-only recipe snapshots
  `$T/recipe_bcsupply_{b921,b63d,e766,ee92}` (about 1.5 MB each), and the code snapshot `$T/code_bcsupply_0a70`.
- Nothing over 1 GiB.
