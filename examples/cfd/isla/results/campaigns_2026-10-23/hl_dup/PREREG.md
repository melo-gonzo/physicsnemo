# HL-DUP stage 2 pre-registration (proposal; nothing submitted)

Written 2026-10-24 at about 06:25 EDT, after stage 1 (`RESULTS.md`). Known before writing: all of stage 1, including
the (a), (b) and (c) scores of the six standing checkpoints. No deduplicated-data training has run. The trigger in the
brief is "duplicates carry more than 1 percent of the drawn measure". It is met: the copies carry 42.1 % on average and
3.2 % at the minimum, and more than 1 % on all 180 scored cases.

## Question

Trained and scored on the wing without its duplicated cells, where do ISLA and both baselines stand on HiLift? In
particular, does baseline / ISLA (0.92 on pressure and wall shear, both with and without the copies in the metric) move?

## The correction: a recipe transform, never the dataset

It works like `PlaceGroundUnderBody` on DrivAerML: the files stay as they are and the recipe removes the defect at load.

- **Exclusion index.** For every case, an index of its extra copies: `np.packbits(extra)` over the stored wing cells,
  where `extra` marks every cell whose sorted vertex triple repeats a lower-indexed cell's.
  - It is computed once, read-only, by `lhr/hl_census2.py` (exact grouping, verified).
  - It is stored as a sidecar laid out by case under `$T/hldup/masks/`. The 180 val cases exist; the 510 training cases
    are still to compute.
  - It is a data-quality index, not a geometry input: no model reads it, and ISLA still computes every geometry
    quantity from the boundary it receives.
- **Reader option** `exclude_cells_from:` on `MeshReaderWithGlobalData`.
  - The cyclic block of k cells is drawn over all N stored cells, exactly as today (same generator, same block).
  - The drawn extra copies are then dropped, and the kept cells' Horvitz-Thompson factor stays N / k.
  - Each first copy keeps inclusion probability k / N, so the measure-weighted sums estimate the deduplicated wing
    without bias, and only the block's slice of the mask is read (about 25 KB).
  - The token count becomes about 186,000 (6.4-7.5 % fewer, varying by case).
  - The option goes in the shared dataset config, so every arm gets the same data: a data correction, not an
    ISLA-specific feature.
- **Dataset configs.** `highlift_surface_medium_dedup.yaml` (and `_posed`) are `highlift_surface_medium.yaml` plus that
  option. Nothing else changes; `DropDegenerateCells` stays.
- **Tests.** On a synthetic mesh with planted copies:
  - the transform's measure sum equals the deduplicated mesh's area in expectation, checked to 1e-3 over 2,000 draws;
  - the draw equals today's when the mask is empty;
  - a mask of the wrong length is refused.
  The tests run in the recipe suite on LHR cpu-short.

## Arms (two seeds each, 42 and 43; 500 epochs; the standing's hyperparameters, precision and node type)

| arm | code / recipe | as the standing row |
|---|---|---|
| ISLA 0.9.0 (dedup) | code_h090 / recipe_h090 + the transform | h090_hl_isla_seed4x |
| GeoTransolver, unit drive (dedup) | code_a2r / recipe_a2r_hl + the transform | udrv_hl_gt_seed4x |
| Transolver 512 x 16, unit drive (dedup) | code_a2r / recipe_a2r_hl + the transform | udrv_hl_t_seed4x (trains at 100k cells) |

The baselines keep their released code; only the shared dataset config changes. Their saved configs must show the
intended forward_kwargs. Job names are `isla.hldupx.*`, under a new registry row.

## Scoring

Float32 on the 180 val cases, epoch 500, with the deduplicated draw (k = 200,000 drawn, about 186,000 kept). Protocols
are native and posed. Area is degenerate for ISLA, as before, so it is reported but not read. Comparators: the
standing's checkpoints under (b), the same predictions without the copies in the metric, and (c), with the copies also
removed from the input. Both rows exist. Every config row is checked for epoch 500, and every protocol for 180 cases.

## Rule (pressure and wall shear, native; G = seed-mean baseline / ISLA, both dedup-trained)

The reference is the stage-1 (b) ratio: GeoTransolver 0.917 / 0.922 and Transolver 0.924 / 0.909 (pressure / wall
shear). Two-seed gaps are 1-16 %, so a shift counts only when both seeds agree.

- **Gap narrows:** on both fields and for both baselines, G rises by at least 0.03, and both seeds' ratios exceed the
  reference.
- **Gap widens:** on both fields and for both baselines, G falls by at least 0.03, with both seeds below the reference.
- **Unchanged:** for each baseline, G is within 0.03 of the reference on both fields.
- **Mixed:** anything else. It is reported per baseline and field, and no verdict is claimed.

Beside the rule:

- ISLA dedup against ISLA (b) and (c), per AoA band (4-12 and 14-22);
- posed / native for ISLA (the R-POSE question, 1.009 on the stored data);
- ISLA's length unit on the training cases.

## Predictions

| # | prediction | probability |
|---|---|---|
| P1 | ISLA dedup native pressure < 0.0589 (its (b)) by at least 3 % | 0.55 |
| P2 | each baseline's dedup native pressure is within 5 % of its (b) (0.0540, 0.0544) | 0.70 |
| P3 | rule verdict "unchanged" | 0.50; "narrows" 0.30; "widens" 0.10; mixed 0.10 |
| P4 | ISLA dedup is far better than ISLA (c) (0.0832): below 0.065 | 0.85 |
| P5 | ISLA posed / native stays within 1.005-1.015 | 0.65 |

## Cost

- **Masks.** 510 training cases, about 37 process CPU-h (about 45 core-h allocated), about 2.5 h wall on 2 cpu-short
  jobs. They add about 18 GB on LHR scratch; the 180 val masks hold 5.9 GB.
- **Training.** On LHR GB300, one 4-GPU node per lane, at the standing's rates: ISLA about 8 h per lane, GeoTransolver
  about 15 h, Transolver about 9 h. That is about 64 + 122 + 73 = 260 GPU-h for 6 lanes, 4-6 normal-QoS links each, and
  about one calendar day if nodes are free.
- **Smokes and scoring.** About 1 GPU-h of smokes (short QoS) and about 2 GPU-h of scoring.
- **Total.** About 263 GPU-h and 50 core-h.

## Open points for the supervisor or Peter

1. Whether a per-case exclusion index read by the recipe is allowed under the boundary-only ruling (no precomputed
   geometry input, no per-case cache). I read it as a data correction, like an orientation flip, since no model reads
   it. The exact alternative inside the loader costs about 4 min of CPU per load, which is far too slow for training.
2. This lane's deadline (11:06 EDT) is far shorter than the retrain. An approved stage 2 needs an owner for the chains,
   either a new lane or this one relaunched.

## Amendment before submission (supervisor, received 2026-10-24 06:44 EDT)

The supervisor approved stage 2 as registered (md5 d65a75bc) on these terms. Nothing had been submitted, and no
deduplicated-data number existed.

1. **The exclusion index.**
   - *The supervisor's reading* (Peter may veto it). It is a cache of a recipe data correction, like
     PlaceGroundUnderBody. No model reads it, so the boundary-only ruling of 2026-10-17 holds and the no-geometry-cache
     rule does not apply.
   - *Fingerprint.* The cache must be unable to drift. Each mask carries a fingerprint of its source: the cell count
     plus the md5 of a fixed strided sample of 1,024 rows of the connectivity. The reader checks it at every load and
     refuses a mismatch, with no fallback.
   - *Manifest.* A manifest holds the full md5 of each case's connectivity.
   - *Builder.* The mask builder lives in the recipe's src, with an exact mask-free path used by the tests. A cpu-short
     check shows that the stored masks equal a fresh computation on at least 3 cases.
   - *Storage.* The masks are read-only under `$T/hldup/masks`, listed with md5s in a manifest.
   - *Name.* The reader option is `drop_duplicate_cells:`, in the shared dataset config.
2. **Decision use.** The deduplication goes into the canonical HiLift configuration on principle (the supervisor's
   decision). Stage 2 measures its effect on all three architectures and sets the book's HiLift standing. Its verdict
   does not decide adoption.
3. **Code.** The transform, the mask builder and their tests are committed on a new branch `psharpe/isla-hl-dedup` off
   origin/psharpe/isla (`-s`, hooks on, plain push, never merged), apart from the snapshot that runs.
4. **Cost.** The kept token counts (about 186,000) are mostly not multiples of 8, so each arm is priced from its smoke.
   The cap is 320 GPU-h plus 50 core-h; a projection over the cap goes to the supervisor before submission.
5. **Root section.** Using LHC001's outline as a shared root section is declined: HiLift's symmetry plane stays out.
6. **Ownership.** This lane computes the 510 training masks, builds and tests the transform, and smokes every arm. If
   every smoke unit passes its registered checks, it submits the six lanes without waiting:
   - order: ISLA, then GeoTransolver, then Transolver, two lanes at a time;
   - at most 12 jobs, with short QoS at most 2 for the smokes;
   - registry row `^isla\.hldupx\.`, expected_end 2026-10-26;
   - HANDOFF_HLDUP.md by 10:45 EDT, then one message with the job ids.
   The supervisor then resumes this lane to own the chains, score and read.
7. **Disk and drafts.** `masks_main` is kept. UPSTREAM_DRAFT.md stays a draft for Peter.

**Smoke checks registered here** (each arm, before its lanes go). Every check must hold:

- finite losses over 20 training steps;
- the saved resolved config shows dataset `highlift_surface_medium_dedup`, the reader's `drop_duplicate_cells` set, and
  the arm's intended forward_kwargs (the baselines: `U_inf_dir`);
- the drawn boundary holds 6-8 % fewer cells than 200,000 (100,000 for Transolver);
- the fingerprint check ran (logged);
- no traceback;
- projected training cost within the cap.
