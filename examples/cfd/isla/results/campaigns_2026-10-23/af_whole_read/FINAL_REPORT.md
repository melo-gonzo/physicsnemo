Status: DONE

On AirfRANS, reading the far field makes ISLA's error worse on every field, though its pose holds. ISLA-DTT is worse still, and its pose breaks through a tie in its octant trees. 397.5 of the 520 GPU-h cap were used. All checks and the supervisor's far-field guard passed (details: `RESULTS.md`, `read.md`).

**Results (W = trained with the far field; B = the eager-trained standing; full_test, float32, two seeds):**

| pair | pressure | velocity | nut | posed / native |
|---|---|---|---|---|
| W-ISLA / B | **harms** 1.349 (abs(D)/2SE 2.48) | **harms** 1.204 (2.62) | **harms** 1.282 (4.21) | **holds**, 1.0001-1.0014 |
| W-DTT / B | **harms** 1.943 (1.77) | **harms** 1.531 (1.89) | **harms** 1.893 (1.63) | **broken**, 1.53-2.52 (B 1.002) |

- **Where it harms (the bands by wall distance).** For W-ISLA velocity and nut, the harm is at the wall:
  - velocity is 1.43x in the first decade off the airfoil (1e-6 to 1e-5 chords) and 1.01x beyond 0.1 chord;
  - below the softening's knee (3.05e-4 chords) the ratios are 1.24 (velocity) and 1.15 (nut), against 1.12 and 1.02 above it, on both seeds.

  By the registered criterion that points to the near-wall softening (1e-6 L) as the cause. W-ISLA's pressure harm is not near the wall (1.01-1.07 there, 1.28 at 0.01-0.1 chord). W-DTT's harm is spread over every band.
- **Why W-DTT's pose breaks.** I found this with a probe after the read; it is analysis, not a registered rule.
  - `tree_read.normal_octants` sorts each cell into a tree by the sign of its normal's components.
  - The far field's straight edges have normals on the frame's axes, so one component is about 2e-6 (as low as 1.7e-8), a sign tie.
  - Over the 200 test cases, posing moves 16-45 % of the far-field cells (median 29 %) to another tree, against a median of 2 airfoil cells (`out/octant_pose_summary.txt`).
  - It is the lug's octant tie that chapter 05 already describes. The frame sign does not flip (0 flips in 200 cases).
- **Pre-training facts** (all 1,000 cases, `FACTS.md`):
  - L is 305.3 chords with the far field, against 0.411-0.437 without it (725x). So 1e-6 L = 3.05e-4 chords.
  - First-node heights: median 2.10e-6 chords, 1st percentile 1.59e-6.
  - Interior points in the linear part of the softening: 20.2 % with the far field, against 0.56 % (the wall nodes only) without it.
  - z shifts by +2.86, so the constants were refit (full_train reynolds {9.065, 0.1338}).
  - Tokens: 2,406-3,140, all kept.
  - canonical_frame eigenvalue gap 0.033; axis 2's sign comes from the airfoil (skew 5e-8 to 8e-5).
- **Cost.** Per step the far field costs nothing: W / B smoke step 0.984 (ISLA) and 1.005 (DTT), memory 1.02-1.03.
- **Predictions.** I gave "harms" 0.25-0.40 on every field and it happened on all six. The near-wall predictions P8 and P9 held; P11 (W-DTT pose holds) and P12 (ISLA smoke step at least 1.00) failed.

**What ran.**
- Commit 8e93f2307 (reader, configs, refit constants, tests), then the DRV-GROUND merge 4b772ec59 at your steer. Both are on psharpe/isla-whole-boundary, pushed with a plain push.
- Tests: recipe 540 passed (545 after the merge), ISLA test directory 273 passed. A CPU forward on 2 real cases was finite.
- LHR: cpu-short 361286-361288, 361835 and 381148; smoke 361876; lanes 362002, 362003, 362006 and 362007, with their chained links; scoring 377591 and 380907; near-wall inference 377592, 377736 and 380908; reductions 377846 and 381004 (377636 failed because B-ISLA's old dumps had been removed, so I re-dumped them). The full job list is in `JOBS.md`.

**Rulings relied on.**
- Peter, 2026-10-23 17:08: canonical ISLA reads the whole boundary.
- Your decisions: the far field is `freestream`, with no per-cell fields and no dataset change.
- Your approval with conditions 1-4.
- Your steer to use the devin/ model for subagents; I launched none.

**New items for project memory.**
- On LHR, `cp -a` fails on Lustre ACLs; use `cp -r --preserve=timestamps`.
- B-ISLA's near-wall dumps now live in `$T/evals_afwhole_nw/`.

**What is left (your and Peter's decisions; I launched nothing):**
1. NEARWALL's contingency is triggered by the near-wall harm, so the held viscous softening asinh(dU/nu) is the registered next test.
2. ISLA-DTT needs a tie-free octant rule before its far-field accuracy can be read apart from the tie.
3. W-ISLA's pressure harm has an untested candidate cause: the far field holds 99.86 % of the measure.
4. You decide whether the W runs replace the AirfRANS standings. The chapter changes are in `NOTEBOOK_DRAFT.md`.
5. Side finding, no action: on 168 case files, the stored airfoil `normals` disagree with the winding (`out/airfoil_normals_cases.tsv`).

Nothing is still running, and the registry row is removed.

**What stays on disk.**
- Worktree `/home/psharpe/gh/physicsnemo-isla-whole-boundary` (1.1 GB, clean; its head 4b772ec59 is merged into psharpe/isla in 0bdb011f8, so it can be removed; left in place, and no further commit to the branch without telling the supervisor).
- Scratch folder: 6.2 MB.
- LHR: `$T/runs/afwhole_af_*`, about 130 GB each (524 GB in total, a checkpoint every 2 epochs as in the standings). I recommend pruning them to the final checkpoint once you have read them.
- LHR: `$T/evals_afwhole_nw` (7.9 GB of dumps), `$T/recipe_afwhole` (read-only), `$T/kernel/afwhole{,b}` and `$T/afwhole`.
