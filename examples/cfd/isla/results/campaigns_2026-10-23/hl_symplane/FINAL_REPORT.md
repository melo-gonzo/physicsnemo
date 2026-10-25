Status: DONE

HiLift's symmetry plane is built at load from the closed root cap in the dataset's own source STL, and all 690 canonical cases pass the per-case check. Branch `psharpe/isla-hl-symplane`, head **bc4081e75**, is pushed and not merged. It merges cleanly with origin/psharpe/isla 142485ef0. Details are in `RESULTS.md`, the jobs in `JOBS.md`, and the book changes in `BOOK_CHANGES.md` (md5 4c2a1604). Paths below are relative to this folder.

**What ran**
- Commits, all by one lane-writer child on devin/claude-opus-5.5:high, hooks on:
  - d12834fcd: the cap, the tolerance, `check_root`, `points_fingerprint` and the record builder `tools/build_symmetry_check.py`;
  - ae5b89931: `AddSymmetryPlane` in the reader, a generalized `whole_surface_transforms` hook, and the configs;
  - bc4081e75: the census in a code comment, and a test fixed so it passes on deployments.
- LHR cpu-short jobs: 373775, 373970, 374043-374045, 374259, 374310-374312, 374330, 374340, 374370, 374371, 374461. They used 9.0 core-h (cap 40); the census was 6.2 of it.
- GB300 short job 374346: 0.27 GPU-h (cap 3).
- The registry row was added before the first job and is now removed. Nothing is running.

**Census.** The tool ran on every -medium folder with a wing. That is the canonical manifest's 690 cases, which the cluster dedup configs also read.
- **690 of 690 pass.**
- Worst on-plane vertex: 0.061264 in, the same vertex on every case.
- Worst open-edge midpoint: 0.0601 in on 689 cases. On geo_LHC024_AoA_8 it is 0.1266 in (0.94 tau). Two 20-in root edges there have their ends within 0.004 in of the outline, so the distance is those edges' own chord sagitta.
- The record is `$T/hldup/symplane/hl_symmetry_check_medium.json` (md5 bc2ed653). It and its parts are read-only, with md5s in `MANIFEST`.

**Tolerance.** tau = max over the cap's chords of h + eps = 0.13412 + 0.00035 = **0.1345 in**.
- eps = sqrt(2) × the float32 spacing at 2583 in. It bounds the change in a point-to-segment distance from storage rounding.
- h is each chord's tangent-triangle bound, L / (cot b1 + cot b2). The b are the end turnings plus their float32 uncertainty.
- The maximum falls on a 1-in chord near the flap whose ends turn 11.8° and 20.5° in the same sense.
- Measured against tau: the CFD vertices reach 0.46 tau. HL-DUP's 0.011 in compares two CFD outlines, a different quantity, and sits at 0.08 tau.

**Against curator's plane** (LHC001_AoA_10 and _12; -medium and full copies agree):
- The equator is bitwise equal.
- Area is +4.6e-9 relative; the hole area ratio is 0.99999.
- Hole Hausdorff distance: 0.0613 in from the stored hole to ours, 0.00085 in from ours to the stored.
- Every normal is +y. The stored holes have 11 degree-4 vertices; ours is one simple loop.
- The seam is watertight with the inlet and outlet.

**Moves with the body.** Eight samples, including the posed file at rotations of 121° and 165°, gave BAD 0, with residuals at most 1.5e-5 of the extent.

**Baselines.** GeoTransolver's and Transolver's forward_kwargs and targets are **bitwise unchanged, 24 of 24**. The plane adds 12-13 ms per load.

**Tests and smoke**
- Recipe tests: 605 passed, 12 skipped. The 2 test_campaign_c_hooks failures have the same messages as on the base.
- The 3 real-data tests that check the record against the exact check pass, including two broken roots.
- GB300 smoke with the plane: 682.3 ms per step, 32.51 GB, 98,300 tokens (the plane adds 1,032). The far-field arm, which is BC-SUPPLY's whole-boundary smoke, gives 682.0 ms, 32.23 GB and 97,268 tokens. Step time measures data loading in both.

**BOOK_CHANGES.md.** I ran the cells against 142485ef0 and the merged tree.
- No cell fails.
- Three generated tables gain `symmetry`, in chapters 05, 06 and 20.
- Prose edits with replacement text, in 07, 10, 20 (six places), 21 and 06:
  - remove the two "In flight" callouts on the symmetry plane, in 10 and 20;
  - cut "every piece but the symmetry plane";
  - state the check, the tolerance and the curator comparison.

**Rulings I relied on.** Peter's approval of about 19:10 EDT. Your 19:28 decision to keep the verdicts in one fingerprinted dataset-level record under $T/hldup/symplane, which holds no geometry. The whole-boundary and boundary-only rulings. The condition type comes from the dataset's paper: "The symmetry plane is treated with a no-stress boundary condition" (arXiv:2605.19565v2, section 2.3 and appendix A.2.1). The README documents none.

**For project memory:** the record rule as you stated it at 19:28. The exact check costs 13-23 s and a 6.4 GB scan per wing, so it cannot run at every load.

**For you to decide**
1. A case that failed the check would get no plane, and `isla_surface_highlift` would then raise a KeyError on it. No canonical case fails, but the full 1,800-case root has no record yet. Building one would take about 15 core-h. I recommend building it only when a run needs that root.
2. The open-edge midpoint check includes a long edge's own sagitta, so it is at 0.94 tau on LHC024. A future case with a longer root edge could fail although its root is correct. I recommend leaving the check as it is unless one does.
3. Merge, and the retrain of the HiLift standings with the plane.

**On disk.** The worktree `/home/psharpe/gh/physicsnemo-isla-hl-symplane` is clean. This folder is 2.3 MB. On LHR: `$T/hldup/symplane` (0.7 MB), `$T/hlsymplane` (186 MB, mostly the baseline-identity .pt files), `$T/kernel/hlsymplane` (empty), the read-only snapshots `$T/code_hlsymplane_b4b1` and `$T/recipe_hlsymplane_{base_b4b1,m1_d128,head_ae5b,head_bc40}`. Nothing is over 1 GiB.
