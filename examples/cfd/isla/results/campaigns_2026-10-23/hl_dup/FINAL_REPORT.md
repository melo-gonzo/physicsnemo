Status: PARTIAL. Stage 1 is done; stage 2 (approved 06:44 EDT) has its six lanes running, and scoring and reading are
handed off (HANDOFF_HLDUP.md).

**What ran** (LHR; every job and md5 is in JOBS.md):

- cpu-short:
  - census 364757, 364882, 364883, 365608;
  - masks 367046, 367047;
  - hole jobs 365609, 365673, 366727, 366767, 367338, 367416, 367560, 367577, 367605, 367622, 367657.
- Short QoS: scorer 366726 (365654 failed), smokes 367139 and 367146.
- GPU so far: about 2.8 GPU-h (1.27 for stage 1, about 1.5 for the smokes). CPU: about 45 core-h.
- Branch psharpe/isla-hl-dedup, commit add3da918, pushed; its worktree is removed.

**Census** (RESULTS.md section 1). 266 of 268 cases (all 180 geometries) store 18.6-19.4 M extra copies of
25,400-26,000 root cells. These are 6.4-7.5 % of the cells and a median 46 % of the wing area (range 3.6-52.7 %),
with up to 492 k copies of one cell. The copies are exact (indices repeated) and their fields are equal up to float32.
LHC001_AoA_10 and _12 instead hold 6.6 M reversed pairs (134 in²).

**Exposure** (all 180 scored cases). The copies are 6.8 % of the tokens and carry 42.1 % of the drawn measure (median
47 %). ISLA's length unit changes by a factor of 0.77-1.19 without them.

**(a)/(b)/(c), baseline / ISLA** (seed means; score_table.md):

| | pressure a / b / c | wall shear a / b / c |
|---|---|---|
| GeoTransolver | 0.913 / 0.917 / 0.679 | 0.924 / 0.922 / 0.697 |
| Transolver | 0.923 / 0.924 / 0.695 | 0.912 / 0.909 / 0.689 |

The defect does not explain ISLA's deficit: (b) moves every ratio by 0.004 or less. Under (c), the input shift, ISLA
loses 41 % on pressure against (b) and the baselines 5-6 %.

**Hole** (revised after the second read). Every on-plane edge together covers only about 70 % of LHC001's outline on
the five broken roots. The stored vertices cover all of it, and about 2,950 on-plane vertices per root belong to no
cell, so the root triangles are missing, not the geometry. A closing of each case's own edges and vertices gives
611,842 in², 0.05 % short of curator's hole, but only at 1.5x the vertex spacing. A robust rebuild is future work.

**Stage 2.**

- 690 masks (MANIFEST.tsv); the mask check passed on 4 of 4 cases.
- All registered smoke checks passed.
- Lanes: ISLA 367678 / 367679, GT 367681 / 367682, T 367686 / 367687. ISLA seed 43's first link hit the Errno 521 cache
  race; its successor 367683 is queued.
- Projection about 270 GPU-h of the 320 cap.

**Rulings relied on.** Datasets stay unmodified, with masks as a sidecar. The supervisor reads the boundary-only ruling
as allowing a data-correction cache (Peter may veto). The LHC001-outline repair was declined.

**On disk.**

- LHR: `$T/hldup/masks` (23 GB, needed by the lanes), `masks_main` (3.1 GB, kept), `$T/hldup/*` (small),
  `$T/recipe_hldup{,x}_{isla,base}`, `$T/kernel/hldupx_*`, `$T/evals_hldup` (4 MB) and `$T/bcsidecar/hilift_hole` (13 MB).
- Local scratch folder: 12 MB.
