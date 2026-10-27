# FOREST-NEARTIE: the face forest's near-tie blow-up, its risk, and a fix (final report, 2026-10-27 15:17 EDT)

Status: DONE. The fix is staged on psharpe/isla-014-neartie and waits for Peter's review, because it changes what
canonical ISLA computes. It is pushed and not merged.

What ran:
- All jobs ran on LHR (`isla.neartie.*`); job ids and results are in JOBS.md. Scripts are in lhr/ (copies under
  `$T/kernel/neartie`); data copied back are in data/.
- GPU-h: 8.32 of 15.
- Snapshots: code_8097 (8097ed00f), code_6f1e (6f1e527f1), code_981c (the fix) and the four test snapshots. Each one's
  MD5SUMS matches the local git archive.

## 1. Risk (RISK_TABLE.md, risk_rows.tsv)

The face forest (8097ed00f) and the octant forest (6f1e527f1) ran on the same draws. The kept-cell sha256 matched on
all 5,376 draws compared.

| sample | face / octant pairs median | p99 | max | > 2x | > 3x | > 5x |
|---|---|---|---|---|---|---|
| AeroZoo census, 320 mean_stable cases (train 200, val 60, test 60, stratified by family) x 5 draws | 0.954 | 1.30 | 2.23 | 1 | 0 | 0 |
| AeroZoo near-tie Monte Carlo, 60 cases x 60 draws | 1.034 | 2.05 | 8.40 | 40 (1.1%) | 15 (0.42%) | 3 (0.08%) |
| DrivAerML Volume, whole boundary, 8 x 4 | 0.837 | 0.86 | 0.86 | 0 | 0 | 0 |
| AirfRANS with far field / lug / Tensile2d, 24 cases each (kept whole: every draw is the same) | 1.08 / 0.97 / 1.07 | | 1.16 / 1.00 / 1.29 | 0 | 0 | 0 |

**Memory.**
- Measured on 640 real chunked training steps (bf16, eager, one GB300): peak reserved = 147.2 GiB + 152 B per pair.
- The worst measured step had 179M pairs and peaked at 172.3 GiB (the octant forest on the same draw: 158.7 GiB).
- The worst draw found, AF-CONFIRM's case 0 at draw 22, has 742M pairs, so its step is predicted at about 252 GiB.
  The GB300's capacity is 276.6 GiB.
- AF-CONFIRM's 1.4B-pair OOM ran on r090 code, which has no chunked step. 0.14.0's recipe splits the step into chunks
  by a predicted peak, but in eager mode that prediction is too low by 3.6 to 4.0 GiB:
  - a forced capacity of 150 GiB gave a prediction of 149.1 GiB and a measured 152.7 GiB;
  - so a step OOMs at 942 to 968M pairs (one chunk), and in about 2% of each band that needs several chunks
    (chunk_sim.py).

**OOM chance of a 250-epoch AeroZoo ISLA-DTT lane: about 13%** (risk_model.py; uncertain by about a factor of 3).
- That is 0.145 expected OOM draws per lane, at 4 ranks and 2,800 draws per epoch.
- Each lane would also get about 2.4 two-pass steps above 1e9 pairs and 0.4 steps that read every pair (about 4e10).
- The model: on the face forest, pairs grow 2.5x at eta 0.02, 8.7x at 0.05, 33x at 0.1 and 160x at 0.2, and reach
  every pair (about 540x) from eta 0.45 (a forced-eta sweep). The token gap follows a Rice law fitted per case.
- With the fix, the same model gives 0.

**A resume replays the draw.**
- The recipe's checkpoint holds no RNG state, and `set_seed(seed + rank)` runs at every start. (The brief assumed the
  RNG was checkpointed; it is not, but the replay follows from the reseeding.)
- Two resumes from copies of one checkpoint drew bitwise the same starts and kept cells. An uninterrupted run drew
  different ones (job 429718).
- So a crash inside a resumed segment recurs at the same step on every later resume: a crash loop. Every chained link
  also restarts the same sequence of starts.

**Which cases are near ties.**
- Only the external_opposed_hemisphere and external_opposed_sphere families. Their spherical far fields have two or
  three nearly equal variances.
  - Full-resolution gap/E is 30 to 1,000 there, against at least 4,550 for boxes and at least 1,840 for ducts.
  - Median eta: 2.0e-3 (hemisphere), 4.3e-4 (sphere), 4.1e-6 (box).
- AF-CONFIRM's "near-cubic box" case 0 is case-...0001, an opposed-hemisphere case.

## 2. Root of the draw dependence, and the options (OPTIONS.md, POSE_TABLE.md)

The forest's frame, origin and eta come from the drawn tokens, not from the full-resolution boundary:
- `BoundaryForest._build` runs on `boundary_r` / `log_frac`, which are the kept cells with their Horvitz-Thompson
  weights.
- The full-resolution frame is computed only inside `sample_boundary_cells`, for the Morton order of the draw.
- Near a tie, the token gap g is set by the draw's noise (a Rice law about the full-resolution gap), so eta = E / (g -
  E) varies from draw to draw.
- 0.14.0's separation diameter then treated the frame's rotation as an error of every coordinate in proportion to its
  distance R from the centroid (kappa = 3u + eta). Every small node far from the centroid was opened.

The options:
- (a) **Frame and eta from the full-resolution boundary** makes the near ties per-case; it does not remove them.
  - The smallest full-resolution g/E is 30, so the blow-up stays at up to 4.4x the octant forest's pairs, now on every
    draw of such a case.
  - It re-groups 83-90% of AeroZoo's pairs.
- (b) **Bounding spheres** are immune to eta but cost +29% pairs on AeroZoo, +37% on DrivAerML and +28% on the lug.
- (c') **The fix.** The rotation moves the point and the cells together, so it widens each node only in proportion to
  its own size: by f = (1 + sqrt(d) eta (1 + theta)) / (1 - sqrt(d) eta), at the angle theta / f.
  - It is capped by a frame-free certificate: in every frame, the node's cells lie in the ball of half its diagonal
    about its box's centre.
  - The guarantee is the same as before: an admitted node is separated in the exact canonical frame.
  - On the near-tie Monte Carlo, the pairs are at most 1.27x the octant forest's. On all 1,632 saved draws they are
    never more than 0.14.0's. At any eta they are at most 3.65x their eta-0 pairs.
- Pose (the P check of DTT-POSE-EXP): no candidate changes the outputs at random weights. Posed pairs unmatched (median
  / max), 0.14.0 against the fix:
  - lug: 1.0e-6 / 1.75e-2 against 1.2e-6 / 1.75e-2;
  - Tensile2d: 8.7% against 8.7%;
  - AirfRANS: 7.0% against 8.2%;
  - AeroZoo near-tie cases: 17% / 43% against 18% / 64% (octant 20% / 61%). No forest is pose-invariant at a near tie:
    the frame turns inside the degenerate plane, and the trees turn with it.

## 3. The fix: psharpe/isla-014-neartie, 981ca5aaf (on 8097ed00f; worktree /home/psharpe/gh/physicsnemo-isla-014-neartie)

The diff:
- `separation_diameter_sq` gets the new bound (tree_read.py +44 / -22).
- Three tests (+149 lines):
  - a near-spherical boundary whose frame nearly ties. It fails on 8097ed00f (2.99x the eta-0 pairs) and passes now
    (1.39x);
  - the certificate under rotations within the bound, and under any rotation past it (4 parametrizations);
  - the diameter's monotone, bounded growth.
- Goldens: `record.py --regenerate goldens_v1.pt`.
  - 12 of 90 cases change: the G-FOREST forest_* cases, in diameter_sq only.
  - The CPU A/B against 8097ed00f is bitwise on the other 86 variants (ab/compare.txt).
- The committed code reproduces the evaluated variant's pairs on 1,632 of 1,632 saved draws, and its P rows (job
  430771). On AF-CONFIRM's case 0, the worst pairs drop from 742M to 115M (octant: 94M); the draw-22 step peaks at
  162.6 GiB.

Tiers, all with 0 failed, each +6 new tests on 72b624ae4's counts:

| tier | torch 2.13 | torch 2.12 |
|---|---|---|
| GB300, whole directory | 1,324 passed, 6 skipped (mix-order, by design), 1 xfailed | 1,330 passed, 1 xfailed |
| CPU full | 781 passed, 533 skipped, 1 xfailed | 781 passed, 533 skipped, 1 xfailed |

**What changes for each dataset**, the fix against 0.14.0 (pairs; then outputs at random weights, native pose):
- AeroZoo: the median draw is unchanged (1.000); near-tie draws lose up to 55% of their pairs.
- AirfRANS (far field): pairs fall 17% (0.815-0.849), because the far field puts the airfoil's tiny nodes at
  R / D >> 1. Outputs change by at most 4.2e-7, and 5.1% of the pairs are re-grouped.
- Lug: pairs 1.000, outputs 1e-8, 4.5e-5 of pairs re-grouped.
- Tensile2d: pairs 1.000 (0.989-1.0), outputs 3e-7, 2.9e-5 of pairs re-grouped (at most 0.86%).
- DrivAerML whole boundary: pairs 0.999.

**Accuracy evidence still missing:**
- A training A/B on AeroZoo's opposed-sphere and hemisphere cases.
- A training A/B on AirfRANS (17% fewer pairs near the airfoil, a coarser read there), two seeds each.
- Posed scoring of trained checkpoints. The P check here uses random weights only.
- Lug, Tensile2d and DrivAerML change too little to need one.

## Rulings relied on, and items for memory

Relied on:
- Peter's 09:40 ruling: architecture changes go to Peter.
- The principled-changes-only yardstick: the fix is a tighter error bound, with no constants.
- The supervisor's test-tier rules.

New, for the supervisor:
1. The recipe's `_CHUNK_GIB` predicts the peak too low by about 4 GiB per 200k points in eager mode. Raising the
   margin, or fitting per execution mode, would remove the OOM window independently of the forest.
2. A crash loop on resume is real because the RNG is not checkpointed. Checkpointing each rank's RNG, or reseeding
   per epoch, would end it.
3. AeroZoo's near ties are the spherical far-field families. AirfRANS with its far field pairs posed against native
   differ by 7% on 0.14.0 already.

## Left, and what stays on disk

- Nothing is running. The registry row is removed.
- Worktree: /home/psharpe/gh/physicsnemo-isla-014-neartie (branch pushed). Drop or keep it per Peter's review.
- This folder: 19 MB (jsonl, logs, scripts; the A/B tensors and the base source copy were deleted).
- LHR: `$T/kernel/neartie`, with code snapshots and logs (about 0.8 GB). The 15 GB of saved forest inputs were
  deleted after this report.

**For the book.** When the frame's variances nearly tie (AeroZoo's spherical far fields), 0.14.0's face forest
treated the frame's rotation as an error that grows with the distance from the centroid. On 1 draw in 1,000 of those
cases it read more than 5 times the octant forest's pairs, and a 250-epoch AeroZoo lane had about a 13% chance of an
OOM. Bounding the rotation as a motion of the point and the cells together, capped by a frame-free ball, keeps the
pairs within 1.27x of the octant forest's and changes nothing measurable elsewhere except AirfRANS, which reads 17%
fewer pairs. Pending Peter.
