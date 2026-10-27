# Rule 1 on AeroZoo: root cause, blast radius, proposed fix (AF-CONFIRM Part B, 2026-10-27)

Supervisor's brief of 11:13 EDT 10-27 (items 1-4). No registered check or verdict here; Part B is paused. Jobs (LHR, all
`isla.affixfcaz.*`, short QoS): smoke 425238, diag 425612, dump 425735/425850, probe 425802/426073, root 426320, blast
426526, DrivAerML 426760 (failed: split name) and 427731. Scripts: `lhr/diag_az/` (here) = `$T/kernel/affixfcaz/diag/`.
Part B GPU-h so far 2.90 (sacct).

## 1. Root cause

The forest is built in the whole boundary's canonical frame (`canonical_frame` over all cells). Rule 1, as committed in
c0fa1fc28, computed the frame-rotation bound per condition:
`eta = frame_rotation_bound(x_g, w_g, u)`, a Davis-Kahan bound `E / (g - E)` from **that condition's own covariance**
(E = 4u tr C, g = its smallest eigenvalue gap). That covariance is not the one the frame comes from.

A square inlet or outlet has two nearly equal in-plane variances. The model's systematic token draw (a uniform random
start) keeps a different subset of the square's cells each step, so the drawn gap moves near zero: g/E ranged from 9 to
324 over 13 draws of the dumped case. eta then ranges from 0.124 down to 3e-3. When g falls below E, the bound declares a
tie and returns 0.

eta enters twice:
- the face tolerance `tol + eta`, which spread the planar outlet over 5 faces;
- `kappa_u = 3u + eta` in `separation_diameter_sq`, where
  D' = (D (1 + theta kappa) + 2 (1 + theta) kappa R) / (1 - kappa - ...).

With kappa near 0.12 almost no node is admitted, and every point opens leaves.

The dumped OOM batch (inlet and outlet 804,806 cells each, no_slip 12,302, slip 0), training draw:
- the outlet's g/E was 9.07 and its eta 0.124, against eta_whole 8.8e-4;
- rule 1 built **1,405M** pairs, 1.20B of them near pairs in the outlet's 5 trees, with up to 15,194 pairs per point;
- the same inputs without groups (DTT-POSE) built 54.0M, and the octants (code_r090) 47.0M.

Other draws of the same case built 59-105M pairs. With every group given the whole boundary's eta (V2), every draw built
49.4-52.5M. Details: `diag_az/root-426320.log` and `diag_az/probe-426073.log`.

## 2. Blast radius (pair counts only; V0 = rule 1 as is, V1 = no groups = DTT-POSE's faces, V2 = rule 1 with the whole boundary's eta)

`diag_az/BLAST_TABLE.md`, from `diag_az/blast/*.jsonl` (`blast_summary.py`). AeroZoo used 40 cases (full.train 24,
full.val 8, full.test 8) with 5 draws each. The lug and Tensile2d used 20 cases x 5 draws each. AirfRANS (far field)
used 24 x 2. DrivAerML Volume with its whole boundary (BC-SUPPLY's pipeline: vehicle, no_slip, slip, inlet, outlet) used
6 x 4.

| dataset | case-draws | V0 > 3x V1 | V0/V1 median / p90 / max | V2/V1 median / p90 / max | min group g/E | V2 = V0 bitwise |
|---|---|---|---|---|---|---|
| AeroZoo (40 cases) | 200 | 1 (3.45x), plus the smoke's batch (30x) | 0.96-1.00 / 1.09-1.22 / 3.45 | 0.94-1.00 / 1.03-1.09 / 1.26 | 35 | 0 / 200 |
| DrivAerML Volume, whole boundary | 24 | 0 | 0.95 / 0.97 / 0.97 | 0.95 / 0.97 / 0.97 | 2.6e5 | 0 / 24 |
| lug | 100 | 0 | 1.003 / 1.026 / 1.03 | 1.003 / 1.026 / 1.027 | 8.6e4 | 0 / 100 |
| Tensile2d | 100 | 0 | 1.038 / 1.048 / 1.07 | 1.038 / 1.048 / 1.065 | 7.3e4 | 5 / 100 |
| AirfRANS (far field) | 48 | 0 | 0.787 / 0.823 / 0.83 | 0.789 / 0.826 / 0.834 | 7.0e4 | **0 / 48** |

Only AeroZoo's square inlets and outlets come near a tie. On AeroZoo, 9 of 770 group covariances had g/E < 100, and 33
had g/E < 1000.

**The singularity is not rule 1's alone.** The bound `E / (g - E)` diverges as g falls toward E, and DTT-POSE's ungrouped
forest has the same singularity on the whole boundary. On AeroZoo full.train case 0 (slip 812,918 cells, a near-cubic
box), eta_whole reached 4.8e-2 on one draw:
- V1 (DTT-POSE, no groups) built **652M** pairs, with up to 5,188 per point;
- V2 built 493M;
- V0 built 83M.

Other draws of the same case gave V1 99-144M. So V2 removes rule 1's own error but leaves DTT-POSE's near-tie blow-up
wherever the whole boundary nearly ties.

## 3. Proposed fix (V2): commit 3e6cc1636 on psharpe/isla-affix-fc-r090-eta (parent e066a8101; pushed, never merged)

`eta` is computed once, from the whole boundary about its centroid (the frame's own inputs), and used for every group.
- **Principled:** it bounds the rotation of the frame the forest actually uses.
- **Pose-invariant:** it is a function of the boundary alone, as before.
- **Bitwise on single-condition boundaries:** with one group, or groups=None, it is the value computed before. The
  existing bitwise test passes.
- **New test:** a square condition with g about 8E. It fails on c0fa1fc28's tree_read (diameter inflation 2.1x) and
  passes with the fix. Locally, test_isla_affix_forest.py passes 7 of 7 and test_isla_affix_quadrature.py passes.
- **It changes AirfRANS's forest:** V2's pair lists differ from V0's in 48 of 48 case-draws, though the counts agree to
  0.3%. F's and FC's AirfRANS lanes therefore test the unfixed rule 1. FC's lanes 423644/5 have not started; they are
  pending on Priority since 09:13.
- **What it does not fix:** DTT-POSE's whole-boundary near-tie (case 0 above). On AeroZoo with V2, 1 of 200 case-draws
  read 1.26x V1, and V1 itself reached 12x its ordinary count on case 0.

## 4. The near-tie singularity, and C without rule 1

- A principled cure of the near-tie is DTT-POSE's design question, not rule 1's. I have none ready. Two directions,
  both untested:
  - (a) Choose the frame inside a nearly degenerate eigenspace by a further invariant, as `canonical_frame` already does
    for sign ties. This needs a principled tie tolerance.
  - (b) Test admission with frame-free quantities, such as a bounding sphere about the node centroid in place of the
    frame's AABB, so the frame's rotation no longer enters D' and eta is needed only for the face tie band.
- **C without rule 1** (nodes that mix conditions): each node's log m becomes log sum_k M_{m,k} exp(s_k), with M_{m,k}
  the node's measure of condition k and s_k = -log A_k + beta_k. M_{m,k} is computed once without gradient by the
  forest's bottom-up sums (`node_means` of the one-hot conditions times the node measure), and the logsumexp over K <= 5
  conditions carries beta's gradient. About 25 lines in tree_read.py plus one test. On pure nodes it equals today's C:
  the other terms are exp(-inf).
