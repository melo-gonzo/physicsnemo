# Two code facts about ISLA-DTT's forest, handed to lane DTT-POSE (DTT-KNOBS, 2026-10-24 02:57 EDT)

Found by DTT-KNOBS stage 1 (inference only; `STAGE1.md`). Both are now DTT-POSE's scope (supervisor, 02:54 EDT). DTT-KNOBS
proposes no fix and its stage-2 arms train on `code_cmpaf_fix` with both facts present, as the control did. Every number
below is from a reviewed report (`<part>/REPORT.md`, revised after `<part>/REVIEW.md`); paths are relative to
`scratch/isla-dtt-knobs-2026-10-23/` unless they start with `$T`
(`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`, LHR).

## 1. The forest is built from bf16-rounded coordinates under bf16 training

**Code** (the same in `$T/code_cmpaf_fix` and at psharpe/isla-geo-bundle f13eb5ce0):
- `tree_read.py` `BoundaryForest.build`: `points_f, normals_f = points @ frame, normals @ frame` (line 247 in
  code_cmpaf_fix, 267 at f13eb5ce0). `BoundaryForest.point_pairs`: `points_f = points.detach() @ self.frame` (346 / 366).
  Plain matmuls; `tree_read_pairs` wraps the build only in `torch.no_grad()`. The file's own float32 helper
  `frame_coordinates` (column products) is not used on these lines.
- `model.py` line 1907 (f13eb5ce0) calls `self._tree_read_pairs(...)` after the autocast-disabled block (1871-1876) has
  closed, and `forward` calls `_forward_tokens` outside the disabled blocks at 1647 / 1664. So under
  `torch.autocast(bfloat16)` both matmuls return bf16.
- What is bf16 then (DK-TREEPREC, checked against `cluster_tree.py`): the trees' coordinates and boxes (`aabb`,
  cluster_tree 867) and `node_total_area` (`.to(dtype)`, 879), which becomes `forest.measure`, the node log-measures
  (tree_read 557, 565) and the denominator of `node_means` (407). Float32: `diameter_sq` (`pow(2).sum`, autocast
  upcasts) and the walk's box table (`torch.cat` promotes, 664; queries cast to `box.dtype`, 1743). Float32 regardless:
  centroids and normals, aggregated from the float32 points (tree_read 262).
- Training and in-training validation both run under it: `recipe_r090/src/train.py` 266 (autocast in `forward_pass`),
  380 (`precision`), 405; `val_epoch` delegates to `_run_epoch` (657, 673). Every ISLA-DTT run's `resolved_config.yaml`
  has `precision: bfloat16`. Every float32 standing score builds a float32 forest.

**Numbers.**
- Pair lists differ, float32 against bf16 (DK-TREEPREC, jobs 364407 smoke, 364468 production; 1.16 GPU-h): 82 % of
  AirfRANS points per case and 99 % of lug points get a different list (AirfRANS 21.6 / 20.9 pairs per point, lug 118 /
  118). Pairs per point over datasets (DK-COST, jobs 363435, 364009, random weights): AirfRANS +1 to +10 %, AeroZoo -34
  to +9 %, DrivAerML -2 to +9 %.
- Accuracy, reading the training (bf16) forest T against the float32 forest R on the same checkpoints, native
  (`treeprec/treeprec_table.md`, paired sign test `treeprec/signtest.md`): AirfRANS pressure 0.999, velocity 1.002, nut
  1.007; lug 1.004. Small but consistent (T worse in 132-134 of 200 AirfRANS cases on velocity and nut, z 4.5-4.8;
  365 / 343 of 600 lug cases); pressure level.
- Pose: the bf16 forest is not pose-invariant. Lug posed T / R = 1.90 [1.95, 1.85]; AirfRANS posed 1.033 / 1.025 /
  1.012. Two candidate mechanisms, untested which dominates: the uncentred `points @ frame`, and autocast's bf16 cast of
  the lab-frame inputs (`points / gauge`, model.py 1824-1830) before the matmul; the posed sets also translate by one
  RMS radius with no CenterMesh.
- Surface form (HiLift-size, queries = the cells; DK-COST): bf16 adds a near-field cell term that theta does not
  shrink, 1,309-1,537 pairs per point flat over theta 3-0.5. bf16 / float32 = 709 / 569 (1.25x) at 50k cells and
  2,170 / 916 (2.37x) at 400k, p99 23,041 pairs per point (`cost/cost_table.md`, HiLift section).

**How to reproduce the training forest in a float32 probe:** wrap `BoundaryForest.build` and `point_pairs` in
`torch.autocast('cuda', dtype=torch.bfloat16)`. Swapping only the two matmuls for bf16 matmuls is not faithful (it
leaves `diameter_sq` and the walk in bf16): its lists differ from training's for 0.3-4 % (AirfRANS) and 18-24 % (lug)
of points. DK-TREEPREC checked its T arm bit for bit (`torch.equal` on lists, boxes, diameters, measures) against a
whole-model bf16 forward of the unmodified code, on 12 distinct forests (3 cases x 4 dataset-protocols), each twice.
Probe: `treeprec/probe_treeprec.py` (md5 9e2d6211; LHR copy `$T/kernel/dttknobs_treeprec/probe_treeprec.py`), outputs
`$T/kernel/dttknobs_treeprec/out` and `treeprec/raw/`.

## 2. The lug's posed score differs from native at every theta > 0

**Numbers** (DK-SENS, job 363160, float32, r090fix_lug_dtt_seed42/43, 600 cases): posed / native = 1.53 / 1.98 at
theta 1 [s42, s43], 1.02-1.98 across theta and leaf settings, and 1.0000 / 1.0001 at theta 0 (the exact read, which
builds no clusters). The posed float32 lists hold 1.17-1.41x the pairs (1.29x at theta 1: 151 against 118 per point).
AirfRANS posed equals native to 0.00-0.25 % at every setting, with identical pair counts. (The standing
`lug_dtt_read.md` already shows posed / native 1.528 / 1.977.)

**Measured** (DK-SENS, job 364211, the lug bf16 run, native pose; the diagnostic `normals @ canonical_frame` ran under
that job's autocast, so this is a bf16 measurement and the float32 share was not measured): 84 % of the lug's cells
(median over cases, range 76-91 %) have a normal component below 1e-5 in absolute value in the canonical frame
(`sens/lhr/probe_sens.py` line 261, `near_zero_normal_share`; LHR `$T/kernel/dttknobs_sens/out_v3`). For those cells
`normal_octants`' `c > 0` test rests on rounding residues, natively as well as posed.

**Inferred, not tested:** that the posed flip of those signs builds the different forest. Not done: posed octant labels
compared with native ones; a tie-breaking rule tried; the frame's own near-ties excluded (the canonical frame's skews are
about 1.2e-6 against `CANONICAL_SKEW_TIE = 1e-9`). The coherence test is not involved at theta 1 (vacuous at theta >=
0.919 in 3D, see below). Probe: `sens/lhr/probe_sens.py` (md5 323f5396; LHR `$T/kernel/dttknobs_sens/probe_sens.py`),
tables `sens/sens_table.md` (posed rows) and `sens/sens.json`.

## Related fact (not a defect)

The normal-coherence admission test (`1 - |mean n| < theta^2 / 2`, tree_read.py `coherent_length`) is vacuous for theta
>= 0.919 in 3D and >= 0.765 in 2D, because inside one sign octant |mean n| >= 1/sqrt(d). Measured smallest node |mean n|:
0.872 (AirfRANS), 0.705 (lug), so it binds only below theta 0.505 and 0.768 there; no node is refused at theta >= 1 on
any dataset (DK-COST, DK-SENS). A fix to the octant split would change which cells share a node and therefore the read
at every theta > 0; at theta 0 nothing changes.

## Not touched

DTT-KNOBS opened no file of TIE-EXP (`scratch/isla-tie-2026-10-22/`) and made no code change; the octant question may
overlap TIE-EXP's medial-tie work, which DTT-POSE should check with the supervisor.
