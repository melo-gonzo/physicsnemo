Status: DONE
Revised after REVIEW.md, 2026-10-24 02:35 EDT (first version: `REPORT_v1.md`; changes: `REVISION_NOTE.md`).

**DK-TREEPREC (2026-10-24 02:12 EDT).** The tree mismatch explains none of ISLA-DTT's float32 standing gap. Reading the training (bf16) forest instead of the float32 forest makes the native error slightly and consistently worse on velocity, nut and the lug (+0.2 % to +0.7 %), and leaves it level on pressure (-0.2 % / +0.0 % for seeds 42 / 43). ISLA-DTT ÷ ISLA moves from 0.994 / 1.129 / 1.342 to 0.994 / 1.131 / 1.351 on AirfRANS and from 1.394 to 1.399 on the lug, so the training forest widens the gap rather than closing it. The pair lists still differ for most points: 82 % on AirfRANS, 99 % on the lug.

**What ran.** LHR short-QoS jobs 364407 (smoke, 2.6 min) and 364468 (production, 14.8 min), with probe `probe_treeprec.py` on code_cmpaf_fix + recipe_r090. **1.16 of 2 GPU-h.** Nothing is running, the slot is released, and nothing was committed (scratch only). Details: `JOBS.md`.

**Gate passed.** R reproduces the standing's per-run native scores: AirfRANS to ≤ 1.8e-6; the lug to 0.00221933 / 0.00183431 (the standing is given to 4 digits). Every case matches the scorer's record to ≤ 6.4e-6. Posed R also reproduces `lug_dtt_read.md` (0.003391 / 0.003627). The repeat floor is d ≈ 3e-7, and the repeat's pair lists are identical to R's.

**What the code does.** The four runs trained under bf16 autocast: `precision: bfloat16` in every `resolved_config.yaml` and `augment: False`. The in-training validation (`metrics.jsonl`) also ran under bf16: `val_epoch` calls `_run_epoch` and then `forward_pass` inside `get_autocast_context(precision)` (train.py 266, 380, 405), so it read the bf16 forest too.
- Under autocast, the bf16 output of `points @ frame` reaches these parts of the tree:
  - in bf16: the trees' coordinates and boxes (`aabb`), and the trees' `node_total_area`, which becomes `forest.measure`, the node log-measures and the denominator of `node_means`;
  - in float32: `diameter_sq`, because autocast upcasts `pow`/`sum`, and so the walk's box table (bf16-rounded query coordinates cast to float32);
  - in float32 regardless: centroids and normals, which are aggregated from the float32 points.
- **The literal emulation the brief proposed is not the training tree.** Replacing only the two matmuls by bf16 matmuls, without autocast, leaves `diameter_sq` and the walk in bf16. Its lists differ from training's for 0.3-4 % of points (AirfRANS) and 18-24 % (lug).
- **The T arm used here:** `BoundaryForest.build` and `point_pairs` called under `torch.autocast('cuda', bfloat16)`, everything else float32. The forest depends only on the geometry, so the 3 check cases of each (dataset, protocol) give 12 distinct forests, each checked twice (once per seed). On every one, its (qi, si, n_node) lists, boxes, diameters and measures equal those of a whole-model bf16 forward of the unmodified code, bit for bit, from bit-identical inputs.

**Results** (`treeprec_table.md`, `treeprec.json`; paired sign test `signtest.md`). T ÷ R of the two-seed mean, [seed 42, seed 43]:

| | AirfRANS pressure | AirfRANS velocity | AirfRANS nut | lug von Mises |
|---|---|---|---|---|
| native | 0.999 [0.998, 1.000] | 1.002 [1.002, 1.003] | 1.007 [1.007, 1.007] | 1.004 [1.003, 1.004] |
| native: cases where T is worse [s42, s43] (z) | [99, 112] of 200 (-0.1, 1.7) | [132, 132] of 200 (4.5, 4.5) | [134, 132] of 200 (4.8, 4.5) | [365, 343] of 600 (5.3, 3.5) |
| two-seed spread (R) | 0.3 % | 5.6 % | 1.1 % | 19 % |
| posed | 1.033 [1.041, 1.024] | 1.025 [1.037, 1.013] | 1.012 [1.017, 1.008] | **1.90** [1.95, 1.85] |

- **Native.** Most lists change:
  - AirfRANS: 82 % of points per case (10 % of cases at 100 %), at 21.6 pairs per point under R and 20.9 under T;
  - lug: 99 % of points (78 % of cases at 100 %), at 118 pairs per point under both arms.
- **Per-case prediction change.** d(T, R) is 0.11-0.18 of the case's own error on AirfRANS and 0.23-0.25 on the lug (median). Mean d is 0.0017 / 0.0008 / 0.0026 on AirfRANS (pressure / velocity / nut) and 0.0005 on the lug. The change is a small systematic bias, not noise: the repeat floor is d ≈ 3e-7, and the paired sign test above finds T worse in most cases for velocity, nut and the lug, while pressure is level. The per-case T/R medians are, for seeds 42 / 43: pressure 0.9997 / 1.0030, velocity 1.0022 / 1.0020, nut 1.0080 / 1.0057, lug 1.0033 / 1.0025. Restoring float32 node measures (the TL arm) leaves T/R within 0.004 of T.
- **Posed.** The bf16 forest is not pose-invariant. Two mechanisms move its rounding with the pose, and which one dominates is untested:
  - the frame keeps no centre (model.py 1824-1830), so the bf16 output of `points @ frame` (tree_read 247, 346) is rounded on a grid that depends on where the points sit;
  - autocast casts the uncentred lab-frame inputs (points, gauge) to bf16 before the matmul, and that rounding changes under a rotation alone.
  The posed sets also translate each case by one RMS point radius with no CenterMesh (`*_posed.yaml`), which coarsens the bf16 grid further. On the lug posed, T doubles the error (0.00351 to 0.00667), and DTT ÷ ISLA posed goes from 1.770 to about 3.4. These checkpoints never trained posed (`augment: False`), so posed T is the forest the bf16 code would build in that pose, not one they learned.

**Meaning.**
- **Sure:** the float32 standing's gap is not a tree-mismatch artefact. Reading the training forest moves native DTT ÷ ISLA by -0.0008 / +0.0026 / +0.0092 (pressure / velocity / nut) and +0.0050 (lug), against 0.994 / 1.129 / 1.342 / 1.394. It widens every gap, including pressure's, where ISLA-DTT is the one ahead. So the float32 standing does not inflate the gap through the tree mismatch; the training forest's gap is larger by at most 0.009. The two-seed spread is the wrong yardstick for a paired change on one checkpoint (for the record, nut's 0.68 % is about 0.6 of its 1.10 % spread).
- **Not tested here:** whether bf16 training, as opposed to reading the bf16 tree, causes DK-GAP's lug gap.
- **Side finding:** on the lug, the float32 pair lists change with the pose too (151 pairs per point posed, against 118 native), at 1.53 / 1.98 times the native error. The cause is not isolated: the forests and the query and boundary subsamples were not compared across poses, so frame ties and pose-dependent inputs are not separated. This matches the standing and is DK-SENS's territory.

**For project memory.**
- A bf16-trained ISLA-DTT reads a bf16 forest in training and validation. A float32 score of it is slightly and consistently better, by 0.2-0.7 % native on velocity, nut and the lug, and level on pressure, so it understates the DTT-ISLA gap slightly and never inflates it.
- The faithful way to reproduce training's forest in a float32 probe is `torch.autocast` around build and walk. Swapping only the matmuls is not faithful, because autocast also upcasts `pow`/`sum` there.

**Rulings relied on:** float32 scoring on the standing's sets in the scorer's order (COMMON_RULES), the reproduction gate, GPU work on LHR only, no code change to the reference path (the patch lives in this probe's process only). No fix is proposed. **Left:** nothing; nothing is running.

**On disk:** this folder (4.3 MB, `raw/` included; the revision added `REPORT_v1.md`, `REVISION_NOTE.md`, `signtest.md` and `signtest_treeprec.py`) and `$K` (4.2 MB).
