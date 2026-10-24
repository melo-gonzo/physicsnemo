# DTT-POSE-EXP: does training with DTT-POSE's forest cost ISLA-DTT native accuracy? (pre-registration)

Lane DTT-POSE-EXP (research-lane, launched 2026-10-24 at 06:51 EDT by the supervisor 01a13c2c). Written 06:55-07:07
EDT 10-24, **before any job of this lane: nothing has been submitted, no smoke, probe, test, checkpoint or score of an
arm exists.** The snapshot below is built (file operations on the LHR login node only). `T` is the LHR task root
`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`; `K` is this
lane's launcher `$T/kernel/dttposex`.

## 1. Question and why

ISLA's contract (book chapter 04) makes the prediction a function of the boundary meshes and their conditions alone.
DTT-POSE (`scratch/isla-dtt-pose-2026-10-24/MECHANISMS.md`, `DESIGN.md` md5 8420670e, approved 06:07 EDT) found that
ISLA-DTT's Barnes-Hut forest broke it: under bf16 autocast the forest was built from bf16-rounded coordinates, and on
the lug rounding decided the octant labels and exact box-test ties, so the posed lug read a different forest (standing
posed / native 1.528 / 1.977). Its fix (D1 build and walk outside autocast; D2 centred float64 canonical coordinates;
D3 cube-face classes with a tie band; D4 a banded box test; D5 model.py carries each token's normal error bound) makes
the forward pose-invariant to roundoff for any weights and its bf16 forest bitwise the float32 one, but it changes
ISLA-DTT's function. **Q: does ISLA-DTT trained with the new forest lose native accuracy against ISLA-DTT trained
with the old one, on the lug and on AirfRANS?** Pose (P) and precision (Q) are checks, not questions.

## 2. What was seen before registering (disclosure)

- The controls' scores in full: `scratch/isla-ref090-2026-10-20/lug_dtt_read.md` (r090fix_lug_dtt von Mises native
  0.002219 / 0.001834, posed 0.003391 / 0.003627; ISLA-DTT / r090fix_lug_isla 1.394), `eag_af_dtt.md` and
  `eag_af_isla.md` (r090eag_af_dtt and r090eag_af_isla per seed, native and posed), DK-GAP's tables
  (`scratch/isla-dtt-knobs-2026-10-23/gap/gap_table.md`), CODE_FACTS.md, DTT-KNOBS's PREREG and stage-1 numbers
  quoted there.
- DTT-POSE's DESIGN.md, MECHANISMS.md, JOBS.md and `results/final_tables.md`: the random-weight verification on
  0.12.0 (job 366947) and the DrivAerML step timing (366948). Of its trained-checkpoint read on the 0.9.0 port (job
  366899, `code_dttpose_090p`) I saw only the empty table header (its reducer failed on a missing file); I have not
  opened its raw outputs and will not before the read.
- TIE-EXP's PREREG and RESULTS (the precedent), including its lug P result: on 0.9.0 (code_cmpaf_fix) interacting
  ISLA's lug posed / native is 1.246 / 1.619, carried by the medial ties of 0.9.0's geometry, which this snapshot
  keeps (section 6, P).
- The controls' link history from sacct (section 9).
- Nothing of any arm exists.

## 3. Snapshot (built; one change and nothing else)

`$T/code_dttposex` = `cp -a $T/code_cmpaf_fix` (the standings' 0.9.0 code: 92d9d7800 + td0152 + the COMPILE-AF
model.py 45b24fb8) patched with DTT-POSE's self-contained 0.9.0 port
`scratch/isla-dtt-pose-2026-10-24/patch090/dttpose_forest_090.patch` (md5 5ab4f654; copied to `snap/` here and to
`$K/snap/`), `patch -p1`, no rejects. Then a SNAPSHOT_HEAD line was appended, MD5SUMS regenerated (3,248 files, md5
6b7add01; SNAPSHOT_HEAD 375e6538), and the tree made read-only (`chmod -R a-w`).

| file | code_cmpaf_fix | code_dttposex |
|---|---|---|
| physicsnemo/experimental/models/isla/tree_read.py | c99dbc37 | **00d83cb9** |
| physicsnemo/experimental/models/isla/model.py | 45b24fb8 | **fc8d0fa9** |
| everything else (sampling.py 435c4cdf, geometry.py, tests, td0152) | | identical |

Overlay proof (`snap/overlay_proof.txt`, md5 e7e8b975): `diff -rq code_cmpaf_fix code_dttposex` lists only MD5SUMS,
SNAPSHOT_HEAD, tree_read.py and model.py; `diff -rq code_dttpose_090p code_dttposex` (DTT-POSE's probe copy of the
same port) lists only MD5SUMS and SNAPSHOT_HEAD; `md5sum -c` passes. (code_cmpaf_fix's own MD5SUMS lists model.py as
65727d07, the pre-COMPILE-AF file; its file on disk is 45b24fb8, checked. So the MD5SUMS diff shows three lines.)

**The port against the branch** (static checks, this machine): the port's model.py changed lines are identical to
`git diff 0673bff1d 40b411df5 -- model.py` (D5). Its tree_read.py differs from the branch's (99aac192 at 0673bff1d,
40b411df5 and 4242962e0) in exactly two lines of `canonical_frame`: 0.9.0's covariance and projection stay float64
GEMMs (`(w x).t() @ x`, `x @ axes`) where the branch uses column sums. DTT-POSE kept them because `canonical_frame` is
shared with 0.9.0's token draw; they are the same function to float64 roundoff. The version commit 4242962e0 (0.12.0)
is not ported, so 0.9.0 checkpoints load in both directions: **an arm checkpoint loads silently into code_cmpaf_fix
and a control checkpoint into code_dttposex** (section 5's in-process proof exists for this).

**Does the compiled blocks' graph change? Statically no.** `compile_blocks` compiles only each slice block's
`block.forward(h, log_w, r, n_hat, g_hat, EPS, vecs, tens, geo_kernel=...)` (model.py `_block_callables`). The forest
is built in `_forward_tokens` before the block loop, and its pairs are read by `kernel_reads[i]` (`_TreeKernelRead`),
which runs eagerly between blocks. D5's `normal_tol` stays on the boundary TensorDict: the blocks' `tokens` are built
from named fields and never carry it. The read's output enters the next block through
`h = cat([h_b, h_p + delta])`, and `_TreeKernelRead` (unchanged byte for byte by the port) returns
`.to(h_q.dtype)`, so `h` keeps its dtype and shape. The blocks' code, inputs' dtypes and shapes, and the token draw
are unchanged, so Dynamo traces the same graphs. This is confirmed in the smoke (section 5, item G); only if G fails
are the blocks certified per LANE_RULES "Compiled regions" on a smoke checkpoint before the lug lanes go.

## 4. Snapshot checks before any training (gating; registered before their results)

Instrument `probe_x.py` (written after approval from DTT-POSE's `probe/probe_impl.py`, adapted to 0.9.0 and
recipe_r090; md5 in JOBS.md before it runs), one code version per process, each process printing its
`tree_read.__file__`, its md5 and `hasattr(tree_read, "normal_faces")`. Cases: the lug's 24 test cases at stride 25
(as 366947), native and from the scorer's posed twin `ginot_lug_volume_posed`; AirfRANS 24 full_test cases at stride
8, native and self-posed (FixedRandomPose keyed by case, translate 1, native draw held: the posed AirfRANS file
resamples on read, DTT-POSE JOBS.md). A small ISLA-DTT with DTT-POSE's protocol: the dataset's model config at hidden
64, 2 layers, 8 slices, `query_independent=true`, every parameter `randn * 0.1`, seed 0. Theta 1, leaf 4.

- **(a) The forest is DTT-POSE's.** Process 1 (code_dttposex + recipe_r090) captures every argument of
  `tree_read_pairs` (tokens, queries, wall distance, gauge unit, theta, eps, `normal_tol`) and its forest and pairs
  on each case and protocol. Process 2 (`$T/code_dttpose_c3` = 4242962e0, tree_read 99aac192, the branch's file) calls
  its own `BoundaryForest.build` / `point_pairs` on the captured arguments. **Pass:** on every case and protocol, the
  pair lists (point and entry indices, node cell sets), node boxes, node measures, separation diameters and per-cell
  classes are bitwise equal (`torch.equal`). Reported beside: max |F_090 - F_ref| of the two frames. If any case
  differs, the replay is repeated with the snapshot's frame injected into process 2; equality then attributes the
  difference to the two `canonical_frame` lines alone. Either way a difference is a failed check, reported to the
  supervisor.
- **(b) Pose, random weights.** On the 24 lug cases, float32, per case `RMS(vm_posed - vm_native) / RMS(vm_native)`
  of the predicted von Mises. **Pass:** median <= 2.2e-7 and max <= 1.74e-5. Source: DTT-POSE DESIGN section 3 (12
  cases: 1.6e-7 [1.1e-7, 2.2e-7]) and its 24-case verification on the same stride (366947: median 1.53e-7, max
  1.74e-5, the generic residual of one case). Beside, not gating: the same on code_cmpaf_fix (the defect; 0.11.0 read
  2.1e-6) and on code_dttposex at theta 0 (the read with no clusters: the floor of the rest of 0.9.0's model, whose
  geometry lacks the medial-tie rule). **Stated risk:** 0.9.0's geometry may put its own pose residual above the
  bound (TIE-EXP: interacting ISLA's lug posed / native 1.25 / 1.62 on 0.9.0). If (b) fails and the theta-0 floor
  is within a factor 2 of the theta-1 residual, the excess is not the forest's: I report it and recommend proceeding;
  the supervisor decides. (Superseded by amendment A1.2, section 8: the fallback is decided.)
- **(c) Precision.** On all 48 cases, both protocols: the forest of a bf16-autocast forward (the whole model under
  `torch.autocast("cuda", bfloat16)`, as training) against the float32 forward's. **Pass:** bitwise equal (lists,
  boxes, measures, diameters, classes) on every case.
- **(d) The whole ISLA test directory** (`test/experimental/models/isla/`, the snapshot's own 0.9.0 tests) on LHR
  cpu-short (`scratch/isla-surface-bc-2026-10-23/lhr/tests_cpu.sbatch`'s command), on code_dttposex and on
  code_cmpaf_fix (the real base), one job running both in turn. **Pass:** every failure or error on the snapshot is
  either on the base with the same assertion message, or one of the two failures by design, declared now:
  (E1) `test_isla_tree_read.py` errors at collection with `ImportError` for `normal_octants` (removed by D3; ISLA is
  unreleased, no shims); (E2) `test_isla_release.py`'s ISLA-DTT golden (`isla_dtt_v1`) mismatches, since the function
  changes, while every interacting golden passes. Anything else fails the check.
- **(d2) The 0.9.0 tree-read contract on the new forest** (my addition: E1 would otherwise leave the forest
  untested). A copy of the snapshot's `test_isla_tree_read.py` with only `normal_octants` dropped from its import
  and its one octant test (`test_boundary_forest_splits_by_normal_octant_and_its_nodes_are_tight`) deselected (diff
  recorded), run in a temporary copy of the snapshot's test tree against code_dttposex and code_cmpaf_fix, in the
  same job. Every failure the base lacks is listed with its message and attributed to a registered design change
  (D2's float64 frame and centred `origin`, D3's faces, D4's separation diameter) or not. One I cannot attribute
  fails the check.

A failed check stops the lane before training; I report it to the supervisor with the numbers.

## 5. Arms, launch, smokes, scoring

| arm | rows of `$K/lanes.tsv` | control | its row | trained |
|---|---|---|---|---|
| lug | 0-1: dttposex_lug_dtt_seed42 / 43 | r090fix_lug_dtt_seed42 / 43 | `$T/kernel/r090fix/lanes.tsv` rows 0-1 | compile_blocks=true (fix-trained path) |
| AirfRANS | 2-3: dttposex_af_dtt_seed42 / 43 | r090eag_af_dtt_seed42 / 43 (also DTT-KNOBS stage 2's control) | `$T/kernel/r090eagaf/lanes.tsv` rows 2-3 | eager |

Each row is its control's row byte for byte apart from idx, group (`dttposex`), run id and code (`code_dttposex`); the
`diff` is recorded in JOBS.md. So: lug `ginot_lug_volume`, `isla_lug`, `train_split=train val_split=test
+training.val_interval=10 model.hidden=512 model.n_slices=512 model.compile_blocks=true ++model.query_independent=true`,
20,000 cells; AirfRANS `airfrans_uvolume_2d`, `isla_volume_airfrans`, the same extra on `full_train / full_test` plus
`compile=false model.compile_blocks=false`, 100,000 cells; both lr 3.75e-4, 500 epochs, bf16 (LANE_PRECISION), seeds 42
and 43, `$T/recipe_r090`.

**Launcher `$K`**: lanes.sbatch and ckpt_preflight.py byte copies (8eecec8f / e34bd611, shared by r090fix and
r090eagaf); site.env = kernel/dttknobs_s2's (a0b8d0ca: r090eagaf's settings, 03:59:00 normal-QoS links,
`--segment=1`, an exclude-node file) with its own `exclude_nodes.txt` of the five bad nodes; submit.sh = r090eagaf's
with the prefix `isla.dttposex.`. Jobs `isla.dttposex.<slug>` (lanes), `.smoke`, `.check`, `.tests`, `.score`,
`.probe`. Chained singleton links; a link is never cancelled or resubmitted except a duplicate after NODE_FAIL (the
oldest kept). Event guard (as SLICES-AF): validation loss above 4x its running minimum after epoch 10, a non-finite
loss, or a BLOCKED marker stops that lane and is reported; it is not a verdict. Up to 4 lanes at once, submitted two
at a time, the lug first; at most 2 of my jobs on short QoS; user-wide short QoS kept at 12 or fewer.

**Smokes** (after approval, before any lane; short QoS; one job `isla.dttposex.smoke`): DTT-KNOBS's smoke.sbatch
(b943c4b6) and smoke_wrap.py (f6233bd4) copied with this lane's paths and job name, through `$K`'s lane launcher with
NO_CHAIN, 50 training steps per unit. Units: **L** (row 0), **Lc** (the control's row 0 on code_cmpaf_fix, from a
smoke-only table; same node), **A** (row 2), **Ac** (the control's row 2 on code_cmpaf_fix). The wrapper adds one
print at the first training forward, `SMOKE FOREST`: tree_read path and md5, `normal_faces` present, the forest's class
count and non-empty trees, `origin` present, frame dtype; and on the first 3 forwards it rebuilds the forest from the
same arguments with autocast disabled and prints whether it is bitwise the autocast one (Q in training). A unit
passes if: rc 0, no traceback, no non-finite loss, no OOM; its resolved_config.yaml equals the control's
(`runs/r090fix_lug_dtt_seed42`, `runs/r090eag_af_dtt_seed42`) except run_id and paths; for L and A, SMOKE FOREST
shows md5 00d83cb9, `normal_faces` and `origin` present, 6 classes (lug) / 4 (AirfRANS; 2-D faces and octants both
number 4, so there the md5 and `origin` decide), and the rebuild is bitwise equal on all 3 forwards. Each unit records
median step and peak memory.
- **G (graph).** L and Lc run with `TORCH_LOGS=recompiles,graph_code`. Pass: the rank-0 Dynamo graphs of the compiled
  blocks, in order, are textually identical after deleting source-location comments (paths and line numbers differ:
  the patch shifts model.py's lines); the recompile counts are equal. If G fails, a cert unit trains row 0 for 2
  epochs and the blocks are certified per LANE_RULES "Compiled regions" (block-wise against float64 eager on that
  checkpoint, per tensor group, in each precision within eager's own one-ulp change) with the program's current
  certification instrument, named in JOBS.md before it runs; the lug lanes wait for it.
- When every smoke unit and every check of section 4 passes, the lanes are submitted without a further approval,
  and one short message gives the job ids. A failed unit or check is reported at once.

**Scoring, exactly as each control's** (r090fix's score.sbatch 3a5e4ca9 and r090eagaf's bd40af29 run the same command
and differ only in job name and default roots): a copy with job name `isla.dttposex.score`, roots `$K` /
`$T/evals_dttposex`, the five nodes excluded, and one change: `infer.py` is started through `score_wrap.py`, which
wraps nothing numeric. It prints `SCOREPROOF` lines from inside the scoring process: tree_read path and md5,
`normal_faces` and `origin` present, and per forest built the class count and non-empty trees (counts over all
builds), then runs `infer.py` unchanged by `runpy`. Rows: float32, the final checkpoint, `infer_split=val` (lug
600 test cases at 200,000 cells, AirfRANS 200 full_test cases at 100,000), eager (`compile=false
model.compile_blocks=false` appended to the lug rows, as r090fix's rows), protocols **native and posed**. Area is not
scored: on both datasets it equals native by construction (degenerate; the lug standing says so, and DTT-KNOBS did not
score it on AirfRANS).

**Checks before reading any row** (a failed check makes the row invalid and its rule "incomplete as registered"):
`"epoch": 500` in its config row; 600 / 200 `infer_step` rows with the control's sample ids; `SNAPSHOT_USED` =
code_dttposex and DATASET_USED as the control's; the score log's extra as the lane's; and the SCOREPROOF lines show
md5 00d83cb9, `normal_faces` and `origin` present, and 6 classes on the lug.

## 6. Rules (registered; the supervisor's requirements)

CM is the scorer's case-mean relative L2 (`infer_summary`, else the mean of the `infer_step` rows), native. Fields:
lug von Mises (`von_mises_l2`, the standing's only field, the headline); AirfRANS pressure, in-plane velocity
(`velocity_l2`; never velocity_z) and nut. Slots: 2 runs per arm, 4 per comparison; posed is a reading of the same
runs (no independent number).

**N, native accuracy, per field:**
- `D = ln( mean_seeds CM(arm) / mean_seeds CM(control) )`;
- `SE = sqrt((s_arm^2 + s_c^2) / 2)`, s = a pair's two-seed sample sd (n - 1) of ln CM. The controls' own:
  lug **s_c = 0.1347** (0.00221933 / 0.00183431, DK-GAP's unrounded values of the standing); AirfRANS s_c = **0.0022 /
  0.0395 / 0.0078** (pressure 0.012428 / 0.0123887; velocity 0.00737341 / 0.00697252; nut 0.0207112 / 0.0204842).
- **cost** if D > ln 1.10 and |D| > 2 SE; **gain** if D < ln 0.90 and |D| > 2 SE; otherwise **no change**.
- **Sub-bar flag** (SLICES-DRV's): per field, up if arm / control > 1.03 with both arm seeds above both control seeds;
  down if < 0.97 with both below. It changes no label.
- **Resolution, stated now.** On the lug 2 SE binds, not the ln 1.10 bar: with s_arm = s_c, 2 SE = 0.269, so a cost
  label needs a ratio above 1.31 (1.21 if the arm's seeds agree exactly). The control's seeds are 21 % apart. On
  AirfRANS the ln 1.10 bar binds on every field. The seed-matched ratios (same seed, init and data order; TIE-EXP's
  lug pair landed within 1 % of each other's seed) are reported beside; they change no label. (Amendment A1.3,
  section 8, adds a registered paired reading for the lug.)

**P, pose (a check):** for each lug seed, posed / native of the scorer's von Mises CM lies in [1/1.02, 1.02]. A failure
is a defect and is reported at once. Derivation: (i) the posed protocol (`ginot_lug_volume_posed`: uniform SO(3)
rotation and a translation by one RMS radius, no CenterMesh, float32) changes every input by its float32 rounding;
the trained-model floor of that alone on this set is at most 0.25 % (interacting ISLA on the fixed geometry,
TIE-EXP: 1.0024 / 1.0015; ISLA-DTT on 0.9.0 at theta 0, with no clusters, DK-SENS 363160: 1.0000 / 1.0001, which
also bounds what 0.9.0's medial ties do to the ISLA-DTT form). (ii) The forest's own residual (DTT-POSE, 24 lug
cases, 0.12.0): 0.0001 % of pairs unmatched at the median, 0.07 % mean, 2 % in the worst case. Scaled as the old
forest's posed excess scales with its unmatched share (52 % of pairs, +53 / +98 %, so about 1-2 % of error per 1 % of
pairs), that is about 0.1-0.15 % on the case mean. (i) + (ii) is about 0.4 %; 2 % leaves a margin of 5. The
control's 1.53 / 1.98 is far outside it. Beside: AirfRANS posed / native (standing 1.0017-1.0022).

**Q, precision (a check):** on every scored case of every arm checkpoint (lug 600 native + 600 posed, AirfRANS 200 +
200 posed, the scorer's dataset files, each batch loaded once and read in both precisions; probe_x.py with the
checkpoint loaded, whole-model forward up to the forest), the bf16-autocast forest equals the float32 forest bitwise (lists, boxes, measures, diameters, classes). A failure is a defect and is
reported at once.

**Adoption (the supervisor's decision rule, registered):** DTT-POSE's forest goes into canonical ISLA if no field of
either dataset reads **cost** under N, and P and Q pass. If any field reads cost, the supervisor takes it to Peter with
the numbers and neither of us adopts it then. Gains and flags are reported, not decisions. If an arm or a row is lost
or invalid, its rule is "incomplete as registered" and escalated; no verdict is improvised.

**Beside, not verdicts:**
- arm / interacting ISLA: lug against r090fix_lug_isla (mean 0.0014537; the control's 1.394, seed-matched 1.517 /
  1.270); AirfRANS against r090eag_af_isla (0.0131631 / 0.011794, 0.0064407 / 0.00627008, 0.0157382 / 0.0149675; the
  control's 0.994 / 1.129 / 1.342);
- DK-GAP's AirfRANS regions: per run, native, pooled over the 200 cases, e_R in the wake (in-plane velocity) and
  beyond one chord (nut), with DK-GAP's own instrument (`gap/probe_gap.py` 2a213669, `reduce_gap.py`'s `af_regions`
  6be03f4b) on code_dttposex; gate: its per-case metrics equal the scorer's to 1e-5; reported as ratios to the control
  and to interacting ISLA (the control's gap 1.95 / 1.99 wake velocity, 1.48 / 2.47 nut beyond one chord);
- pairs per point (float32, native, every scored case, from the Q probe; the control's with code_cmpaf_fix on its own
  checkpoints) and step time (smoke L / Lc and A / Ac on one node; the lanes' epoch times against the controls');
- the posed D, SE and label; case medians; the share of cases where the arm beats the control.

## 7. Predictions (probabilities mine, written before any result)

| | prediction | p |
|---|---|---|
| S1 | (a) bitwise on all 48 cases x 2 protocols (the two frame lines could flip a rounding) | 0.85 |
| S2 | (b) passes (0.9.0's geometry could add its own pose residual) | 0.65 |
| S3 | (c) passes | 0.95 |
| S4 | (d) passes with exactly E1 and E2 as new failures | 0.70 |
| S5 | G passes (graph unchanged) | 0.90 |
| S6 | every smoke unit passes | 0.85 |
| N1 | lug von Mises: no change / gain / cost | 0.85 / 0.08 / 0.07 |
| N2 | AirfRANS: no change on all three fields | 0.85 |
| N3 | any field of either dataset reads cost | 0.10 |
| F1 | a sub-bar flag up on at least one field / down on at least one field | 0.30 / 0.25 |
| P1 | P passes for both lug seeds | 0.85 |
| Q1 | Q passes on every case | 0.95 |
| A1 | the adoption condition is met | 0.75 |
| B1 | lug arm seed-mean native error below the control's | 0.55 |
| B2 | lug pairs per point 3-8 % below the control's (DTT-POSE: 117 -> 112) | 0.70 |
| B3 | smoke step time of L within 5 % of Lc, of A within 5 % of Ac | 0.70 |

Why: on the octant-trained checkpoints the bf16 training forest against the float32 one moved native accuracy by
0.4 % on the lug and 0.1-0.7 % on AirfRANS (DK-TREEPREC), and the new forest holds about the same pairs, so a
retrained model should adapt to its own forest. Faces in place of octants regroup which cells share a node, which a
model trained on it absorbs. The lug might gain a little, because its training forest is no longer decided by
rounding at every step, but its 21 % seed spread hides anything under about 30 %.

## 8. Amendments

Allowed only before results as LANE_RULES define them, marked "Amendment before submission" or "before results",
with what was known when written.

### A1. Amendment before submission (supervisor), received 2026-10-24 07:10 EDT (by `date`)

Approval of md5 80bae24a09c8d06028a421631c1cfc59 as registered, **cap 650 GPU-h**, with these amendments. **What was
known:** no job of this lane exists (no smoke, check, probe, test or arm); the snapshot code_dttposex was built; nothing
else beyond section 2.

1. **Times.** The lane was launched at 06:51 EDT 10-24, and the harness deadline is about 00:51 EDT 10-28 (corrected
   in place above). HANDOFF_DTTPOSEX.md by 18:00 EDT 10-27 stands.
2. **Check (b)'s fallback, decided now.** If (b)'s bound fails but the theta-1 residual is within a factor 2 of the
   theta-0 floor on both the median and the max over the 24 lug cases, the excess comes from 0.9.0's geometry, not the
   forest: proceed, and report it. If the theta-1 residual exceeds twice the floor on either the median or the max,
   stop and tell the supervisor. Also reported beside (b), not gating: the share of forest pairs unmatched posed
   against native on the 24 lug cases (probe_pose.py's metric), next to DTT-POSE's 0.12.0 numbers (366947: median
   0.000118 %, max 1.97 %, 8 of 24 identical).
3. **Lug paired reading (registered).** The change adds no parameters, so a seed gives the same initialization and
   data order in both codes. For von Mises, with r_s = CM(arm, seed s) / CM(control, seed s), s = 42, 43:
   **cost (paired)** if r_42 > 1.10 and r_43 > 1.10; **gain (paired)** if r_42 < 1/1.10 and r_43 < 1/1.10; otherwise
   no paired label. A cost (paired) counts as a cost under the adoption rule (it goes to Peter); a gain (paired) is
   reported. AirfRANS keeps N alone, since its ln 1.10 bar binds already.
4. **Reference of check (a).** JOBS.md records the DTT-POSE head it compares against, 4242962e0 (`$T/code_dttpose_c3`),
   tree_read.py md5 99aac192. The supervisor freezes DTT-POSE's forest and tells me if it must change. The lanes keep
   code_dttposex either way; before any merge the supervisor checks that the merged forest equals this snapshot's
   bitwise on (a)'s captured arguments (so (a)'s captures are kept).
5. Registry row `^isla\.dttposex\.` with expected_end 2026-10-27; up to 4 lanes at once, as registered. When every
   check and smoke passes as registered, the lanes are submitted without waiting, and one short message gives the job
   ids.

The adoption rule therefore reads: adopt if no field of either dataset reads cost under N, the lug reads no cost
(paired), and P and Q pass.

## 9. Cost, cap, schedule

Measured from the controls' sacct (all links, including two lost restarts and two links cancelled mid-run):
r090fix_lug_dtt 171.3 + 153.0 = **324 GPU-h** for the pair (25 links, 81 h of 4-GPU node time); r090eag_af_dtt
110.8 + 110.0 = **221 GPU-h** (19 links). (The brief's 480 GPU-h for the lug pair is not what sacct shows for these
two lanes.) DTT-POSE measured the new forest's build and walk slightly faster than the old (lug 115 -> 95 ms per
forward on 0.11.0) and D5 adds one float64 pass over the kept cells, so the arms should cost about what the controls
did; the smokes re-price them (arm GPU-h = control GPU-h x smoke step ratio L / Lc and A / Ac).

| item | GPU-h |
|---|---|
| lug arm pair | about 325 (290-360) |
| AirfRANS arm pair | about 220 (200-245) |
| smokes (4 units, one short job) | about 4 |
| snapshot checks (a)-(c), one short job | about 2 |
| scoring (8 rows: 2 array tasks) | about 8 |
| Q probe, pairs, regions | about 6 |
| contingency: cert unit if G fails | about 4 |
| **total** | **about 570 (530-630)** |

**Cap: I ask for 650 GPU-h** (inside the provisional 760). If failures or reruns project past it, I ask first.
Schedule, if approved by about 08:00 EDT 10-24: checks and smokes 08:00-10:00 (two short jobs, one cpu-short job);
lanes from about 10:00. AirfRANS lanes end about 16:00 EDT 10-25 (the control: 28 h of links); the lug lanes, the
longest (43 h of links, about 2.3 days with queueing), about 12:00-24:00 EDT 10-26. Scoring, Q and regions within
about 3 h after each; the read on 10-27 morning. HANDOFF_DTTPOSEX.md by 18:00 EDT 10-27 if anything is unread; the
harness deadline is about 00:51 EDT 10-28 (corrected, A1.1). Registry row `^isla\.dttposex\.` (expected_end 2026-10-27) at my first job,
removed after my last.
