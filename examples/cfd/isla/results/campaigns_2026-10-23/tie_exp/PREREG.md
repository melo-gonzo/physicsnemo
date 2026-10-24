# TIE-EXP: pre-registration (MEDIAL-TIE DESIGN section f; written 2026-10-22 15:11-15:40 EDT, before any TIE result)

Lane TIE-EXP, launched 2026-10-22 ~14:50 EDT by the supervisor ("ISLA Autoresearch", 01a13c2c). Design:
`scratch/isla-medial-tie-2026-10-22/DESIGN.md` sections a-c, f, g, j. Budget and eager path approved by the supervisor at
15:00 EDT (conditions 1-6 below).

## 0. What is known at registration ("before results")

- No TIE checkpoint, smoke or pre-step output exists. Nothing from pre-steps (i) or (ii) has been seen.
- **B, the lug (r090eag_lug_isla 42/43): nothing eager-trained on the lug has been scored.** At 15:00 EDT both lanes are
  training (epochs 337-342 of 500, jobs 332257 / 332267 and their pending successors). REF-090-CLOSE (lane 39090df0) finishes
  and scores them about 10-23 00:00-03:00 EDT; this lane only reads their score records.
- **B, Tensile2d (r090eag_t2dz 42/43): scored and seen** (REF-090 PREREG §13, `scratch/isla-ref090-2026-10-20/t2d_control.md`):
  native displacement 0.00783 / 0.00629, stress 0.00964 / 0.00794, plastic strain 0.41040 / 0.25005; posed / native
  (displacement) 1.0001 / 0.9997.
- Other experiments seen: r090fix_lug_isla (fix-trained, compiled path) native 0.00146 / 0.00144, posed 0.00198 (ATTRIB-090 §4);
  its compensated flip residual median 1.5e-4 / 1.6e-4 (job 333695); the geometry flip mismatch at 8,428 interior nodes of 60
  lug test cases (job 334176). The TIE code's GB300 timing and losses on the lug (DESIGN j: "its losses differ slightly").

## 1. Arms

| arm | code | recipe | runs | training |
|---|---|---|---|---|
| B (reused) | `$T/code_cmpaf_fix` (92d9d7800 + td0152 + COMPILE-AF model.py 45b24fb8) | `$T/recipe_r090` | r090eag_lug_isla_seed42/43 (kernel/r090eagaf rows 4-5), r090eag_t2dz_isla_seed42/43 (kernel/r090eag rows 0-1) | eager: `compile=false model.compile_blocks=false`, bfloat16 (LANE_PRECISION), lr 3.75e-4, 500 epochs, augment=false, 4 GPUs DDP |
| TIE | `$T/code_tie` = code_cmpaf_fix + d3d9da49d's geometry.py (2d39efb4) and tree_read.py (c1d34bc1) | `$T/recipe_r090` (unchanged) | tie_lug_isla_seed42/43, tie_t2dz_isla_seed42/43 | B's lane rows byte for byte apart from run id and code |

Overlay proof: `evidence/overlay_proof.txt` (`diff -rq` and fresh md5s of all 3,248 files: only SNAPSHOT_HEAD, MD5SUMS,
geometry.py, tree_read.py differ; both files equal `git show d3d9da49d:<path>`). Between 92d9d7800 (B's code) and the branch
base 817e5ddbf only model.py's compile options and segment_softmax.py's Triton import differ (not on the eager path; B and TIE
both keep B's model.py); geometry.py, tree_read.py and sampling.py are identical there.

**(a) Joint attribution.** TIE changes two things at once: off-wall closest-feature ties take the least-norm Clarke
subgradient, and on-wall nodes take the least-norm point of the hull of the incident cells' normals, non-unit there (15 % of
the lug's training queries). It also changes normal_octants at exact zero components (DESIGN e; inert unless the canonical
frame aligns exactly with face normals). A verdict attributes to these tie rules jointly, never to one of them.

**Stated assumption (supervisor condition 4), not a result.** If the cell count of section 3 (iii) shows that the in-model
token draw keeps every boundary cell on both datasets, the proposed token-draw change (DESIGN i.2) cannot touch them; FOLD-SIGN
(lane 666bae9d) counts the wall-distance sign changes of the winding-number sign (DESIGN i.1) on the lug and Tensile2d meshes,
expected 0. If both hold, TIE stands for the whole geometry bundle (ties, on-wall rule, fold sign, token draw) on these two
datasets. The FOLD-SIGN count is recorded in RESULTS when the supervisor passes it on.

## 2. Scoring (as B was scored)

`kernel/tie/score.sbatch` = kernel/r090eagaf/score.sbatch with roots kernel/tie and evals_tie; rows = B's rows with run id and
code_tie: recipe src/infer.py, final checkpoint, float32, fully eager (`compile=false model.compile_blocks=false`),
infer_split=val, native / area / posed, 200,000 cells; the lug's 600 test cases, Tensile2d's 100 val_100 cases. Before
reading: `"epoch": 500` in every config row and the case count (600 / 100). Metric: the infer_summary case mean (or the mean of
the infer_step rows where infer.py refuses the 2-D summary): lug `von_mises_l2`; Tensile2d `displacement_l2`, `stress_l2`,
`q_l2` (plastic strain), as `scratch/isla-ref090-2026-10-20/t2d_control.py` reads them. B's lug rows are REF-090-CLOSE's
`$T/evals_r090eag/r090eag_lug_isla_seed4x/{native,posed}`, B's Tensile2d rows `$T/evals_r090eag/r090eag_t2dz_isla_seed4x`.
Protocol non-degeneracy: the posed sets rotate (uniform SO(3) / SO(2)) and translate each case with no CenterMesh, so posed
changes every input on both datasets; the area protocol is scored as B was but read by no rule here.

## 3. Pre-steps and the gate (no training; registered before their results)

Scripts in `launcher/probe/` (mirrored at `$T/kernel/tie/probe/`); probe recipe `$T/recipe_tieprobe` = recipe_att090 +
`conf/model/isla_tensile2d_tie_flip.yaml` (isla_tensile2d with FlipBoundaryOrientation on `outer` first), probe-only.

**(i) Geometry-only flip residual** (`geo_flip_probe_tie.py`, ATTRIB-090's geo_flip_probe.py generalized; cpu-short array
`pre_cpu.sbatch` tasks 0-1, and the same probe on the GPU path in `pre_gpu.sbatch`): the first 60 lug test cases, native
against winding-flipped, base (code_cmpaf_fix) and branch (code_tie). The residuals do not assume a unit direction:
`r = |nu_F + nu_N|`, `|d_F + d_N| / L`. (The original's angle test reads an exactly flipped short vector as a mismatch.)
Gate (i) passes when, on both CPU and GPU:
- the base reproduces the defect: at least one interior node with unit-angle mismatch above 1e-3 (ATTRIB: 8,428);
- the branch has no node with `r > 1e-6` outside DESIGN c.2's stated exception (a zero vote sum within the error bound:
  `|d_N| <= 1e-6 L` and `d_F == d_N`; counted and reported), and `max |d_F + d_N| / L <= 1e-6` over all nodes.

**(ii) Prediction-level compensated flip residual** (`flip_probe_tie.py`, ATTRIB-090's flip_probe.py generalized, same
compensation: qt.seed's s and asinh(s) columns, geo_logit / geo_feat columns 2:4 negated; eager float32; `pre_gpu.sbatch`,
short QoS): the same weights under base and branch geometry. Lug: r090fix_lug_isla_seed42, 60 test cases. Tensile2d:
r090eag_t2dz_isla_seed42, 100 val cases. Gate (ii) passes when:
- branch: the median over cases of `d_F` is at most 1e-6 for every target field, on both datasets;
- base, lug: median `d_F` above 1e-5 (reproduces ATTRIB's 1.5e-4: the probe sees the defect);
- the uncompensated control `d_U` has median above 1e-2 on both datasets under both codes (the flip matters, so a small `d_F`
  is the compensation's doing).

**(iii) Supervisor condition 4** (`cellcount.py`, `pre_cpu.sbatch` tasks 2-3): every case of the lug's train (2,400) and test
(600) splits and Tensile2d's train_400 and val_100, as the training config loads them: the total cells of the named
condition meshes against `boundary_tokens` (200,000 in B's resolved configs), and the reader's boundary against
`subsample_n_cells` (20,000). Reported, not gating; it decides only the stated assumption of section 1.

**Smokes (supervisor condition 2; gating).** One epoch of each dataset through the TIE lane launcher (`NO_CHAIN=1`, smoke
run dirs, `training.num_epochs=1`), short QoS. Pass: rc 0, "Training completed", finite losses, no traceback; resolved config
precision bfloat16, compile false, compile_blocks false, lr 3.75e-4, seed, hidden 512, n_slices 512, boundary_tokens 200,000;
the link-start line names code_tie and recipe_r090; and the projected total cost from the smoke's median step time stays
under the 220 GPU-h cap (else I ask before launching).

**Gate.** The four arms are submitted only if (i), (ii) and both smokes pass as registered. If (i) or (ii) fails, I stop and
report the numbers to the supervisor.

## 4. Reading rules (registered verdicts stand as written)

Notation: `X̄` = seed mean (42, 43) of the case-mean error; `l_s = log X_s`.

**F (symmetry).** On the TIE checkpoints (`flip_probe_tie.py`, code_tie; lug 60 test cases, Tensile2d 100 val cases): the
median over cases of `d_F` is at most 1e-6 for every target field, for both seeds, on both datasets: "exact flip symmetry
restored". Otherwise "a residual source remains", to be located before anything else is read as a property of the tie
rules. Beside it (not a rule): the same probe on B's checkpoints r090eag_* with code_cmpaf_fix.

**N (native).** `r = X̄_TIE / X̄_B`, native. Sigma per dataset and field (supervisor condition 3): `sigma = max(sigma_reg,
s_pool)`, where `s_pool^2 = [sum_B (l_s - mean_B l)^2 + sum_TIE (l_s - mean_TIE l)^2] / 2` (2 dof, the pooled two-seed
spread of B's and TIE's seeds; with two seeds per arm the SE of `log r` is then `sqrt(s^2/2 + s^2/2) = s`). Both numbers are
reported beside the verdict.
- `|log r| <= 2 sigma`: "no cost". `log r > 2 sigma`: "costs native accuracy". `log r < -2 sigma`: "gain".
- **(b) sigma_reg and sources.**
  - Lug: **0.036**, ATTRIB-090 RESULTS §4 (`scratch/isla-attrib090-2026-10-21/RESULTS.md`, "Pooled σ(log) 0.036 (4 dof,
    native)"; fix-trained and eager-trained lug arms). 2 sigma_reg = 0.072, r in [0.930, 1.075].
  - Tensile2d: the eager-trained spread of r090eag_t2dz 42-45 (the supervisor's condition 3; REF-090 PREREG §13,
    `scratch/isla-ref090-2026-10-20/t2d_control.md`, "0.9.0" row, σ log of the eager column): **displacement 0.232, stress
    0.330, plastic strain 0.303**. The brief named `t2d_fourseed.md`; its per-seed σ(log) are smaller (fix-trained 0.158 /
    0.144 / 0.149, bug-trained 0.025 / 0.040 / 0.237; pooled two-vs-two 0.113 / 0.106 / 0.198, its "SE of the log ratio"
    × √2), so including them in the max changes nothing; they are recorded here as the brief's source. 2 sigma_reg for
    displacement = 0.464: r in [0.629, 1.590].
- Tensile2d's N verdict is displacement's (its low-noise field, as REF-090 §13 decides by displacement); stress and plastic
  strain carry their own labels beside it.

**P (pose; lug primary).** `e = Pbar / Nbar - 1` per arm, from the arm's own posed and native seed means at epoch 500 (so
**(c) e_B is read from B's own r090eag_lug rows**, not from ATTRIB's fix-trained 0.37).
- `e_TIE <= 0.5 e_B`: "the medial ties carry most of the lug's pose sensitivity". `e_TIE >= 0.8 e_B`: "they do not; look
  elsewhere". Between: "partial".
- Non-degeneracy guard: P is read only if `log(1 + e_B) > 2 sigma_lug` (B has a pose excess above seed noise); otherwise P is
  "not readable: B shows no pose excess" and nothing is claimed from it.
- Tensile2d is the control (not a verdict): predicted `|log((1 + e_TIE) / (1 + e_B))| <= 2 sigma_T2D` (displacement).

**Decision (DESIGN f).** Adopt (with section d's kernel or a per-mesh cache) if F passes and N is not "cost" on either
dataset. P decides whether the 2026-10-18 note's other options are still needed.

**Incomplete as registered.** If a rule cannot complete (B's lug rows never land, an arm is lost, a scorer gives up), it is
reported "incomplete as registered" with what the available slots show, and escalated; no verdict is improvised.

## 5. Predictions (probabilities)

| item | prediction | p |
|---|---|---|
| gate (i) | base reproduces; branch clean on CPU and GPU | 0.85 |
| gate (ii) | branch median d_F <= 1e-6 (the float32 floor of the forest's table order) | 0.70 |
| F | restored on both datasets | 0.65 |
| N, lug | no cost / gain / cost | 0.70 / 0.10 / 0.20 |
| N, Tensile2d (displacement) | no cost | 0.90 |
| P, lug | most / partial / do not | 0.40 / 0.30 / 0.30 |
| P control, Tensile2d | within 2 sigma | 0.95 |

## 6. Cost (supervisor condition 1)

DESIGN f's 73 GPU-h counted one compiled-path lane's figure (REF-090: lug 15.8 h x 4 GPUs = 63 GPU-h per lane) for both
seeds. B trains eager with compile_blocks=false, measured on B itself (sacct, train.log): the lug at 0.215 s/step, about 26
epochs/h, 500 epochs in about 19 h on a 4-GPU node, **76 GPU-h per seed**; Tensile2d 3.25 h per seed, **13 GPU-h**. TIE adds
about 35 ms per lug step (DESIGN j), +16 %: lug about 2 x 88 = 175, Tensile2d about 2 x 13 = 26, **about 200 GPU-h**, plus
about 3 for pre-steps, smokes, scoring and flip probes. Cap 220 GPU-h for everything (supervisor 15:00 EDT); ask before
going past it.

## 7. Run conventions

Launcher `$T/kernel/tie` (copies of kernel/r090eagaf; md5s in LAUNCH_MD5S.txt), job names `isla.tie.lug.s42`,
`isla.tie.lug.s43`, `isla.tie.t2d.s42`, `isla.tie.t2d.s43`, singleton chains of 03:59:00 normal-QoS links (a pending final
link that fits under 2 h moves to short QoS), bad nodes excluded. Registry row `^isla\.tie\.` (expected_end 2026-10-25) at
launch, removed when done. A code-review defect that changes the features means cancel and relaunch from scratch; TIE
checkpoints are never resumed across a code change.

## 8. Amendment A1 before submission (written 2026-10-22 16:05 EDT; before any TIE arm exists)

What was seen when writing it: the pre-step outputs of jobs 337675, 337993 and the smoke 337910 (listed in JOBS.md);
no TIE checkpoint, score or flip residual of any arm (none exists).

1. **Instrument of gate (ii) and rule F: `flip_probe_tie2.py`** (md5 63145842) replaces `flip_probe_tie.py`. ATTRIB-090's
   compensation negates the geometry columns only, which is complete for the lug's single scalar output (von Mises) but not
   for outputs of rank >= 1. ISLA assembles those from the head's coefficients and products of the true basis `[n, rhat,
   pc(rhat, n)]` (K = 0), and `n` and `pc(rhat, n)` are odd under the flip. v2 also negates the head rows (weight and bias) of
   every coefficient whose multi-index holds an odd number of them. Evidence: v1 on Tensile2d (337675) gives d_F median 1.59
   (displacement) and 0.61 (stress) under base and branch alike, while the scalar q is 2.7e-5 (base) and exactly 0
   (branch); v2 (337993) gives exactly 0 on the branch in every field. On the lug v2 equals v1 (head signs all +1; same
   numbers, 1.49e-7). The thresholds of gate (ii) and F are unchanged.
2. **Gate (i), CPU half:** run on a short-QoS GPU node with `geo_flip_probe_tie2.py --geom-device cpu` (the CPU path of
   exact_boundary_geometry; the dataset loads as in training). The recipe datapipe pins memory and cannot run on a CPU-only
   node (338014). Thresholds unchanged. Cell counts (section 3 (iii)) move to the same job.

## 9. Amendment A2 before submission (written 2026-10-22 16:05 EDT; supervisor steer of 16:00 EDT)

What was seen when writing it: everything A1 lists, the code_tie pre-step and smoke outputs (337675, 337993, 337910), and
TIE-REVIEW (`scratch/isla-tie-review-2026-10-22/REVIEW.md`). No TIE arm, score or code_tie2 output exists.

1. **The code under test changes.** d3d9da49d (code_tie) has a blocking defect (TIE-REVIEW finding 1): with float32 inputs
   the on-wall rule returns u = 0 at most on-wall nodes. The on-wall directions are float32 unit normals cast to float64
   without renormalizing, and `_min_norm_points` assumes exactly unit inputs (geometry.py:637, 270, 281-282). The code under
   test becomes **code_tie2** = code_cmpaf_fix + geometry.py and tree_read.py of the TIE-FIX commit on
   psharpe/isla-medial-tie (taken only by `git show <sha>:<path>`), proven as code_tie was. Its SHA and md5s are recorded
   in A3 before any code_tie2 job runs. code_tie stays as it is, read-only, never used again. Launcher `$T/kernel/tie2`
   (kernel/tie's files with code_tie2 and roots kernel/tie2 / evals_tie2); probe outputs `$T/kernel/tie2/probe/out`. Arm
   names, recipe, rows and rules are unchanged.
2. **Superseded records.** Every pre-step and smoke on code_tie (337675, 337993, 337910; 338161 cancelled, 337674 / 337934
   / 338014 failed before any geometry) was **run on the defective snapshot, superseded**, and is not read for the gate. The
   base-side (code_cmpaf_fix) outputs of those jobs are kept as records only; the gate job reruns the base beside the branch.
   (My code_tie geometry probe had counted |u| < 1 at 52,506 of 192,172 lug surface nodes and 15,544 of 45,692 Tensile2d
   boundary nodes, consistent with the defect; flip residuals cannot see it, since u = 0 is odd.)
3. **Gate item (a): feature validity in float32** (`f32_validity.py`). Every val case of the lug (600) and Tensile2d (100),
   loaded as in training (stored float32), geometry from the float32 inputs against the same computation on float64 copies
   of them; on-wall nodes are d64 == 0. GPU path (as in training) on all cases, and the CPU path on the lug's first 60 and all
   100 Tensile2d cases. Pass (both paths, both datasets): every on-wall `|u32 - u64| <= 1e-5`, and the count of on-wall nodes
   with `|u| < 0.5` from float32 inputs equals the float64 count, in total and in every case. The count is reported. The base
   (code_cmpaf_fix) is run beside it as a record (the lug's first 60 cases, Tensile2d's 100), not gated.
4. **Gate item (b): the two geometry test files on GB300 from code_tie2** (`test_isla_interior_geometry.py`,
   `test_isla_tree_read.py`; `python -m pytest -rA`, CUDA visible, short QoS). Pass: no failure or error, and no skipped
   test except `test_saved_posed_airfrans_case_runs`, which is gated on the ISLA_AF_POSED_SAMPLE file and is not a CUDA test.
   Every CUDA test (`skipif(not torch.cuda.is_available())`) must run and pass.
5. **The gate becomes:** arms launch only if (i), (ii), (a), (b) and both smokes pass on code_tie2, as registered. Gate (i)'s
   exception (DESIGN c.2, `|d_N| <= 1e-6 L` and `d_F == d_N`) covers TIE-REVIEW finding 3's points by its text (float32 d
   rounded to 0 at a tie); any such node is reported with its examples, whichever column the probe counts it in.
6. Instruments: `$T/kernel/tie2/probe/` holds copies of geo_flip_probe_tie2.py (cb98a614), flip_probe_tie2.py (63145842),
   cellcount.py (8cf5c0c1), the new f32_validity.py and gate2.sbatch (md5s in JOBS.md and kernel/tie2/LAUNCH_MD5S.txt).

## 10. Amendment A3 before submission: evidence for the stated assumption of section 1 (written 2026-10-22 19:37 EDT)

What was seen when writing it: everything A2 lists, and FOLD-SIGN's counts below, passed on by the supervisor (16:31 EDT).
No TIE arm, score or code_tie2 output exists.

FOLD-SIGN (lane 666bae9d) compared the winding-number sign with d3d9da49d's geometry, on CPU, with the datapipe's float32
inputs (LHR cpu-short jobs 338225 and 338281):
- Lug, all 600 test cases: 0 sign changes in 3,768,183 off-wall nodes; |d| and u bitwise equal at every node; identical
  on-wall sets; 0 nodes in the far-field error band (|w - sigma/2| < 3e-2), minimum margin 0.471.
- Tensile2d val_100: 0 sign changes in 922,124 off-wall nodes, bitwise equal, band 0, minimum margin 0.5 (exact in 2-D).
- AirfRANS full_test, 200 cases (not an arm here): 0 sign changes in 35,645,073 off-wall nodes, bitwise equal, band 0.

Together with the cell count of section 3 (iii), these are the evidence for the stated assumption that TIE stands for the
whole geometry bundle on these two datasets. **Caveat:** they were measured against d3d9da49d, before TIE-FIX. TIE-FIX changes
the on-wall directions (where d = 0 and there is no sign), and may change d's sign where its float32 magnitude rounds to 0.
FOLD-SIGN repeats the comparison against the fix commit. The assumption is stated for code_tie2 only once that repeat is
recorded here, before any arm result. If the repeat shows sign changes, the assumption is withdrawn and TIE stands for the
tie rules only.

## 11. Amendment A4 before submission: gate item (a) gets an exact reference (written 2026-10-22 19:40 EDT; supervisor 16:40 EDT)

What was seen when writing it: everything A3 lists, and TIE-FIX's finding, passed on by the supervisor: near-duplicate
normals on a torus give on-wall u = 0 from float64 inputs as well. So a float32-against-float64 comparison alone can pass
while both are wrong. No code_tie2 output exists.

Gate item (a) becomes, on every val case of the lug (600) and Tensile2d (100) at their stored float32 precision (GPU path),
and on the lug's first 60 and all 100 Tensile2d cases on the CPU path; `f32_validity.py` v2 (md5 ae9ad05f):
1. every on-wall `|u32 - u64| <= 1e-5` (as registered in A2);
2. every on-wall `|u32 - uref| <= 1e-5` and `|u64 - uref| <= 1e-5`. `uref` is the exact least-norm point of the hull of
   the incident cells' normals. The incident cells are those holding the node as a vertex, welded by exact coordinates.
   Their normals come from `compute_cell_normals` of the float64 copy in stored vertex order, renormalized in float64, with
   degenerate cells (zero or non-finite normal) dropped and distinct directions only. `uref` is solved by TIE-REVIEW's
   `exact(c, tol=1e-14)` (probes/probe_minnorm_fail.py), batched over the nodes with the same direction count, with one
   correction to the copy: `best.norm()` becomes `best[0].norm()`, since the original raises AttributeError whenever a
   second subset is feasible. The batched solve equals `exact` to 3.4e-16 on 480 random sets and to 5.5e-15 on all 642
   vertices of a float32 icosphere (|uref| 0.9956-0.9973, the reviewer's 0.99596 at vertex 12). The job checks 50 nodes per
   case against `exact` (pass: within 1e-12). Every on-wall node must be a mesh vertex with a valid reference (counts
   `nonvertex` and `ref_invalid` both 0), else (2) is incomplete as registered.
3. the count of on-wall nodes with `|u| < 0.5` from float32 inputs, from float64 inputs and from the exact reference agree,
   in total and in every case.
All three counts are reported (and the zero counts beside them). The base runs beside it as a record, not gated. Everything
else in the gate is unchanged.

## 12. Amendment A5 before submission: code_tie2, and gate (a) items 4-6 (written 2026-10-22 19:53 EDT; supervisor 19:50 EDT)

What was seen when writing it: everything A4 lists; FIX_COMMIT.txt; my local CPU checks of the instrument on a posed float32
icosphere (below). No code_tie2 job has run; no TIE arm, score or code_tie2 output exists.

1. **Code under test (completes A2.1):** `$T/code_tie2` = code_cmpaf_fix + geometry.py (**f1810d97**) and tree_read.py
   (**c1d34bc1**) of **19fb0a754e8b55f5d33ed30681f96c8ff88ac059** (FIX_COMMIT.txt; psharpe/isla-medial-tie). Against
   d3d9da49d only geometry.py changes. On-wall directions are renormalized in float64, degenerate cells are dropped from the
   on-wall set, `_min_norm_points` keeps only certified candidates, the tie key is a stacked unique, and a tie's d is signed on a
   float64 distance. MD5SUMS 49aeb192 (3,248 files), SNAPSHOT_HEAD 517e199a. Overlay proof: `evidence/overlay_proof_tie2.txt`
   (only SNAPSHOT_HEAD, MD5SUMS and the two files differ from B's snapshot; both equal `git show 19fb0a754:<path>`;
   code_tie's MD5SUMS unchanged). Gate (b) runs the test files of 19fb0a754 (a read-only `git archive 19fb0a754 test
   pyproject.toml` at `$T/kernel/tie2/testtree`) against code_tie2.
2. **Gate (a) instrument v3: `f32_validity3.py` (md5 95c82f1f)**, replacing v2 (never run). Items 1-3 as in A4. New items:
   - (4) at the routine's boundary: in the probe only, `geometry._min_norm_points` is wrapped. Every row of every call (input
     direction sets and outputs, from the float32 and the float64 computation, on-wall and off-wall) is compared with
     `exact(c, tol=1e-14)` (TIE-REVIEW's, copied in, with the `best[0]` correction of A4). It is batched over rows of equal k,
     with the copied `exact` itself run on 20 rows per call beside it (pass within 1e-12). Pass: `|u - exact| <= 1e-5` on every
     row. Reported: row count, worst error, and the rows where the routine returns exactly 0 (no candidate passed its
     certificate), split into exact point 0 (origin in the hull) and nonzero.
   - (5) gray-zone queries: `1e-12 < |d64| < 64 * 2^-24 * M`, M = max Euclidean norm over the case's query points and
     boundary vertices. Count per dataset; pass if 0. **If the lug or Tensile2d has any, I stop and tell the supervisor.**
   - (6) boundary cells whose normal (compute_cell_normals from the float32 and the float64 inputs) is zero or non-finite.
     Count per dataset, reported (not gating).
   Meshes: the lug's test split (600) and Tensile2d's val_100 (its scored set), plus a seeded random 20 % of the training
   meshes (seed 20261022: lug 480 of 2,400, Tensile2d 80 of 400), from the stored float32 inputs.
   Paths: the **CPU path** as cpu-short array `gate_a_cpu.sbatch` (31 shards, at most 4 at once, `dataloader.pin_memory=false
   dataloader.use_streams=false`: memory placement only) on all those meshes; and the **GPU path** (training path) in the gate
   job on the test sets. Both must pass. The base runs beside it on the GPU path as a record (lug 60, Tensile2d 100).
   Instrument check before registration (local CPU, posed float32 icosphere, 642 vertices + 1,280 edge midpoints + 3,000
   random points): on d3d9da49d, 616 of 1,509 float32 rows fail (routine 0, exact nonzero; worst 0.9994) and 210 vertices have
   |u| < 0.5, the reviewer's count. On 19fb0a754: 0 rows fail (worst 6.2e-15), 0 vertices short, max |u32 - uref| 1.2e-7.
3. **Transfer to TIE-FIX2's coming commit:** if every row of item 4 passes on both paths, that commit (a rule for rows with no
   certified candidate, plus a test) cannot change these datasets' features, and the arms stay on code_tie2 without a
   rebuild. If any row has no certified candidate and a nonzero exact point, this transfer argument does not hold; I report it
   and ask before launching.
4. **Gate:** arms launch only after (i), (ii), (a) items 1-6 (6 reported only), (b) and both smokes pass on code_tie2.
   Smokes: `$T/kernel/tie2/smoke/smoke.sbatch`, criteria of section 3.

## 13. Amendment A6 before any arm result: transfer to 34654e01d, and the supervisor's decision rule for (a)(1)/(2) (written 2026-10-22 20:33 EDT)

What was seen when writing it: everything A5 lists. I have not read any code_tie2 gate output. The supervisor read partial
output at 20:15 EDT (`logs/f32v3_gpu_lug_code_tie2.log`, 105 lug case lines) and reported it at 20:16 EDT (message headed 20:20; time corrected by the supervisor): on 18 of the 105
cases, one on-wall node has |u32 - u64| just above 1e-5 (max 1.124e-5, case 01432), with |u32 - uref| about the same and
|u64 - uref| <= 3e-10. All the other counts on those lines are 0.

1. **Transfer to 34654e01dd58c4175ed4097104cdfe53c067a839** (TIE-FIX2; psharpe/isla-medial-tie). Against 19fb0a754 it changes
   only geometry.py's `_min_norm_points`, and only on rows with no certified candidate (`certified` false). There
   19fb0a754 returns 0, and 34654e01d returns the separating candidate with the least duality gap, or 0. Every certified row
   takes the same `norm.argmin` pick in both, so it is bitwise equal. In 19fb0a754 a row with no certified candidate returns
   exactly 0, so it is counted in item (4)'s `b_zero_out`. **If `b_zero_out` is 0 on both paths, both precisions and both
   datasets, code_tie2 computes exactly what 34654e01d computes on every row the lug and Tensile2d produce, and the arms stay
   on code_tie2.** If it is not 0, the rows are reported and I ask before launching.
2. **Supervisor decision rule (approved 20:16 EDT; message headed 20:20), fixed before the full gate (a) output is read.** If (a)(1) and the u32 half of
   (a)(2) fail and every other gate item passes, a cpu-short diagnosis (`f32_diag.py`, at most 30 min) runs first. For every
   on-wall node with |u32 - u64| > 1e-5 on the lug and Tensile2d it reports:
   - (i) `max |n32 - n64|` over the node's incident cells (each cell normal computed from the float32 inputs in float32
     arithmetic, and from the float64 copy in float64; both renormalized in float64);
   - (ii) the sine of the smallest angle of those cells;
   - (iii) `r = |u32 - u64| / max |n32 - n64|`;
   - (iv) the node's |u32 - u64| under code_cmpaf_fix, as a record.
   It also reports the node count, and the max of max_du_off_wall for code_tie2 and the base (record, not gated).
   - r <= 2 on every such node, and every other gate item passes: a supervisor-approved deviation from (a)(1)/(2), recorded
     here before launch with its cause, the bound `|u32 - uref| <= r max |n32 - n64|` and the max r. The arms launch without a
     further wait.
   - r > 2 at any node: no launch. Each such node is reported with K, the pairwise angles of its normals, and |uref|.
   - Any other gate item fails: stop and report, as registered.

## 14. Amendment A7 before any arm result: supervisor-approved deviation from gate (a)(1)/(2) (written 2026-10-22 21:10 EDT)

What was seen when writing it: the full code_tie2 gate (jobs 341643, 341644, cpu-short 341647-342048), the diagnosis 342298
and the transfer and base-(ii) job 342299 (`gate2/gate_summary.txt`, `gate2/diag_transfer_summary.txt`, `gate2/DIAG_NODES.md`).
No TIE arm exists.

1. **Gate as read.** Failed as registered: (a)(1) and the u32 half of (a)(2), on the lug only, and f32_validity3's
   instrument self-check (`b_check_dev` 5.2e-9 > 1e-12, on 3 float64 rows with k = 5 near-duplicate directions). Passed:
   - (i) the geometry flip on the CPU and GPU paths, both datasets: 0 nodes with |nu_F + nu_N| > 1e-6, max |d_F + d_N| 0. The base
     reproduces the defect (lug 8,428 GPU and 3,142 CPU interior mismatches).
   - (ii) branch: lug d_F median 1.46e-7 (max 2.3e-7), Tensile2d exactly 0 in every field, d_U medians 1.4-93. The base lug
     d_F median is 1.53e-4 (342299).
   - (b): 92 passed, 1 skipped (the ISLA_AF_POSED_SAMPLE file test, allowed).
   - Both smokes: rc 0, eager, bfloat16, lug 0.243 s/step, Tensile2d 0.215 s/step.
   - (a)(3): on-wall counts of |u| < 0.5 are 0 / 0 / 0 on both datasets. (a)(4): b_over 0, worst 2.1e-8 over 5.1M rows.
     (a)(5): gray zone 0 on both datasets (min |d|/band 1,800). (a)(6): 0 degenerate cells on both.
   - Cell counts: every boundary cell kept on both datasets (max 8,568 and 496 against 200,000).
   - A6.1 transfer to 34654e01d, measured directly (f32_transfer.py, 342299): 0 uncertified rows and 0 bitwise differences in
     5,143,196 rows (both datasets, both precisions). code_tie2 computes exactly what 34654e01d computes on these data.
   Tensile2d passes every item of (a) (max |u32 - u64| 7.2e-8).
2. **Cause, as measured.** The routine returns the exact least-norm point of the normals it receives, at both precisions:
   |u32 - exact(n32)| <= 3.6e-8 and |u64 - exact(n64)| <= 6.5e-12. The point itself is ill-conditioned where the hull has a thin
   face spanned by near-duplicate normals. These are the lug's split hex faces: the normals at the 190 nodes form clusters
   0.0005-0.003 degrees wide, 9-12.9 degrees apart (K = 4-6, |uref| 0.9937-0.9969). The face's thinness bounds the movement,
   so |u32 - u64| stays at or below the within-cluster spread (max 2.07e-5) while r = |u32 - u64| / max |n32 - n64| reaches
   2,261 (median about 300; max |n32 - n64| 4.8e-9 to 7.5e-8; sin of the smallest cell angle 0.50-0.70, not thin cells). The
   base gives |u32 - u64| <= 8.8e-8 at those nodes. The supervisor's 20:16 rule assumed one-to-one propagation (r <= 2). That
   premise was wrong, and **the supervisor overrode its do-not-launch branch at 21:10 EDT, after seeing the diagnosis.**
3. **Bound:** |u32 - uref| <= 2.1e-5 at 190 of about 3.8M lug on-wall nodes (101 in 73 of 600 test cases, 89 in 70 of 480
   sampled training cases); Tensile2d at most 7.2e-8 everywhere.
4. **Why F, N and P are unaffected (stated before any result).** F compares a winding flip within one precision, where the
   rule negates u bitwise, so conditioning does not enter it. For N and P the network reads these features under bf16
   autocast, about 200 times coarser than 2e-5, and only 0.005 % of on-wall nodes are affected.
5. The conditioning of the least-norm point at near-duplicate normals goes to the design agenda as its own question, not this
   experiment's.

The arms are submitted on code_tie2 as registered (`$T/kernel/tie2`, lanes 0-3), with the rules of section 4 unchanged.

## 15. Amendment A8 before any arm result: transfer to ONWALL's on-wall rule (written 2026-10-22 21:44 EDT; supervisor 21:18 EDT)

What was seen when writing it: everything A7 lists; the arms are pending (no TIE result exists).

Beside A3's fold-sign evidence. ONWALL (psharpe/isla-onwall 1e2c431cd0de9a8cf6714a35f2e75f4450c27cd3) puts a query on the wall
when its float64 distance is within its closest cell's bound E = (u_p + (3d + 4) u_64)(|x| + max |v|), in place of the fixed
1e-12 cutoff. In float32, E <= 2 * 2^-24 * M, 32 times smaller than gate (a)(5)'s gray band 64 * 2^-24 * M. The gray count is 0
on every gated lug and Tensile2d case (min |d| / band 1,800). So no query of those cases changes its on-wall status under
ONWALL's rule, and the arms on code_tie2 stand for that rule too on these datasets. ONWALL's own measurement agrees: 0 queries
moved on 3 lug and 3 Tensile2d cases at both precisions.

Beside gate (b): the full ISLA suite on 19fb0a754 passed on CPU at 20:59 EDT, 290 passed, 49 skipped, 0 failed
(`scratch/isla-main-2026-10-14/continuity-bisect-2026-10-22/suite_19fb0a754.log`).

## 16. Amendment A9, records only: FOLD-SIGN's repeat against 19fb0a754 (written 2026-10-23 11:37 EDT; supervisor 04:03 EDT)

This amendment changes no rule and no verdict.

**What I had seen when writing it.**
- Tensile2d rule F, read 01:25 EDT 10-23: the TIE d_F is exactly 0 in every field for both seeds. B beside it: displacement
  8.0-8.2e-7, stress 1.4-1.5e-6, q 2.7-3.0e-5.
- The Tensile2d TIE score files exist (`$T/evals_tie2`, local copy `read/evals_tie2/`). I checked their completeness (epoch 500,
  100 cases, summary row, .done, SNAPSHOT_USED code_tie2), but not their metric values.
- Of the lug arms: the monitor's status lines (epoch, train.log age, link states, a zero count of non-finite loss lines) and the
  acceptance checks of 21:45 EDT 10-22 (resolved configs, step times, no traceback). No validation curve or loss value of a
  lug arm.
- The one-epoch smoke losses (code_tie 337910, code_tie2 341644).
- Of B's lug pair: training-loss log lines on 10-22 (epochs 342 at 14:52 EDT, 496 / 482 at 20:38 EDT), no score.
- The supervisor had seen B's lug scores from 22:04 EDT 10-22 on. I have opened none of the files on the supervisor's
  blinding lists (22:34 EDT and 01:00 EDT) since 22:04 EDT 10-22.

**The record.** A3 (section 10) stated the whole-bundle assumption for code_tie2 "only once that repeat is recorded here, before
any arm result". FOLD-SIGN ran the repeat against 19fb0a754 before any TIE arm existed: LHR 341328 (lug, Tensile2d) and 341486
(AirfRANS), which completed 19:44-19:53 EDT 10-22, 19fb0a754 against the merge 702f60f97 (DESIGN k.4 of
`scratch/isla-medial-tie-2026-10-22/DESIGN.md`; FOLD-SIGN's JOBS.md). The counts were the same as against d3d9da49d:
- lug test, 600 cases: 3,768,183 off-wall nodes, 0 sign changes, |d| and u bitwise, on-wall sets equal, 0 in the band, minimum
  margin 0.471;
- Tensile2d val_100: 922,124 off-wall nodes, all 0, margin 0.500;
- AirfRANS full_test, 200 cases: 35,645,073 off-wall nodes, all 0, margin 0.500.
GEO-AUDIT reproduced these counts at 03:20-03:29 EDT 10-23 (346922_10, 347401, 347555), with a preflight check that each
version ran its own search (`scratch/isla-geo-audit-2026-10-23/AUDIT.md`, which holds no scores).

**Status at the read.** No amendment through A8 recorded the repeat, and this record comes after the first arm result (the
Tensile2d arms completed 00:44-01:14 EDT 10-23; rule F on Tensile2d was read at 01:25 EDT). So A3's condition is reported as
"not met as registered". The transfer statement to the whole bundle is reported beside the verdicts, labelled "recorded late
(A9); the repeat ran before any arm existed".
