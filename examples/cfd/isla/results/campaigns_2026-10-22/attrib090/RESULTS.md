# ATTRIB-090 results

Rules: PREREG.md §5 as amended in §9 (md5 502cca2a). Jobs and md5s: JOBS.md. Reduction: `reduce.py` (tables
`t2d_eager_native.md`, `t2d_eager_posed.md`, `scorer_native.md`; JSON `results_native_eager.json` etc.).

## 1. Tensile2d (read 2026-10-21 ~02:50 EDT; 100 val cases, epoch 500, float32; four seeds 42-45 per arm)

Verdicts on the fully eager scores (§9a), job 312322 (arms, CONT) and COMPILE-AF's `evals_cmpaf_eager` (base seeds 42 / 43).
Scorer-path verdicts beside, labelled "trained under the compiled path; Tensile2d divergence small" (jobs 312119 / 312120).
B = 0.9.0 (r090_t2dz_isla 42 / 43 + att_t2dz_base 44 / 45, same snapshot and row). CONT = E2 CONT (gcont_cont_t2dz 42 / 43 +
attc_t2dz_cont 44 / 45). Pooled σ(log) over the six arms (18 dof): displacement 0.123, stress 0.063, q 0.263.

Displacement, native, eager (per seed 42 / 43 / 44 / 45; arithmetic seed mean; ÷ B = exp(Δ), the ratio of geometric seed means,
which differs from the ratio of the mean column by 0.8-1.1 % (COND 1.134, SDF 1.136, ALL 0.928); R1; R2):

| arm | per seed | mean | ÷ B | Δ / SE | R1 / R2 | scorer path ÷ B, R2 |
|---|---|---|---|---|---|---|
| B (0.9.0) | 0.00729, 0.00697, 0.00737, 0.00708 | 0.00717 | 1 | – | base | 1 |
| CONT (E2, 0.6.0) | 0.00662, 0.00626, 0.00711, 0.00716 | 0.00679 | 0.945 | −0.057 / 0.087 | no; **R4: the step is not real** | 0.935, no |
| COND (one labelled boundary) | 0.00683, 0.00723, 0.01001, 0.00849 | 0.00814 | 1.122 | +0.115 / 0.087 | no measurable effect | 1.113, same |
| SDF (stored SDF back) | 0.00995, 0.00704, 0.00752, 0.00810 | 0.00815 | 1.127 | +0.119 / 0.087 | no measurable effect | 1.112, same |
| QEMB (0.6.0 query embedding) | 0.00987, 0.00823, 0.00877, 0.00830 | 0.00879 | **1.223** | +0.201 / 0.087 | **moves, opposes** | 1.220, opposes |
| ALL (COND + SDF + QEMB) | 0.00720, 0.00518, 0.00724, 0.00700 | 0.00666 | 0.919 | −0.084 / 0.087 | no measurable effect | 0.908, same |

Stress (eager ÷ B): CONT 0.927, COND 1.017, SDF 1.021, QEMB 1.020, ALL 0.943, none moves. Plastic strain q: none moves
(σ 0.26). Posed equals native to three digits for every arm (`t2d_eager_posed.md`).

**Reading.**
- **R4: the Tensile2d step is not real.** With four seeds each, 0.9.0 ÷ CONT on displacement is 1.058 (CONT ÷ B 0.945,
  Δ = −0.057 against 2 SE = 0.17). The two new CONT seeds (0.00711, 0.00716) sit inside 0.9.0's range (0.00697-0.00737), and
  E2's two seeds (0.00662, 0.00626) were the low ones. The registered 1.115 was mostly two-seed noise. Per R4 no cause is named
  and S4 does not fire (no lug mirrors).
- **0.9.0's displacement is the most reproducible of the six arms** (four seeds within 5.6 %, on these defect-trained models).
  Each other arm has one seed 10-33 % from the mean of its other three: CONT s43 10 % low, QEMB s42 17 % high, ALL s43 28 % low,
  SDF s42 32 % high, COND s44 33 % high. (Corrected 2026-10-22 from the second read: "5.7 %" and "15-40 % away" were wrong,
  and CONT failed the old wording.)
- **The one solid effect is QEMB's 1.223.** All four QEMB seeds lie above all four 0.9.0 seeds. So the BC-NAMES
  prediction-point embedding, which puts the distance channels in the query's own seed layer, beats the 0.6.0 one
  (a type offset plus a separate feature MLP added to the seed embedding) on Tensile2d displacement by about 18 %
  (1 − 1/1.223). The result does not isolate why.
  Beside it: ALL also uses the 0.6.0 embedding, but with the stored-SDF features in its MLP and one labelled boundary.
  It does not show the cost (0.919), so the pieces interact.
- Beside the rule (not a registered verdict), for REF-090: the Tensile2d flag, 0.9.0 ÷ A2 = 1.041 (1.045 eager), is smaller
  than the 5.6 % spread of the four 0.9.0 seeds, but both A2 seeds (eager 0.00679, 0.00685) still lie below all four 0.9.0
  seeds (lowest 0.00697).
- For the program: four seeds see no accuracy cost on Tensile2d anywhere in the CONT → 0.9.0 bundle. The 0.9.0 standing
  rests on principled pieces: exact geometry, CONT, named conditions, no stored SDF. Confidence: high that there is no ≥ 10 % step; moderate on the
  QEMB direction (one arm, four seeds, a 2.3 SE effect).

## 2. AirfRANS, interacting ISLA (read 2026-10-22 ~02:10 EDT; 200 full_test cases, 100,000 points, epoch 500, float32)

Label: **fix-trained** (PREREG §9b). REF-090-CLOSE's training-path measurement on AirfRANS, T = r090fix_af_isla ÷ bug-trained
r090_af_isla (both eager) = 1.024 / 0.985 / 0.998, inside the A2-pair seed gap
(scratch/isla-ref090-2026-10-20/r090_final_tables.md, section REF-090-FIX AirfRANS), so these are the model's own function in
substance. B = r090fix_af_isla (eager scores, `evals_r090fix`); CONT = att2c_af_cont (E2's row retrained under the fix,
eager); arms on code_att090fix (fixed compiled scorer). Pooled σ(log) 0.058 / 0.052 / 0.060 (5 dof). Tables:
`relaunch_native.md`, `relaunch_posed.md`.

| arm | p (s42, s43) | v | nut | ÷ B (p / v / nut) | φ (p / v / nut) | R2 |
|---|---|---|---|---|---|---|
| B (0.9.0, fix-trained) | 0.01222 (0.01299, 0.01146) | 0.00633 | 0.01556 | 1 | – | base |
| CONT (E2 0.6.0, fix-trained) | 0.01569 (0.01564, 0.01574) | 0.00856 | 0.02197 | 1.286 / 1.353 / 1.411 | 1 | the step D |
| **NOSC** (no ISLA scales) | 0.01503 (0.01502, 0.01504) | 0.00857 | 0.02072 | **1.232 / 1.354 / 1.331** | **0.83 / 1.00 / 0.83** | **carries the step on all three** |
| SDF (stored SDF back) | 0.01279 (0.01204, 0.01355) | 0.00653 | 0.01574 | 1.047 / 1.030 / 1.009 | 0.18 / 0.10 / 0.03 | no measurable effect |
| QEMB (0.6.0 query embedding) | 0.01207 (0.01244, 0.01171) | 0.00627 | 0.01599 | 0.989 / 0.991 / 1.025 | −0.04 / −0.03 / 0.07 | no measurable effect |

Posed ÷ B reproduces native to three digits (NOSC 1.231 / 1.353 / 1.331).

**Reading (relabelled 2026-10-22 10:10 EDT, PREREG §10).** NOSC's numbers and its R1 / R2 labels stand, but its attribution is
confounded: NOSC − B changes four things at once (the physical nondimensionalization U L / ν; a standardized global input z
against the raw speed in m/s, about 31-94, fed unstandardized into every seed; nut's units; the fp32 head). So **the scales
together with the standardized global input carry the step (confounded; NOSC-STD pending)**. Taking all of that out puts back
83-100 % of the step on every field, and both NOSC seeds sit at CONT's level. The stored-SDF removal and BC-NAMES' query embedding have no measurable effect on
AirfRANS. Triggers: S1 fires (AF-RAWU, AF-NUT0, DTT-NOSC launched 02:14 EDT to split the scales into the Reynolds-number input
and the nut units, and to test ISLA-DTT); S2, S3 do not. Principled or side effect: the scales replace an unstandardized
dimensional input (a speed of 30-90 m/s fed raw into every token's seed) by the dimensionless log Reynolds number, and make
nut scale with U L per case. Whether that is accuracy for the right reason depends on NOSC-STD (§10): if standardizing the old input alone recovers the gain,
the 0.9.0 gain came mostly from fixing an unstandardized input, not from the physics of the scales.

## 3. AirfRANS, ISLA-DTT and R5 (read 2026-10-22 ~06:40 EDT)

B = r090fix_af_dtt (eager); DTT-CONT = att2c_af_dtt (code_gcont + the variable; deviation 1); arm D = E3b anc_dtt, **bug-trained,
eager re-scores** (COMPILE-AF impact/). Pooled σ 0.026 / 0.016 / 0.070.

| arm | p | v | nut | ÷ B | R1 |
|---|---|---|---|---|---|
| B (0.9.0 ISLA-DTT, fix-trained) | 0.01265 (0.01274, 0.01257) | 0.00713 | 0.02149 | 1 | – |
| DTT-CONT | 0.01404 (0.01360, 0.01447) | 0.00833 | 0.02199 | 1.109 / 1.169 / 1.026 | moves on p, v |
| arm D (A2 era, bug-trained, eager) | 0.01395 | 0.00868 | 0.02246 | 1.103 / 1.218 / 1.047 | moves on p, v |

- DTT-CONT ÷ arm D = 1.006 / 0.960 / 0.979: exact geometry and CONT did not move ISLA-DTT.
- **R5: ρ_CONT = DTT-CONT ÷ CONT = 0.895 on pressure (≤ 0.95): the reversal happened in the 0.9.0 bundle.** At CONT, ISLA-DTT
  led ISLA by about 10 % on pressure (velocity 0.973, nut 1.001). At 0.9.0 (fix-trained), ISLA-DTT ÷ ISLA is 1.035 / 1.126 / 1.381.
- Both forms gained from the bundle, ISLA more: ISLA 0.778 / 0.739 / 0.709, ISLA-DTT 0.901 / 0.856 / 0.977 (B ÷ CONT).
- Beside the rules: the compile defect cost ISLA-DTT about 10 % at 0.9.0 (r090fix_af_dtt ÷ bug-trained r090_af_dtt, both eager,
  0.898 on pressure), against T ≈ 1 for ISLA. Part of REF-090's "ISLA-DTT barely moved" was the defect.
- Why ISLA gains more: open until DTT-NOSC reads; DTT-NOSC carries the same raw-speed confound as NOSC (labelled when read).

## 4. The lug (provisional, read 2026-10-22 09:05 EDT; 600 test cases, epoch 500, float32)

Labels: **fix-trained; pending the lug training-path read** (REF-090-CLOSE's lug T = fix-trained ÷ bug-trained = 0.768: on the
lug the training path matters). B = r090fix_lug_isla (eager scores). CONT for now = E2's bug-trained gcont_cont_lug scored
eager (labelled); the registered CONT is the retrained att2c_lug_cont (running; the trigger fired). The lug's verdicts are
directional (noisy field). Pooled σ(log) 0.036 (4 dof, native).

| arm | native (s42, s43) | ÷ B | R1 / R2 | posed ÷ B |
|---|---|---|---|---|
| B (0.9.0, fix-trained) | 0.00145 (0.00146, 0.00144) | 1 | base | 1 (0.00198) |
| CONT (E2, bug-trained, eager; provisional) | 0.00177 (0.00177, 0.00178) | 1.220 | the step D (provisional) | 0.951 |
| FLIP (E2's winding, fix-trained) | 0.00158 (0.00151, 0.00166) | **1.088** | moves; directional | 1.320 (s42 0.00348) |
| ENC (0.5.0 encoding, **eager-trained**) | 0.00166 (0.00169, 0.00162) | **1.141** | moves; directional | 0.908 |

L-ENC scored on the fixed compiled path (beside, labelled "compiled differs from eager on 5 of 60 cases of one early checkpoint;
JOBS.md deviation 2"): 0.00170 / 0.00162, mean 0.00166, the same as eager.

**Reading (provisional).**
- R6, ENC: taking CONT out at 0.9.0 costs the lug 14 % (P12, which expected a gain, fails). E2 had measured the opposite
  (CONT ÷ CONTROL 1.128, both seeds) on bug-trained models with the flipped winding. But ENC trained eager and B on the fixed
  compiled path, and on the lug the training path moves the error by about 23 %, so the like-for-like base for ENC is
  REF-090-CLOSE's eager-trained r090eag_lug_isla (due about 18:00-20:00 EDT); both readings will be reported, the registered
  verdict on the registered base.
- R6, FLIP: E2's winding (normals into the solid) costs 9 % against the stored winding of 0.9.0, both FLIP seeds above both B
  seeds. P11 (no effect, 0.80) fails. 0.9.0's choice is the better one, so the REF-090 lug standing has no orientation defect,
  but ISLA is not invariant in distribution to a global winding flip on the lug. The flip negates every odd geometric input
  consistently (s, sign(d) u, the pseudo-normal at the 33 % of nodes with s = 0, the boundary normals), so a network whose odd
  inputs all entered through linear maps with symmetric initializations would be indifferent; ISLA's are not all linear (the
  per-layer geometry region acts nonlinearly on normal-relative products). **Withdrawn (2026-10-22, PREREG §10):** blocks.py maps
  rel·n and rel·m linearly (geo_logit, geo_feat) before any nonlinearity, so the lug ISLA code admits an exact compensating weight
  negation, and FLIP and B are draws from the same distribution; the zero-training flipped-weight test decides it (§10).
- **Zero-training flip test (job 333695, `launcher/flip_probe.py`, beside R6).** B's checkpoints with qt.seed[:, -2:] and
  geo_logit / geo_feat [:, 2:4] negated in every block, run eager float32 on the flipped lug, against B's own predictions on the
  native lug, 60 cases per seed: median d 1.5e-4 / 1.6e-4, max 4.7e-4 / 4.6e-4 (s42 / s43); case-mean metric 0.001347 against
  B's 0.001343 (s42) and 0.001335 against 0.001333 (s43), 0.1-0.3 % apart. Uncompensated (B's weights on the flipped lug), d is
  1.4-1.7 and the metric 1.35-1.60: the flip matters, and the negation undoes it. The residual is above float32 roundoff (B's
  compiled-vs-eager floor is about 2e-7), so the flip is a reparametrization up to a small residual, presumably the geometry's
  float ties under the swapped vertex order at the 33 % of nodes with s = 0 exactly (not isolated). It is about 30 times smaller
  than FLIP's 8.8 %. **Reading beside R6:** FLIP and B are draws from almost the same distribution of trained models, so FLIP's
  1.088 (both seeds above both B seeds, chance 1 in 6; z 1.4-2.3 by the σ source, reviewer M2) reads as seed noise, not a
  winding convention. R6's registered label (moves; directional) stands as computed.
- **Training is equivariant under the column negation (checked against the code, 2026-10-22).** The recipe's optimizer is
  `build_muon_optimizer` (src/utils.py@92d9d7800): torch Muon for every 2-D parameter, which includes qt.seed.weight and every
  block's geo_logit.weight and geo_feat.weight, and AdamW (elementwise) for the rest (their biases are untouched by the
  negation); weight decay 1e-4 on both. For D = diag(±1) acting on the input columns, a gradient G of W becomes G D; Muon's
  momentum is linear, its Newton-Schulz iteration X ← (a I + b X Xᵀ + c (X Xᵀ)²) X has X Xᵀ unchanged by X → X D (and the
  transposed form, used for tall matrices, gives D on the other side consistently), the Frobenius normalization and the
  shape-only `match_rms_adamw` learning-rate scale are invariant, so orth(G D) = orth(G) D; AdamW, decoupled weight decay,
  global gradient-norm clipping and round-to-nearest are symmetric under negation; the initial weight distributions are
  sign-symmetric. So training on the flipped lug from W0 D follows W_t D step by step, and FLIP and B are draws from the same
  distribution of outcomes up to the measured geometric residual. That is why FLIP's 1.088 reads as seed noise.
- **Where the residual lives (job 334176, `launcher/geo_flip_probe.py`, geometry only, 60 cases, 575,702 nodes).** The flip is
  exact everywhere except at closest-feature ties: d_F = −d_N to 2.6e-7 L on every node; all 192,172 surface nodes (s = 0) stay
  exactly 0 and their ν flips to within 6.7e-4 rad; the boundary normals flip to 1.8e-7 and the gauge to 7e-9. But on **8,428
  interior nodes (2.2 % of the interior, 1.5 % of all nodes; 78-147 per case)** the direction u, and so ν, changes by up to π
  while |d| is unchanged: the node is equidistant (to float32 resolution) from two or more surface features, and the closest-cell
  search (`exact_boundary_geometry` → `_closest_features`, the float32 Warp BVH, physicsnemo/experimental/models/isla/geometry.py
  around lines 188-195 at 92d9d7800) keeps whichever it meets first, which depends on the vertex order. Examples: a node at
  s = −0.102 with u_N = (−0.997, −0.071, 0) and u_F = (−0.997, +0.071, 0); rows of lattice nodes on a corner bisector (angle 0.14
  rad); mid-thickness nodes (angle π). These are the medial ties of DESIGN_NOTE_medial_ties.md (the same nodes drive the lug's
  posed ÷ native of 1.1-1.4). So the hidden convention is an implicit tie rule: first-found closest feature.
- **Proposed principled tie rule (not implemented; a model change for the supervisor to decide):** where several surface
  features attain the minimum distance within float32 resolution of d, take ν as the minimum-norm element of the Clarke
  subdifferential of the signed distance, i.e. the mean of the candidates' unit directions sign(d) u_k (zero when they cancel,
  as on a mid-plane, meaning no preferred direction). It is deterministic, independent of vertex order and of the search,
  equivariant under the winding flip and covariant under rigid motions, so it removes both this residual and the medial part
  of the lug's pose sensitivity. It needs the search to return every feature within d_min + tol, not only the first.
- The lug CONT step D is provisional until att2c_lug_cont lands.
