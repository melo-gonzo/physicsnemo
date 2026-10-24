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
  interior nodes (2.2 % of the interior, 1.5 % of all nodes; 62-245 per case, median 130)** the direction u, and so ν, changes by up to π
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
- Erratum (2026-10-22): the per-case count of tie nodes is 62-245, median 130 (nodes.jsonl, md5 0a543a07); "78-147" was read off
  the log's last three cases. The total 8,428 is right.
- The lug CONT step D is provisional until att2c_lug_cont lands.

## 2b. AirfRANS stage 2: RAWU and NUT0 (read 2026-10-22 17:50 EDT; fix-trained; B and D_f as §2; pooled σ 0.053 / 0.046 / 0.057, 7 dof)

| arm | p (s42, s43) | v | nut | ÷ B | φ | R2 |
|---|---|---|---|---|---|---|
| RAWU (scales on, raw U_inf_mag in place of z; **confounded with input standardization**) | 0.01577 (0.01606, 0.01548) | 0.00855 | 0.02076 | 1.292 / 1.351 / 1.333 | 1.02 / 0.99 / 0.83 | carries the step on all three |
| NUT0 (scales on, Re on, nut without units) | 0.01199 (0.01237, 0.01161) | 0.00638 | 0.01541 | 0.982 / 1.009 / 0.990 | −0.07 / 0.03 / −0.03 | no measurable effect |

**Reading.** nut's units (U L) and, with them, the fp32 head and output_stats path, carry nothing measurable (NUT0). Putting the
raw speed back in place of z, with everything else of the scales kept, puts back the whole step (RAWU ≈ NOSC ≈ CONT). So the
AirfRANS gain from E2 CONT to 0.9.0 sits entirely in the one global scalar: z = standardized log10(U L / ν) against the raw
speed in m/s. Whether that is the physics (a Reynolds number) or the standardization (an O(1) input instead of 31-94) is what
NOSC-STD (§10, running) and STDU (RAWU with the standardized speed; launched 17:53 EDT because RAWU moved by R1: 339539 / 339541)
decide. Interim label: "the standardized global input, with or without its physical nondimensionalization (NOSC-STD, STDU
pending)".

### 4b. The lug with the retrained CONT (read 2026-10-22 20:10 EDT; labels: fix-trained; pending the lug training-path read; directional)

CONT = att2c_lug_cont (E2's row retrained on code_gcont + TORCHINDUCTOR_MIX_ORDER_REDUCTION=0, eager-scored): 0.00167 (0.00162,
0.00173). Pooled σ(log) 0.043 (4 dof). **D = CONT ÷ B = 1.151**: fix-trained 0.9.0 beats fix-trained CONT on the lug by 13 %, the
reverse of REF-090's bug-trained 0.9.0 ÷ CONT = 1.092. FLIP 1.088: Δ 0.085 < 2 SE 0.087, **no measurable effect** (with the
earlier bug-trained CONT it had read "moves" at σ 0.036); consistent with the reparametrization test. ENC 1.141, moves (R6: taking
CONT out at 0.9.0 costs 14 %; eager-trained against a fix-trained base, r090eag pending). Label (deviation 4, accepted): "trained under code_gcont + TORCHINDUCTOR_MIX_ORDER_REDUCTION=0; its compiled block 2
deviates from eager on 5 of 100 lug cases, up to 4.5e-4 of the block's update" (applies to the CONT row and so to every φ and
the FLIP and ENC reads against D).  Detail:
att2c_lug_cont's compiled path deviates from eager by up to 8.6e-4 on 5 of 100 cases (block 2), beyond the float64 floor; its
scores are eager.

**L-ENC against the eager-trained base (beside the registered verdict).** r090eag_lug_isla (eager-trained 0.9.0, REF-090-CLOSE):
0.001376 / 0.001771, a 29 % seed gap. ENC ÷ r090eag = 1.062 native (Δ 0.060; with this pair's own σ the SE is 0.13), 0.812 posed.
So against the like-for-like eager base, taking CONT out shows no measurable cost; the registered 1.141 is against the fix-trained
base, and the lug's training-path sensitivity (fix ÷ bug 0.768; eager ÷ fix here 1.074) is as large as the effect. **R6 for ENC
(registered): moves, directional; beside it: not distinguishable from noise against the eager base.**
**Deviation 4 diagnosis (343394):** the per-block option reproduces the variable's deviation exactly (case 391 8.1e-4 against
8.6e-4), so 0.6.0's lug graph has a second, unidentified fusion difference; 0.9.0's per-block path reads ≤ 4.4e-7 on the lug.

## 3b. DTT-NOSC (read 2026-10-22 23:10 EDT; fix-trained; B = r090fix_af_dtt; pooled σ 0.026 / 0.022 / 0.063, 4 dof)

DTT-NOSC (ISLA-DTT with no scales: the raw speed in m/s, nut standardized, bf16 head; **same raw-speed confound as NOSC**):
p 0.01600 (0.01569, 0.01631), v 0.00873, nut 0.02483; ÷ B = **1.264 / 1.225 / 1.158**, moves on all three; φ against ISLA-DTT's own
step (DTT-CONT ÷ B = 1.109 / 1.169 / 1.026) is 2.27 / 1.30 / 5.75.

**Reading.** ISLA-DTT gains from ISLA's scales (with the standardized input) about as much as ISLA does on pressure (DTT 1.264,
ISLA NOSC 1.232) and less on velocity (1.225 against 1.354) and nut (1.158 against 1.331). Taking the scales out of 0.9.0 costs
ISLA-DTT more than its whole CONT → 0.9.0 step: DTT-NOSC is 14 % worse than DTT-CONT on pressure. So for ISLA-DTT the rest of the
bundle (between E2's code and 0.9.0, without the scales) costs about 12 % on pressure, which the scales more than repay. The
candidates for that cost on ISLA-DTT are the stored-SDF removal (which also took sdf_normals out of ISLA-DTT's tree-read kernel)
and the query embedding; S2 did not fire (SDF and QEMB do not move interacting ISLA), so the DTT-SDF and DTT-QEMB mirrors were
not run and this is not attributed. For the AeroZoo ISLA vs ISLA-DTT decision: the reversal of the order on AirfRANS is not that
ISLA-DTT ignores the scales; it gains from them too, but it also loses something elsewhere in the bundle that ISLA does not.
**Later (2026-10-23):** the sentence "ISLA-DTT gains from ISLA's scales (with the standardized input) about as much as ISLA does"
is no longer supported as worded: DTT-NOSC carries the same raw-speed confound that §2c resolved for ISLA, and by analogy with
§2c DTT-NOSC's step is most likely the standardization of the global input, not the scales; no ISLA-DTT standardization arm was
run. The bundle-cost reasoning stands unchanged: DTT-NOSC is 14 % worse than DTT-CONT on pressure, with the raw speed in both.

## 2c. NOSC-STD (PREREG §10; read 2026-10-23 00:18 EDT; fix-trained; B and D_f as §2; pooled σ 0.063 / 0.045 / 0.059, 8 dof)

NOSC-STD (no ISLA scales; the inlet speed standardized, (U_inf_mag − 62.116) / 17.914, as the fair-drive baselines read it):
p 0.01228 (0.01326, 0.01129), v 0.00651 (0.00669, 0.00633), nut 0.01625 (0.01708, 0.01543); ÷ B = **1.003 / 1.029 / 1.043**,
no measurable effect on any field; φ = 0.01 / 0.10 / 0.12.

**Registered reading (§10): on all three fields φ ≤ 0.3 and no move by R1, so input standardization carries the step; the
scales do not.** With the speed standardized, a 0.9.0 model without ISLA's scales (no Reynolds number, nut in the dataset's
units, a bf16 head) is as accurate as 0.9.0. The whole AirfRANS gain from E2 CONT to 0.9.0 (0.778 / 0.739 / 0.709) came from
replacing an unstandardized dimensional input (the raw speed, 31-94 m/s, fed into every token's seed) by an O(1) one. RAWU (raw
speed with the scales on) and NUT0 (nut units off) agree: the global scalar's scaling is everything, nut's units nothing. STDU
(scales on with the standardized speed; running) is the last check of the same reading.

**Principled or side effect.** The gain is real and for a sound reason (standardized inputs), but it is a side effect of the
ISLA-SCALES change, not of its physics: on AirfRANS ν is constant and L nearly so, so log10(U L / ν) is a monotone
reparametrization of U, and standardizing U alone gives the same accuracy. The A2-era and 0.8.0 AirfRANS configs carried an
input defect (the raw speed) that the baselines' fair-drive configs had already fixed (DRIVE-FIX). The scales remain the
principled way to make this automatic (a declared velocity source yields an O(1) Reynolds number by construction), and they
matter where the physics varies more than one scalar (several ν or L). **Proposed model change (not made):** ISLA should never
feed an unscaled dimensional rank-0 global into its seeds: a rank-0 global without `units` should be standardized by fitted
constants (as `isla_constants.reynolds` is), or refused, so that no config can reintroduce the defect.

## 2d. STDU (PREREG §10, conditional on RAWU; read 2026-10-23 06:49 EDT; fix-trained; B and D_f as §2; pooled σ 0.076 / 0.048 / 0.056, 9 dof)

STDU (ISLA's scales kept: U_inf as the velocity source, nut in U L, the fp32 head; the standardized inlet speed
(U_inf_mag − 62.116) / 17.914 in place of the standardized log Reynolds number): p 0.01235 (0.01355, 0.01115), v 0.00643
(0.00672, 0.00614), nut 0.01548 (0.01566, 0.01530); ÷ B = **1.008 / 1.015 / 0.995**, φ 0.03 / 0.05 / −0.02, **no measurable
effect** on any field (posed 1.008 / 1.015, the same).

**Reading.** With the scales on, a standardized speed serves as well as the standardized log Reynolds number, and RAWU (the same
with the raw speed) carries the whole step. So the form of the global scalar (log Re or speed) does not matter on AirfRANS; its
standardization does. This closes the AirfRANS attribution: **the CONT → 0.9.0 gain is the standardization of the one global
scalar** (NOSC-STD and STDU both at 0.9.0's level; NOSC and RAWU, the raw speed with and without the scales, both at CONT's).
Label: fix-trained (T on AirfRANS ≈ 1, scratch/isla-ref090-2026-10-20/r090_final_tables.md, section REF-090-FIX AirfRANS).

**Float64 models of the probes (supervisor's check, 2026-10-23).** Every float64 model I used scores as eager float32 does:
case-mean metric float64 ÷ eager float32 = 1.0002 (att2c_lug_cont_seed42, f64_probe_g2.py, 100 cases), 1.0003 and 1.0000
(att_lug_enc_seed42 / 43, f64_probe.py, 60 cases). So the float64 floors behind deviations 2 and 4 are valid (unlike the
code_a2r case REF-090-CLOSE found).

Erratum (2026-10-23 09:51 EDT): §2d's "NOSC and RAWU, the raw speed with and without the scales" should read "without and with"
(NOSC has no scales, RAWU has them). Nothing else changes.

## 3c. The ISLA-DTT mirrors DTT-SDF and DTT-QEMB (PREREG §11 as amended by §12, md5 dcd52514; read 2026-10-23 23:37 EDT; fix-trained)

Read once, after all four lanes were scored together (job 362799; every config row "epoch": 500, 200 cases). B = r090fix_af_dtt;
D_DTT = log(DTT-CONT ÷ B) = +0.103 / +0.156 / +0.026; C_f = log(DTT-NOSC ÷ DTT-CONT) = +0.131 / +0.047 / +0.121 (p / v / nut).
Pooled σ over the ISLA-DTT arms, now with the mirrors: **0.042 / 0.029 / 0.064 (6 dof)**, up from 0.026 / 0.022 / 0.063 at §3b
(DTT-SDF's two pressure seeds are 12 % apart). So SE = 0.042 / 0.029 / 0.064 and SE(C_f) the same. Tables: relaunch_native.md,
relaunch_posed.md (reduce2.py, section af / dtt), s12_native.md, s12_posed.md (read_s12.py).

| arm | field | seed 42 (÷ B, ÷ DTT-CONT) | seed 43 (÷ B, ÷ DTT-CONT) | ÷ B | ÷ DTT-CONT | Δ ± SE | R1 | φ | R2 | ψ (SE ÷ C_f) | §12(b) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DTT-SDF | p | 0.01394 (1.094, 1.025) | 0.01242 (0.989, 0.858) | 1.040 | 0.938 | +0.039 ± 0.042 | no | +0.38 | no measurable effect | −0.30 (0.32) | does not carry it |
| DTT-SDF | v | 0.00755 (1.062, 0.922) | 0.00733 (1.025, 0.864) | 1.043 | 0.892 | +0.043 ± 0.029 | no | +0.27 | no measurable effect | – (C_f < ln 1.05) | not read |
| DTT-SDF | nut | 0.02118 (1.064, 0.959) | 0.02209 (0.957, 1.009) | 1.009 | 0.984 | +0.009 ± 0.064 | no | +0.36 | no measurable effect | −0.07 (0.53) | does not carry it |
| DTT-QEMB | p | 0.01214 (0.953, 0.892) | 0.01276 (1.015, 0.882) | 0.983 | 0.887 | −0.017 ± 0.042 | no | −0.16 | no measurable effect | +0.13 (0.32) | does not carry it |
| DTT-QEMB | v | 0.00682 (0.960, 0.834) | 0.00735 (1.028, 0.867) | 0.994 | 0.850 | −0.006 ± 0.029 | no | −0.04 | no measurable effect | – (C_f < ln 1.05) | not read |
| DTT-QEMB | nut | 0.02163 (1.087, 0.980) | 0.01912 (0.828, 0.873) | 0.949 | 0.925 | −0.053 ± 0.064 | no | −2.07 | no measurable effect | +0.44 (0.53) | does not carry it |

(÷ B and ÷ DTT-CONT of the seed means are exp(Δ), ratios of geometric seed means; per-seed ratios are against the same seed.)
Posed reads the same to the third digit (s12_posed.md); posed rotates and translates the input, but ISLA-DTT is invariant to it,
so posed is not an independent number.

**Registered reading.** (a) R1 / R2: neither mirror moves any field (|Δ| < 2 SE on all six); both "no measurable effect".
(b) §12: on pressure and nut, neither DTT-SDF nor DTT-QEMB carries ISLA-DTT's bundle cost ("does not carry it" on all four);
on velocity no cost is located (C_f below the 5 % floor). **The cost (DTT-NOSC 14 % worse than DTT-CONT on pressure) is not in
either piece alone, as far as two seeds can show.** Labels: fix-trained.

**Beside the rule (not a verdict).**
- Power: at this σ a mirror moves pressure only if |Δ| ≥ 0.084, i.e. ψ ≥ 0.64. The 2 SE upper ends of ψ on pressure are 0.34
  (DTT-SDF) and 0.77 (DTT-QEMB); on nut 0.98 and 1.49. So putting the stored SDF back is ruled out as carrying even half of the
  pressure cost; the old prediction-point embedding is not ruled out as carrying part of it; nut constrains neither.
- Directions: DTT-SDF's pressure Δ points the wrong way (stored SDF back, ISLA-DTT no better, ψ −0.30); DTT-QEMB's point
  toward the cost on p and nut (ψ +0.13 / +0.44), both within noise.
- §12(c) caveat: C_f is measured with the raw speed on both sides; the mirrors run with 0.9.0's scales and standardized
  input. A cost that exists only with the raw speed (an interaction with the input defect of §2c) would read as ψ ≈ 0 here.
  Untested: no ISLA-DTT standardization arm (NOSC-STD's mirror) was run.
- What remains in the bundle for ISLA-DTT: the gauge L, the upstream merge and drift, the interaction above, or SDF and QEMB
  together (no DTT-ALL arm). Not attributed.
- Peter's boundary-only ruling: DTT-SDF would not have brought the stored field back in any case; it shows no measurable loss
  from removing it.
