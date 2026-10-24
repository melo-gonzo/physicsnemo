# Lug von Mises ladder: A2 → CONTROL → CONT → 0.9.0 (archaeology, 2026-10-21)

Setup shared by every rung below unless noted: ginot_lug_volume with train / test, 600 test cases, 20k training cells, scored at 200,000,
float32 scorer, final checkpoint (epoch 500), seeds 42 / 43, isla_lug with hidden 512 / 512 slices, lr 3.75e-4, **bf16 training**.
Metric: relative L2 of von_mises.

## Ladder (absolute; seed mean (s42, s43))
| rung | runs | code / recipe snapshot | native | posed | posed÷native | seed gap | source |
|---|---|---|---|---|---|---|---|
| CHAR (old) | cl_r_lug_isla | code_v20-era, raw slice reads | 0.0040 (0.0035, 0.0046) | 0.0065 (0.0070, 0.0060) | 2.01 / 1.30 | 29 % | lug_a2_table.md; results/char/lug.json |
| A2 | a2r_lug_isla_seed42/43 | code_a2r / recipe_a2r = `git archive 7e167820a` + cluster lug yamls from recipe_v8 | 0.001726 (0.001602, 0.001850) | 0.005974 (0.006633, 0.005315) | 4.14 / 2.87 | 15.5 % | scratch/isla-a2-rerun-2026-10-17/lug_a2_table.md; results/char/lug_a2.json |
| GEO-EXACT / GEO2 | – | – | **not scored on the lug** (GEO1 was AirfRANS only, #sec-nb-geo1-airfrans) | | | | |
| E2 CONTROL | gcont_ctl_lug_isla_seed42/43 (275258 / 275410) | code_gctl, HEAD c9fec58bf (geo-exact b40537f9b + isla ea397e04f); recipe_gcont | 0.001546 (0.001457, 0.001635) | 0.001690 (0.001568, 0.001813) | 1.08 / 1.11 | 12.2 % | isla-geo-cont-2026-10-18/lug_table_2026-10-19.md |
| E2 CONT | gcont_cont_lug_isla_seed42/43 (275261 / 275413) | code_gcont, HEAD 84aef1db4 (ckpt 0.6.0); recipe_gcont | 0.001744 (0.001717, 0.001771) | 0.001840 (0.001804, 0.001875) | 1.05 / 1.06 | 3.1 % | same |
| REF-090 ISLA | r090_lug_isla_seed42/43 (298080 / 298081; scorer 308284) | code_r090 / recipe_r090 = `git archive 92d9d7800` + lug yamls from recipe_az_c | 0.001904 (0.001816, 0.001993) | 0.002139 (0.002073, 0.002206) | 1.14 / 1.11 | 9.7 % | isla-ref090-2026-10-20/RESULTS_partial.md L97-114; r090_final_tables.md L50-54 |
| REF-090 ISLA-DTT | r090_lug_dtt_seed42/43 | same, `++model.query_independent=true` | **not scored** (at epoch 225 of 500 when RESULTS_partial was written) | | | | |

## Inputs per rung (verified on LHR: `$T/recipe_*/datasets/ginot_lug_volume.yaml` and `conf/model/isla_lug.yaml`)
| rung | dataset yaml (md5) | FlipBoundaryOrientation | sign of built-in d inside | stored SDF via query_schema | boundary conditions | geometry / encoding |
|---|---|---|---|---|---|---|
| A2 | 01e0e69b | no | negative (stored triangles wound outward) | yes (sdf rank 0, sdf_normals rank 1) | one `surface` | nearest boundary token (medial / structured-mesh ties), sign·log |
| CONTROL | 56d73aaa | **yes** (first transform; after FixedRandomPose in posed) | positive | yes | one `surface` | exact geometry, full-res boundary_meshes, in-model draw, sign·log; ckpt 0.5.0 |
| CONT | 56d73aaa | **yes** | positive | yes (stored sdf also gets asinh) | one `surface` | as CONTROL + [s, asinh(s/1e-6)], direction sign(d)·u; ckpt 0.6.0 |
| 0.9.0 | **01e0e69b (= A2's, no flip)** | **no** | **negative**: the r090 isla_lug.yaml says the triangles stay as stored, so d has the stored sdf's sign (≤ 0 inside) | **no** (the comment says "No query_schema") | four: fixed / hole / loaded (bc_pressure) / free via SplitBoundaryByLabels | CONT encoding + per-condition input layers, whole-model compile=true |

The winding story: 84aef1db4 only changed the comment in isla_lug.yaml ("pair with FlipBoundaryOrientation"). The flip lived only in
recipe_gcont's lug yamls (lug_flip.py, build_snapshots.sh). recipe_r090 copied the lug yamls from recipe_az_c without the flip. 0.9.0's config states this on
purpose (it runs with the negative side inside). **So A2 and 0.9.0 saw outward winding (d<0 inside), while CONTROL and CONT saw it flipped (d>0).** Under
the CONT encoding, s→−s and ν→−ν is a global sign change. Since asinh is odd, a network could in principle absorb it. But ISLA's odd invariants
(n·g, ρ·n, …) change sign, so this is not a guaranteed no-op. 0.9.0 is therefore not like-for-like with CONT on orientation.
BC-NAMES (isla-bc-names-verify-2026-10-19/REPORT.md L45, L66) only checked the build, speed and memory of the four-condition lug (fixed 280 / hole 294 /
loaded 294 / free 5,792 = 6,660 cells). It produced no accuracy number.

## Ratios (native; seed-matched s42, s43 in parentheses)
| step | what changed | ratio of means | seed-matched | attributed? |
|---|---|---|---|---|
| CONTROL ÷ A2 | exact geometry + single-input design + flip + code drift + ckpt 0.5.0 (bundled) | 0.896 | (0.909, 0.884) | bundled, not separated |
| CONT ÷ CONTROL | continuous encoding only (registered A/B, same recipe) | 1.128 | (1.178, 1.083) | **yes**, the only clean step; flagged (both CONT seeds above CONTROL's range) |
| **CONT ÷ A2** | net of the two above | **1.010** | (1.072, 0.957) | – |
| **0.9.0 ÷ CONT** | no stored SDF query, four BC inputs, **no flip**, whole-model compile, code 84aef1db4→92d9d7800 | **1.092** | (1.058, 1.125) | **not attributed** (bundled) |
| 0.9.0 ÷ A2 | everything | 1.104 | (1.134, 1.077) | – |
| 0.9.0 ÷ CONTROL | | 1.232 | (1.246, 1.219) | – |

Posed: CONT÷A2 0.308; 0.9.0÷CONT 1.163; 0.9.0÷A2 0.358. Area = native everywhere.

## Answer: the CONT step does not fully explain 0.9.0's 1.104
- In absolute terms CONT is at A2's level (1.01×). The encoding's +12.8 % cost mostly cancels CONTROL's −10.4 % gain. Of the 1.104, **≈1.01 is
  reached at CONT and ≈1.09 comes after CONT**, in unattributed 0.9.0 changes. RESULTS_partial.md L109 says the 1.10 is "in line with the E2
  GEO-CONT flag". That is misleading as attribution: the GEO-CONT flag is measured against CONTROL, and CONTROL is 0.90× A2.
- Candidates for the 0.9.0÷CONT 1.09, none tested on the lug: (a) dropping the stored sdf / sdf_normals query input; (b) splitting into four conditions
  plus bc_pressure; (c) orientation reverted to unflipped; (d) compile=true and code drift. The Tensile2d note lists (a) and (b) for its 1.04 / 1.135.
- Noise: the single-seed σ(log e) for lug VM is 0.102 (GEO-CONT PREREG L134-140, from A2). For a two-seed mean the ratio SD is ≈0.10, i.e. ±10 %. So
  0.9.0÷CONT 1.09 (≈0.9σ), CONT÷A2 1.01 and CONTROL÷A2 0.90 (≈1σ) are each within about 1σ. Only CONT÷CONTROL shows both seeds separated. The 0.9.0
  seed ranges overlap A2's and CONT's: 0.9.0 s42 0.001816 < A2 s43 0.001850, and 0.9.0 s42 > CONT s43 0.001771, so 0.9.0 lies above CONT on both seeds.
  Within-arm gaps run 3 % to 15.5 %, so a 10 % step is not resolvable with 2 seeds. Treat the ladder as directional only.
- Posed: the large posed gain (A2 4.1 / 2.9 → about 1.1) comes from the CONTROL step (exact geometry). The remaining 1.05 to 1.14 matches the medial ties
  (DESIGN_NOTE_medial_ties.md: 0.80 % of nodes reverse direction, in 331 of 600 cases; not implemented). solids_ties_2026-10-18.json and lug_ties_2026-10-04.json
  document the A2-era nearest-token ties. lug_prec_2026-10-05.json covers the CHAR-era bf16 vs fp32 scorer sensitivity. None of these changes the
  0.9.0 native attribution.

## Not like-for-like flags
- CHAR → A2: different code (raw vs normalized slice reads) and hidden size (CHAR ties diagnostics used hidden 768). Context only.
- A2 → CONTROL: bundles 4+ changes, including the flip.
- CONT → 0.9.0: bundles the input changes **and the orientation reversal**. It also changes the recipe, data yaml and compile.
- ISLA-DTT lug: no score yet, so there is no interacting ISLA×DTT rung for the lug.
