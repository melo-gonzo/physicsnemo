# Tensile2d ISLA ladder (archaeology, 2026-10-21)

Common setup on every rung below: plaid_tensile2d_2d, `train_split=train_400 val_split=val_100`, scored on 100 val cases,
trained at 20k cells, scored at 200,000 cells, epoch 500, float32, seeds 42/43, isla_tensile2d hidden 512 / n_slices 512
(GC PREREG.md:59; REF-090 RESULTS_partial.md:3). Dataset yaml (`plaid_tensile2d_2d{,_area,_posed}.yaml`) is a cluster-only
file, copied unmodified from LHR recipe_v8 into recipe_a2r, recipe_geo2 / recipe_gcont, recipe_bcn and recipe_r090
(A2R PREREG.md:52-56; REF-090 PREREG.md:24). Its chain includes RestrictToPlane and SplitGlobalVector (per the model yaml
comments); I did not read the yaml itself on LHR. No ISLA-DTT run on Tensile2d was found.
GEO-EXACT / GEO2: not scored on Tensile2d. The design only discusses a 2-D brute-force segment query
(isla-geo-exact-design REPORT.md:207), and geo2-az only copies the dataset file. BC-NAMES verify (isla-bc-names-verify-2026-10-19)
ran tests and 20-step smokes (lug, AirfRANS, DrivAerML, AeroZoo). **No Tensile2d run and no accuracy numbers.**

## Rungs (native errors, seed mean (s42, s43))

| rung | runs | code (commit) / recipe | stress | displacement | q |
|---|---|---|---|---|---|
| old CHAR "ISLA 512 with 512 slices" | CHAR record | pre-A2 | 0.0138 (0.0134, 0.0141) | 0.0133 (0.0147, 0.0118) | 0.328 (0.348, 0.308) |
| A2 | a2r_t2dz_isla_seed42/43 (LHR job 251668) | code_a2r = 7e167820a, recipe_a2r md5 5276dff9 | 0.00781 (0.00702, 0.00859) | 0.00688 (0.00679, 0.00696) | 0.415 (0.369, 0.460) |
| E2 CONTROL | gcont_ctl_t2dz_isla_seed42/43 (275257 / 275409) | code_gctl = c9fec58bf (geo-exact b40537f9b + isla ea397e04f), recipe_gcont md5 937b1231 | 0.0084 (0.0083, 0.0086) | 0.0067 (0.0072, 0.0062) | 0.361 (0.177, 0.544) |
| E2 CONT (ckpt 0.6.0) | gcont_cont_t2dz_isla_seed42/43 (275260 / …) | code_gcont = 84aef1db4, recipe_gcont | 0.0084 (0.0082, 0.0086) | 0.0064 (0.0066, 0.0062) | 0.368 (0.365, 0.370) |
| REF-090 (0.9.0) | r090_t2dz_isla_seed42/43 (scorer 299734) | code_r090 = 92d9d7800, recipe_r090 md5 bb4ee858 | 0.00886 (0.00900, 0.00872) | 0.00716 (0.00725, 0.00708) | 0.286 (0.335, 0.238) |

Posed: A2 0.0085 / 0.0075 / 0.416 (posed÷native 1.06-1.12 stress/disp); CONTROL and CONT posed = native (0.9999-1.0002);
REF-090 posed = native exactly. Area = native everywhere. Old CHAR posed 0.0139 / 0.0132 / 0.328.
Sources: scratch/isla-a2-rerun-2026-10-17/tensile2d_a2_table.md (+ evidence/REPORT.md:15-17 for 5-digit A2 per seed);
scratch/isla-geo-cont-2026-10-18/t2d_table.md (4-digit only; t2d_scores.json has more); scratch/isla-ref090-2026-10-20/RESULTS_partial.md:3-17.

## What changed between rungs (Tensile2d-relevant)

| step | change | like-for-like? |
|---|---|---|
| old → A2 | full A2 code/recipe refresh (many changes); stress becomes one symmetric tensor, etc. | No (different code era); R-A2: 0.568 / 0.519 / 1.265 |
| A2 → CONTROL | geo-exact (exact built-in distance from boundary meshes) + psharpe/isla merge; model.py = code_geo2's. **Still reads stored sdf / sdf_normals via query_schema** (c9fec58bf isla_tensile2d.yaml:60-62). Boundary: one `outer` mesh, BCs as per-cell 0/1 labels bc_loaded/bc_fixed/bc_pinned + bc_pressure (boundary_schema, yaml:55-59), same as A2 | Mostly (same splits/scoring); code differs. CONTROL÷A2: 1.077 / 0.977 / 0.870 (REPORT.md:104) |
| CONTROL → CONT | only the continuous encoding: [s, asinh(s/1e-6)] for built-in distance **and the stored-sdf channel**, sign(d)·u direction (PREREG.md:39); stored sdf still read (84aef1db4 yaml:63-64) | Yes, registered pair. CONT÷CONTROL 0.999 / **0.956** (per seed 0.983, 0.929) / 1.019; no field flagged |
| CONT → 0.9.0 | (a) BC-NAMES 3c4a44105 + 434449af0 + bb07cf413: SplitBoundaryByLabels splits `outer` into **four** named conditions loaded (bc_pressure rank 0) / fixed / pinned / free (rest), each with its own input layer and offset; prediction points get their own input layer replacing shared embedding/type offset/query MLP; ckpt 0.7.0. (b) f259a120e: **stored sdf / sdf_normals removed from query_schema** (distance only built-in). (c) merge 0e77a710e (ckpt 0.8.0) and whatever else to 92d9d7800 (0.9.0), incl. train.py next-batch lookahead | Bundled; not separated |

Boundary count note: REF-090 says "Tensile2d has three conditions" (RESULTS_partial.md:17); the 0.9.0 yaml declares four
(three labelled + `free`) (isla_tensile2d.yaml:72-77 at HEAD and 92d9d7800). Before BC-NAMES there was one boundary mesh.

## Settling CONTROL / CONT vs A2 vs 0.9.0 (displacement, native)

| ratio | seed mean | per-seed |
|---|---|---|
| 0.9.0 ÷ A2 | 1.041 | both 0.9.0 seeds (0.00725, 0.00708) above both A2 seeds (0.00679, 0.00696) |
| CONTROL ÷ A2 | 0.977 | CONTROL seeds 0.0072 / 0.0062 straddle A2 |
| CONT ÷ A2 | 0.934 | 0.0066 / 0.0062, both below A2 |
| 0.9.0 ÷ CONT | ≈1.115 (0.00716/0.0064, 4-digit) | both 0.9.0 seeds above both CONT seeds |
| 0.9.0 ÷ CONTROL | ≈1.07 | s42 0.00725 vs CONTROL s42 0.0072 ≈ equal; CONTROL range [0.0062, 0.0072] |

Two-seed gaps (max/min−1): A2 2.5 %; CONTROL ~16 % (REPORT.md:52 "CTL 0.142" log gap); CONT ~6 % (0.056);
0.9.0 2.4 %; old CHAR 24 % (0.214). Single-seed σ assumed in GC PREREG = 0.017 (from the A2 pair alone;
REPORT.md:52 calls it "contested", 3 of 4 pairs ≥ 5.6 %). Reading: against CONT (0.8.0's predecessor branch arm)
0.9.0 is ~+11 %, beyond both CONT seeds; but CONTROL's own pair spans 16 %, so a regression of this size sits inside
the observed seed spread of sibling arms. The 1.041 vs A2 is consistent in sign per seed but within ~2 multi-pair σ
(multi σ 0.094 for T2D disp, REPORT.md:68). Not proven to be a step; plausibly partly noise.

## Stored SDF on Tensile2d
- A2: read (7e167820a yaml:60-62 query_schema sdf, sdf_normals).
- CONTROL: read (c9fec58bf yaml:60-62), alongside the new built-in exact distance.
- CONT: read, with the asinh encoding also applied to it (PREREG.md:39,179; probe PREREG.md:224 stored pair [-5.06,0]/[0,13.74]).
- 0.9.0: not read (f259a120e "ISLA's AirfRANS, Tensile2d and lug configs read no stored SDF").

## Attribution in the record
- Attributed: old→A2 (R-A2); CONTROL→CONT (registered, displacement 0.956, adopted per #sec-nb-geo-cont-verdict, notebook line 38505).
- Recorded but not a registered comparison: CONTROL÷A2, CONT÷A2 (partners.json, REPORT.md:104).
- **Not attributed:** CONT→0.9.0. REF-090 lists BC-NAMES per-condition input layers and removal of stored-SDF query input as
  candidate causes only (RESULTS_partial.md:16-17); no Tensile2d BC-NAMES or no-SDF ablation was run.

## Residual risks
- CONTROL/CONT numbers are 4-digit from t2d_table.md; per-seed 5-digit values are in t2d_scores.json (not read).
- Dataset yaml transforms not read directly (cluster-only); "three conditions" vs four declared is a wording discrepancy to confirm.
- Commits between 0e77a710e and 92d9d7800 not enumerated beyond the yaml log.
