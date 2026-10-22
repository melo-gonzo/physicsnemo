# AirfRANS ISLA ladder (archaeology, read-only, 2026-10-21)

Common setup unless flagged: airfrans_uvolume_2d (uniform-interior files), full_train / full_test, 200 test cases, 100,000 points,
500 epochs, seeds 42/43, width 512, 512 slices, lr 3.75e-4, bf16 training (lane default), scored float32 on the final checkpoint,
native per-case relative L2. Per-seed values in brackets (42, 43).

## Interacting ISLA

| rung | lanes | code / commit | recipe | p | v | nut | source |
|---|---|---|---|---|---|---|---|
| AF-UNIFORM raw reads (pre-A2) | afu_r_isla | code_v20 | recipe_v8 | 0.02076 (0.01968, 0.02185) | 0.01198 (0.01161, 0.01236) | 0.02424 (0.02361, 0.02486) | results/char/airfrans_uniform.json; airfrans_a2.json "raw_read_isla" |
| A2 replication | afu_bn2_isla | code_v20a2 | (H-RESID) | 0.01791 (0.01820, 0.01761) | 0.00946 (0.00949, 0.00943) | 0.02127 (0.02118, 0.02136) | airfrans_a2.json "replication" |
| **A2 reference** | afu_a2s_isla | code_a2s = ebf841578 | recipe_v11 | 0.01799 (0.01854, 0.01745) | 0.00946 (0.00970, 0.00921) | 0.02230 (0.02325, 0.02134) | results/char/airfrans_a2.json |
| GEO1 exact geometry | afu_geo1_isla | code_geo1 (psharpe/isla-geo-exact) | recipe_v11 | ~0.01564 (~0.01517, ~0.01612) | ~0.00887 (~0.00867, ~0.00907) | ~0.02087 (~0.02046, ~0.02126) | derived from ratios in notebook #sec-nb-geo1-airfrans (0.869/0.938/0.936; per seed .818/.924, .894/.985, .880/.996) × A2; no absolute file found locally |
| E1 HEADNORM CONTROL | hnaf_ctrl_isla | code_hnc = ea397e04f (A2, 0.4.0) | recipe_hn | 0.01828 (0.01778, 0.01877) | 0.00950 (0.00939, 0.00962) | 0.02143 (0.02097, 0.02190) | scratch/isla-headnorm-2026-10-18/RESULTS_AF_2026-10-19.txt |
| E1 HEADNORM | hnaf_hn_isla | code_hn = 8f8c67fde (0.5.0) | recipe_hn | 0.01808 (0.01848, 0.01767) | 0.00958 (0.00960, 0.00956) | 0.02039 (0.02039, 0.02039) | same |
| E3b arm A | anc_isla (A) | code_anc (off ea397e04f) | recipe_anc | 0.01835 (0.01808, 0.01861) | 0.00971 (0.00940, 0.01002) | 0.02257 (0.02112, 0.02402) | scratch/isla-anchor-2026-10-18/RESULTS.md |
| E2 GEO-CONT CONTROL | gcont_ctl_af_isla | code_gctl = c9fec58bf (geo-exact b40537f9b + isla ea397e04f merged) | recipe_gcont | 0.01698 (0.01671, 0.01726) | 0.00928 (0.00989, 0.00867) | 0.02164 (0.02271, 0.02056) | notebook #sec-nb-geo-cont-verdict (l.38521-38527) |
| E2 GEO-CONT CONT | gcont_cont_af_isla | code_gcont = 84aef1db4 (ckpt 0.6.0) | recipe_gcont | 0.01574 (0.01584, 0.01565) | 0.00864 (0.00850, 0.00877) | 0.02056 (0.02036, 0.02076) | same |
| **REF-090 (0.9.0)** | r090_af_isla | code_r090 = git archive 92d9d7800 (+td 0.15.2) | recipe_r090 (92d9d7800 + cluster files) | 0.01186 (0.01251, 0.01121) | 0.00644 (0.00654, 0.00633) | 0.01581 (0.01623, 0.01539) | scratch/isla-ref090-2026-10-20/RESULTS_partial.md (scorer 308199); r090_final.json |

## ISLA-DTT (`query_independent=true`)

| rung | lanes | code | p | v | nut | source / flag |
|---|---|---|---|---|---|---|
| AF-UNIFORM DTT | afu_dtt | code_dtt16 (td14), recipe_v8; **384 wide, 256 slices, lr 5e-4, forest read kernel_read=mean θ=1** | 0.01996 (0.01855, 0.02137) | 0.01190 (0.01179, 0.01201) | 0.02706 (0.02778, 0.02633) | airfrans_uniform.json "controls". NOT like-for-like (size, lr, read). |
| E3b arm D | anc_dtt | code_anc (A2 ea397e04f lineage, nearest-token geometry, stored SDF, raw U_inf_mag) | 0.01361 (0.01356, 0.01367) | 0.00861 (0.00865, 0.00857) | 0.02231 (0.02137, 0.02324) | isla-anchor RESULTS.md; D/A 0.742/0.887/0.988 |
| 0.9.0 DTT | r090_af_dtt | code_r090 (92d9d7800) | 0.01379 (0.01408, 0.01350) | 0.00738 (0.00740, 0.00736) | 0.02206 (0.02195, 0.02218) | RESULTS_partial.md; DTT/0.9.0 1.163/1.147/1.396 |

**No ISLA-DTT measurement exists on GEO1, GEO-CONT CONTROL or CONT code** (GEO1 rows 10-13, GEO-CONT lanes gcont_{ctl,cont}_af_isla, HEADNORM: interacting only). The AF-UNIFORM DTT is on a different config, so the only like-for-like DTT pair is E3b-D → 0.9.0 DTT.

## Model inputs per code version (isla_volume_airfrans.yaml)

| code | query geometry | stored SDF / sdf_normals read | global input | output normalization |
|---|---|---|---|---|
| code_a2s ebf841578 (A2, E1, E3b) | nearest boundary token | yes (query inputs) | U_inf_dir + raw U_inf_mag (m/s) | dataset stats (pre-scales) |
| code_geo1 | exact closest point on full boundary | yes | same | same |
| code_gctl / code_gcont 84aef1db4 | exact; CONT: [s, asinh(s/1e-6)], ν = sign(d)·u (CONT also on stored-sdf channel) | yes | U_inf_dir + raw U_inf_mag | same |
| 92d9d7800 (0.9.0) | exact + CONT, full-mesh gauge, BCs by name | **no** ("stored sdf and sdf_normals ... are not read") | U_inf as farfield Dirichlet BC; ISLA scales: L = RMS pairwise airfoil distance, U = abs(U_inf), nu = 1.56e-5 via SetGlobalField; **standardized log10 Re** replaces raw speed; isla_constants injected (Re mean 6.205, std 0.1347) | v/abs(U_inf), Cp dimensionless; nut standardized by output_stats; head + redimensionalization in fp32 |

Scoring is like-for-like across A2, GEO1, E1, E3b, E2, REF-090: the same 200 full_test cases at 100k points, float32, 500 epochs, 2 seeds,
and REF-090 checked the same case order. Flags: (i) the AF-UNIFORM DTT config, above; (ii) GEO1 absolute numbers are derived from ratios here, not read
from a file; (iii) the pre-A2 raw-read arm uses code_v20/recipe_v8 (an older snapshot, same files and scorer); (iv) controls drift: E1 CONTROL, E3b A and
A2 are all A2-lineage, yet span 0.0180-0.0184 p and 0.0213-0.0226 nut (seed noise about 3-8 %, partners.json σ 0.036-0.061).
GEO-CONT CONTROL (exact geometry, 0.01698) is worse than GEO1 (exact, ~0.01564) on a different snapshot (the geo-exact branch
merged with ea397e04f). So "exact geometry" carries ±8 % across snapshots/seeds.

## Attribution status

| step | single change? | measured ratio (p / v / nut) |
|---|---|---|
| raw reads → A2 normalized reads | yes (A2 / AF-UNIFORM ISLA) | 0.867 / 0.789 / 0.920 (cross-snapshot v20→a2s; replication bn2 agrees) |
| A2 → GEO1 exact geometry | yes, same recipe, controlled pair | 0.869 / 0.938 / 0.936 |
| A2 → HEADNORM | yes (E1) | 0.989 / 1.008 / 0.952 (head-frame normalization; check whether it is in 0.9.0, not verified) |
| exact → CONT encoding | yes (E2) | 0.927 / 0.931 / 0.950 |
| interacting → DTT, A2 lineage | yes (E3b D/A) | 0.742 / 0.887 / 0.988 |
| interacting → DTT, 0.9.0 | yes (REF-090 DTT/0.9.0) | 1.163 / 1.147 / 1.396 |
| **E2 CONT → 0.9.0 interacting** | **NO, bundled, unattributed** | 0.754 / 0.745 / 0.769 |
| A2 → 0.9.0 interacting | bundle (R1, read as standing move per PREREG §2) | 0.659 / 0.681 / 0.709 |
| E3b D → 0.9.0 DTT | bundle, unattributed | 1.013 / 0.857 / 0.989 |

The CONT → 0.9.0 bundle (PREREG l.62-65) contains: removing the stored-SDF query input; ISLA's own scales (L, U, nu; log-Re global
in place of raw U_inf_mag; dimensionless outputs; fp32 head); BCs by name (U_inf enters as a farfield Dirichlet value instead of a global
leaf); the full-mesh gauge; upstream/main merge; code + recipe drift since 84aef1db4. No AirfRANS accuracy arm isolates any of these:
the bc-names verify (REPORT.md l.49) is smoke/throughput only, and the notebook/scratch hold no "scales" or "Option A" AirfRANS accuracy run.
The Tensile2d note in RESULTS_partial suspects SDF removal and per-condition layers as the cause of a regression there, also untested. On
AirfRANS the ~25 % gain is most plausibly the scales/log-Re/nondimensional-output change, since geometry was already exact+CONT.
That is a hypothesis.

Why DTT did not move: its 0.9.0 change against E3b-D is ~1.0 on p and nut and 0.86 on v. The A2-lineage D already beat interacting A by 26 % on p,
which suggests the query-independent read was insensitive to the nearest-token geometry pathology that GEO1/CONT fixed for interacting ISLA
(A's error swings 26-63× with companion points; D's does not). So the geometry gains (GEO1, CONT, ~0.81 p combined) were mostly gains that
interacting ISLA needed and DTT did not. Whatever the bundle did for interacting ISLA (scales etc.) helped DTT only on velocity. This is
inference: no DTT arm exists on geo1/gctl/gcont, so the split between "geometry fixes" and "0.9.0 bundle" for DTT cannot be measured.
Also note D fails float64 query independence at 2e-6 (anchor RESULTS), which does not affect accuracy.

## Needed to attribute
Single-change AirfRANS arms on 92d9d7800: (a) restore the stored-SDF query channel; (b) scales off (raw U_inf_mag global, old output norm);
(c) DTT on code_gcont (84aef1db4) to locate the DTT step. Expect ±5 % seed noise, so effects under ~7 % need more than two seeds.
