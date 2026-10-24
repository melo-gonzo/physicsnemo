# PROBES-090 results: QUERY-POP-090 and REMESH-090 (AirfRANS, DrivAerML Volume, AeroZoo ISLA; AeroZoo ISLA-DTT pending)

Written 2026-10-21 02:45 EDT. Registration: PREREG.md (md5 4f188984, approved 2026-10-20 23:59 EDT). Jobs: JOBS.md.
Reductions: `results/qpop/reduced.{json,md}`, `results/remesh/reduced_full.{json,md}`; labels re-tabulated by
`analysis/label_qpop.py` and `analysis/label_remesh.py` into `results/qpop/labels_af_vol.md` and `results/remesh/labels.md`.

**Provenance (applies to every AirfRANS row).** Every arm ran fully eager in float32, as registered. The AirfRANS
checkpoints (0.9.0 both forms, GEO1, A2) were trained under the compiled-path defect (`qpop/COMPILED_PATH_EVIDENCE.md`;
COMPILE-AF fix b62a93f2c). So their rows are **the eager function of weights trained under the compiled-path defect**.
The clean read is a registered follow-up on the retrained REF-090 AirfRANS checkpoints (about 5 GPU-h, amendment before
those results). DrivAerML Volume is unaffected: eager equals its scorer to 1e-5.

## 1. Validity gates (PREREG §1), read before any probe number

| run | reference | cases | max rel. diff | gate | pass |
|---|---|---|---|---|---|
| QUERY-POP r090_af_isla s42 / s43 | float64 eager | 20 | 2.3e-4 / 1.6e-4 | 1e-3 | yes |
| QUERY-POP r090_af_dtt s42 / s43 | float64 eager | 20 | 1.9e-4 / 8.9e-5 | 1e-3 | yes |
| QUERY-POP afu_geo1_isla s42 / s43 | float64 eager | 20 | 1.7e-4 / 1.4e-4 | 1e-3 | yes |
| QUERY-POP afu_a2s_isla s42 / s43 | float64 eager | 20 | **6.2e-3 / 1.3e-2** (medians 8.7e-4 / 9.8e-4) | 5e-3 | **no** |
| QUERY-POP r090_vol_isla s42 / s43 | REF-090 scorer | 48 | 9.6e-6 / 5.0e-6 | 1e-3 | yes |
| REMESH r090_af_isla s42 / s43 | float64 eager | 20 | 2.3e-4 / 1.6e-4 | 1e-3 | yes |
| REMESH r090_af_dtt s42 / s43 | float64 eager | 20 | 1.9e-4 / 8.9e-5 | 1e-3 | yes |
| REMESH r090_vol_isla s42 / s43 | REF-090 scorer | 48 | 1.9e-5 / 4.0e-6 | 1e-3 | yes |

Coverage complete (200 / 48 cases, epoch 500, no duplicates). **A2 is "incomplete as registered (validity gate failed)".**
Its probe numbers have not been read. The supervisor's condition for a beside-row was that nearest-token flips explain the
excess (masked validity near GEO1's 2e-4). They do not (`diag_a2/A2_FLIPS.md`, jobs 312123, 312205, 312264): 201-488
index flips per case sit in the far field and masking them leaves the difference unchanged. The excess comes from the
sign of A2's signed distance at the ~550 on-wall query points per case (u · n* = 0 exactly there, so its sign is a
roundoff coin flip; 238-286 flip between float32 and float64). With float64 query geometry, float32 A2 is within 2.0e-4
of float64 on all 10 worst cases. So A2 stays incomplete, with no beside-row. Its geometry input is ill-posed on the
wall, and float64 is no reference there either.

**Protocol non-degeneracy.** Every QUERY-POP mode changes the query population on every case (`reduced.json`
`nondegeneracy_ranges`): half 50,000 / 100,000 native-only tokens; same +50,000 / +100,000 tokens not in the native
set; far +100,000 / +200,000 new points; wall +100,000 / +200,000 resampled (14k-15k / 25k-27k distinct); chunks4 4
groups; sub1000 / 100 / 10 / 1 1 / 10 / 100 / 100 groups. Every REMESH variant changes the model's boundary input and
its prediction on every case (`pred_changed` = 1 on 200 / 48 of 200 / 48). Exact variants keep the geometry: measure
and L within 1e-7, points' distances within 1e-6 L. The AirfRANS ISLA `repeat` is bitwise; ISLA-DTT and Volume repeats
differ at 3e-7 to 3e-6 (GPU atomics). No row is degenerate.

**Frame check of the `same` companions (CAP-AUDIT's CenterMesh defect; supervisor, 2026-10-21 afternoon).** The
Volume `same` companions are raw stored rows mapped into the pipeline frame by a similarity fitted per case on the native
block's own raw rows (centre from the means, scale 0.2 = 1/L_ref). The probe asserted on every case that this mapping
reproduces the native points, and recorded the residual: 4.4e-7 to 1.19e-6 pipeline units over all 48 cars (extent
about 24 units, so about 5e-8 relative, float32 roundoff; `reduced.json` `geometry_ranges.frame_err`). The far, wall,
half, chunk and subset modes are built from the native draw itself. **The Volume numbers stand unchanged.** AeroZoo's
chain has no CenterMesh and no coordinate scaling (`aerozoo_uvolume.yaml`, 2026-10-19 correction), so its companions'
identity mapping is right (blind smoke frame_err 0). qpop090.py v2 (d282a4f3; v1 ffdcdb69 kept) tightens the Volume
assertion to 1e-6 × extent and adds the same per-case assertion for AeroZoo before the AeroZoo runs.

## 2. QUERY-POP-090 (r per field, seed mean [s42, s43]; full tables in `results/qpop/labels_af_vol.md`)

| arm | half | chunks4 | sub1000 | sub100 | sub10 | sub1 | same (+50%) | far | wall |
|---|---|---|---|---|---|---|---|---|---|
| AF ISLA 0.9.0, p / v / nut | 1.11 / 1.12 / 1.14 **material** | 1.33 / 1.36 / 1.40 **material** | 4.7 / 5.0 / 5.7 **severe** | 27 / 19 / 39 **severe** | 74 / 65 / 105 **severe** | 85 / 91 / 118 **severe** | 0.95 / 0.95 / 0.95 not material | 136 / 62 / 133 **severe** | 20 / 30 / 22 **severe** |
| AF ISLA-DTT 0.9.0 | roundoff | roundoff | roundoff | roundoff | roundoff | roundoff | roundoff | roundoff | roundoff |
| AF GEO1, p / v / nut | 1.14 / 1.09 / 1.15 **material** | 1.44 / 1.30 / 1.40 **material** | 5.6 / 4.6 / 5.4 **severe** | 26 / 16 / 23 **severe** | 156 / 47 / 81 **severe** | 237 / 74 / 87 **severe** | 0.93 / 0.96 / 0.95 not material | 30 / 30 / 50 **severe** | 52 / 21 / 59 **severe** |
| Volume ISLA 0.9.0, p / v / nut | 1.01 / 1.01 / 1.02 not material | 1.03 / 1.03 / 1.15 not material | 4.5 / 4.0 / 5.5 **severe** | 13 / 9.8 / 12 **severe** | 18 / 13 / 15 **severe** | 21 / 13 / 18 **severe** | 1.00 / 1.00 / 1.00 not material | 20 / 16 / 14 **severe** | 6.2 / 4.5 / 4.5 **severe** |

- ISLA-DTT is the instrument check, and it passes: in all ten modes, both seeds, every key has |r − 1| ≤ 4.3e-6, and the
  median d is 2.6e-7 to 8.2e-7.
- Both seeds agree on every label. On the large ratios the two seeds differ by up to 2× (AirfRANS far pressure 91 / 181,
  GEO1 wall nut 79 / 40, Volume wall pressure 3.8 / 8.6); on the ratios below 1.5 they agree within 6%.
- Median d (prediction moves): AirfRANS ISLA chunks4 0.010 / 0.0055 / 0.013 (p / v / nut); sub1 0.94-1.0 / 0.55-0.60 /
  1.4-2.1; far 0.84-1.6 / 0.38 / 2.0. Volume: half 0.006 / 0.009 / 0.012; far 0.72-0.85 / 0.93-0.97 / 1.1-1.2.
- `same` lowers the error: AirfRANS 0.95 and Volume 0.996. Adding points at the training law helps a little; changing
  the law hurts a lot.
- Volume chunks4 nut is 1.154 (1.12 / 1.19). That is above 1.10 on both seeds, but nut is outside the registered
  "material" fields (p or v), so it is reported here and not labelled.

## 2b. QUERY-POP-090 on AeroZoo `full`, ISLA (job 322430; read 2026-10-21 23:00 EDT)
Subsample: every 4th test case (179 of 716), F = 25. Validity against §12(e) max 3.1e-4 / 1.7e-4; frame assertion
frame_err 0 on every case. Seed means r (p / v / nut / tke / omega): half 1.00 / 1.01 / 1.00 / 1.00 / 1.00, chunks4 the same,
same 1.00, far 1.04 / 1.05 / 1.03 / 1.01 / 1.02 (median d 0.10 on p), wall 1.44 / 1.24 / 1.30 / 1.21 / 1.62 (material,
not severe), sub1000 1.09 / 1.05 / 1.04 / 1.06 / 1.04, sub100 1.40 / 1.18 / ..., sub10 1.96 / 1.54 / ..., sub1 3.99 / 2.92 /
2.30 / 1.45 / 4.37 (severe). Both seeds agree on every label. Native errors are large (p 0.31 / 0.29, v 0.12 / 0.12), so
the ratios are compressed while the predictions still move (d up to 0.3 on p with wall-heavy companions). Full table:
`results/qpop_az/labels_az_isla.md`; copied to the §12(f) file.

## 3. REMESH-090 (r seed mean [s42, s43], median d; full tables in `results/remesh/labels.md`)

| arm / variant | pressure r | velocity r | nut r | d p / v / nut | label |
|---|---|---|---|---|---|
| AF ISLA refine2 / refine4 / refine31 | 0.9999 | 1.0000 | 1.0000 | 3e-5 / 2e-5 / 2e-5 | **invariant** (all three) |
| AF ISLA-DTT refine2 | 1.011 [1.009, 1.013] | 1.004 | 1.002 | 2e-3 / 8e-4 / 2e-3 | invariant |
| AF ISLA-DTT refine4 | 1.036 [1.024, 1.048] | 1.013 | 1.007 | 3e-3 / 1e-3 / 2e-3 | small move |
| AF ISLA-DTT refine31 | 1.060 [1.050, 1.069] | 1.019 | 1.036 | 3e-3 / 1e-3 / 2e-3 | small move |
| AF ISLA tok800 / 600 / 400 / 200 | 1.018 / 1.37 / 2.39 / 5.21 | 1.004 / 1.11 / 1.53 / 2.78 | 1.006 / 1.07 / 1.36 / 2.51 | tok400: 0.018 / 0.005 / 0.009 | small move / **sensitive** / **sensitive** / **sensitive** |
| AF ISLA-DTT tok800 / 600 / 400 / 200 | 1.046 / 1.50 / 2.63 / 5.64 | 1.039 / 1.30 / 2.00 / 3.94 | 1.011 / 1.14 / 1.58 / 2.78 | tok400: 0.027 / 0.009 / 0.018 | small move / **sensitive** ×3 |
| AF ISLA decim400 / decim200 (new geometry) | 1.31 / 1.91 | 3.37 / 7.45 | 1.04 / 1.24 | 0.007 / 0.021 / 0.003 (400) | **sensitive** (robustness, not invariance) |
| AF ISLA-DTT decim400 / decim200 | 1.19 / 1.35 | 3.11 / 6.87 | 1.08 / 1.28 | | **sensitive** (robustness) |
| Volume tok100k | 1.0012 | 1.0005 | 1.0006 | 2e-3 / 3e-3 / 3e-3 | **invariant** (d 1.3-1.4× the redraw floor) |
| Volume tok400k | 0.9997 | 0.9997 | 0.9998 | 1e-3 / 2e-3 / 2e-3 | **invariant**, within draw noise in d (0.87× floor) |
| Volume subdiv4 | 1.0017 | 1.0052 | 0.9996 | 5e-3 / 9e-3 / 2e-3 | **invariant** (d 2.9 / 3.4 / 1.2× the redraw floor) |
| Volume stl (new geometry) | 2.95 [2.67, 3.23] | 1.84 [1.87, 1.81] | 1.48 [1.24, 1.71] | 0.09-0.12 / 0.09 / 0.06-0.12 | **sensitive** (robustness, not invariance) |
| Volume redraw_u0 / u25 (floor) | 1.0000 / 1.0003 | 0.9999 / 1.0001 | 0.9997 / 1.0003 | 1.5e-3 / 2.2e-3 / 2.1e-3 | invariant |

- AirfRANS draw floor at a binding budget: redraw400 against tok400 gives r 1.06 / 1.03 / 1.03 and d 0.031 / 0.010 /
  0.017 (ISLA s42). Two different 400-token draws are farther apart than either is from native. The 400-token loss is
  draw variance, not a bias (labelled "within draw noise" by the registered d criterion; r is not compared there).
- 0.9.0 against GEO1 at 400 area-law tokens: 0.9.0 ISLA 2.39 / 1.53 (p / v), against GEO1's 3.1-4.0 / 2.5-2.9
  (`remesh/FACTS_CHECKED.md` #5). The count effect shrank but did not vanish.

## 4. Analyses beside the registered reading (post hoc, labelled)

**(a) Volume STL, by distance to the wall** (job 310919, probe v3 `--stl-diag`; 48 cars; rows reproduce the
registered all-points ratios exactly). r = STL ÷ native per band, s42 / s43:
| points | per case | pressure r | velocity r | nut r |
|---|---|---|---|---|
| inside the STL (1.6%) | 3,202 | 10.7 / 15.2 | 10.6 / 17.1 | 70 / 119 |
| native wall distance < 1e-3 L | 64,889 | 3.44 / 4.22 | 2.55 / 2.45 | 2.92 / 5.16 |
| 1e-3 L to 1e-2 L | 78,660 | 1.16 / 1.15 | 1.39 / 1.36 | 1.04 / 1.04 |
| ≥ 1e-2 L | 56,451 | 1.011 / 1.006 | 1.007 / 1.002 | 1.000 / 1.001 |
Beyond 0.01 L the STL boundary moves the error by ≤ 1.1%. The registered "sensitive" comes from the near-wall region.
**How far apart the surfaces are** (`diag_stl/STL_OFFSET.md`, jobs 312063, 312116; 48 cars, 4 M area samples per
surface, exact point-to-triangle distances): at 0.01 L they coincide. No point of either surface is 0.01 L from the
other, CFD → STL is at most 2.2e-3 L (4.7 mm), and only 0.005-0.04% of the CFD wall is farther than 1e-3 L. But they are
offset everywhere by a median 1.3e-5 L (27 µm; 99th percentile 7.8e-5 L), the chord error of a 753k-triangle
tessellation against a 15-20M one, and 34% of the CFD wall lies inside the STL. That offset is about 10× the softening
of ISLA's log-distance channel (1e-6 L). It is also the whole wall distance of the 5.5% of interior nodes that sit ON the
CFD wall (d < 1e-6 L), whose distance the STL moves from 0 to about 1.1e-5 L (29% of them land inside the STL).
Reading: by the supervisor's 0.01 L criterion the offset is not comparable, so the registered "sensitive" stands
without a "geometry differs" relabel. The surfaces differ only at the 1e-5 L scale, which ISLA resolves next to the
wall. The cause (the on-wall nodes' distance channel) is consistent with these numbers but was not tested
separately.

**(b) ISLA-DTT refinement against the tree-read opening angle θ** (jobs 311989-311991; θ patched in the probe
process; `results/remesh_theta/summary.md`). Pressure r, seed mean, refine2 / refine4 / refine31:
θ = 1.0 (as trained) 1.011 / 1.036 / 1.060; θ = 0.5 1.014 / 1.027 / 1.032; θ = 0.25 1.003 / 1.007 / 1.001;
θ = 0 (exact read) 1.0006 / 1.003 / 1.0008. Velocity and nut fall the same way, to at most 1.0009 / 1.0001 at θ = 0.
So ISLA-DTT's discretization sensitivity is the tree approximation, and it vanishes as θ → 0. Peter's knob rule credits
this. Caveat: these weights were trained at θ = 1, and changing θ at inference changes the function. The native error
rises from 0.0144 / 0.0075 / 0.023 (θ = 1) to 0.0316 / 0.0121 / 0.096 (θ = 0) on s42. A model trained at small θ would
be needed to price this.

## 5. Predictions (PREREG §6) against outcomes
| # | prediction (p) | outcome |
|---|---|---|
| Q1 | ISLA-DTT roundoff in every mode, AF and AeroZoo (0.93) | AF true; AeroZoo pending (handoff) |
| Q2 | AF ISLA severe in far (0.85); in wall (0.75) | true (136 / 62); true (20 / 30) |
| Q3 | AF ISLA material in chunks4 (0.6); half (0.45); same (0.3) | true (1.33 / 1.36); true (1.11 / 1.12); false (0.95) |
| Q4 | AF small subsets 1-100 severe (0.8); 1,000 severe (0.55) | true; true (4.7 / 5.0) |
| Q5 | Volume severe in far (0.85); half within 1.05 on all fields (0.7) | true (20 / 16); true (1.011 / 1.010 / 1.020) |
| Q6 | AeroZoo ISLA severe in far (0.55); material (0.4) | false (far 1.04 / 1.05, d 0.10 / 0.045); false (half, chunks4, same all ≤ 1.01) |
| Q7 | far ratio at 0.9.0 below A2 (0.5); GEO1 between (0.4) | incomplete as registered (A2 failed validity; no beside-row, §1). On GEO1 → 0.9.0, far rose: p 30 → 136, v 30 → 62, nut 50 → 133. Wall fell: p 52 → 20, nut 59 → 22 (v 21 → 30) |
| R1 | AF exact refinement invariant, both forms (0.65) | false: ISLA invariant on all three; ISLA-DTT invariant on refine2 only (refine4, refine31 small moves, from θ, §4b) |
| R2 | AF 400 and 200 tokens sensitive (0.85) | true (both forms) |
| R3 | Volume subdiv4 invariant (0.5); within draw noise (0.65) | true; false (d 2.9-3.4× the redraw floor on p, v) |
| R4 | Volume 400k invariant (0.55); 100k sensitive (0.3) | true; false (100k invariant too) |
| R5 | Volume STL sensitive (0.45); |r − 1| ≤ 0.05 (0.4) | true; false (2.95 / 1.84 / 1.48) |

## 6. What it means (confidence in brackets)
- **Query independence.** Interacting ISLA at 0.9.0 depends strongly on its query population. This holds on both
  datasets, and on AirfRANS more than at GEO1 under far-field companions [high: both seeds, every label agrees]. The
  practical forms matter. On AirfRANS, 4 chunks raise the error by 33-40% and half the points by 11-14%. Small queries
  are unusable: 1,000 points give 4-6×, single points 85-118×. On DrivAerML Volume, chunking and halving are benign
  (≤ 3.4% on p and v), but small queries are not (4-21×). ISLA-DTT is query-independent to float32 roundoff, as built. On AeroZoo `full` the interacting form's error barely moves
  under chunking, halving or far-field companions (≤ 6%), its predictions still move (d 0.1-0.3), and single-point
  queries cost 1.4-4.4× [high for the subsample of 179 cases; native errors there are large].
  For §12(f)/§13 this is the measured size of the principled case for ISLA-DTT [high, for these eager functions; the
  AirfRANS weights were trained under the defect].
- **Discretization.** Interacting ISLA 0.9.0 is invariant to exact re-panelization on AirfRANS (d 3e-5) and to 4×
  subdivision and the token budget (100k-400k) on DrivAerML Volume [high]. ISLA-DTT moves 1-6% under refinement because
  of its θ = 1 tree read, which vanishes as θ → 0 [high for the mechanism; the cost of a small θ is not priced]. Token
  counts far below the boundary's own cell count (AirfRANS 400 or fewer) cost 1.5-5.6×, from draw variance. A watertight
  STL in place of the CFD wall costs 1.5-3× overall, all of it within 0.01 L of the wall [post hoc, high]. The two
  surfaces differ by a median 1.3e-5 L, so ISLA's near-wall predictions need the wall at the precision of the mesh the
  interior was built on. This replaces
  the book's unmeasured "5× more robust to an STL-like boundary" sentence.

## 7. AirfRANS on fix-trained weights (PREREG §8; "fix-trained weights (the training-path measurement; not the AirfRANS standing)")

ISLA pair r090fix_af_isla_seed42/43 (code_cmpaf_fix, recipe_r090, epoch 500), fully eager float32. QUERY-POP job 325715,
REMESH job 326381 (325779 failed on a code guard; JOBS.md). Validity against the official evals_r090fix records: probe
`full` / native vs official max 0.0 on 200 / 200 cases, both seeds (the official scorer ran eager, and eager ISLA on
AirfRANS is bitwise repeatable). Beside it, this lane's own infer_ref (the fixed compiled path) differs from the official
record by at most 3.8e-5 / 2.8e-4. Tables: `results/qpop_afix/labels_afix_isla.md`, `results/remesh_afix/labels_afix_isla.md`.

| mode / variant | fix-trained, seed mean (p / v / nut) | defect-trained (§§2-3) | label (both) |
|---|---|---|---|
| half | 1.11 / 1.13 / 1.14 | 1.11 / 1.12 / 1.14 | material |
| chunks4 | 1.33 / 1.37 / 1.41 | 1.33 / 1.36 / 1.40 | material |
| sub1000 / sub1 (p) | 4.6 / 94 | 4.7 / 85 | severe |
| same | 0.95 / 0.95 / 0.95 | 0.95 / 0.95 / 0.95 | not material |
| far | 98 / 54 / 131 | 136 / 62 / 133 | severe |
| wall | 19 / 29 / 24 | 20 / 30 / 22 | severe |
| refine2 / refine4 / refine31 | 1.0001 / 1.0001 / 1.0001 (d 3e-5) | 0.9999 (d 3e-5) | invariant |
| tok800 / 600 / 400 / 200 (p) | 1.02 / 1.32 / 2.41 / 5.32 | 1.02 / 1.37 / 2.39 / 5.21 | small / sensitive ×3 |
| decim400 / decim200 (v) | 3.45 / 7.63 | 3.37 / 7.45 | sensitive |

**ISLA-DTT pair** (r090fix_af_dtt_seed42/43; QUERY-POP 330576, REMESH 330583; validity against the official records
max 4.0e-6 / 3.2e-6 QUERY-POP, 3.3e-6 / 1.6e-6 REMESH). QUERY-POP: roundoff in all ten modes, both seeds (max |r − 1|
4.5e-6, median d 2.7e-7 to 8.8e-7). REMESH (pressure r): refine2 1.010 (invariant), refine4 1.039 and refine31 1.070
(small moves), tok800 / 600 / 400 / 200 1.08 / 1.62 / 3.06 / 6.86 (small / sensitive ×3), decim400 / 200 velocity 3.20 /
7.04 (sensitive). Defect-trained: 1.011 / 1.036 / 1.060; 1.05 / 1.50 / 2.63 / 5.64. Tables:
`results/qpop_afix/reduced.{json,md}`, `results/remesh_afix_dtt/labels_afix_dtt.md`.

Every ISLA and ISLA-DTT label repeats on the fix-trained weights. The training path does not change interacting ISLA's
query-population dependence or its re-meshing invariance; the far ratio's size moves within its seed spread (91-181
defect-trained, 93-103 fix-trained). The eager-trained rows of §9 follow (JOBS.md handoff (f)).

## 8. Left
- AeroZoo `azc_full` ISLA-DTT (~10-23): QUERY-POP as for ISLA (handoff (d) in JOBS.md), then added to the §12(f) file.
- PREREG §9: the eager-trained AirfRANS rows r090eag_af_{isla,dtt} (handoff (f); ISLA ~21:00 EDT 10-22, ISLA-DTT ~03:00 EDT 10-23).
- Registered follow-up: AirfRANS QUERY-POP and REMESH on the retrained (fixed-compile) REF-090 checkpoints.

## Erratum (2026-10-22, from the reviewer's second read; detail in scratch/isla-review-2026-10-21/second-read-2026-10-22/probes.md)
- The largest STL-to-CFD-wall separation is 7.8e-3 L, not 2.2e-3 L (that was one direction only).
- 'Querying a few points returns predictions off by factors of 5 to 100' should read 'queries of at most 1,000 points raise the error 4.7 to 118 times'.
- 'Re-meshing invariance holds wherever the geometry is exact' overclaims: token budgets well below the cell count are not invariant (600 to 200 tokens, 1.37 to 5.21 times).
- 'The dependence is no smaller than before 0.9.0' is mixed: far-field companions did not shrink, but wall-heavy and single-point modes shrank against GEO1.
- 'Severe on both datasets' is wrong for AeroZoo, where only the single-point mode is severe.
- The job list covers about 17 of about 36 GPU-h.
- The tok200 'within draw noise' beside-label in results/remesh/labels.md rests on d alone. The registered text (PREREG.md:94-96) also needs r within the redraw's, which fails at 200 tokens: pressure 5.21 and eddy viscosity 2.51 exceed redraw200's 5.13 and 2.23. Only tok400 qualifies (r 2.39 / 1.53 / 1.36 against redraw400's 2.53 / 1.60 / 1.38; d 0.56 to 0.63 of the floor). The notebook entry reads 'small / sensitive x3 (400: within draw noise)'.
