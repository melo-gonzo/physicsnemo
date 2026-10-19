<!-- Copy of the program coordinator's read-only drive audit (scratch/isla-drive-audit-2026-10-18/REPORT.md, 2026-10-18), kept here so the book can cite what each model receives. -->

# DRIVE-AUDIT: the drive inputs the baselines read, against ISLA's (2026-10-18)

Read-only audit, run 15:51–16:18 EDT on 2026-10-18. Sources:

- The model and dataset configs the standings lanes ran: LHR `recipe_v8` (CHAR, REF-80 and AF-UNIFORM lanes; SNAPSHOT_HEAD 343b284eb), `recipe_v9` (AeroZoo AZ lanes) and `recipe_a2r` (A2-RERUN, ISLA only). Copies were read here and then deleted.
- The lane tables `$T/kernel/{char/r80_lanes,char/cl_lanes,airfrans_u/lanes,aerozoo/lanes,a2r/lanes}.tsv`. No lane's `extra` column overrides `forward_kwargs`.
- The `resolved_config.yaml` of the lanes still on LHR: `afu_r_{gt,t,isla}_seed42`, `runs_from_aga/cl_r_af_{gt,t}_seed42`, `runs_from_aga/az_full_gt_seed42` and `a2r_suv_isla_seed42`. Each confirms what the configs say.
- Case data, read with `read_globals.py` (LHR td14 venv) and `read_globals_stdlib.py` (JHB, HiLift medium). The raw output is in `globals_*.txt`, and the AirfRANS split statistics are in `airfrans_split_stats.txt`.

Note: `$T/.venv-recipe-td14` does not exist. The venv is at `.../psharpe/physicsnemo-mesh-transformer/.venv-recipe-td14`. HiLift data is not on LHR; it was read on JHB.

## 1. Inputs per dataset × model

Abbreviations: GT = GeoTransolver, T = Transolver, "norm?" = is the input normalized before the model sees it, "vs ISLA" = information compared with ISLA's input.

| dataset | model | global or drive input (source) | units, range across cases | norm? | vs ISLA |
|---|---|---|---|---|---|
| DrivAerML Surface | ISLA | `global_data.U_inf_dir` (rank 1) | unit; (1,0,0) for every car | unit by construction | — |
| | GT | `global_embedding: global_data.U_inf` | m/s; (38.889, 0, 0) for every car (6 of 484 read; the book says the same for all) | **no**, magnitude 38.9 | same (constant: no information) |
| | T | `fx.source: global_data.U_inf` | same as GT | **no** | same |
| DrivAerML Volume | ISLA | `U_inf_dir` | unit, constant | yes | — |
| | GT / T | `global_data.U_inf` | (38.889, 0, 0), constant | **no** | same |
| HiLift (medium) | ISLA | `U_inf_dir` | unit; AoA 4–20° | yes | — |
| | GT / T | `global_data.U_inf` | in/s; \|U\| = 2679.505 on every case; Ux 2518–2673, Uz 187–916 (14 of 690 read) | **no**, magnitude 2,680, varies with AoA | same information, scale ×2680 |
| SHIFT-SUV | ISLA | `U_inf_dir` | unit, constant | yes | — |
| | GT / T | `global_data.U_inf` | m/s; (30, 0, 0) for every case (6 of 996 read) | **no**, magnitude 30 | same (constant) |
| AirfRANS (AF-UNIFORM) | ISLA | `U_inf_dir` (rank 1) + `U_inf_mag` (rank 0) | unit; speed 31.3–93.6 m/s (full_train mean 62.1, std 17.9); AoA −4.9…14.9° | the direction yes; **the speed no** (raw scalar appended to the seed MLP) | — |
| | GT | `global_embedding: global_data.U_inf` | m/s; Ux 30.6–93.4, Uy −7.2…22.9 (all 1,000 names parsed, 10 cases read) | **no**, magnitude 31–94, **varies in magnitude and direction** | same information, scale ×30–94 |
| | T | `fx.source: global_data.U_inf` (concatenated into every point's features) | same as GT | **no** | same, scale ×30–94 |
| Tensile2d | ISLA | `P_std, p1_std … p5_std` (rank 0); boundary `bc_loaded, bc_fixed, bc_pinned, bc_pressure` (P/45) | z-scores over train_400 (raw P −49…−41, p5 55k–90k) | yes | — |
| | GT / T | `params_std` (the same 6 z-scores); nodal `tag_top, tag_bottom, tag_bottomleft` | O(1) | yes | same |
| GINOT lug | ISLA | `hole_radius_std, length_std, thickness_std, pressure_std`; boundary `bc_fixed, bc_hole, bc_loaded, bc_pressure` (P/75.3) | z-scores (raw pressure 60–96 MPa) | yes | — |
| | GT / T | `params_std` (the same 4 z-scores); interior `sdf, sdf_normals` only | O(1) | yes | **less, given explicitly**: no BC labels (see F4) |
| DeepJEB++ | ISLA | `load_dir` (rank 1), `is_force`, `is_torque`; boundary `bc_fixed, bc_loaded, bc_load` (/20.43) | load_dir unit (8 cases); one-hot; load magnitude 35.6k–565k enters only through `bc_load` | yes | — |
| | GT | `global_embedding: [load_dir, load_type]`; local `bc_fixed, bc_loaded, bc_load` | the same values | yes (unit, one-hot, standardized) | same |
| | T | `fx.source: [load_dir, load_type]`; the same per-cell BCs in `embedding` | the same values | yes | same |
| AeroZoo | ISLA | `U_inf_dir, log_re_std, inlet_{tke,omega,intermittency,retheta}_std`; per face of four boundary meshes: role one-hot and prescribed `bc_velocity, bc_pressure, bc_tke …` (nondimensional) | unit; log_re_std ≈ ±1.7 (Re 1e3–1e8); z-scores | yes | — |
| | GT | `global_vector` = concat of the same 8 globals; local `sdf_wall, sdf_wall_normals, sdf_slip` | the same values | yes | **less**: no inlet or outlet faces, no per-face prescribed values; a secondary inlet (1 case in 7) is invisible to it |
| | T | not run on AeroZoo | | | |

AeroZoo has no Transolver lanes, so it has no T row.

## 2. Flags

**F1. AirfRANS, GT and T: a raw, varying freestream vector of magnitude 31–94 m/s. Severity HIGH.**
- AF-UNIFORM `afu_r_gt_seed{42,43}` and `afu_r_t_seed{42,43}`, and the earlier `cl_r_af_*`, all read `global_data.U_inf` (checked in their resolved configs).
- This is the HiLift situation at about 1/40 of the scale. The drive varies in direction (AoA) and in magnitude (the Reynolds number), so the baselines need it. At these magnitudes the untrained projector is already collapsed: at 38.9 it occupies 1.0–3.8 of 256 slices (`tbl-projector`, `results/geotransolver_drive_conditioning_2026-09-08.json`).
- The pattern of errors fits a direction-conditioning failure. Baseline ÷ ISLA (`results/char/airfrans_uniform.json`, native mesh) is:
  - pressure 0.59 (GT) and 0.65 (T);
  - eddy viscosity 0.80 and 0.87;
  - **velocity 3.05 and 5.59**.

  Velocity scaled by \|U\| is dominated by the far field, which equals (cos α, sin α). A relative L2 of 0.037 corresponds to about 2° of direction error. The angle is carried only by Uy (−7…23) on a vector whose norm varies threefold.
- Evidence on HiLift for the size of the effect: at 35 cases, the unit-drive baselines improved 3.4× and 3.8× by the median per case, on 180 of 180 cases (`sec-nb-udrv-verdict`). At 210 cases the "2.2×" became 0.83–0.91× (`sec-nb-udrvl-210-verdict`). At 1,260 cases the effect was smaller, a slow start: the physical-drive GT was at parity, and the unit-drive GT reached 0.88× (`sec-nb-udrv-1260-verdict`).
- Expected outcome: AirfRANS has 800 training cases at a scale between DrivAerML's (null) and HiLift's, so a partial-to-large effect on velocity is likely. ISLA's 3–5.6× velocity lead is the comparison most at risk in the book. Confidence: moderate on the direction of the change, low on its size.
- The book (`09-airfrans.qmd` line ~255) says the baselines get "the inlet velocity through the unit-direction rule". The configs contradict this.
- ISLA's own `U_inf_mag` is also raw (31–94) and goes into its seed MLP. That can only cost ISLA; it is not a fairness issue against the baselines.

**F2. HiLift, GT and T: raw U_inf of magnitude 2,680, varying with AoA. Severity HIGH; already known.**
- `U_inf_dir` carries the same information, because \|U\| is 2679.505 on every case read.
- This is the finding that prompted the audit; another lane is relaunching these baselines. This audit confirms that `U_inf_dir` alone is the complete fix, with no speed channel needed.

**F3. DrivAerML Surface, DrivAerML Volume and SHIFT-SUV, GT and T: a raw but constant vector, (38.9,0,0) or (30,0,0). Severity LOW to MEDIUM.**
- The vector carries no information, so the information matches ISLA's. The issue is conditioning only.
- Evidence:
  - GT on the DrivAerML surface: unit and physical drive gave 0.0556 against 0.0548 (null; `sec-nb-uw-verdict`; 435 cars, 10k cells).
  - GT on the DrivAerML volume: changes under 2% (null; `sec-nb-udrv-int-verdict`).
  - **Transolver has never been controlled on a constant drive at these magnitudes.** At HiLift's fixed angle (a constant drive of 2,680), Transolver went from 0.091 to 0.032 with the unit drive. A constant 38.9 in `fx` adds a bias of up to about 13 to the first-layer pre-activations at initialization, against O(1–4) from the coordinates.
  - No control at the CHAR protocol (200k cells, the REF widths) exists for either baseline.
- The book is wrong in three places. `05-protocol.qmd` (unit-direction rule) says "every dataset pipeline … writes the far-field direction as a unit vector", which is true but is not what the baselines read. `07-drivaerml.qmd` line ~436 says the baselines read the direction "through the unit-direction rule". `10-shift-suv.qmd` line ~223 says "Every model receives the unit direction only".

**F4. GINOT lug, GT and T: no boundary-condition labels. Severity LOW to MEDIUM.**
- ISLA reads which faces are clamped, the hole, the loaded half of the hole, and the pressure on those faces. The baselines read the interior nodes with `sdf` and `sdf_normals` and the four parameter z-scores, which include the pressure.
- The clamp and the loaded half sit at fixed places in the lug's canonical frame, and augmentation is off. The baselines can therefore infer the labels from the coordinates, so the information is equal in principle, but the labels are not given to them explicitly.
- ISLA leads by 1.21× (GT) and 1.26× (T) on von Mises stress (`results/char/lug_a2.json`).
- The book (`12-lug.qmd` line ~251) says the baselines receive "the nodal condition tags". The lug data has no nodal tags (interior point_data holds only `sdf`, `sdf_normals` and `von_mises`); Tensile2d has them.

**F5. AeroZoo, GT: no channel for the inlet and outlet faces, or for a secondary inlet's velocity. Severity LOW.**
- This is documented in `20-aerozoo.qmd` lines 454–457.
- The asymmetry favors ISLA, yet GT leads on almost every field. GT ÷ ISLA on the full split is 0.76 (velocity), 0.62 (pressure), 0.58 (eddy viscosity), 0.98 (k) and 0.95 (ω); on the non-box test split it is 0.30–0.97.
- ISLA leads only on the Reynolds test split: 1.46 on eddy viscosity and 1.14 on k (`results/campaigns_2026-10-09/aerozoo/aerozoo_reduction.json`). Those two leads are the only AeroZoo numbers that could come from the extra inputs.

**No flag.**
- DeepJEB++: the same values, all O(1).
- Tensile2d: the same z-scores, and the same BC tags as nodal fields.
- AeroZoo globals: the same 8 standardized leaves for both models.

## 3. Recommended reruns and costs

| # | what | lanes | override (recipe config only; no model code) | cluster, GPU-hours |
|---|---|---|---|---|
| R1 (do it) | AirfRANS baselines with the unit direction and a standardized speed | `afu_r_gt_seed{42,43}`, `afu_r_t_seed{42,43}`, as e.g. `afu_ud_{gt,t}_seed{42,43}` | new dataset file `airfrans_uvolume_ud.yaml` = `airfrans_uvolume.yaml` + `SplitGlobalVector{field: U_inf_mag, names: [U_inf_mag_std], mean: 62.116, std: 17.914}` (full_train statistics). GT: `'forward_kwargs.global_embedding=[global_data.U_inf_dir,global_data.U_inf_mag_std]' model.global_dim=4`. T: `'forward_kwargs.fx.source=[global_data.U_inf_dir,global_data.U_inf_mag_std]' model.functional_dim=4` | LHR (where AF-UNIFORM ran); ≈25 (GT) and 34 (T) GPU-h per seed from the train.log active time, ≈120 GPU-h in all (≤170 by `char_lanes_cost`). A list-valued global input already runs in the DeepJEB++ baseline configs; `forward_kwargs.resolve_spec` pads (B,) to (B,1) before it concatenates |
| R1b (optional, your decision) | ISLA AirfRANS with `U_inf_mag_std` in place of the raw `U_inf_mag`, so both sides follow the rule "magnitudes as standardized scalars" | 2 lanes | `model.global_schema` swaps `U_inf_mag` for `U_inf_mag_std` on the same new dataset file | ≈190 GPU-h (94 per seed) |
| R2 (cheap control) | Transolver on a constant drive | REF Transolver 512×16 at 100k, DrivAerML Surface, seeds 42/43 | `forward_kwargs.fx.source=global_data.U_inf_dir` | ≈70 GPU-h. If within 3%: no reruns; correct the book text (F3). If not: rerun T on DrivAerML Volume and SHIFT-SUV (≈175 GPU-h) and GT on all three (≈365 GPU-h) |
| R3 (optional) | GT lug with explicit BC labels | 2 lanes (GT 256) | `++dataset_reader_overrides.drop_in_file_boundaries=false 'forward_kwargs.geometry=[boundaries.surface.cell_centroids,boundaries.surface.cell_data.normals,boundaries.surface.cell_data.bc_fixed,boundaries.surface.cell_data.bc_hole,boundaries.surface.cell_data.bc_loaded,boundaries.surface.cell_data.bc_pressure]' model.geometry_dim=10` (geometry is a generic feature path in GeoTransolver). T has no config-only path: it would need nodal tags written by a recipe transform | ≈95 GPU-h |
| R4 (no action now) | AeroZoo GT inlet and outlet channels | — | possible later: `ComputeSDFFromBoundary` on `inlet` and `outlet` into `local_embedding` (`functional_dim` 8→10); a secondary inlet's velocity has no config-only path | ≈200 GPU-h per lane (AGA, `az_full_gt` ran ≈52 active hours) |

The recipe defaults on `psharpe/isla` (`unified_external_aero_recipe/conf/model/{geotransolver_surface,geotransolver_volume,geotransolver_volume_highlift,transolver_surface,transolver_volume}.yaml`) still read `global_data.U_inf`. Any new lane inherits the confound until those defaults change to `U_inf_dir`. The change is to recipe config only; it is listed for the main session and was not made here.
