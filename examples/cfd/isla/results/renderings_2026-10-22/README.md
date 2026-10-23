# RENDER-090 artifacts (ISLA book, 2026-10-22)

The book's field renderings, trained-routing probes and GB300 cost sweep, re-made from the
current standings: ISLA 0.9.0 (the standing's ISLA run, seed 42) beside the standing's
baselines, GeoTransolver 256 and Transolver 512 x 16 at the fair drive (seed 42). Same
cases, cameras, colour scales and layout as `../renderings_2026-10-18/`; only the models
differ. Every number below sits in a file in this directory, with the run, the job and the
command that produced it.

Compute: LHR (GB300), short QoS, inference and microbenchmarks only. Task dir on LHR:
`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`,
lane dir `$T/book_figs_2026-10-22` (scripts/, dump/, routing/, cost/, sbatch_logs/).
Local scratch: `~/gh/physicsnemo/scratch/isla-render090-2026-10-22/` (dump/, routing/,
cost/, scores/, geom/, logs/; not committed). Every run reads its own code and recipe
snapshot and its own saved `norm_stats.pt`; no snapshot was modified. The scripts ran on
LHR as copied there before the jobs; the committed copies differ from those only by the
license header and the hooks' formatting, except the local-only changes named below
(`render_car.py` array files, `render_cloud.py` point order, `thin_routing.py`).

## Method

- **Dumps** (`scripts/dump.sbatch`, `scripts/dump_wrap.py`): each row of
  `scripts/fig_rows.tsv` is its scorer's inference command (the scorer's
  `score_rows.tsv`: dataset, model, extra, code and recipe snapshots, cells; float32,
  `infer_split=val`, the final checkpoint), run through `dump_wrap.py`, which restricts
  the sampler to the case and writes `.npz`. New against 2026-10-18: with `replay=1` the
  wrapper replays the scorer's reads of the earlier cases before the case is drawn, so a
  random draw (DrivAerML and SHIFT-SUV cells, the DrivAerML Volume and AirfRANS points)
  is the scorer's draw.
- **Scorer check** (`render_lib.scorer_checked`): every per-case error in a panel title
  is the standing's own scorer record of that case (`metrics.jsonl`, phase `infer_step`;
  the last row per case, as the CHAR reducer reads it), and each json records it under
  `scorer_checks` beside the dump's own row. Largest relative difference over all panels:
  1.0e-5 (DrivAerML Volume ISLA: compiled blocks are not bitwise reproducible run to
  run); DeepJEB++ ISLA 1.3e-6; zero on every other panel.
- **Rows reused from 2026-10-18** (`ckpt_root` `REUSE:<row>`): the CHAR baselines that
  are still the standing's arms on DeepJEB++ (`cl_r_jeb_gt`, `cl_r_jeb_t`), Tensile2d
  (`cl_r_t2dz_gt`, `cl_r_t2dz_t`) and the lug Transolver (`cl_r_lug_t`). Their
  checkpoints are no longer on LHR; the 2026-10-18 dump of the same run, case and command
  (LHR job 275252) stands in, and its row equals the CHAR record
  (`results/campaigns_2026-10-02/char/protocols/`).
- **Point order** (`render_cloud.check_same_points`): ISLA 0.9.0 returns a
  multi-condition surface in the conditions' sorted order (DeepJEB++: fixed, free,
  loaded), so the DeepJEB++ baselines' arrays are permuted onto ISLA's cells by an exact
  one-to-one nearest-neighbour match (distance 0); recorded as `point_order`.
- **Case rank**: the cases are the 2026-10-18 cases (median ISLA error under the
  nearest-token form). Each figure title gives the case's rank by ISLA 0.9.0's error among
  the scorer record's cases instead of calling it the median.
- **Rendering**: locally on the CPU (`CUDA_VISIBLE_DEVICES= nice -n 19`, VTK 9.7.1 /
  pyvista 0.49 off-screen, matplotlib), from the local copies of the dumps, on the case
  geometries copied from LHR into the scratch `geom/` (DrivAerML runs 171 and 129
  single-solid STLs, SHIFT-SUV run 00187 merged STL, the DeepJEB++ 611-092-IS06_hor FEM
  surface).
- **File limit** (the hooks' 5,000 KB): the car arrays store the points and the truth once
  (all three models are on the same 200,000 cells) and the posed arrays in
  `drivaerml_surface_00079_pose.npz`; the DrivAerML routing probe is committed through
  `scripts/thin_routing.py` (below).

## Run map (`scripts/fig_rows.tsv`)

| dataset | ISLA 0.9.0 | GeoTransolver 256 | Transolver 512 x 16 | scorer records |
| --- | --- | --- | --- | --- |
| DrivAerML Surface (native, posed) | `r090_dr_isla_seed42` (code_r090) | `r80ud_dr_gt_h256_seed42` | `r80ud_dr_t_h512_l16_c100k_seed42` (200,000 cells) | `evals_r090`, `evals_ud2`, `evals_tset200k` |
| DrivAerML Volume | `r090_vol_isla_seed42` (code_r090) | `clud_vol_gt_seed42` | `clud_vol_t_seed42` | `evals_r090`, `evals_ud2` |
| SHIFT-SUV | `r090_suv_isla_seed42` (code_r090) | `clud_suv_gt_seed42` | `clud_suv_t_seed42` | `evals_r090`, `evals_ud2` |
| DeepJEB++ | `r090_jb_isla_seed42` (code_r090) | `cl_r_jeb_gt_seed42` (reused) | `cl_r_jeb_t_seed42` (reused) | `evals_r090`, CHAR |
| Tensile2d | `r090eag_t2dz_isla_seed42` (eager-trained, code_cmpaf_fix) | `cl_r_t2dz_gt_seed42` (reused) | `cl_r_t2dz_t_seed42` (reused) | `evals_r090eag`, CHAR |
| AirfRANS (phase 2) | `r090eag_af_isla_seed42` | `afu_ud_gt_seed42` | `afu_ud_t_seed42` | `evals_r090eag`, `evals_afu_ud` |
| lug (phase 2) | `r090eag_lug_isla_seed42` | `lugbc_gt_seed42` | `cl_r_lug_t_seed42` (reused) | `evals_r090eag`, `evals_lugbc`, CHAR |

Each run was checked from its saved `resolved_config.yaml` on LHR (width, slices or
layers, seed 42, forward_kwargs: the baselines read `U_inf_dir`, the lug GeoTransolver its
four boundary-condition labels, the AirfRANS baselines `U_inf_dir` and `U_inf_mag_std`)
and has `Training completed` in its `train.log`; every scorer record is at epoch 500
(DeepJEB++ 40).

## Artifacts

`<s>` is the local scratch; commands run in `scripts/`, PNGs written to
`book/figures/renderings/`, arrays to `arrays/`.

| file | what | produced by |
| --- | --- | --- |
| `drivaerml_surface_cp_090.png`, `drivaerml_surface_cf_090.png`; `arrays/drivaerml_surface_00079.npz`, `arrays/render_car.json` | DrivAerML Surface car 00079 (run 171), left side, orthographic, on its STL: c_p and \|tau_w\|/q_inf, truth / three models / errors | dump rows 0, 2, 4 (LHR jobs 336002, 336068, 336003); `python render_car.py <s>/dump <s>/geom/drivaer_171_single_solid.stl.pmsh <png> <arrays> pose` |
| `pose_cp_090.png`, `pose_geometry_090.png`; `arrays/drivaerml_surface_00079_pose.npz` (numbers in `render_car.json` `pose`) | the scorer's posed protocol on the same car (rotation 169.5 deg, \|t\| = 1.55 m), mapped back to the body frame (map-back distance <= 5.7e-7 m) | dump rows 1, 3, 5 (job 336068); same command |
| `drivaerml_volume_pressure_090.png`, `drivaerml_volume_velocity_090.png`; `arrays/drivaerml_volume_00033.npz`, `arrays/render_2d_drivaerml_volume.json` | DrivAerML Volume car 00033 (run 129): the 11,053 scored points within 15 cm of the symmetry plane, c_p and \|u\|/U_inf | dump rows 21, 22, 23 (jobs 336069, 336004); `python render_2d.py <s>/dump <png> <arrays> drivaerml_volume,tensile2d drivaerml_volume=<s>/geom/drivaer_129_single_solid.stl.pmsh` |
| `tensile2d_stress_090.png`; `arrays/tensile2d_00238.npz`, `arrays/render_2d_tensile2d.json` | Tensile2d validation specimen 00238: plane-stress von Mises [MPa] on the 9,547 nodes | dump row 15 (job 336070), rows 16, 17 reused; same command |
| `deepjebpp_von_mises_090.png`, `deepjebpp_displacement_090.png`; `arrays/deepjebpp_60073.npz`, `arrays/render_cloud_deepjebpp.json` | DeepJEB++ test bracket 60073 (611-092-IS06_hor), three-quarter view, on its FEM surface (50,024 triangles, one per scored cell) | dump row 9 (job 336070), rows 10, 11 reused; `python render_cloud.py <s>/dump <png> <arrays> deepjebpp,shift_suv deepjebpp=<s>/geom/deepjebpp_611-092-IS06_hor_surface shift_suv=<s>/geom/shift_suv_00187_merged_surfaces.stl.pmsh` |
| `shift_suv_cp_090.png`; `arrays/shift_suv_00184.npz`, `arrays/render_cloud_shift_suv.json` | SHIFT-SUV validation case 00184 (run 00187), left side, orthographic, on its STL (5,010,474 triangles) | dump rows 18, 19, 20 (jobs 336069, 336070); same command |
| `routing/r090_dr_isla_seed42_s00079.{npz,json}`, `routing/r090eag_t2dz_isla_seed42_s00238.{npz,json}` | trained routing of ISLA 0.9.0 on DrivAerML Surface 00079 (200,000 tokens) and Tensile2d 00238 (462 boundary tokens, 9,547 prediction points), schema of `../renderings_2026-10-18/routing/` plus `n_query` and `probe_metrics` in the json: the probe's own scorer row, equal to the standing's record to 1.0e-6 on DrivAerML (the probe runs the blocks eagerly, the scorer compiled them) and exactly on Tensile2d | `scripts/slices_probe_090.py`, a port of `slices_probe_a2.py` to 0.9.0's forward (the scorer's `infer.py` through `dump_wrap.py`, `model.compile_blocks=false compile=false`, read-only hooks), via `PROBES=dr,t2d sbatch routing.sbatch`, LHR job 336200; then `python thin_routing.py <s>/routing/<tag>/<tag>.npz ../routing/<tag>.npz` |
| `airfrans_pressure_090.png`, `airfrans_velocity_090.png`; `arrays/airfrans_00215.npz`, `arrays/render_2d_airfrans.json` | AirfRANS test sample 00215 (Re 2.86e6, angle of attack -2.06 deg), c_p and in-plane \|u\|/U_inf on a Delaunay of the 100,000 scored points (the same points for all three models: the replayed draw is the scorer's) | dump row 6 (job 345308), rows 7, 8 (job 336070); `python render_2d.py <s>/dump <png> <arrays> airfrans` |
| `routing/r090eag_af_isla_seed42_s00215.{npz,json}` | trained routing of the eager-trained AirfRANS ISLA 0.9.0 on test sample 00215 (1,046 boundary tokens, 100,000 prediction points; 60,000 stored, 616 of them boundary); the probe's own row equals the scorer's exactly | `PROBES=af sbatch routing.sbatch`, LHR job 345309; `thin_routing.py` |
| `lug_von_mises_090.png`; `arrays/lug_00633.npz`, `arrays/render_2d_lug.json` | lug test case 00633, top face (the 1,060 nodes at z = thickness of the 12,720 scored nodes), von Mises [MPa] | dump row 12 (job 343286), row 13 (job 336070), row 14 reused; `python render_2d.py <s>/dump <png> <arrays> lug` |
| `routing/r090eag_lug_isla_seed42_s00633.{npz,json}` | trained routing of the eager-trained lug ISLA 0.9.0 on test case 00633 (7,520 boundary tokens, 12,720 prediction points, all stored); the probe's own row equals the scorer's to 3e-6 | `PROBES=lug sbatch routing.sbatch`, LHR job 343287; `thin_routing.py` (copied unchanged) |
| `cost_sweep_ref_090_gb300_bf16.json` | training-step time (median, p10, p90 of 25 steps after 8 warm-up steps) and peak allocated memory against token count, the method of `../renderings_2026-10-18/cost_sweep_ref_a2_gb300_bf16.json`: DrivAerML Surface 25k-800k cells for ISLA 0.9.0, GeoTransolver 256 and Transolver 512 x 16 (fair drive); DrivAerML Volume 100k-400k prediction points for ISLA 0.9.0 and ISLA-DTT 0.9.0; the tail arm `isla_surface_090_codeR090` | `scripts/bench_step_090.py` via `scripts/cost.sbatch`, LHR job 336005; `python cost_reduce.py` |

`thin_routing.py` copies a probe of at most 60,000 tokens unchanged (Tensile2d, the
lug). The AirfRANS probe, which stored 60,000 of its 101,046 tokens, keeps `points`,
`keep` and the per-token fields as they are, with the top three territories and no
`measure` or `top_a_density`. The DrivAerML probe keeps the 60,000-token subset that
03-isla.qmd's `load_slices` draws: `points` for those tokens only with `keep` = 0..n-1
(so `points[keep]` reads as before) and `token_index` their indices among the 200,000,
the top three territories of each probed layer (the json keeps eight), and no `measure`
or `top_a_density` (the book reads neither). Per-slice fields and the json's statistics
are over all tokens. The full probe output is in the scratch `routing/` and on LHR.

Cost sweep: ISLA 0.9.0 is code_cmpaf_fix (code_r090, 92d9d7800, with per-block compile
at Inductor's mix-order reduction off, as psharpe/isla compiles every block since
b62a93f2c), whole-model compile off and `compile_blocks` on as the r090 lanes'
`resolved_config.yaml` records, bf16, width 512, 512 slices; ISLA-DTT is
`query_independent=true` with its tree read at theta = 1. The tail arm is the same ISLA
on code_r090, the snapshot the r090 DrivAerML lanes trained on. The baselines read
`U_inf_dir`, as the fair-drive arms do (the same shapes as `U_inf`). `bench_step_090.py`
is `bench_step.py` building the model as recipe_r090's `train.py` does (output stats and
dataset constants). ISLA 0.9.0 reads the DrivAerML Volume vehicle whole and draws
min(cells, 200,000) boundary tokens inside the model, so on the volume task the token count
is the prediction points only (the A2 sweep had N boundary cells plus N prediction points).

## Numbers (scorer records, per-case relative L2, normalized space)

| case | field | ISLA 0.9.0 | GeoTransolver | Transolver | ISLA 0.9.0 rank |
| --- | --- | --- | --- | --- | --- |
| DrivAerML Surface 00079 | pressure | 0.0389 | 0.0413 | 0.0387 | 22 of 48 |
| | wall shear | 0.0632 | 0.0656 | 0.0625 | |
| posed protocol, same car | pressure, posed | 0.0389 (posed minus native rms c_p 7.5e-4) | 1.041 (rms 0.41) | 1.396 (rms 0.49) | |
| DrivAerML Volume 00033 | pressure | 0.0393 | 0.0482 | 0.0527 | 29 of 48 |
| | velocity | 0.0611 | 0.0993 | 0.1135 | |
| | eddy viscosity | 0.0637 | 0.0625 | 0.0638 | |
| SHIFT-SUV 00184 | pressure | 0.0417 | 0.0405 | 0.0370 | 49 of 99 |
| | wall shear | 0.0606 | 0.0641 | 0.0615 | |
| DeepJEB++ 60073 | von Mises | 0.469 | 0.446 | 0.547 | 4144 of 5540 |
| | displacement | 0.150 | 0.115 | 0.165 | |
| Tensile2d 00238 | stress | 0.0082 | 0.0060 | 0.0071 | 26 of 100 |
| | displacement | 0.0051 | 0.0059 | 0.0044 | |
| lug 00633 | von Mises | 0.00090 | 0.00153 | 0.00192 | 101 of 600 |
| AirfRANS 00215 | pressure | 0.0116 | 0.0103 | 0.0113 | 113 of 200 |
| | velocity (in-plane) | 0.0052 | 0.0260 | 0.0220 | |

The DeepJEB++ case was the median under the nearest-token form and is in ISLA 0.9.0's
worst quartile; it was kept so the figure differs from the 2026-10-18 one only in the
models.

Cost (median step ms / peak GiB): ISLA 0.9.0 surface 102/9 at 25k, 246/64 at 200k,
944/253 at 800k; GeoTransolver 160/49 at 200k, 523/197 at 800k; Transolver 232/85 at
200k, out of memory at 800k; ISLA 0.9.0 volume 422/95 at 100k, 796/190 at 400k; ISLA-DTT
0.9.0 volume 1202/129 at 100k, 1833/183 at 200k, out of memory at 400k. code_r090 against
code_cmpaf_fix: within 4% at every count but 50k (7.5%); same peak memory. The baselines
reproduce the 2026-10-18 sweep within 5% from 100k cells; at 25k-50k, where a step is
mostly overhead, by up to 22% (Transolver at 25k: 67 against 86 ms).

## Pending

None: every figure of `FIGURES_TO_REPLACE.md` has its `_090` replacement. The LHR dumps
(`$T/book_figs_2026-10-22/dump`, `routing`) are kept as the source the jsons cite.
