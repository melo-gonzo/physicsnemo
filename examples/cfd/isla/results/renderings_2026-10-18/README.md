# FIG lane artifacts (ISLA book overhaul, 2026-10-18)

Every number below sits in a file in this directory, along with the run, the job and the command that produced it.
Compute: LHR (GB300), short QoS, inference and microbenchmarks only. Task dir on LHR:
`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`,
lane dir `$T/book_figs_2026-10-18` (scripts/, runs/ = read-only copies of final checkpoints, dump/, cost/, stream/,
routing/, sbatch_logs/). Local scratch: `~/gh/physicsnemo/scratch/isla-book-overhaul-2026-10-18/fig/`.

## Artifacts

| file | what | produced by |
|---|---|---|
| `stream_norms.json` | H-RESID residual stream: per checkpoint, the mean token norm of h after the last block (`stream_out`), the per-block norms, the per-block amplification (total, slice/broadcast step, MLP step) and the sum of log10 amplifications, for the AeroZoo `full` lanes: raw reads at 3.75e-4 (seeds 42 and 43, to epoch 249), raw at half rate (uniform, windowed), arm A and arm A2 (seeds 42 and 43, epochs 1-249) | `scripts/stream_reduce.py` over the probe jsons of LHR jobs 224693 (2026-10-15), 240803/240804 (2026-10-16) and **270135** (this lane: `scripts/stream_ext.sbatch`, epochs 99-249 and the raw seed-43 series; log `stream_ext_270135.log.txt`). Probe `hresid_probe.py` unmodified (md5 514f3c6b...). |
| `cost_sweep_ref_a2_gb300_bf16.json` | training-step time (median, p10, p90 of 25 steps after 8 warm-up steps) and peak allocated memory against token count, DrivAerML Surface 25k-800k cells: ISLA 512x12 (A2, compiled fullgraph/dynamic/max-autotune-no-cudagraphs, the lane's compile_blocks=true), GeoTransolver 256, Transolver 512x16 (torch.compile as recipe_v8's train.py); DrivAerML Volume 100k-400k (N boundary cells + N queries): ISLA and ISLA-DTT (compile=false, compile_blocks=true, per isla_volume.yaml's DTT note). Real training batches; data loading not timed; the four GPU streams of one node ran concurrently, one arm per GPU | `scripts/bench_step.py` via `scripts/cost.sbatch`, LHR job **271948**; reduced by `scripts/cost_reduce.py` |
| `arrays/airfrans_00215.npz`, `arrays/render_2d_airfrans.json` | AirfRANS test sample 00215 (Re 2.86e6, AoA -2.06°), the case of median ISLA (afu_a2s_isla_seed42) pressure error among 200 test cases (scorer's metrics.jsonl, LHR evals_afu): coordinates (float32) and c_p and \|u\|/U_inf truth and predictions of ISLA (afu_a2s_isla_seed42, A2), GeoTransolver 256 (afu_r_gt_seed42), Transolver 512x16 (afu_r_t_seed42), float16, in the window around the airfoil; json: per-sample relative L2 (normalized space, the scorer's metric) | `scripts/dump_wrap.py` via `scripts/dump.sbatch` rows 6-8 of `scripts/fig_rows.tsv`, LHR job **271987**; `scripts/render_2d.py dump png arrays airfrans` |
| `book/figures/renderings/airfrans_pressure.png`, `airfrans_velocity.png` | truth / ISLA / GeoTransolver / Transolver (shared viridis scale) and prediction minus truth (shared symmetric DIVERGING scale), Delaunay of the 100,000 scored points | `scripts/render_2d.py` |

Per-sample relative L2 on AirfRANS 00215 (from `arrays/render_2d_airfrans.json`): pressure ISLA 0.0156, GeoTransolver
0.0118, Transolver 0.0113; velocity ISLA 0.0074, GeoTransolver 0.0226, Transolver 0.0467. Note: the dump draws the
100,000 interior points in a single-sample run; ISLA's pressure error of this case in the scored record is 0.01543
(the draw differs from the record's), and all three models in the dump see identical points.

| `routing/<run>_s<sample>.{npz,json}` | trained routing of the A2 reference ISLA checkpoints (final epoch 500) on the median case: per layer dominant slice, its weight, slice masses q and qw, anchors and anchor normals; for layers 0, 5, 11 the 8 largest-territory slices' soft assignment and routing density (schema of results/probes/slices_2026-09-22); json: per-layer n_eff, entropies, anchor spread. Runs: a2r_dr_isla_seed42 (DrivAerML 00079, all 200,000 tokens stored), afu_a2s_isla_seed42 (AirfRANS 00215), a2r_lug_isla_seed42 (lug 00633), a2r_t2dz_isla_seed42 (Tensile2d 00238) | `scripts/slices_probe_a2.py` via `scripts/routing.sbatch`, LHR job **275271** |

## Renderings of the median cases (2026-10-19)

All rendered locally on the CPU (`CUDA_VISIBLE_DEVICES= nice -n 19`, VTK 9.7.1 / pyvista 0.49 off-screen on llvmpipe,
matplotlib) from the dumps of LHR jobs 271987 and 275252 (local copy `fig/dump/`); each json beside the arrays records
the case, the scorer's per-sample relative L2 of every rendered field (metrics.jsonl of the same run, normalized space,
as in results/char) and, under `source`, the fig_rows.tsv row, run id and LHR dump job of every panel. Panels are labelled
by architecture (ISLA = A2 reference 512 x 12, GeoTransolver = 256, Transolver = 512 x 16; run map below). Fields:
viridis, 0.5-99.5th percentile of the truth; errors: the book's DIVERGING map, symmetric, 99th percentile of the errors.
Why the first car run was blank: on this machine's software OpenGL, VTK draws a mesh coloured by CELL data as white
when lighting is on (a plain-colour or point-data mesh draws correctly); `render_lib.render` now converts cell values to
point data (mean of the adjacent cells) and also resets the clipping range. The car points were also misplaced by 1 m:
infer.py writes the centred points divided by L_ref = 5 m, so the body frame is x = 5 x* + center (`render_car.to_body`).

| file | what | command |
|---|---|---|
| `book/figures/renderings/drivaerml_surface_cp.png`, `drivaerml_surface_cf.png`; `arrays/drivaerml_surface_00079.npz`, `arrays/render_car.json` | DrivAerML Surface car 00079 (run 171), left side, orthographic, on the case's STL (753,238 triangles; each triangle takes the nearest scored cell of that model, median distance 5.7 mm, ISLA and GeoTransolver on the same 200,000 cells, Transolver on its 100,000): c_p and \|tau_w\|/q_inf, truth / three models / errors | `python scripts/render_car.py fig/dump fig/geom/drivaer_171_single_solid.stl.pmsh <png> <arrays> pose` |
| `book/figures/renderings/pose_cp.png`, `pose_geometry.png` (numbers in `render_car.json` `pose`) | the scorer's posed protocol (FixedRandomPose: rotation 169.5°, \|t\| = 1.55 m) on the same car: per model, native c_p, posed c_p mapped back to the body frame (x = L_ref R^T x* + c; the posed and native draws are the same cells, map-back distance <= 6e-7 m), posed minus native; and the posed car itself coloured by ISLA's posed prediction | same command |
| `book/figures/renderings/airfrans_pressure.png`, `airfrans_velocity.png` (re-rendered) | as above; panels now labelled by architecture, json keys ISLA / GeoTransolver / Transolver (were "GeoTransolver 256", "Transolver 512 x 16") with a `source` block; numbers unchanged | `python scripts/render_2d.py fig/dump <png> <arrays> airfrans` |
| `book/figures/renderings/tensile2d_stress.png`; `arrays/tensile2d_00238.npz`, `arrays/render_2d_tensile2d.json` | Tensile2d validation specimen 00238: plane-stress von Mises from the scored sig11, sig22, sig12 on a Delaunay of the 9,547 nodes. The dump is in normalized units; the figure multiplies by the shared stress std 37.837 MPa (plaid_tensile2d_2d.yaml), so the scale is MPa | `... render_2d.py ... tensile2d` |
| `book/figures/renderings/lug_von_mises.png`; `arrays/lug_00633.npz`, `arrays/render_2d_lug.json` | lug test case 00633, the top face (the 1,060 nodes at z = thickness of the 12,720 scored nodes): von Mises = 150.2 + 92.4 x the dump's normalized value (ginot_lug_volume.yaml), MPa | `... render_2d.py ... lug` |
| `book/figures/renderings/deepjebpp_von_mises.png`, `deepjebpp_displacement.png`; `arrays/deepjebpp_60073.npz`, `arrays/render_cloud_deepjebpp.json` | DeepJEB++ test bracket 60073 (611-092-IS06_hor), three-quarter view, on the case's FEM surface (50,024 triangles, copied from LHR `datasets/deepjebpp_pdmsh/611-092-IS06_hor/.../boundaries/surface`, 4.4 MB, to `fig/geom/`): every triangle is one scored cell (centroid match <= 1.1e-5 mm). von Mises [MPa] and \|u\| [mm] as infer.py writes them; the panel titles' displacement L2 is the scorer's vector metric | `python scripts/render_cloud.py fig/dump <png> <arrays> deepjebpp fig/geom/deepjebpp_611-092-IS06_hor_surface` |

DrivAerML Surface 00079, per-sample relative L2 (`arrays/render_car.json`): pressure ISLA 0.0415, GeoTransolver 0.0424,
Transolver 0.0406; wall shear ISLA 0.0612, GeoTransolver 0.0667, Transolver 0.0668. Posed protocol, pressure: ISLA
0.0415 (posed minus native rms c_p 7.2e-4, 0.16 % of the native c_p norm), GeoTransolver 0.885 (rms 0.38), Transolver
0.778 (rms 0.35).
Tensile2d 00238 (`arrays/render_2d_tensile2d.json`): stress ISLA 0.0067, GeoTransolver 0.0060, Transolver 0.0071;
displacement 0.0068, 0.0059, 0.0044. Lug 00633 (`arrays/render_2d_lug.json`): von Mises ISLA 0.00137, GeoTransolver
0.00173, Transolver 0.00192. DeepJEB++ 60073 (`arrays/render_cloud_deepjebpp.json`): von Mises ISLA 0.403,
GeoTransolver 0.446, Transolver 0.547; displacement 0.077, 0.115, 0.165.

## Case selection

The representative case of each dataset is the case of median per-case ISLA error (seed 42, final checkpoint, native
protocol, first headline field of `results/char/<dataset>_a2.json`), from the scorers' metrics.jsonl
(`scratch/isla-a2-rerun-2026-10-17/scores/` and LHR `$T/evals_afu/`): DrivAerML Surface 00079 (run 171, pressure),
DrivAerML Volume 00033 (run 129, pressure), SHIFT-SUV 00184 (run 00187, pressure), lug 00633 (von Mises), Tensile2d
00238 (stress), DeepJEB++ 60073 (611-092-IS06_hor, von Mises), AirfRANS 00215 (pressure).

## Run map (`scripts/fig_rows.tsv`)

ISLA A2 reference: `a2r_dr_isla_seed42`, `a2r_vol_isla_seed42` (trained on JHB, final .mdlus copied to LHR
`$F/runs`), `a2r_suv_isla_seed42`, `a2r_lug_isla_seed42`, `a2r_t2dz_isla_seed42`, `jbn2_isla_seed42`,
`afu_a2s_isla_seed42`. Baselines (CHAR record arms): `r80_dr_gt_h256`, `r80_dr_t_h512_l16_c100k` (scored at 100,000
cells), `cl_r_{vol,suv,t2dz,lug,jeb}_{gt,t}`, `afu_r_{gt,t}`, all seed 42, copied from AGA. Each row is the scorer's
infer.py command (snapshot, recipe, venv, cells) for that run.

## Pending at hand-off (see the lane report)

- DrivAerML Surface rendered 2026-10-19 (above).
- All 24 rows of `scripts/fig_rows.tsv` are dumped (LHR jobs 271987 and 275252): `$F/dump/<tag>/<protocol>/*.npz` with
  the scorer's metrics.jsonl beside each, copied to the local scratch `fig/dump/`; only AirfRANS is packed and rendered.
- Not started: HiLift (dataset not on LHR; A2 lanes at epoch 285/500 on JHB), AeroZoo mid-span planes, hero view.
