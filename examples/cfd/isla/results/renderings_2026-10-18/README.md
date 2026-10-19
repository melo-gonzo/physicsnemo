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

- DrivAerML Surface native + posed predictions are dumped (LHR job 275252 rows 0-5, `$F/dump/dr_*`); the renderer
  `scripts/render_car.py` (side view of c_p and c_f, and the pose figure) ran once locally at hand-off and every VTK
  panel came out blank (white), so `common_crop` found nothing; likely cause: the camera's clipping range is not reset
  after `camera_position` is set in `render_lib.render` (add `p.reset_camera_clipping_range()`). Not yet fixed.
- LHR job 275252 (rows 10-23: DeepJEB++, lug, Tensile2d, SHIFT-SUV, DrivAerML Volume baselines and ISLA volume) and
  275271 (A2 routing probe, `scripts/slices_probe_a2.py`, DrivAerML/AirfRANS/lug/Tensile2d) were queued or running;
  outputs land in `$F/dump/<tag>/native/*.npz` and `$F/routing/`.
- Not started: HiLift (dataset not on LHR; A2 lanes at epoch 285/500 on JHB), AeroZoo mid-span planes, hero view.
