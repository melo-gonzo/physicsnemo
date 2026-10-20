# HILIFT-090 Part B: can ISLA 0.9.0 run on HiLift as it stands? (2026-10-20)

**Verdict: it fits. No decimated geometry asset is needed for the HiLift surface task.** ISLA 0.9.0 (psharpe/isla
75ac723d8) trains on `highlift_surface_medium` at 200,000 cells with 64.5 GiB allocated / 66.4 GB reserved per GB300
(of about 277 GB) and the same step time as A2 (warm epochs: 0.338-0.340 s median trainer step against A2's 0.345-0.346 s
on the same node; epoch wall 65.8-67.6 s against 64.5 s).

## Why the premise did not hold

- The full-resolution boundary path of 0.9.0 belongs to the interior task only. `ISLA.forward` takes the Warp BVH
  (`exact_boundary_geometry`) and the in-model systematic draw (`sample_boundary_cells`) only when `prediction_points`
  is given. A surface task (`prediction_points=None`, HiLift's `isla_surface_highlift`) "reads every cell it is given
  (its outputs are predicted there) and samples none" (model.py docstring at 75ac723d8).
- The cells it is given are already sampled. `datasets/highlift_surface_medium.yaml` reads
  `boundaries/vehicle` with `subsample_n_cells: ${sampling_resolution}` (a cyclic contiguous block of 200,000 cells),
  then `DropDegenerateCells` (53-137 cells). DrivAerML Surface does the same, and its REF-090 lanes run now
  (r090_dr: 64.5 GB, 0.24 s/step).
- The on-disk cell order is pre-shuffled. Three 200k blocks of `geo_LHC001_AoA_16` (at 0, N/3 and 2N/3) each span
  the full wing extent (x 93-2625 in, y 0-1160 in) with about 2.8 distinct vertices per cell, so a block is a uniform
  sample. A contiguous block is not a spatial window here (unlike AeroZoo's OpenFOAM order).
- The note "HiLift would need a decimated watertight surface" comes from the interior-task design
  (scratch/isla-geo-exact-design-2026-10-16/REPORT.md, "Input cost"). It applies to a HiLift *volume* task
  (`highlift_volume.yaml` exists), not to this one.
- Consequence for the 0.9.0 contract: on a surface task the "exact full-mesh gauge" is measured on the 200k given cells
  (each with its second moment), not on the full mesh. That is a sample estimate, as on DrivAerML in REF-090.

## Face counts (CFD wing surface, `boundaries/vehicle`, 690 cases)

| | triangles | vertices |
|---|---|---|
| min | 248,651,261 | 126,043,378 |
| median | 277,230,797 | 140,414,061 |
| max | 301,959,583 (`geo_LHC132_AoA_14`) | 152,856,939 |

Cells plus points are about 8.3 GB per case (int64 cells, float32 points); the 14 TB dataset is mostly the cell fields.
Only 20 cases (5 geometries) carry a `geo_*.stl.pmsh` (668k triangles), so the STL is not a general geometry asset.
Every case has `geo_*_element_labels.{npz,json}` (run-length slat / main / flap / other per cell); Part A uses them.
File: `face_counts.txt` (case, triangles, vertices, STL triangles, STL vertices; -1 = no STL).

## Measurements (LHR, GB300, bf16, 4 GPUs, the lane launcher unchanged)

| job | what | result |
|---|---|---|
| 299686 (short) | REF-090 smoke harness (`kernel/h090/smoke/smoke.sbatch`), lane `h090_hl_isla_seed42`, 50 training forwards | ISLA 0.9.0, K=1, unscaled; peak 64.52 GiB allocated, 66.38 GiB reserved on every rank; first step 159 s (compile), median 0.598 s forward-to-forward (cold cache); losses finite and falling (total 0.53 to 0.17) |
| 299686 | same harness on the A2 row | not a real failure: the wrapper reads `ISLA._scaled`, which A2 lacks |
| 299755 (short) | paired, one node: h090 row 0, then A2 row 8 (`kernel/a2r_hl` row, code_a2r / recipe_a2r_hl), each 420 s through `lanes.sbatch`, trainer `Step` / `Mem` lines from step 5 | below |

| epoch | 0.9.0 median / mean (s) | A2 median / mean (s) | 0.9.0 epoch wall | A2 epoch wall |
|---|---|---|---|---|
| 0 (h090 cold page cache, A2 warm) | 0.443 / 0.578 | 0.344 / 0.406 | 162.9 s | 191.0 s |
| 1 | 0.340 / 0.452 | 0.346 / 0.427 | 67.6 s | 64.5 s |
| 2 | 0.338 / 0.423 | 0.345 / 0.438 (97 steps) | 65.8 s | – |

Memory: 66.38 GB (0.9.0) against 65.98 GB (A2), the trainer's `Mem`. The A2 lane itself ran 57 s per epoch on LHR
(epochs 285-499). Where the time goes: there is no BVH build, pre-draw feature pass or draw on this path. The step is the
same trunk as A2's on the same 200k tokens; the extra 0.9.0 work (gauge with per-cell second moments, conditions by name)
does not show in the step time.

## Fallback asset design (only if a HiLift interior task or full-surface geometry is wanted later)

- Method: quadric-error-metric decimation (Garland-Heckbert) of each case's `boundaries/vehicle`, with the knob a
  tolerance eps in chord units (L_ref = 275.8 in): eps/c in {1e-4, 3e-5, 1e-5}. Accept a level when the two-sided
  distance (full-mesh cell centroids to the decimated surface, and decimated vertices to the full mesh) has p99.9 <= eps c
  and max <= 3 eps c. Report both in chord units. Lock the boundary edges and sharp creases so that slat and flap gaps
  (1-2 % c, about 2.8-5.5 in) cannot close; eps = 1e-4 c is 0.028 in. The full mesh's mean edge is about 0.12 in
  (4e-4 c), so the curved leading and trailing edges set the face count. Estimate 1-5 M triangles; one probe case settles it.
- Sidecar per case in our own area, never in the dataset:
  `$T/highlift/geo_decim/eps<e>/<case>/vehicle.pmsh` (float32 points, int32 cells) plus `meta.json` (eps, faces,
  distance stats, md5s of the source `cells`/`points` memmaps, tool version).
- Recipe: a reader-side transform `AttachSidecarBoundary(root, name="vehicle_geometry")` keyed on the sample's
  `source_path` case id. It adds the sidecar as a second DomainMesh boundary *before* the augmentations, so the
  rotation, translation and CenterMesh apply to it identically. It skips DropDegenerateCells (QEM leaves no zero-area
  cells; check it) and it is never subsampled.
- What the model reads: `boundary_meshes.vehicle: boundaries.vehicle_geometry` for geometry (gauge, wall distance, drawn
  tokens). The predictions stay at the full mesh's sampled cells (`prediction_points` = the 200k cell centroids of
  `boundaries.vehicle`). For a surface target that means running ISLA's interior path with the queries on the surface,
  an API question for the model, not for the asset.
- Cost for 690 cases: at about 2 M faces, 36 MB per case, so about 25 GB and about 3.5 k files. QEM on 250-300 M
  triangles needs about 100-150 GB RAM and on the order of 10-20 min per case single-threaded, about 150-250 CPU-h on
  `-p cpu` (one case per job, `--mem=200G`), plus about 690 x 1 min of GPU for validation.
- Validation: on 20 probe cases spanning AoA and geometry, exact distances with the Warp BVH on the *full* mesh
  (277 M triangles fit on a GB300). Then the downstream check: ISLA's wall-distance channels and directions for
  off-surface probe points, computed from the full mesh against the sidecar. Bar: |delta s| / s <= 1e-3 beyond 1e-3 c
  from the wall.
