# BOUNDARY-AUDIT: what the stored case files hold on their boundaries, and ISLA reading the whole boundary

Research lane of the ISLA program, launched 2026-10-23 17:52 EDT by the supervisor (01a13c2c); written 2026-10-23 about
20:10 EDT. Scratch: this folder (git-ignored). LHR folders: `$T/bdaudit/` (data reads) and `$T/kernel/bdaudit/` (GB300
cost), `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`.
Rulings relied on: Peter 2026-10-23 17:08 EDT (canonical ISLA reads every stored piece of the PDE boundary, by name with
the condition type from the dataset); boundary conditions as boundary meshes, full resolution, nothing precomputed;
datasets never modified; orientation flips are recipe transforms. Lengths are in metres unless stated (raw files; the
DrivAerML datapipe's unit is L_ref = 5 m, so a datapipe length times 5 is metres).

## 1. Summary

| dataset (task) | class | what changes |
|---|---|---|
| DrivAerML Volume | **(b)**, changed | `isla_volume.yaml` reads vehicle, no_slip, slip, inlet, outlet (commit `c1e07b7fd`, pushed) |
| DrivAerML Surface | **(b)**, not changed | same files store the four domain meshes; an ISLA surface task predicts at every cell it is given, so adding them needs model or recipe code (section 5) |
| AirfRANS | **(c)** | far field missing from the pdmsh (in the raw release, 200 chords out; the interior is a crop) |
| SHIFT-SUV | **(c)** | only the vehicle (open at the tyre patches) on all 1,991 files; ground, walls, inlet, outlet not stored |
| HiLift | **(c)** | only the wing on 1,798 of 1,800 files; 2 files also store inlet, outlet, symmetry |
| AeroZoo | (a) | ISLA reads all four stored pieces |
| lug | (a) | ISLA reads the whole stored surface, split into its four labels |
| Tensile2d | (a) | ISLA reads the whole stored polyline, split into its four labels |
| DeepJEB++ | (a) | ISLA reads the whole stored surface, split into its three labels |

Findings that change the plan, reported to the supervisor as progress updates (18:45 and 19:39 EDT):
- **DrivAerML ground offset.** On 297 of 484 DrivAerML files, the four stored domain meshes sit 2.4 mm below the CFD
  mesh (section 6). The config reads them as stored; whether to correct the frame is Peter's call.
- **The whole-boundary cost is almost all one defect.** On DrivAerML Volume the extra cost comes from the Warp seed query
  (`signed_distance_field`), not from the 5,054 added cells. Without that query, the whole boundary costs 0.99x the
  body-only step on 0.9.0 and 1.11x on the bundle (section 4).

## 2. What the stored files hold (raw memmaps, read in cpu-short jobs)

Sources: `out/*.jsonl` (one JSON line per case, `harness/audit.py`, summarized by `harness/summarize.py` into
`summary_1.txt` and `summary_meta.txt`). "All" means every case file of the dataset was read for that property.

| dataset | cases read | boundary meshes (name: cell type, cells) | condition-type field | orientation of stored normals | closure |
|---|---|---|---|---|---|
| AirfRANS (`airfrans_pdmsh`) | all 1,000 | airfoil: segments, 843-1,217 | none (cell_data normals, quadrature_measure) | into the fluid: 1,988,656 of 2,000,000 sampled interior nodes on the + side, 8 on the - side, the rest on the wall (all 1,000) | closed polyline, 1 component, 0 open or non-manifold vertices (all 1,000); the domain is not closed: the interior is cropped, no outer boundary stored |
| DrivAerML (`PhysicsNeMo-DrivaerML`, Volume and Surface) | all 484 | vehicle: triangles, 14.6-20.4 M; no_slip 722; slip 2,888; inlet 722; outlet 722 | none: the four domain meshes have no cell data; the names are the condition types | domain meshes into the fluid on all 484 (face-offset winding test, 20 faces per piece per case, 9,680 per piece: winding -1 on the +n side, 0 on the -n side); vehicle into the body (1,840-1,851 of about 1,895 interior nodes nearest it on the - side; 3 cases), which the dataset's FlipBoundaryOrientation reverses | domain union: 1 component, 114 open edges (all on the ground plane, 480 m of non-conforming seams), 0 non-manifold (all 484); with the vehicle, 2 components and 7,425-7,497 open edges (the vehicle meets the ground without shared vertices; 3 cases); interior nodes have winding about -1 about the union (8 per case, 3 cases) |
| SHIFT-SUV (estate, fastback) | names and cell data: all 1,991; geometry: 5 | vehicle: triangles, 4.79-5.33 M | none | into the fluid: 9,751 of 10,000 sampled interior nodes on the + side (5 cases) | vehicle open: 3,190 open edges, 5.75 m long, at z -0.0229 to -0.0201 m (the tyre contact patches), 0 non-manifold (5 cases); no domain pieces stored |
| HiLift (`PhysicsNeMo-HighLiftAeroML`) | names and cell data: all 1,800; geometry: 1 | vehicle: triangles, 248-303 M; on geo_LHC001_AoA_10 and _12 only: inlet 1,952, outlet 1,952, symmetry 11,030-11,031 | none on the domain pieces | domain pieces into the fluid on LHC001_AoA_10 (50 faces each: winding -1 on the +n side); vehicle not checked (no interior points) | on LHC001_AoA_10: the union of inlet, outlet and symmetry has 10,935 open edges, and all 10,924 of their vertices are vehicle vertices (the wing root on the symmetry plane) |
| DeepJEB++ (`deepjebpp_pdmsh`) | names and cell data: all 61,440; geometry: 300 spread | surface: triangles, 25,184-50,154 | bc_fixed, bc_loaded (one-hot), bc_load (vector) | out of the solid: winding 1 on the -n side at 8,996 of 9,000 sampled faces (300 cases) | closed manifold, 0 open, 0 non-manifold, 0 same-direction edges (300) |
| Tensile2d (`plaid_tensile2d_pdmsh`) | all 500 | outer: segments, 424-496 | bc_loaded, bc_fixed, bc_pinned (one-hot), bc_pressure | into the solid: 949,373 of 1,000,000 sampled nodes on the + side, 0 on the - side (all 500) | closed polyline, 1 component (all 500) |
| lug (`ginot_lug_pdmsh`) | all 3,000 | surface: triangles, 4,368-8,568 | bc_fixed, bc_hole, bc_loaded (one-hot), bc_pressure | out of the solid: winding 1 on the -n side at 60,000 of 60,000 sampled faces (all 3,000) | closed manifold, genus 1 (Euler 0), 0 open, 0 non-manifold (all 3,000) |
| AeroZoo (`aerozoo_v1.2_shuf_pdmsh`) | names and cell data: all 3,816; geometry: 40 spread | inlet, outlet, no_slip, slip: triangles (slip empty on 461 of 3,816; one file, shard-002 case ...2857-acd3d5a9db63, has an extra empty boundary `inle`) | prescribed.dirichlet.* (inlet), prescribed.dirichlet.gauge_pressure and reverse_flow.* (outlet), prescribed.dirichlet.velocity (no_slip); names are the types | out of the fluid (sampled interior nodes on the - side for all four pieces; face-offset winding 1 on the -n side), which the dataset's FlipBoundaryOrientation reverses | union open edges 0 on 8 of 40, non-manifold edges 0 on 24 of 40; 800 sampled interior nodes have winding 1.0 about the union (40 cases) |

The three solid datasets do not share one orientation convention: Tensile2d's normals point into the solid, the lug's and
DeepJEB++'s out of it. ISLA's sign is relative and `isla_lug.yaml` documents the lug's sign, so this is noted, not changed.

## 3. What each config reads

| dataset | ISLA config: boundary meshes | baselines (see section 8 and `BASELINE_PARITY.md`) |
|---|---|---|
| DrivAerML Volume | was: vehicle; **now: vehicle, no_slip, slip, inlet, outlet** | Transolver, GeoTransolver: interior points + sdf to the vehicle STL; GLOBE: vehicle |
| DrivAerML Surface | vehicle (the reader loads `boundaries/vehicle` only) | vehicle cell centroids and normals; GLOBE: vehicle |
| SHIFT-SUV, HiLift | vehicle | as DrivAerML Surface |
| AirfRANS | airfoil | interior points + the stored sdf |
| AeroZoo | inlet, outlet, no_slip, slip | GeoTransolver: all four, merged; Transolver: sdf to no_slip |
| lug | fixed, hole, loaded, free (all cells of `surface`) | Transolver: stored sdf |
| Tensile2d | pinned, loaded, fixed, free (all cells of `outer`) | Transolver: nodal tags, stored sdf |
| DeepJEB++ | fixed, free, loaded (all cells of `surface`) | Transolver: per-cell labels |

## 4. Config change, length unit, tokens and cost (DrivAerML Volume)

**Commit.** Branch `psharpe/isla-whole-boundary` (worktree `/home/psharpe/gh/physicsnemo-isla-whole-boundary`, from
`origin/psharpe/isla` b5fee74b3), commit `c1e07b7fd` "Recipe: isla_volume reads the whole boundary of the DrivAerML Volume
domain ...", `-s`, hooks on (all passed), pushed with a plain push, not merged.
- `conf/model/isla_volume.yaml`: `forward_kwargs.boundary_meshes` adds `no_slip`, `slip`, `inlet` and `outlet`, and
  `model.boundary_schemas` gives each `{}`, its condition named as in the file (no field: the files store none). The
  domain meshes are read as stored, since they point into the fluid on all 484 files. Starting point: FLOOR-DRV's
  `isla_volume_floor.yaml` (564f803cc); the composed model block is the same.
- There is no second config: `isla_volume_floor.yaml` is on `psharpe/isla-floor-drv` only. If that branch merges, the
  file duplicates the reference and should go.
- No dataset file changes: `drivaer_ml_volume_reference*.yaml` already keep the in-file boundaries
  (`drop_in_file_boundaries: false`) and translate and scale every boundary with the interior.
- `tests/test_isla_configs.py`: the volume test builds the five boundaries and runs collate, forward and loss on them,
  and asserts the five names and their empty schemas. The mesh-contract test asserts the five names for `isla_volume`.
  The dataset test asserts that the files keep the in-file boundaries and drop only the STL.
- Tests (CPU, `nice -n 19`): `test_isla_configs`, `test_utils`, `test_forward_kwargs`, `test_synthetic_configs`,
  `test_aerozoo_configs` and `test_airfrans_configs` ran, 254 passed (`pytest_recipe_3.log`). The full-size forward and
  training pass ran on LHR (steer 3). In job 360028 (row `branch_isla_volume`), the branch's own `isla_volume`
  (snapshots `$T/code_bdaudit` + `$T/recipe_bdaudit`, both `c1e07b7fd`) took 20 training steps on real DrivAerML Volume
  cases. Losses were finite, the resolved config lists the five boundary meshes, F = 17,249,814, the step took 2,896 ms,
  and peak memory was 140.6 GB. The model code did not change, so the ISLA test directory was not rerun.

**Length unit, Reynolds input and near-wall channel** (`out/drvwall_c*.jsonl`, `harness/drv_walldist.py`, 5 cases:
run_1, 188, 279, 373, 462; L over all 484 for the domain pieces alone from `drvdom.jsonl`).

| quantity | body only (vehicle) | whole boundary (five pieces) |
|---|---|---|
| ISLA's L (measure-weighted RMS pairwise distance, cells with their second moments, full resolution) | 1.99-2.17 m (median 2.11) | 59.50-59.51 m (domain pieces alone: 59.55 m on all 484) |
| boundary measure | vehicle 31.9-36.0 m^2 | + 17,120 m^2 of domain (vehicle 0.19-0.21 %) |
| 1e-6 L, below which asinh(d / (1e-6 L)) is linear | 2.0-2.2 um | 59.5 um |
| smallest positive wall distance of the stored interior nodes | to the vehicle: 58.5-60.6 um (exact point-to-triangle, 5 cases; 7.6-9.1 M nodes per case are vehicle vertices, d = 0) | to the domain box: 44.0-45.3 um on the 187-file group (run_1, run_188; 1.56-1.58 M nodes on the box, d = 0); 2.40 mm on the 297-file group (run_279, 373, 462; about 5,140 nodes per case outside the stored box, section 6) |
| channel at the first node off the car (d about 60 um) | asinh(28) = 4.0, the logarithmic part | asinh(1.0) = 0.89, the linear part |
| ISLA's Reynolds input | none: `isla_volume.yaml` declares no viscosity and no `reynolds`; the files store U_inf = 38.889 m/s and nu = 1.0 (all 484; a placeholder, not air) | none |
| Re = U L / nu if it were declared, file nu = 1.0 | 77-85 | 2,314 |
| Re with air at 1.5e-5 m^2/s (SHIFT-SUV's stored nu; not DrivAerML's) | 5.2-5.6e6, z = log10(Re/1e6) = 0.71-0.75 | 1.54e8, z = 2.19 |
| shift of z, for any nu | | +1.45 = log10(59.5 / 2.11) |
| boundary tokens (200,000, water-filled by cell count) | vehicle 200,000 | vehicle 194,946; no_slip 722, slip 2,888, inlet 722, outlet 722 (kept whole) |

**Cost on GB300** (`cost_table_isla.md`, `cost_table_dtt.md`, `cost_seed_table.md`; harness `harness/cost_gb.sbatch`,
`cost_dtt.sbatch`, `cost_seed.sbatch` with DV-COST2's `timed_train.py` and my `timed_seed_train.py`, which also times
the seed query inside each call). Setup as DV-COST2's `final_gb.sbatch`: recipe `train.py`, REF-090 settings, bfloat16,
eager (`compile=false`, `compile_blocks=true`), 200,000 interior points, seed 42, the same cases in the same order in
every arm. Body only = `isla_volume` of `recipe_r090_fd`; whole = `isla_volume_floor` of `recipe_r090_fd`, the same five
meshes as `c1e07b7fd`, whose own run gives the same step (2,896 ms against 2,897). Codes: 0.9.0 = `code_cmpaf_fix`,
bundle = `code_dvcost2a` (f13eb5ce0). Reported: medians of steps and calls 6-20, mean of two passes with the GPUs
rotated and the order of body and whole reversed. ISLA-DTT = `+model.query_independent=true`.

| code | model | geometry call, body / whole | step, body / whole | ratio | added | Warp seed in the call, body / whole | step without the seed, body / whole | ratio | peak memory, body / whole |
|---|---|---|---|---|---|---|---|---|---|
| 0.9.0 | ISLA | 98 / 2,421 ms | 570 / 2,897 ms | **5.08x** | +2,327 ms | 55 / 2,387 ms | 518 / 513 ms | **0.99x** | 140.5 / 140.6 GB |
| 0.9.0 | ISLA-DTT | 97 / 2,452 ms | 1,728 / 4,242 ms | **2.45x** | +2,514 ms | (55 / 2,387)* | 1,673 / 1,855 ms* | 1.11x* | 200.4 / 209.9 GB |
| bundle | ISLA | 364 / 2,768 ms | 850 / 3,309 ms | **3.89x** | +2,459 ms | 55 / 2,410 ms | 811 / 896 ms | **1.11x** | 140.8 / 141.1 GB |
| bundle | ISLA-DTT | 365 / 2,814 ms | 1,986 / 4,619 ms | **2.33x** | +2,634 ms | (55 / 2,410)* | 1,931 / 2,209 ms* | 1.14x* | 200.9 / 211.2 GB |

\* The seed was timed only in the ISLA runs (360725). The ISLA-DTT "without the seed" columns subtract the ISLA seed of
the same code, since the geometry call is the same function (its total matches ISLA's within 1-2 %). The ISLA rows of
360725 reproduce 360028 within 1 %.

- **The seed query is the cost.** In training, `signed_distance_field` takes 2.39-2.42 s of the 2.43-2.80 s
  whole-boundary call (55 ms body only). Isolated on GEO-FAST's dump of the same case it takes 2.01-2.02 s (LHR
  359447, 359546), and the gap to the in-training 2.4 s was not traced. Without it, the five pieces add -6 ms per ISLA
  step on 0.9.0 and +85 ms on the bundle, whose ball traversal grows on the larger domain (156 to 235 ms isolated,
  GEO-FAST).
- **After GEO-FAST's seed commit** (approved, not yet on origin): its float32 seed on ISLA's cell tree, bitwise in every
  output (LHR 360424, `scratch/isla-geo-fast-2026-10-23/lhr/out_p1f`), takes 27 ms on the vehicle and 21 ms on the whole
  boundary. Adding those to the step without the seed projects the bundle's ISLA step at about 0.84 s body only and
  0.92 s whole, **about 1.09x**, and ISLA-DTT at about 1.96 and 2.23 s, **about 1.14x**. This is arithmetic on
  measured parts, not a measurement of that commit. 0.9.0 has the same Warp seed (`_closest_cell`) and stays at 5.1x
  unless the fix is ported.
- FLOOR-DRV measured 1.58x in float32 (3.33 against 5.27 s). The added time is about the same (+1.9 s there,
  +2.3-2.6 s here); the ratio is larger here because bfloat16 shortens the rest of the step.
- GPU use: 360028 (9:37), 360541 (7:20), 360725 (7:22), each one 4-GPU node on short QoS: **1.62 GPU-h** of the 6 GPU-h cap.

## 5. Class (c), and the surface tasks: what is missing and what supplying it would take

No dataset was built or modified.

- **AirfRANS** (all 1,000 cases). The pdmsh stores only the airfoil, and its interior is a crop: nodes in about
  [-2.16, 4.23] x [-1.63, 1.62] chords (5 cases, `out/afbox.jsonl`). The raw release's `_internal.vtu` has the same
  bounds, so the crop is in the release itself. The crop's edges are not a boundary of the PDE (the flow crosses them),
  and no condition is stored on them.
  - The far field is in the raw release (`users/psharpe/datasets/airfrans/Dataset/<case>/<case>_freestream.vtp`, all
    1,000 cases present). It is 1,664-1,763 line cells on the square +-200 chords, with U, p and nut point data (5 cases
    read, `out/afraw.jsonl`, job 360548).
  - Supplying it means adding the freestream curve to each pdmsh as a named boundary, whose type (inflow or outflow
    freestream, Dirichlet from U_inf) would come from the AirfRANS setup.
  - ISLA's L would then be set by the far field: about 327 chords for the +-200 square, against 0.41-0.44 now.
  - The interior would remain a crop 200 chords inside that boundary. Whether a cropped interior with a distant far
    field is acceptable, or whether the crop box itself needs a condition (which could only come from the solution), is a
    judgment for Peter.
- **SHIFT-SUV** (estate 996 and fastback 995 files). Only `vehicle` is stored, and it is open at the tyre contact
  patches.
  - The interior nodes fill the box x -40 to 80 m, y -20 to 20 m, z -0.024 to 19.976 m (5 cases, `out/suvbox.jsonl`).
    About 245,000 nodes lie on z = -0.024, the ground, 1.1 mm below the vehicle's lowest point.
  - Missing: ground, two sides, roof, inlet and outlet. Their geometry is the box's faces, recoverable from the interior's
    bounding planes. Their condition types are not in the case files (`units.json` holds only global data), so they
    would have to come from the SHIFT-SUV solver setup.
  - The case folders also hold `merged_surfaces.stl.pmsh` and `merged_surfaces_filled.stl.pmsh`, not inspected.
- **HiLift** (1,800 files). 1,798 store only the wing; geo_LHC001_AoA_10 and _12 (both in the medium manifest) also
  store inlet, outlet and symmetry, without cell data.
  - On those 2, the three pieces meet the wing root on the symmetry plane (section 2).
  - Missing on 1,798: the far field (inlet, outlet) and the symmetry plane. The case folders hold only the pdmsh (and the
    STL on LHC001); where the release keeps the other cases' domain meshes I could not tell from LHR.
- **The surface tasks: DrivAerML Surface (class b), SHIFT-SUV and HiLift.** An ISLA surface task (`interior=False`)
  predicts at every cell of every boundary it is given (model.py, the `forward` and `boundary_tokens` docstrings). So
  domain pieces cannot be added by config: the model would predict on cells that have no targets, and the surface
  readers (`MeshReaderWithGlobalData`, pattern `.../boundaries/vehicle`) load the vehicle only.
  - Reading the four DrivAerML domain meshes on the Surface task would need one of two code changes: boundaries read but
    not predicted on in a surface task (model change), or the surface task posed as an interior task whose prediction
    points are the vehicle's cell centroids (recipe and dataset change). This is a design choice for Peter; nothing was
    changed.

## 6. DrivAerML: the stored domain meshes are 2.4 mm low on 297 of 484 files

From `drvdom.jsonl` (all 484) and `harness/interior_bbox.py` on 8 files (4 per group, jobs 360408 and 360409,
`out/drvbox_*.jsonl`):
- On 187 files the ground (no_slip and the low faces of slip, inlet and outlet) is at z = -0.31760 m and the roof at
  19.68240 m.
- On 297 files they are at -0.32000 and 19.68000. On every one of the 8 checked, the interior nodes span -0.31760 to
  19.68240 and the vehicle's lowest point is -0.31760.
- So on the 297, the stored ground is 2.4 mm below the wheels and the lowest CFD nodes (about 1.6 M nodes lie on
  z = -0.3176), and about 5,140 nodes per file lie above the stored roof (signed distance -2.4 mm).
- In ISLA's whole-boundary unit the offset is 4e-5 L = 40 x (1e-6 L), and it is 54 x the smallest positive wall distance
  on the matching files.
- run_171 (FLOOR-DRV's case) is in the matching group.

The config reads the meshes as stored. A per-file translation would be a correction of the data's frame, which is Peter's
call (reported 19:39 EDT).

## 7. Baseline parity, facts only

See `BASELINE_PARITY.md` (read-only survey by a child scout, with file:line citations) for every dataset. For the
changed dataset, DrivAerML Volume, checked against CAP-AUDIT's resolved configs
(`$T/cap/runs/cap_drv_{t,gt}_s42/resolved_config.yaml`):
- **What they see now.** Transolver (`transolver_volume`): `embedding = [interior.points, sdf, sdf_normals]`, `fx =
  U_inf_dir`. GeoTransolver (`geotransolver_volume`): `geometry = interior.points` (volume points, not a boundary), the
  same local embedding. The sdf is computed from the vehicle STL (`ComputeSDFFromBoundary(stl_geometry)`). Both set
  `dataset_reader_overrides.drop_in_file_boundaries: true`, so the domain meshes are not even loaded. They see the
  ground, walls, inlet and outlet only implicitly, through coordinates in the dataset frame: z is the height above the
  ground, up to CenterMesh's per-draw shift and the 1/5 m scale, and the sampled interior fills the box. GLOBE
  (`globe_volume`): `boundary_meshes: {vehicle}`; the domain meshes are loaded but not passed.
- **What the released code could take from the recipe config without code changes** (static reading, nothing run):
  - GLOBE: the five meshes by name, in `forward_kwargs.boundary_meshes` and `model.boundary_source_schemas` (its
    `forward` takes a dict of named meshes).
  - GeoTransolver: one `geometry` tensor, so the five would have to be merged into one mesh. `MergeBoundaries` needs a
    non-empty shared cell field, which the DrivAerML domain meshes lack. `HarmonizeBoundaryFields` with `one_hot`
    supplies one, as on AeroZoo. Both are dataset transforms a model config can append (`dataset_append_transforms`)
    once `drop_in_file_boundaries` is false. The merged vehicle (14.6-20.4 M cells) would also need subsampling for
    geometry tokens; AeroZoo subsamples to 50,000.
  - Transolver: per-point channels only. It can take the distance to the union or to each piece through further
    `ComputeSDFFromBoundary` entries (`boundary_name` is one boundary, so a union needs `MergeBoundaries` first), with
    `embedding_dim` raised to match.
  - The scout also found statically that `drop_in_file_boundaries: true` with the dataset's
    `FlipBoundaryOrientation(vehicle)` should raise. CAP-AUDIT's Transolver and GeoTransolver runs on this dataset with
    that override completed, so it does not in practice; not investigated.

## 8. Jobs and files

All jobs are on LHR, named `isla.bdaudit.*`. The registry row `^isla\.bdaudit\.` was added at 18:10 EDT and removed at
20:04 EDT (backups `.ACTIVE_EXPERIMENTS.bak*`, cmp-checked). Details are in `JOBS.md`.
- cpu-short: 359338-359341, 359406-359409 (359406-359408 hit the 20-minute limit and were split and rerun), 360023-360026,
  360408-360411, 360548, 360557, 360696.
- GB300 short QoS: 360028, 360541, 360725.
- Files kept: `out/` (JSON lines of every read, about 11 MB), `out/cost/` (the cost jobs' logs, metrics and resolved
  configs), `harness/`, `dsconf/` (the cluster's dataset configs), `summary_*.txt`, `cost_*_table.md`, `pytest_recipe_*.log`,
  `BASELINE_PARITY.md`.
- On LHR: snapshots `$T/code_bdaudit` and `$T/recipe_bdaudit` (read-only, about 60 MB), `$T/bdaudit/` and
  `$T/kernel/bdaudit/`.
- Worktree: `/home/psharpe/gh/physicsnemo-isla-whole-boundary` (its branch is pushed).
- The one local copy of a case (a Tensile2d sample, 700 KB) was deleted after its check.
