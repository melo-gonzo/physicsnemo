# SETUP-RESEARCH: the CFD setup of SHIFT-SUV and HiLift, where their missing boundary pieces are, and what supplying them costs

Research lane of the ISLA program, launched 2026-10-23 about 20:25 EDT by the supervisor (01a13c2c); written 2026-10-23
about 20:40 EDT. Read-only: no dataset, GitHub, or message writes. Scratch: this folder. LHR folder:
`$T/setupres/` (`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`).
Dataset roots on LHR: `D=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets`,
`$D/PhysicsNeMo-ShiftSUV/{estate,fastback}/run_*/` and `$D/PhysicsNeMo-HighLiftAeroML/geo_LHC*/`.

Rulings relied on: Peter 2026-10-23 17:08 EDT (canonical ISLA reads the whole boundary of the PDE domain, condition types
from the dataset, never invented; datasets never modified; a gap is reported, not filled). LANE_RULES (read-only lane
limits; heavy reads in cpu-short jobs).

## 0. The finding that frames the rest: the domain pieces we already read are synthesized

- **DrivAerML.** The four domain meshes that BOUNDARY-AUDIT found in every DrivAerML file (inlet, outlet, slip, no_slip;
  722 / 722 / 2,888 / 722 triangles) are not CFD surfaces. physicsnemo-curator `examples/cae/drivaerml/main.py`
  (origin/main, PR #58, commit 60ed919, Corey Adams, 2026-06-08) runs
  `BoundaryInjectionFilter(BoxTunnelBoundaries(x_min=-40, x_max=80, y_min=-22, y_max=22, z_height=20, x_bl=-2.339))`, whose
  comment reads "Synthesize the rectangular wind-tunnel boundaries (inlet/outlet/slip/no_slip) the dataset ships without,
  inferring the floor height per sample from the vehicle surface". Each box face is a 20 x 20 grid (`n_per_side=(20, 20)`),
  so 2 x 19^2 = 722 triangles per face, and `slip` merges four faces (2,888). The floor height is
  `z_floor_from_boundary(vehicle) = vehicle.points[:, 2].min()` (`src/physicsnemo_curator/domains/mesh/boundaries/_geometry.py`).
  Read with `git -C ~/gh/physicsnemo-curator show origin/main:<path>`. This method probably also explains BOUNDARY-AUDIT's 2.4 mm
  ground offset on 297 DrivAerML files; I did not check that (another lane, DRV-GROUND, is registered on it).
- **HiLift, the 2 files with domain pieces.** These are synthesized too (section 2.2).
- **The generator module is written for both datasets.** The `generators.py` docstring lists `BoxTunnelBoundaries` as
  "rectangular wind tunnel (DrivAerML, ShiftSUV)" and `HemisphereBoundaries` as "hemispherical open-road domain with a
  symmetry plane (HighLiftAeroML)". There is no SHIFT-SUV example in curator (`git ls-tree origin/main examples/cae/`:
  ahmedml, crashsim, drivaerml, drop, vti_grids).

So "condition types from the dataset" already rests, on DrivAerML, on geometry built from the paper's documented domain
and named by the setup. Option (ii) or (iii) below would give SHIFT-SUV and HiLift pieces of the same kind. I sent this
to the supervisor as a progress update at 20:30 EDT.

## 1. SHIFT-SUV

### 1.1 Documented setup

| item | documented | source |
|---|---|---|
| solver | Luminary Cloud GPU-native finite-volume solver, 2nd order in space and time; transient DDES on a Spalart-Allmaras backbone (shear-layer-adapted length scale, VTM grey-area mitigation, shielding function); about 45 M-cell hex-dominant ANSA meshes, one per variant; 20,000 iterations, full-scale dt 1.78e-4 s, averaged over the final 15,000 | [AB-UPT report, arXiv 2510.15808 sec. 3.2.1](https://arxiv.org/html/2510.15808); [HF card](https://huggingface.co/datasets/luminary-shift/SUV); `croissant/shift-suv.json` `convergence_criterion` in [shift-neurips-2026](https://github.com/shift-neurips-2026/shift-neurips-2026) (commit d1ec365) |
| vehicle | no-slip walls on the vehicle body | AB-UPT sec. 3.2.1 ("The rest of the vehicle surfaces are modeled as no-slip wall boundaries"); croissant `boundary_conditions` |
| wheels | rotating-wall condition | AB-UPT sec. 3.2.1; croissant; [GeoTransolver, arXiv 2512.20399 sec. 4.1.2](https://arxiv.org/html/2512.20399v1) |
| ground | translating floor ("rolling road", "modeling road-vehicle relative motion"). Belt layout and belt speed for SHIFT-SUV are not stated. The SHIFT-Truck card says its "solver setup, boundary conditions and rolling-road floor system match the SHIFT-SUV pipeline" and gives a 5-belt rolling road moving at the freestream speed for the truck | AB-UPT sec. 3.2.1; croissant; GeoTransolver sec. 4.1.2; [SHIFT-Truck card](https://huggingface.co/datasets/luminary-shift/Truck) |
| inlet | uniform inflow, 30 m/s along x | AB-UPT; croissant `freestream_speed`; GeoTransolver; LHR `units.json` (U_inf 30.0). **But** the HF card's Notes (changelog 8/25/2026) give full-scale inflow 31.2928 m/s |
| side walls, roof | **not documented** in any source found | (the Truck card names "tunnel walls, inlet, outlet and floor patches" as removed from its export, so the SUV pipeline has such patches) |
| outlet | **not documented** | |
| domain shape and extent | **not documented** | |
| fluid | air, 1.225 kg/m^3, mu 1.81e-5 Pa s, 293.15 K; yaw 0 | HF card Notes |

Sources searched without finding the domain, wall, or outlet conditions: the HF card, the AB-UPT report, GeoTransolver,
the shift-neurips-2026 DATASHEET.md, DATASET.md and croissant files (`rg -i 'belt|rolling|ground|slip|outlet|inlet|tunnel|domain|symmetry|wall'`
over the clone), Luminary's SHIFT introduction blog
([link](https://luminary.ai/resources/introducing-luminary-shift-models-a-suite-of-physics-ai-foundation-models-to-transform-engineering-design/)),
and Luminary's one-page DrivAer V&V summary (`src/luminary_drivaer_vv_2024.pdf`). The only remaining source would be Luminary
themselves (shift@luminarycloud.com, from the HF card). Writing to them takes another person's time, so it needs Peter's approval.

### 1.2 What the stored files hold, and whether the interior matches a domain

There is no documented extent to compare against. The stored interior defines the box exactly:

| quantity | value | source |
|---|---|---|
| interior bounding box | x -40 to 80, y -20 to 20, z -0.024 to 19.976 m (120 x 40 x 20 m) on all 7 files read (5 estate from BOUNDARY-AUDIT, plus estate and fastback run_00001) | `out/suv.jsonl` (`suv_faces`, job 361150); `../isla-boundary-audit-2026-10-23/out/suvbox.jsonl` |
| CFD nodes on each bounding plane (estate / fastback run_00001) | x = -40: 3,792 / 3,792; x = 80: 3,792 / 3,792; y = -20: 11,280 / 11,459; y = 20: 11,657 / 11,454; z = -0.024: 244,496 / 244,426; z = 19.976: 19,210 / 18,873 | `out/suv.jsonl` |
| interior | 50.4 M / 49.9 M nodes with pressure_average and velocity_average point data; cells shape [0, 1], so a point cloud with no volume cells | `$D/PhysicsNeMo-ShiftSUV/estate/run_00001/domain.pdmsh/_tensordict/interior/_tensordict/meta.json` (cat over ssh) |
| vehicle in the box | x -0.84 to 3.79 m (4.63 m long), y +-1.024 m, top 1.52-1.68 m, lowest z -0.02285 m | `suvbox.jsonl`, `out/suv.jsonl` |

Nodes lie on all six planes, so the interior is the whole CFD volume, and its six bounding planes are the faces of the
domain. From the box: the inlet is 39.2 m (8.5 vehicle lengths) ahead of the nose, the outlet 76.2 m (16.5 lengths) behind
the tail, the side walls 19.0 m from the vehicle, and the roof about 18.3 m above it. Frontal blockage is 2.464 m^2 / 800
m^2 = 0.31 % (reference area from the HF card).
The box is not DrivAerML's: y is +-20 m here against +-22 m in curator's DrivAerML constants. Curator's
`z_floor_from_boundary` would put the SHIFT-SUV floor at the vehicle's lowest point, -0.02285 m, which is 1.15 mm above the
CFD ground at -0.024 m.

### 1.3 Where the missing geometry is

- **`merged_surfaces.stl.pmsh`.** The vehicle again: the same point set as `boundaries/vehicle` (2,520,898 / 2,483,877
  points), the same 5,038,762 / 4,964,720 triangles, the same 3,190 open edges, and no cell data (`suv_stl`, `out/suv.jsonl`;
  estate / fastback run_00001).
- **`merged_surfaces_filled.stl.pmsh`.** The same points plus 3,091 / 3,087 triangles. They are caps at z -0.02285 to
  -0.0201 m, centroids at x -0.067 to 2.853 m and |y| up to 0.87 m: the four tyre cuts. Afterwards 73 / 81 open edges and 10
  non-manifold edges remain. The HF card calls these "patches filling the holes at the tire/floor intersection (fully
  closing the vehicle)". They are Luminary's fill and not a CFD boundary: CFD nodes lie in the gap between the caps and
  the ground. Within 5 mm (in x, y) of the cap centroids, 19,517 / 19,608 interior nodes lie at z -0.02391 to -0.0195 m
  above the ground, and 446 / 455 on it (`suv_gap`, `out/p2_console.txt`, job 361188). How the CFD closed the tyres is not
  stored.
- **No other file on LHR.** The case folders hold only `domain.pdmsh` and the two STL pmsh (`ls` over ssh); the dataset
  root holds `units.json` with global data only. No raw SHIFT-SUV copy was found under `users/psharpe/datasets/` or by name
  in the other users' folders (`find -maxdepth 3 -iname '*suv*'`).
- **Public release (HF, gated by terms).** Each run has `DRAG/LIFT(_COEFFICIENT).csv`, `merged_surfaces.stl`,
  `merged_surfaces_filled.stl`, `merged_surfaces.vtp` (191 MB) and `merged_volumes.vtu` (3.25 GB), plus `residuals.csv` and
  a png (`curl https://huggingface.co/api/datasets/luminary-shift/SUV/tree/main/AeroSUV_full_scale_estate_transient/run_00001`).
  `merged_volumes.vtu` is the full volume mesh with cells ("VTK UnstructuredGrid", connectivity, faces; DATASET.md), so its
  outer faces are the CFD domain boundary, including whatever closes the tyres. It has no patch names, and no file states
  conditions.

## 2. HiLift (HiLiftAeroML)

### 2.1 Documented setup ([arXiv 2605.19565v2](https://arxiv.org/abs/2605.19565v2), sec. 2.3, App. A.2.1, Table 2; [HF card](https://huggingface.co/datasets/nvidia/HiLiftAeroML))

| item | documented |
|---|---|
| solver | Cadence Fidelity Charles: explicit, unstructured finite-volume compressible Navier-Stokes, 2nd order in space and 3rd order in time; WMLES with dynamic Smagorinsky and an equilibrium wall model; Fidelity Stitch Voronoi meshes of 300-500 M cells |
| domain | "The semi-span airframe is mounted on the symmetry plane of a hemispherical domain with a radius of ≈75 × MAC"; MAC c_ref = 275.80 in, so about 20,685 in (525 m) |
| inlet | "A uniform plug flow is specified at the inlet (the forward half of the hemisphere)", a uniform freestream at the case angle of attack |
| outlet | "A characteristic non-reflecting boundary condition with a specified outlet pressure is applied to the rear half" (Poinsot & Lele 1992) |
| symmetry plane | "no-stress boundary condition" |
| walls | all aircraft surfaces viscous, equilibrium wall model |
| flow | M 0.2, Re_MAC 1.6e6, AoA 4-22 deg; English units (slug, in, s, Rankine) |

**The documented radius does not match the stored interior.** On the 9 intact HiLift files that store an interior (below),
the interior spans x and z +-24,533.42 in and y 0.53 to 24,446 in. Radii reach 24,994 in, which is 90.6 MAC
(`hl_interior`, `out/hl.jsonl`, job 361149). The radial point density is smooth from 70 to 91 MAC, with no edge at 75 MAC:
3,621-5,748 points per 1-MAC bin. 9,284 points lie beyond 24,533.42 in. The histograms are identical on LHC001_AoA_4 and
LHC002_AoA_16 (`hl_radius`, `out/p2_console.txt`, job 361188), so the far-field grid is shared by all cases. The stored
fluid region therefore reaches about 88.9-90.6 MAC (623-635 m), not about 75 MAC. The volume is an octree export of the
Voronoi grid (paper App. B.3), so its outermost points sit up to about 2 % outside a sphere. Peter is an author of the paper
and can settle which radius is right.

### 2.2 What is stored, and where the 2 files with domain pieces came from

- **1,800 files.** All store `boundaries/vehicle` (248-303 M triangles; BOUNDARY-AUDIT) with `global_data` L_ref = 275.8 in
  (the MAC) on all 1,800 (`../isla-boundary-audit-2026-10-23/out/hilift_meta.jsonl`). The dataset root's `units.json`
  says L_ref = 1156.75 in (the half-span); that is a documentation mismatch, noted only.
- **Interiors.** 11 of 1,800 store an interior: LHC001_AoA_4/8/14/20/22 and LHC002_AoA_10/12/14/16/18/20, with 225-248 M
  points, coordinates only and no point data. Two of these are truncated copies: LHC002_AoA_10 has 38.9 MB of 2.78 GB and
  LHC002_AoA_20 has 1.12 GB of 2.81 GB, both written 2026-08-17 13:10 (`out/hl_sizes.txt`, `harness/hl_sizes.sh`). The
  surface task does not read interiors, so this does not affect current runs.
- **geo_LHC001_AoA_10 and _12: inlet, outlet and symmetry, synthesized.** The dataset's
  `.psnc/interactive/46a2b1bb6ff75d13.db` (a physicsnemo-curator run database, read read-only with sqlite3 on the login
  node) records one pipeline run, started 2026-06-10T19:53Z. Its `run_dir` is
  `/lustre/fs1/.../users/coreya/workdir/physicsnemo-curator`, its output `/lustre/fs1/.../datasets/PhysicsNeMo-HighLiftAeroML`,
  and it read `HighLiftAeroMLSource` ("volume/boundary VTU pairs"). Its steps: `TauWallVectorFilter`, `DropZeroPressureFilter`,
  `PointDataToCellDataFilter`, then `BoundaryInjectionFilter(generator=HemisphereBoundaries)`
  ("Synthesize and inject CFD-domain outer boundaries"), then `RandomPermutationFilter(seed=42)`. It completed idx 0 and 1,
  the two files, in 983 s and 850 s with a 41.6-42.5 GB peak; the injection stage took 65 s. Its log says
  "Injected 3 boundary(ies) ['inlet', 'outlet', 'symmetry']". The files' inner `meta.json` dates (2026-06-10 13:26-13:27
  PDT) match the run. The other cases' files date from April-May 2026.
- **The synthesized geometry** (`hl_domain`, `out/hl.jsonl`):
  - inlet and outlet: 1,024 points and 1,952 triangles each, on a sphere of radius 24,533.42 in (88.95 MAC). That is
    curator's `radius_from_interior`, max(|x|, y, |z|) of the interior, so the interior must have been present when the files
    were written. Neither file holds an interior on LHR now (no `interior` folder), although the DomainMesh `meta.json`
    lists one.
  - the inlet/outlet split follows each file's `U_inf` (the case's AoA).
  - symmetry: an 11,030-11,031-triangle disk at y = 0, out to the same radius, with a hole at the fuselage and wing-root
    silhouette (inner radius 204.7 in).
  - no cell data on any of the three (BOUNDARY-AUDIT).
  - the vehicle spans x 92.5 to 2,625.9 in, y 0 to 1,159.9 in, z 17.7 to 728.5 in, so the far field is about 9.7 aircraft
    lengths away.
- **Public release.** Each case has `boundary_<case>.vtu.tgz` (12.7 GB), `volume_<case>.vtu.tgz` (24.0 GB), the STL, the
  STP, `boundary_dual_area_*.npy`, CSVs and pngs (`curl https://huggingface.co/api/datasets/nvidia/HiLiftAeroML/tree/main/geo_LHC001_AoA_10`).
  The paper describes the boundary VTU as "the complete, time-averaged flow field on the aircraft surface" (App. F.7), so
  it holds no far field or symmetry plane (I did not open it). The volume is an octree export (App. B.3), whose outer faces
  are a staircase and not the hemisphere. No HiLift raw files are on LHR. The 2-file run read them on `/lustre/fs1`, where
  `PhysicsNeMo-HighLiftAeroML-medium/.COPY_DONE` says it was "written ... on oci-aga-slurm-1-dc-02"; I did not check that
  cluster.

## 3. Options, cost, and recommendation

Common to every option: SHIFT-SUV and HiLift are surface tasks in the recipe. `shift_suv_surface.yaml` and
`highlift_surface.yaml` read the pattern `.../boundaries/vehicle` with `MeshReaderWithGlobalData` and end with
`MeshToDomainMesh` (`../isla-boundary-audit-2026-10-23/dsconf/`). BOUNDARY-AUDIT section 5 found that an ISLA surface task
predicts at every cell of every boundary it is given. So any domain piece also needs the change that lets a surface task
read boundaries it does not predict on, which is Peter's design choice (audit section 5). That is about 1-2 agent-days in
the model or recipe, plus tests, and is not counted again below. The reader also subsamples the vehicle
(`subsample_n_cells`) before any transform, and the augmentations (rotation about z, translation, `CenterMesh`) must move
the pieces with the vehicle. So the reader and transforms would carry a DomainMesh, not a Mesh: about 0.5-1 day, needed
for (ii) and (iii).

| option | SHIFT-SUV | HiLift |
|---|---|---|
| **(i) read existing files** | **Not possible.** No file holds a domain piece; the interior has no cells. The filled STL could replace the open vehicle (config only, about 0.5 day), but it is Luminary's fill, not the CFD surface, and it supplies no ground, walls, inlet or outlet. | **Not possible beyond 2 of 1,800 files**, and those pieces are synthesized (section 2.2). Reading them takes a config change, about 0.5 day, and gains nothing for training or evaluation. |
| **(ii) recipe transform from bounding planes and documented conditions** | The box is exact: CFD nodes lie on all six planes, identical on 7 files. Use it as config constants; reading 50 M interior nodes per sample in a surface task would cost about 600 MB of reads per step. Pieces: inlet, outlet, ground, two sides, roof, with the floor at the interior's z_min, not the vehicle's. About 2 agent-days with tests; the per-step cost is a few thousand triangles. **Blocked on conditions:** the walls, roof and outlet types are undocumented, and the belt layout is unknown. The tyre-to-ground closure cannot be inferred. | Radius as a constant (24,533 in measured, or 75 MAC documented: Peter's call); 1,789 files have no interior to read it from. The inlet/outlet split by `U_inf` is cheap per sample. The symmetry disk needs the wing-root loop at y = 0 from the full-resolution vehicle (140 M points, 280 M triangles), which the reader has already subsampled. Extracting it per sample costs tens of seconds and more than 10 GB per load. About 3-4 agent-days, and too slow per step unless the hole is approximated. **Not recommended.** |
| **(iii-a) new dataset version: a synthesized sidecar** (curator generators; boundary-only pdmsh per case in a new versioned folder, read beside the vehicle) | A SUV generator or config: `BoxTunnelBoundaries`-style faces with y +-20 m, the floor from the interior's z_min, and names from the SUV setup, not DrivAerML's slip/no_slip floor split. About 1 day of code plus 1-2 CPU-h over 1,991 files; less than 1 MB per file. Still blocked on the undocumented condition types. | `HemisphereBoundaries` with a fixed radius over the stored vehicles. The injection stage took 65 s per case in the 2-file run, and the read is the full vehicle (about 8 GB), so about 2-5 min per case, 60-150 CPU-h over 1,800, within cpu-short limits. About 1-2 days of code; less than 1 MB per case. The radius needs Peter's ruling. |
| **(iii-b) new dataset version from the raw CFD meshes** | Extract the outer faces of the public `merged_volumes.vtu`: the exact walls and the CFD's own tyre closure. Needs the HF terms accepted (Peter's call) and about 6.5 TB downloaded (1,991 x 3.25 GB; not on LHR); face extraction of 45 M-cell polyhedral meshes, roughly 100-300 CPU-h; then split the faces by plane. 3-5 days with transfer. Condition types still undocumented. | Not useful: the boundary VTU is the aircraft only, and the volume is an octree whose outer faces are a staircase (43 TB to download). |

**Recommendation: (iii-a) for both datasets**, together with the surface-task read-without-predict change and the
DomainMesh reader. Reasons:

1. It is the method already behind the DrivAerML domain pieces that canonical ISLA reads, and behind the 2 HiLift files.
   One provenance class across datasets is more principled than a recipe-only exception.
2. The geometry is the same as in (ii), but it is built once, with its generator, constants and sources recorded in the
   files. There is no per-step cost and no frame plumbing in the augmentations.
3. HiLift's symmetry hole needs the full-resolution vehicle, which the datapipe no longer has after subsampling.
4. (iii-b) would add only SHIFT-SUV's tyre-to-ground closure, since the other faces are exact planes either way. That is
   worth it only if Peter wants CFD-exact boundaries.

Confidence: high on what is stored and where it came from (read from the files and the run database). Medium on the HiLift
radius: the interior's far field is unambiguous, but the paper says ≈75 MAC.

**Decisions only Peter can make, before any of this:**
1. SHIFT-SUV's side-wall, roof and outlet condition types, and its belt layout, are not in any public source. The rule
   "never invented" means they need Luminary's answer (contact from the HF card) or Peter's ruling on how to label an
   undocumented piece. I recommend a short question to Luminary. Asking takes another person's time; I can draft it.
2. HiLift radius: the stored interior's far field (24,533 in, 88.95 MAC; the value the 2 synthesized files use), or the
   paper's ≈75 MAC. I recommend the stored value, since it is the CFD's own data, and a correction note in the paper.
3. Whether synthesized pieces count as "from the dataset". DrivAerML's already are (section 0).
4. SHIFT-SUV inflow speed: the HF card gives 31.2928 m/s for full scale; the stored `U_inf`, AB-UPT, GeoTransolver and the
   croissant metadata give 30 m/s. This matters for an inlet condition's value.

## 4. Commands, jobs and files

- cpu-short jobs (LHR, account coreai_modulus_cae): 361149 `isla.setupres.hl` (HiLift `hl_domain` x 2, `hl_interior` x 3;
  2 of the interior reads failed on the truncated memmaps), 361150 `isla.setupres.suv` (`suv_stl`, `suv_faces` x 2 each),
  361188 `isla.setupres.p2` (`hl_radius` x 2, `suv_gap` x 2). Each ran under 1 minute. Registry row `^isla\.setupres\.`
  added 20:31 and removed 20:37 EDT (backups `watchers/.ACTIVE_EXPERIMENTS.bak.setupres*`, cmp-checked; DRV-GROUND's row
  added in between was kept).
- Harness: `harness/setup_probe.py`, `harness/setup_probe2.py`, `harness/probe.sbatch`, `harness/probe2.sbatch`,
  `harness/*.calls`, `harness/hl_sizes.sh` (login-node listing). md5s of the first job's files were checked identical on
  LHR and locally (`md5sum`). Outputs: `out/hl.jsonl`, `out/suv.jsonl`, `out/p2_console.txt`, `out/hl_sizes.txt`, and the
  slurm logs; on LHR, `$T/setupres/out/`.
- Login-node reads (listing, headers, `meta.json`, the 64 KB sqlite database): commands as quoted in sections 1-2, each
  `timeout ... ssh -o BatchMode=yes lhr2 '...'`.
- Web sources: as linked above. The shift-neurips-2026 clone was read at commit d1ec365686fa (2026-05-06). The Luminary
  V&V PDF is saved as `src/luminary_drivaer_vv_2024.pdf` (one page, no setup details).
