# BC-SIDECAR: the boundary pieces SHIFT-SUV and HiLift do not store, prototyped

Research lane of the ISLA program, launched 2026-10-23 at about 23:46 EDT by the supervisor (01a13c2c); written
2026-10-24 at about 01:50 EDT. It prototypes the supervisor's supply plan, which Peter has not yet approved, on branch
`psharpe/isla-bc-sidecar`, worktree `/home/psharpe/gh/physicsnemo-isla-bc-sidecar`, from SURFACE-BC's e636f29bd.
Nothing merges without Peter. On LHR, T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0.
Every job, harness md5 and snapshot is listed in `JOBS.md`. Raw records are in `out/`, `out_smoke/` and `logs/`.

## 1. SHIFT-SUV: the box is the same on every file

I read all 1,991 files (996 estate, 995 fastback) in cpu-short jobs 363013 and 363014, which took 1.47 process-h
(`lhr/suv_box.py`; summary in `out/suv_box_summary.txt`). The result:

- **The box.** On all 1,991 files, the interior's exact float32 bounding box is x -40 to 80 m, y -20 to 20 m,
  z -0.024 to 19.976 m. No file differs, and every file stores only `boundaries/vehicle`.
- **CFD nodes on the planes.** Every file has nodes on all six planes: 3,792-4,029 on each end plane, 11,280-11,947
  on each side plane, 242,597-246,369 on the floor and 18,565-19,477 on the roof.
- **The vehicle.** Its lowest z is -0.0228564 to -0.0228537 m, so the ground lies 1.1436-1.1463 mm below it (median
  1.1448 mm). It spans x -0.871..-0.829 to 3.793 m, |y| up to 1.024 m and top heights 1.429-1.745 m.

The box is therefore a constant of the dataset, and the recipe builds it from constants.

**Split.** I used the recommended one: inlet x_min, outlet x_max, floor z_min, side_walls y_min + y_max, roof z_max.
The evidence gives no reason to split differently: inflow (uniform, 30 m/s) and floor (translating) are documented,
while walls, roof and outlet are not. The two side walls are the same kind of face by symmetry, so they share one
condition.

## 2. Code (branch `psharpe/isla-bc-sidecar`, pushed; commits 456b19260 and 12fce7bfc, `-s`, hooks on)

- **`AddBoxBoundaries`** (`src/domain_transforms.py`) builds the six faces of a box from constants. Each face is a
  20 x 20 grid, curator's `box_face_patch` (722 triangles per face), with normals into the box. Named pieces may merge
  faces, and every face must be used exactly once. On a bare Mesh it returns the DomainMesh that
  `MeshReaderWithGlobalData(domain_boundaries=...)` returns. The five SHIFT-SUV dataset files run it as their first
  transform and end with `SurfaceMeshToDomainMesh`, so CenterMesh, rotation, translation, FixedRandomPose and the
  L_ref scaling all move the box with the vehicle.
- **Two SHIFT-SUV files also run `SubsampleMesh`** (`shift_suv_estate_surface`, `shift_suv_fastback_surface`). That
  step acts on every mesh in the sample, so the box stays whole only while `sampling_resolution` exceeds 1,444 cells
  (true at the configured 200,000; noted in those files).
- **`MeshReaderWithGlobalData(domain_boundaries_root=...)`** reads the named boundaries from a sidecar directory laid
  out like the case files. `highlift_surface_whole.yaml` uses it (with the new key
  `dataset_paths.highlift_aero_ml_boundaries`); it is `highlift_surface.yaml` plus that reader option, with
  `SurfaceMeshToDomainMesh` as its last step.
- **Model configs.** `isla_surface_shift_suv.yaml` reads the vehicle and the five box pieces and predicts at the
  vehicle. It does not flip the vehicle, since SHIFT-SUV's vehicle points into the fluid (BOUNDARY-AUDIT). The new
  `isla_surface_highlift_whole.yaml` reads the wing, inlet, outlet and symmetry and flips the wing for ISLA only: its
  signed volume is negative (wound into the body) on the 4 cases checked.
- **A fix in `forward_kwargs.walk_path`.** It now resolves a TensorDict's own key before attributes of the same name.
  Before the fix, `boundaries.floor` resolved to `TensorDict.floor`, a method.
- **Tests.** I added `tests/test_box_boundaries.py`: the box on a synthetic case, what it refuses, the whole chain of
  each SHIFT-SUV file moving the box with the vehicle (a scaled-similarity fit), the sidecar reader, and the HiLift
  whole file against the body-only one. I also added two config tests in `test_isla_configs.py` and one
  `walk_path` test.
- **Results.** The whole ISLA test directory passes on LHR cpu-short (363573: 280 passed, 47 skipped). The recipe
  tests give 554 passed, 12 skipped and 2 failed (363575). The 2 failures are the `test_campaign_c_hooks`
  `initialize_slurm` TypeError inside a Slurm job; the base (362118) fails them with the same message. The ISLA model
  code is the same as `code_surfbc` (129c66294).

## 3. HiLift: the generator matches curator's own files, but it fails on the other geometries

The code is curator ee6aeda, read-only via `git archive` into `curator_ee6aeda/`. Its boundaries code is unchanged
since 60ed919, so it is the code behind the 2026-06-10 run.

**Validation against the 2 stored files** (job 363290, radius taken from the stored pole, 24,533.418 in = 88.95 MAC):

| piece | points (gen / stored) | cells (gen / stored) | max coordinate difference, sorted points | oriented cells identical |
|---|---|---|---|---|
| inlet, both cases | 1,024 / 1,024 | 1,952 / 1,952 | 0.0 | yes |
| outlet, both cases | 1,024 / 1,024 | 1,952 / 1,952 | 0.0 | yes |
| symmetry, AoA_10 | 11,253 / 11,253 | 11,039 / 11,031 | 0.0 (same point set) | no: 8 cells differ |
| symmetry, AoA_12 | 11,254 / 11,254 | 11,037 / 11,030 | 0.0 | no: 7 cells differ |

**The hole outline.** On AoA_12 the hole is 612,114 against 612,119 in^2 (the silhouette loop: 612,137); it is
5,857 against 5,858 in long, with x 92.5-2,583.3 and z 91.0-728.5 in on both, inner radius 204.67 in, and a
Hausdorff distance of 0.40 in between the hole vertices. On AoA_10 it is 611,535 against 612,119 in^2, Hausdorff
18.5 in, with 2 loose ends.

The differences come from curator's loop walk. At degree-4 vertices it takes the first branch, so the result depends
on vertex order. On AoA_10, relabelling the wing's vertices at random once gave a disk of 11,040 cells and once a
broken one (job 363439, `out/hlcheck.jsonl`). So the generator reproduces curator's pieces, up to that walk.

**The 5 other cases** (LHC030_AoA_4, LHC065_AoA_8, LHC100_AoA_14, LHC140_AoA_18, LHC180_AoA_22) are written under
`$T/bcsidecar/hilift/` with a SIDECAR.json each. Their symmetry planes are wrong:

- Curator's silhouette walk finds 927-960 small loops, covering 46-54 % of the root edges.
- The disks' hole areas are -5,158 to +1,704 in^2, against about 612,000.
- On LHC100, some normals point to -y.
- 17-22 k points are unused.

**The cause, on the wing as stored** (`out/hldiag*.jsonl`, `out/hlnm.jsonl`, `out/hldedup_*.jsonl`):

- **Duplicated cells.** These geometries hold 18.7-19.4 M duplicated cells, copies of only 25,500-25,800 distinct
  cells. Single edges on y = 0 are shared by up to 1.08 M cells (for example near x 118, z 168 in on LHC030). The
  duplicates are 7 % of the cells and 119 k-1.62 M in^2 of the 1.79-3.32 M in^2 summed wing area. LHC001 has 6.6 M
  duplicated cells, but they total only 134 in^2.
- **An open root.** With the duplicates dropped, the root on y = 0 is still open: 7,376-7,460 open edges with
  1,616-1,630 loose ends (about 812 chains). It also has 21-308 non-manifold edges and 0.02-2.1 M cells lying in the
  plane.
- **The sample.** Of 21 geometries sampled (one AoA_10 case per ninth geometry, LHC001-LHC180), only LHC001 has a
  root that closes into one loop (11,189 edges, every vertex of degree 2).

The root therefore needs a repair, or the hole a different construction, before a symmetry plane can be supplied for
1,798 files. The far field (inlet, outlet) does not depend on the wing and is valid on every case.

**Cost per case** (generation and write only; 8 cores, 120 GB allocation):

| case | wing cells | wall s | CPU s | peak RSS |
|---|---|---|---|---|
| LHC001_AoA_10 / _12 | 285 M / 290 M | 7.4 / 15.1 | 27 / 30 | 18.7 / 19.0 GB |
| LHC030_AoA_4 | 259 M | 28.9 | 56 | 17.1 GB |
| LHC065_AoA_8 | 281 M | 39.6 | 65 | 18.5 GB |
| LHC100_AoA_14 | 295 M | 33.1 | 60 | 19.3 GB |
| LHC140_AoA_18 | 279 M | 37.2 | 73 | 18.4 GB |
| LHC180_AoA_22 | 292 M | 35.4 | 63 | 19.1 GB |

**Projection for all 1,800 cases.** About 27 process CPU-h, or 37 at the worst case. Allocated at 8 cores for the
worst wall time, that is 158 core-h: 1,800 x 40 s is about 20 wall-hours, spread over cpu-short jobs. The sidecar is
under 1 MB per case (4.4 MB for 7). Dropping the duplicates costs 5-8 min and 31-36 GB more per case (about
190 CPU-h for 1,800), and it does not fix the root on its own.

## 4. GB300 smokes (job 363460, short QoS, 1 node, 3:24, 0.23 GPU-h)

Setup: 20 training steps, bfloat16, eager with compile_blocks, augment on, seed 42. Code is `code_surfbc` and the
recipe snapshot is `$T/recipe_bcsidecar` (12fce7bfc minus the comment). SHIFT-SUV runs at 200,000 cells on
`shift_suv_surface`. For HiLift, "body" is `isla_surface_highlift` and "whole" is `isla_surface_highlift_whole` on
the one case LHC030_AoA_4, at 100,000 cells (`out_smoke/smoke_summary.txt`).

| arm | losses finite | loss, step 1 to 20 | median step, steps 6-20 | peak memory |
|---|---|---|---|---|
| SHIFT-SUV body (the box built, not read) | yes | 0.151 to 0.067 | 164.5 ms | 65.4 GB |
| SHIFT-SUV whole (vehicle + 5 box pieces, 7,220 extra tokens) | yes | 0.138 to 0.069 | 196.1 ms (1.19x) | 67.5 GB |
| HiLift body | yes | 0.606 to 0.146 | 682 ms | 33.1 GB |
| HiLift whole (sidecar) | yes | 0.884 to 0.161 | 675 ms (0.99x) | 38.9 GB |

What the rows show:

- **SHIFT-SUV.** The 1.19x is about SURFACE-BC's: 207,220 tokens is not a multiple of 8, and ALIGN8 is looking at
  that.
- **HiLift.** The steps take about 680 ms in both arms, probably because data loading of the 260 M-cell wing
  dominates. The whole arm's symmetry plane is the invalid one from section 3, so this row tests only the plumbing.

## 5. Decisions that remain Peter's

1. **The HiLift wing data.** About 19 M duplicated cells on 20 of 21 sampled geometries inflate HiLift's surface
   measures and draws (about 7 % of uniform draws; 7-50 % of the area). Should the program report this to the
   dataset's owners, and correct for it in the recipe (dropping duplicate cells as a transform)? This affects the
   HiLift standing whether or not a symmetry plane is ever supplied.
2. **The HiLift symmetry plane.** It needs the root repaired (closing the about 812 open chains on y = 0) or another
   construction of the hole, as a new lane. Alternatively, a whole-boundary HiLift run could read only the far field
   for now, which Peter's ruling would count as an incomplete boundary.
3. **Where the HiLift sidecar lives and what its pieces are called.** I recommend a sibling folder of the dataset,
   such as `PhysicsNeMo-HighLiftAeroML-boundaries/`, laid out like the case files, with curator's names inlet,
   outlet and symmetry. For now it lives under `$T/bcsidecar/hilift`.
4. **The radius.** The stored interiors and curator's 2 files use 24,533.42 in (88.95 MAC); the paper says about
   75 MAC.
5. **The SHIFT-SUV pieces and their names** (inlet, outlet, floor, side_walls, roof), and whether to ask Luminary for
   the undocumented wall, roof and outlet conditions.

## For the book

No standing changes from this lane. If the duplicate-cell finding is confirmed, the HiLift chapter's data section
will need a sentence and an In-flight callout once a decision is made.
