Status: DONE

DRV-GROUND, 2026-10-23, 20:33 to 21:18 EDT. Scratch: `scratch/isla-drv-ground-2026-10-23/`.

**What ran.** LHR cpu-short only, no GPU, at most 4 jobs at once: facts 361183 (smoke), 361189, 361190; check
361391 (smoke), 361419-361422 (361382 cancelled, bad argument). Branch psharpe/isla-drv-ground, commit c4419b299
(from c1e07b7fd), pushed with a plain push. Read-only LHR snapshot `$T/recipe_drvground`. Details in `JOBS.md`.

**Task 1: the facts, all 484 files** (`classify_summary.txt`, `out/classify.tsv`).
- 187 files match: the union of the four domain meshes equals the interior bounding box bitwise.
- 297 files are offset: z only, +2.40299 mm at the ground and +2.40326 mm at the roof. The two differ by one float32
  ulp at 19.68 m, so it is one translation. x and y offsets are 0, box sizes are equal, and point and cell counts,
  cells and x, y are identical to run_1's. dz per point is 2.40230 to 2.40326 mm (float32 rounding).
- On all 484, the vehicle's lowest z equals the interior's lowest z bitwise (-0.317597 m). The ground rule's
  shift equals the offset on all 484. Interior extremes are the same on every file.
- As stored, 5,117 to 5,227 interior nodes per offset file lie above the roof (1,527,826 in all); none on the 187.
- Provenance (with the curator code the supervisor pointed to): `.bc_surgery_checkpoint.jsonl` in the dataset
  folder lists exactly the 297 offset runs, all `"z_floor": -0.32`, rewritten in place on 2026-06-02 01:48 UTC.
  On 184 of the 187 matching files the meshes were written with the vehicle on 2026-06-01, as `main.py` writes a
  case (`out/mtimes.txt`).
- So the rule of task 2 holds on every file, and I wrote the transform without stopping.

**Task 3: the code (c4419b299).**
- New recipe transform `PlaceGroundUnderBody(ground, body, boundaries, axis=2)` in `src/domain_transforms.py`.
  It translates the named meshes by the body's lowest z minus the ground's, and refuses a non-planar ground or
  missing names. Its docstring records the defect, the counts and the provenance.
- It runs as the first transform in `drivaer_ml_volume_reference{,_area,_posed}.yaml` (in posed, before
  `FixedRandomPose`), so it acts in raw metres before CenterMesh, ComputeFreestreamDirection and
  NonDimensionalizeByMetadata.
- `drivaer_ml_volume.yaml` (the baselines) gets a comment only. Its reader cell-subsamples the vehicle and compacts
  points, so the vehicle's lowest z there changes from draw to draw, and none of its models read the domain meshes.
  The other DrivAerML configs read only the vehicle.
- Tests: shifted and unshifted synthetic cases plus refusals (`tests/test_domain_transforms.py`), and a config test
  that the transform is first in all three reference files (`tests/test_isla_configs.py`). Recipe tests: 528
  passed, 12 skipped. ISLA test directory on CPU: 273 passed, 47 skipped. Hooks on, `-s`.
- The ruff-format hook also reformatted a few existing lines in the two files I touched, around
  FlipBoundaryOrientation and its test.

**Task 4: all-file check through the real chain** (`check_summary.txt`, `out/check_*.jsonl`).
- `build_dataset(drivaer_ml_volume_reference.yaml)` ran with the real reader and the instantiated chain, step by
  step. On 484 of 484 files: the vehicle was read whole; after the transform the ground equals the interior's lowest
  z and the vehicle's; slip, inlet and outlet reach exactly the interior's highest z; x, y extents and the whole
  bounding box equal the interior's; x, y are unchanged; the shift is 0 on 187 files and 2.4029911 mm on 297.
- Signed distance to the six face planes (oriented by the cells' normals) for every interior node (77.6 G):
  outside after the transform 0 on every file; as stored, 1,527,826, all above the roof, on the 297 files.
  After the transform, 1.50 to 1.66 M nodes per file lie on the floor and 5,117 to 5,246 on the roof.
- At the end of the whole chain (centred, nondimensionalized), the ground still equals the vehicle's lowest z on
  484 of 484, and 0 of 9.68 M drawn interior nodes lie outside.

**Task 5.** `UPSTREAM_DRAFT.md`: a draft for Peter only, not sent. It covers the files, the offsets, the provenance
file, a check snippet and a fix.

**Meaning.** The 297 files' four domain meshes were written with a hard-coded floor; the CFD mesh and the vehicle
are consistent on every file. The transform restores the curator's own rule bitwise and is a no-op on the 187
matching files. I am highly confident, since every check is exact and covers all files. ISLA runs on these
configs before c4419b299 (FLOOR-DRV uses run_171, a matching file, so it is unaffected) read the low box on
offset files.

**Rulings relied on.** Datasets are never modified (Peter); orientation flips are recipe transforms (Peter);
the whole boundary (Peter, 17:08). The supervisor's decision (not Peter's): fix this in the recipe, with the ground
moved onto the vehicle's lowest point. For memory: `PlaceGroundUnderBody` is first in every DrivAerML Volume
reference chain, and a new DrivAerML config that reads the domain meshes needs it, with the vehicle read whole.

**Left.** Merge of psharpe/isla-drv-ground (supervisor's call). Nothing is running and the registry row is gone.
On disk: worktree `/home/psharpe/gh/physicsnemo-isla-drv-ground` (full checkout; remove once merged or dropped),
LHR `$T/recipe_drvground` (1.4 MB) and `$T/drvground` (1.7 MB). The raw domain-mesh copies are deleted.
