Status: PARTIAL. SHIFT-SUV is done. On HiLift, curator's generator reproduces its own 2 files, but it makes a valid
symmetry plane on LHC001 only: the other wings store about 19 M duplicated cells and an open root. Supplying HiLift's
symmetry plane needs a decision on a new lane, and the duplicates need a decision on the dataset itself.

**What ran.** All on LHR, all job names isla.bcsidecar.*; `JOBS.md` lists every job id and harness md5.
- cpu-short: 19 jobs, 16.3 core-h.
- GB300: one short-QoS job, 363460, 0.23 GPU-h.
- Branch `psharpe/isla-bc-sidecar`, pushed: 456b19260 (the code and tests) and 12fce7bfc (a comment). Worktree
  `/home/psharpe/gh/physicsnemo-isla-bc-sidecar`.
- The registry row is removed and no job is left running.

**The SHIFT-SUV box.** I checked all 1,991 files (996 estate, 995 fastback); no file differs.
- The interior's exact box is x -40..80, y -20..20, z -0.024..19.976 m on every file, with CFD nodes on all six
  planes.
- The ground lies 1.1436-1.1463 mm below the vehicle's lowest point.

**The SHIFT-SUV code.**
- `AddBoxBoundaries` builds the box (curator's 20 x 20 grid per face, normals into the fluid) as the first transform
  of the five SHIFT-SUV dataset files, so every pose and CenterMesh moves it with the vehicle. The pieces are inlet,
  outlet, floor, side_walls and roof.
- `isla_surface_shift_suv.yaml` reads the box and predicts at the vehicle.
- Fix in `walk_path`: `boundaries.floor` used to resolve to `TensorDict.floor`, a method.

**Tests.** The whole ISLA test directory passes (280 passed, 47 skipped; job 363573). The recipe tests give 554
passed, 12 skipped and 2 failed (363575); the 2 failures fail with the same message on the base (362118).

**HiLift validation** (`REPORT.md` section 3).
- On LHC001_AoA_10 and _12, inlet and outlet are identical (0.0 coordinate difference, the same oriented cells).
- The symmetry disk has the same points and 7-8 cells differ. The hole is 612,114 against 612,119 in^2 on AoA_12;
  the gaps come from curator's order-dependent loop walk.
- On the 5 other cases (`$T/bcsidecar/hilift/`), the generated symmetry planes are invalid. Of 21 sampled
  geometries, only LHC001 has a closed root loop.
- The others store 18.7-19.4 M duplicated cells (about 25,700 distinct cells), which is 7 % of their cells and
  7-50 % of their summed area. Even with those dropped, about 812 open chains remain on y = 0.
- The reader (`domain_boundaries_root`) and `highlift_surface_whole.yaml` work.

**HiLift cost.** 7-40 s wall, 27-73 CPU-s and 17-19 GB peak per case. For all 1,800 cases that is about 27 process
CPU-h, or 158 core-h allocated. Deduplication would add 5-8 min and 31-36 GB per case.

**Smokes** (20 steps, all losses finite; `out_smoke/smoke_summary.txt`).
- SHIFT-SUV: 164.5 ms and 65.4 GB body only, against 196.1 ms and 67.5 GB whole. The 1.19x is the unaligned token
  count, as SURFACE-BC found.
- HiLift (LHC030, the invalid symmetry plane, so plumbing only): 682 ms and 33.1 GB against 675 ms and 38.9 GB.

**Rulings I relied on.** Whole boundary (Peter, 17:08) and option A (23:26); never modify a dataset; the LANE_RULES
limits on cpu-short and short QoS. I added one memory candidate: HiLift wings carry duplicated cells (this lane, if
the supervisor confirms it).

**Decisions that remain Peter's** (`REPORT.md` section 5).
1. Report the duplicate cells to the dataset's owners, and correct for them in the recipe?
2. A lane to repair the wing root or to construct the hole another way, or a HiLift run with the far field only.
3. Where the HiLift sidecar lives and what its pieces are called (I recommend a sibling folder named
   `PhysicsNeMo-HighLiftAeroML-boundaries` with names inlet, outlet, symmetry).
4. The radius: 88.95 MAC stored, against the paper's about 75 MAC.
5. The SHIFT-SUV piece names, and whether to ask Luminary for the conditions on the walls, roof and outlet.

**What is left.** The symmetry planes for HiLift's 1,798 files, and the duplicate-cell question; nothing is running.

**What stays on disk.**
- The worktree above.
- This scratch folder (1.6 MB).
- On LHR: `$T/bcsidecar` (6 MB: harness, outputs, 7 sidecars, the curator copy), `$T/kernel/bcsidecar` (0.2 MB) and
  `$T/recipe_bcsidecar` (1.5 MB, read-only snapshot).
- Nothing over 1 GiB.
