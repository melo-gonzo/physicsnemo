# HL-DUP: duplicated wing cells in HiLiftAeroML and what they do to the HiLift standing

Research lane of the ISLA program, launched 2026-10-24 at 02:06 EDT by the supervisor (01a13c2c); written 06:25 EDT.
Stage 1 only (no training). Jobs, harness md5s and snapshots: `JOBS.md`. LHR $T =
`/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`.

**Answer.** Yes, and by a lot. On every geometry the wing stores about 18.9 M extra copies of about 25,700 root cells.
Those copies carry 42 % of the Horvitz-Thompson measure of the 200,000 tokens that the standing draws (median 47 %), and
the same share of the scorer's measure weights. Excluding them from the metric (b) raises every arm's pressure error by
about 38 % (wall shear by about 6 %) but leaves the ranking: baseline / ISLA stays at 0.91-0.92 on pressure and wall
shear. Removing them from the input as well (c) costs ISLA 41 % more pressure error than (b), against 5-6 % for the
baselines. That is a distribution shift, because
the models trained with the duplicates. So the defect does not explain ISLA's current deficit, but ISLA's input is very
sensitive to it. The duplicates far exceed the brief's 1 % threshold for stage 2, which is proposed in `PREREG.md`.

## 1. Census (exact, every geometry; counts from `out/census/*.json`, table `census_table.md`, summary `census_summary.json`)

- **Coverage.** 275 case reads on cpu-short: 268 distinct cases. They are all 180 scored (val) cases read from the
  `-medium` copy the standing read, and one AoA_10 case for each of the 62 other geometries. They also include all 10
  angles of attack of LHC001, LHC030 and LHC140, plus the BC-SIDECAR hole cases. All 180 geometries are covered.
  Seven cases were read from both copies, and the counts are identical.
  Cost: 20.2 process CPU-h (21.8 core-h allocated), 3-12 min per case, peak RSS 16-20 GB.
- **Method** (`lhr/hl_census2.py`; v1 `hl_census.py` gave identical results on the 273 cases without a hash collision).
  Each cell's sorted vertex-index triple is hashed to 64 bits. Cells whose hash repeats are grouped exactly by a lexsort
  on the triple; the lowest index of each group is kept, and every other cell in the group is an extra copy. Coincident
  points are found by their float32 coordinate bits, which also catches copies that match by coordinates only. A check
  confirmed that raw memmap reads equal `Mesh.load`.
- **Two kinds of duplication.**
  - *Root kind*, 266 of 268 cases, including LHC001 at 8 of its 10 angles. Per case: 18.57-19.45 M extra copies (median
    18.89 M; 6.4-7.5 % of the cells) of 25,425-25,974 distinct cells. Their extra area is 61,762-1,905,310 in² (median
    1.46 M), which is **3.6-52.7 % of the summed wing area (median 46.3 %)**. The largest multiplicity is 424,278-491,701,
    and about 1,800 cells per case have more than 1,000 copies. The distinct duplicated cells cover only 2,015-22,037 in².
  - *Pair kind*, LHC001_AoA_10 and LHC001_AoA_12 only: 6.61 M and 6.66 M cells, each stored exactly twice with reversed
    orientation, totalling 134 in². These are spread over the whole wing (y up to 1,160 in). LHC001_AoA_10 also holds
    228,548 zero-area cells. This is the case BC-SIDECAR sampled for LHC001, which is why that lane reported LHC001 as
    the exception.
- **Not a property of the geometry alone.** On LHC030 and LHC140 the number of copies is constant across angles
  (18.78-18.82 M and 18.65-18.68 M), but their area is not: 119 k to 1.29 M in² on LHC030, and 1.28-1.71 M in² on
  LHC140. The kind can change within a geometry (LHC001). The meshes differ by angle of attack (251-301 M cells).
- **Where.** All root-kind copies lie in x 95.8-2,530 in, z 29-727 in, at the wing root. The copy-weighted y_max is at
  most 1.03 in at the 90th percentile; a few distinct cells reach the tip. Pooled over all reads:
  - 42 % of the extra copies have no vertex on y = 0 (|y| < 1e-3), 35 % have one, 19 % two and 4 % three.
  - The copies with three such vertices lie in the plane and total 42,817 in².
  - The extra area sits mostly on the copies with one (168 M in² pooled) or two (106 M in²) vertices on y = 0.
- **Exactness.** Every root-kind copy repeats the vertex indices of its first copy: 83 % as the same row, 16 % as a
  cyclic rotation of it, and 0.26 % reversed (nearly all of those from the pair kind). Copies by coordinates only number
  10,707 over all 275 reads, a negligible count. No cell repeats a vertex.
- **Cell fields.** Every stored field (P, T, RHO, U, TAU_WALL, RMS(P), RMS(T)) is bitwise equal to the first copy on
  93-97.5 % of the copies. The rest differ by at most float32 rounding: 3.1e-5 on PROJ(AVG(P)), 9.8e-4 on
  PROJ(AVG(U)), 6e-8 on AVG(TAU_WALL). No copy has a NaN. The copies carry the same solution.
- **Zero-area cells.** Median 131 per case (range 99 to 232,527, the maximum on LHC001_AoA_10). Of 808,804 over all
  reads, 320,483 are extra copies, so dropping the duplicates removes about 40 % of them.
- **Masks.** For every scored case, `np.packbits(extra)` is in `$T/hldup/masks/` (5.9 GB; 35 MB per case). The masks of
  the main-dataset reads are in `masks_main/` (3.1 GB).

## 2. Exposure in the standing's pipeline

I ran the standing's own val loader (recipe_h090's `highlift_surface_medium`, `infer.yaml` with the scorer's overrides)
with a reader hook. The hook gives each drawn cell a flag that no model reads; on 3 samples its draw equals
`MeshReader`'s own. All 180 scored cases ran, not 10 (`out/expose/expose.jsonl`, `hl_expose.py`):

| per case, after the full transform chain | min | median | mean | max |
|---|---|---|---|---|
| duplicate share of the ~200,000 tokens | 6.4 % | 6.8 % | 6.8 % | 7.5 % |
| duplicate share of the drawn Horvitz-Thompson measure | 3.2 % | 47.0 % | 42.1 % | 52.7 % |
| ISLA's length unit (`ISLA._gauge`) without / with them | 0.772 | 1.141 | 1.098 | 1.186 |

The measure share exceeds 1 % on 180 of 180 cases and 10 % on 174. The scorer weights its relative L2 by the same
measure (`target_measure`), so the published HiLift numbers put about 42 % of their weight on copies of about 25,700
root cells. ISLA's length unit moves by -23 % to +19 % per case when the copies go.

## 3. Effect on the standing (inference only; job 366726, short QoS, 1 node, 17:43; `score_table.md`, `score_summary.json`)

The standing's own code, recipe snapshot, infer.py, scorer and checkpoints were used. Every row is epoch 500 on 180
cases. Under (a), each per-case metric of the five headline fields equals the standing's stored score to within
1.5e-5 relative (8.3e-5 over all 11 metric keys, a component field), which validates the hook.

- (a) is the standing as scored.
- (b) uses the same predictions, scored without the extra copies.
- (c) removes the extra copies before the draw, as in `hldup.draw`: a cyclic block of 200,000 over the kept cells,
  with factor n_kept / k. It is a distribution shift from training, and it scores only the distinct cells, so compare it
  with (b).

Seed means; seeds in `score_table.md`:

| field | protocol | ISLA | GeoTransolver | Transolver | GT / ISLA (s42, s43) | T / ISLA (s42, s43) |
|---|---|---|---|---|---|---|
| pressure | a | 0.04272 | 0.03902 | 0.03941 | 0.913 (0.914, 0.912) | 0.923 (0.955, 0.890) |
| pressure | b | 0.05888 | 0.05400 | 0.05443 | 0.917 (0.917, 0.917) | 0.924 (0.952, 0.897) |
| pressure | c (shift) | 0.08324 | 0.05652 | 0.05784 | 0.679 (0.651, 0.709) | 0.695 (0.689, 0.701) |
| wall shear | a | 0.09634 | 0.08903 | 0.08783 | 0.924 (0.926, 0.922) | 0.912 (0.924, 0.899) |
| wall shear | b | 0.10273 | 0.09473 | 0.09335 | 0.922 (0.924, 0.920) | 0.909 (0.921, 0.896) |
| wall shear | c (shift) | 0.13829 | 0.09635 | 0.09524 | 0.697 (0.685, 0.709) | 0.689 (0.685, 0.693) |

Velocity, density and temperature follow the same pattern; under (a) and (b) every baseline / ISLA ratio is 0.89-0.98.
Per case, ISLA's pressure is lower than GeoTransolver's on 53 / 54 / 9 cases under (a) / (b) / (c), and lower than
Transolver's on 22 / 20 / 10. The median per-case pressure b / a is 1.37-1.38 for every arm. Against (b), the shift (c)
costs ISLA +41 % on pressure and +35 % on wall shear, GeoTransolver +5 % / +2 %, and Transolver +6 % / +2 %. The median
per-case pressure c / b is 1.64 / 1.57 for the ISLA seeds and 1.06-1.17 for the baselines. Per case, the shift's damage
(log c / b, pressure) correlates only weakly with the duplicates' measure share (r = 0.09-0.11 for ISLA, 0.06-0.12 for
the baselines) and with the change of ISLA's length unit (r = -0.10 / -0.11 against |log ratio|), so neither explains
which cases suffer (`corr_cb.py`, `corr_cb.json`).

**What it means.**

- The metric's weighting of the copies inflates no arm's ranking: removing them from the metric changes the ratios by at
  most 0.004 on pressure and wall shear (seed means), and by at most 0.009 on any field and seed.
- ISLA reads cell measures and the baselines do not, so ISLA alone has learned an input in which 42 % of the boundary
  measure sits on repeated root cells. When that measure is removed, its predictions break. A model trained on
  deduplicated data may differ, and stage 2 asks that question.
- Confidence: (a) and (b) are firm on 180 cases. The GeoTransolver ratios agree across seeds within 0.004; the
  Transolver ratios differ by up to 0.065 between seeds (pressure 0.955 / 0.890 under (a)), but every seed's ratio is
  below 1 under both (a) and (b). Under (c) both ISLA seeds agree within
  5 %. (c) measures sensitivity to the defect, not the quality of a model trained without it.

## 4. The symmetry plane's hole (prototype; output only in `$T/bcsidecar/hilift_hole/`, local copy `out/hole/`)

Three constructions were tried on the 7 BC-SIDECAR cases, with the duplicates removed:

- **Footprint union** (`hl_hole.py`): the union of the deduplicated wing cells touching y = 0, projected onto the
  plane, grown by eps and filled.
- **Open-edge closing** (`hl_hole2.py`): the open on-plane edges, buffered at half the root's largest edge length
  (sensitivity at 0.5x and 2x).
- **Coverage checks**: how much of LHC001's outline each case's open edges cover (`hl_hole3.py`), every on-plane
  edge covers (`hl_hole4.py`, `hl_hole5.py`), and every stored vertex near the plane covers (`hl_hole6.py`); then a
  closing from each case's own open edges and on-plane vertices (`hl_hole7.py`).

- **LHC001_AoA_10 and _12 (clean roots).** The footprint gives 612,137.3 in², against 612,136.6 in² for curator's
  stored hole, measured by polygonizing its inner boundary edges. BC-SIDECAR's 612,119 in² came from another area
  method on the same file. The area ratio is 1.0000011, the Hausdorff distance 0.011 in and the
  symmetric difference 0.66 in². The result is stable to eps 0.001-0.1 in and to a band of 1e-4. Every on-plane vertex
  lies inside the result. The open-edge closing gives 612,136.8 in² (ratio 1.00000025, Hausdorff 0.48 in). The disk
  minus the hole triangulates validly.
- **The five broken roots** (LHC030_4, 065_8, 100_14, 140_18, 180_22). No construction works:
  - the footprint falls apart into 809-813 pieces, with no filled region of 1 in² or more;
  - the open-edge closing leaves 0-41 in² at eps = h_max / 2 (0 on LHC030_AoA_4), and at most 465 in² even at eps 11 in;
  - the root holds 1,624-1,628 loose ends and 55-300 non-manifold edges.
- **What the open edges show.** On all five broken roots, every on-plane vertex used by a cell lies on LHC001's
  outline (within 0.011 in; median 1.4e-7 in). Their open edges total only 3,283-3,290 in, against the outline's
  5,852 in, and cover 69.3-69.4 % of the outline within 0.5 in (85.3-85.5 % within 2 in).
- **No other edge holds the rest** (`hl_hole5.py`, job 367605; `out/hole/hole5.jsonl`). Deduplication keeps one copy
  of every cell, so it removes no edge. Counting every on-plane edge (open, shared and non-manifold: 7,562-7,853 per
  root), the coverage is still 69.4-69.6 % within 0.5 in (85.4-85.5 % within 2 in). Shared and non-manifold edges
  cover at most 0.6 % of the 1,790-1,794 in of outline that the open edges miss, and edges with both ends within
  0.1 in of the plane add at most 0.6 %. On LHC001 every class covers 100 %, and polygonizing all its on-plane edges
  gives curator's hole to a Hausdorff distance of 0.002 in (`hl_hole4.py`, `hole4.jsonl`).
- **The vertices do hold it** (`hl_hole6.py`, job 367622, reading every point of each stored wing; `hole6.jsonl`).
  Every wing, broken or not, stores 11,237-11,249 vertices at |y| < 1e-3, at the same positions on all seven cases:
  96.1 % of LHC001's outline lies within 0.5 in of one and 100 % within 2 in. On the broken roots the cells touching
  the plane use only about 8,290 of them. The other 2,937-2,959 per root belong to no cell, since copies share their
  first copy's vertices (`hl_hole7.py`; 0 such vertices on LHC001). So the files hold the whole root outline as
  vertices; what is missing is the root triangles that connect about 30 % of it.
- **A construction from each case's own data** (`hl_hole7.py`, job 367657; `hole7.jsonl`, `*_hole7.png`). The case's
  open edges and all its on-plane vertices are each grown by eps, unioned, filled and shrunk. eps is half the 99.9th
  percentile of the vertex-to-second-neighbour distance, 0.65 in.
  - LHC001_AoA_10 and _12: 612,136.7 in² against curator's 612,136.6 (ratio 1.0000002, Hausdorff 0.43 in).
  - Five broken roots: at eps 0.65 in the band still has 12-14 gaps and no hole forms. At 1.5x eps (0.98 in) it closes
    into one hole of 611,842 in² on every root, 295-297 in² (0.048 %) less than LHC001's stored hole, with a Hausdorff
    distance of 0.62 in.
- **Verdict.** The hole can be built from each broken wing's own data, but not yet robustly: the closing needs a
  scale chosen above the vertex spacing, and it leaves a 0.05 % area deficit. A robust construction would rebuild the
  missing root strip topologically: chain the orphan vertices into the open-edge loops in outline order (each orphan
  chain has a vertex spacing of at most 1.3 in), then check the closed loop against curator's hole on LHC001 and
  against the 612,137 in² that all seven cases share. That is roughly a 2-4 h piece of work, beyond this lane. Using
  LHC001's outline as a shared root section stays declined.

## 5. Cost

- GPU: 1.27 GPU-h of the 4 GPU-h cap. Job 365654 (failed at Hydra config resolution, 1:17, 0.09 GPU-h) and job 366726
  (17:43, 1.18 GPU-h; it includes the CPU exposure probe).
- CPU: about 22 core-h allocated on cpu-short (jobs 364757, 364882, 364883, 365608, 365609, 365673, 366727, 366767),
  plus about 1.5 core-h for the hole checks after the second read (367338 and 367416, partial; 367560 and 367577 ran out
  of memory; 367605, 367622, 367657).
  Each case took 3-12 min and 16-20 GB with the hash method, against BC-SIDECAR's 5-8 min and 31-36 GB.
