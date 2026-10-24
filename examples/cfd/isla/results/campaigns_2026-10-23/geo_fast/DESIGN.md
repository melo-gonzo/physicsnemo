# GEO-FAST: a fast path for ISLA's geometry call (phase 1 design, 2026-10-23)

Lane GEO-FAST, launched 2026-10-23 17:52 EDT by the supervisor (01a13c2c). Reference: `psharpe/isla-geo-bundle` at
`f13eb5ce0` (snapshot `$T/code_dvcost2a`, md5-checked against git; geometry.py md5 1f0df670). Base: `code_cmpaf_fix`.
All timings GB300, LHR short QoS, one GPU per process, synchronized medians of 5 calls after warm-up, the first
training call of each dataset dumped by `dump_train.py` (REF-090 settings, bfloat16, float32 geometry inputs).
Records: `JOBS.md`; scripts and outputs in `lhr/` (mirror of `$T/kernel/geofast/`). Baseline step costs are q.4's
(job 346068), not the brief's rounding: +7 / +29 / +17 / +49 percent on AirfRANS / lug / AeroZoo / DrivAerML Volume.

## 0. Summary

1. **The whole boundary costs about 2 s per call on DrivAerML Volume, at every ISLA version, and nearly all of it is the
   Warp seed query.** With the floor strip, slip walls, inlet and outlet (isla_volume_floor, 564f803cc), the call takes
   2,376 ms at f13eb5ce0 and 2,057 ms at the base, against 339 and 92 ms for the vehicle alone. Of that, 2,010 ms is
   `signed_distance_field`, the Warp mesh query that only seeds the search radius: Warp's default LBVH build degenerates
   on a box 24 m long around a 1 m car (380 ms to build, 1,508 ms to query; its SAH build queries in 14 ms but takes
   17 s to build). ISLA's own cell tree (63-bit Morton codes in the canonical frame) does not degenerate.
2. **A prototype that is bitwise equal to the reference runs.** It keeps the reference's arithmetic and changes only how
   the candidate pairs and the winding sum are produced: the same tree with its nodes in depth-first preorder, built by
   one kernel; a float32 nearest-cell seed on that tree in place of the Warp mesh query; the reference's ball test and
   winding traversal on packed node records. On every 3-D dump (DrivAerML Volume vehicle and whole boundary, AeroZoo,
   three lug cases; float32 and float64), all six outputs of the search and `w` are bitwise equal: 0 differing
   queries of 200,000 per DrivAerML and AeroZoo case. Over the geometry and tree-read test files on CPU, 0 of 146,645
   queries differ in 136 compared calls. Op time against the reference: 292 vs 308 ms (DrivAerML vehicle), 342 vs 2,358
   (whole boundary), 109 vs 123 (AeroZoo), 27 vs 31 (lug).
3. **The traversals cannot be done well in Triton, so I propose Warp for them, and Triton for the elementwise and
   scan phases.** A naive Triton winding traversal is bitwise equal to Warp's but 10x slower: Triton cannot branch per
   lane, so every step pays for the rare float64 solid angle. With block-level branches it reaches Warp's speed on
   AeroZoo and the lug (1.0x and 0.8x; DrivAerML 1.3-1.5x), but `w` is then no longer bitwise (max |dw| 1e-8; the
   compiler contracts the arithmetic differently). A bitwise `w` is required: on a triangle soup `w` sits at exactly
   1/2, so a rounding change can flip the sign there. A Warp traversal compiles the same source as the reference, so it
   is bitwise by construction.
4. **Expected saving (section d):** call 339 to about 210 ms on DrivAerML vehicle (-38 %), 2,376 to about 270 ms on the
   whole boundary (8.8x; below the base's 2,057), 136 to about 80 ms on AeroZoo (-40 %), 47 to about 25 ms on the lug
   (-45 %), and 26.5 to about 17 ms on AirfRANS if the optional 2-D kernel is built. The remaining gap to the base
   on the vehicle (about 120 ms) is the winding number's far field and the float64 ball test, which the reference's
   numerics fix.
5. **Decisions needed:** listed in section h, each with the answer I recommend.

## a. Where the time goes at f13eb5ce0

Per-phase times from `prof_phase.py`'s synchronizing line tracer (traced sum is 5-30 % above the untraced call);
tables in `lhr/phase_tables_p1.md`, from LHR 359447 (f13eb5ce0) and 359546 (base).

| phase (ms) | DML vehicle | DML whole | AeroZoo | lug 0004 | AirfRANS 2-D |
|---|---|---|---|---|---|
| Q / F | 200k / 16.65M | 200k / 16.65M | 200k / 1.35M | 10.5k / 6.7k | 100k / 1,017 |
| **call, untraced median** | **338.8** | **2,376.3** | **136.1** | **46.8** | **26.5** |
| base call (code_cmpaf_fix) | 91.9 | 2,056.5 | 18 (q.4) | 8 (q.4) | 12 (q.4) |
| weld + canonical cells | 28.8 | 28.9 | 7.3 | 5.9 | 2.0 |
| float64 normals and copies | 7.6 | 7.6 | 1.4 | 1.1 | 0.9 |
| dedupe of identical cells | 18.1 | 18.2 | 2.4 | 1.3 | 0.8 |
| seed query (Warp mesh BVH + float64 bound) | 53.6 | **2,012.8** | 13.4 | 6.4 | - |
| tree build + escape pointers | 39.5 | 38.6 | 16.8 | 12.2 | - |
| winding moments | 21.3 | 21.2 | 5.2 | 3.8 | 0.5 |
| joined ball + winding traversal | 156.3 | 235.1 | 93.7 | 8.8 | - |
| exact phase (incl. least-norm, 2-D candidates) | 34.6 | 34.6 | 13.2 | 19.8 | 17.7 |
| winding sign (2-D: the brute-force sum) | 0.2 | 0.2 | 0.3 | 0.3 | 11.4 |
| host syncs / kernel launches | 76 / 2,776 | 80 / 2,798 | 52 / 2,005 | 64 / 2,115 | 36 / 869 |
| peak GiB | 18.3 | 18.3 | 1.5 | 0.1 | 0.2 |

What the traversal pays for (`probe_layout.py`, an instrumented copy of the joined kernel; per query, mean / p99):

| | DML vehicle | DML whole | AeroZoo | lug |
|---|---|---|---|---|
| nodes entered | 13,864 / 21,529 | 18,112 / 27,759 | 12,573 / 14,527 | 1,162 / 1,427 |
| winding far-field nodes | 6,766 / 10,739 | 8,958 / 13,835 | 6,189 / 6,886 | 571 / 701 |
| winding near leaves (float64 solid angles) | 2.1 / 18 | 22.6 / 39 | 4.0 / 19 | 7.9 / 21 |
| ball leaves tested / float64 distances | 60 / 25 | 34 / 22 | 51 / 3.2 | 8.2 / 5.8 |
| candidates | 3.8 / 10 | 5.4 / 21 | 1.5 / 6 | 5.7 / 12 |

The winding number's far field is most of the node visits: about 7,000 expansions per query at beta = 2, fixed by
the tree and beta, so it cannot shrink without changing `w`. The ball test visits few nodes but pays float64
distances. Separately timed on the reference's layout (DrivAerML vehicle): ball only 77 ms, winding only 124 ms, joined
154 ms.

The seed on the whole boundary (`probe_seed.py`, LHR 359546): Warp's constructors on the same float32 mesh.

| constructor | DML whole build / query | DML vehicle build / query | AeroZoo build / query |
|---|---|---|---|
| default (LBVH) | 380 / 1,508 ms | 18.6 / 22.9 | 1.8 / 5.3 |
| sah | 17,294 / 13.9 | 15,942 / 15.3 | 466 / 3.9 |
| median | 13,679 / 16.5 | 13,841 / 18.1 | 421 / 4.4 |

The base has the same seed (`_closest_cell`), so FLOOR-DRV's 1.58x whole-boundary step at 0.9.0 is mostly this query.

## b. What the fast path replaces

The fast path is a second custom op, `physicsnemo::isla_closest_candidates_fast`, with the reference op's signature,
outputs and fake. It replaces the steps that produce the candidate pairs and the winding sum, and the mesh-only
preparation where a kernel gives the same integers. Everything that decides an output stays the reference's code.

| reference step | fast path | why it is the same |
|---|---|---|
| `_cell_tree`: Morton order, `build_lbvh_topology` (host level loop), parents, escape pointers, boxes per level | the same Morton order and midpoint topology, its nodes numbered in depth-first preorder by one Warp kernel (each sorted position descends from the root and writes the node whose range starts there); boxes and moments by the reference's bottom-up kernels | same tree, renumbered: every traversal visits the same nodes in the same order |
| node arrays in separate tensors, cells in input order | one packed record per node (box, skip pointer, cell; and centre, far radius, dipole, spread), the left child at `node + 1`, cells and corners in leaf order | an index permutation of the inputs |
| seed: `signed_distance_field` (Warp mesh BVH built per call, float32 query) | a float32 nearest-cell search on the cell tree (stack, nearer child first), then the reference's float64 bound at that cell | any cell bounds the closest distance (below); this one is near, as Warp's is |
| joined ball + winding traversal | the reference's ball kernel and winding kernel, separately, on the packed records, queries in Morton order | same tests, same arithmetic, same order |
| weld (3 argsorts, cummax), canonical cells (27 `torch.where` launches), dedupe (4 argsorts) | Triton: one scan kernel for the weld's run starts, one elementwise kernel for the canonical order and parity; dedupe by two stable sorts of packed keys | integer and comparison operations only: same integers |
| exact phase (about 60 torch ops, 30 syncs; least-norm by K groups) | Triton, one kernel per query over its candidate slots for the distances, bounds, first minimum, on-wall, feature keys, valence test, tie test and directions, with `enable_fp_fusion=False` and torch's operation order; `_grouped_min_norm` stays torch | bitwise if the kernel rounds as torch's separate kernels do: measured, and kept only if bitwise (else the torch exact phase is shared code) |
| 2-D (optional): torch brute force in chunks, Warp winding sum | one Warp kernel per query: float64 candidates within the 3-D radius formula, then the reference's winding expression in cell order | as 3-D; the winding sum is the same expression in the same order |

Unchanged and shared: `_closest_features`, `exact_boundary_geometry` after the op, `_grouped_min_norm`, `_winding_sign`'s
threshold, the float64 normals, every output dtype. The op has no gradient; `d` and `u` get theirs from the shared
torch code after it, so equal op outputs give equal gradients.

**Why fewer candidates give the same outputs.** The reference searches within `R = B + 3 max_j E_j` (times 1 + 1e-12),
where `B` is the float64 distance to the seed cell plus that cell's bound, so `B >= D_min`. Its outputs read only the
pairs that can tie, `D_k - E_k <= min_j (D_j + E_j)` (so `D_k <= D_min + 2 max E`), and, for the valence test, every
cell that contains such a pair's closest point (distance at most `D_k`). All of them are within `D_min + 3 max E <= R`
for every seed, so any seed gives the same outputs; a near seed only keeps the pair count low.
The same padding as the reference (box test, float32 lower bound, overflow pass) is kept, and a NaN distance (a
degenerate cell) is kept as there.

## c. Prototype results (scratch code, not the branch)

`lhr/fastproto.py` (preorder tree, packed winding, and a variant with a shrinking-radius nearest search) and
`lhr/fastproto_b.py` (variant B: float32 seed, reference radius, packed ball). `probe_fast.py` runs the reference op and
the prototype on the same inputs and counts the queries where any output differs bitwise; it prints the md5 of the
reference geometry.py it ran (the prototype registers no op). LHR 359652 (first variant) and 360424 (B).

| case | precision | queries | queries differing (any of 6 outputs) | w differing | reference op | variant B op |
|---|---|---|---|---|---|---|
| DML vehicle | f32 / f64 | 200,000 each | 0 / 0 | 0 / 0 | 308 / 308 ms | 292 / 284 |
| DML whole boundary | f32 / f64 | 200,000 each | 0 / 0 | 0 / 0 | 2,359 / 2,307 | 342 / 318 |
| AeroZoo | f32 / f64 | 200,000 each | 0 / 0 | 0 / 0 | 124 / 123 | 109 / 108 |
| lug 0004, 0011, 0012 | f32 / f64 | 29,473 each | 0 / 0 | 0 / 0 | 30-32 | 27 |
| geometry + tree-read tests (CPU, `cmp_plugin.py`) | both | 146,645 in 136 calls | 0 | 0 | | |

Not yet covered: the tie set as a set of feature ids (the probe compares the tie flag and direction, a function of the
set), the 17 remaining lug cases, Tensile2d, AirfRANS, the adversarial sets, gradients, checkpoints (section e). 99 of
the 235 test calls were not compared: 96 are 2-D and 3 have repeated cells, which the prototype does not handle yet.

Prototype pieces, DML vehicle / whole / AeroZoo / lug (ms): preorder tree 25 / 25 / 3.1 / 2.7 (reference tree
31 / 31 / 11 / 8); seed32 27 / 21 / 21 / 1.0; packed ball at the reference radius 56 / 75 / 9.3 / 2.6 (reference ball
only 77 / 108 / 14 / 3.0); packed winding 61 / 104 / 55 / 2.9; packing the winding records 27 / 27 / 2.7 / 0.4 (to be
written by the moments kernel instead). The preorder layout alone speeds the reference's joined kernel 154 to 145
(DML), 232 to 214 (whole), 92 to 74 (AeroZoo), with pairs and `w` bitwise.

Triton winding traversal (`triton_spike.py`, LHR 359942 and 360066; time / w bitwise against the Warp kernel):

| | Warp | Triton, naive (fp fusion on) | Triton, block-level branches (fp fusion on) | same, fp fusion off |
|---|---|---|---|---|
| DML vehicle | 61 ms | 749 ms, bitwise | 91 ms, max abs dw 1.4e-8 | 92 ms, 1.8e-5 |
| DML whole | 104 | 926, bitwise | 132, 1.3e-8 | 140, 1.0e-4 |
| AeroZoo | 55 | 568, bitwise | 56, 9.1e-9 | 55, 2.4e-6 |
| lug | 3.1 | 6.6, bitwise | 2.4, 5.1e-9 | 2.6, 3.2e-8 |

The differences are float32 rounding of the far-field terms, from where the compiler contracts multiply-adds, which
can change between Triton versions (LHR has 3.7.0, this machine 3.7.1).

## d. Kernel plan and expected saving

Module `physicsnemo/experimental/models/isla/geometry_fast.py`: Warp kernels for the tree, seed, ball and winding
(Warp is already the geometry's dependency), Triton kernels for the weld scan, the canonical cells and the exact phase,
loaded through importlib with `HAS_TRITON` as in geo_kernel.py. The reference stays in geometry.py; `_closest_candidates`'s
exact phase moves into a helper both ops call (bitwise checked against f13eb5ce0), unless the Triton exact phase is
bitwise, in which case only the fast op uses that kernel.

Steps, each committed with its tests and checks:

1. The exact phase as a shared helper; reference bitwise against f13eb5ce0 (CPU and CUDA, refload.py).
2. The fast op with the Warp pieces (tree, packed records written by the bottom-up kernels, seed, ball, winding),
   repeated cells handled as the reference does (two trees). The selection (section f).
3. Triton weld scan and canonical cells; Triton exact phase if bitwise.
4. Optional: the 2-D kernel (AirfRANS, Tensile2d).
5. The equivalence campaign (section e) and the final timing (section g).

Expected call time, from the measured pieces (traced-equivalent sums scaled to the untraced call; basis in brackets):

| phase (ms) | DML vehicle | DML whole | AeroZoo | lug |
|---|---|---|---|---|
| weld + canonical (Triton scan and sort network) [the integer ops, a few passes over 8M points] | 29 to 5 | 29 to 5 | 7 to 1.5 | 6 to 1 |
| dedupe (two stable sorts) | 18 to 10 | 18 to 10 | 2.4 to 1.5 | 1.3 |
| seed [seed32 measured] | 54 to 27 | 2,013 to 21 | 13 to 15 | 6.4 to 4 |
| tree + moments + records [preorder 25; records in the moments kernel] | 61 to 40 | 60 to 40 | 22 to 6 | 16 to 5 |
| ball + winding [packed, measured separately] | 156 to 117 | 235 to 179 | 94 to 65 | 8.8 to 5.7 |
| exact phase [fused kernel; least-norm stays torch: 14 / 14 / 2.3 / 9.8] | 35 to 20 | 35 to 20 | 13 to 5 | 20 to 12 |
| **call** | **339 to about 210** | **2,376 to about 270** | **136 to about 80** | **47 to about 25** |

Training step, if the step falls by the call saving, as it did in q.4: DrivAerML vehicle about 850 to 720 ms (base 570, so
+49 % to about +27 %), AeroZoo about 590 to 535 (+17 % to about +6 %), the lug about 180 to 160 (+29 % to about +14 %).
The whole-boundary step has not been measured at f13eb5ce0; it should fall by about 2.1 s.
These are estimates. The exact-phase and weld lines are the least certain (the Triton bitwise requirement may keep them
in torch, which would leave about 40 ms on DrivAerML and 15 ms on the lug). Peak memory should not exceed the
reference's (no Warp mesh copy, one tree); it will be measured.

What the fast path cannot remove under the rule that the numerics stay the reference's: the far-field winding sum (61-104
ms at DrivAerML size; it is beta = 2 on this tree, float32 multipoles with float64 offsets), and the float64 ball test.
Fewer visits need a coarser far field (DESIGN q.3 measured wider trees and leaves: slower) or a dual-tree traversal,
which changes `w` within its far-field error; I do not propose either.

## e. Correctness criteria, concrete

All checks compare the fast op with the reference op on the same CUDA device in one process: the package's reference
op and the fast op have different names (`isla_closest_candidates`, `isla_closest_candidates_fast`); every probe prints
the md5 of the geometry files it ran and which op ran each side (LANE_RULES). Counts, not samples.

C1. **The reference is unchanged.** At each commit, the package's reference path against f13eb5ce0's geometry.py
(loaded by refload.py): `d`, `u`, their gradients in the queries and the vertices, every op output and `w`, bitwise, on
the 52 cases of `probe_speed_bitwise.py` on CPU (single-threaded) and CUDA, and on the 3-D dumps on CUDA.
Zero differences required.

C2. **Discrete decisions, every query, zero tolerance:** the closest cell id, `tied`, `on_wall`, the sign of `d`, and
the tie set as the set of (query, cell) pairs that pass the tie test, captured inside the exact phase on both sides and
compared as sets per query. Also the set of features (vertex or edge key, or cell interior) of those pairs.

C3. **Continuous outputs:** the closest-point weights (float64), the tie direction, `d`, `u` and `w`: bitwise required for
the Warp pieces and for `w` (a soup puts `w` at exactly 1/2). If a Triton exact-phase kernel is not bitwise it is not
used; the fallback is the shared torch code. So the bound is 0 ulp, measured as the count of queries with any bit
different, and the max abs and max ulp difference if any appear.

C4. **Gradients:** of `d` and `u` in the queries and the vertices, float32 and float64, bitwise on the 52 cases (CUDA,
fast forced) and on one DrivAerML, AeroZoo and lug dump at 20k queries.

C5. **Case sets, each reported as counts of queries and of disagreements, float32 and float64:**
- the test meshes: the whole ISLA test directory on GB300 with a plugin (as `cmp_plugin.py`) that runs both ops on CUDA
  at every geometry call (icosphere, ellipse, torus, open cube, soups, the finned cube, boxes, NACA, refined and split
  meshes with repeated cells);
- adversarial sets (`adv_sets.py`, new): vertices, edge midpoints and centroids of an icosphere, a raw torus, a box and
  a soup, posed and unposed; centroids moved by E/4 and 4E along the normal; near-duplicate normals (a refined flat
  patch with 1e-9 perturbations); medial-axis queries (box and plate mid-planes, cube diagonals, the torus core circle);
  zero-area cells; queries in float32 at offsets 1 and 1e3; repeated and opposite coincident cells;
- real dumps: the 20 lug cases, 4 AeroZoo cases, 2 DrivAerML Volume vehicle and 2 whole-boundary cases at 200k
  queries; Tensile2d and AirfRANS (4 cases each) once the 2-D kernel exists, else reported as reference-only.
Pass: zero disagreements in C2 and C3 in every set.

C6. **Checkpoints:** one trained 0.9.0 ISLA checkpoint per dataset (AirfRANS, Tensile2d, lug, AeroZoo, DrivAerML
Volume), one case each, evaluation forward under `geometry_kernel="reference"` and `"fast"`: outputs equal to float32
roundoff (expected bitwise, since the geometry is); max abs and relative difference reported.

C7. **Tests:** the whole ISLA test directory on CPU at each commit (auto resolves to the reference there), with the
assertion messages of any failure compared with f13eb5ce0's; on GB300 the geometry and tree-read files with the fast
path forced and under auto. New tests (`test_isla_geometry_fast.py`), run whenever CUDA and Triton are present: the
fast op against the reference on the test meshes and the adversarial sets (bitwise), repeated cells, an empty
candidate overflow, the selection (auto, forced, logged once), and the torch.compile fake.

A speedup counts only at a commit where C1-C7 pass.

## f. Selection

As `resolve_geo_kernel`: a writable ISLA attribute `geometry_kernel` ("auto", "reference" or "fast"), resolved at the
first forward pass per device, cached, logged once; under auto the fast path when the points are on CUDA, Triton
imports (the HAS_TRITON of the new module) and the dimension is 3 (2 too if the 2-D kernel is built), else the
reference; a forced value overrides it. `exact_boundary_geometry(..., kernel="reference")` takes the resolved name; the
model passes it. Not in the state dict: no version bump, `_refuse_earlier_state_dict` unchanged.

## g. Final timing (phase 2)

The geometry call and the training step at the base (`code_cmpaf_fix`), f13eb5ce0 (`code_dvcost2a`) and the head, on
AirfRANS, the lug, AeroZoo, DrivAerML Volume vehicle and whole boundary (recipe_r090_fd, isla_volume_floor), with peak
memory: q.4's harness (`timed_train.py`), all versions on one node, two passes with the GPUs rotated, plus the
equivalence checks at the same commit. About 2 node-hours of short QoS in all; phase 1 used 1.05 GPU-h.

## h. Decisions needed (recommended answer first)

1. **Warp for the traversals, Triton for the elementwise and scan phases** (section c), instead of all kernels in
   Triton. Recommended: yes. All-Triton would give a non-bitwise `w` or a 10x slower traversal.
2. **Gate:** auto picks the fast path on CUDA when Triton imports, as Peter's ruling says. If the Triton pieces turn out
   not to be bitwise and are dropped, the fast path would be Warp only; I would then gate it on CUDA alone and say so.
   Recommended: gate on Triton as long as a Triton kernel is part of the path.
3. **The reference's own seed.** The 2 s whole-boundary seed is a performance defect of the reference (and of the
   base), not of its numerics. I recommend leaving the reference as it is (it stays the definition and the CPU
   fallback, and training runs on CUDA), and fixing it only in the fast path. The alternative, the float32 seed on the
   cell tree in the reference too, gives the same outputs bitwise and would also speed the reference on CUDA.
4. **Scope:** the 2-D kernel (AirfRANS +7 %, about 10 ms of 26) last and optional. Recommended: yes.

For the program (FLOOR-DRV, BOUNDARY-AUDIT): any whole-boundary DrivAerML Volume run, at any ISLA version, pays about
2 s per geometry call until a fast path lands; at the base this is most of FLOOR-DRV's 1.58x step.
