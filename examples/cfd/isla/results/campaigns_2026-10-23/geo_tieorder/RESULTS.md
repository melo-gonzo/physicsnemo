# GEO-TIEORDER results (2026-10-25)

Head = `$T/code_tieorder_head1`: fc6ed19d0 plus the branch's three changed files (geometry.py 9e9c0242,
test_isla_geometry_band.py eede1b84, test_isla_interior_geometry.py 4038c39a; committed as listed in FINAL_REPORT.md).
Base = fc6ed19d0's geometry (`$T/code_drawexact_base`, geometry.py e02fea5a). Every comparison runs each version in its
own process (tieorder_probe.py prints the geometry.py md5 and the op that ran).

## 1. The head is the prototype, and fast is the reference (GB300 job 378186, `out/final_378186/bitwise.md`)

Prototype (7fdd06e7, the design's measured rule) against head, every dump, float32 and float64: 0 queries differ in d,
u, sign, tied or on-wall. Kernel fast against reference at the head: bitwise on every dump and precision (d, u, tied,
on_wall). The writers' CPU checks add d, u and both gradients on 32 cases x 2 kernels (REPORT_WRITER2.md).

## 2. The function change, base -> head (job 378186, `out/final_378186/change.md`; HiLift: cpu-short 378382, `out/hlcrop_378382/hilift.md`)

float32 (the recipe's precision), reference kernel:

| dataset (dump, queries) | on wall | u changed: count (on wall), max, median | d changed | sign / wall-label changes | renumbering moves base -> head | float32 vs float64 mismatches (on the wall in both) base -> head |
|---|---|---|---|---|---|---|
| DrivAerML whole boundary s42 (200k) | 12,888 | 3,094 (3,087), 4.5e-3, 2.8e-4 | 0 | 0 / 0 | 2,577 -> 0 | 3,087 of 12,888 -> 0 |
| DrivAerML s43 (200k) | 12,821 | 3,217 (3,212), 4.4e-3, 2.8e-4 | 0 | 0 / 0 | 2,692 -> 0 | 3,212 of 12,821 -> 0 |
| AirfRANS s42..s45 (100k each, 2-D) | 546-584 | 70-93 (all), 1.6e-4, 0.6-1.1e-4 | 0 | 0 / 0 | 39-48 -> 0 | 70-93 -> 0 |
| lug, 20 test cases (6-13k each) | 2,532-3,856 | 1-22 per case (0), 6e-8 (most 2e-16) | 2 queries in one case (1.2e-10) | 0 / 0 | 1-15 -> 0 | 0 -> 0 |
| AeroZoo (200k; none on the wall) | 0 | 0 | 0 | 0 / 0 | 0 -> 0 | n/a |
| Tensile2d (11k, 2-D) | 438 | 0 | 0 | 0 / 0 | 0 -> 0 | 0 -> 0 |
| HiLift, two cases, spanwise slab of about 2.4M cells, 25k vertex + 25k centroid queries each (CPU) | 50,000 | 19,561 and 21,215 (all; 19,177 and 20,834 at vertices, about 380 at centroids), max 1, median 1.2-1.3e-3 | 0 | 0 / 0 | 14,571 and 16,415 -> 0 | 19,189 of 25,370 and 20,834 of 25,356 -> 271 and 183 |

float64 queries (the float32 dumps cast up): DrivAerML u changes at 5.3-5.5k queries by at most 1.6e-15 and d at about
850 by at most 2.8e-17 (all off the wall, T1's rebuild about a canonical cell); renumbering moves 5,543 and 5,216 -> 0.
Lug: u at 18-80 queries per case, mostly by at most 2.8e-16 and by up to 1.8e-8 at a few, all off the wall; d by at most
3.5e-18; renumbering moves 14-64 per case -> 0. AeroZoo: u at 135 (<= 2.5e-16), d at 25 (<= 2.8e-17); renumbering 129 -> 0.
AirfRANS and Tensile2d: no change.

The certified case is unchanged: at every query the base already ties, u is bitwise the head's, on every dump and in
both precisions (`out/final_378186/certified_same.txt`, GB300).

HiLift: the canonical HiLift task (surface, option A) never calls the geometry. The full-resolution wing (0.27 billion
cells, 0.4 billion vertex and centroid queries per case; REPORT_HILIFT_DUMP.md) runs out of GPU memory on a GB300 in
**the base as well as the head** (`_winding_moments`, 264 GiB in use; job 378186, `out/final_378186/hilift.log`).
So the measurement runs on a spanwise slab of about 2M wing cells plus the far field (probes/hl_crop.py; cpu-short
378382), on CPU, base and head on the same crop. Fast equals reference there too (CPU).

HiLift is where the change is largest, and it shows the base defect at its worst:
- In the base, renumbering moved u at 29-33 % of the slab's queries.
- At the head, u moves by more than 0.1 at about 1,000 queries per case, by 1e-2 to 0.1 at 1,100-1,600, and by 1e-3 to
  1e-2 at about 9,000, almost all at vertices (`out/hlcrop_378382/hilift.md`, `big_LHC002.txt`; probes/hl_big.py).
- Of the queries that move by more than 0.1, 876 of 1,078 and 719 of 983 have a member cell whose stored float32
  vertices leave its normal undetermined (tau = 2, s >= 1, a sliver). 141 and 118 have two opposite-facing normals (a
  fold). The base reported one member, chosen by cell order. The head reports their least-norm point, the Clarke value
  of the stored mesh. |u| < 0.9 goes from 409 to 880 and from 323 to 723 queries of 50,000.
- The remaining 271 and 183 precision mismatches are all queries whose float32 on-wall group strictly contains the
  float64 group, which section 3(i) of DESIGN.md predicts (probes/hl_groups.py, local CPU: groups equal 0, float64
  strictly inside float32 271 of 271 and 183 of 183).
- One float64 query (a centroid 1.3e-10 off the wall) still flips the sign of d under renumbering. That is the winding
  sign (DESIGN.md section 1, last row): its BVH far field depends on Morton ties. It is outside T1 and T2.
- Pose on the slab, on the wall in both poses: max |u_posed - R u| 1.98 -> 1 and 2 -> 1; RMS 0.092 -> 0.072 and 0.095 ->
  0.078; queries > 1e-3: 17,113 -> 16,077 and 18,632 -> 19,055; tied flips 4,710 -> 397 and 3,349 -> 866.

## 3. Pose (GEO-BAND's protocol: a fixed rotation and a translation by the boundary's RMS radius, float32; job 378186)

On the queries on the wall in both poses, base -> head:

| dataset | max \|u_posed - R u\| | queries > 1e-3 | tied flips | off-wall tied flips |
|---|---|---|---|---|
| DrivAerML s42 | 1.45e-2 -> 3.1e-3 | 352 -> 17 | 630 -> 0 | 2,492 -> 2,492 |
| DrivAerML s43 | 1.02e-2 -> 2.3e-3 | 425 -> 17 | 692 -> 2 | 2,472 -> 2,472 |
| AirfRANS s42..s45 | 0.39-0.84 -> 0.30 | 392-431 -> 235-271 | 453-499 -> 28-40 | unchanged |
| lug, 20 cases | 2.2e-5 -> 2.2e-5 (RMS down slightly) | 0 -> 0 | 0 -> 1,785-2,821 per case | unchanged |
| Tensile2d | 9.2e-6 -> 4.7e-6 | 0 -> 0 | 0 -> 279 | 0 -> 0 |
| AeroZoo | no query on the wall | | | 6 -> 6 |

u becomes more pose-stable everywhere. The tie *flag* now flips on planar on-wall regions of the lug and Tensile2d (the
review's finding 2): their normals are bitwise equal in one pose and differ by rounding in the other. The flag reaches
the outputs only through u's gradient in the vertices (none at a tie). On DrivAerML and AirfRANS the flag flips fall,
because the base's certification itself flipped under pose there.

## 4. Inference check (child ea41ae15, GB300 job 377708; `infer/INFER_CHECK.md`)

No 0.12.0-trained checkpoint exists. GEO-BAND's set was rescored with old (fc6ed19d0) and new (prototype) geometry:
TIE-EXP's lug seeds 42/43, AF-WHOLE seed 42 at epochs 197 and 500, and FLOOR-DRV fdF_s42 (DrivAerML, one case, step
9000). Largest field |log(N / O)|: DrivAerML 2.7e-4 (held-out pressure), AirfRANS 6.5e-5, lug 6.7e-7. Largest per-case:
AirfRANS 1.18e-3 (one case of 200), lug 3.0e-4. Largest posed / native move: 5.1e-5. No field exceeds the 1e-3 bar.

## 5. Cost (GB300; GEO-THINFAST's timing harness, bf16 training, interleaved passes, 4 runs per node)

| case (Q / F) | call, fast: base -> head [ms] | call, reference | training step, fast [ms] | job |
|---|--:|--:|--:|---|
| lug (9,808 / 6,488) | 36.1 -> 38.1 (+5 %) | 85.9 -> 95.0 (+11 %) | 243 -> 254 (+5 %) | 378187 (head) |
| lug, prototype | 37.6 -> 40.7 (+8 %) | 85.1 -> 90.2 (+6 %) | 252 -> 249 (-1 %) | 377673 |
| AirfRANS 2-D (100,000 / 2,746), prototype | 72.1 -> 75.4 (+5 %) | 73.9 -> 82.4 | 538 -> 549 (+2 %) | 377673 |
| AeroZoo (200,000 / 501,802), prototype | 90.6 -> 92.4 (+2 %) | 131.7 -> 147.5 | 650 -> 652 (0 %) | 377673 |
| DrivAerML Volume (200,000 / 16,958,756) | 399.6 -> 407.7 (+2 %) | 514.5 -> 520.0 (+1 %) | 1,242 -> 1,256 (+1 %) | 378187 (head) |
| DrivAerML Volume, prototype | 405.4 -> 408.2 (+1 %) | 513.0 -> 522.6 | 1,259 -> 1,261 (0 %) | 377915 |

The fast call costs 2-8 ms more (its passes differ by up to 2 ms). The lug's training step moves by -1 % and +5 % in
the two jobs, where its passes differ by up to 9 ms. So the step cost is at most about 5 % on the lug and 1 % on
DrivAerML. Tables: `out/timing_proto_table.md`, `out/timing_head1_table.md`.

## 6. Suites

At the merged head 3a1a15d48 (psharpe/isla b86c7391d, ISLA 0.13.0, merged in; geometry files unchanged by the merge),
`$T/code_tieorder_merge1`:
- CPU, the whole ISLA directory (cpu-short 378676): 468 passed, 250 skipped, 0 failed.
- GB300, the whole ISLA directory with every CUDA test, under auto (job 378687): test_isla_compile.py 8 + 4 passed; the
  rest 711 passed, 1 skipped, 0 failed.

Before the merge, at code_tieorder_head1 (0f26e6db7's code):

- CPU, the whole ISLA directory (cpu-short 378185): 476 passed, 251 skipped, 0 failed (16 min).
- GB300, the whole ISLA directory with every CUDA test, under auto (job 378186, two GPUs): test_isla_compile.py 9 passed
  ("not cpu") + 5 passed ("cpu"); the rest of the directory 720 passed, 1 skipped, 0 failed.
- Locally (writers): the four geometry files 143 passed, 62 skipped; pre-commit clean on the three files.
