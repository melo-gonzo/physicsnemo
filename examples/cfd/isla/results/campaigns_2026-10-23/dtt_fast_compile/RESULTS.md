# DTT-FAST piece COMPILE: the compiled path of ISLA-DTT (results)

Written 2026-10-24 03:40 EDT. Reference f13eb5ce0 (LHR snapshot `$T/code_dvcost2a`, md5-checked against git:
`lhr/*_physicsnemo.md5`; only `segment_softmax.py` differs, in the Triton import, no runtime change). GB300, torch
2.12.0+cu130, Triton 3.7.0. Jobs and GPU-h (4.79 of 6): `JOBS.md`. Probe `lhr/cx.py` (runtime overlays, reference
functions called unchanged; each JSON line names the md5 of the model files it ran). Tables: `tables_363857.md`,
`tables_364478.md`, `tables_365108.md` (certification), `timing_all.md` (model-step timing), `recipe_steps_365234.md`,
`recipe_steps_365375.md` (real training steps).

## Recommendation

1. **Adopt `compile_read.patch`** (tree_read.py, model.py, one test helper; 229 lines): under `compile_blocks` the
   read also compiles its point-side and cell-side LayerNorm with the key, value and query maps and the output
   projection (P), `node_means` as three graphs reused for every forest depth (M), and the pair invariants with the
   basis MLP per chunk (B), each with `dynamic=True` and `{mix_order_reduction: False, emulate_precision_casts: True}`.
   The reference path is the `compiled=False` branch, unchanged; selection follows `compile_blocks` (an execution
   choice, not in the state dict). Expected saving: **3-7% of the model step and 1-8 GiB of peak**, against the current path in the
   same processes (DrivAerML vehicle 200k -4.5%, 181.5 -> 173.3 GiB; AeroZoo -6.6%, 157.4 -> 149.7 GiB; AirfRANS
   -3.7%, 67.8 -> 65.9 GiB; lug -3.1%), and 3-14% of real recipe steps measured side by side. Certified to the bound wherever the current compiled path is (section 2), whole
   ISLA test directory 321 passed / 46 skipped on CPU (cpu-short 365063). Bounded recompiles: 60 real training steps
   per dataset over changing cases add no graph after step 10.
2. **Decision for the parent: the current compiled blocks move bf16 rounding points.** Inductor by default elides
   the bf16 round trip between fused ops (`emulate_precision_casts=False`), so `compile_blocks` today computes a
   (slightly more precise) different bf16 function from eager. It passes the accuracy bound (its distance to float64
   equals eager's to within 1.0-1.4x in every tensor group), so it is not a correctness defect by that standard, and
   every trained result used it. `block_precision_casts.patch` (one line) restores eager's rounding points: bf16 block
   outputs then match eager to 1 bf16 step (AeroZoo block 0: 5.8e-2 -> 5.0e-3 relative; parameter gradients vs eager
   1.8e-1 -> 3.5e-2), free on 3-D cases (fused geometry region: -0.3 to -1.3%) but **+11 to +14% step time on
   AirfRANS** and +5% on the lug (eager geometry region inside the compiled block). I recommend **not** adopting it:
   it buys no accuracy, costs 2-D time, and changes the function the existing checkpoints were trained under. The
   read patch keeps eager's rounding points either way.
3. Not compiled, and why: the forest build, walk and plan (data-dependent sizes, host-bound: pieces FOREST/WALK);
   `segment_softmax_read` (already Triton) and the kernel term `basis @ W` (one GEMM; FUSEDREAD removes it); the
   geometry call (GEO-FAST); the head and the mesh-side prologue (shared with ISLA, within PROFILE's 66 ms "rest";
   not tried).

## 1. What is compiled in an ISLA-DTT training step at f13eb5ce0

`compile_blocks` compiles only `_SliceBlock.forward` (x12; the fused geometry kernel is a custom-op leaf, the eager
geometry region of 2-D cases is traced inside the block). Everything else runs eager. Device ms per step from
PROFILE (`../profile/PROFILE.md`, DrivAerML vehicle 200k, bf16, eager blocks) and the measured effect of compiling it:

| region | where | device ms (PROFILE) | compiled here? | measured effect (in-process, `timing_all.md`) |
|---|---|---|---|---|
| 12 slice blocks | blocks.py | 641 (363 compiled) | yes, `compile_blocks` | E -> C: -12% (DrivAerML 200k), -17% (AeroZoo), -36% (AirfRANS), -14% (lug) |
| geometry call `exact_boundary_geometry` | geometry.py | 327 (2343 whole boundary) | no (GEO-FAST) | - |
| forest + walk + PairPlan | tree_read.py | 103 (+~160 host idle) | no: data-dependent, host-bound | - |
| pair invariants + basis MLP (B) | `_chunked_basis` | ~170 of 296 | patch | -0.3 to +0.6% alone; -0.6 to -1.7% on top of P+M (job 365445) |
| kernel term `basis @ W` | `_TreeKernelRead` | 126 of 296 | no (one GEMM; FUSEDREAD) | - |
| `segment_softmax_read` | segment_softmax.py | 431 | no (Triton) | - |
| `node_means` (M) | tree_read.py | 52, 8088 launches | patch | -1.2 to -2.3% (one graph per depth, N, recompiles; M does not) |
| read LayerNorms, q/k/v, out (P) | `_TreeKernelRead` | 47 | patch | -1.1 to -1.5%, -7 GiB peak (fewer saved float32 copies) |
| head, frame, loss, prologue | model.py | ~66 (shared with ISLA) | no (not tried) | - |

## 2. Certification (compiled vs eager, block by block, on real ISLA-DTT checkpoints)

Checkpoints (read-only): AeroZoo `azc_full_dtt_seed{42,43}` epoch 250, AirfRANS `r090eag_af_dtt_seed{42,43}` epoch
500; plus DrivAerML vehicle and whole boundary at random init (no trained DrivAerML ISLA-DTT exists). Inputs: PROFILE's
first training batch of each dataset. Two instruments:

- **Isolated, full size** (AeroZoo s42/s43 200k points, 53.8M pairs; AirfRANS s42/s43 all 180,593 points): each
  block, each read and the basis on the eager float32 run's own captured inputs (the same tensors for every arm),
  float32 and bf16, beside float64 eager on the same inputs (the read's float64 on 20,000 points, exact for those
  rows by query independence; the basis's on 2 chunks). Then each arm's own accumulated forward.
- **Gradients, 20k tokens and points**: loss, outputs and every parameter gradient (4 groups), float64, float32 and
  bf16 autocast; pair lists (sorted, `torch.equal`, node counts) against eager.

Current path (C) and the patch, max over blocks (relative to the reference's max abs):

| case | prec | eager vs f64 | C vs eager | C vs f64 | patch read (P+M) vs eager | patch basis vs eager | eager run-to-run |
|---|---|---|---|---|---|---|---|
| AeroZoo s42 blocks | f32 | 8.5e-4 | 1.5e-5 | 8.5e-4 | - | - | 0 |
| AeroZoo s42 blocks | bf16 | 1.4e-1 | 5.8e-2 | 8.2e-2 | - | - | 0 |
| AeroZoo s42 reads | f32 / bf16 | 7.7e-6 / 6.1e-2 | | | 4.5e-6 / 1.5e-2 | | 7.1e-7 / 6.7e-3 |
| AeroZoo s42 basis | f32 / bf16 | 5.2e-7 / 7.3e-3 | | | | 3.7e-7 / 4.9e-3 | |
| AirfRANS s42 blocks | f32 / bf16 | 8.2e-6 / 2.5e-2 | 2.9e-6 / 1.3e-2 | 6.2e-6 / 1.7e-2 | - | - | 0 |
| AirfRANS s42 reads | f32 / bf16 | 1.3e-6 / 1.8e-2 | | | 1.1e-6 / 8.5e-3 | | 8.2e-7 / 4.2e-3 |
| AirfRANS s43 blocks | f32 / bf16 | 2.2e-6 / 5.8e-2 | 1.3e-6 / 6.9e-2 | 2.5e-6 / 2.2e-2 | - | - | 0 |

(AeroZoo block 0 is ill-conditioned: eager float32 itself is 8.5e-4 from float64.) bf16 read differences of the patch
are 1-2 bf16 steps (eager's own run-to-run, from atomics, is 1 step). Accumulated over 12 blocks and reads, bf16
outputs: eager vs float32 eager 2.8e-1 (AeroZoo s42), 8.6e-2 (AirfRANS); C and the patch the same (2.8e-1, 9.2e-2).

Gradient certification, ratio = arm's distance to float64 eager over the same-precision eager run's own; max over
the 4 parameter-gradient groups and the output fields, the scalar loss in brackets. Patched snapshot from job 365108;
the overlay of the same regions (C+P+N+B, Cemu+P+M+B) and the current path C from jobs 363857 and 364478 (range over
the two):

| case | f64 patched: outputs / grads vs eager | patched f32 | patched bf16 | overlay f32 / bf16 | current C f32 / bf16 |
|---|---|---|---|---|---|
| AeroZoo s42 | 1.9e-14 / 9.0e-14 | 1.17 (1.00) | 0.99 (1.14) | 1.33-1.63 / 1.02-1.05 | 1.57-1.60 / 1.01 |
| AeroZoo s43 | 5.9e-15 / 4.9e-14 | 1.21 (1.11) | 1.06 (0.96) | 1.23-1.27 / 1.03-1.06 | 1.25-1.46 / 1.05 |
| AirfRANS s42 | 1.3e-14 / 6.3e-15 | 1.57 (3.42) | 1.34 (0.86) | **2.20-2.24** (3.41) / 1.00-1.07 | **2.01-2.28** (3.41) / 1.00 |
| AirfRANS s43 | 1.2e-14 / 1.1e-15 | 1.23 (1.00) | 1.10 (0.58) | 1.30-1.49 / 1.31-1.42 | 1.34-1.51 / 1.00-1.31 |
| DrivAerML vehicle (random init) | overlay 5.4e-15 / 2.4e-14 | - | - | 1.00 / 1.38 | 1.00 / 1.38 |
| DrivAerML whole (random init) | 1.0e-14 / 2.3e-13 | 1.00 (1.11) | 1.61 (**2.40**) | 1.15 / 1.26 (1.34) | 1.00 / 1.13 (1.70) |

Float64 differences are roundoff (<= 2.3e-13 relative; the reference's CUDA atomics are in it too). Pair lists equal
in every case and precision. Where the 2x bound fails, the current compiled path fails it too, so it is a property of
compiling the blocks, not of the read patch: on AirfRANS s42 in float32 the eager run is unusually close to float64
(loss 8.0e-8, kernel-basis gradient 2.8e-7 relative), and every compiled arm sits at 2-3 float32 ulps (the patched
snapshot 1.57 on the tensors, 3.4 on the loss, like C). The DrivAerML whole bf16 loss ratio (patch 2.40, C 1.70,
overlay 1.34 in another process) is one scalar whose eager value happens to be close to float64 (8.9e-4 against 5e-3
to 2.4e-2 on the other cases); its gradient groups are at 1.61.

## 3. Timing

Model step (train mode, bf16, forward + loss + backward, no optimizer, the same draws in every arm; arms alternated in
one process, medians of 8 steps; percentages against C in the same jobs; full table with every appearance `timing_all.md`):

| case | Q | eager E | current C | C + read patch | patched snapshot itself | Cemu (blocks with casts) |
|---|---|---|---|---|---|---|
| DrivAerML vehicle | 100k | 1522 ms, 155.0 GiB | 1347, 127.5 | 1251-1312 (P+N+B / Cemu+P+M+B), 121 | - | 1326 |
| DrivAerML vehicle | 200k | 2146, 218.1 | 1879, 181.5 | 1779 (-4.5%), 173.3 | 1798 | 1868 |
| DrivAerML whole | 200k | - | 4073, 188.4 | 4018 (Cemu+P+M+B), 180.3 | - | - |
| AeroZoo (s42) | 200k | 1497, 194.1 | 1240, 157.4 | 1136 (-6.6%), 149.7 | 1162 | 1186 |
| AirfRANS (s42) | 180,593 | 1459, 82.8 | 927, 67.8 | 899 (-3.7%), 65.9 | 776 | 1060 (+14%) |
| AirfRANS (s42) | 100k | - | 604, 37.8 | 585 (-3.2%), 36.7 | - | 671 (+11%) |
| lug (random) | 9048 | 416, 7.7 | 358, 6.3 | 340 (-3.1%), 6.0 | - | 368 (+5%) |

AirfRANS times differ by up to 15% between processes (C: 820 to 979 ms over 7 processes; the patched snapshot's 776
was measured in its own process), so only same-process comparisons count; the others agree across jobs within 3%.

Real recipe steps (`train.py`, REF-090 recipe, 60 steps, data + forward + backward + optimizer, median of steps 12-60;
base and patched side by side on one node, swapped between slots, job 365375; DrivAerML from job 365234):

| case | base | patched | change | data wait | peak base / patched |
|---|---|---|---|---|---|
| DrivAerML vehicle 200k | 2008 | 1889 | -5.9% | - | 186.6 / 178.5 GiB |
| AeroZoo 200k, slot 1 / 2 | 1433 / 1270 compute | 1233 / 1149 | -14% / -10% | 10-12 ms | 170.7 / 162.6 |
| AirfRANS 100k, slot 1 / 2 | 658 / 652 compute | 565 / 630 | -14% / -3% | 4 ms | 38.7 / 37.6 |

On a node shared by four recipe processes the host-bound phases make steps vary by 10-15% between processes (job
365234 had AeroZoo patched 15% slower in a different slot; its step-by-step trace shows isolated 2-3 s spikes, not a
uniform slowdown), so the in-process 4-7% is the number to plan by. Compile cost: the read's graphs add about 10-15 s
to the first steps (9-10 graphs; 1-2 graphs once more at the first change of a size class, then none).

## 4. Bf16 rounding points (the brief's constraint)

The read regions compile with `emulate_precision_casts=True`, so every bf16 tensor eager rounds is rounded at the same
point; the F.linear inputs and outputs keep their autocast dtypes. Evidence: in bf16 the compiled read regions differ
from eager by 1-2 bf16 steps on full-size inputs (eager's own run-to-run is 1 step); the CPU toy case gives 0 for the
projections alone. Without the flag (as `compile_blocks` runs today) compiled blocks differ from eager by up to 5.8e-2
relative in bf16 against 5.0e-3 with it. The read patch adds no Inductor option beyond these two.

## Patches and checks

- `compile_read.patch` (md5 c9aaacba; generated by `lhr/make_patch.py ref compile_read.patch` from f13eb5ce0's files
  in `ref/`): applies to f13eb5ce0 with `patch -p1`. CPU: `test_isla_tree_read.py` 36 passed / 6 skipped; `lhr/patchcheck.py`
  PASS (compile_blocks on vs off: float64 3.1e-15, float32 6.7e-7, pairs equal, all three compiled regions ran);
  whole ISLA test directory 321 passed / 46 skipped (LHR cpu-short 365063 on `$T/code_dttfast_compile_p1`). GPU:
  job 365108 above. The test change is one line: the exact-read stub in `_with_exact_read` accepts the new keyword.
  No new test of the compiled branch on CUDA yet: add one that runs a small ISLA-DTT with `compile_blocks` on and off
  and compares outputs and gradients (patchcheck.py's body) when CUDA and Triton are present.
- `block_precision_casts.patch` (md5 ee940cef): one line in model.py's `_BLOCK_COMPILE_OPTIONS`; not recommended
  (Recommendation 2).

## Ideas for DTT-KNOBS

None from this piece: the patch changes no pair, criterion or rounding point.
