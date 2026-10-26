# FUSED-PAD results (branch psharpe/isla-fused-pad ee1917561 against its base e44015436)

Written 2026-10-25 13:05 EDT. Jobs and commands: JOBS.md, JOBS_tests.md. Plan and verdict definitions, fixed before
the GPU certification ran: CERT_PLAN.md. Every base/branch comparison ran each version in its own process
(snapshots `$T/code_fusedpad_base` and `$T/code_fusedpad_ee19175`, `diff -rq`: only geo_kernel.py and model.py differ;
each run printed its geo_kernel.py/model.py md5: base 8303218e/bbd84a23, branch 08ae866e/6749140b). GB300, torch
2.12.0+cu130, bf16 recipe configs (hidden 512, 12 layers, 512 slices).

## 1. Branch

- `psharpe/isla-fused-pad` = origin/psharpe/isla e44015436 + ee1917561 (cherry-pick of 0c1e71214; 80a58e1e8 not
  taken). Pushed to origin. Writer report: `writer/REPORT_WRITER.md`.
- Conflicts resolved:
  - model.py `resolve_geo_kernel`: it keeps KERNEL-BUGS' autocast condition (`amp in (None, torch.bfloat16)`). It
    takes FUSED-PAD's `n_spatial_dims <= 3` and `max(n_global_vectors, 1) + n_block_vecs <= MAX_DIRECTIONS`. The
    module and method docstrings state both conditions once.
  - test_isla_kernels.py: the base's `@pytest.mark.cuda @requires_cuda` and its `monkeypatch` argument are kept, and
    FUSED-PAD's parametrization over d in {3, 2} and K = 0 is added. The new tests use the base's markers.
  - Two base CUDA tests pinned the old routing and were adapted:
    - `test_geo_kernel_auto_disqualifiers_on_cuda` drops the 2-D and no-direction cases;
    - `test_the_model_repeats_bitwise_on_cuda_with_kernels_auto` now expects `fused` in 2-D, and passes on the GB300,
      so the fused 2-D region repeats bitwise.
- pre-commit on the 7 changed files (COLUMNS=250): all hooks pass.
- Merge pre-check, read-only (`git merge-tree`): the branch merges cleanly with origin/psharpe/isla-int64-offsets
  21707ae5f and with origin/psharpe/isla f16c3f9f0 (PERF-RECIPE). A temporary commit c3906cf16 of all three, on no
  branch, passed the GPU tier (job 386125, section 7).

## 2. Tests

| suite | job | result |
|---|---|---|
| whole ISLA dir, GB300, every CUDA test (submit.sh gpu) | 384198 | 1,232 passed, 5 xfailed (the known strict xfails), 0 failed, 0 errors; 8 invocations rc 0 |
| whole ISLA dir, CPU (submit.sh full) | 384678 | 708 passed, 508 skipped, 5 xfailed, 0 failed |
| recipe tests (local, CUDA hidden) | - | 615 passed, 18 skipped, 0 failed |
| GPU tier on the merge pre-check c3906cf16 (FUSED-PAD + INT64-OFFSETS + f16c3f9f0) | 386125 | 1,240 passed, 5 xfailed, 0 failed; 8 invocations rc 0 |

## 3. Speed and memory per iteration (job 384802, `out/speed_384802/TABLE.md`)

Each round put 4 processes on one node at the same time: base, branch, base, branch, with the GPUs swapped from one
round to the next. prof4.py's barrier started the timing windows after every process had compiled and warmed up, so
no Inductor compile overlapped a timing window. Each process timed the recipe's own `_run_epoch` on the replayed first
batch: 3 warm-up steps, then the median of 12 steps. Peak is the allocated peak.

| case | base ms (2 processes) | branch ms (2 processes) | branch / base | peak GiB base -> branch |
|---|---|---|---|---|
| AirfRANS (far field), compile_blocks | 602, 549 | 297, 271 | **0.49** (575 -> 284) | 27.35 -> 26.65 |
| AirfRANS, eager blocks | 974, 883 | 357, 379 | **0.40** (928 -> 368) | 33.07 -> 31.46 |
| lug, compile_blocks | 235, 250 | 260, 247 | 1.04 (242 -> 253) | 5.43 -> 5.58 |
| lug, eager blocks | 381, 374 | 311, 305 | **0.82** (377 -> 308) | 6.70 -> 6.55 |
| Tensile2d, compile_blocks | 215, 221 | 223, 216 | 1.01 (218 -> 220) | 3.81 -> 3.92 |
| Tensile2d, eager blocks | 331, 360 | 295, 294 | **0.85** (346 -> 294) | 4.50 -> 4.40 |

The host-bound compiled lug and Tensile2d steps are neutral, as PERF-SURVEY found for the lug. The lug trains
eager today, so its relevant comparison is base eager 377 ms against branch eager 308, or against branch compiled 253
if a compiled path is certified.

## 4. 3-D models with K > 0: bitwise as before (job 384785, `out/BITWISE_3D.txt`)

Each chain ran three processes on one GPU, sharing one node-local Inductor/Triton cache: base (cold), branch, base.
Each process trained 8 recipe steps with compile_blocks (3 loader steps, 1 warm-up, 4 timed) and then saved every
parameter.
- **DrivAerML Volume** (whole boundary, 200k): all 431 parameter tensors bitwise equal in base1 = branch = base2, and
  the mean losses are equal. The branch compiled none of its 4 FX graphs: every one was a cache hit on the base's, so
  the compiled graphs are identical.
- **AeroZoo** (200k): with the recipe's 12 loader workers, even base against base differs (430 of 432 tensors; the
  cases are the same, so the loader's threaded draws are not reproducible). With `num_workers=1 prefetch_factor=1`,
  all 432 tensors are bitwise equal in base1 = branch = base2, with 4 of 4 FX graph cache hits.
- By the code: for d = 3 and K > 0, `fused_geo_region` takes neither padding branch and passes `max(k, 1) = k`, and
  `resolve_geo_kernel` returns `fused` as before.
- Not run: DrivAerML Surface and HiLift, which take the same path.

## 5. Exactness of the eager path (bullet 1)

**(a) PERF-SURVEY's instrument, rerun on the branch** (prof4 `--exact`: model-level groups, initial weights, the first
batch, L2 distances; job 384785; `out/cert_384785/EXACT_TABLE.md`):

| | AirfRANS (34 groups) | lug (33) | Tensile2d (34) |
|---|---|---|---|
| float64 fused vs eager region, max over groups | 5.6e-14 (eager rerun 5.9e-14) | 7.2e-14 (rerun 8.1e-14) | 3.8e-13 (rerun 4.9e-14; largest in the gradient groups) |
| float32 fused/eager distance-to-float64 ratio, median / max | 0.999 / 1.29 | 1.000 / 1.000 | 0.976 / 1.14 |
| float32 groups above eager + eager's one-ulp change | **0** | **0** | **0** |
| bf16, fused median inside eager's 8-sample family (indicative) | 33 / 33 | 31 / 32 (1 below) | 27 / 33 (5 above, max 1.15x) |

On Tensile2d, float64 is at roundoff (under 1e-12), but about 100 times the eager region's own rerun spread there:
the fused kernel sums in a different order.

**(b) The same check on real checkpoints, block-wise, per tensor** (V3 of cert_fp.py, 2 processes x 1,152 groups;
`out/cert/CERT_TABLES.md`):
- float64: max over groups of d(fused64, region64) is 5.0e-14 (af), 1.8e-13 (lug) and 5.8e-14 (t2d).
- bf16: 0 failing groups.
- float32: 1-4 of 1,152 groups per process exceed the region + mean one-ulp change, by at most 1.9x. All are routing
  or anchor gradients (geo_logit.weight, grad_ms, assign.gain). In every case the fused value lies inside the fused
  path's own one-ulp family.
- On aggregated groups (cert2, `out/cert2/AGG_TABLES.md`, l2): 0 failing groups in every precision, on all three
  datasets.

## 6. The compiled path (compile_blocks): certification on real checkpoints

Instrument `lhr/cert_fp.py` (from SLICE-NORMAL-EXP's cert_blocks.py):
- It certifies the regions the interior training step compiles: the boundary side's block forward (`slices=True`) and
  the points' `block.read`. Their inputs are captured from eager `query_chunks` passes.
- Checkpoints: AirfRANS afwhole_af_dtt_seed42 (airfoil + far field; strict load); lug r090fix_lug_dtt_seed42 (strict
  load); Tensile2d r090fix_t2dz_isla_seed42. The Tensile2d checkpoint is interacting ISLA 0.9.0, the only real
  Tensile2d one: its blocks are trained, and the ISLA-DTT kernel reads start at their initial weights.
- Runs: job 384785 (2 val cases, 2 cold processes per code and dataset, 8 + 8 draws each); job 385643 (cert_fp2,
  case 0, 2 more cold processes each, which also log the numerators for aggregated groups).

**V1, the compiled-regions rule (compiled vs float64 <= eager vs float64 + mean one-ulp change).** Failing groups per
process, fp32 / bf16:

| dataset | per-tensor (1,152 groups), base | per-tensor, branch | aggregated l2 (72 groups), base | aggregated l2, branch | float64 > 1e-12, per-tensor base / branch (max) |
|---|---|---|---|---|---|
| AirfRANS | 21 / 10, 19 / 11 | 16 / 7, 19 / 8 | 0 / 0, 0 / 0 | 0 / 0, 0 / 0 | 1 / 2 (1.0e-12 / 3.3e-12) |
| lug | 25 / 10, 26 / 11 | 28 / 11, 30 / 10 | 0 / 0, 0 / 0 | 0 / 0, 0 / 0 | 0 / 0 (2.5e-13 / 3.8e-13) |
| Tensile2d | 30 / 31, 30 / 30 | 17 / 25, 17 / 26 | 3 / 6, 3 / 6 | 1 / 5, 1 / 5 | 2 / 2 (1.9e-12 both, same groups) |

- Per tensor, **the base fails V1 on every dataset**, AirfRANS included, as C4 found for the lug. The branch fails in
  the same class of groups (routing gradients assign.*, geo_logit, norm_assign, and the slice outputs out_ms/out_z;
  almost all in the boundary-side forward) at similar counts: lower on AirfRANS and Tensile2d, +3-4 groups in fp32
  on the lug.
- On aggregated groups (DTT-FAST/PERF-SURVEY style: per block call, outputs / input gradients / parameter gradients,
  l2), AirfRANS and the lug pass in every group for both codes. Tensile2d fails 1/5 groups on the branch and 3/6 on the
  base.
- float64: at most 3.3e-12 per tensor (AirfRANS fwd3.grad_assign.gain, the base's own worst group, 1.0e-12). On
  aggregated groups it is at most 3.9e-13 (l2).

**V2, bf16 equivalence (19:30 rule: 16 compiled samples per family from 2 cold processes; job 384785).** Per tensor:

| dataset | branch median inside base compiled range | above base max (worse) / below min (better) | null: base p1 inside base p0 (8 vs 8) | branch median / base median over groups |
|---|---|---|---|---|
| AirfRANS | 1,098 / 1,152 | 47 / 7 | 1,099 / 1,152 (26 / 27) | 1.004 |
| lug | 1,093 / 1,152 | 43 / 16 | 1,117 / 1,152 (15 / 20) | 1.002 |
| Tensile2d | 1,090 / 1,152 | 44 / 18 | 1,109 / 1,152 (19 / 24) | 1.001 |

On aggregated groups (cert2, l2, case 0): inside 56 / 72 (af), 62 / 72 (lug), 56 / 72 (t2d). The base-vs-base null
is 69 / 72, 69 / 72 and 66 / 72. Tables: `out/cert/ONESIDED.md`, `out/cert2/AGG_TABLES.md`.

**What the asymmetry is** (`out/cert/ONESIDED.md`):
- The fused region and the eager region are equivalent in bf16. Without compile, the branch's eager family against
  the base's eager family is worse in 10 / 16 / 19 groups and better in 7 / 21 / 14: symmetric, median ratio 1.000.
- The base's compiled path is *more accurate* than eager: it is below eager's range in 122 / 203 / 286 groups. Inductor
  elides bf16 casts inside the eager geometry region (DTT-FAST's finding; the program keeps Inductor's default
  casts).
- The fused Triton kernel is opaque to Inductor and reproduces eager's bf16 roundings by design (geo_kernel.py's
  docstring). The branch's compiled path therefore loses that bonus in and right after the geometry region: the
  worse groups are the read's grad_z / grad_ms and the forward's out_protos / out_ms / assign.
- Of the groups where the branch is above the base compiled maximum, 42 / 39 / 33 (of 47 / 43 / 44) still lie
  inside or below the base's *eager* range. The excess over the base compiled maximum has a median of 1.07-1.14x and a
  maximum of 4.5x.
- Against the base's eager family, the branch's compiled path is worse in 16 / 18 / 40 groups and better in
  79 / 172 / 263.

**The lug's certification outcome:**
- Per tensor (C4's reading), V1 fails for the base (25-26 fp32 and 10-11 bf16 groups) and for the branch (28-30 and
  10-11). As in C4, the rule cannot certify the lug's compile_blocks on either code.
- On aggregated groups, V1 passes for both codes, and float64 is at 1.8e-15 (l2). bf16 V2 against the base compiled
  family is 62 / 72 groups inside (null 69 / 72).
- So the lug's result is the same as AirfRANS': the branch does not change the lug's standing under either reading.

## 7. Merge pre-check (job 386125)

GPU tier on c3906cf16 = FUSED-PAD + INT64-OFFSETS 21707ae5f + origin/psharpe/isla f16c3f9f0, a commit object on no
branch: 1,240 passed, 5 xfailed, 0 failed (every test ran; the only skips are the 5 strict xfails).

## GPU-h

0.51 (384198) + 0.76 (384802) + 1.91 (384785) + 0.87 (385643) + 0.52 (386125) = 4.57 of the 5 cap.
