Status: DONE (13:12 EDT 10-25). Open for the assembly: which certification reading applies (DECISIONS_NEEDED.md).

**Branch.** `psharpe/isla-fused-pad` at **ee1917561** is pushed. It is origin/psharpe/isla e44015436 plus a cherry-pick
of 0c1e71214; 80a58e1e8 is not taken. Nothing was merged. The worktree is /home/psharpe/gh/physicsnemo-isla-fused-pad
(1.1 GB).
- Conflicts:
  - `resolve_geo_kernel` keeps KERNEL-BUGS' autocast condition and takes `n_spatial_dims <= 3` and
    `max(K, 1) + P <= MAX_DIRECTIONS`.
  - The tests keep the base's `@pytest.mark.cuda @requires_cuda` and `monkeypatch`.
  - Two base CUDA tests pinned the old routing. The disqualifier test drops its 2-D and K = 0 cases, and the 2-D
    determinism test now expects `fused` and passes bitwise.
- Merge pre-check: the branch merges cleanly with INT64-OFFSETS 21707ae5f and with origin f16c3f9f0. A temporary
  commit of all three, on no branch, passed the GB300 tier (386125).

**Suites** (all on ee1917561 unless noted):

| suite | result |
|---|---|
| GB300, every CUDA test (384198) | 1,232 passed, 5 xfailed, 0 failed |
| CPU full tier (384678) | 708 passed, 0 failed |
| recipe tests | 615 passed, 0 failed |
| pre-commit, 7 changed files | clean |
| merge pre-check c3906cf16, GB300 (386125) | 1,240 passed, 0 failed |

**Speed and memory per iteration** (GB300, ABAB in synchronized windows, 384802; `out/speed_384802/TABLE.md`):

| case | compiled ms | eager ms | peak GiB |
|---|---|---|---|
| AirfRANS (far field) | 575 -> 284 (**0.49x**) | 928 -> 368 (**0.40x**) | -0.7 / -1.6 |
| lug | 242 -> 253 (1.04x, host-bound) | 377 -> 308 (**0.82x**) | ±0.15 |
| Tensile2d | 218 -> 220 (1.01x) | 346 -> 294 (**0.85x**) | ±0.1 |

- **3-D models with K > 0 run bitwise as before.** After 8 compiled recipe steps, every parameter matches between
  base, branch and base on DrivAerML Volume (431 tensors) and AeroZoo (432 tensors). The branch's 4 compiled graphs
  were all FX graph cache hits on the base's. AeroZoo needs `num_workers=1`: its 12-worker loader is not bitwise even
  base against base (`out/BITWISE_3D.txt`).

**Eager certification (the fused region against the eager reference region), passes:**
- PERF-SURVEY's instrument, rerun on the branch:
  - float64 is at roundoff: 5.6e-14 on AirfRANS and 7.2e-14 on the lug, the eager region's own rerun spread. It is
    3.8e-13 on Tensile2d, under 1e-12 but about 100x the eager rerun spread.
  - float32: 0 groups above eager plus eager's one-ulp change, on all three datasets.
- On real checkpoints, block by block:
  - aggregated groups: 0 failures;
  - per tensor: 1-4 of 1,152 groups exceed by up to 1.9x, all inside the fused path's own one-ulp family.

**Compiled certification** (cert_fp.py, the training step's block forward and read, on real checkpoints: AirfRANS
afwhole_af_dtt_seed42, lug r090fix_lug_dtt_seed42, Tensile2d r090fix_t2dz_isla_seed42; bf16 families of 16 per code
from 2 cold processes):
- **Per tensor (C4's reading), the base itself fails the rule on every dataset, AirfRANS included.** Failing groups
  per process, fp32 / bf16:

  | dataset | base | branch |
  |---|---|---|
  | AirfRANS | 19-21 / 10-11 | 16-19 / 7-8 |
  | lug | 25-26 / 10-11 | 28-30 / 10-11 |
  | Tensile2d | 30 / 30-31 | 17 / 25-26 |

  The failures are in the same routing-gradient groups for both codes.
- **On aggregated groups (DTT-FAST style), AirfRANS and the lug pass in every group for both codes.** Tensile2d fails
  1 / 5 groups on the branch and 3 / 6 on the base. float64 is at most 3.3e-12 per tensor and 3.9e-13 aggregated.
- **The lug's outcome:** it fails per tensor, as at base (C4), and passes aggregated. FUSED-PAD changes neither verdict.
- **bf16 against the base's compiled family:**
  - The branch's median lies inside the base's range in 1,090-1,098 of 1,152 groups, against a base-vs-base null of
    1,099-1,117. The median ratio over groups is 1.00.
  - The branch is above the base's maximum in 43-47 groups, and below its minimum in only 7-18.
  - Cause: the base's compiled path is more accurate than eager in and after the geometry region, because Inductor
    elides eager's bf16 casts there. The fused kernel keeps eager's roundings by design.
  - Without compile, the branch's eager family against the base's is symmetric (10-19 groups worse, 7-21 better).
  - Of the 43-47 groups above the base compiled range, 33-42 lie inside or below the base's *eager* range.

**For the 0.14.0 assembly:**
1. Rule which reading certifies compile_blocks. Under the per-tensor reading, AirfRANS would also train eager, which
   no current plan says.
2. Rule which family judges a compiled path that calls a Triton kernel: the base's compiled family, or the eager one
   as well. My recommendation, with the numbers: DECISIONS_NEEDED.md.
3. Merge after INT64-OFFSETS, or with it. Both merge cleanly and the combined GB300 tier passes.
4. No version bump on the branch.

What I relied on: LANE_RULES (compiled regions, the 19:30 bf16 rule, GPU tests, cross-version processes, git) and
C4's per-tensor precedent. I amended CERT_PLAN.md before any GPU result: the float64 verdict uses C4's 1e-12 bound,
because a float64 one-ulp input change is far below the reordering roundoff of a compiled region.

For project memory:
- AeroZoo's 12-worker loader is not bitwise run to run.
- In bf16, Inductor's elided casts make a compiled eager region more accurate than its eager run.

**Jobs and GPU-h.** 4.57 of the 5 GPU-h cap: 384198, 384802, 384785, 385643, 386125, plus CPU jobs 384678 and the
dry runs. Details are in JOBS.md and RESULTS.md. Nothing is running, and the registry row is removed.

**On disk:**
- Locally: the worktree above (1.1 GB) and this folder (75 MB).
- On LHR: `$T/code_fusedpad_{base,ee19175}`, read-only and reusable, and `$T/kernel/fusedpad` (79 MB).
