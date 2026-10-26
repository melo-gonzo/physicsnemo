Status: DONE

**What ran** (2026-10-25, 11:06 to 12:51 EDT; 1.43 of the 4 GPU-h cap; all jobs on LHR GB300 / cpu-short, rows in JOBS.md)
- Branch psharpe/isla-int64-offsets, from origin/psharpe/isla e44015436, pushed to origin and not merged.
  Worktree: /home/psharpe/gh/physicsnemo-isla-int64-offsets (1.1 GB).
- Commits, each signed off with hooks on:
  - df266e504: cherry-pick of 80a58e1e8 with -x. It applied without FUSED-PAD. Only the test's marks conflicted; they now use this branch's @pytest.mark.cuda / @requires_cuda.
  - 330497096: the read kernels take int64 pair offsets whatever the plan's dtype.
  - 389b5092c: the ball search runs in blocks of rows under Warp's limit of 2^31 entries per array.
  - 21707ae5f: the fused geometry backward reads r and n_hat with their own strides. This is a non-width bug I found during the audit.
- Children: 3 bug-hunters wrote the audits; 1 lane-writer was the worktree's only writer for the three fix commits; 1 lane-reviewer reviewed. Its verdict was "OK with notes": no P0 or P1, and its P2/P3 points are answered below and in AUDIT.md.
- Jobs:
  - 383912: the cherry-picked test on a GB300.
  - 384199: strides probe.
  - 384972: gpu tier.
  - 385335: CPU full tier.
  - 385599: goldens A/B.
  - 385665: the new tests against the base code.
  - 385803: A/B rerun of the one flagged output.

**Audit table:** AUDIT.md. It has one row per kernel and expression, the maximum value at production scale, and a verdict. The detailed tables are AUDIT_A_read.md, AUDIT_B_tree.md and AUDIT_C_geometry.md; the sizes used are in SCALES.md. In short:
- **Fixed, geo_kernel fwd/bwd.** Tokens x slices reach 3.1e9 in the unchunked AeroZoo forward, and the backward's per-tile partials reach 9.3e9. Fixed by df266e504.
- **Fixed, fused_read and segment_softmax by-point kernels.** `p*F` and `p*H` were int32 whenever the plan's offsets were int32. `PairPlan.build` gives int32 offsets for int32 indices, so they wrapped above 2^27 or 2^28 pairs. The model's own plans keep int64 offsets, so this was latent. Fixed by 330497096.
- **Fixed, geometry `_ball_candidates`.** Its Q x 64 slots sat in one 1-D Warp array, which Warp refuses (ValueError) from Q = 2^25 = 33.5M. The DrivAerML Volume full case (about 1.5e8 points) therefore could not compute its geometry. Fixed by 389b5092c.
- **Safe or guarded, everything else.** This covers the tree walk and tree build (guards fall back to int64 paths), sampling's float64 cumsum, fused_basis (launches capped at 4.2M pairs), and the other geometry and geometry_fast kernels (largest value 5e7 to 3e8, or memory runs out before 2^31).

**Tests and suites**
- New tests:
  - slow GPU: `test_fused_kernel_read_with_int32_offsets_past_2_27_pairs` (about 21 GiB), `test_segment_softmax_read_with_int32_offsets_past_2_28_pairs` (about 17 GiB), `test_ball_search_past_2_31_slots[fast|reference]` (2^25 + 64 queries, about 60 GiB);
  - CUDA: `test_fused_geo_region_reads_strided_inputs_as_contiguous_ones`;
  - CPU fast: `test_ball_search_in_blocks_of_rows_equals_one_launch[fast|reference]`;
  - plus the cherry-picked `test_fused_geo_region_offsets_past_int32` (about 100 GiB).
  
  Each past-the-limit test compares the last rows, bitwise, with the same rows run alone.
- **CPU full tier (385335):** 710 passed, 478 skipped, 5 xfailed.
- **GB300 gpu tier (384972):** every group rc 0, 0 failures. All 8 new cases passed there; none was skipped.
- **Fails on the base (385665):** each new test fails on e44015436's code. The geo_kernel, read and segment-softmax tests hit an illegal memory access. The ball search gets Warp's ValueError at 2,147,487,744 entries. The strides test's torch.equal fails.

**Bitwise evidence** (goldens A/B harness; base and head each in their own process; logs in `$T/kernel/isla_tests/out/gpu_385599/` and `gpu_385803/`)
- **CPU:** 110 of 110 case variants PASS, bitwise.
- **CUDA:** 154 of 155 PASS. One output was flagged: `query_chunks_f32@cuda:reference out.range0.p`, a declared atomics family that was 1 ulp (1.49e-8) apart across the two sides while each side's 5 repeats were bitwise.
- **The rerun at 20 repeats (385803)** shows that output varying by the same 1 ulp within base and within head. All four comparisons pass: two base-head and two A/A.
- So no output below the limits changed.

**What it means.** I am highly confident that full-resolution forwards no longer wrap in the ISLA kernels: every fix is checked by a test that faults or raises on the base and passes on the head. Production training (B = 1, int64 plans) was not affected by fixes 2 and 4.

**For the supervisor / Peter**
- Peter decides the branch's merge, as with FUSED-PAD. It does not need FUSED-PAD: 80a58e1e8 applied alone.
- The DrivAerML Volume full case now gets past Warp, but its ball search still holds about 1.6-1.8 kB per query, about 240 GiB at 1.5e8 points. Chunking `exact_boundary_geometry` over queries is the natural next step (its outputs are per query). I did not do it: it is new work.
- Torch's CUDA sort refuses more than INT_MAX elements (the reviewer's reading, not run). That caps `PairPlan.build` at 2^31 pairs per chunk, about 2.6M-point chunks on DrivAerML. It fails loudly; I did not fix it (listed in AUDIT.md).
- Two notes from the reviewer, left as they are to avoid a rebase:
  - the `_fkr_source_sum_kernel` cast in 330497096 is defensive, a no-op today;
  - the new strides test uses `@requires_cuda` where `requires_cuda_triton` fits, following its neighbours.
- No new rulings. The registry row `^isla\.int64\.` is removed (backups in this folder). Nothing is running.

**Left on disk**
- the worktree (1.1 GB);
- this folder (244 KB);
- on LHR, the read-only snapshots `$T/kernel/int64/{snap_base,snap_onbase}` and `$T/kernel/isla_tests/snap/{cp1,strides1,gpu1,full1,ab1,onbase1,ab2}`.
