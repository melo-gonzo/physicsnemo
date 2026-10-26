# AUDIT: int32 offsets in ISLA's Triton and Warp kernels (INT64-OFFSETS, 2026-10-25)

Code: physicsnemo/experimental/models/isla/ at psharpe/isla e44015436. Branch psharpe/isla-int64-offsets.
Production sizes are in SCALES.md:
- training: 200k points, 200k tokens, S = 512 slices, pairs per chunk up to 163M;
- full cases: AeroZoo 5.87M points (F = 1.35M cells); DrivAerML Volume about 1.5e8 points (F = 16.65M);
  HiLift is a surface task (about 2.7e8 cells, no query geometry, no forest);
- read widths: H = 8, F = 16, KC = 4, C = 8.

Detailed per-expression tables, with line numbers and IR checks, are in:
- AUDIT_A_read.md (fused_read.py, fused_basis.py, segment_softmax.py);
- AUDIT_B_tree.md (tree_build.py, tree_walk.py, sampling.py, tree_read.py bookkeeping, geo_kernel.py);
- AUDIT_C_geometry.md (geometry.py, geometry_fast.py).

Limit: 2^31 = 2.147e9. Warp's `a[i, j]` computes byte offsets in 64-bit, but each index is a C++ `int`, and a
Warp array's shape is int32, so a Warp array of 2^31 or more entries is refused (ValueError).

| kernel | expression | max value at production scale | verdict |
|---|---|---|---|
| geo_kernel `_geo_fwd_kernel`, `_geo_bwd_kernel` | `pid_b*l_sb + offs_n[:,None]*l_sn + offs_s` ((B,N,S) logits, bias, mix and their gradients) | (N+Q)·S = 6.07e6·512 = 3.1e9 (unchunked AeroZoo `forward`) | **fixed** (df266e504: int64 program ids) |
| geo_kernel `_geo_bwd_kernel` | per-tile partials `tile*S*3`, `tile*NW` | n_tiles·S·3 = 9.3e9 (same case, block 1) | **fixed** (same int64 ids) |
| geo_kernel both | per-point loads and stores `pid_b*r_sb + offs_n*r_sn`, pooled `offs_n*po_sn` | (N+Q)·12 = 7.3e7 | safe (int64 since df266e504) |
| geo_kernel `_geo_region_bwd` (host) | n_hat read with gn's strides; gr written with r's strides | not a width bug: wrong gradients, and writes past gr's end, for B > 1 with inputs not contiguous along N (GB300 job 384199: gradient differences up to 1e2) | **fixed** (21707ae5f; production uses B = 1, so it was not reached) |
| fused_read `_fkr_fwd_kernel`, `_fkr_bwd_point_kernel` | `p = start + arange`; `basis + p*F`, `dbasis + p*F` (start = `q_offsets[s]`, loaded as the plan's dtype) | (P−1)·16 = 2.6e9 at P = 163M. int64 in the model, because `walk_range_pairs` keeps the offsets int64. `PairPlan.build` of int32 indices gives int32 offsets, which wrap at P > 2^27 | **fixed** (330497096: offsets cast to int64 in the kernel) |
| segment_softmax `_ssr_fwd_kernel`, `_ssr_bwd_point_kernel` | `ker + p*H`, `dker + p*H`, `logm + p` | (P−1)·8 = 1.3e9 at 163M; 1.1e10 unchunked AeroZoo. With int32 offsets they wrap at P > 2^28 | **fixed** (330497096) |
| fused_read `_fkr_source_sum_kernel` | `range(a, b)` from `item_off` | items·96 ≈ 7e7; `item_off` is always int64 | safe; cast anyway (330497096) for the same pattern |
| fused_read others (`_fkr_bwd_source_kernel`, `_nm_*`), `inv*2H`, `s*(2HF)` | explicit `.to(tl.int64)` already | up to 2.6e9 (`inv*2H`) | safe (int64) |
| fused_basis `_pair_rows`, `_inv/_basis_fwd/_basis_bwd_kernel` | `tile*BLOCK + arange` (int64 tile); `p*F`, `p*G` | 4.19M pairs per launch (`TREE_READ_PAIR_CHUNK`)·16 = 6.7e7 | safe |
| tree_walk `_count_kernel`, `_fill_kernel` (Warp) | `tid`, `q*n_t + t`, `pts[q*d + k]` | n_q·8 = 1.2e9 (count over all DrivAerML Q = 1.5e8) | guarded (n_q·max(d, n_t) ≥ 2^31 → reference walk). The fallback runs out of memory past 2.68e8 interior points; no production case reaches that |
| tree_walk fill writes | `qi[far_at]`, `si[near_at]`, `scratch[...]` | pairs of the walked points (163M per chunk) | guarded (n_pairs ≥ 2^31 → int64 reference walk) |
| tree_walk prefix sums, `walk_pair_counts`, recipe `cum` | torch int64 | 1.2e11 (DrivAerML all Q) | safe (int64) |
| tree_read plan narrowing | `qi/bj/perm/qi_by_b .int()` | below 2^31, guarded by `P < 2**31` | safe (guarded) |
| tree_build `_node_count/_node_fill_kernel` | `wp.int64(tid)`, offsets | N = 2e5 tokens; nodes < 4e5 | safe (n ≥ 2^31 guarded → reference build) |
| sampling `isla_cumsum` | float64 cumsum over F cells; int64 positions | F = 2.7e8 (HiLift) entries | safe (no int32) |
| geometry `_ball_candidates.search` (Warp) | one 1-D Warp array of rows·cap slots (cap = 64) | Q·64 = 9.6e9 (DrivAerML full case); 3.8e8 (AeroZoo full) | **fixed** (389b5092c: launches over blocks of (2^31−1)//cap rows). Before the fix Warp refused at Q ≥ 2^25 = 33.5M with a ValueError, not silently |
| geometry ball and winding kernels | `planes[c*4+e]`, `corners[c*3+k]`, node ids int32 | 4F = 6.7e7; 2F = 3.3e7 | safe |
| geometry `_nearest_cell`, `_winding`, `_packed_winding`, BVH kernels | `query[tid]`, `c*3`, node ids | Q = 1.5e8; 3F = 5e7 | safe |
| geometry `_packed_winding_sum` buffer | (2F−1) records of 88 B | 2.9e9 bytes, shape 3.3e7 | safe (the byte size is never an int32) |
| geometry_fast `_closest_weights_kernel` | `(pid*256 + arange).to(int64)`: the product is int32 | P ≈ 3e8 candidate pairs (DrivAerML full) | safe. It would wrap at 2^31 pairs (about 1.2e9 queries), where its inputs alone need about 270 GB |
| geometry_fast `_canonical_kernel`, `_run_start_kernel` | `K*c` with c = pid*256 + arange (int32) | 3F = 5e7 (would be 8.2e8 for HiLift's cells, which make no geometry call) | safe; it would wrap at F ≥ 7.2e8 |
| geometry host keys and masks (torch) | `min*V + max`, `li*F + pk`, (Q, 64) masks | Q·F = 2.5e15 | safe (int64) |

Not fixed, by choice. Each of these is a loud failure or an out-of-memory error, not a wrong value, and no
production case reaches it:
- **The count pass's fallback past 2^28 interior points.** It runs out of memory.
- **`_closest_weights_kernel` and `_canonical_kernel`.** Memory or cell counts stop them before 2^31. A test past
  their limits would need about 270 GB, or 7e8 cells.
- **torch's CUDA sort and unique refuse more than INT_MAX elements** (the lane reviewer read this from
  libtorch_cuda's messages; not run). They cap two things:
  - `PairPlan.build`'s `argsort` at 2^31 pairs per chunk (segment_softmax.py:96). DrivAerML reaches it with chunks
    of about 2.6M points; training chunks are 163M pairs.
  - The geometry's candidate-pair sort and unique (geometry.py:~1961) at 2^31 pairs (about 1.2e9 queries).
- **Memory of the DrivAerML Volume full case after 389b5092c.** Warp no longer refuses it, but the ball search still
  holds all Q x 64 slots and (Q, 64) masks at once: 1.6-1.8 kB per query (measured on CPU), about 225-250 GiB at
  1.5e8 queries. Memory, not Warp, now limits that case. Chunking `exact_boundary_geometry` over queries would bound
  both; its outputs are per query (the reviewer's CPU probe: bitwise).

Evidence (GB300, LHR):
- Each new test past its limit **fails on the base** (job 385665, head's tests on e44015436's code):
  - geo_kernel, read and segment-softmax tests: illegal memory access;
  - ball search, both kernels: Warp ValueError at 2,147,487,744 entries;
  - strides test: torch.equal fails.
- Each **passes on the head** (gpu tier 384972: 8 new cases passed, none skipped).
