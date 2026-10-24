Status: DONE

**Outcome.** Thin cells of nonzero area are fixed in the reference and the fast path, which stay bitwise equal to each other. `psharpe/isla-geo-thin` is pushed and ready to merge at `6bac57cbd`; it was not merged into `psharpe/isla`. Every model-visible output (d, u, the winding sign, the tie and on-wall decisions) and every gradient is bitwise unchanged on every real dump under both kernels. The op's internal closest-point weights differ at 11 of 200,000 DrivAerML whole-boundary queries per draw, all on-wall queries on ordinary tunnel-floor cells, by 1-2 ulps; the new point is closer to exact in 20 of 22 rows. No version change, no new op (the supervisor ruled at 08:51).

**Commits** (`git commit -s`, hooks on, plain push):
- `91bd3236e`: F1-F3 and `test_isla_geometry_thin.py` (19 CPU + 19 CUDA tests; 13 of the CPU tests fail at 7213c70c5).
- `672d6373a`: the bounds labelled as first-order estimates, plus a test that pins the switch on ordinary cells.
- `eee0c551e` and `6bac57cbd`: merges of `origin/psharpe/isla` 1b25eb899 and cce0fb53d. Neither changed geometry code.

**Design** (`DESIGN.md`, md5 6b342832; approved f7f8ed68 at 06:27, then the 08:36 amendment):
- F1, certified plane bound: edge normals are outward by vertex order, not by a rounded sign test. A cell whose float64 normal error exceeds u32 gets NaN planes.
- F2: the float64 ball test rejects a cell only if Ericson's distance minus its certified error is beyond the radius. K = 168 is derived, and the exact rational check stays within 0.0115 of the bound.
- F3: guarded pairs get an accurate closest point (the face foot in the longest edge's frame, or the edge points). The switch to it is not certified: its 12 and 13 units are first-order estimates (worst cases 77 and about 17.5 units; largest observed error 2.03).

**Phase A** (`RESULTS.md`, md5 50924a0e):
- Census of 43 meshes over every dataset (census table in RESULTS A1; raw data in `lhr/out/census/`):
  - No stored mesh has an uncertain normal (largest error 6.4e-6 eps32).
  - Cells past Ericson's float32 failure index: AeroZoo 2 in the dump and 15 in stored case 0; HiLift 18,774 (surface task, which calls no query geometry).
  - 0 coplanar folds and 0 axis-plane overlaps. AeroZoo has 4 near folds, which change no recipe query.
  - DrivAerML's whole boundary has 110 T-junction vertices. No drawn query has its closest point at one, and removing them leaves all 200k queries bitwise unchanged, in both draws.
- Recipe draws against an exact rational brute force (310,545 queries checked, f32 and f64, both kernels): 0 raises, 0 wrong d, cell, sign, on-wall decision or u. No standing and no final rerun is exposed.
- Adversarial float64 queries on the stored AeroZoo meshes triggered a CUDA device-side assert in both kernels at the base, and gave 0 wrong at the head. Float32 queries were already correct at the base.
- Mechanism: Ericson's weights misplace the closest point by about u64 L/rho^2, which produces wrong d or cell, or a wall point taken as off the wall. The float32 plane bound and the float64 rejection then drop the true cell, so the query has no candidate. The winding number never failed.
- Tie direction (A4): one-ulp vertex moves change u by at most 3.9e-5 on the lug and Tensile2d, with 0 active-set changes. It is Hoelder-1/2 at near-duplicate normals.

**Checks at the head** (`RESULTS.md` final sections):
- Bitwise against 7213c70c5 (ops renamed; all op outputs, w, d, u, gradients): 104 of 104 GB300 cases and 60 of 60 CPU cases (one thread), under both kernels. The cases cover AirfRANS with its far field (4 draws), AeroZoo, the 20 lug cases, Tensile2d and DrivAerML whole boundary. DrivAerML whole boundary was also run at 2 draws x 200k queries; there only the 11 weight rows differ.
- Adversarial thin sets on CPU: 0 wrong of 517,296 queries.
- Fast path: equals the reference on 2.56M queries and 16.7M candidate pairs; FAST-PLUGIN 0 of 584,953 differ.
- Tests: the ISLA directory on cpu-short (368129) 395 passed, 120 skipped. Recipe tests (368221) 551 passed, 12 skipped, 2 failed: `test_campaign_c_hooks`, already failing with the same E lines at 7213c70c5 (DTT-POSE's 367075). GB300 geometry, tree-read, fast, degenerate, thin and release files (368128): 282 passed, 1 skipped, under both auto and forced.
- Cost of the DrivAerML whole-boundary call: reference 410 to 474 ms (+15%), fast 343 to 405 ms (+18%). By part: F2 +32-34 ms in the ball search, F3 +27 ms (launch overhead, about the same on AeroZoo's 93 guarded pairs), `_thin_bounds` 8.5 ms.

**Limits recorded:**
- Overlapping coplanar cells.
- T-junctions.
- Near-duplicate normals.
- The band edge at medial points. Lug case 1884 node 6997 has gap/band 1.12 native and 0.86 posed, so |du| = 0.707 there, which accounts for all of DTT-POSE's 1.74e-5 excess. A band measured about the centroid removes it. Tie membership still differs at 4-6k nodes per case, from bitwise normal ties, with |du| <= 2.2e-5.

**Tighter guard:** a certified bound cannot reach zero guarded pairs on DrivAerML or the lug. That needs K/1000, and the observed margin allows at most about 1/87. The cost lever is a fused guard and accurate-point kernel (about 25 ms per call); this is left for 0.12.0.

**Rulings relied on:**
- Peter: correctness first (2026-10-23 17:08); boundary-only input (2026-10-17).
- Supervisor: approvals at 06:27, 08:36, 08:51 and 09:23.
- LANE_RULES.

For project memory: "ISLA geometry: thin cells certified (F1, F2: K = 168 derived); F3's switch and the tie bands' 13-unit share are first-order estimates; the option-C follow-up lane goes with 0.12.0 (GEO-THIN, 2026-10-24)."

**GPU and running jobs:** 3.39 GPU-h of the 8 GPU-h cap, over 11 GB300 short jobs. Nothing is running, and the registry row has been removed. The job list is in `JOBS.md`.

**On disk:**
- The worktree `/home/psharpe/gh/physicsnemo-isla-geo-thin` (1.1 GB; pushed, clean; remove it after the merge).
- On LHR: `$T/kernel/geothin` (1.1 GB, mostly the canonical dumps in `dumps/`, kept for the option-C lane) and `$T/code_geothin3` (57 MB).
- Scratch: 3.7 MB.

---

## For the option-C lane (self-contained; line numbers are `geometry.py` at `6bac57cbd`)

The tie band is E_k = (u + 13 u64)(|x| + R_k), with `coef` at line 1747 and `err` at line 1820. The on-wall test is D_min - E_min <= 0. The tie test is D_k - E_k <= min_j(D_j + E_j), with a `reached` check on each feature. The docstring is at lines 1696-1705.

Computations whose error enters the bands or the switch, with their first-order worst cases in units of u64 (|x| + R_k):

| computation | where | worst case derived | observed max |
|---|---|---|---|
| Ericson face, edge and vertex regions (`_closest_point_weights`, sdf.py:292; `_triangle_distance2`, which F2's kernel test uses) | derivation at lines 343-386 (`_ERICSON_K`) | K l^2/(rho^2 L) along the face (steps 1-5); a region decided wrongly by rounding stays within that (step 6) | 0.0115 of the bound |
| edge points (`_segment_t`, line 176) | step 5 of `_FOOT_UNITS`, lines 395-425 | 14.5 u l + 3 u R_k, about 17.5 units | at most 0.18 E on real dumps (float64) |
| accurate face foot (`_accurate_weights`, line 471) | lines 395-425 | 77 units | 2.03 units |
| switch threshold (`_closest_weights`, lines 206-245) | 13 - 12 = 1 unit | not derivable from the worst cases | 9 of 520 wrong at 25 units, 0 at 1 unit |
| pseudo-normals for on-wall and tie directions (`n64`, `_closest_features`) | not derived | angle error about u64/rho | enter u only (A4: 3.9e-5 under one-ulp moves) |
| least-norm point (`_min_norm_points`, line 551; `slack` at line 591) | 2 d eps, not derived as a bound | Hoelder-1/2 at near-duplicate normals | 0 active-set changes on real nodes |
| winding number (`_solid_angle`, line 1234; far field in `_winding_kernel`) | not derived; sign only, outside the bands | — | 0 wrong signs on clearly off-wall queries |

What a full derivation still needs:
- A worst case, not a first-order estimate, for each row. That needs second-order terms, or interval and exact arithmetic for the uncertain pairs.
- A band that is pose-invariant. Measure |x| about the boundary's centroid. Keep the vertex term at least u32 times the stored coordinates, since the inputs' float32 storage rounding is irreducible; centring R_k too broke case 1909.
- The closest-point arithmetic done in the boundary's centred frame (NEARWALL's GEOMETRY_NOTE).
- The bitwise distinct-direction test at planar stars replaced by a certified one. That test, not the band, causes the 4-6k membership flips per lug case.

My estimate of the real outputs a derived band would change (unmeasured; low confidence):
- Float32 and bfloat16-cast queries: the float64 share is 2.4e-8 of the band, so widening it from 13 to 77 units moves the band by about 1e-7 relative. I expect 0 to a few changed ties or on-wall decisions per 200k-query draw.
- Float64 queries: the band would widen about 5.5 times. Ties could grow by the same order as today's count, 10.9k per 200k-query DrivAerML draw, so thousands of u values would change per draw, and d would change at newly on-wall queries.
- Centring the band changes every dataset's posed outputs at band-edge nodes: about 1 in 24 lug cases, as at node 6997.
