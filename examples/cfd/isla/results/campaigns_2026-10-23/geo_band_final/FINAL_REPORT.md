Status: DONE (phases 0, A, B, C; phase D not started, estimate below)

**Outcome.** The tie and on-wall bands of ISLA's exact query geometry now rest on derived worst cases, and the search arithmetic runs in the boundary's centred frame. The branch `psharpe/isla-geo-band` is pushed at `2e9cd7fda` and was not merged into `psharpe/isla`:
- `9e0d52e27` is the change (`git commit -s`, hooks on).
- `2e9cd7fda` merges `psharpe/isla` at `eba038f08` into it (`--no-ff --signoff`).
- The branch merges with DTT-POSE's frozen `0f06e4065` without conflict (`git merge-tree`).
- No version bump: ISLA 0.12.0 is folded at merge.
- Fast and reference stay bitwise equal. The fast path needed no change of its own.

**The rule (VF, approved 11:24)**, one canonical rule with no flag:
- The band is `E_k = u (|x| + R_k) + C u64 (|x'| + R'_k)`. The first term is the inputs' storage rounding, in the frame they were given in. The second is float64's arithmetic in the frame centred on `c`, the measure-weighted centroid computed with fixed-point sums.
- C = 91 in 3-D and 17 in 2-D (was a first-order 13 and 10). The 1-unit switch threshold is certified by construction.
- A tie needs two certified-distinct directions.
- The offset x - p is formed in float64.

**What ran** (8 jobs: 6 on GB300 short, 6.13 of the 8 GPU-h cap; 2 on cpu-short; details in `JOBS.md`):
- 368333: phase 0 timing.
- 368478 and 368608: phase A probes.
- 369316: checks.
- 369317: inference check.
- 369573: final GB300 suites.
- 369318 and 369574: the ISLA directory on cpu-short.
- 369319: recipe tests.
- Children: a lane-writer (3c542568, implementation) and a lane-reviewer (2c9a2209, independent re-derivation).

**Results** (`RESULTS.md`, `DERIVATION.md` v2, `PHASE0.md`, `INFER_CHECK.md`):
- **Phase 0 cost table** (sent 10:30, `PHASE0.md`): versus FAST-MERGE's head, the lug's geometry call rose 44 to 77 ms (+75 %, reference) and 37 to 78 ms (+112 %, fast). DrivAerML whole boundary rose 431 to 500 ms (reference) and 354 to 433 ms (fast).
- **Derivation:** the accurate face foot's worst case is 87 units. Its worst path is a face foot that rounding rejects; the first draft missed it and over-counted the term along the edge. The kept guarded pair is worst at 88, plus 3 for the centring and second order: C = 91. K = 168 is a worst case. The reviewer's corrections are applied to `DERIVATION.md` and the code comments. One point stays open: the reviewer counts the 2-D brute-force margin for float64 queries at 33.5 > 32; I count `err_max` as already covering it. Either way only candidates change, so no output does.
- **Native change, float32:**
  - 0 winding-sign and 0 on-wall changes on every dataset.
  - d changes at rounding level. The largest relative change is 0.7 %, at AirfRANS's first nodes, where the old float32 cancellation was that wrong.
  - u changes by up to 6.5e-3 at 7.9k DrivAerML ties, 1.6e-5 on the lug, and not at all on Tensile2d.
- **Pose (translation of one RMS radius):**
  - Tie-membership flips fall from 90,221 to 25 (lug) and from 231 to 0 (Tensile2d).
  - DrivAerML u flips go from 3,000 to 3,706 per 400k. These are band-edge flips, each within its certified direction uncertainty.
  - NEARWALL's compute rounding falls to 1.9e-7: DrivAerML was 0.0156, AirfRANS off-wall 0.29.
  - Lug case 1884's node 6997 remains a band-edge node, answered correctly for its own inputs in each pose, with an excess of 1.74e-5.
- **Bitwise equality with VF:** 106 of 106 runs on every dump, covering every op output, d, u and the gradients. The fast path differs at 0 of 588,689 queries.
- **Cost** on the DrivAerML whole-boundary call: +13 ms reference (+2.8 %), +14 ms fast (+3.5 %).
- **Tests:**
  - Whole ISLA directory on GB300 with every CUDA test: 548 passed, 1 skipped (environment-gated), 0 failed.
  - Whole ISLA directory on CPU: 409 passed, 0 failed.
  - Recipe tests: 551 passed, 2 failed. Both are `test_campaign_c_hooks`, with the same E line as the base.
  - The new far-from-origin and posed planar-star tests fail at 921472488.
- **Inference check (pre-registered, gated):** O reproduces 36 of 36 of TIE-EXP's registered summaries. Every \(|\log(N/O)| \le 2.96\times10^{-5}\) and every posed/native change \(\le 4.4\times10^{-5}\), against limits of 0.01. No retrain is needed.
- **Sensitivity checks** (not gated; both lanes trained with 0.9.0's geometry; N/O only, since their reads are blinded): AF-WHOLE seed 42 at epoch 197 has N/O of 0.99994-1.00003. FLOOR-DRV fdF s42 at step 2000 has 0.99990-1.00014.

**What it means:** the geometry's decisions are certified and its arithmetic no longer depends on where the boundary sits, at little cost and without harming trained models. Confidence is high: the code is bitwise equal to the measured rule and the bounds were re-derived independently.

One limit is irreducible. A case's float32 storage rounding grows with its distance from the origin, so band-edge nodes can still flip under translation. The extreme case is AirfRANS with its far field translated by one boundary RMS radius: 16k near-wall nodes become on-wall in every variant.

**Rulings relied on:**
- Peter: correctness first (2026-10-23 17:08); boundary-only input (2026-10-17).
- Supervisor, 11:24: VF; G4 in; the inference criterion and the two sensitivity checks; the comment above `_ERICSON_K` to match the derivation.
- Supervisor, 12:25: merge `eba038f08`, then the whole directory on GB300.

For project memory: "ISLA geometry bands: storage term u (abs x + R_k) in the inputs' own frame (a centroid-measured storage term is not a bound); arithmetic share C = 91 / 17, derived and independently re-derived, in the centred frame; certified tie directions; float64 x - p (GEO-BAND, 2026-10-24, for 0.12.0)."

**Phase D, not started.** It does not fit the remaining 1.9 GPU-h: development, verification and timing need about 4. What it would take:
- One Warp kernel over the candidate pairs: the guard (`_thin_bounds`' kappa, l^2, `_ericson_error`) and, on guarded rows, `_accurate_weights` with its exact cross-product expansion and the switch.
- It replaces about a hundred small torch launches, made twice per call (the seed and the pairs). GEO-THIN measured that cost at about 25 ms per call; on the lug it is most of the 33 ms the thin-cell fix added.
- Bitwise equality needs the kernel's module compiled with `fuse_fp: False`. Warp 1.17 defaults to FMA contraction (`--fmad=true`), which would break the Dekker splits.
- It also needs torch's own reduction order for the 3-component norms and sums. Matching that is empirical; the alternative is to make the reference use the same explicit elementwise form, which changes outputs at rounding level and needs approval.
- Estimate: half a day of writer time, one GB300 bitwise check on every dump, and one timing job.

**Left and on disk:** nothing is running, and the registry row is removed.
- Worktree `/home/psharpe/gh/physicsnemo-isla-geo-band`: about 1.1 GB, pushed and clean. Remove it after the merge.
- LHR: `$T/code_geoband1` and `$T/code_geoband2`, 57 MB each; `$T/kernel/geoband`, 6.7 MB. The copied checkpoints and large outputs are deleted.
- Scratch folder: a few MB.
- Book changes are drafted in `NOTEBOOK_DRAFT.md`.
