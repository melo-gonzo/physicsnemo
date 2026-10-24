Status: DONE (ALIGN8, 2026-10-24 01:55 EDT). The slowdown is real and located. A padding fix that changes no output is pushed. In the current standings, though, the GPU-h at stake is small (about 1-3 % of the rerun), not a large share as the brief assumed: AeroZoo and DrivAerML, which carry most of the cost, happen to be aligned.

**What ran.** All on LHR, short QoS, one node each:
- GPU: 362691 (op benchmark), 362753, 363327 and 363671 (timing and profiles). 363671 was cancelled at 55 min to stay under the cap.
- cpu-short: 362791-3 and 363324-6 (the ISLA test directory), 362817-8 (exposure counts, child lane ALIGN8-EXPO).
- GPU use: 5.86 GPU-h of the 6 GPU-h cap. The registry row was added at 23:11 and removed at 01:49.
- Snapshots, md5s and the full log are in `JOBS.md`. Harness: `lhr/align_time.py` (the recipe's own pipeline, collate and model build; bf16 autocast; medians of 10 after warm-up).

**Reproduction** (GB300, bf16, step in ms). Full table, with the forward pass and every dataset: `results/repro_table.md`. "Base" is 32fa15389; "fix" is e60be94d4 at j = 1..7.

| case, path | N0 | base N0 | base odd N | base even N, not mod 8 | fix | odd / fix |
|---|---|---|---|---|---|---|
| DrivAerML Surface, eager | 200,000 | 349 | 470 | 422 | 351 | 1.34 |
| DrivAerML Surface, compile_blocks | 200,000 | 219 | 313 | 268 | 223 | 1.40 |
| AeroZoo, eager | 400,000 | 723 | 966 | 872 | 731 | 1.32 |
| AirfRANS ISLA, eager (standing) | 100,992 | 662 | 727 | 701 | 668 | 1.09 |
| AirfRANS ISLA-DTT, eager (standing) | 100,992 | 820 | 891 | 863 | 833 | 1.07 |
| lug / Tensile2d / DeepJEB++, eager | 10k / 6.5k / 24k | | | | | see below |

SURFACE-BC's +20 % was an even count. Odd counts cost +32-40 %. compile_blocks is hit as hard as eager, because its token-axis bmms are still cuBLAS calls. The forward pass alone loses +6-27 %.

The small cases are launch-bound and their wall times are noisy (10 to 25 % spread run to run), so wall time does not resolve them. Profiled GPU time does: lug +5 % at N+1, Tensile2d +8.6 %, DeepJEB++ under 1 %, and the fix brings both lug and Tensile2d back to the aligned value. One wall-time reading on the lug has the fix about 10 % slower even at j = 0, where it pads nothing. I read this as noise but have not proven it; see what is left.

**Cause** (`results/gemm_362691.jsonl`, profiles in `results/out_v*/prof_*.txt`). It is not the per-token linear layers, the softmax, or the broadcast: those run at the same speed at any N. Two groups of operations slow down:
- **Token-axis GEMMs in the slice attention.** These are the bmms whose token axis is the reduction axis or the output's inner axis: the slice pooling `einsum(bns,bnh->bsh)` (0.63 to 2.30 ms at odd N, 1.36 at even N), the prototype attention `einsum(sh,bnh->bsn)` (1.19 to 2.94 ms) and its value sum (2.94 to 6.50 ms). cuBLAS leaves its sm103 nvjet kernels for cutlass_80 ones when the token stride is not 16 bytes.
- **LayerNorm backward.** Its weight-gradient reduction runs on 32-row tiles above 65,536 rows and takes the aligned path only when N is a multiple of 32 (`layer_norm_kernel.cu`). This costs 1.49 to 2.54 ms per call, and it is why an 8-aligned count (200,008) still cost 377 ms against 350 ms. It reaches DrivAerML, AeroZoo and AirfRANS.

**Exposure.** Details: `exposure/exposure.tsv` (N mod 8 histograms), `exposure/exposure_classes.tsv` (odd, even, multiple of 8, multiple of 32; per case and per 4-rank step) and `exposure/exposure_gpuh.md`.
- N is fixed per case. The in-model draws change which cells are taken, never how many.
- Aligned on every case (multiples of 32): AeroZoo (all 7 splits; the water fill only splits 200,000 among roles), DrivAerML Volume (body or whole), DrivAerML Surface (body) and SHIFT-SUV.
- Misaligned per case: AirfRANS 87 %, AirfRANS + freestream 74 % (always even), lug 64 %, Tensile2d 89 %, DeepJEB++ 75 %, and DrivAerML Surface whole boundary 100 % (205,054 on every case).
- With 4 DDP ranks waiting on the slowest, 98-100 % of steps on those datasets are exposed.
- GPU-h at stake, from sacct actuals of the standing lanes: about 14 each for AirfRANS ISLA and ISLA-DTT (178 and 221 GPU-h lanes), about 16 for AirfRANS + freestream, and about 6 for DrivAerML Surface whole if adopted. Upper bounds for the small-token lanes are lug at most 8 (ISLA) and 16 (DTT), Tensile2d at most 4, and DeepJEB++ at most 10.
- In all, about 28 GPU-h measured and at most about 70 per two-seed set: 1-3 % of a 2,500-4,000 GPU-h rerun.
- Counterfactual: a count one off on AeroZoo would cost +32 % per step, about 1,100-2,200 GPU-h on REF-090's AeroZoo lanes. The fix removes that exposure for good.

**Fix.** Branch `psharpe/isla-align8` (pushed with a plain push, not merged), two commits made with `-s` and hooks on:
- 0fef70a42: pads to a multiple of 8.
- e60be94d4: pads to `TOKEN_ALIGNMENT = 32`, appended with `F.pad`.

`_forward_tokens` appends `-N mod 32` padding tokens after the boundary and prediction tokens: every field zero and log-weight -inf, the way passive tokens are. They hold no mass in any softmax over points, the tree read and the head read only the real tokens, and the gauge is computed before padding. No routing mass of any prediction point changes, so the supervisor stop in task 3 did not apply.
- It has no flag or option, adds no parameter, and leaves ISLA at 0.9.0, so the state dict is unchanged.
- Why 32 and `F.pad`: v1 at 8 left the LayerNorm cost. v1's separate block of padding tokens also made the whole-model compile guard on `N % 8 == 0` and recompile (test_isla_compile surface).
- Test changes: the new `test_isla_token_padding.py`, and the query-encoding test's hook now reads the query tokens by position, not as the last tokens.

**Equivalence** (CPU, padded against unpadded; `probes/equiv.py`, results in `probes/equiv_v2.jsonl`). Cases: surface (1,003 tokens), interior ISLA, interior ISLA-DTT (421 tokens), and 2-D interior (150 tokens).

| precision | max abs output difference | max abs gradient difference |
|---|---|---|
| float32 | 1.2e-6 (outputs about 1-2) | 3.4e-5 (gradients up to 105) |
| float64 | 2.3e-15 | 7.1e-14 |

The ISLA test directory on CPU (363324-6) gives 276 passed, 47 skipped (CUDA-only), 0 failed: the base's 273 plus the 3 new tests.

**Before and after.** The fix holds every j at the aligned time:
- DrivAerML Surface: eager 470 → 351 ms, compile_blocks 313 → 223 ms.
- AeroZoo, eager: 966 → 731 ms.
- AirfRANS eager: ISLA 727 → 668 ms, ISLA-DTT 891 → 833 ms.
- At j = 0, where nothing is padded, the fix costs at most +1 %.

**For the program.** Merging e60be94d4 before the final rerun removes this slowdown on every dataset, including the canonical whole-boundary AirfRANS and DrivAerML Surface. It is cheap insurance against an expensive accident on AeroZoo and DrivAerML. I am confident in the cause and the equivalence, and moderately confident in the GPU-h estimates: the harness has no DDP or dataloader, so the fractions are upper estimates for a training step.

**Rulings relied on:** the LANE_RULES (LHR only, hooks on, no merge, whole-test-directory rule), the brief's numerical-meaning stop, and Peter's whole-boundary ruling for which variants count.

**For project memory:**
- In bf16 on GB300, ISLA steps cost +20 % at even token counts that are not multiples of 8, and +32-40 % at odd counts, eager and compile_blocks alike. The cause is the slice attention's token-axis bmms plus the LayerNorm-backward 32-row tile.
- `psharpe/isla-align8` pads the token axis to a multiple of 32, equal to roundoff.

**Left:** the merge, for the supervisor. Optionally, an interleaved repeat of the lug and Tensile2d wall times to settle the noisy small-case reading, and DeepJEB++'s compile_blocks timing after the fix (not measured: job cancelled). Nothing is running.

**On disk:** worktree `/home/psharpe/gh/physicsnemo-isla-align8` (1.1 GB; pushed, remove after merge or drop); LHR snapshots `$T/code_align8{a,b,c,d}` (57 MB each) and `$T/recipe_align8`; scratch folder 9.4 MB.
