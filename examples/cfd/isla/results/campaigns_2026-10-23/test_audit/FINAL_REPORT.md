Status: DONE (TEST-AUDIT; audit and plan only; updated 23:40 EDT 10-24 with the determinism probe the supervisor asked for at 23:28)

**What ran.** No code, branches, commits or GitHub writes. Code at 5b70c8686, from the read-only snapshot `code_mainmerge_base`. Details in `JOBS.md`.
- LHR cpu-short 376354: the ISLA directory plain and under `coverage --branch`, and the recipe tests under coverage.
- GB300 376355: the ISLA directory under coverage, 1.77 GPU-h.
- GB300 377000: the determinism probe, 0.18 GPU-h, after the cap was raised to 2.7. Lane total 1.95 GPU-h. Registry rows added and removed.
- Four read-only children: `REPORT_{QUALITY,PROPMAP_MODEL,PROPMAP_KERNELS,P0_INVENTORY}.md`.
- Local probes: `probes/rtol_probe.py`, which ran unlocked before the 23:16 rule; `probes/det_probe.py`, run on CPU under the lock.

**Coverage** (`COVERAGE.md`; Triton and Warp kernel bodies excluded). Non-kernel lines, CPU / CPU ∪ GPU:

| | modules |
|---|---|
| 100 % | blocks, heads, sampling, geometry_fast (on CPU ∪ GPU) |
| 93-99 % | geometry, tree_build, tree_read, inputs, model 96.2 / 96.5, scales, tree_walk |
| low | segment_softmax 66 / 81, fused_basis 52 / 85, fused_read 51 / 77, geo_kernel 31 / 68 |
| total | 88.2 / 94.4 % lines; 78.9 / 85.1 % branches |

- The low ones are mostly CUDA backward, which runs on autograd's own device thread. coverage.py does not trace that thread, although the tests do run the code.
- Never run: several rank-1 vectors on one condition (`inputs.py:211`); the ball search's second pass (`geometry.py:1114`); geo-kernel S > 256; 38 untested raises.
- Runtime: CPU 951 s, of which 9 compile tests take 874 s; GPU 28-30 min. 241 of 702 tests skip on CPU.
- No flaky tests. The recipe's `test_campaign_c_hooks` fails 2 tests inside sbatch.

**MOD-008** (`AUDIT.md` §1).
- 008a partial: about 18 attributes are not asserted.
- 008b met in form only: one small fp32 configuration at 1e-5/1e-4, and no `query_chunks` golden.
- 008c met for the surface task only: there is no ISLA-DTT `.mdlus`.

**Determinism** (measured; `DETERMINISM.md`). GPU eager with kernels auto, run 6 times:
- **Bitwise** for the geometry, the kept cells, the tokens, the fused read and geo region, every block, the outputs, and the surface task.
- **Not bitwise:**
  - The draw's float64 HT weights differ by ulps, from CUDA's `cumsum`. They become bitwise under `use_deterministic_algorithms`, or with every cell kept.
  - Where a hit lands on a line boundary, a kept cell flips. That is the likely cause of REYNOLDS-OUT's AeroZoo case 0 (inferred). One hook in their `reyout_repeat.py` would confirm it.
  - The reference (unfused) tree read differs by about 2e-6 on CUDA.
- My recommendation:
  - no global deterministic mode;
  - goldens read CPU bitwise, GPU (kernels auto) bitwise except the weights;
  - the reference read judged against its 5-run spread;
  - the A/B harness compares `kept` first, and a flip there makes the comparison inconclusive.

**Gaps, ranked** (`AUDIT.md` §2):
1. No bitwise module golden.
2. ISLA-API's converted interior tests run ISLA-DTT with a read that adds zero. Sent to the supervisor at 23:20.
3. Interior cell permutation is untested.
4. Patch-and-compare tests could pass vacuously after LIB-MOVE.
5. ISLA-API deletes the sampling-under-compile proof that LIB-MOVE row 9 cites.
6. Fast geometry is checked only in 3-D.
7. Chunking is never tested with the draw active.

Also: no gradcheck or opcheck; no interior parity test. The inherited `rtol=1e-5` is measured safe to tighten (8 calls depend on it).

**Plan** (`TEST_PLAN.md`; three writers):

| phase | lines | writer | content | start |
|---|---|---|---|---|
| P0 | ~1,300 | W1 | A/B harness, plus committed goldens of at most 1 MB | now |
| P1 | ~450 | W1 | MOD-008a/b/c | after ISLA-API merges |
| P2 | ~750 | W2 | properties | after ISLA-API merges |
| P3 | ~850 | W3, GPU | gradcheck and opcheck, kernels, determinism tests | now |
| P4 | ~150 | — | tiers: `-m "not slow"` about 80 s; a 4-way GB300 launcher about 8 min | last |

`TEST_PLAN.md` also lists, for each LIB-MOVE phase, the tests that must come first, the tests that move, and those that stay. About 10-12 GPU-h; done by about 10-28.

**For you to decide:**
1. ISLA's CPU CI on `-m "not slow"`.
2. A 1 MB cap on committed golden data.
3. `live()` added inside ISLA-API's conversion.
4. Triton tested only on LHR.
5. **New:** make the draw deterministic with exact int64 fixed-point sums. It changes which cells are kept at boundary hits, so it is a model change for Peter. The alternative that changes no numbers is CPU cumsums at a host sync.

**Rulings relied on.**
- Peter's 22:05 test-suite direction.
- The GPU-tests rule.
- The registry rule.
- The custom-op rule.
- The 23:16 lock rule.
- The supervisor's 23:31 cap raise.

**On disk.** This folder, about 2.4 MB. No worktrees, venvs or running jobs.
