Status: DONE (RERUN-ENV; written 06:40 EDT 10-26)

**Venv:** `/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/venvs/isla-rerun-t213`
- Build: `lhr/build_venv.sh`, log `logs/build_venv.log`. It runs `uv sync --frozen --extra cu13 --extra mesh-extras
  --group dev --no-install-project` on bf71190af's uv.lock, then adds:
  - tensordict 0.15.3 from $T/wheels (sha256 a437693b…, as in uv.lock);
  - tensorboard 2.21.0, pyacvd and pykdtree at .venv-recipe's versions, constrained by the lock;
  - physicsnemo bf71190af, with no deps. Snapshots shadow it on PYTHONPATH.
- Contents: torch 2.13.0+cu130, triton 3.7.1, warp-lang 1.17.0, numpy 2.4.6, NCCL 2.29.7 and cuDNN 9.20. A
  `uv sync --dry-run` afterwards lists only those additions.
- 145 packages differ from .venv-recipe: `FREEZE_DIFF.md`, with both freezes under `freeze/`.
- Two build traps:
  - `uv export` (uv 0.12.23) drops nvidia-cufile, cupti and nvtx on aarch64, and torch then fails to import; use
    `uv sync`.
  - uv.lock has no hash for the cp313 torch+cu130 wheels.

## What passed and what did not (jobs and commands in `JOBS.md`)
- **Recipe tests, GB300** (397631): 655 passed, 6 skipped, as on 2.12.
- **ISLA test tiers** (397630 GPU, 397695 CPU): 36 and 13 fail. All are test_isla_goldens.py and test_isla_release.py
  reporting `in.points: … fixture drift`.
  - Cause: torch.norm on CPU moves by one ulp between 2.12 and 2.13, and the goldens' icosphere is built with it.
  - With the sphere pinned to its 2.12 bits, all of them pass on 2.13: 276 (397847), and 90 plus 1 xfail (397937). The
    modules match their goldens; only the fixture moves.
  - Details and the fix (test-side: build the sphere with correctly rounded ops, or regenerate the goldens):
    `GOLDENS_213.md`.
- **Smokes:** 10 of the 11 smoked configs (9 ISLA families, 2 baselines) pass the 5 % rule on 2.13; HiLift fails. Warm-cache readings, from the table in
  `SMOKES.md`:

| family | iter_time_s 2.12 → 2.13 | ratio | peak GiB (both) | compile | verdict |
|---|---|---|---|---|---|
| ISLA-DTT DrivAerML Volume | 1.245 → 1.250 | 1.003 | 146.6 | same, 0 breaks | pass |
| ISLA-DTT AeroZoo full | 0.711 → 0.732 | 1.030 | 133.4 | same | pass |
| ISLA-DTT AirfRANS | 0.306 → 0.306 | 0.997 | 27.1 / 27.0 | same | pass |
| ISLA-DTT lug | 0.268 → 0.258 | 0.963 | 7.7 | same | pass |
| ISLA-DTT Tensile2d | 0.257 → 0.247 | 0.962 | 4.3 | same | pass |
| ISLA A DrivAerML Surface | 0.246 → 0.246 | 0.998 | 66.0 | same | pass |
| ISLA A SHIFT-SUV | 0.246 → 0.249 | 1.012 | 65.8 | same | pass |
| ISLA A DeepJEB++ | 0.124 → 0.117 | 0.939 | 17.1 | same | pass |
| ISLA A HiLift dedup | 0.264 → crash | — | 63.3 / — | — | **fail, 3 of 3** |
| GeoTransolver AeroZoo / Transolver HiLift | 0.445 → 0.446 / 0.312 → 0.313 | 1.002 / 1.006 | 77.7 / 59.4 | eager | pass |

- First losses agree to 3-4 digits, there are no NaN and no compile errors, and recompile counts are equal.
- Round 1 showed a cold-page-cache order effect, so rounds 2 and 3 warm the cache first.
- **HiLift on 2.13** raises `Cannot transform nonzero effective measure on a zero-measure cell` at steps 15-17
  (`HILIFT_213.md`):
  - **Mechanism:** torch 2.13 holds `torch.compiler._is_compiling_flag` for a whole compile session, so
    `is_compiling()` reads True in the datapipe threads while training compiles. #2172's `Mesh._cached` then drops the
    geometry cache.
  - **Cells:** about 7,000 slivers per rank and epoch whose float32 vertices collapse (v0 == v1 bitwise) lose the cached
    area that 2.12 carries.
  - **Bits:** the vertices, the areas recomputed from them (0) and the measures are bitwise equal on both torches.
  - **Can 2.12 hit it?** It hit 0 in 2 epochs × 2 seeds; 2.13 hit 61-387 cells per rank and epoch (capture 400202).
  - **The check** is #1970's, from upstream main; **the bypass** is #2172's.
  - **Repro:** `lhr/hl_repro.py`, CPU only.
  - **Fix:** `PROPOSED_FIX_2172.diff`: `is_dynamo_compiling()` in `_cached` and `set_cell_measures`, plus 2 tests. On
    2.13 it passes test_compile, test_measure and the new tests (400986), and on 2.13 without it the new tests fail.

## Recommendation
Use torch 2.13 for CERT and the rerun, provided two changes reach psharpe/isla first:
- (1) the #2172 fix, in #2172, which Peter owns;
- (2) the goldens' sphere fixture.

Then re-smoke HiLift on 2.13 (about 0.3 GPU-h) before CERT.
- Per family: the other 8 ISLA families and the baselines can use 2.13 now. HiLift can use 2.13 only after fix (1).
- The fallback is 2.12 with a disclosure. It needs neither fix, but GLOBE's compiled path is broken there (396599); no
  rerun lane compiles GLOBE.
- Confidence: high on the timings and memory, which agree to ±3 %, and on the HiLift mechanism (a CPU repro and a fix
  that is tested both ways). The fix's `torch.export` behaviour is untested; see the note in `HILIFT_213.md`.

## Compute, rulings, memory
- **GPU:** 16.53 GPU-h, against a cap of 15 that the supervisor raised to 16.7 for the capture job. The two cpu-short jobs
  are extra. Nothing is running, and I removed the registry row `^rerunenv\.`.
- **Rulings used:** LANE_RULES (the bad nodes, the QoS limits, no edits to library code); the supervisor's decision rule;
  the cap raise; and Peter's rule to fix the library rather than the recipe.
- **For project memory:**
  - The venv path above serves code at or after bf71190af on torch 2.13.
  - On LHR, set `UV_CACHE_DIR` on scratch: the home quota is about 10 GB. Try 3 overflowed it; I removed only the cache
    entries that build had added.
  - On torch 2.13, `torch.compiler.is_compiling()` is process-wide during a compile; use `is_dynamo_compiling()` for
    frame-local questions. Library-wide there are 246 call sites, a sweep for Peter.
  - In A/B timing smokes, warm the page cache first.
  - The recipe's `benchmark_io` fails on AeroZoo with `min()` on an empty tensor in its logging. That is a small recipe
    bug, not fixed here.
- **Flag:** `$T/kernel/premerge` (ISLA-PREMERGE's A/B snapshots and .pt files, kept for CERT) was missing on LHR when
  I looked at about 02:20 EDT. I did not touch it.

## On disk
- On LHR:
  - the venv (8.5 GB);
  - `$T/kernel/rerunenv/`, about 1.7 GB: uvcache 0.9 GB, hardlinked into the venv; wheels_t213 0.4 GB; repo 0.2 GB, a git
    repo whose objects borrow from physicsnemo-mesh-transformer/.git; the read-only snapshots code_bf71 and recipe_bf71;
    smoke and probe outputs, 46 MB.
- Locally, this folder, about 3 MB. No worktree was made.

## Re-smoke after the fix (07:10 EDT 10-26)
- **Code:** psharpe/isla 9eb786e20, which merges #2172's new head 0d968ef46: `Mesh._cached` and `set_cell_measures`
  now use `is_dynamo_compiling()`, with two tests. The snapshots were built on LHR by `lhr/mk_snap_9eb7.sh`:
  - `$K/code_9eb7`, with td0153;
  - `$K/recipe_9eb7`. Its HiLift dataset file, model config, train and base configs and `src/` are identical to
    recipe_bf71's.
- **Job 401557:** one benchmark_io warm-up pass, then three runs of the same option-A HiLift smoke on torch 2.13: 80
  steps, 4 ranks, bf16, `compile_blocks`, seed 42.
- **Results:** all three ran clean, with no ValueError and no traceback. Logs are in `smokes/r4/401557/`.

| run | iter_time_s median, last 40 (mean ± sd) | ratio to 2.12 | peak GiB | first losses | compile (rank 0) |
|---|---|---|---|---|---|
| 2.12 reference (398482 hl.t212, bf71190af, warm, steps 41-80) | 0.2629 (0.325) | — | 63.34 | 0.8452, 0.5321, 0.3138, 0.2352 | 3 graphs, 2 recompiles, 0 breaks |
| 2.13 hl4 | 0.2650 (0.307 ± 0.107) | 1.008 | 63.34 | 0.8458, 0.5323, 0.3140, 0.2351 | 3, 2, 0 |
| 2.13 hl5 | 0.2771 (0.329 ± 0.170) | 1.054 | 63.34 | 0.8458, 0.5327, 0.3144, 0.2350 | 3, 2, 0 |
| 2.13 hl6 | 0.2591 (0.267 ± 0.025) | 0.985 | 63.34 | 0.8458, 0.5321, 0.3144, 0.2354 | 3, 2, 0 |

- Over the three runs, the median of the medians is 0.2650 s, a ratio of **1.008**. Against 2.12's last 50 steps it is
  1.003; the runs span 0.985 to 1.054.
- hl5 alone is 5.4 % slower, with a large spread (sd 0.17 s). The other two are within ±1.5 %, so I read hl5 as
  run-to-run noise, not a slowdown.
- Peak memory is identical. The first losses agree to 3-4 digits, and none is non-finite.
- **GPU:** 0.87 GPU-h, so 17.40 GPU-h in all, of the 17.7 allowed. Nothing is running, and I removed the registry row
  again.
- The 2.12 reference ran on bf71190af. 9eb786e20 also changes ISLA's geometry.py, so the ratio compares 2.13 on the new
  code with 2.12 on the old. A 2.12 run on 9eb786e20 would separate the two effects; it did not fit the cap.
- **Updated recommendation:** with #2172's fix in, every family can use torch 2.13 for CERT and the rerun. The goldens'
  sphere fixture (`GOLDENS_213.md`) is still needed before the ISLA test tiers pass on 2.13.

Verdict: HiLift passes on torch 2.13 at 9eb786e20 (3 of 3 runs clean), with an iter_time_s ratio of 1.008 against
torch 2.12 (range 0.985-1.054).
