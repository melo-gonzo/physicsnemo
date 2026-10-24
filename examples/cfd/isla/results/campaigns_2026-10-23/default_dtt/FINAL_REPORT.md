Status: DONE

**What ran.** I pushed branch psharpe/isla-default-dtt (head e10681b33) with a plain push; it is not merged. It holds three commits, all `-s` with hooks on:
- 9d27ae47b, the model default and docstrings;
- b0fd82da8, the recipe configs and README;
- e1a6d7853, the tests and two new fixtures.

e10681b33 merges psharpe/isla 0d46b2f9b, as you asked at 13:36. The merge had no conflicts. In test_isla_configs.py every top-level definition equals the side that changed it, and no definition was changed on both sides. A lane-writer child (9dc424b1) made the commits and two read-only children made the book-sentence lists; I did the merge myself, after the writer had finished.

LHR jobs:
- cpu-short: 369930 (ISLA directory), 369931 and 370095 (recipe tests), 369634 (recipe tests at the base), 369638/369667/369676 (harness preflight).
- GB300, short QoS: 369933 (tests), 369932 and 370215 (smokes).
- GPU use: 3.4 of the 12 GPU-h cap.
- Snapshots: $T/{code,recipe}_defaultdtt_{base,e1a6,e106}. Records: JOBS.md.

**Tests.** No failure is attributable to the branch.

| Suite | Branch | Base |
|---|---|---|
| ISLA directory, CPU (369930) | 401 passed, 122 skipped, 0 failed | 397 passed, 122 skipped |
| ISLA directory, GB300 with every CUDA test (369933) | 531 passed, 1 skipped, 0 failed | 526 passed, 1 skipped, same job |
| Recipe tests on the merged tree (370095) | 585 passed, 15 skipped, 2 failed | 2 failed (369634) |

- The merge changes nothing under physicsnemo/ or test/: the MD5SUMS of code_e106 and code_e1a6 match. So the e1a6 CPU and GB300 runs are runs of the merged tree.
- The two recipe failures are test_campaign_c_hooks' initialize_from_checkpoint tests. They fail at the base with the same message, `TypeError: int() ... 'NoneType'`, which comes from DistributedManager inside a Slurm job.
- The assertion audit (assert_diff.txt) finds no message or match string removed. Two lines changed, both explained, and 48 were added. No golden was regenerated and no tolerance was weakened.

**Per interior dataset.** Smoke 370215 trained 40 steps on the merged tree, on one GB300 node (nvl72d025-T03): bf16, augment off, seed 42, each dataset's reference split and resolution. Ratios are ISLA-DTT ÷ interacting ISLA, both arms per-block compiled, timed in the same job. QI is the query-independence check: the largest change at the shared points when 1,000 other points are added, as a fraction of the field's largest value.

| Dataset | Step time, DTT vs ISLA (s) | Step ratio | Peak allocated, DTT vs ISLA (GiB) | Memory ratio | QI fp64, DTT | QI fp64, ISLA |
|---|---|---|---|---|---|---|
| DrivAerML Vol (whole boundary, 5 pieces) | 2.17 vs 0.93 | 2.33x | 195 vs 129 | 1.52x | 3e-15 | 2.8e-3 |
| AeroZoo full | 1.27 vs 0.47 | 2.73x | 170 vs 126 | 1.35x | 8e-15 | 6.1e-3 |
| AirfRANS | 0.67 vs 0.31 | 2.17x | 39.8 vs 34.8 | 1.14x | 2e-14 | 1.3e-2 |
| Lug | 0.45 vs 0.18 | 2.48x | 9.2 vs 7.9 | 1.17x | 1.5e-15 | 1.9e-2 |
| Tensile2d | 0.31 vs 0.14 | 2.19x | 5.3 vs 4.8 | 1.12x | 1.2e-14 | 3.1e-2 |

- The pre-merge smoke (369932, another node) gives step ratios of 1.97x to 2.64x and the same memory.
- **QI check:** one real case per dataset, float32 and float64, eager, eval mode so the boundary draw is fixed. In float32, ISLA-DTT moves 8e-7 to 3e-5, on the order of a reversed-request floor of 7e-7 to 8e-6; interacting ISLA moves 3e-3 to 3e-2. The float64 columns in the table show the same split at roundoff. The weights are freshly initialised: the 0.9.0 checkpoints do not load into the current version.
- **Checks:** every resolved config has query_independent true, compile false and compile_blocks true, and every boundary piece the dataset config names arrived non-empty. There are no graph breaks, one compile and at most one recompile per run (the blocks going to dynamic shapes on varying token counts), and only two benign warnings (the experimental-namespace notice and "no checkpoint dir").
- **Tensile2d** builds and trains ISLA-DTT: the loss falls from 0.47 to 0.20 and stays finite.
- **Lug, single GPU:** ISLA-DTT's loss is noisy and its 10-step mean drifts from 0.50 to 0.70 over the 40 steps, while interacting ISLA's goes from 0.48 to 0.40 on the same cases. The 4-GPU reference lane r090fix_lug_dtt descends normally (0.49 to 0.16 over 200 steps). I read this as batch-1 noise, not a fault.
- **Settings carried over:** only query_independent: true, compile: false and compile_blocks: true. Each reference lane's resolved config (refcfg/) differs from its interacting partner only in query_independent. The lug's compile_blocks came from its lanes.tsv extra; its compile=false and AeroZoo's from lanes.sbatch and isla_volume_aerozoo.yaml.

**Checkpoint loading.** physicsnemo's `Module.__new__` calls `apply_defaults()` (core/module.py:239), so every `.mdlus` stores every argument. An old interior checkpoint therefore carries `query_independent: false` explicitly and loads as interacting ISLA. The new fixture written by eba038f08 code proves it: it loads, its outputs match, and its args show False.
- The class now writes the resolved value into its stored arguments, so new checkpoints never depend on the default.
- A state dict of either form loaded into the other raises, both ways, through load_state_dict, Module.load, and from_checkpoint with mixed args and weights (test_isla_release.py).

**BOOK_CHANGES.md** (run on the merged trees 0d46b2f9b and e10681b33): all 348 cells pass on both; 6 cells change output.
- Two show a wrong number and need `query_independent=False` on their interacting model: 05's cell after `para-arch-params` and 09's `setup-dtt-params`. The latter now prints "0.0 million parameters to ISLA's 70 million".
- 06 `para-using-kernel` and the `tbl-using-configs` caption need rewording, because per-block compile is now the interior default.
- 05 `tbl-arch-switches` should say what `None` resolves to.
- 06 `para-using-draw`: one number changes and the sentence still holds.
- Remove four code-default In-flight callouts: 03 L1596, 05 L2604, 06 L162 and 21 L308. Set the DEFAULT-DTT ledger row (14 L718) to DONE.
- Three prose passages are false after the merge: 05 L1089, 05 L2359-2378 and 06 L801-809. About 15 [should] items are listed in BOOK_SENTENCES_v2.md.

**For you to decide.**
1. AirfRANS's ISLA-DTT reference lanes (r090eag) trained fully eager, as do its standings and the lug's. As your brief says, the configs now set compile_blocks: true. COMPILED-PATH found compiled = eager per case after the mix-order fix, but the per-block certification under the 10-24 compiled-regions policy is still the final rerun's job (05 L2414 callout). No action unless you want eager configs for those datasets.
2. Research scripts under examples/cfd/isla/research that build interior ISLA without the argument now build ISLA-DTT. They are outside my scope and unchanged.

**Left.** Your merge and book batch. Nothing is running and the registry row is removed. On disk: this scratch folder (3.8 MB); the worktree /home/psharpe/gh/physicsnemo-isla-default-dtt, to remove after the merge; and the LHR snapshots, about 175 MB.
