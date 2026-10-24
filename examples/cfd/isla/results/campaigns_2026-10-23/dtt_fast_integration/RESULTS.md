# DTT-FAST integration pass 1: results (2026-10-24 04:58 EDT)

Branch `psharpe/isla-dtt-fast`, pushed, head **5fa4efb17** (639508864, then the removal of `tree_read_checkpoint` in f1ae05c75, then a
test-only commit; see the last sections). ISLA stays at version 0.11.0 (`_version` 11), and the state dict is unchanged.
`psharpe/isla-dttfast-qchunk` is not merged; a second pass merges it.

## Merges and commits (`git merge --no-ff --signoff`, hooks on)

| commit | what | conflicts |
|---|---|---|
| da21b38c6 | merge `origin/psharpe/isla` 1e7f03d7d (ISLA 0.11.0, the geometry bundle) | none. 0.11.0's version, `_version` and refusal text are kept as is. The merge also takes the book figure renames already on psharpe/isla |
| 023b0556d | merge `origin/psharpe/isla-dttfast-forest` 60ae18202 | model.py (Execution docstring; `tree_read_checkpoint` beside `tree_build`; `walk=` beside `build=`) and tree_read.py (imports, the `walk`/`build` arguments, the docstring). Every conflict had both sides adding lines in one place, so both were kept (`resolve_forest.py`) |
| cd26d1876 | merge `origin/psharpe/isla-dttfast-fusedread` 6aada61cd | model.py (docstring, attribute, `kernel=`), tree_read.py (signature, docstring) and the test file (forest tests beside the fused-read tests). Again both sides were kept, the review's P2.3 included (`resolve_fusedread.py`); ruff-format added two blank lines at the join |
| 639508864 | **one execution attribute** | model.py +22/-26, test_isla_tree_read.py +47/-14 |
| f1ae05c75 | **no `tree_read_checkpoint`** (parent's note of 04:15) | model.py +9/-31, test_isla_tree_read.py -100 |
| 5fa4efb17 | fast-vs-reference forest and walk tests no longer pin the reference's autocast precision (parent's note of 04:22, DTT-POSE) | test_isla_tree_read.py +4/-6 |

**The fold (639508864).** `ISLA.tree_read_kernel` is the only execution attribute of the tree read. It takes
`"auto"` (the default) or `"reference"`, and any other value raises `ValueError` at the forward pass.
- The model passes the attribute itself to `tree_read_pairs(walk=..., build=...)`, and
  `resolve_tree_read_kernel` (`"fused"`/`"reference"`) as `kernel=`.
- So under `"auto"` each piece makes its own check and logs it once:
  - the forest build: `resolve_tree_build` (CUDA, Warp, d in {2,3}, under 2^31 cells);
  - the walk: `resolve_walk` (CUDA, forest depth within the stack, under Warp's 2^31 elements);
  - the fused read and fused node means: CUDA, Triton, not float64.
- `"reference"` runs all three references.
- The model attributes `tree_walk` and `tree_build` and the `"fused"` value are gone. The module-level forcing
  arguments stay for tests: `point_pairs(walk="fast")`, `BoundaryForest.build(build="fast")` and
  `tree_read_pairs(kernel="fused")`.
- The class docstring's Execution paragraph is rewritten.
- `tree_read_checkpoint` was left as it was in this commit, then removed in f1ae05c75 (last section).
- Tests:
  - those that set `m.tree_walk`, `m.tree_build` or `m.tree_read_kernel = "fused"` now patch the resolver;
  - a new test, `test_tree_read_kernel_is_the_tree_reads_one_execution_attribute`, checks that `"auto"` and
    `"reference"` reach both the build and the walk, that the state-dict keys are unchanged, and that `"fused"` and
    `"fast"` are refused. It fails on cd26d1876 (the walk receives `"auto"` when the attribute is `"reference"`) and
    passes at the head.

## Checks

| check | result |
|---|---|
| test_isla_tree_read.py, CPU, locally | 69 passed, 29 skipped (CUDA-only) |
| whole ISLA test directory, LHR cpu-short 365676 (code_dttfast_integ_p1) | **358 passed, 75 skipped, 0 failed** |
| recipe tests, same job | 521 passed, 12 skipped, **2 failed**: `test_campaign_c_hooks.py::test_initialize_from_checkpoint_{loads_weights_only,rejects_architecture_mismatch}`, a `TypeError: int() argument must be ... not 'NoneType'` in `physicsnemo/distributed/manager.py:350/398`. 365890 ran both on 0.11.0 (1e7f03d7d) with the same recipe snapshot: same 2 failures, same message. They come from the base (likely the cpu-short node's launcher environment), not from this branch |
| CUDA tests on a GB300 (365884): test_isla_tree_read.py + test_isla_kernels.py + test_isla_compile.py | **146 passed**, 0 skipped |
| 0.11.0 state dict loads unchanged (local, small, two processes: `probes/sd_probe.py`) | an ISLA-DTT `.mdlus` and its state dict written by 1e7f03d7d (model.py md5 1febf1e3) load into the head (23e19365) through `Module.from_checkpoint` and through `load_state_dict(strict=True)`. Keys are the same (87 tensors), outputs bitwise equal (max abs diff 0.0), and `tree_read_kernel` loads as `"auto"` and is not in the state dict |

## Equivalence (365884; tables `out/g1_365884/equiv_table.md`, `out/g2_366177/spread_table.md`)

Inputs are PROFILE's first-batch dumps: DrivAerML vehicle (`drv_body`) and whole boundary (`drv_whole`), AeroZoo,
AirfRANS and the lug. They are raw model inputs (boundary meshes, prediction points, global data), so each version
computes its own geometry; nothing needed rebuilding, and geometry.py is the same file in both (md5 6ba44b1d).
- Each run is in its own process and prints the md5 of every tree-read file and of model.py, plus the resolver logs.
- The head under `"auto"` logged `tree_build` fast, walk fast, `tree_read_kernel` fused and geo_kernel fused.
- Each comparison is one training step at 2,000 points with the reference's parameters (seed 42, read projections
  N(0, 0.02)), in float64, float32 and bf16 autocast.
- The reference ran twice, the head under `"auto"` once, and the head with `tree_read_kernel="reference"` once.
- Distances are max abs difference over a tensor group divided by the reference's max abs over that group. The groups
  are the loss, the outputs, and the gradients of `blocks`, `kernel_basis`, `kernel_reads` and the other parameters.

| case | pairs (every precision, all 4 runs) | float64: head vs ref (ref vs ref) | float32: worst ratio | bf16: worst ratio |
|---|---|---|---|---|
| drv_body | identical (1,362,988 pairs at f64/f32, 1,360,155 under bf16) | <= 2.1e-14 (1.0e-14) | 1.003 | 1.000 (kernel_basis 0.85) |
| drv_whole | identical (1,648,429 / 1,648,645 / 1,621,815) | <= 3.4e-15 (4.3e-15) | 1.000 | 1.025 (loss) |
| AeroZoo | identical (557,046 / 557,184 / 465,050) | <= 2.7e-15 (3.7e-15) | 1.011 | 1.004 |
| AirfRANS | identical (42,706 / 42,706 / 41,172) | <= 1.7e-15 (2.0e-15) | 1.004 | 1.000 |
| lug | identical (243,195 / 242,767 / 240,981) | <= 3.3e-15 (3.7e-15) | 1.002 | 1.002 |

- **Ratio** = (head's distance to the reference's float64 run) / (the reference's own float32 or bf16 distance to it),
  taken over every group. The bound is 2; the worst is 1.025.
- **float64** differences are at roundoff, at most 2.1e-14 relative, against a bound of 1e-12. They are the same size
  as the reference's own run-to-run spread (CUDA atomics).
- Under bf16 the pair lists differ from the float32 lists in both versions alike, because the forest is built from
  bf16 coordinates. The two versions' lists are identical in every precision.
- **Forced reference on the head vs 0.11.0.** In g1, float32 and float64 sit at the reference's own spread. Under bf16
  some gradient groups differed in AeroZoo (blocks 4.3e-3) and AirfRANS (kernel_basis 1.6e-3), where g1's one pair of
  reference runs happened to be bitwise equal. Job 366177 repeated each 4 times:
  - AeroZoo: the reference's own spread is blocks 4.31e-3 and other params 6.28e-4, the same as forced reference vs
    reference;
  - AirfRANS: all 8 runs are bitwise equal.
  - So the forced reference equals 0.11.0 to the reference's run-to-run spread. g1's AirfRANS difference was one
    nondeterministic backward (bf16 atomics).

## Step time and peak (365884; bf16 autocast, `compile_blocks=True`, fused geometry kernel; median of 10 synchronized steps after 2 warm-up steps; reference and head on the same GB300)

| case | points | 0.11.0 step ms | head step ms | change | 0.11.0 peak GiB | head peak GiB |
|---|---|---|---|---|---|---|
| DrivAerML vehicle | 100k | 1320 | 937 | -29% | 127.8 | 112.4 |
| DrivAerML vehicle | 200k | 1883 | 1370 | -27% | 180.5 | 150.8 (-16%) |
| DrivAerML whole | 100k | 1462 | 1034 | -29% | 131.2 | 113.1 |
| DrivAerML whole | 200k | 2119 | 1537 | -27% | 187.3 | 152.1 (-19%) |
| AeroZoo | 100k | 910 | 615 | -32% | 115.0 | 109.1 |
| AeroZoo | 200k | 1225 | 881 | -28% | 156.6 | 145.6 (-7%) |
| AirfRANS | 100k | 608 | 495 | -19% | 38.1 | 37.7 |
| AirfRANS | 180,593 (all) | 811 | 692 | -15% | 68.1 | 67.4 |

The pair counts are the same in both versions at every size. Raw records are in `out/g1_365884/perf_*.jsonl`.

## Jobs and GPU-h

GPU: 365884 (28:41, 1.91 GPU-h), 366177 (3:09, 0.21 GPU-h) and 366384 (17:58, 1.20 GPU-h), **3.32 of the 4-GPU-h
cap**. CPU (cpu-short): 365676, 365890 and 366407. Details are in JOBS.md.

## `tree_read_checkpoint` removed (f1ae05c75)

- **Why.** With query chunking on, which is ISLA-DTT's only training path from pass 2, read checkpointing trains no
  more points on a GB300. Both forms reach 1.6M points on DrivAerML vehicle and whole boundary, and both stop at 3.2M
  on the same `torch.sort` INT_MAX limit, not a memory error (QCHUNK `RESULTS.md` section 7, item 3).
- **What it would still buy.** The peak at 400k vehicle is 123 GiB chunked against 26.5 GiB with `"block"`, at +13%
  step time. That matters only on GPUs under about 130 GiB, where the chunked step at 200k (121 to 126 GiB) would not
  fit.
- **The rule.** Options are scaffolds (the supervisor's 02:39 note), so the attribute goes.
- **Where the code is kept.** The feature stays on `origin/psharpe/isla-dttfast-ckpt` (5f0d01d98: d26643e26 adds it,
  5f0d01d98 binds `kernel_pairs` for the closure).
- **What was removed:**
  - the attribute and its sentence in the Execution docstring;
  - the `torch.utils.checkpoint` import and the `checkpoint` wrapping in the block loop;
  - the two tests and their helper, exactly the 100 test lines d26643e26 added;
  - the `kernel_pairs = None` binding. It was needed only for the `layer` closure (5f0d01d98: a closure cell broke
    the fullgraph compile), and the loop has no closure now.
- **The result.** The block loop and its surroundings in model.py are 0.11.0's again, line for line.
- **Checks:**
  - The tree-read file locally: 67 passed, 28 skipped.
  - The whole ISLA directory on LHR cpu-short (366407, code_dttfast_integ_p1b): 356 passed, 74 skipped, 0 failed
    (358 and 75 before, less the removed tests).
  - The recipe tests: 521 passed, with the same 2 `test_campaign_c_hooks` failures as on 0.11.0.
  - CUDA test files on a GB300 (366384): tree_read 95, kernels 34 and compile 14 passed. That is 143, against 146
    before less the 3 removed cases; the compile file covers the whole-model fullgraph compile that 5f0d01d98 had
    fixed.
- **Equivalence.** Not rerun, since the forward and backward are unchanged: the path with `"off"` was the only one
  the equivalence and timing jobs ran.

## Tests ready for a new reference forest (5fa4efb17)

- **Why.** DTT-POSE will change ISLA-DTT's reference forest, probably built with autocast disabled. The parent's
  note of 04:22 says to stop matching the current reference under bf16 autocast and to keep the fast-vs-reference
  tests parametrized over the reference.
- **Audit.** Every forest and walk equivalence test builds both sides through `BoundaryForest.build(build=...)` or
  `point_pairs(walk=...)`, or calls GLOBE's traversal directly (the opening-threshold test). None hard-codes
  reference outputs.
- **The change.** Two assertions pinned the current reference's bf16 boxes under autocast: one in
  `test_fast_walk_gives_the_reference_lists_in_the_reference_order` and one in `_check_fast_build`. They are
  dropped. `_forest_differences` already compares every field's dtype and bits with the reference's, so a
  reference built another way drops in.
- **Checks.** The edited tests pass locally (18 cases, 1 GiB cgroup cap). The change is test-only, so there was no
  LHR rerun.

## For pass 2

- QCHUNK's `tree_read.walk_pairs` collides with WALK's `tree_walk.walk_pairs`: `tree_read.py` already imports the
  latter by that name. Nothing to do in this pass.

## Notes

- The verification of fusedread (`../verify_fusedread/`) finished at 04:15 with "ok, would merge" and no blocking
  finding.
- At about 04:02 a local CPU dry run of the probe (11.7 GB) ran without the heavy-work lock. The supervisor flagged it,
  and every later probe ran on LHR.
- Ideas for DTT-KNOBS: none new from this pass.
