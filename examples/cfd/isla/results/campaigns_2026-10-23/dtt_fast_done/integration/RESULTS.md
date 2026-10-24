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

# Pass 2 (2026-10-24 05:43-07:45 EDT)

Branch `psharpe/isla-dtt-fast`, pushed, head **757a5a877**. ISLA 0.11.0 (`_version` 11), state dict unchanged.

## Merges and commits (`--no-ff --signoff`, hooks on)

| commit | what | conflicts and resolution |
|---|---|---|
| 4a6243021 | merge `origin/psharpe/isla` 7213c70c5 (exact geometry fast path, `geometry_kernel` / `resolve_geometry_kernel`, GEO-DEGEN) | model.py only, both sides adding at one place: the Execution docstring (its `geometry_kernel` sentence placed after the `geo_kernel` one, then ours) and two new resolvers (`resolve_tree_read_kernel`, then `resolve_geometry_kernel`). Version, `_version` and refusal texts are psharpe/isla's unchanged (`resolve_p2_isla.py`) |
| 77c18bfe5 | merge `origin/psharpe/isla-dttfast-qchunk` 7b2b968ae (`ISLA.query_chunks`, the recipe's chunked step and size rule) | tree_read.py (6 regions) and the test file; model.py merged textually. `tree_read_walk` takes pass 1's `walk=`, `build=`, `kernel=` and records `kernel` on `TreeReadWalk`, so the chunked path walks and builds fast and reads fused under `"auto"`, and runs every reference under `"reference"`; the range function builds the fused read's basis after each range's plan (`_chunked_basis_from_plan`) and its fused node means. qchunk's `tree_read.walk_pairs` is **`walk_range_pairs`**. P2.8: `ISLA._tree_read_walk` replaces the lambda switch. The CUDA float64 test's capture patches `walk_range_pairs` (`resolve_p2_qchunk_tree_read.py`) |
| 1190d01c8 | review P2.5, P2.6, P2.9 | DDP reduction over every `requires_grad` parameter, raising on a missing gradient; one startup line when `compile=true` (the chunked step is eager); the chunked branch uses `_run_epoch`'s shared tail (skips only `loss.backward()`), `spans` for the shadowed `ranges` |
| e1ad25e4a | review P2.7 (as 0cb28ba41 has it) | `chunked_forward_backward` refuses targets with `batch_dims != 1`; new test fails on 1190d01c8's train.py, passes here |
| 757a5a877 | the size rule refit (below) | `_ONE_CHUNK_GIB` and its comment; the test checks the prediction against held-out measured rows |

Not merged: `origin/psharpe/isla-dttfast-basis` (its fix commit 96d5a146d appeared at 06:31, the checks were submitted at
06:32; the parent's pass 3). Not done of the taste items: qchunk's `_CaseInputs` NamedTuple and the explicit keyword
signature of `tree_read_pairs` (4f0722591 has both).

## P2.2 (float32 master leaves): settled by QCHUNK, bf16 leaves kept

Parent's 06:27 note: QCHUNK measured it (jobs 366714, 366811; `../qchunk/RESULTS.md` section 9). Master leaves cost nothing
in time and +2.2 to +3.0 GiB of peak; the bf16 ratios did not change beyond run-to-run noise (the 1.4x on DrivAerML moves
between runs in float32 too: kernel_basis GPU noise). So the bf16 leaves stay. My e2 (367245) agrees on this branch:
master leaves leave every ratio where the bf16 leaves have it (table below).

## The size rule refit (757a5a877)

`fit_one_chunk.py` (QCHUNK's) on the one-chunk `hold` rows of `cost_p2.py` (QCHUNK's `cost_l.py`) on the integrated head,
compile_blocks, 200k tokens, GB300 (jobs 366927 AeroZoo, 367203 DrivAerML; `out/p2/refit_{bfloat16,float32}.txt`). Peak
reserved above the memory allocated at the decision point, least squares, no intercept, the largest point count of
each dataset held out; the margin is the largest held-out |residual|, rounded up.

| precision | rows (fit / held out) | GiB per M pairs | GiB per 100k points | fit max abs residual | held out (measured - predicted) | margin | before (QCHUNK) |
|---|---|---|---|---|---|---|---|
| bfloat16 | 10 / 3 (100k-600k) | 0.082 | 24.75 | 0.74 | AeroZoo 600k +0.23, vehicle 400k -0.82, whole 400k -0.82 | 0.9 | 0.324, 24.22, 10.9 |
| float32 | 9 / 3 (50k-250k) | 0.151 | 45.49 | 1.15 | AeroZoo 250k -0.37, vehicle 150k +4.35, whole 150k +5.41 | 5.5 | 0.647, 43.08, 4.5 |

The fused read's per-pair memory is a quarter of the reference read's (0.082 against 0.324 GiB per million pairs). With
the stored coefficients every one of the 25 measured rows' peak is within its prediction (smallest slack 0.10 GiB, the
whole boundary at 150k in float32). The float32 held-out jump at 150k on DrivAerML (+4.4/+5.4 GiB) is reserved-minus-
allocated fragmentation (the allocated fit's held out is <= 1.1 GiB), as QCHUNK found at 400k in bf16.

## Checks

| check | result |
|---|---|
| whole ISLA test directory, LHR cpu-short 367067 (code p2b = e1ad25e4a) | **398 passed, 122 skipped, 0 failed** |
| recipe tests, 367067 (e1ad25e4a) and 367409 (757a5a877) | 529 passed, 12 skipped, 2 failed: the pre-existing `test_campaign_c_hooks` pair, the same `TypeError: int() argument ... 'NoneType'` (manager.py:350/398) as on 0.11.0 in 365890 |
| CUDA tests on a GB300 (366928) | test_isla_tree_read.py 100 passed; test_isla_kernels.py + test_isla_compile.py 48 passed |
| locally | test_train_helpers.py 22 passed, 1 skipped (the two-rank test runs on LHR); the new P2.7 test fails on 1190d01c8; the tree-read file's query_chunks / fused-basis / attribute tests 8 passed (the two theta=0 chunk cases need over 1 GiB: run on LHR) |
| 0.11.0 ISLA-DTT checkpoint (local `probes/sd_probe.py`, two processes: 7213c70c5 saves, the head loads) | `from_checkpoint` and `load_state_dict(strict=True)`: 87 tensors, outputs bitwise (max abs diff 0.0) |
| smoke step, recipe's chunked step and rule at 200k, bf16, compile_blocks (366927, `out/p2/g1/smoke.jsonl`) | DrivAerML vehicle and whole boundary (relative_l2) and AeroZoo (case_relative_l2): one chunk, peak 148 / 153 / 130 GiB; AirfRANS (mse, 180,593 points): 2 chunks, 27 GiB; lug (mse, 9,048): 1 chunk, 5 GiB. Losses and gradients finite |

## Equivalence (366928; `out/p2/e1/table_<case>.md`, `equiv_table.md`)

Head e1ad25e4a (`"auto"`: fast build and walk, fused read) against `origin/psharpe/isla` 7213c70c5, each run in its own
process printing every model file's md5. PROFILE's five dumps at 2,000 points (targets from the dumps), the reference's
parameters. One training step per run in float64, float32 and bf16 autocast through the recipe: the reference's
unchunked `forward_pass` (twice under mse, once under relative_l2); the head's `chunked_forward_backward` in chunks of
700 (mse: one pass over 3 chunks; relative_l2: the two-pass form, capacity 0); the head unchunked with
`tree_read_kernel="reference"`; and the head chunked with `tree_read_kernel="reference"`. float64 has no recipe precision,
so its step runs the same algorithms with a float64 loss. Ratio = (head's distance to the reference's float64 run) /
(the reference's own distance), worst over loss, outputs and the gradient groups (blocks, kernel_reads, kernel_basis,
other).

| case | pairs (all runs, every precision) | float64 chunked vs ref (mse / rl2) | float32 ratio (mse / rl2) | bf16 ratio (mse / rl2) | forced reference vs ref: f64, f32, bf16 (ref vs ref2) |
|---|---|---|---|---|---|
| DrivAerML vehicle | identical (1,362,988 f64/f32; 1,360,155 bf16) | 7.1e-14 / 4.3e-14 | 1.02 / 1.02 | 1.09 / 1.26 | 8.2e-15, 1.5e-6, 5.6e-3 (6.8e-15, 1.5e-6, 5.0e-3) |
| DrivAerML whole | identical | 3.1e-13 / 6.5e-13 | 1.24 / 1.24 | **2.10 / 5.19** (loss) | 4.0e-15, 1.8e-6, 3.9e-7 (3.7e-15, 1.8e-6, 6.4e-5) |
| AeroZoo | identical | 3.2e-14 / 1.9e-14 | 1.01 / 1.02 | 1.26 / 1.26 | 2.7e-15, 1.2e-6, 2.5e-10 (2.0e-15, 7.0e-7, 1.4e-3) |
| AirfRANS | identical (42,706; 41,172 bf16) | 7.6e-14 / 3.6e-14 | 1.13 / 1.00 | **6.36 / 4.03** (grad blocks) | 1.6e-15, 8.3e-7, 2.1e-5 (1.8e-15, 7.4e-7, 0) |
| lug | identical | 7.4e-14 / 7.4e-14 | 1.00 / 1.01 | 1.03 / 1.11 | 4.6e-15, 1.9e-6, 1.2e-2 (6.4e-15, 1.7e-6, 4.6e-3) |

- **Pairs**: every range's (point, source) pairs in global point numbers, lexsorted, md5-equal to the reference's in every
  run and precision, node-pair counts equal; the head's chunked runs read `fused` under "auto" and `reference` under
  "reference".
- **float64**: at most 6.5e-13 relative (bound 1e-12); the chunked step reorders sums (the unchunked forced reference is at
  the reference's own 2e-15 to 8e-15).
- **float32**: worst ratio 1.24 (bound 2).
- **Forced reference** (head, `tree_read_kernel="reference"`, unchunked) against the reference: float64 and float32 at
  the size of the reference's own run-to-run difference (one sample each); bf16 within 2.5x of one sample (lug 1.2e-2 vs
  4.6e-3), as pass 1's four-run spread (g2 366177) showed for bf16 atomics.
- **The chunked step with the reference read** gives the same ratios as with the fused read (float32 <= 1.24; bf16 AirfRANS
  6.40, whole boundary 2.14): the bf16 excess below is not the fast pieces'.

### bf16: two cases over 2x, from `query_chunks` itself (e2 367245, a1 367366)

- AirfRANS, grad blocks: the group ratio is set by one tensor, `blocks.3.assign.weight` (the slice prototypes' map; its
  own max is 6.5e-2 of the group's). Its bf16 gradient is noise in the reference too: the reference's error is 5.96e-2 of
  the group max, about its whole size; the chunked step's is 3.79e-1. The excess is broader than that tensor, in the first
  blocks: `geo_feat` of blocks 0-2 at 16-20x the reference's bf16 error (2.8e-2 against 1.8e-3 of the group max),
  `broadcast` of blocks 0-2 and `blocks.0.mlp` at about 4x. In float32 all of these are at or below the reference's error
  (0.7x on `blocks.3.assign`). On the DrivAerML vehicle the same listing is at 1x except `blocks.4.assign` at 2.5x of a
  small value (`out/p2/a1/a1.log`).
- Whole boundary, the loss only (outputs 0.96, every gradient group <= 1.06): head 1.3e-3 from float64 against the
  reference's 2.5e-4 (relative_l2) / 6.2e-4 (mse), one scalar each.
- Reproducible (rerun 6.36 / 2.10, 5.19), the same with **one chunk of every point** (6.40; 2.10, 5.19) and with **float32
  master leaves** (6.36, 4.00; 2.10, 5.19), and with the reference read. So it is neither the chunk accumulation (P2.2)
  nor the fast pieces. The likely cause (not isolated further): `query_chunks` computes each leaf's gradient from the
  points and from the boundary side in separate backward passes, each rounded to bf16 before they are added, where the
  unchunked step sums both inside one matmul; on a gradient that cancels (as this one does) the extra rounding is large
  relative to the result. This is a moved bf16 rounding point in CHILD_RULES' strict sense, inherited from QCHUNK.
- CHILD_RULES' bound is for float32 (bf16 is reported beside the reference's own distance), and the parent's 04:22 note
  stops work on matching the current bf16 reference, so nothing was changed. Master leaves (P2.2) do not help here.

## Ideas for DTT-KNOBS

- The chunked step's bf16 gradient split: keeping the points' and the boundary side's contributions to each leaf in one
  float32 sum would need the points' consumers of the leaves (slice logits, readback, read sources) to produce float32
  gradients, which moves forward rounding points. Out of scope here.

## Jobs and GPU-h

366927, 366928, 367203, 367245 on GPU: **3.90 of the 5 GPU-h cap**; 367067, 367366, 367409 on cpu-short. JOBS.md "Pass 2".
Raw equivalence `.pt` files (55 GB) deleted after the tables; tables, logs, jsonl and the parameter states kept.

# Pass 3 (2026-10-24 07:48-09:40 EDT)

Branch `psharpe/isla-dtt-fast`, pushed, head **07ce32eb1**. ISLA 0.11.0 (`_version` 11), state dict unchanged.

## Commits (`--no-ff --signoff` merges, hooks on)

| commit | what |
|---|---|
| 011306ec3 | merge `origin/psharpe/isla` 1b25eb899 (ALIGN8: `TOKEN_ALIGNMENT = 32`, `_pad_tokens`). Three model.py conflicts, all ALIGN8's edits of the old one-piece forward against qchunk's split into `_boundary_tokens` / `_query_tokens` / `_forward_tokens`: HEAD's helpers kept, ALIGN8's three edits re-applied in `_forward_tokens` (pad after the concatenation; the kernel reads update `n_boundary:n_real`; the head reads `n_boundary:n_real`) (`resolve_p3_align8.py`) |
| cd4c958bd | merge `origin/psharpe/isla-dttfast-basis` 96d5a146d. tree_read.py: the fallback-log set, and `tree_read_walk`'s docstring (HEAD's + the basis clause); the fused-basis hook itself auto-merged into `walk_range_pairs`, which builds the basis of the unchunked read and of every query chunk. Test file: both sides' appended tests kept (`resolve_p3_basis.py`) |
| ea63ebc62 | tests: the CUDA fused-basis test also builds one range of points through `walk_range_pairs` (as `query_chunks` does) and checks the fused kernel ran; `test_query_chunks_equal_the_forward_pass_and_its_gradients` asserts its token count is not a multiple of 32 (so the forward pads and the chunks do not) |
| 90805de66 | part P: under `compile_blocks` the read's LayerNorms and q/k/v/out projections run through `torch.compile(dynamic=True)` with `{mix_order_reduction: False}` (Inductor's default casts), in the forward and in `query_chunks`; CPU float64 test compiled vs eager |
| 023c0f3d6 | **revert of P** (certification, below); the tree equals ea63ebc62 |
| 07ce32eb1 | **fix**: `fused_basis._geometry_args` detaches the geometry; CUDA test `test_fused_basis_leaves_the_geometrys_graph_alone` |

### ALIGN8 against ISLA-DTT's paths (checked in code and by test)

- Unchunked forward: the passive point tokens are concatenated, then padded; the tree read's pairs are built from the
  unpadded boundary and point tokens before padding, so no padded token is a boundary cell of the forest, a point of
  the walk, or read by the fused read or `node_means`; the kernel reads update points `n_boundary:n_real`; the head and
  the loss see `n_boundary:n_real` only. Padding tokens have log-weight -inf (no mass in any softmax).
- `query_chunks`: `_QueryChunks` calls `_boundary_tokens` / `_query_tokens` directly, never `_forward_tokens`, so neither
  the boundary side nor any chunk is padded (the chunks' token axes are the chunk sizes: the recipe's 100k chunks are
  aligned, a last chunk and the boundary side generally not, so `query_chunks` does not get ALIGN8's speedup; noted, not
  changed). The CPU test `test_query_chunks_equal_the_forward_pass_and_its_gradients` (theta 0 and 1, plain and boundary
  vectors) compares a padded forward with unpadded chunks at 420 tokens (not a multiple of 32): outputs and every gradient
  to float64 roundoff; it passes (theta 1 locally, all four in the cpu-short suites).

## The bug found and fixed (07ce32eb1)

The e1 equivalence on AeroZoo (the only dump with rank-1 boundary inputs) and j1's one-chunk timing on AeroZoo failed in
float32 and bf16 with `RuntimeError: Trying to backward through the graph a second time` in the first chunk's
backward. Cause: the fused basis (`_FusedBasis.apply`) took the walk's geometry as inputs, and `src_vecs` is the
learned rank-1 map of the boundary inputs, so the autograd Function's node had an edge into the boundary side's graph:
the first chunk's backward ran into it and freed it, the second failed. (The reference basis computes the invariants
under `no_grad`; the Function's backward returns `None` for the geometry, so the edge carried nothing.) Unchunked steps
were unaffected except for a wasted traversal. Fix: the geometry enters detached. Evidence:
- `lhr/p3/fix_check.py` (two ranges' backwards through the fused basis, `src_vecs = exp(leaf)`): FAILS on ea63ebc62
  with that error, PASSES on 07ce32eb1 (e2 367923); the same mechanism reproduced on CPU with a stand-in Function.
- New CUDA test (same body) passes on the head (tree_read file 109 passed on CUDA, e2).
- AeroZoo equivalence on the head (below) now runs, with identical pairs and ratios at 1.

## Part P (COMPILE's read projections): left out

**Timing** (j1 367616; `time_p.py`; A = ea63ebc62 no P, B = 90805de66 P; separate processes A1 B1 .. A4 B4 on one GPU per
case; bf16, compile_blocks, 200k points, median of 20 steps after 3; `out/p3/p_tables.md`):

| case | step | A medians ms | B medians ms | paired B - A (mean +- sd, range) | % | peak GiB A / B |
|---|---|---|---|---|---|---|
| DrivAerML vehicle | unchunked | 1198, 1198, 1200, 1198 | 1169, 1171, 1169, 1169 | -29.2 +- 1.7 (-32 to -27) | -2.4 | 150.6 / 143.6 |
| DrivAerML vehicle | one-chunk `query_chunks` | 1115, 1114, 1116, 1116 | 1087, 1087, 1084, 1088 | -28.7 +- 2.2 (-32 to -27) | -2.6 | 134.1 / 125.1 |
| AeroZoo | unchunked | 780, 780, 779, 784 | 751, 747, 751, 750 | -31.1 +- 3.0 (-34 to -28) | -4.0 | 145.6 / 138.7 |
| AeroZoo | one-chunk | failed in both A and B (the bug above) | | | | |

Every pair favours P, by 10-17x the pairs' sd: the gain is outside the spread.

**Certification** (the supervisor's compiled-region policy; `cert_p.py` / `cert_p2.py`; COMPILE's checkpoints AeroZoo
`azc_full_dtt_seed{42,43}` epoch 250 at 200k points and AirfRANS `r090eag_af_dtt_seed{42,43}` epoch 500 at every point;
each of the 3 regions x 12 blocks on the inputs eager captured, float32 and bf16 against float64 eager on the same
inputs, forward output and the gradient of every input; 1440 tensor groups):
- float64 compiled vs eager: <= 1.8e-15 everywhere.
- j1 (allowance = eager's change under one random one-ulp draw of the activation and output gradient): **1438/1440**;
  az42 block 11 `_source_states` float32 `grad_ln_w` (excess 1.08 of the allowance) and `grad_w_value` fail.
- e2 (allowance = the mean over 8 such draws, decided before the rerun, STATUS 08:47): **1439/1440**; those two pass, and
  az42 block 3 `_query_states` float32 `grad_ln_w` fails exactly at the bound (d_comp 5.4247e-7 against d_eager
  1.9703e-7 + allowance 3.4544e-7; excess 1.00). Every bf16 group passes in both (`out/p3/p_tables.md`,
  `out/p3/p_cert2.md`).
- So P does not pass as registered in either measurement, by 8% and then by rounding, both on float32 LayerNorm-weight
  (and one value-weight) gradients of one checkpoint. By the brief P stays out (023c0f3d6). **Decision for the parent**:
  read the policy's bound as met (the failures are at the size of the allowance's own draw-to-draw spread) and re-apply
  P (`git revert 023c0f3d6`, which restores 90805de66's test too), or keep it out.

## Checks

| check | result |
|---|---|
| ISLA test directory, cpu-short 367618 (90805de66, P) | 405 passed, 129 skipped, 0 failed |
| ISLA test directory, cpu-short 368116 (07ce32eb1, the head) | **404 passed, 130 skipped, 0 failed** (P's CPU test gone, the new CUDA test skipped); recipe 529 passed + the same 2 pre-existing failures |
| recipe tests, 367618 | 529 passed, 12 skipped, 2 failed: the pre-existing `test_campaign_c_hooks` pair (same TypeError as 365890) |
| CUDA tree_read file | 109 passed on 90805de66 (e1) and on 07ce32eb1 (e2) |
| CUDA kernels file, 07ce32eb1 (e2) | 29 passed, 5 failed: `test_compile_blocks_float32_matches_eager_at_every_token_count[12001-12003]` and `test_compile_blocks_query_independent_float32_at_every_token_count[6001, 6002]`, `assert err > 5e-6` with err 1.8e-7 to 2.7e-7. On the reference 1b25eb899 the same 5 fail with the same assertion (1.9e-7 to 2.8e-7): **pre-existing from ALIGN8** (the padded token count is a multiple of 32, so the forced-on mix-order reduction no longer shows pytorch#195845; the tests' guard needs an unaligned count) |
| CUDA compile file, 07ce32eb1 (e2) | 12 passed; then the CPU-only warm-cache test exceeded the job's timeout on the GPU node (it passes in cpu-short); the max-autotune CUDA test passed on the head and on 1b25eb899 (e3 368114) |
| test_isla_token_padding.py on a GB300 node, head (e3) | 5 passed (CPU tests; also in both cpu-short suites) |
| 0.11.0 checkpoint load (local `sd_probe.py`, 1b25eb899 saves, head loads) | strict, 87 tensors, outputs bitwise (max abs 0.0) |

## Equivalence (e1 367617 for 4 dumps on 90805de66; e2 367923 for AeroZoo on 07ce32eb1; reference 1b25eb899)

P is inactive here (eager blocks) and the fix does not change values, so e1's four cases stand for the head. Tables
`out/p3/e1/equiv_table.md`, `out/p3/e2/table_aerozoo.md`; method as pass 2.

| case | pairs | float64 chunked vs ref (mse / rl2) | float32 ratio (mse / rl2) | bf16 ratio (mse / rl2) | forced reference vs ref: f64, f32, bf16 (ref2 vs ref) |
|---|---|---|---|---|---|
| DrivAerML vehicle | identical | 7.1e-14 / 4.3e-14 | 1.07 / 1.02 | 1.09 / 1.26 | 9.0e-15, 1.8e-6, 5.4e-3 (7.6e-15, 2.2e-6, 5.7e-3) |
| DrivAerML whole | identical | 3.1e-13 / 6.5e-13 | 1.14 / 1.14 | 2.10 / 5.20 (loss) | 4.7e-15, 1.7e-6, 3.2e-6 (3.8e-15, 1.7e-6, 0) |
| AeroZoo | identical | 3.2e-14 / 1.9e-14 | 1.02 / 0.98 | 1.26 / 1.26 | 4.8e-15, 9.4e-7, 5.7e-3 (3.5e-15, 1.8e-6, 5.0e-10) |
| AirfRANS | identical | 1.5e-13 / 9.1e-14 | 1.09 / 1.03 | 6.38 / 4.01 (grad blocks) | 1.8e-15, 8.0e-7, 0 (1.7e-15, 8.0e-7, 0) |
| lug | identical | 6.6e-14 / 6.6e-14 | 1.00 / 1.01 | 1.15 / 1.15 | 5.0e-15, 2.0e-6, 4.6e-3 (4.0e-15, 1.5e-6, 1.5e-2) |

- float64 <= 6.5e-13 (bound 1e-12); float32 worst 1.14 (bound 2); the chunked step with the reference read gives the same
  ratios (AirfRANS bf16 6.42, whole bf16 2.14).
- bf16, as found (not worked on; `../bf16diag/`): the same two excesses as pass 2 (AirfRANS grad blocks 6.4 / 4.0, whole
  boundary loss 2.1 / 5.2), unchanged by ALIGN8 and the fused basis; the other cases <= 1.26.

## Ideas for DTT-KNOBS

None new. (Padding the query chunks to `TOKEN_ALIGNMENT` would be an execution change, not a knob: noted for a later
pass, not done.)

## Jobs and GPU-h

367616 (0.92), 367617 (1.93), 367923 (1.38), 368114 (0.33), then the addendum's 368336 (0.17) and 368482 (0.17): **4.90 of the 5 GPU-h cap**; cpu-short 367618, 368116, 368338, 368337, 368483. The AirfRANS whole-boundary check below (368565, 368662) used 0.83 of the granted extra GPU-h: **5.73 of 6**. JOBS.md "Pass 3". Raw equivalence `.pt` files on LHR deleted after the tables.

## Addendum: the boundary side of query_chunks padded (parent's note of 09:40; 5e82a9b98)

BF16DIAG (`../bf16diag/REPORT.md`) traced pass 2's AirfRANS bf16 excess to `_QueryChunks` running the boundary side on
the boundary tokens alone (1,017 on AirfRANS): at that odd count cuBLASLt computes the slice anchors with split-K
partials rounded to bf16. **5e82a9b98** applies its remedy's `__init__` hunk on top of the 1b25eb899 merge (no second
copy of `_pad_tokens` / `TOKEN_ALIGNMENT`): the boundary side is padded with `_pad_tokens(tokens, TOKEN_ALIGNMENT)` after
the walk, and the kernel reads' sources take the real cells `h[:, :n_cells]` only. The section above that says
`query_chunks` runs unpadded is superseded for the boundary side; the points' chunks stay unpadded.

- New CPU test `test_query_chunks_pad_the_boundary_side_to_the_token_alignment`: the first block's token axis is 320 for
  300 cells; on 07ce32eb1's model.py it fails (`[300] == [320]`). `test_query_chunks_equal_the_forward_pass_and_its_gradients`
  now also asserts the boundary count is unaligned (300) and still matches the padded forward to float64 roundoff.
- ISLA test directory on 5e82a9b98, cpu-short 368338: **405 passed, 130 skipped, 0 failed**; recipe 529 passed + the 2
  known `test_campaign_c_hooks` failures.

### bf16 equivalence with the one-ulp yardstick (f1 368336, f2 368482; tables `out/p3/f1/compare_*.md`)

BF16DIAG's probe `diag3_probe.py` unchanged (copied to `lhr/p3/f/`): every float32 parameter moved one ulp (signs from
the sample k and the name, identical in both versions), 2,000 points. Reference 1b25eb899 unchunked (its family: the
yardstick); head 5e82a9b98 chunked (700) under `"auto"` (fast build and walk, fused read and basis; pairs md5-equal to
the reference's in every run). Ratio = distance to the truth over the reference's unperturbed bf16 distance; truth the
reference's float64 step (AirfRANS) or float32 step (DrivAerML, AeroZoo: f1 put both versions on one GPU and the
reference's float64 step left the head no memory, so f2 reran the head families and float32 truths). min / median / max
over each family:

| case | family | loss | outputs | blocks | kernel_reads | kernel_basis | other |
|---|---|---|---|---|---|---|---|
| AirfRANS mse (8) | ref | 0.94 / 1.19 / 1.36 | 0.87 / 0.96 / 1.02 | 0.90 / 1.03 / 1.05 | 1.00 / 1.08 / 1.16 | 1.00 / 1.35 / 1.70 | 0.95 / 0.98 / 1.00 |
| | head | 0.94 / 1.10 / 1.31 | 0.93 / 1.01 / 1.13 | **0.74 / 0.98 / 1.01** | 1.00 / 1.00 / 1.10 | 1.00 / 1.25 / 1.44 | 0.93 / 0.98 / 1.00 |
| AirfRANS relative_l2 (4) | ref | 0.74 / 1.06 / 1.20 | 0.87 / 0.96 / 1.00 | 0.98 / 1.01 / 1.02 | 1.00 / 1.00 / 1.00 | 1.00 / 1.01 / 1.02 | 0.99 / 1.00 / 1.01 |
| | head | 0.97 / 1.01 / 1.10 | 0.93 / 1.01 / 1.01 | 0.82 / 0.99 / 1.00 | 0.89 / 0.89 / 1.04 | 0.91 / 1.04 / 1.14 | 1.00 / 1.01 / 1.03 |
| DrivAerML whole mse (7) | ref | 0.47 / 2.10 / 6.70 | 1.00 / 1.05 / 1.25 | 0.87 / 1.18 / 3.38 | 0.82 / 0.86 / 1.22 | 1.00 / 1.00 / 1.34 | 0.77 / 1.00 / 1.39 |
| | head | 1.49 / 2.46 / 6.78 | 0.96 / 1.05 / 1.24 | 0.81 / 0.99 / 1.89 | 0.72 / 0.79 / 1.27 | 0.83 / 1.00 / 1.33 | 0.65 / 0.84 / 1.54 |
| DrivAerML whole relative_l2 (7) | ref | 1.00 / 1.97 / 15.78 | 1.00 / 1.05 / 1.25 | 0.85 / 1.18 / 2.07 | 0.53 / 0.85 / 1.00 | 0.71 / 1.00 / 1.56 | 0.38 / 0.80 / 1.00 |
| | head | 4.25 / 5.27 / 17.76 | 0.96 / 1.05 / 1.24 | 0.72 / 0.99 / 1.50 | 0.53 / 0.79 / 0.88 | 0.75 / 0.84 / 2.00 | 0.68 / 0.88 / 0.93 |
| AeroZoo mse (7) | ref | 0.82 / 0.88 / 1.00 | 0.96 / 1.01 / 1.12 | 0.93 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.97 / 0.98 / 1.00 |
| | head | 0.79 / 0.90 / 1.00 | 0.91 / 0.99 / 1.26 | 0.92 / 0.99 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.01 / 1.01 | 0.96 / 0.98 / 1.00 |

Reading:
- **AirfRANS: fixed.** The head's blocks ratios are 0.74-1.01 (pass 2: 6.4 under mse, 4.0 under relative_l2); every
  group's head family lies within or below the reference's. 1 of 427 gradient tensors has every head run above every
  reference run, 22 the other way.
- **AeroZoo**: the families coincide (outputs max 1.26 against 1.12, one sample; medians equal).
- **DrivAerML whole boundary** (boundary already aligned, so the padding changes nothing here): every gradient group's
  median is at or below the reference's; three maxima of 7 sit above the reference's maximum of 7 (mse kernel_reads 1.27
  vs 1.22 and other 1.54 vs 1.39; relative_l2 kernel_basis 2.00 vs 1.56), and no gradient tensor has every head run
  above every reference run (0 of 431 under both losses). The scalar loss under relative_l2 sits higher (median 5.27
  vs 1.97, inside the reference's 1.00-15.78 range); by the compiled-region policy a scalar loss is reported, not gated,
  and BF16DIAG found this loss within the reference's one-ulp spread at 757a5a877 too.
- So by the diagnosis's yardstick the pass-3 head's bf16 step is equivalent to the reference on all three cases; the
  remaining excess of pass 2 was AirfRANS's, and it is gone.

## Addendum 2: FINAL-TABLE's AirfRANS whole-boundary 100k loss outlier (parent's note of 10:27; jobs 368565, 368662)

FINAL-TABLE (job 368334) found the 18-step mean loss of 5e82a9b98 at +3.4e-3 relative on AirfRANS whole boundary at
100k points (huber; the chunked step in one chunk of 100,000), where 07ce32eb1 was -1.2e-5. Its logs show the curves
equal at step 1 and parting from step 4. This case's boundary side is 1,017 + 1,727 = 2,744 cells: a multiple of 8, not
of 32, so 5e82a9b98 pads it to 2,752. Versions: reference 1b25eb899 (unchunked), 07ce32eb1, head 5e82a9b98; snapshots
ref1b25, p3b, p3c; probes `lhr/p3/g/` (BF16DIAG's `diag3_probe.py` with `DIAG_COMPILE=1`, and FINAL-TABLE's `step.py`
with `--seed-base` and `--ulp`, both copies); tables `out/p3/g1/compare_*.md`, curves `out/p3/g1/curves/*.json`.

**18-step curves** (the recipe's `_run_epoch`, compile_blocks, bf16, seed-42 init as FINAL-TABLE; mean loss of the 18
steps; seed bases 1000 and 2000, one-ulp perturbed starts 1 and 2 at base 1000):

| version | base 1000 | base 2000 | ulp 1 | ulp 2 |
|---|---|---|---|---|
| reference 1b25eb899 | 0.250671 | 0.250603 | 0.250586 | **0.252097** |
| 07ce32eb1 | 0.250767 | 0.250554 | 0.250519 | 0.250735 |
| head 5e82a9b98 | 0.250647 | 0.250600 | 0.250597 | 0.250622 |

The reference's own family spans 0.250586-0.252097 (+6.0e-3: its ulp-2 run jumps at step 10, 0.2211 against 0.212-0.213);
FINAL-TABLE's head run (0.251394, +3.4e-3 against its reference run) lies inside it, and the head's four runs here sit at
the family's centre. **The outlier is trajectory noise.** (Every boundary cell is kept on AirfRANS, so the two seed bases
change the curves only at run-to-run level.)

**One training step by the one-ulp family** (huber, 100k, compile_blocks; 7 samples each; truth the head's float64
chunked step; ratio min / median / max, blocks gradients and the other parameters' gradients):

| family | blocks | other | gradient tensors above the reference in every run |
|---|---|---|---|
| reference 1b25eb899 | 0.53 / 0.87 / 1.44 | 0.93 / 1.06 / 1.22 | - |
| head 5e82a9b98 | 0.77 / **1.32 / 2.90** | 0.99 / 1.24 / 1.58 | 2 of 428 |
| 07ce32eb1 | 0.43 / 0.72 / 0.96 | 0.91 / 1.00 / 1.18 | 0 |
| head, reduced-precision reduction off (368662) | 0.50 / 0.90 / 1.53 | 0.91 / 1.08 / 1.11 | 0 |
| 07ce32eb1, reduced-precision reduction off | 0.58 / 1.05 / 1.92 | 0.90 / 1.07 / 1.23 | 0 |

- At this shape the padding gives the head a tail: 3 of 7 samples at 2.2-2.9x, above the reference's maximum, in
  blocks 0-2's routing and geometry weights (where BF16DIAG's AirfRANS tail sat). 07ce32eb1, unpadded at 2,744 (a
  multiple of 8), has none.
- Turning `allow_bf16_reduced_precision_reduction` off removes it (median 0.90, no tensor above in every run), so it is
  BF16DIAG's mechanism (bf16 split-K partials in the slice anchors) at the padded count 2,752: the open risk BF16DIAG
  named, that padding only steers cuBLAS's shape heuristics.
- It is about half the size of the tail the padding fixed (1,017 cells: up to 6.8x), and its effect on 18 training steps
  is inside the reference's own spread. The scalar loss column (reference family 1.00-61.75, since the unperturbed
  reference's loss error is 1.3e-5 by chance) is reported in the tables, not gated.
- Not changed in code. Choices for the parent: keep the padding (better at 1,017, slightly worse at 2,752, noise in
  training); or pad to another multiple; or set `allow_bf16_reduced_precision_reduction = False` for the recipe (the
  only cure for every shape; program-wide, changes the reference's bf16 numerics at some shapes, needs a timed
  benchmark and the supervisor).

# Pass 4 (2026-10-24 13:15-15:15 EDT)

Head `psharpe/isla-dtt-fast` at **9967490fe** (pushed), from 5e82a9b98. Reference for equivalence: `origin/psharpe/isla`
at eba038f08. Launchers and probes in `lhr/p4/` (mirrored at `$T/kernel/dttfast_integ/p4/`); copied outputs in
`out/p4/` (the `.pt` files were deleted after the tables).

## Merges and commits (`--no-ff --signoff`, hooks on)

| commit | what |
|---|---|
| 02be03412 | merge `origin/psharpe/isla` eba038f08. One conflict in model.py: eba038f08 makes the surface task's `prediction_boundaries` required and adds `out_rows` (the head reads only the prediction boundaries' cells). QCHUNK had split `forward` into `_case_inputs` and `_forward_tokens`, so `_case_inputs` now takes `prediction_boundaries` and returns `out_rows`, `forward` passes them to `_forward_tokens`, and `_QueryChunks` ignores them (interior only). Script `resolve_p4_isla.py`. KERNEL-TESTS' fix of the 5 mix-order tests comes with it. No `query_independent` default change is in eba038f08. |
| d41641559 | merge readbwd e8848168c (clean) |
| d9f0e0cd3 | merge basisbwd e9fc0d9a0 (clean) |
| 14367b61f | merge plan 16d813ed0 (clean) |
| 6cfa550f4 | per-point pair counts (+25/-24 lines in 6 files): `tree_walk.walk_pairs(count_only=True)` returns each point's count `(n_near + n_far).view(n_q, n_t).sum(1)` with no host read; `BoundaryForest.frame_pairs(count_only=True)` returns the reference's `bincount(qi)`; `walk_pair_count` becomes `walk_pair_counts` `(Q,)`; `_QueryChunks.n_pairs` becomes `pair_counts`. PLAN's range-walk test now checks the counts against `bincount(qi)`. |
| 26450bb51 | merge rule 746d788b1. Two conflicts in the recipe's `train.py` and its test, both resolved to RULE's side. Then RULE's pair counts, which ran `searchsorted` on the all-points lists, come from `chunks.pair_counts`: one cumulative sum `cum` gives the total (`cum[-1]`) and each span's exact pairs (`cum[bounds].diff()`). Script `resolve_p4_rule.py`. |
| 9967490fe | the size rule refit (below) and its held-out test rows |

**How PLAN and RULE were reconciled.** PLAN walks each chunk's points only when the chunk runs, so at the decision point
no all-points list exists. The walk's count pass, which PLAN already ran for `chunks.n_pairs`, writes each
(point, tree) count at `q * n_t + t`. Summing over trees gives each point's count, and a cumulative sum gives RULE's
exact per-span pairs for any spans. So the decision still comes after the boundary side and the count pass, before any
point-side forward, with no list built. RULE's all-reduce (the minimum of the fitting size and of `-n` over the ranks)
is unchanged, so every rank makes the same one-or-two-pass choice. The per-span check that follows stays rank-local, as
RULE has it. On CPU, both of RULE's DDP tests and all its chunk tests pass at 26450bb51 (10 passed).

## The size rule refit (9967490fe)

Rows come from RULE's `cost_rule.py`, ported to the head's API (`lhr/p4/mk_probes.py`), on GB300 in job a1 370118. Each
row is the peak reserved memory minus the memory allocated at the decision. Arms: one chunk of every point, and chunks
of a half to a quarter, in one pass (huber) and in two (relative_l2, case_relative_l2). Datasets: DrivAerML vehicle
alone (`isla_volume_vehicle` = 5e82a9b98's `isla_volume.yaml`), DrivAerML whole boundary (eba038f08's `isla_volume`),
and AeroZoo. The largest point count of each dataset is held out. The fit is RULE's `fit_rule.py`. Fits:
`out/p4/fit_*.txt`; every row against the chosen coefficients: `out/p4/check_*.txt`.

| precision | fit | GiB per M pairs | GiB per 100k points | margin | rows (fitted / held out) | smallest slack over all rows |
|---|---|---|---|---|---|---|
| bfloat16 | with intercept (4.05) | 0.136 | 23.25 | 7.6 = 4.05 + largest held-out residual 3.47 | 20 / 7 | +0.09 GiB |
| float32 | no intercept | 0.231 | 44.97 | 4.0 (largest residual of any row, 3.93) | 10 / 6 | +0.06 GiB |

The old coefficients were bfloat16 (0.0753, 24.40, 6.2) and float32 (0.160, 44.02, 6.6). With them, the new held-out
float32 rows were under-predicted by 7-9 GiB: DrivAerML vehicle 200k predicted 227.0 GiB against 235.7 measured,
whole boundary 150k 203.0 against 210.5. In float32 the fit with an intercept held out worse (margin 10.6 GiB against
3.74 without), so the no-intercept fit was kept. Every measured peak lies within its prediction. The recipe test's
held-out rows are the new measurements.

## Checks

| check | result |
|---|---|
| whole ISLA test directory on a GB300 with every CUDA test (a1 370118 GPU 0, head 26450bb51) | **608 passed, 1 skipped, 0 failed** (29:43) |
| whole ISLA test directory on CPU, cpu-short (370119 on 26450bb51; 370461 on 9967490fe) | 438 passed, 163 skipped, 0 failed (both) |
| recipe tests, cpu-short (same jobs) | 565 passed, 12 skipped, 2 failed: the known `test_campaign_c_hooks` pair (`int(None)` TypeError in `physicsnemo/distributed/manager.py`, the same message as on 0.11.0 since pass 1) |
| 0.11.0 checkpoint (local `probes/sd_probe.py`; eba038f08 saves, 9967490fe's model loads; model.py is unchanged since 26450bb51) | `from_checkpoint` and `load_state_dict(strict=True)`: 87 tensors, outputs bitwise equal (max abs 0.0) |
| local CPU | query_chunks and range-walk tests (8 passed), surface-conditions file (7 passed), recipe chunk tests (10 passed), chunk-rule test after the refit (2 passed) |

## Equivalence against eba038f08 (a1 370118; tables `out/p4/a1/table_<case>.md`, `equiv_table.md`)

Each version ran in its own process and printed its md5s (eq_probe.py). Inputs: 2,000 points, chunks of 700 (forced
through `train._chunk_spans`, after the recipe's own decision was printed), and pass 2's parameters.

| case | per-chunk lists = walk of every point (raw) | pairs | float64 max rel (chunked mse / rl2) | float32 ratio max (mse / rl2) | bf16 single-run ratio (mse / rl2) | forced reference, chunked: f32 ratio / f64 |
|---|---|---|---|---|---|---|
| DrivAerML vehicle | True | identical | 7.1e-14 / 4.3e-14 | 1.01 / 1.01 | 1.09 / 1.26 | 1.01 / 7.1e-14 |
| DrivAerML whole boundary | True | identical | 3.1e-13 / 6.5e-13 | 1.14 / 1.14 | 2.10 / 5.20 | 1.14 / 3.1e-13 |
| AeroZoo | True | identical | 3.2e-14 / 1.9e-14 | 1.02 / 0.99 | 1.26 / 1.26 | 1.02 / 3.2e-14 |
| AirfRANS | True | identical | 1.9e-13 / 1.2e-13 | 1.00 / 1.22 | 1.31 / 1.10 | 1.00 / 1.9e-13 |
| lug | True | identical | 6.0e-14 / 6.0e-14 | 1.00 / 1.00 | 1.00 / 1.21 | 1.00 / 6.0e-14 |

How to read the table:
- **Pair lists.** "raw" means the per-chunk lists, concatenated in range order with each range's start added back,
  are `torch.equal` to `forest.frame_pairs` over every point (qi, si and node counts). "identical" means the lexsorted
  global pairs and node counts equal the reference's.
- **float64 and float32.** Both meet CHILD_RULES: float64 within 1e-12, float32 within 2x.
- **bf16 single-run ratios.** These are not a yardstick. The DrivAerML whole-boundary 2.10 and 5.20 are the same values
  as passes 2 and 3, and the one-ulp families below judge them.
- **`tree_read_kernel="reference"`, unchunked, against the reference's run-to-run spread** ("forced-ref" against
  "ref2" in the tables).
  - float64 and float32: inside the spread on every case. For example, float32 on DrivAerML vehicle gives
    1.57e-6 against 1.55e-6.
  - bf16: inside the spread on four cases. On AeroZoo it is 5.7e-3 against one ref2 sample of 1.4e-3. Pass 1 measured
    the reference's 4-run spread there at 4.31e-3.

### bf16 one-ulp families (a1 GPUs 1-2 and b1 GPUs 2-3; tables `out/p4/a1/compare_*.md`, `out/p4/b1/compare_*.md`)

`diag_probe.py` ran with `compile_blocks`. The reference ran unchunked and the head chunked (700 points, mse, auto
kernels). Each version ran in 2 cold processes with private Inductor, Triton and Warp caches, 9 samples per family
(ulp 0-8), and a float64 truth.

| case | group: reference family min-max | head family min-max | gradient tensors above in every run / below / total |
|---|---|---|---|
| lug | blocks 0.48-1.00; other 1.00-1.11 | blocks 0.49-**1.48** (one run of 9; median 0.52 vs 0.51); other 1.00-**1.12** | 2 / 4 / 426 |
| AirfRANS | blocks 0.82-3.44; loss 0.68-1.02 | blocks 0.71-**3.93** (one run; median 0.99 vs 1.00); loss 0.71-1.23 (not gated) | 1 / 5 / 427 |
| AeroZoo | every group | inside, every group | 1 / 19 / 432 |
| DrivAerML vehicle | every group | inside, every group | 0 / 12 / 427 |
| DrivAerML whole boundary | blocks 1.00-6.25; loss 0.42-1.48 | blocks 1.26-4.41; loss 0.40-2.09 (not gated); other groups inside | 0 / 0 / 431 |

Outputs, kernel_reads and kernel_basis are inside, or at or below the reference's range, on every case. Three group
maxima lie above the reference family's maximum: lug blocks, lug other (1.12 against 1.11), and AirfRANS blocks. Each
comes from a single run of 9, and the family medians are at or below the reference's. At most 2 of about 430 tensors
sit above the reference in every run. This is the pattern pass 3 accepted for DrivAerML whole ("3 maxima of 7 slightly
above the ref's max, 0 of 431 tensors systematically above"). Read strictly, the rule ("inside that family's range")
is exceeded by those three maxima. **The parent judges.** I recommend accepting them on the same grounds as pass 3.

## Step time, A/B (b1 370460: 5e82a9b98 = A against 9967490fe = B; 3 alternations, A B / B A / A B, median of 15, bf16, compile_blocks, FINAL-TABLE's `step.py`)

| case | A ms | B ms | B - A per alternation | B / A | peak allocated GiB A / B | B's chunks |
|---|---|---|---|---|---|---|
| DrivAerML vehicle 200k | 1178.4, 1178.1, 1178.0 | 1130.4, 1117.6, 1114.4 | -47.9, -60.5, -63.6 | **0.949** | 106.0 / 139.5 | 1 (A: 2 x 100k) |
| DrivAerML vehicle 400k | 1804.9, 1802.3, 1808.1 | 1615.1, 1614.2, 1611.7 | -189.9, -188.1, -196.4 | **0.894** | 108.1 / 205.2 | 1 (A: 4 x 100k) |
| DrivAerML whole 200k | 1303.9, 1310.8, 1309.3 | 1257.5, 1249.1, 1242.3 | -46.4, -61.7, -67.0 | **0.954** | 107.3 / 143.0 | 1 |
| AeroZoo 200k | 750.2, 750.2, 743.8 | 751.3, 750.5, 731.8 | +1.1, +0.3, -11.9 | 1.000 | 125.5 / 126.0 | 1 (A: 1, case_relative_l2) |
| AirfRANS (180,593 points) | 655.5, 649.3, 661.0 | 795.9, 794.2, 791.0 | +140.4, +144.8, +130.0 | **1.212** | 26.6 / 47.0 | 1 (A: 2 x 100k) |
| AirfRANS whole boundary | 867.8, 742.0, 733.9 | 878.6, 869.3, 884.8 | +10.9, +127.3, +150.9 | **1.18** | 27.4 / 47.9 | 1 (A: 2 x 100k) |
| lug (9,048 points) | 248.2, 236.5, 242.3 | 262.6, 267.7, 269.1 | +14.3, +31.2, +26.8 | **1.105** | 5.5 / 5.5 | 1 (A: 1) |

**What slows AirfRANS and the lug (c1 370666).** Arm C is the merged model with A's recipe (huber in chunks of 100k).
Its runs are on one node, in the order C B C. The guard of A's first run was too short there (my launcher), so A comes
from b1, except for the lug.

| case | A | C (merged model, old rule) | B (merged model, new rule) | the merges (C - A) | the rule (B - C) |
|---|---|---|---|---|---|
| AirfRANS | 655 (b1) | 650.9, 655.5 | 794.1 | about 0 | **+141 ms (+22%), +20 GiB** |
| AirfRANS whole boundary | 734-868 (b1, bimodal) | 734.0, 736.7 | 874.7 | about 0 | **+139 ms (+19%), +20 GiB** |
| DrivAerML vehicle 200k | 1178 (b1) | 1133.8, 1118.6 | 1128.1 | **-52 ms** | about 0 |
| lug | 243.0 (c1) | 278.5, 264.2 | 274.3 | **+28 ms (+11%)** | about 0 |

So two findings.

1. **The RULE lane's "fewest chunks" makes AirfRANS about 20% slower.** On AirfRANS, one chunk of 180k points is
   140 ms slower than two chunks of about 90k, and it peaks 20 GiB higher. RULE premised its rule on "one chunk of
   every point is the fastest form", but that was measured on DrivAerML, where one chunk is the same speed at 200k or
   faster. Whether to cap the chunk size (for example at the old 100,000 points, or a per-dataset optimum) is a design
   decision for the parent. I made no change, since this pass is on the critical path and RULE's rule is what the
   brief merges.
2. **The merged model is about 11% slower on the lug (+20-30 ms) with the rule unchanged, and about 4% faster on
   DrivAerML.** PLAN measured the lug unchanged. The likely source is upstream rather than a DTT-FAST lane:
   eba038f08 brings GEO-THIN's thin-cell geometry (921472488), which the book records as +15 and +18% on the
   whole-boundary geometry call, and A (5e82a9b98) predates it. This is not measured here; the GPU-h cap is spent.
   Timing the lug on 5e82a9b98 with only eba038f08 merged would settle it.

## Ideas for DTT-KNOBS

None new: every change here is a merge or a count, and none changes the pairs, the opening test or the arithmetic.

## Jobs and GPU-h

a1 2.58 + b1 1.69 + c1 0.53 = **4.80 of the 5 cap** (JOBS.md, "Pass 4").

# Pass 5 (2026-10-24, head a08adb0ec, pushed)

Brief: `BRIEF_PASS5.md`. Jobs: `JOBS.md` "Pass 5". Launchers and probes: `lhr/p5/` (= `$T/kernel/dttfast_integ/p5/`).
Copied outputs: `out/p5/a1` (job 371043), `out/p5/b1` (job 371199), `out/p5/tests` (cpu-short 371132).

## Commits

| commit | what |
|---|---|
| baeb92d59 | merge `origin/psharpe/isla` b66013630 (`--no-ff --signoff`, clean). ISLA 0.12.0, `_version` 12, refusal text and `isla_default_v1.mdlus` are b66013630's exactly (`git diff origin/psharpe/isla HEAD` is empty on `data/` and `geometry.py`) |
| d5321d547 | test only: `test_query_chunks_equal_the_forward_pass_and_its_gradients` gains a case `far_sphere` (icosphere about 1e3 of its size from the origin; one point at its centre, which the geometry ties; 12 on cell centroids, which it puts on the wall; 30 near). The test asserts the tie and the on-wall points first, then chunked = forward in outputs and every gradient (float64 roundoff). 8 cases pass locally |
| c6287db86 | candidate: each chunk's point tokens padded to `TOKEN_ALIGNMENT` for the blocks' reads, the kernel reads and the head on the real points, as forward has them (+ a test, failing on d5321d547 with `[37] != [64]`) |
| a08adb0ec | revert of c6287db86 (tree = d5321d547), because alignment is not the cause (below) |

**GEO-BAND's geometry through the chunked path.** GEO-BAND adds no model input: the band, the tie directions and the
centred frame live inside `exact_boundary_geometry`, whose signature is unchanged. `_case_inputs` computes the
geometry once for every prediction point, and `forward` and `query_chunks` both call it. The centred frame's origin
(`_centroid`) is a function of the boundary cells alone, so a point's geometry does not depend on the other points or
on the chunking. The `far_sphere` case tests this on a model whose points take the tie and on-wall branches.

## Why one chunk of AirfRANS was 20 percent slower (pass 4's decision 1)

**Neither alignment nor chunk size causes it.** The excess comes from the compiled point-side backward while the
process has seen only one chunk shape. Probe `span_probe.py` runs the recipe's own `_run_epoch` on the dump with the
spans forced (the recipe's decision runs and is logged first). Its arms are interleaved in one process, in 3 rounds
(order reversed on round 1). Each arm-round is 2 warm steps, 5 plain steps and 5 steps with synchronized phase marks.
Tables: `out/p5/a1/span_table.md`.

AirfRANS, 180,593 points, huber, bf16 (d5321d547). Times in ms, medians of 5:

| arm | spans | round 0 | round 1 | round 2 | point-side backward, round 0 / 1 / 2 | peak alloc GiB |
|---|---|---|---|---|---|---|
| one (RULE's choice) | 180,593 | **796.0** | 655.8 | 652.6 | **526** / 380 / 379 | 47.0 |
| two balanced | 90,297 + 90,296 | 669.4 | 667.2 | 667.1 | 389 / 388 / 389 | 24.3 |
| two, bound at 32k | 90,304 + 90,289 | 654.7 | 653.5 | 662.5 | 383 / 382 / 383 | 24.3 |
| chunks of 100k (old rule) | 100,000 + 80,593 | 653.9 | 656.1 | 661.5 | 381 / 381 / 381 | 26.6 |
| one, first 180,576 (aligned; probe only) | 180,576 | 636.0 | 642.0 | 641.2 | 366 / 366 / 366 | 47.0 |
| two, first 180,576 | 90,288 + 90,288 | 660.6 | 665.0 | 672.8 | 388 / 388 / 389 | 24.3 |

- The excess is the first arm's alone: 796 ms in round 0, the point-side backward 526 ms against 380. Once any other
  point count has run, the same one chunk takes 653-656 ms, level with two chunks (654-669) and with chunks of 100k
  (654-662). Every other phase (boundary side, chunk forwards, boundary backward, optimizer) is the same in every
  round.
- AirfRANS whole boundary shows the same: one chunk 872.5 in round 0, then 721.8 and 725.0. Chunks of 100k give
  726-731.
- Job b1 (371199) ran a process with one chunk only, 2 rounds: 785.7 and 785.4 ms, backward 526 in both. A process
  that never sees a second shape stays slow.
- Pass 4's `step.py` repeats one batch, so RULE's one chunk never leaves its first graph. The old rule ran 100,000 and
  80,593 in its first step, which put a second shape in that step.
- This fits Inductor's default automatic dynamic shapes (my reading; not traced): the first compile of
  `block.read` specializes to the static count, and the first new count recompiles it with the point axis dynamic.
  At 180,593 the static graph's backward is 146 ms slower. At 400,000 it is not (DrivAerML round 0 backward 603 against
  609-610 later).
- Training batches vary in count: AirfRANS reads 156k-210k points per case, capped at `sampling_resolution` 200,000,
  so a training run leaves the static graph after its first distinct count.
- A process compiled dynamic from its first call (`assume_static_by_default=False`) is not that state: 1,580 ms,
  backward 962. Every dimension becomes symbolic there, not just the point axis. So that check confirms nothing and
  is reported only.

DrivAerML vehicle 400k, every arm aligned except `@399983`, ms:

| arm | round 0 | round 1 | round 2 | point-side backward | peak alloc GiB |
|---|---|---|---|---|---|
| one (RULE's choice) | 1645.8 | 1619.7 | 1613.9 | 603-610 | 205.0 |
| one, first 399,983 (unaligned) | 1660.8 | 1642.1 | 1655.6 | 637 | 205.0 |
| two of 200k | 1637.7 | 1635.3 | 1642.1 | 622 | 140.3 |
| 4 x 100k (old rule) | 1668.2 | 1664.2 | 1666.5 | 651 | 107.4 |

**The size rule stays a pure memory rule, with no cap.** Once the graph is past its first shape, step time is flat in
chunk size on AirfRANS: one chunk 653-656, two 654-669. On DrivAerML 400k one chunk is the fastest: 1614-1646, against
1664-1668 for 4 x 100k. The pass-4 recommendation, the fewest chunks of at most 100k, would cost DrivAerML 400k about
+46 ms (+3%) and buy nothing on AirfRANS. Peak memory stays as RULE predicts it: 47.0 GiB for one AirfRANS chunk
against 24.3 for two.

**Alignment.** Two points matter here.
- An unaligned count costs about 2% at these sizes, all of it in the point-side backward. AirfRANS shows 366 ms against
  380 (180,576 against 180,593 points). DrivAerML shows 637 against 603-610 (399,983 against 400,000).
- The block-read padding (c6287db86) does not remove that cost. In job a1 its arms ran on the same node:

| case | arm | without padding | with padding |
|---|---|---|---|
| AirfRANS | one chunk, rounds 1-2 | 655.8, 652.6 | 652.1, 657.5 |
| AirfRANS | two chunks | 667.1-669.4 | 661.4-676.2 |
| AirfRANS whole boundary | one chunk, rounds 1-2 | 721.8, 725.0 | 720.5, 725.9 |
| DrivAerML 399,983 points | backward | 637 | 637 |

Round 0 with padding was 781 ms, the same static-graph excess. The cost sits in what forward also leaves unpadded: the
kernel reads' projections and the head on the real points. So the padding came out (a08adb0ec), as the brief's
condition ("if alignment is the cause") requires.

**bf16 families of the padded candidate (b1, `out/p5/b1/compare_airfrans_pad.md`).** On AirfRANS they are the head's
(below) to within run noise: blocks 0.96 / 1.01 / 3.93 against the head's 0.93 / 0.99 / 3.93, with 1 of 427 tensors
above. So padding does not explain pass 4's AirfRANS blocks maximum either.

**Pass-4 decision 2, accepted on pass 3's grounds.** The parent accepted the three single-run bf16 group maxima above
the reference family (lug blocks 1.48 against 1.00, lug other 1.12 against 1.11, AirfRANS blocks 3.93 against 3.44),
whose medians are at or below the reference's. Pass 3's grounds: 0-2 of about 430 tensors are systematically above.
Pass 5 shows the same pattern (below).

## Checks

| check | result |
|---|---|
| whole ISLA test directory on a GB300 with every CUDA test (b1 371199, a08adb0ec's tree = snapshot p5a; five streams, the compile file split by `-k` into 3 + 5 + 6 = its 14 tests) | **636 passed, 1 skipped, 0 failed** (3 + 5 + 6 + 166 + 456 passed + 1 skipped; pass 4 had 608 + 1). Logs `out/p5/b1/pytest_s*.log` |
| whole ISLA test directory on CPU, cpu-short 371132 (p5a) | **456 passed, 173 skipped, 0 failed** (629 collected = `collect_all.txt`) |
| recipe tests, same job | 590 passed, 15 skipped, 2 failed: the known `test_campaign_c_hooks` pair, with the same `TypeError: int() argument must be ... not 'NoneType'` at `physicsnemo/distributed/manager.py:350` as since pass 1, where it failed the same way on psharpe/isla 1e7f03d7d (365890). `manager.py` and the test file are unchanged from 1e7f03d7d to the head |
| 0.12.0 ISLA-DTT checkpoint (local `p5/sd_probe.py`; b66013630 saves, model.py 2168550b; the head loads, its model.py the same since the revert) | `from_checkpoint` and `load_state_dict(strict=True)`: 87 tensors, outputs bitwise (max abs 0.0) |
| local CPU | `-k query_chunks_equal` 8 passed (with `far_sphere`); `-k query_chunks` 10 passed on c6287db86 |

## Equivalence against b66013630 (b1 371199; `out/p5/b1/equiv_table.md`, `table_<case>.md`)

`eq_probe.py` (pass 4's) ran each version in its own process and printed its md5s: head model.py 89df35aa, reference
2168550b, geometry.py 3212d5a9 on both. Inputs: 2,000 points, pass 2's parameters with torch's state-dict metadata
dropped. 0.12.0 refuses the version-11 record the reference wrote under 0.11.0; the layout is the same. The
reference's unchunked mse step ran against the head's chunked step, the recipe's `chunked_forward_backward` with
chunks of 700 forced, which are unaligned.

| case | per-chunk lists = walk of every point (raw) | pairs (f64, f32) | float64 max rel | float32 ratio max (gated groups) |
|---|---|---|---|---|
| DrivAerML vehicle | True | identical | 7.1e-14 | 1.01 |
| DrivAerML whole boundary | True | identical | 3.1e-13 | 1.14 |
| AeroZoo | True | identical | 3.2e-14 | 1.02 |
| AirfRANS | True | identical | 1.9e-13 | 1.05 |
| lug | True | identical | 6.0e-14 | 1.00 |

### bf16 one-ulp families (b1; `out/p5/b1/compare_{airfrans,drv_body}.md`)

`diag_probe.py` ran with `compile_blocks`: the reference unchunked, the head chunked at 700. There were 2 cold
processes per version, 7 one-ulp samples, and a float64 truth.

| case | group: reference family min / median / max | head family min / median / max | tensors above in every run / below / total |
|---|---|---|---|
| AirfRANS | blocks 0.82 / 0.99 / 3.42; kernel_basis 0.66 / 1.00 / 1.34 | blocks 0.93 / 0.99 / **3.93** (one run of 7); kernel_basis 0.66 / 0.75 / 1.08 | 1 / 7 / 427 |
| DrivAerML vehicle | kernel_basis 0.95 / 1.06 / 1.12; other 1.00 / 1.00 / 1.47 | kernel_basis 0.90 / 1.02 / **1.14** (one run); other 1.24 / 1.24 / 1.24 | 0 / 11 / 427 |

Loss, outputs and kernel_reads are inside the reference's range. Three single-run maxima lie above it: the two in the
table, and AirfRANS other at 1.02 against 1.00 (median 0.98 against 0.97). Their medians are at or below the
reference's, or at most 0.01 above it. This is the pattern of the
accepted pass-4 decision 2 (AirfRANS blocks 3.93 against 3.44 then).

## Step time A/B (a1 371043: 9967490fe = A against c6287db86 = B; 3 alternations, A B / B A / A B, median of 15, bf16, compile_blocks, pass 4's `step.py`)

The head a08adb0ec is c6287db86 without the padding. At the aligned counts (DrivAerML, AeroZoo) the padding is a no-op
in code, so those rows are the head's own. At the unaligned ones (AirfRANS, the lug) the diagnosis measured the
padding as neutral; the lug (9,048 points to 9,056) was not re-timed.

| case | A ms | B ms | B / A | peak alloc GiB A / B |
|---|---|---|---|---|
| DrivAerML vehicle 200k | 1106.9, 1101.3, 1101.9 | 1123.2, 1114.9, 1120.6 | 1.017 | 139.5 / 139.5 |
| DrivAerML vehicle 400k | 1596.4, 1605.0, 1581.6 | 1626.4, 1611.4, 1612.9 | 1.010 | 205.2 / 205.2 |
| AeroZoo 200k | 727.3, 725.8, 723.6 | 736.8, 731.3, 734.8 | 1.012 | 126.0 / 126.0 |
| AirfRANS (180,593) | 791.9, 790.3, 781.0 | 791.6, 781.2, 779.1 | 0.988 | 47.0 / 47.0 |
| AirfRANS whole | 861.6, 869.3, 870.5 | 856.4, 858.7, 851.8 | 0.985 | 47.9 / 47.9 |
| lug (9,048) | 272.6, 266.7, 262.4 | 289.2, 266.6, 257.3 | 1.000 | 5.5 / 5.5 |

- On DrivAerML and AeroZoo B is 1-1.7% slower, outside the alternations' spread (DrivAerML 200k: +8 to +22 ms in each
  pair). Nothing on DTT-FAST's side changed there, so this is b66013630's geometry (GEO-BAND's centroid and bands). I
  did not isolate it.
- AirfRANS and its whole boundary are 1.2-1.5% faster, and the lug is the same.
- Both arms of AirfRANS still run the static one-chunk graph (about 790 / 860 ms), as `step.py` repeats one batch.
- The job had 8 CPUs (no `--cpus-per-task`) for 4 timing processes, as pass 4's b1 did, so the absolute times may
  carry CPU contention. A and B shared it alike.

## Ideas for DTT-KNOBS

None: no change in this pass touches the pairs, the opening test or the arithmetic.

An execution idea outside DTT-KNOBS: mark the prediction points' axis dynamic for the compiled block reads
(`torch._dynamo.mark_dynamic` in `_QueryChunks.__call__`), so that a run whose counts never vary does not keep a slow
static graph (AirfRANS at 180,593: +146 ms per step). It is not measured, and it does not matter in training with
varying counts.

## Jobs and GPU-h

a1 1.32 + b1 0.77 = **2.09 of the 2.2 cap** (JOBS.md, "Pass 5"). cpu-short: 371044 (cancelled after 12 min: it ran
c6287db86 before the revert), 371132, 371354.

## Remerge: origin/psharpe/isla b4b10d5a9 (DEFAULT-DTT and GEO-THINFAST), head 4d281a702 (pushed)

The parent's note of 15:26, read at 16:36 after the checks above had run: DEFAULT-DTT had landed at 16:18 EDT (merge
1100fdcf2, with GEO-THINFAST 2a521b20b), so I merged again and ran the checks once on that head. GPU-h: +0.70, within the
extra 0.8 the note allows.

| commit | what |
|---|---|
| 89710f345 | merge `origin/psharpe/isla` b4b10d5a9 (`--no-ff --signoff`). One conflict, in the recipe README: DEFAULT-DTT's sentence on the interior configs, then DTT-FAST's on the chunked step. model.py merges automatically: DTT-FAST's model.py delta against b4b10d5a9 is line for line its delta against b66013630. `query_independent=None` resolves in `__init__` to the task (`self.query_independent = self.interior`), so the recipe's `_chunked_step` (`getattr(core, "query_independent")`) chunks interior defaults. One DTT-FAST test, `test_query_chunks_need_query_independent_points`, relied on the old interacting default and now passes `query_independent=False` |
| 4d281a702 | recipe test: `test_chunked_step_chunks_isla_dtt` also asserts that `query_independent=None` (the interior default) takes the chunked step. Locally, `-k chunk`: 10 passed |

Version 0.12.0 and `_version` 12 are unchanged.

| check | result |
|---|---|
| whole ISLA test directory on a GB300 with every CUDA test (c1 371487, 5 streams, code_dttfast_integ_p5c = 4d281a702) | **709 passed, 1 skipped, 0 failed** (3 + 5 + 6 + 166 + 529) |
| ISLA directory on CPU, cpu-short 371488 | **461 passed, 241 skipped, 0 failed** (702 collected) |
| recipe tests, same job | 597 passed, 5 failed. Two are the known `test_campaign_c_hooks` TypeErrors (same message). Three are DEFAULT-DTT's new config tests on `isla_volume_vehicle`, a file only my LHR recipe snapshot carries (5e82a9b98's vehicle-only config, added for the probes), which those tests glob. On the branch, without it, `tests/test_isla_configs.py tests/test_utils.py` pass locally: 145 passed |
| 0.12.0 checkpoint (local `p5/sd_probe.py`; b4b10d5a9 saves, model.py 8716db38; the head loads, fa031026) | strict, 87 tensors, outputs bitwise (max abs 0.0) |

**Equivalence against b4b10d5a9** (c1, `out/p5/c1/table_*.md`; each version in its own process, md5s printed: head
model.py fa031026, reference 8716db38, geometry.py e02fea5a on both):

| case | raw | pairs | float64 | float32 ratio |
|---|---|---|---|---|
| DrivAerML vehicle | True | identical | 7.1e-14 | 1.01 |
| DrivAerML whole | True | identical | 3.1e-13 | 1.14 |
| AeroZoo | True | identical | 3.2e-14 | 1.02 |
| AirfRANS | True | identical | 1.9e-13 | 1.07 |
| lug | True | identical | 6.0e-14 | 1.00 |

**bf16 families** (c1 + cpu-short 371598; `out/p5/c1/compare_*.md`):

| case | group | reference family min / median / max | head family | tensors above in every run / below |
|---|---|---|---|---|
| DrivAerML vehicle | kernel_basis | 0.95 / 1.06 / 1.12 | 0.90 / 1.02 / **1.14** | 0 / 11 of 427 |
| DrivAerML vehicle | every other group | | inside | |
| AirfRANS | blocks | 0.82 / 1.00 / 1.46 | 0.71 / 0.99 / **3.94** (one run) | 0 / 6 of 427 |
| AirfRANS | other | 0.97 / 0.99 / 1.01 | 0.99 / **1.03 / 1.07** | |
| AirfRANS | loss, outputs, kernel_reads, kernel_basis | | inside | |

**For the parent: AirfRANS's "other" group.** Its family median, 1.03, lies above the reference family's maximum of
1.01. That is more than a single-run maximum. Against b66013630 in b1, with the same DTT-FAST code, it was 0.98 (inside,
max 1.02 against 1.00).

Between b1 and c1 only the shared upstream geometry changed, and it changed in both arms alike. The reference's own
family moved between the two jobs:

| group | reference family in b1 | reference family in c1 |
|---|---|---|
| blocks max | 3.42 | 1.46 |
| kernel_basis | 0.66-1.34 | 1.00-1.99 |
| baseline loss distance | 6.82e-4 | 5.82e-4 |

So 7-sample families move by more than the head's excess here, and no tensor is above the reference in every run.

I recommend accepting this on decision 2's grounds. Read strictly, the rule is exceeded on "other" (a 4% shift of a
group whose reference range is 4% wide), so the call is the parent's.

**Not re-run on the remerged head:** the step A/B. It would cost about 1.3 GPU-h and does not fit the extra 0.8.
GEO-THINFAST changes the lug's fast geometry call, so the lug row above (on b66013630) is likely pessimistic for the
head. The A/B rows above stand for b66013630's merge.

GPU-h, pass 5 total: 1.32 + 0.77 + 0.70 = **2.79** (cap 2.2 + 0.8 granted for the remerge).
