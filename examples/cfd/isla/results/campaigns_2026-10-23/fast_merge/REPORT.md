# FAST-MERGE report (2026-10-24, written 04:5x EDT)

## The merge

`5219f2e8f` on `psharpe/isla-geo-fast` ("Merge remote-tracking branch 'origin/psharpe/isla' into psharpe/isla-geo-fast",
`git merge --no-ff --signoff`, author Peter Sharpe). Parents `1086cf42e` (the branch) and `045dd14fe` (origin/psharpe/isla).
Tree `5561c0ce`, the tree `git merge-tree` predicted. No conflicts. Pushed with a plain push (`1086cf42e..5219f2e8f`).
Not merged into psharpe/isla. No other commit was needed: the merge is the only change.

## Per-path equality (merge base `fb263fa29`)

| paths | count | check | result |
|---|--:|---|---|
| changed only on the branch | 4 (geometry.py, geometry_fast.py, test_isla_geometry_degenerate.py, test_isla_geometry_fast.py) | merge = `1086cf42e` | all equal |
| changed only on psharpe/isla | 471 (the 0.11.0 version commit, book, results, recipe) | merge = `045dd14fe` | all equal |
| changed on both | 1 (model.py) | each side's hunks, as +/- lines, found unchanged in the merge's diff against the other parent | both intact: the branch's 44 lines (md5 c66529f7) and psharpe/isla's 30 lines (md5 2b21abc8) |

Against origin/psharpe/isla, the merge changes exactly the branch's five files (`--diff-filter=AM`), with no deletions or
renames. pre-commit on those five files passed every hook: ruff check, ruff format, license, large files, import-linter
(`work/precommit.log`). `pyproject.toml` and `uv.lock` are unchanged against origin/psharpe/isla.

## No version change

`__model_checkpoint_version__ = "0.11.0"`, `_version = 11`, as psharpe/isla. Test data, all 6 files under
`test/experimental/models/isla/data/`: the merge, origin/psharpe/isla and the worktree have the same md5:

| file | md5 |
|---|---|
| isla_backbone_v1.pth | 0d6fe827 |
| isla_default_checkpoint_output_v1.pth | cc4d7089 |
| isla_default_v1.mdlus | 59af2e3f |
| isla_default_v1.pth | 08ab6fa6 |
| isla_dtt_v1.pth | 6b8eed5b |
| isla_interior_v1.pth | 860c23e3 |

## Snapshots on LHR (read-only, MD5SUMS, SNAPSHOT_HEAD; `lhr/build_snapshots.sh 5219f2e8f`)

- `$T/code_fastmerge`: GEO-DEGEN's `code_geodegen3` with the 3 files the merge changes there (model.py, isla_default_v1.mdlus,
  test_isla_release.py) overlaid. All 3,173 files of `physicsnemo/` and `test/` match `git archive 5219f2e8f` by md5, and the
  snapshot has no extra files.
- `$T/recipe_fastmerge`: git archive of the recipe at `5219f2e8f` (tar md5 5058baf5), plus the 14 cluster-only dataset files of
  `recipe_surfbc2`.

## Equivalence (GB300, job 365757, nvl72d047-T10; logs `lhr/out_checks/`)

The head's geometry.py is md5 dd492ac1, the same as GEO-DEGEN's 1086cf42e: the merge leaves geometry.py unchanged.

- **(a) The merged head against 1e7f03d7d** (the merged 0.11.0 reference, geometry.py md5 6ba44b1d). It was loaded by
  `probes/refload3.py`, which renames its one op. Each log prints the op each side ran: REF `isla_closest_candidates_geometry_ref`,
  NEW `isla_closest_candidates` or `_fast`. Compared bitwise: every op output, w, d, u, and the gradients in the queries and the
  vertices, in float32 and float64.

  | sets | runs | queries | differing, head reference | differing, head fast |
  |---|--:|--:|--:|--:|
  | small (20 lug cases, icosphere, torus, with and without repeats, ellipse) + DrivAerML vehicle | 52 | 827,884 | 0 | 0 |
  | DrivAerML whole boundary | 2 | 400,000 | 0 | 0 |
  | AeroZoo, the 20 lug dump cases, AirfRANS, Tensile2d | 48 | 1,038,022 | 0 | 0 |
  | **total, per kernel** | **102** | **2,265,906** | **0** | **0** |

- **(b) Fast against reference on the merged head, bitwise** (GEO-DEGEN's probes, unchanged):
  - `probe_c2.py`, on GEO-FAST's adversarial sets and every dump: 86 runs, 1,958,096 queries (250,286 on the wall, 166,000
    tied, 5,941,818 candidate pairs). 0 differ in pairs, ties, op rows, w, d or u.
  - `probe_degen.py`, on the degenerate sets: 152 set-and-precision runs, 291,100 queries, 1,919 zero-area cells. fast 0, op
    mismatch 0, raises 0, wrong d, cells or signs 0. The TOTAL line is identical to GEO-DEGEN's 365177; only the path line
    differs.
  - The forced-fast GPU tests (`fast_plugin.py`) made 375 calls and compared 521,081 queries: 0 differ.
- **(c) `geometry_kernel="auto"`** (`probe_auto.py`; v1 md5 abba25c4 in 365757, v2 in 365898):
  - With Triton installed it resolved to `fast` on cuda:0. It was resolved once (cache `(('cuda', 0), 'fast')`), logged once
    over 3 forwards, and every forward ran `_closest_candidates_fast`.
  - It resolved to `reference` in three cases: with the import blocked (`sys.modules['triton']=None`), with the distribution
    reported absent, and with `HAS_TRITON` patched False as the tests do (each logged once, each running `_closest_candidates`).
    It also resolved to `reference` on CPU, checked locally.
  - No state-dict key names a kernel. A `load_state_dict` starts at `auto`, unresolved. A `.mdlus` saved with `fast` loads
    at `auto`.
  - No import statement in the package names triton (AST scan). `geometry_fast` loads triton with `importlib.import_module`.
  - In each mode, auto's outputs equal forced reference's bitwise.
  - Across modes, the outputs differ until the slice blocks are forced eager, because `geo_kernel` is also gated on Triton.
    With `PROBE_GEO_KERNEL=eager` (365898), `main` (fast) and `noimport` (reference) give the same output md5, 688589f3.

## Tests

| where | job | what | result |
|---|---|---|---|
| LHR cpu-short | 365758 | the whole ISLA directory on `code_fastmerge` | **362 passed, 100 skipped, 0 failed** (11:39) |
| LHR cpu-short | 365758 | the recipe tests on `recipe_fastmerge` | 521 passed, 12 skipped, 2 failed (`test_campaign_c_hooks.py::test_initialize_from_checkpoint_{loads_weights_only,rejects_architecture_mismatch}`) |
| LHR cpu-short | 365934 | that file at both parents, same environment | 2 failed, 2 passed at `045dd14fe` and at `1086cf42e`, with the same message |
| LHR cpu-short | 365934 | head, with `SLURM_*` unset | the file 4 passed; the recipe suite **523 passed, 12 skipped** |
| local CPU | - | the recipe suite at `5219f2e8f` | **523 passed, 12 skipped** (`work/recipe_tests_local.log`) |
| GB300 | 365757 | interior-geometry, tree-read, geometry-fast, geometry-degenerate and release files, forced fast | **241 passed, 1 skipped** |
| GB300 | 365757 | the same files under auto | **241 passed, 1 skipped** |

- The 2 recipe failures come from the environment, as BUNDLE-MERGE found. The assertion message is `TypeError: int() argument
  must be ... not 'NoneType'` at `int(os.environ.get("SLURM_NPROCS"))`: `save_checkpoint` calls
  `DistributedManager.initialize_slurm` inside an sbatch step without srun. Both parents fail the same way, and the head
  passes once the `SLURM_*` variables are unset.
- 365934 is recorded as FAILED only because its final `rm` hit the read-only tensordict copy. Every test step ran, and the
  node's per-job `/tmp` is gone (cleanup job 366144).
- The one GPU skip is the `ISLA_AF_POSED_SAMPLE` test, which is not a CUDA test.

## Timing (GB300, job 365898, nvl72d022-T03; GEO-FAST's final2 protocol)

Protocol: recipe_r090 train.py under `timed_train2.py` (md5 845fe1b4), REF-090 settings, bfloat16, eager, the same five cases
and inputs, 4 runs at once with the GPUs rotated. Each value is the median after 5 warm-up calls or steps. The arms:

- base: `code_cmpaf_fix`, 0.9.0, rerun in this job.
- head, reference: `code_fastmerge` with `GEO_KERNEL=reference`.
- head, auto: `code_fastmerge`, where auto resolved to fast.

The head arms ran 3 passes and the base 2. Full table, with each pass and GEO-FAST's 363375 base beside it:
`lhr/timing_table.md` (md5 c32ba428), built by `lhr/reduce_timing.py`.

Book shape (`lhr/book_table.md`, md5 e4e2f81e; the mean of passes 1 and 2, as the book's table):

| case (Q / F) | geometry call: base / reference / fast [ms] | step: base / reference / fast [ms] | step ratio: reference / fast |
|---|--:|--:|--:|
| AirfRANS 2-D (100000 / 1,018) | 11.7 / 29.9 / 30.1 | 365 / 386 / 425 | 1.06 / 1.16 |
| lug (9808 / 6,488) | 7.9 / 44.0 / 36.7 | 141 / 179 / 171 | 1.27 / 1.21 |
| AeroZoo (200000 / 501,802) | 18.1 / 103.0 / 83.0 | 505 / 592 / 579 | 1.17 / 1.15 |
| DrivAerML Volume, vehicle (200000 / 16,953,702) | 98.0 / 369.5 / 307.7 | 568 / 861 / 787 | 1.52 / 1.39 |
| DrivAerML Volume, whole boundary (200000 / 16,958,756) | 2,420.7 / 431.4 / 353.7 | 2,897 / 928 / 843 | 0.32 / 0.29 |

Against GEO-FAST's 363375: this job's base is within 2% of 363375's base on every case. The head's geometry calls are within
about 1 to 6% of GEO-FAST's head rows; this job is slightly slower overall.

**The AirfRANS fast step of 425 ms is an outlier.** Its pass 2 (`head2`, GPU 0, after the base) has a step median of 458 ms,
against 392 and 386 in passes 1 and 3. Its geometry call in that pass is 29.3 ms, equal to the others, so the extra time
does not come from the geometry. The geometry call under fast equals the reference's on AirfRANS (30.1 against 29.9 ms)
because 2-D takes the same brute-force segment search under both kernels. The fast kernel speeds up only the 3-D search. On
passes 1 and 3 the AirfRANS fast step is 389 ms (1.07x), as in GEO-FAST's 389 / 382.

Peak memory is the same under reference and fast on every case, within 0.1 GiB.

## GPU-h

365757: 4 GPUs for 4:40. 365898: 4 GPUs for 24:33. Total **1.95 GPU-h** of the 3 GPU-h cap. 365758, 365934 and 366144 ran on
CPU only. The registry row `^isla\.fastmerge\.` was added at 04:03 and removed at 04:47 EDT (backups
`ACTIVE_EXPERIMENTS.tsv.bak` and `.bak2`).
