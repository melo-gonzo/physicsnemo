Status: DONE (ISLA-PREMERGE; written 01:10 EDT 10-26)

**Branch:** `psharpe/isla-premerge-1026` at **737f1c48c**, pushed to origin with a plain push. `git ls-remote` showed no
such branch before the push and shows 737f1c48c after it. It contains psharpe/isla 2ed83c0cf, upstream/main 289e57382
(unmoved at 01:00) and the five PR heads, each unmoved at 01:00 by `gh pr view`. Not merged anywhere.

## Inputs that move (read first)
- **Baseline loader leaves:** none move on the final tree. Seven area-protocol configs moved, then were fixed; all are
  scoring sets, not training sets:
  - GT and T on `drivaer_ml_surface_area`, `shift_suv_surface_area` and `highlift_surface_medium_area`;
  - ISLA on `shift_suv_surface_area`.
  - GB300 job 395320, 9f279664c vs 8112112f2: `PoissonBiasedSubsampleMesh` kept a different number of cells (for
    example 99,897 vs 99,608), so every leaf of those batches moved. CPU was bitwise.
  - Cause: #2033 replaced psharpe/isla's `set_generator` fix (5cd0ab835). Transforms now get an index-less
    `torch.Generator("cuda")`. Four recipe subsamplers compared devices with `!=` and dropped the seeded generator:
    `src/domain_transforms.py` :212, :313, :363 and :1014.
  - Fix 737f1c48c (approved, D3): the four sites use `physicsnemo.utils.weighted_multinomial._devices_match`, plus
    tests. Job 396490: **7/7 configs bitwise to 9f279664c**. Job 396510: the CUDA test fails on the pre-fix recipe and
    passes with the fix.
- **ISLA loader leaves:** none move. **No training config moved.**
- **Final tree, GB300:** 11,365 of 11,365 leaves bitwise over 73 composing configs (`LEAF_AB.md`; configs
  `lhr/leafcfgs.tsv`, md5 5ede9e30). This includes all 7 AeroZoo splits for ISLA and GT, T full, and GT/T on
  `drivaer_ml_volume` (the baselines' DrivAerML Volume config).
- **4 configs fail identically on 9f279664c and on the final tree** (both sides, GB300 and CPU):
  - GT and T on `drivaer_ml_volume_reference`: PlaceGroundUnderBody refuses the dropped domain meshes. This is the
    plan's section 7 static risk. The baselines read `drivaer_ml_volume`, which is bitwise.
  - ISLA on `drivaer_ml_surface_area.yaml` and on `shift_suv_surface_fastback.yaml`: `boundaries.inlet` not found.
    These are stale cluster-only files (`$T/recipe_dttfast_rule_b`, identical in the 40 newest recipe snapshots) that
    lack the tunnel pieces and the CFD box.
  - **Flag:** the DrivAerML Surface standing lists the area protocol. ISLA's rerun cannot score area on that file, as
    SLICE-NORMAL-EXP already found. SHIFT-SUV's fastback file serves only the book extra `suv_xfam`.
- **GLOBE compiled crashes with #2172 on torch 2.12** (reported to you at 01:00).
  - The error is `InductorError: TypeError: list indices must be integers or slices, not NoneType`.
  - 9f279664c + #2172 alone crashes; torch 2.13 runs (job 396599).
  - Repro: `globe_repro/README.md`, with the traceback and `minifier_launcher.py`. No rerun lane compiles GLOBE.

## Merges and commits (all `-s`, hooks on, Peter Sharpe, no trailers)
| sha | what | conflicts → resolution |
|---|---|---|
| 72f0a9ccf | forceref_integrate.py made to pass the hooks (header, ruff format, 3 docstrings; no logic). Needed before the drop's edit; see note 1 | — |
| a2919006e | recipe and results stop using cache_host_views and read_only_memory_maps | — |
| 55ae2e67a | merge #2033 1b1f8a236 | CHANGELOG, readers/mesh.py, test_mesh_readers.py: the 7 #2033 files are now equal to 1b1f8a236; recipe `_load_sample(index, generator, interior_n_points)` overrides (domain_transforms.py 695, 742) fixed in the merge, note 2 |
| 39482cb04 | merge #2172 3cf768033 | test_measure.py: all tests kept |
| bd27f5fa2 | recipe ComposeQuadratureMeasure drops copy=True | — (probe: identical in 6 dtype cases) |
| d19af528c | merge #2173 70161f801 | clean; `muon_cuda_graph: false` in base.yaml, false in all 29 composed configs |
| 3af62c800 | merge #2166 8afbeb233 | pyproject `tensordict[zarr]>=0.15.3,<0.16.0`; uv.lock regenerated (`uvx --from uv==0.12.20 uv lock --upgrade-package tensordict`, then `uv lock --check` exit 0) and equal to 8afbeb233's |
| cb275c637 | merge #2171 21340864e | train.py, README, test_train_helpers.py: #2171's loop head, `stop`, iter_time_s and peak-memory text verbatim; ISLA's chunked step and `_average_gradients_` kept; the chunked step prefetches after chunked_forward_backward and before step (probe), note 3 |
| 8112112f2 | merge psharpe/isla-lib-sampling fe86a0011 | CHANGELOG only; `git diff fe86a0011 HEAD` = `git diff 9f279664c cb275c637` line for line |
| ca35a1a5c | merge psharpe/isla 2ed83c0cf | `git diff HEAD^1 HEAD --stat` empty |
| 737f1c48c | recipe: the four subsamplers keep their seeded generator on CUDA | — |

## Drops (line at 9f279664c → replacement)
- `MeshDataset(cache_host_views)`: mesh_dataset.py 87/113/124/129/241/252, plus the recipe's src/datasets.py
  433/512/739/766/918/1022/1063/1088, deleted. No config sets it (checked locally and on LHR).
- `CenterMesh(use_measure_weighting)`: transforms/mesh/transforms.py 217/236/240/246/290. No config sets it.
- `read_only_memory_maps`: readers/mesh.py 29-30 and 42-82, plus test_mesh_reader_readonly_memmap.py.
  forceref_integrate.py 32/74/103 now load with `mode="r"`.
- `set_generator` index-less-cuda fix: mesh_dataset.py 169-175. Its CUDA effect on four recipe subsamplers is
  restored in the recipe by 737f1c48c.
- Recipe `copy=True`: domain_transforms.py:1049.
- Taken from #2033: the per-index cache lock, `close()` and the shared interior-count draw.

## Checks on the final tree
physicsnemo/, test/, pyproject.toml and uv.lock are identical at 8112112f2 and 737f1c48c.

| check | job / where | result |
|---|---|---|
| ISLA gpu tier (`lhr/submit_td.sh` = isla_tests/submit.sh with td0153) | 395316 | 8 invocations rc 0: 1,248 passed, 5 xfailed |
| ISLA full CPU tier | 395268 | 716 passed, 516 skipped, 5 xfailed |
| recipe tests, CPU, local .venv + td0153 (W3) | local | 631 passed, 24 skipped |
| recipe tests, GB300 | 396510 | 655 passed, 6 skipped (test_campaign_c_hooks deselected) |
| test/mesh; test/models/globe; test/optim (LHR cpu-short, torch 2.13 venv) | 395261 | 4009 passed; 250 passed; 28 passed |
| test/datapipes -k "not domino" | 395261 | 815 passed, 1 failed: `test_mesh_datapipe[cuda]` (DALI without CUDA). It fails the same way on 9f279664c (395950) |
| recipe on that venv | 395261 / 395950 | 604 passed, 2 failed. Both are `test_campaign_c_hooks` (`int(None)` in a Slurm env), with the same messages on 9f279664c |
| goldens A/B CPU / CUDA vs fe86a0011 | 395260 / 395951 | 110/110 and 155/155 PASS; CPU sha256 10,571/10,571 equal; CUDA: no deterministic tensor differs |
| model A/B eager, 12 families | 395951, 396490 | 11 BITWISE; GLOBE within its own run-to-run spread (`AB_TABLES.md` §1) |
| compiled A/B | 395951, 396490 | ISLA surface and DTT: A/B of the same order as two cold compiles; GT/T outputs move 8 and 2 bf16 ulp (the expected #2172 movement); GLOBE crashes (above) (§3) |
| book render (book_render.lock, `PYTHONPATH=td0153:<worktree>`, CUDA hidden) | `render.sh`, `render_full.log` | rc 0; no quarto-unresolved-ref in log or 23 HTML files; 21 TCP kernel warnings only |
| `pre-commit run --files` (27 changed files) / `import-linter --all-files` | `precommit_*.log` | pass / pass |

## Overlay and .venv (`TD0153_OVERLAY.md`)
- `$T/td0153` and local `td0153/` hold the pure-Python wheel; its sha256 a437693b… matches uv.lock at 8afbeb233.
- The doc lists every snapshot script that copies 0.15.2: isla_tests/submit.sh:48 (ready swap `lhr/submit_td.sh`),
  the mk_snap.sh variants, prp7's `$K/td0152`, local PYTHONPATH and book runs.
- The shared .venv needs only tensordict 0.15.0 → 0.15.3. Its zarr 3.4.0 meets the new zarr>=3.1.1. I did not change
  it.

## Notes for the main session
1. The brief said to stop on a hook failure in untouched code. W1's three contact_supervisor calls timed out, so it
   fixed the hooks in a separate commit (72f0a9ccf, no logic change).
2. The `_load_sample` fix in 55ae2e67a goes beyond the list. Without it the case-key and VTK readers raise TypeError.
3. Peak memory now resets at each epoch's start (#2171). The first epoch no longer counts pre-epoch peaks.
4. 737f1c48c imports a private library helper. A local copy is the alternative.
5. The book render prepends td0153, since the tree needs 0.15.3.
- New for project memory:
  - Code at or after this premerge uses `$T/td0153`, not td0152.
  - GLOBE `compile=true` fails on torch 2.12 with #2172.
- Every merge ran as briefed. Rulings used: Peter's "fair game" merge permission, and the supervisor's D1-D4 (configs;
  GPU cap 3 → 4.5 → 6; the recipe fix).

## GPU, files, and disk
- GPU: 5.88 of the 6 GPU-h cap. Nothing is running. The registry row `^premerge\.` has been removed.
- Tables and logs in this folder (11 MB): `JOBS.md`, `LEAF_AB.md`, `AB_TABLES.md`, `MD5_vs_isla.txt` (per-file md5s of
  the 28 files that differ from 2ed83c0cf), `REPORT_W1-3.md`, `DECISIONS_NEEDED.md`, `cmp/`, `logs/`, `goldens/` and
  `globe_repro/`.
- On disk:
  - Worktree `/home/psharpe/gh/physicsnemo-isla-premerge-1026` (1.2 GB, clean). Remove it once the branch merges.
  - LHR `$T/kernel/premerge` (36 GB, mostly the model A/B `.pt` files of 395951 and 396490, kept for follow-up on
    GLOBE and CERT). It also holds the read-only snapshots `code_/recipe_{base9f,headcb27,libsfe86,fin8112,only2172}`
    and `recipe_{base9fx,fin737f}`.
  - LHR `$T/td0153`, `$T/wheels/` and `$T/kernel/isla_tests/snap/fin8112_{full,gpu}`.
  - I deleted the leaf-hash `.pt` files.
