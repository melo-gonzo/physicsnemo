# MERGE-PREP report (written 2026-10-24 about 05:45 EDT)

Lane MERGE-PREP of the ISLA program, launched 04:18 EDT by the supervisor (01a13c2c). Two merge branches, each merged
with origin/psharpe/isla 7213c70c5 (FAST-MERGE's merge, ISLA 0.11.0 with the fast geometry path) and tested on the merged
tree. Part 1 is on psharpe/isla as 1b25eb899 (the supervisor's merge). GPU use: 6.98 GPU-h of the cap, which the
supervisor raised from 4 to 9 at 05:58 (366130 1,964 s, 366327 1,089 s, 366600 370 s, 366724 1,412 s, 367119 1,444 s;
4 GPUs each). Registry row ^isla\.mergeprep\. added at 04:44 and removed at 07:27 (backups
watchers/.ACTIVE_EXPERIMENTS.bak.mergeprep_{add,rm}).

LHR: `T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`,
launchers and logs in `$T/kernel/mergeprep` (local copies in `lhr/`). Snapshots, read-only, with SNAPSHOT_HEAD and
MD5SUMS: `$T/code_mergeprep_a8` (1062d6086, for the timing), `$T/code_mergeprep_a8m` + `$T/recipe_mergeprep_a8m`
(17c9df5c0), `$T/code_mergeprep_bm` + `$T/recipe_mergeprep_bm` (e94d91c86), `$T/code_mergeprep_isla7213` (7213c70c5, for
the base comparison), `$T/code_mergeprep_bm2` + `$T/recipe_mergeprep_bm2` (2b15490b2). Each code snapshot is code_align8a (32fa15389 + td0152) with the ISLA model and test directories
replaced; `git diff 32fa15389 <head>` outside those directories is empty for both heads, so nothing else differs.

## Part 1: token padding, psharpe/isla-align8, head 17c9df5c0 (pushed)

| commit | what |
|---|---|
| 1062d6086 | 1a: in eager mode an aligned token count runs no padding (`_pad_tokens` returns the tokens as they are; the kernel read and the head read use the unpadded token set when nothing was padded); new test `test_an_aligned_token_count_runs_no_padding` (surface, interior) |
| 17c9df5c0 | 1c: `git merge --no-ff --signoff origin/psharpe/isla` (7213c70c5), no conflicts, no rerere replay |

**1a, cost where nothing is padded.** e60be94d4 did run the pad path at aligned counts: `F.pad` with zero padding is a
clone (7 or 8 `constant_pad_nd` per forward), ISLA-DTT gained 2 `slice_backward` and 2 `add` from an empty `h[:, n:]`
slice, and the surface task's gradients were not bitwise (the full-range head slice). Probe `probes/bitwise_aligned.py`,
one tree per process, 20 cases (surface 3-D and 2-D, interior, ISLA-DTT, 2-D interior; f32 and f64; train and eval; all
at multiples of 32): base 32fa15389 against e60be94d4 `probes/bitwise_aligned.txt` (DIFFERENCE); against 1062d6086
`probes/bitwise_aligned_fix.txt`: outputs and every gradient bitwise and the same aten compute ops, all 20. After the 1c
merge, 7213c70c5 against 17c9df5c0: the same, all 20 (`probes/bitwise_aligned_7213_vs_17c9df5c0.txt`). Whole-model
compile still pads (a `pad == 0` test would guard the dynamic count); the compile key-order test passes locally.

**1b, timing** (`results/TIMING.md`, `results/time_table.md`; job 366130, base code_align8a against fix code_mergeprep_a8,
base, fix, base, fix per case and mode, natural and aligned counts interleaved in each process). Nothing padded: eager
runs exactly the base's kernels (GPU time 1.000, 1.000, 0.999), compile_blocks 0.97-1.00, every wall ratio inside the
base's own run-to-run spread. Padded: GPU time per step 0.885 (lug eager), 0.888 (lug compile_blocks), 0.923 / 0.870
(Tensile2d), 0.998 / 0.956 (DeepJEB++, 50,040 cells: already a multiple of 8); wall level or faster (lug compile_blocks
0.94). Accept under the brief's rule. ALIGN8's open lug reading (fix 10 % slower at j = 0) was noise. Whole-model
compile was not measured: one process did not finish compiling within its 720 s guard, base or fix.

**1c, per-path merge check** (`merges/m3_align8_isla.tsv`, `probes/merge_check.sh`: each path against both parents and a
fresh `git merge-tree`, which never reads rerere). 88 paths: 85 take psharpe/isla's version, 2 the branch's
(test_isla_query_encoding.py, test_isla_token_padding.py), and model.py is a clean line combination that equals the fresh
merge. In model.py the branch's diff against its base equals the merged file's diff against 7213c70c5, and psharpe/isla's
diff equals the merged file's diff against 1062d6086 (`merges/m3_*.diff`).

**Tests on 17c9df5c0.**
- ISLA test directory, CPU, LHR cpu-short 366325: 367 passed, 100 skipped, 0 failed (17:15).
- Recipe tests, local (a short check under LANE_RULES; cpu-short refused on the user-wide submit limit twice):
  523 passed, 12 skipped, 0 failed (`results/recipe_tests_align8.log`).
- GB300 366327 and 366600: test_isla_release.py + test_isla_token_padding.py 38 passed; the geometry files
  (interior_geometry, geometry_fast, geometry_degenerate) 166 passed, 1 skipped (needs an external sample).
- test_isla_compile.py on GB300: incomplete on both trees. In 366600, with a 360 s guard, 17c9df5c0 passed its first 3
  tests and 7213c70c5 its first 5 before the guard; in 366327 the align8 group (geometry + compile) hit its 1,080 s guard
  inside the compile file. No test failed; compile tests are slow here on both. Not rerun: 0.2 GPU-h remain of the cap.
  The CPU compile tests pass in 366325.
- Hooks on the 3 files the branch changes against origin/psharpe/isla: all passed (`merges/hooks_align8.txt`).

**Supervisor's checks of 05:58 EDT (part 1, head 17c9df5c0).**
- (a) test_isla_compile.py with every CUDA test, GB300 job 366724 (nvl72d053-T02, short QoS, 2,700 s guard,
  `--durations=0`, `lhr/tests_compile.sbatch` md5 3baee5f7): 14 passed, 0 failed, 22:52
  (`results/out_compile_366724/align8.log`). The base run was not needed. The slowest test was the max-autotune
  full-resolution draw, at 196 s.
- (b) Every book chapter whose cells build or run ISLA (03, 04, 05, 06, 09, 10, 12, 19, 20, index; found by grepping the
  cells for `ISLA(`, `from_checkpoint`, `boundary_meshes=`, `recipe_model(` and the geometry and draw calls). Each was run
  with `probes/run_cells_all.py` on 7213c70c5 and on 17c9df5c0 (the examples trees are identical), local CPU, nice 19,
  one chapter at a time, MemAvailable 19 GB. All 293 cells pass on both trees. 3 cells print differently
  (`results/book_diff_7213_vs_17c9df5c0.txt`):
  - 05 `code-interior` and `code-read-only`: the source listings show the padding lines (`_pad_tokens`, `n_real`).
  - 06 `tbl-using-contracts`: float64 contract residuals at roundoff, such as rotation 6.5e-15 / 2.0e-15 becoming
    6.4e-15 / 2.2e-15, and geometry x7 1.7e-15 / 6.5e-16 becoming 1.4e-15 / 7.0e-16. The demo's 80-cell airfoil pads
    to 96 tokens, so the summation order changes, as the fix allows where it pads.
  - No other printed number changes. In the next book batch, these become the new values, and the prose around the
    two listings could mention the padding.

## Part 2: whole boundary and surface task, psharpe/isla-boundary-merge, head e94d91c86 (pushed)

| commit | what |
|---|---|
| 6215af9a7 | merge of origin/psharpe/isla-whole-boundary 4b772ec59, no conflicts |
| 658289991 | merge of origin/psharpe/isla-surface-bc e636f29bd: 4 conflicts with ce598a72d |
| f6c8e5429 | a surface task that names every given boundary reads its outputs from the token state itself (bitwise 0.11.0) |
| 8dcfc24f5 | 2b: the request contract in the `prediction_boundaries` docstring |
| e94d91c86 | `git merge --no-ff --signoff origin/psharpe/isla` (7213c70c5), no conflicts |

**Per-path checks.** 6215af9a7 (`merges/m1_whole_boundary.tsv`): 116 paths, each one parent's version, none combined.
658289991 (`merges/m2_surface_bc.tsv`): 136 paths; 131 one parent's version, test_isla_configs.py a clean combination
equal to the fresh merge, and the 4 conflicted files below. e94d91c86 (`merges/m4_boundary_isla.tsv`): 59 paths, model.py
the only combination, clean, equal to the fresh merge; both sides' model.py diffs are preserved exactly.

**2a, the conflicts** (rerere recorded preimages only; it replayed nothing).
- model.py: one docstring hunk (the boundary-token paragraph). psharpe/isla added the float64 draw sentence; the branch
  rewrote the surface-task sentence. Both kept. The rest of model.py: the branch's diff against the merge base equals the
  merged file's diff against psharpe/isla except in that sentence, and the same for psharpe/isla's side
  (`merges/model_*.diff`).
- test_isla.py, test_isla_nd.py, test_isla_release.py (reformatted on psharpe/isla's side): `probes/resolve_fmt.py`
  ruff-formats (ruff 0.17.0, the hook's pin) the base, psharpe/isla's and the branch's versions, merges them line by
  line, and formats the result R. Each had 0 line conflicts in both directions. AST(R) equals AST of the other route
  (the branch's file with psharpe/isla's edits applied), and every top-level node equals the version of the side that
  changed it; no node changed on both sides (`merges/resolve_tests.txt`). ruff-format leaves R unchanged.
- ISLA stays 0.11.0: no version, constructor or state-dict change (the keyword owns no parameters).

**2a, bitwise against psharpe/isla** (`probes/bitwise_merge.py`, one tree per process, 28 cases: surface 3-D at 203 and
224 cells, 2-D surface, interior with one and two boundaries, ISLA-DTT, 2-D interior; f32 and f64; train and eval). As
first merged (658289991), interior forwards were bitwise. Vehicle-only surface forwards had bitwise outputs and points,
but gradients that were not bitwise. The cause is the surface branch's own `out_rows`, always set, so naming every boundary
read `torch.cat([tokens[:, 0:N]])`, a copy that reorders gradient accumulation and costs a copy on every DeepJEB++ and
HiLift step. f6c8e5429 sets `out_rows` to None when every given boundary is named (decided by name, so no shape guard).
Then all 28 cases are bitwise against 045dd14fe, and after the 7213c70c5 merge all 28 against 7213c70c5
(`probes/bitwise_merge_isla_vs_boundary_merge_fix.txt`, `probes/bitwise_merge_7213_vs_e94d91c86.txt`). The supervisor was
told at about 05:00 EDT; drop f6c8e5429 to keep the surface branch as written.

**2b.** 8dcfc24f5 adds to the `prediction_boundaries` docstring: the prediction is a quadrature over the request, so a
request must cover the body (half a car raises the DrivAerML Surface pressure and wall-shear errors 10 to 18 times);
a larger request or the mean of several lowers the error, with the spread between requests falling about as one over the
square root of the cell count.

**Tests on e94d91c86.**
- ISLA test directory, CPU, LHR cpu-short 366326: 369 passed, 100 skipped, 0 failed (12:48).
- Recipe tests, local: 553 passed, 12 skipped, 0 failed (`results/recipe_tests_boundary-merge.log`). The 2 environmental
  test_campaign_c_hooks failures appear only inside sbatch, so they do not appear here.
- GB300 366327: test_isla_release.py + test_isla_surface_conditions.py 40 passed; geometry files 166 passed, 1 skipped.
- Hooks on every file the branch changes against origin/psharpe/isla 7213c70c5: all passed
  (`merges/hooks_boundary_merge_e94.txt`).

**2c, the book** (`book/BOOK_CHANGES.md`, no book file edited). Chapter 05: 41/41 cells pass; two print new content (the
conditions table lists the five DrivAerML pieces for isla_surface and isla_volume and AirfRANS's freestream; the stage-1
listing gains the `out_rows` block). Chapter 06: 11 of 32 cells fail on the merged tree, all because the demos call
surface models without the now-required `prediction_boundaries` (4 raise, 7 depend on them); with seven one-argument
edits (listed) all 32 pass and print the same numbers, except `para-using-scales` (AirfRANS nu_t/(UL) median 7.5e-5 to
1e-7, "3 to 4 orders" to "3 to 7", from AF-WHOLE's constants refit) and the conditions table. The night lane's
`run_cells.py` cuts cells at a fence printed inside a string (13 false failures in chapter 05 on both trees);
`probes/run_cells_all.py` does not.

## 2d: part 1 merged in, psharpe/isla-boundary-merge head 2b15490b2 (pushed)

Part 1 went onto origin/psharpe/isla as 1b25eb899 (on 0158ed4c0, the BOOK-GEOFAST book merge); it adds no code beyond
17c9df5c0. 2b15490b2 is `git merge --no-ff --signoff origin/psharpe/isla` into e94d91c86. One conflict, in model.py's
head-read hunk: rerere replayed the rehearsed resolution, and model.py equals the reviewed trial ea44c3c55 byte for
byte. test_isla_token_padding.py took the trial's version (its surface calls name their boundary). The code and recipe
trees equal the trial's; only origin's new book and records files differ. Per path (`merges/m5_boundary_2d.tsv`): 106
paths are one parent's version, model.py is the conflicted file, and test_isla_token_padding.py is the edit above. Hooks
on every file the branch changes against 1b25eb899 pass (`merges/hooks_boundary_2d.txt`). Snapshots:
`$T/code_mergeprep_bm2`, `$T/recipe_mergeprep_bm2` (model.py md5 cae32d49, equal to the worktree's).

Tests on 2b15490b2:
- ISLA test directory, CPU, cpu-short 367120: 374 passed, 100 skipped, 0 failed (19:26).
- Recipe tests, local (cpu-short refused on the submit limit): 553 passed, 12 skipped, 0 failed
  (`results/recipe_tests_2b15490b2.log`).
- GB300 367119 (nvl72d053-T02, `lhr/tests_gpu_2d.sbatch` md5 e3d85872, 45-minute guard per pytest, `--durations=0`):
  test_isla_compile.py 14 passed (23:19); release + surface conditions + token padding 45 passed; geometry files 166
  passed, 1 skipped (`results/out_gputests2d_367119/`).
- The book on 2b15490b2 against 1b25eb899 (ten chapters): see `book/BOOK_CHANGES.md`. Chapter 06 fails the same 11
  cells until the seven edits are applied. Chapter 20's `para-limits-whole-boundary` cell needs a change, because it
  maps `isla_surface` to SHIFT-SUV.

**A gap the merge exposes (not fixed; for the supervisor).** `isla_surface.yaml` now names `boundaries.inlet`,
`.no_slip`, `.outlet` and `.slip`, and none of the SHIFT-SUV dataset files reads them, because SHIFT-SUV stores no
tunnel (SURFACE-BC REPORT section 5). So ISLA can no longer train on SHIFT-SUV with `isla_surface`, nor take SHIFT-SUV
as an `extra_datasets` entry. I infer this from the forward_kwargs resolver, which raises KeyError on a missing path; I
did not run it. A body-only SHIFT-SUV config (for example `isla_surface_suv`, reading `vehicle` and predicting there) would
keep it trainable until a tunnel sidecar exists. Under Peter's whole-boundary ruling that config would be a documented
gap, not the canonical form.

## 2d rehearsal (not pushed)

Merging 17c9df5c0 into e94d91c86 conflicts in one hunk of `_forward_tokens`, the head read. Resolution: the prediction
points `tokens[:, n_boundary:n_real]`, else the prediction boundaries' rows when `out_rows` is set, else
`tokens` or `tokens[:, :n_real]`; and test_isla_token_padding.py's two surface calls name their boundary
(`prediction_boundaries=None if q is not None else ["wall"]`). Trial commit ea44c3c55 on the local branch
`mp-trial-2d` of the boundary-merge worktree: bitwise and same compute ops against both e94d91c86 and 17c9df5c0 on the
20 aligned cases (`probes/bitwise_2d_trial_vs_*.txt`); token padding, surface conditions, query encoding and release
tests 40 passed locally; hooks pass. rerere has recorded this resolution.
