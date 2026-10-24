Status: DONE

**The merge is ready for the supervisor.** `5219f2e8f` on `psharpe/isla-geo-fast` merges origin/psharpe/isla `045dd14fe` into
`1086cf42e` (`--no-ff --signoff`, no conflicts, tree `5561c0ce` as `git merge-tree` predicted). Pushed with a plain push; not
merged into psharpe/isla. Every equivalence check is bitwise 0, and no failure comes from the merge. Details:
`scratch/isla-fast-merge-2026-10-24/REPORT.md` and `JOBS.md`.

**What ran.**
- GB300, short QoS, the five bad nodes excluded: 365757 (checks and GPU tests, 4:40) and 365898 (timing, 24:33).
- LHR cpu-short: 365758 (the suites), 365934 (the recipe failure at both parents) and 366144 (cleanup).
- **GPU-h: 1.95 of the 3 cap.** The registry row was added and is now removed; nothing is running.

**Equality per path** (merge base `fb263fa29`):
- The 4 paths only the branch changed equal `1086cf42e`; the 471 only psharpe/isla changed equal `045dd14fe`.
- model.py, changed by both: each side's hunks are intact in the merge, byte for byte.
- Against origin/psharpe/isla the merge changes only the branch's 5 files. Hooks pass on them, import-linter included.
- pyproject.toml and uv.lock are unchanged. ISLA stays 0.11.0 (`_version` 11). No shim, no flag.
- All 6 test-data files equal origin/psharpe/isla's (isla_default_v1.mdlus 59af2e3f; the rest in REPORT.md).

**Equivalence on the merged head (GB300):**
- **(a) Against 1e7f03d7d**, loaded with its op renamed (each log prints the op each side ran): 0 of 2,265,906 queries differ
  over 102 runs, under the reference kernel and under fast. Compared: every op output, w, d, u and their gradients. The sets:
  the small set, DrivAerML vehicle and whole boundary, AeroZoo, the 20 lug cases, AirfRANS and Tensile2d.
- **(b) Fast against reference:**
  - 0 of 1,958,096 queries differ over 86 runs on the adversarial sets and the dumps.
  - 0 of 291,100 differ over 152 degenerate runs. Its TOTAL line is identical to GEO-DEGEN's 365177.
  - 0 of 521,081 differ in the forced-fast GPU tests.
- **(c) auto:**
  - It resolves to fast on GB300, and to reference when Triton is blocked, absent from the metadata, or patched off.
  - It is resolved once, logged once, absent from the state dict and `.mdlus`, and Triton loads only through importlib.

**Tests:**
- Whole ISLA directory, CPU on LHR: 362 passed, 100 skipped, 0 failed.
- GB300 geometry, tree-read, geometry-fast, degenerate and release files: 241 passed, 1 skipped (a sample-file test), both
  forced and under auto.
- Recipe tests on LHR: 521 passed, 12 skipped, 2 failed in `test_campaign_c_hooks.py`.
  - The failure is `TypeError` at `int(os.environ.get("SLURM_NPROCS"))`, from the sbatch environment.
  - It fails the same way at both parents, in the same job.
  - It passes with `SLURM_*` unset (523 passed, 12 skipped), and locally (523 passed, 12 skipped).

**Timing for the book** (365898, GEO-FAST's final2 protocol, mean of passes 1 and 2; `lhr/book_table.md`):

| case | geometry: base / ref / fast [ms] | step: base / ref / fast [ms] | step ratio ref / fast |
|---|--:|--:|--:|
| AirfRANS 2-D | 11.7 / 29.9 / 30.1 | 365 / 386 / 425* | 1.06 / 1.16* |
| lug | 7.9 / 44.0 / 36.7 | 141 / 179 / 171 | 1.27 / 1.21 |
| AeroZoo | 18.1 / 103.0 / 83.0 | 505 / 592 / 579 | 1.17 / 1.15 |
| DrivAerML vehicle | 98.0 / 369.5 / 307.7 | 568 / 861 / 787 | 1.52 / 1.39 |
| DrivAerML whole boundary | 2,420.7 / 431.4 / 353.7 | 2,897 / 928 / 843 | 0.32 / 0.29 |

- \* AirfRANS fast pass 2 is an outlier: its step is 458 ms, against 392 and 386 ms in passes 1 and 3, with the same 29-31 ms
  geometry call. 2-D runs the same segment search under both kernels. Passes 1 and 3 give 389 ms (1.07).
- This job's base is within 2% of GEO-FAST's 0.9.0 base on every case. The table with every pass and GEO-FAST's base beside it
  is `lhr/timing_table.md`.

**For the book.**
- `tbl-using-geometry-cost` (06-using-isla.qmd) can take these columns, with "fast" as the default on GB300.
- With fast as the default, the fast path cuts the geometry call by 17 to 19% in 3-D and the step by 2 to 9%.
- The "In flight" callout on the fast path can then go.
- I recommend the AirfRANS fast step read 389 ms (passes 1 and 3), or GEO-FAST's 389 / 382.

**Rulings relied on.**
- Peter's ruling of 2026-10-23 17:08 on the fast path, defaulting where Triton is installed.
- The brief's no-version-change reasoning.
- LANE_RULES on cross-version op renaming, suites on LHR, and registry rows.

Nothing new for project memory.

**Left.** The supervisor merges `psharpe/isla-geo-fast` into psharpe/isla.

**On disk:**
- The worktree `/home/psharpe/gh/physicsnemo-isla-geo-fast` (GEO-FAST's, about 1 GB, clean, pushed).
- On LHR: `$T/code_fastmerge` (57 MB), `$T/recipe_fastmerge` (1.4 MB) and `$T/kernel/fastmerge` (2.3 MB).
- Scratch: 2.5 MB.
- The Pi file tools reached me only through the dynamic-tool route after the bridge loss; the native tools wrote nothing.
