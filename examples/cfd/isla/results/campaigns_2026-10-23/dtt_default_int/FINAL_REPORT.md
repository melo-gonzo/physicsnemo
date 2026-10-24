Status: DONE

**Commits** (branch psharpe/isla-default-dtt, pushed plainly e10681b33..f9b0014c8; not merged into psharpe/isla; `-s`, hooks on):
- 616992605: `git merge --no-ff --signoff origin/psharpe/isla-geo-band` (3f3fef69f). No conflicts. I reworded the default message before pushing, to the repo's style: "Merge psharpe/isla-geo-band (3f3fef69f): ... as ISLA 0.12.0".
- f9b0014c8 (head): the test fix. It changes test_isla_release.py and deletes the two eba038f08 fixture files.

**How the fixture test reads now.** I deleted the 0.11.0 fixture and its _output.pth, and the test that read them.
- Why delete: two existing refusal tests already cover the case. A probe on the real fixture (fixture_refusal_probe.log) gets exactly the strings those tests match:
  - `test_isla_checkpoint_of_an_earlier_version_is_refused[0.11.0]` matches the error from `Module.from_checkpoint`: "version 0.11.0 is not compatible with current version 0.12.0".
  - `test_isla_0_7_0_to_0_11_0_state_dicts_are_refused_on_every_loading_path[0.11.0-11]` matches the error from `Module.load(strict)`: "before version 0.12.0 (state-dict version 11".
- The old proof now runs at 0.12.0, inside DEFAULT-DTT's own `test_new_interior_checkpoints_store_their_form`. Its `interacting` case builds a model in the test with `query_independent=False`, saves it, and reloads it. A second test would have repeated that one, so I extended it instead. It now also asserts:
  - `set(args) == set(inspect.signature(ISLA.__init__).parameters) - {"self"}`: every constructor argument is stored;
  - `args["interior"] is True`;
  - kernel_basis and kernel_reads are None exactly when the form is interacting.
  - It already checked that query_independent is stored as False, that the model reloads as interacting (routing mass present, no boundary read), and that outputs equal the saved model's with atol = rtol = 0. The old test compared against a recorded forward at 1e-5 (CPU) and 1e-4 (CUDA), so the check is stricter, not weaker.
  - Its docstring names the two refusal tests.
- Assertion audit over the whole directory (assert_before/after/diff.txt; before is 616992605): 702 lines before, 697 after.
  - The 7 removed asserts are all from the deleted test, and assert_diff.txt maps each one to its replacement.
  - 1 line changed (it now also checks `interior`) and 2 were added.
  - No message or match string changed, no tolerance was weakened, and no other test was deleted.

**Research script.** I left it unchanged, because your premise does not hold. examples/cfd/isla/research/qi_global/cluster/model_bench.py already passes `query_independent=True` (line 42), unchanged since its only commit, fdf0475f9. QI-GLOBAL's REPORT.md "Cost" section shows that it benchmarked read-only against the global read, both query-independent; prof.sbatch runs it as `ro` and `global`.
- It cannot run on current ISLA in any case: it passes `query_schema`, `geo_checkpoint` and `kernel_read*`, which are no longer constructor arguments, so it raises TypeError.
- No other research script constructs an interior ISLA directly.

**Suites** (LHR, snapshots $T/code_dttdefint_f9b0 and recipe_dttdefint_f9b0 of f9b0014c8; I read every log, all in logs/ here):

| Suite | This head | DEFAULT-DTT | BAND-MERGE / GEO-BAND |
|---|---|---|---|
| ISLA dir, CPU, 370472 | 414 passed, 132 skipped, 0 failed | 401 | 411 |
| ISLA dir, GB300, every CUDA test, 370473 g0 | 553 passed, 1 skipped, 0 failed | 531 + 1 skip (g2, same job) | 550 + 1 skip (g1, GEO-BAND 3f3fef69f, same job) |
| Recipe tests, cpu-short, 370615 | 585 passed, 15 skipped, 2 failed | 585 / 2 failed | n/a |

- The counts add up exactly: 397 + 4 + 14 − 1 = 414 on CPU, and 526 + 5 + 24 − 2 = 553 on the GB300. The deleted test was 1 case on CPU and 2 on GPU.
- The 2 recipe failures are the test_campaign_c_hooks initialize_from_checkpoint tests. Their 4 `E` lines, `TypeError: int() ... 'NoneType'`, match DEFAULT-DTT's job 370095 line for line.
- The only GB300 skip is ISLA_AF_POSED_SAMPLE. The CPU skips are all CUDA, Triton, or that same sample.
- The GB300 job ran on nvl72d039-T13 with MD5SUMS-OK, and each pytest printed the model.py it ran. GPU use was 2.07 of the 3 GPU-h cap.

**For you to decide / note.**
1. Several research probes build their model by composing the recipe's infer config: qi_kernel/bench_step.py and prec_probe.py; query_independence/{qi,support,full_field,native_block}_probe.py and posed_diag.py; routing_mass/probe_routing_mass.py. Without `++model.query_independent=false` they now get ISLA-DTT. For example, full_field_probe.py adds the override only for its QI arm. All of them read checkpoints from before 0.12.0, which are refused anyway, so I left them unchanged.
2. The DEFAULT-DTT report's "Checkpoint loading" paragraph says an old interior checkpoint loads as interacting. That no longer holds after this merge: it is refused, with its version named. Any book or ledger text drawn from that paragraph should say so.

**Left / on disk.** Nothing is running, and I removed the registry row (backup watchers/.ACTIVE_EXPERIMENTS.bak.dttdefint_rm).
- On disk: the worktree /home/psharpe/gh/physicsnemo-isla-default-dtt, clean at f9b0014c8, to remove after your merge; this folder (0.4 MB, JOBS.md has the full record); and on LHR the read-only snapshots code_dttdefint_{f9b0,3f3f} (57–58 MB each), recipe_dttdefint_f9b0 (1.5 MB) and kernel/dttdefint.
