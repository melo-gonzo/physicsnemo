Status: DONE (TEST-BUILD; phases 1 and 2 including P4; written 08:51 EDT 10-25)

**Branches (all pushed with plain pushes, none merged; base b86c7391d = ISLA 0.13.0).**
- `psharpe/isla-tests-p0` **118268a77**: P0, the A/B harness and module goldens, plus the model goldens and P1 (MOD-008a/b/c).
- `psharpe/isla-tests-p2` **448eb9d92**: P2, properties.
- `psharpe/isla-tests-p3` **a4a6c485b**: P3 and its deferred items.
- `psharpe/isla-tests-p4` **9d8ff72f7**: p0 + merges of p2 and p3 (cec65a1a1), then P4.
- **I recommend merging p4 alone**, since it contains the other three. The order p0, p2, p3, then p4 also merges clean.
- Every branch changes only `test/experimental/models/isla/`. All commits are `-s` with hooks on and no AI trailers (verified).

**Tests added** (CPU times: the 380678 junit, LHR 16 cores; GB300: 380504). Detail: REPORT_W1/W2/W3.md, REPORT_P4_final.md.

| item | files | ids | CPU / GB300 |
|---|---|---|---|
| P0 harness and goldens | goldens/{cases,record,compare}.py, test_isla_goldens.py: 94 cases (12 groups); diff +4,384/−216 with P1 | 276 | 4.9 s / 48.5 s |
| P1, MOD-008a/b/c | test_isla_release.py: 3 configurations, every argument and attribute, f64/rich/2-D/draw/query_chunks goldens, isla_dtt_v1.mdlus, version refusals, one `--regenerate` | 79 | 2.5 s / 6.8 s |
| P2, properties | test_isla_dtt_contracts.py plus edits in 12 files (rtol 0, live reads, renumbering, chunks with the draw, parity, pose, raises); +800/−47 | +34 | 2.3 s for the new file |
| P3 gradcheck and opcheck | test_isla_gradcheck.py | 47 | 5.0 s / 18 s |
| P3 kernels against references | test_isla_kernel_refs.py and the 2-D cases in geometry_fast | 34 (+2-D) | 0.1 s / 30 s |
| P3 determinism | test_isla_determinism.py (draw, kernels, reference read, whole model: surface and DTT, 3-D and 2-D, fp32 and bf16) | 27 | 2.7 s / 31.5 s |
| P3 forced-path guards | _forced.py (sys.monitoring), test_isla_forced_paths.py, 6 guards in tree_read; +2,448/−64 in all | 21 | 1.2 s / 1.5 s |
| P4 | _marks.py; slow marks; _geometry/_kernel/_release_cases.py; 26 duplicate ids deleted; TF32 restore; launchers in `$T/kernel/isla_tests/` | — | — |

**Golden data:** `data/` totals 772,210 B, against the 1 MB cap. The base had 240,239 B. `goldens_v1.pt` is 345,599 B.

**Suite counts** (`VERIFY_FINAL.md`, `JOBS.md`; 0 failed in every run):

| run | job | passed / skipped / xfailed | wall |
|---|---|---|---|
| base b86c7391d, CPU | 380457 | 453 / 240 / 2 | — |
| p0+p2+p3 (cec65a1a1), CPU | 380678 | 726 / 473 / 6 | — |
| p0+p2+p3 (cec65a1a1), GB300 | 380504 | 1213 / 1 / 7 | 8:19 |
| p4 head, full CPU tier | 381812 | 704 / 472 / 6 | — |
| p4 head, fast tier `-m "not slow"` | 381881 | 687 passed | 129 s |
| p4 head, GB300 tier | 381934 | 1191 / 0 / 7 | 7:41 |

- The collected ids add up exactly: base 695 + P0 341 + P2 34 + P3 151 − 16 = 1205.
- P4 deletes 26 duplicate ids and adds 3 fast-tier cases.
- GPU use: 10.01 of the 11 GPU-h the supervisor raised the cap to at 02:46.

**Bugs found** (`BUGS.md`, all reported to you, none fixed; each pinned by a test):
- W3-B1: in 3-D edge regions the gradient of `u` is wrong (detached weights; strict xfail).
- W3-B2: the Triton segment-softmax read drops `logm`'s gradient silently.
- W1-B1: the eager segment-softmax read sums its bf16 gradients in bf16; dv is 29 % off.
- W2-1: fp16 autocast on CUDA raises in the fused geo region.
- W2-2: fp16 autocast on CPU gives NaN when tokens are padded.
- Core gap: `Module.load` ignores a checkpoint's metadata version (strict xfail).
- Note, not filed: mixed devices are refused only by torch's own error.

**Reviews:** each branch was reviewed and its findings fixed (REVIEW_P0, P0_2, P2, P3, P3_2, P4). Every final verdict is approve, except P2: its one-line block was fixed and I checked the diff. A verifier ran the finals (VERIFY_FINAL.md).

**Rulings I relied on:**
- the supervisor's decisions 1-5;
- the 1 MB data cap;
- the 11 GPU-h cap;
- the 23:57 base and quad-diagonal steer;
- the 02:46 and 03:28 notes on GEO-TIEORDER and DRAW-EXACT. Tests tied to them carry the literal tags (`git grep -n 'DRAW-EXACT\|GEO-TIEORDER' test/experimental/models/isla`); they are listed in REPORT_W1/W2/W3.md.

**For project memory:**
- The ISLA GPU tier, and the LANE_RULES "GPU tests" run, is `$T/kernel/isla_tests/submit.sh <worktree> gpu|full|fast <TAG>`. Its README is there, and the master copy is in `lhr/isla_tests/`. Its jobs are named `isla.testbuild.*`, and I have removed that registry row. A lane that uses the launcher sets `PREFIX` or adds its own row.
- ISLA's CPU CI is `-m "not slow"`.
- Process note: the writers' `contact_supervisor` messages never reached me as messages, only as timeouts. I worked from their files.

**Left:**
- The supervisor's merges.
- The 0.14.0 assembly regenerates the DRAW-EXACT-tagged goldens with `python -m test.experimental.models.isla.goldens.record --regenerate goldens_v1.pt` (or the named file).
- TEST_PLAN's LIB-MOVE P5 row still names the deleted `test_fused_segment_softmax_read_matches_eager`. Its replacement is `test_isla_kernel_refs.py::test_segment_softmax_read_obeys_the_2x_error_rule`.
- The recipe's `test_campaign_c_hooks.py` was left alone (it is outside the ISLA directory).
- Nothing is running.

**On disk:**
- Worktrees `/home/psharpe/gh/physicsnemo-isla-tests-{p0,p2,p3,p4}`, about 1.1 GB each, all clean. Remove them after the merge.
- This folder, 5.5 MB.
- On LHR:
  - the final-suite snapshots `$T/code_testbuild_v-*` (7);
  - `$T/kernel/isla_tests/snap/p4fix-*`;
  - logs in `$T/kernel/testbuild/out` (506 MB).
  - I deleted the development snapshots and tars.
