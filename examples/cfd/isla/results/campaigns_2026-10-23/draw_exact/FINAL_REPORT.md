Status: DONE

The token draw is now bitwise the same from run to run on a GB300, and bitwise the same on CPU and GPU given the same float64 inputs. Branch `psharpe/isla-draw-exact`, head **5b5e72c2f**, pushed with a plain push and not merged. No version bump; Peter decides adoption at the 0.14.0 freeze.

**What ran.** All LHR jobs had the five bad nodes excluded. Registry row `^isla\.drawexact\.` was added, then removed at 03:23 EDT. Total 3.20 of 4 GPU-h.
- 377433: design probe over 18 real cases.
- 377520: Inductor debug.
- 378082: round-1 suite, det_probe, head_check.
- 378518: the new tests run on the base.
- 378892: final GB300 suite, det_probe, head_check.
- 378083, 378832, 378893: cpu-short suites.

Commands, md5s and the snapshot audit are in `JOBS.md`.

**Commits:**
- 7a3f91f20: exact int64 line.
- 18fdcd7c0: the tail sums in a fixed-order float scan, and pi = 1 when share ≥ n.
- 02d5451fd: merge of psharpe/isla b86c7391d (ISLA 0.13.0), with one docstring conflict.
- 5b5e72c2f: tests only.

Children: three writers, one merge writer, and lane-reviewer runs on the design (BLOCK, fixed), on diff round 1 (BLOCK, fixed) and on diff round 2 (OK; notes fixed).

**Options** (18 real cases, budget 200,000; full table and text in `DESIGN.md`):

| option | 6 runs bitwise | CPU = GPU | kept cells changed vs today's CPU (eval start / 544 random draws) | DrivAerML draw (ms) | host sync |
|---|---|---|---|---|---|
| today (ATen float cumsum) | 14/18, up to 9 cells, **duplicates** | 12/18 | 0 / 0 (reference) | 19.4 | none |
| A: exact int64 fixed point | 18/18 | 18/18 | 5 / 27 | 21.1 | none |
| B: CPU cumsum | 18/18 | 18/18 | 0 / 0 | 45.6 | 2 or more per draw |
| C: fixed-order Triton scan | 18/18 | 13/18 | 5 / 27 | 20.4 | none |
| D: ATen deterministic scan, toggled | 18/18 | 13/18 | 5 / 27 | 19.8 | none (global state) |

**Recommendation and why.** I recommend, and built, A for the line plus a fixed-order float scan for the tail sums (G):
- The line in exact int64 is the only option that gives the draw's invariants by construction: each pi = 1 cell is kept exactly once, no cell is drawn twice, and a hit on a boundary takes the later cell. The result is the same in any order, on any device and under any compiler, with no sync, for about 1 to 3 ms.
- The tail sums are a fixed-order (Kogge-Stone) float scan, with an explicit pi = 1 when share ≥ n. The reviewer showed that fixed point is wrong for the tail sums: tiny measures gave negative pi, and the precision was relative to the condition's total rather than to the tail. Elementwise IEEE adds give the same bits on CPU and CUDA, at about log2 n ulps.
- B is the only option with no function change, but it costs +26 ms and a sync per forward to keep a float line that is itself wrong at exact ties.

**Determinism before and after:**
- The mechanism is exact ties at the eval start of 0.5. AeroZoo's water fill gives half-integer shares; case 0 has 31250, 58315.5, 52119 and 58315.5. So hits land exactly on condition boundaries and on every cell boundary of a condition kept whole.
- Before, on GPU, case 0 changed by up to 4 kept cells over 6 runs and case 050 by up to 9. Today's GPU draw also **duplicates cells and drops pi = 1 cells**: 0 to 4 per run in case 0, 1 to 13 in case 050. This is shown on case 0's own boundary meshes, read raw rather than through the recipe.
- After (378892): 6 runs bitwise on all 6 real boundaries checked, 0 duplicates, every pi = 1 cell kept, CPU = GPU bitwise, and compiled (fullgraph, dynamic) equal to eager bitwise. det_probe `dtt` is bitwise at every stage in 5 of 5 runs; on the base, 3 of 5 runs differed in the weights. `dtt_ref` still differs from `read0`: that is the reference tree read's index_add, which is out of scope.

**Function change and cost:**
- Against today's CPU, in the round-1 probe: 1 kept cell of 200,000 in 5 of 18 eval draws, and 27 cells over 544 random draws. That is today's own CPU float error, the same size as today's CPU-to-GPU gap. On the final head (head_check, 6 cases × 9 starts) the eval draws changed one cell in 5 of the 6 cases and none in case 050.
- Weights: unchanged in meaning (a/pi); the float64 values change by ulps.
- Effect on outputs: one changed token moved AeroZoo case 0's outputs by 2.8e-6 to 8.8e-6 (REYNOLDS-OUT).
- Cost: DrivAerML draw 21 to 22 ms against 18 to 19 ms, and AeroZoo 8.1 to 8.9 ms against 6.2 to 8.8 ms. These compare different nodes, so they are approximate. Peak memory is unchanged.

**Suites:**
- CPU (378893, head 5b5e72c2f): 466 passed, 245 skipped, 2 xfailed, 0 failed.
- GB300 (378892), every CUDA test: 716 passed, 1 skipped, 2 xfailed, 0 failed.
- The new CUDA determinism and CPU = CUDA tests fail on the base (378518).
- pre-commit on the changed files passed, and every commit ran with hooks on.

**For the supervisor and Peter:**
- (1) Adoption of the function change at the 0.14.0 freeze is Peter's call.
- (2) Today's code has two defects that this branch fixes:
  - GPU evaluation draws on AeroZoo keep duplicate tokens and drop whole cells wherever the shares are half-integers.
  - With equal measures and share = n, pi comes out at 1 minus a few ulps (26 of 244 cases), so a condition kept whole is not exactly whole.
- (3) If TEST-BUILD's p3 branch carries an `xfail(strict)` test for draw determinism, it will XPASS after this merges and needs flipping.
- (4) `ISSUE_DRAFT_INDUCTOR.md` is a draft PyTorch issue about an int64 constant times an int32 index in Triton. It is for Peter and has not been filed.
- (5) For LIB-MOVE row 9: `budget` must be a Python int; a condition whose cells are mostly exactly 0 gets sum pi = n, as on the base; the op name `isla_cumsum` is unchanged.
- (6) Snapshot audit: `code_drawexact_base` and `code_drawexact_head_7a3f91f` are short. Every suite that counts was rerun on complete snapshots, checked file by file with md5: base2, head_02d5451 and head_5b5e72c.

**Rulings relied on:** the brief; LANE_RULES (cap, registry, GB300 suite, no version bump, merge and never rebase); the supervisor's 02:32 merge steer and 02:52 snapshot steer.

**For the book:** the eval-time draw is now exact. The AeroZoo GPU scores so far carry the duplicate-token defect, which is too small to matter: a token-level change moves outputs by about 1e-5.

**On disk:** worktree `/home/psharpe/gh/physicsnemo-isla-draw-exact` (1.1 GB; remove it after the merge), and on LHR `$T/code_drawexact_*` (5 read-only snapshots). Nothing is running.
