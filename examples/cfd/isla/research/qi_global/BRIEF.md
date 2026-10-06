# Lane `isla-qi-global`: an untruncated, exact boundary read for read-only ISLA

Worktree/cwd: `/home/psharpe/gh/physicsnemo-isla-qi-global` (branch
`psharpe/isla-qi-global`, based on `psharpe/isla-qi-kernel` @ bb877be65, which holds the
truncated kernel read). Research folder: `examples/cfd/isla/research/qi_global/`. AGA task
dir: `/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global/`.
Job prefix `isla.global.`.

## Read first
- `examples/cfd/isla/research/qi_kernel/REPORT.md`: the truncated read (R = 0.01 L), with
  its design, tests, cost, final results, band anatomy and the bf16/float32 arithmetic
  finding. Its section "How to read this lane" says why it is not a recommendation.
- `examples/cfd/isla/research/query_independence/RESULTS.md`: the campaign that set the
  question.
- The code and tests:
  - `physicsnemo/experimental/models/isla/kernel_read.py` and `pair_kernels.py`;
  - the wiring in `model.py`;
  - `test/experimental/models/isla/test_isla_kernel_read.py`.

Official results of the truncated read (float32 protocol, seed means). Ratio = arm error /
reference error, so a ratio below 1 means the arm is more accurate. Fields are pressure /
velocity / eddy viscosity.

| arm and dataset | vs interacting | vs read-only |
|---|---|---|
| AirfRANS, truncated read | 0.795 / 0.995 / 1.239 | 0.766 / 0.915 / 0.874 |
| DrivAerML, truncated read | 1.162 / 1.317 / 1.202 | |

On DrivAerML the bf16-scored ratio vs interacting is 1.051 / 0.954 / 1.301. The
eddy-viscosity excess sits in the wake, more than 1 L from the wall, and a 0.01 L read
cannot reach it.

## Question
An untruncated boundary read has every prediction point read every boundary cell, with no
length scale anywhere. Does it close read-only ISLA's wake eddy-viscosity gap on AirfRANS,
while keeping exact query independence and every ISLA invariant?

## Principles (Peter's requirements for this lane)
- **No chosen length scale.**
  - No radius, no falloff tied to a constant and no neighbour count.
  - Any locality must be learned from scale-free invariants: lengths in units of L, or
    relative to the point's own geometry.
- **Classify every constant you add, and justify it in REPORT.md.** A constant must be a
  numerical tolerance (fine) or a capacity knob (fine, but name it). A modeling scale fitted
  to the data is not allowed. R = 0.01 L was of that third kind.
- **Exact evaluation: all pairs, no approximation.**
  - Do not use a tree code (Barnes-Hut or dual-tree). Peter notes that tree aggregation of
    mixed-sign contributions (dipole and higher-order influences, differences of large
    quantities) can have unboundedly large error, and GLOBE never solved this.
  - AirfRANS has about 1,000 boundary cells, so all pairs (about 1e8 per block at 100k
    points) are affordable.
- **Discretization consistency:** a quadrature over the boundary (HT-corrected measure
  fractions), exactly invariant under refinement.
- **Invariants:** exact query independence; rotation, reflection and translation
  covariance; the similarity gauge.

## Design (a starting point; you may improve it if you justify the change)
- **Global mode.** Reuse `kernel_read.py`'s machinery: the pair invariants, the shared basis
  MLP, the per-head values and the zero-initialized output. Add a global mode (for example
  `kernel_read_radius=None`) that reads every cell with no falloff.
- **Pair invariants without R:**
  - `r/L` and `log(r/L)`;
  - the self-similar near-wall invariant `log(r / |d_q|)`, where `d_q` is the point's
    built-in wall distance from the class's nearest-boundary geometry. Allow only a
    numerical guard, and state it;
  - the cosines, as before.
- **Form.** Make the `"mean"` form primary. It is a softmax over all cells with logits =
  learned kernel + q.k + log measure fraction, which is ISLA's own quadrature form. Logits
  linear in log r give power laws, so it can represent any locality. The `"sum"` form is
  optional and secondary.
- **Placement.** Read at every block (matched depth, as the slices are) if the cost allows.
  Otherwise use the last k blocks and state it as a capacity compromise, with its cost.
- **Implementation.** For a boundary of about 1,000 cells:
  - use dense all-pairs tiles (chunked over queries, batched matmuls), not ragged gathers;
  - recompute in backward if memory requires it;
  - measure step time and peak memory at the protocol size before choosing the placement.
- **Defaults unchanged.** With the option off, every existing test and golden passes.

## Tests (float64, before any AGA lane)
These are the kernel lane's set:
- exact query independence: alone, prefix, permuted, chunked;
- rotation, reflection and translation covariance in 2-D and 3-D;
- the similarity gauge;
- boundary refinement: every cell as two coincident half-measure copies, exact;
- measure scale;
- every parameter receives a gradient (DDP);
- bf16 finite;
- the CUDA path matches eager.

## Precision
The kernel lane found that bf16-trained models can depend on bf16 rounding in the token
embeddings: on DrivAerML, float32 scoring is much worse than bf16 scoring.
- Score every arm, and the references, in both float32 (the protocol) and bf16, and compare
  like with like. The kernel lane's re-scoring and probe scripts are in its AGA task dir.
- The verdict uses float32; report bf16 beside it.
- Do not change the training precision.

## Experiments (AirfRANS only)
Protocol: dataset `airfrans_volume_2d`, model `isla_volume_airfrans`, cells 100000, extra
`train_split=full_train val_split=full_test +training.val_interval=10`. lr 5e-4, 500
epochs, bf16, no augmentation, compiled blocks. AirfRANS eddy viscosity varies 10-15%
between runs, so every arm gets four seeds (42 to 45):

| arm | what | seeds |
|---|---|---|
| G | read-only + global read (the test) | 42-45, all new |
| I | interacting, 256 slices | 44 and 45 new; 42 and 43 from the QI campaign |
| P | read-only, 256 slices | 44 and 45 new; 42 and 43 from the QI campaign |
| T | truncated read, R = 0.01 L (the kernel lane's design) | 44 and 45 new; 42 and 43 from the kernel lane |

- The new I and P lanes run this lane's snapshot with the read off.
- The QI references ran an older snapshot (`code_qi2`). Confirm that the default paths are
  identical (the goldens) and say so in the report.
- Order: smoke and cost probe, then the bars in REPORT.md, then launch the full lanes.

## Bars (write them into REPORT.md before launching)
Use four-seed means of the float32 protocol scores, and report the standard error and the
paired ratios.
- **Closes the gap:** all of the following hold.
  - G / I <= 1.03 on pressure and on velocity;
  - G / I <= 1.05 on eddy viscosity;
  - in the wake band (|sdf| > 1 L), G's eddy-viscosity error is within 10% of I's.
- **Helps the wake:** G's eddy-viscosity excess over I (ratio - 1) is at most half of T's,
  and the wake-band error falls with it.
- **No wake effect:** G is within one standard error of T on eddy viscosity.

Also report G / P (what the global read adds to read-only), G / T (global vs truncated),
the band anatomy (error by |sdf|/L band), cost per step and peak memory.

## Out of scope unless you ask the supervisor
- DrivAerML lanes. An exact all-pairs read there is about 4e10 pairs per block. If
  AirfRANS is positive, propose a plan with its cost.
- Tree codes.
- Changing the training precision or the ISLA core (slices, routing temperature).

## Timeline and practicalities
- Targets: implementation and tests in the first ~3 h, smoke and cost by ~4 h, full lanes
  launched by ~5 h. AirfRANS lanes take about 8 h, more for G. Peter checks in tomorrow
  morning; keep REPORT.md current and commit at milestones.
- The `normal` QoS queue has been about 1,000 jobs deep. The kernel lane ran scoring,
  probes and short links under the `short` QoS with limits under an hour. Follow the
  hpc-resources skill on QoS choice.
- Templates in `/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-03-isla-qi-kernel/`:
  - `kernel/` (lanes, submit and the after-lanes watcher);
  - `eval/` (scoring, probes, reduction, bf16 re-scoring).

  Copy them; never write there. The slices lane (`isla.slices.*` jobs) may still be
  running: do not touch it.
- Rules:
  - wrap every ssh in `timeout`;
  - keep each bash call under 30 minutes;
  - start background processes through ssh only as
    `ssh -n -f aga1 'setsid nohup CMD > LOG 2>&1 < /dev/null &'`, and prefer Slurm jobs.

---

The shared context below was written for the first three lanes on 2026-10-04. It still
applies, with two updates: you are a fourth lane, and Peter checks in tomorrow morning
rather than in 10 hours.

# Shared context (all three lanes)

You are one of three independent research lanes Peter Sharpe (the ISLA author) asked for. The other two
lanes run in parallel in their own worktrees and AGA task directories; never touch their files, branches,
task dirs or jobs. Peter will ask for a status report in about 10 hours: keep your REPORT.md current.

## The project
- ISLA (Invariant Slice Attention): PhysicsNeMo experimental model in
  `physicsnemo/experimental/models/isla/` (model.py, blocks.py, geometry.py, heads.py, inputs.py,
  geo_kernel.py = fused Triton geometry kernel). Architecture explained in
  `examples/cfd/isla/book/02-isla.qmd` and `04-architecture.qmd`. Principles: no coordinate reaches a
  learned map (only invariants of relative geometry); every sum over the boundary is a quadrature
  (log cell-measure bias in the softmax over points, Horvitz-Thompson-corrected `_effective_measure`);
  outputs re-assembled from an equivariant basis; exact rotation/reflection/translation covariance,
  similarity gauge (length unit L = measure-weighted RMS pairwise distance), discretization consistency.
- READ FIRST: `examples/cfd/isla/research/query_independence/RESULTS.md` and `PREREG.md` in your
  worktree (the campaign that produced this question), then the model code.
- Interior task = the volume task (`interior=True`): prediction points join the encoder as tokens.
  Default (interacting): each prediction point writes into the slices with an equal share of a learned
  total weight, `q_logw = qt.logw + logsumexp(boundary log-measures) - log(Q)` in
  `ISLA._forward_tensors`. `query_independent=True`: prediction points get log-weight -inf, so they read
  every block's slices/anchors/prototypes at matched depth and never write (exactly query-independent,
  no `qt.logw` parameter). `forward(support_points=...)`: an explicit interior writer set for an
  interacting model.

## Reference numbers (reporting protocol, same code snapshot, float32 scoring on the held-out split,
## seed mean (seed 42 / seed 43), relative L2 of pressure / velocity / eddy viscosity)
- DrivAerML Volume, interacting (256 slices): 0.0404 (0.0413/0.0396) / 0.0611 (0.0625/0.0596) / 0.0868 (0.0887/0.0849)
- DrivAerML Volume, read-only `query_independent=True` (256 slices): 0.0448 (0.0449/0.0447) / 0.0748 (0.0750/0.0747) / 0.1020 (0.1046/0.0993); ratio 1.108 / 1.225 / 1.174
- AirfRANS, interacting: 0.0332 (0.0334/0.0331) / 0.0164 (0.0165/0.0163) / 0.0297 (0.0303/0.0292)
- AirfRANS, read-only: 0.0345 (0.0357/0.0332) / 0.0179 (0.0179/0.0178) / 0.0422 (0.0450/0.0393); ratio 1.037 / 1.088 / 1.417
- Per-lane scorer rows: `examples/cfd/isla/results/qi_2026-10-03/score/<run>/metrics.jsonl`; reduction
  `examples/cfd/isla/results/qi_2026-10-03/qi_reduction_2026-10-03.json`. Read-only's excess on DrivAerML
  sits near the wall (|sdf| < 0.01 L, 72% of points: velocity 1.25x) and in the far field.
- AirfRANS caveat: its interior points are stored in a spatially coherent order and the reader takes one
  contiguous cyclic block, so every AirfRANS request (training and scoring) is a spatial chunk of ~56% of
  the domain. DrivAerML's stored order is shuffled. For read-only arms this only changes which points are
  supervised/scored; for interacting arms it shapes the slices.

## Your worktree and git rules
- Work only inside your worktree (your cwd); it is a git worktree on its own branch, based on
  `psharpe/isla-independent` @ 84b0a4190. Commit with the configured identity (Peter Sharpe). Never add
  AI co-author trailers or "generated with" lines. Never push, rebase or force-push. Small commits.
- The pre-commit hook's ruff-format step reformats whole pre-existing unformatted files (model.py,
  test_isla.py, test_isla_kernels.py are not ruff-formatted on this branch). Run
  `/home/psharpe/gh/physicsnemo/.venv/bin/ruff check <changed files>`; if the only hook failure is
  ruff-format touching pre-existing code, restore those files to your staged version and commit with
  `--no-verify`. Markdown lines <= 88 characters (markdownlint).
- Local Python: `/home/psharpe/gh/physicsnemo/.venv/bin/python`, run from your worktree root so
  `physicsnemo` imports from your worktree (check `python -c "import physicsnemo; print(physicsnemo.__file__)"`).
  Tests: `python -m pytest -q -p no:warnings test/experimental/models/isla/` (170 pass at base, ~5 min;
  CUDA tests use the shared local RTX 4090 laptop GPU, 16 GB; keep local GPU use light).

## AGA cluster (GB300 Slurm)
- First read `/home/psharpe/.pi/agent/skills/hpc-resources/SKILL.md`, its `references/aga.md` and
  `references/slurm-monitoring.md`, and follow them. `ssh aga1` (or `aga2`). Account
  `coreai_modulus_cae`; `-p batch -q normal -N 1 --gpus-per-node=4 --segment=1`; batch wall <= 4 h, so
  long trainings use chained links. Peter authorized whatever AGA resources the experiment needs (stay
  within the skill's GPU budget). The queue currently has ~50 idle nodes and ~900 pending jobs, so expect
  some waiting.
- QI campaign task dir (READ-ONLY templates and checkpoints; do not write there):
  `/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-02-isla-independent/`
  - `kernel/lanes.sbatch` (TSV-driven 3h59 singleton chain links; each link submits its successor, resumes
    from the newest intact checkpoint via `kernel/ckpt_preflight.py`, BLOCKED guards incl. three
    fast-failing links), `kernel/submit.sh`, `kernel/lanes.tsv` (columns idx group run_id dataset model
    extra lr seed cells code), `eval/eval.sbatch` (infer.py scoring + `qi_probe.py`), `eval/qi_reduce.py`,
    `eval/support_probe.py`, `eval/full_field_probe.py`, `recipe_qi/` (recipe snapshot with dataset configs
    `drivaer_ml_volume_reference_sdf`, `airfrans_volume_2d`, `airfrans_volume_2d_all`, `*_400k`;
    `datasets/dataset_paths.yaml` addresses DrivAerML through /scratch/fsw), `code_qi2/` (model snapshot =
    this branch's model code), `runs/qi_{drvol,af}_{int,pas}_seed{42,43}/checkpoints` (epoch 500).
- Make your own task dir `/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-03-<your-slug>/`
  (write only there). Copy what you need from the QI task dir (adapt TASK paths, job-name prefix, lane
  table). Code snapshot: rsync your worktree's `physicsnemo/` + `pyproject.toml` (exclude __pycache__)
  into `code_<tag>/` with a `SNAPSHOT_HEAD` file holding your commit; recipe: copy `recipe_qi` and check
  its `src/*.py` md5s against your worktree's
  `examples/cfd/external_aerodynamics/unified_external_aero_recipe/src/`. Shared venv (read-only use):
  `/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe`,
  with `PYTHONPATH=<code dir>:<recipe>/src`. Job names `isla.<your-slug>.<lane>`, unique to you.
  Keep torch.compile/Triton caches on node-local /tmp (the defaults).
- Lane protocol (matches the references): DrivAerML Volume = dataset `drivaer_ml_volume_reference_sdf`,
  model `isla_volume_sdf`, cells 200000, extra none; AirfRANS = dataset `airfrans_volume_2d`, model
  `isla_volume_airfrans`, cells 100000, extra `train_split=full_train val_split=full_test
  +training.val_interval=10`. lr 5e-4, 500 epochs, `precision=bfloat16 augment=false compile=false`,
  seeds 42 and 43, `compile_blocks` true (config default). Model overrides go in `extra` as
  `++model.<arg>=<value>`. At 256 slices: DrivAerML ~5.2 h (two links), AirfRANS ~7.5-8 h (2-3 links),
  one 4xGB300 node per lane.
- Scoring = infer.py exactly as `eval/eval.sbatch` runs it (float32, `infer_split=val
  +save_predictions=false sampling_resolution=<cells>`, same extra overrides minus training.*). On the
  2-D AirfRANS lanes infer.py writes no summary (a spurious velocity_z key trips its range check after
  all per-case rows are written): average the per-case rows (`qi_reduce.py`'s `summary_of` does this).
- Lessons already paid for: `set -euo pipefail` with `ls | grep` on a missing dir killed a launcher
  (lanes.sbatch has the fix; reuse it); Lustre-hosted compile caches made the first compiled step 6 min;
  one node could not see /home (use /scratch/fsw paths); the VPN can drop (retry ssh with backoff; an ssh
  failure never means a job finished); run a smoke (one short link or a few epochs) before full lanes;
  the library kNN on CUDA (cuML) is not translation-robust and once aborted the CUDA context on posed
  2-D input, so use an exact chunked cdist search for any posed/covariance check of trained models.
- Monitoring: poll with `sleep` inside bash calls of at most 40 minutes, one terse line per lane
  (epoch, val loss, state); reconcile with `sacct`; check `status/BLOCKED_*`. Cancel only your own jobs.
  Before you finish, record every job id and the state of every lane.

## Reporting (required)
- Keep `examples/cfd/isla/research/<your-slug>/REPORT.md` in your worktree current: question, design,
  preregistered bars (written before launching full lanes), code commits, AGA task dir, job ids, lane
  table with status, results vs the references above, verdict, cost (step time, peak memory), open issues
  and next steps. Commit it at each milestone. Mirror a short `STATUS.md` in your AGA task dir.
- Your final message: a concise summary (verdict or current state, key numbers, where the report is,
  what is still running with job ids and expected completion).
- If a decision would change the experiment's meaning (memory cannot fit, a principled design is not
  implementable with the available data, a bar cannot be measured), use contact_supervisor with the
  concrete question, options and your recommended default; otherwise proceed autonomously.
