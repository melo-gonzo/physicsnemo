# Training-step breakdown on GB300 (PERF lane, 2026-10-18)

Compact copy of the coordinator's performance lane. Source report:
`/home/psharpe/gh/physicsnemo/scratch/isla-perf-2026-10-18/REPORT.md` (section 1, A/B
table of section 4) and its `tables_base.md` (copied here verbatim). LHR jobs are listed
in `step_breakdown.json` under `_provenance`; per-step records live on LHR under
`$T/perf_2026-10-18/runs` (`T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`).

- `step_breakdown.json`: the numbers the book reads (host timeline per warm step, GPU
  kernel time, forward module attribution, syncs, launches, memory, rank imbalance,
  the next-batch prefetch A/B).
- `tables_base.md`: the full stage breakdown, three configurations.

Configuration: one node, four GB300 ranks, DDP, bfloat16, per-block compile, AeroZoo
split `family_non_box`, 200,000 cells, seed 42, the decision lanes' training command.
The `single_input_isla` column is the single-input exact-geometry design and is quoted
only where the book describes that design.
