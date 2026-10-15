#!/bin/bash
# AF-SCORE launcher (LHR, 2026-10-15): score afu_* lanes under the native / area / posed protocols with score.sbatch.
# usage: bash kernel/airfrans_u/score_submit.sh <run_id> [<run_id> ...]
# One job per run, named isla.afu.score.<slug> (afu_r_isla_seed42 -> isla.afu.score.r-isla-seed42). Idempotent: a run
# whose three protocols all have a .done marker, or that has a job of that name queued or running, is skipped. A run
# whose train.log lacks "Training completed" is a no-op inside the job (no .done), so re-running this retries it.
# Results: $T/evals_afu/<run>/<protocol>/<run>/metrics.jsonl.
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
AFU="$T/kernel/airfrans_u"; LANES="$AFU/lanes.tsv"; ROOT="$T/evals_afu"
[ $# -ge 1 ] || { echo "usage: $0 <run_id>..." >&2; exit 2; }
mkdir -p "$AFU/sbatch_logs" "$ROOT"
queued=$(squeue -u "$USER" -h -o '%j')
for run in "$@"; do
  awk -F"\t" -v r="$run" 'NR>1 && $3==r {f=1} END {exit !f}' "$LANES" || { echo "skip ${run}: not in ${LANES}" >&2; continue; }
  n=0; for p in native area posed; do [ -f "$ROOT/$run/$p/.done" ] && n=$((n + 1)); done
  [ "$n" -eq 3 ] && { echo "skip ${run}: scored (3/3 .done)"; continue; }
  slug=$(sed -e 's/^afu_//' -e 's/_/-/g' <<<"$run"); name="isla.afu.score.${slug}"
  grep -qxF "$name" <<<"$queued" && { echo "skip ${run}: ${name} already queued or running"; continue; }
  jid=$(sbatch --parsable -J "$name" --export=ALL,AFU_SCORE_RUN="$run" "$AFU/score.sbatch")
  echo "submitted ${jid} ${name} (${n}/3 protocols done before)"
done
