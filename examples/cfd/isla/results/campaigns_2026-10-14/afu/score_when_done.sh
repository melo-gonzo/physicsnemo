#!/bin/bash
# AF-SCORE, one shot (LHR, 2026-10-15): submit scoring for every afu_* lane of kernel/airfrans_u/lanes.tsv whose
# runs/<run>/train.log has "Training completed" and that lacks a .done for any of its three protocols (score_submit.sh
# skips a run that is scored or whose scoring job is queued or running). It does not loop; run it again, or arm it, e.g.
#   ssh lhr2 'bash $T/kernel/airfrans_u/score_when_done.sh'
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
AFU="$T/kernel/airfrans_u"; ROOT="$T/evals_afu"
ready=(); pending=()
for run in $(awk -F"\t" 'NR>1 {print $3}' "$AFU/lanes.tsv"); do
  if ! grep -q "Training completed" "$T/runs/$run/train.log" 2>/dev/null; then pending+=("$run"); continue; fi
  n=0; for p in native area posed; do [ -f "$ROOT/$run/$p/.done" ] && n=$((n + 1)); done
  [ "$n" -eq 3 ] || ready+=("$run")
done
echo "$(date -Is) training: ${#pending[@]} lane(s) not finished${pending[*]:+ (${pending[*]})}"
[ ${#ready[@]} -gt 0 ] || { echo "nothing to score"; exit 0; }
bash "$AFU/score_submit.sh" "${ready[@]}"
