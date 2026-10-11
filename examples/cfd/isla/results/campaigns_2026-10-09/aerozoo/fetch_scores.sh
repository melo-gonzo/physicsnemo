#!/bin/bash
# Fetch the AEROZOO scoring records from AGA into results/campaigns_2026-10-09/aerozoo/scores/ (2026-10-10).
#   scores/<run>/val/metrics.jsonl    the fleet scorer's validation-side record (p200k_fp32/<run>/<run>/metrics.jsonl)
#   scores/<run>/test/metrics.jsonl   the test-side record (p200k_fp32_test/<run>/<run>/metrics.jsonl, eval_test.sbatch)
#   scores/<run>/train_metrics.jsonl  the lane's training record without the per-step rows (config, dataset, val_summary,
#                                     train_summary: the validation trajectory every ten epochs)
#   scores/<run>/train_tail.log       the last 700 step lines of train.log (step time and memory of the final epoch) and
#                                     the completion line
# usage: bash fetch_scores.sh        (from anywhere; idempotent, overwrites)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
RUNS=$(awk -F'\t' 'NR>1{print $3}' "${HERE}/lanes.tsv" | tr '\n' ' ')
for r in ${RUNS}; do
  mkdir -p "${HERE}/scores/${r}/val" "${HERE}/scores/${r}/test"
done
ssh -o ConnectTimeout=30 aga2 "cd ${T}; for r in ${RUNS}; do
  d=/tmp/az_fetch_\$USER/\$r; mkdir -p \$d/val \$d/test
  [ -f p200k_fp32/\$r/\$r/metrics.jsonl ] && cp p200k_fp32/\$r/\$r/metrics.jsonl \$d/val/ || true
  [ -f p200k_fp32_test/\$r/\$r/metrics.jsonl ] && cp p200k_fp32_test/\$r/\$r/metrics.jsonl \$d/test/ || true
  [ -f runs/\$r/metrics.jsonl ] && grep -v '\"phase\": \"train_step\"' runs/\$r/metrics.jsonl > \$d/train_metrics.jsonl || true
  if [ -f runs/\$r/train.log ]; then { grep -E 'Epoch [0-9]+ \[[0-9]+/' runs/\$r/train.log | tail -700; grep -E 'Training completed|=== link start' runs/\$r/train.log | tail -3; } > \$d/train_tail.log; fi
done; tar -C /tmp/az_fetch_\$USER -czf - . ; rm -rf /tmp/az_fetch_\$USER" | tar -C "${HERE}/scores" -xzf -
for r in ${RUNS}; do
  v=$([ -s "${HERE}/scores/${r}/val/metrics.jsonl" ] && grep -c infer_step "${HERE}/scores/${r}/val/metrics.jsonl" || echo 0)
  t=$([ -s "${HERE}/scores/${r}/test/metrics.jsonl" ] && grep -c infer_step "${HERE}/scores/${r}/test/metrics.jsonl" || echo 0)
  e=$([ -s "${HERE}/scores/${r}/train_metrics.jsonl" ] && grep -c val_summary "${HERE}/scores/${r}/train_metrics.jsonl" || echo 0)
  printf "%-26s val_cases=%-5s test_cases=%-5s val_summaries=%s\n" "${r}" "${v}" "${t}" "${e}"
done
