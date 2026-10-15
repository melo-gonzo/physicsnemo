#!/bin/bash
# Fetch the AEROZOO scoring records from AGA into results/campaigns_2026-10-09/aerozoo/scores/ (2026-10-10).
#   scores/<run>/val/metrics.jsonl    the fleet scorer's validation-side record (p200k_fp32/<run>/<run>/metrics.jsonl)
#   scores/<run>/test/metrics.jsonl   the test-side record (p200k_fp32_test/<run>/<run>/metrics.jsonl, eval_test.sbatch)
#   scores/<run>/train_metrics.jsonl  the lane's training record without the per-step rows (config, dataset, val_summary,
#                                     train_summary: the validation trajectory every ten epochs)
#   scores/<run>/train_tail.log       the last 700 step lines of train.log (step time and memory of the final epoch) and
#                                     the completion line
#   scores/<run>/bestval/metrics.jsonl the labeled DIAGNOSTIC record of a chosen (best-validation) checkpoint on the
#                                     validation side (p200k_fp32_bestval/<run>/<run>/metrics.jsonl, eval_bestval.sbatch),
#                                     with the checkpoint file it scored in bestval/CHECKPOINT_USED
#   scores/<run>/bestval_test/metrics.jsonl the same chosen checkpoint scored on the TEST side (p200k_fp32_bestval_test/<run>/,
#                                     eval_bestval.sbatch with BESTVAL_SIDE=test; the best-checkpoint table's protocol), with its
#                                     CHECKPOINT_USED
# Hosts (2026-10-14): the LHR site mirrors AGA's absolute paths. Every lanes.tsv row >= 16 and every row whose run directory
#   exists on LHR (rows 8, 9, 12 and 13: the ISLA Reynolds and blob-bank lanes) is fetched from lhr2; the rows that trained and
#   were scored on AGA (0-7, 10, 11, 14 and 15: ISLA and GeoTransolver full and nonbox, GeoTransolver reynolds and blobs) from aga2.
# usage: bash fetch_scores.sh        (from anywhere; idempotent, overwrites)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
RUNS=$(awk -F'\t' 'NR>1{print $3}' "${HERE}/lanes.tsv" | tr '\n' ' ')
for r in ${RUNS}; do
  mkdir -p "${HERE}/scores/${r}/val" "${HERE}/scores/${r}/test"
done
fetch_from() {
local HOST=$1; shift; local RUNS="$*"
[ -z "${RUNS// /}" ] && return 0
ssh -o ConnectTimeout=30 "${HOST}" "cd ${T}; for r in ${RUNS}; do
  d=/tmp/az_fetch_\$USER/\$r; mkdir -p \$d/val \$d/test
  [ -f p200k_fp32/\$r/\$r/metrics.jsonl ] && cp p200k_fp32/\$r/\$r/metrics.jsonl \$d/val/ || true
  [ -f p200k_fp32_test/\$r/\$r/metrics.jsonl ] && cp p200k_fp32_test/\$r/\$r/metrics.jsonl \$d/test/ || true
  if [ -f p200k_fp32_bestval/\$r/.done ]; then mkdir -p \$d/bestval; cp p200k_fp32_bestval/\$r/\$r/metrics.jsonl p200k_fp32_bestval/\$r/CHECKPOINT_USED \$d/bestval/; fi
  if [ -f p200k_fp32_bestval_test/\$r/.done ]; then mkdir -p \$d/bestval_test; cp p200k_fp32_bestval_test/\$r/\$r/metrics.jsonl p200k_fp32_bestval_test/\$r/CHECKPOINT_USED \$d/bestval_test/; fi
  [ -f runs/\$r/metrics.jsonl ] && grep -v '\"phase\": \"train_step\"' runs/\$r/metrics.jsonl > \$d/train_metrics.jsonl || true
  if [ -f runs/\$r/train.log ]; then { grep -E 'Epoch [0-9]+ \[[0-9]+/' runs/\$r/train.log | tail -700; grep -E 'Training completed|=== link start' runs/\$r/train.log | tail -3; } > \$d/train_tail.log; fi
done; tar -C /tmp/az_fetch_\$USER -czf - . ; rm -rf /tmp/az_fetch_\$USER" | tar -C "${HERE}/scores" -xzf -
}
LOW=$(awk -F'\t' 'NR>1 && $1<16 {print $3}' "${HERE}/lanes.tsv" | tr '\n' ' ')
LOW_ON_LHR=" $(ssh -o ConnectTimeout=30 lhr2 "cd ${T}/runs && for r in ${LOW}; do [ -d \$r ] && printf '%s ' \$r; done; true") "
LHR_RUNS=$(awk -F'\t' -v low="${LOW_ON_LHR}" 'NR>1 && ($1>=16 || index(low, " " $3 " ")) {printf "%s ", $3}' "${HERE}/lanes.tsv")
AGA_RUNS=$(awk -F'\t' -v low="${LOW_ON_LHR}" 'NR>1 && !($1>=16 || index(low, " " $3 " ")) {printf "%s ", $3}' "${HERE}/lanes.tsv")
fetch_from aga2 ${AGA_RUNS}
fetch_from lhr2 ${LHR_RUNS}
for r in ${RUNS}; do
  v=$([ -s "${HERE}/scores/${r}/val/metrics.jsonl" ] && grep -c infer_step "${HERE}/scores/${r}/val/metrics.jsonl" || echo 0)
  t=$([ -s "${HERE}/scores/${r}/test/metrics.jsonl" ] && grep -c infer_step "${HERE}/scores/${r}/test/metrics.jsonl" || echo 0)
  e=$([ -s "${HERE}/scores/${r}/train_metrics.jsonl" ] && grep -c val_summary "${HERE}/scores/${r}/train_metrics.jsonl" || echo 0)
  printf "%-26s val_cases=%-5s test_cases=%-5s val_summaries=%s\n" "${r}" "${v}" "${t}" "${e}"
done
