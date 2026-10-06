#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.global.we
#SBATCH --time=02:00:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
#SBATCH --open-mode=append
# Weighted-error probe (eval/we_probe.py) on up to four rows of eval/we_runs.tsv, one GPU each.
# usage: ROWS="0 1 2 3" [MAX_CASES=2] [OUT_SUB=probe] sbatch eval/we.sbatch
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PHYSICSNEMO_DIST_TIMEOUT_S=900
OUT="${TASK}/we/${OUT_SUB:-probe}"; mkdir -p "${OUT}"
cd "${TASK}/recipe_we"
echo "[$(date -Is)] WE probe on $(hostname): rows ${ROWS}"
g=0
for r in ${ROWS}; do
  IFS=$'\t' read -r IDX RUN M DS CELLS REQ RD CODE X LAWS < <(awk -F'\t' -v i="$r" 'NR>1 && $1==i' "${TASK}/we/we_runs.tsv")
  [ "${X}" = "-" ] && X=""
  echo "[$(date -Is)] gpu ${g}: ${RUN} code=${CODE} ($(cat "${CODE}/SNAPSHOT_HEAD" | head -c 9)) pool=${DS} laws=${LAWS}"
  ( timeout 6900 env PYTHONPATH="${CODE}:${TASK}/recipe_we/src" CUDA_VISIBLE_DEVICES=$g RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$((28000 + SLURM_JOB_ID % 10000 + g)) \
      "${VENV}/bin/python" "${TASK}/we/we_probe.py" --model "$M" --dataset "$DS" --run-id "$RUN" --cells "$CELLS" \
      --request "$REQ" --runs-dir "$RD" --extra "$X" --laws "$LAWS" --task "$TASK" --out "${OUT}/${RUN}.json" \
      --max-cases "${MAX_CASES:-0}" > "${OUT}/${RUN}.log" 2>&1 \
    && echo "[$(date -Is)] ${RUN} done" || echo "[$(date -Is)] ${RUN} FAILED rc=$? (see ${OUT}/${RUN}.log)" ) &
  g=$((g + 1))
done
wait
echo "[$(date -Is)] WE probe task done"
