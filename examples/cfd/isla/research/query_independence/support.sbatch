#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.qi.support
#SBATCH --time=02:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
#SBATCH --open-mode=append
# SUPPORT readout (eval/support_probe.py) on four interacting checkpoints, one GPU each. Rows of SUPPORT_RUNS (env, ';'
# separated): run_id|model|dataset|cells|request|runs_dir|extra. Outputs eval/support/<run>.json and .log.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-02-isla-independent
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
CODE="${TASK}/${CODE_DIR:-code_qi2}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PHYSICSNEMO_DIST_TIMEOUT_S=900
OUT="${TASK}/${SUPPORT_OUT:-eval/support}"; mkdir -p "${OUT}"
cd "${TASK}/recipe_qi"
echo "[$(date -Is)] SUPPORT readout on $(hostname), code ${CODE} ($(cat "${CODE}/SNAPSHOT_HEAD"))"
IFS=';' read -r -a rows <<<"${SUPPORT_RUNS}"
g=0
for row in "${rows[@]}"; do
  IFS='|' read -r RUN M DS CELLS REQ RD X <<<"${row}"
  echo "[$(date -Is)] gpu ${g}: ${RUN} model=${M} dataset=${DS} cells=${CELLS} request=${REQ} runs=${RD} extra='${X}'"
  ( timeout 6600 env PYTHONPATH="${CODE}:${TASK}/recipe_qi/src" CUDA_VISIBLE_DEVICES=$g RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((28000 + SLURM_JOB_ID % 10000 + g)) \
      "${VENV}/bin/python" "${TASK}/eval/support_probe.py" --model "$M" --dataset "$DS" --run-id "$RUN" --cells "$CELLS" --request "$REQ" \
      --runs-dir "$RD" --extra "$X" --task "$TASK" --out "${OUT}/${RUN}.json" --max-cases "${MAX_CASES:-0}" > "${OUT}/${RUN}.log" 2>&1 \
    && echo "[$(date -Is)] ${RUN} done" || echo "[$(date -Is)] ${RUN} FAILED rc=$? (see ${OUT}/${RUN}.log)" ) &
  g=$((g + 1))
done
wait
echo "[$(date -Is)] SUPPORT readout task done"
