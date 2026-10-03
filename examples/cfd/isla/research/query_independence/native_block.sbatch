#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.qi.nativeblock
#SBATCH --time=01:30:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# Region-matched support readout (eval/native_block_probe.py) on the AirfRANS interacting lanes, one GPU each.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-02-isla-independent
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PHYSICSNEMO_DIST_TIMEOUT_S=900
OUT="${TASK}/eval/native_block"; mkdir -p "${OUT}"
cd "${TASK}/recipe_qi"
g=0
for RUN in qi_af_int_seed42 qi_af_int_seed43; do
  ( env PYTHONPATH="${TASK}/code_qi2:${TASK}/recipe_qi/src" CUDA_VISIBLE_DEVICES=$g RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + g)) \
      "${VENV}/bin/python" "${TASK}/eval/native_block_probe.py" --model isla_volume_airfrans --dataset airfrans_volume_2d \
      --dataset-all airfrans_volume_2d_all --run-id "${RUN}" --cells 100000 --request 100000 --extra "train_split=full_train val_split=full_test" \
      --task "${TASK}" --out "${OUT}/${RUN}.json" --max-cases "${MAX_CASES:-0}" > "${OUT}/${RUN}.log" 2>&1 \
    && echo "[$(date -Is)] ${RUN} done" || echo "[$(date -Is)] ${RUN} FAILED rc=$?" ) &
  g=$((g + 1))
done
wait
