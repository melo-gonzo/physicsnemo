#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.qi.fullfield
#SBATCH --time=01:30:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# Full-field readout (eval/full_field_probe.py) on AirfRANS, one GPU per seed: tiles / random / support / passive.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-02-isla-independent
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin" OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PHYSICSNEMO_DIST_TIMEOUT_S=900
OUT="${TASK}/eval/full_field"; mkdir -p "${OUT}"
cd "${TASK}/recipe_qi"
g=0
for S in 42 43; do
  ( env PYTHONPATH="${TASK}/code_qi2:${TASK}/recipe_qi/src" CUDA_VISIBLE_DEVICES=$g RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((30500 + g)) \
      "${VENV}/bin/python" "${TASK}/eval/full_field_probe.py" --run-int "qi_af_int_seed${S}" --run-pas "qi_af_pas_seed${S}" \
      --model isla_volume_airfrans --dataset airfrans_volume_2d_all --cells 100000 --request 100000 \
      --extra "train_split=full_train val_split=full_test" --task "${TASK}" --out "${OUT}/qi_af_seed${S}.json" --max-cases "${MAX_CASES:-0}" \
      > "${OUT}/qi_af_seed${S}.log" 2>&1 && echo "[$(date -Is)] seed ${S} done" || echo "[$(date -Is)] seed ${S} FAILED rc=$?" ) &
  g=$((g + 1))
done
wait
