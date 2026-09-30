#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J smoke.solids
#SBATCH --time=00:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
# Solid-mechanics smoke lanes (2026-09-22): 3 epochs each of ISLA and GeoTransolver on PLAID Tensile2d and GINOT
# PLASTIC_LUG, one lane per array index (0..3). Same torchrun invocation as kernel/af/lanes.sbatch; code_ref snapshot.
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RECIPE="${TASK}/recipe_globin"
CODEDIR="${TASK}/code_ref"
CELLS=20000
case "${SLURM_ARRAY_TASK_ID}" in
  0) RUN_ID=smoke_tensile2d_isla_r1; DATASET=plaid_tensile2d;  MODEL=isla_tensile2d;          LR=5e-4; SPLITS="train_split=train_400 val_split=val_100" ;;
  1) RUN_ID=smoke_tensile2d_gt;      DATASET=plaid_tensile2d;  MODEL=geotransolver_tensile2d; LR=1e-3; SPLITS="train_split=train_400 val_split=val_100" ;;
  2) RUN_ID=smoke_lug_isla_r1;       DATASET=ginot_lug_volume; MODEL=isla_lug;                LR=5e-4; SPLITS="train_split=train_n300 val_split=test" ;;
  3) RUN_ID=smoke_lug_gt;            DATASET=ginot_lug_volume; MODEL=geotransolver_lug;       LR=1e-3; SPLITS="train_split=train_n300 val_split=test" ;;
  *) echo "bad array index"; exit 2 ;;
esac
EXTRA="${EXTRA:-}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=3600
export PYTHONPATH="${CODEDIR}:${RECIPE}/src"
RUN_DIR="${TASK}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "=== smoke start $(date -Is) job=${SLURM_JOB_ID} node=$(hostname) run=${RUN_ID} dataset=${DATASET} model=${MODEL} lr=${LR} splits='${SPLITS}' extra='${EXTRA}' cells=${CELLS}" >> "${RUN_DIR}/train.log"
( while true; do nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader; sleep 120; done ) >> "${TASK}/sbatch_logs/gpu_smoke_${RUN_ID}.log" 2>&1 &
sampler_pid=$!
set +e
srun --overlap "${VENV}/bin/torchrun" \
  --standalone --nproc-per-node 4 \
  "${RECIPE}/src/train.py" \
  model="${MODEL}" ${EXTRA} \
  dataset="${DATASET}" \
  ${SPLITS} \
  sampling_resolution="${CELLS}" \
  run_id="${RUN_ID}" \
  output_dir="${TASK}/runs" \
  checkpoint_dir="${TASK}/runs" \
  training.seed=42 \
  training.num_epochs=3 \
  training.optimizer.lr="${LR}" \
  training.save_interval=1 \
  logging.log_every_n_steps=20 \
  precision=bfloat16 \
  augment=false \
  compile=false \
  >> "${RUN_DIR}/train.log" 2>&1
rc=$?
kill "${sampler_pid}" 2>/dev/null || true
echo "=== smoke end rc=${rc} $(date -Is)" >> "${RUN_DIR}/train.log"
exit "${rc}"
