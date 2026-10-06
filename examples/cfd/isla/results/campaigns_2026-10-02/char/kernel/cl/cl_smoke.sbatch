#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.cs
#SBATCH --time=03:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
#SBATCH --open-mode=append
# CHAR-LANES smoke (2026-10-03): one real epoch of every (dataset, architecture) configuration of kernel/char/cl_lanes.tsv at
# the campaign's reference configurations and cells, seed 42, validation after the epoch (+training.val_interval=1), proving
# each pipeline runs end to end to a validation metric and recording the epoch's peak device memory (the 80 GiB line) and
# wall time. One-shot: no singleton chain, no resubmission (copied from kernel/char/rf1_lanes.sbatch). Same TSV layout as
# cl_lanes.tsv (idx group run_id dataset model extra lr seed cells code epochs; epochs = 1 here); exposes --slug i and
# --manifest a b for kernel/lane_launch.sh. Code snapshot code_v19, recipe_v8.
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RECIPE="${TASK}/recipe_v8"
LANES="${TASK}/kernel/char/cl_smoke.tsv"
lane() {
  local row; row=$(awk -F"\t" -v i="$1" 'NR>1 && $1==i {print; exit}' "${LANES}")
  [ -z "${row}" ] && { echo "bad array index $1" >&2; exit 2; }
  IFS=$'\t' read -r IDX GROUP RUN_ID DATASET MODEL EXTRA LR SEED CELLS CODE EPOCHS <<<"${row}"
  [ "${EXTRA}" = "-" ] && EXTRA=""
  SLUG="${RUN_ID#${GROUP}_}"; SLUG="${SLUG//_/-}"
}
if [ "${1:-}" = "--slug" ]; then lane "$2"; echo "${SLUG}"; exit 0; fi
if [ "${1:-}" = "--manifest" ]; then
  for i in $(seq "${2:-0}" "${3:-0}"); do lane "$i"; printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfalse\t\t\n" "${GROUP}" "$i" "${RUN_ID}" "${DATASET}" "${MODEL}" "${EXTRA:+${EXTRA} }training.num_epochs=${EPOCHS}" "${LR}" "${SEED}"; done
  exit 0
fi
lane "${SLURM_ARRAY_TASK_ID}"
CODEDIR="${TASK}/${CODE}"
TAG="cs${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=3600
export PYTHONPATH="${CODEDIR}:${RECIPE}/src"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_${TAG}"' EXIT
RUN_DIR="${TASK}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "=== smoke start $(date -Is) job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME} node=$(hostname) lane=${SLURM_ARRAY_TASK_ID} run=${RUN_ID} code=${CODEDIR} dataset=${DATASET} model=${MODEL} extra='${EXTRA}' lr=${LR} seed=${SEED} cells=${CELLS} epochs=${EPOCHS}" >> "${RUN_DIR}/train.log"
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then printf "DONE\n" > "${TASK}/RUN_${TAG}"; exit 0; fi
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader
    sleep 300
  done
) >> "${TASK}/sbatch_logs/gpu_${TAG}.log" 2>&1 &
sampler_pid=$!
set +e
srun --overlap "${VENV}/bin/torchrun" \
  --standalone --nproc-per-node 4 \
  "${RECIPE}/src/train.py" \
  model="${MODEL}" ${EXTRA} \
  dataset="${DATASET}" \
  sampling_resolution="${CELLS}" \
  run_id="${RUN_ID}" \
  output_dir="${TASK}/runs" \
  checkpoint_dir="${TASK}/runs" \
  training.seed="${SEED}" \
  training.num_epochs="${EPOCHS}" \
  training.optimizer.lr="${LR}" \
  training.save_interval=1 \
  logging.log_every_n_steps=20 \
  precision=bfloat16 \
  augment=false \
  compile=false \
  >> "${RUN_DIR}/train.log" 2>&1
rc=$?
set -e
kill "${sampler_pid}" 2>/dev/null || true
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then printf "DONE\n" > "${TASK}/RUN_${TAG}"; fi
echo "=== smoke end rc=${rc} $(date -Is)" >> "${RUN_DIR}/train.log"
exit "${rc}"
