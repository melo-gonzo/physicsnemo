#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.trainarea-dr
#SBATCH --time=03:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-5%6
#SBATCH --dependency=singleton
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
#SBATCH --open-mode=append
# TRAIN-AREA-DR (2026-09-20, prereg #sec-nb-trainarea-dr-prereg): the DrivAerML twin of TRAIN-AREA. ISLA, GeoTransolver
# and Transolver trained on 435 cars under an area-proportional Poisson draw (drivaer_ml_surface_area: 1,000,000-cell
# uniform pool, expected 100,000 cells, exact HT weights), 500 epochs, bf16, seeds 42/43; scored with the consistency
# benchmark against the uniform-trained ISO-COST 100k lanes under the same three draws. Manifest group "tadr"; job names
# isla.trainarea-dr.<slug>. Launch: bash kernel/lane_launch.sh kernel/trainarea_dr_aga.sbatch tadr isla.trainarea-dr 0 5
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
CODE="${TASK}/code_lean"
RECIPE="${TASK}/recipe_globin"
SELF="${TASK}/kernel/trainarea_dr_aga.sbatch"
DATASET=drivaer_ml_surface_area
lane() {
  case "$1" in
    0|3) MODEL=isla_surface_reference; EXTRA="+model.geo_kernel=fused";                                    LR=1e-3; SLUG=isla-w192 ;;
    1|4) MODEL=geotransolver_surface;  EXTRA="forward_kwargs.global_embedding=global_data.U_inf_dir";      LR=3e-3; SLUG=gt-w256 ;;
    2|5) MODEL=transolver_surface;     EXTRA="forward_kwargs.fx.source=global_data.U_inf_dir";             LR=3e-3; SLUG=transolver-w256 ;;
    *) echo "bad array index $1" >&2; exit 2 ;;
  esac
  if [ "$1" -le 2 ]; then SEED=42; else SEED=43; fi
  CELLS=100000; EPOCHS=500
  RUN_ID="tadr_dr_${SLUG//-/_}_area_c100k_seed${SEED}"
  SLUG="${SLUG}-area-c100k-s${SEED}"
}
if [ "${1:-}" = "--slug" ]; then lane "$2"; echo "${SLUG}"; exit 0; fi
if [ "${1:-}" = "--manifest" ]; then
  for i in $(seq "${2:-0}" "${3:-5}"); do lane "$i"; printf "tadr\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfalse\t\t\n" "$i" "${RUN_ID}" "${DATASET}" "${MODEL}" "${EXTRA}" "${LR}" "${SEED}"; done
  exit 0
fi
lane "${SLURM_ARRAY_TASK_ID}"
TAG="tadr_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=3600
export PYTHONPATH="${CODE}:${RECIPE}/src"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_${TAG}"' EXIT
RUN_DIR="${TASK}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "=== link start $(date -Is) job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME} node=$(hostname) lane=${SLURM_ARRAY_TASK_ID} run=${RUN_ID} dataset=${DATASET} model=${MODEL} extra='${EXTRA}' lr=${LR} seed=${SEED} cells=${CELLS} epochs=${EPOCHS}" >> "${RUN_DIR}/train.log"
if [ -f "${TASK}/BLOCKED_${RUN_ID}" ]; then echo "lane ${RUN_ID} BLOCKED; no-op"; exit 0; fi
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then printf "DONE\n" > "${TASK}/RUN_${TAG}"; exit 0; fi
cd "${TASK}" && sbatch -J "${SLURM_JOB_NAME}" --array="${SLURM_ARRAY_TASK_ID}" --dependency=singleton "${SELF}" >> "${TASK}/sbatch_logs/tadr_resubmit.log" 2>&1 || true
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader
    sleep 300
  done
) >> "${TASK}/sbatch_logs/gpu_${TAG}.log" 2>&1 &
sampler_pid=$!
"${VENV}/bin/python" "${TASK}/kernel/ckpt_preflight.py" "${RUN_DIR}/checkpoints" >> "${RUN_DIR}/train.log" 2>&1 || true
LOG_START=$(( $(wc -l < "${RUN_DIR}/train.log" 2>/dev/null || echo 0) + 1 ))
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
  training.save_interval=2 \
  logging.log_every_n_steps=50 \
  precision=bfloat16 \
  augment=false \
  compile=false \
  >> "${RUN_DIR}/train.log" 2>&1
rc=$?
set -e
kill "${sampler_pid}" 2>/dev/null || true
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then printf "DONE\n" > "${TASK}/RUN_${TAG}"; exit 0; fi
if [ "${rc}" -ne 0 ]; then
  fp=$(tail -n +"${LOG_START}" "${RUN_DIR}/train.log" 2>/dev/null | grep -hE "step failed at epoch|Error executing job|^[A-Za-z_.]*Error|Could not override|is not in struct|out of memory" | grep -v ChildFailedError | tail -1)
  prev=$(cat "${RUN_DIR}/.last_failure" 2>/dev/null || true)
  printf "%s\n" "${fp}" > "${RUN_DIR}/.last_failure"
  if [ -n "${fp}" ] && [ "${fp}" = "${prev}" ]; then
    printf "identical failure on consecutive segments: %s\n" "${fp}" > "${TASK}/BLOCKED_${RUN_ID}"; exit "${rc}"
  fi
fi
exit "${rc}"
