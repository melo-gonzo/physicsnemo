#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.scalehl
#SBATCH --time=03:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-11%12
#SBATCH --dependency=singleton
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
#SBATCH --open-mode=append
# SCALE-HL (2026-09-20, prereg #sec-nb-scalehl-prereg): does the DrivAerML scaling picture (cells flat above 100k, width
# worth ~7% per doubling for ISLA and GeoTransolver) hold on a second dataset? HiLift medium rung (hl_medium manifest),
# 100,000 cells per case, 500 epochs, bf16, seeds 42/43: ISLA 192/384/768 (width rule), GeoTransolver 256/512 at 3e-3,
# Transolver 256 at 3e-3. The fleet's 200,000-cell medium-rung lanes supply the cells comparison at default widths.
# Manifest group "shl"; job names isla.scalehl.<slug>. Launch: bash kernel/lane_launch.sh kernel/scalehl_aga.sbatch shl isla.scalehl 0 11
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
CODE="${TASK}/code_lean"
RECIPE="${TASK}/recipe_globin"
SELF="${TASK}/kernel/scalehl_aga.sbatch"
DATASET=highlift_surface_medium
ISLA_EXTRA="model.out_scalars=3 model.out_vectors=2 +model.geo_kernel=fused +training.val_interval=25"
GT_EXTRA="forward_kwargs.global_embedding=global_data.U_inf_dir +training.val_interval=25"
T_EXTRA="forward_kwargs.fx.source=global_data.U_inf_dir +training.val_interval=25"
lane() {
  case "$1" in
    0|6)  MODEL=isla_surface_reference; EXTRA="${ISLA_EXTRA}";                  LR=1e-3;   SLUG=isla-w192 ;;
    1|7)  MODEL=isla_surface_reference; EXTRA="${ISLA_EXTRA} model.hidden=384"; LR=5e-4;   SLUG=isla-w384 ;;
    2|8)  MODEL=isla_surface_reference; EXTRA="${ISLA_EXTRA} model.hidden=768"; LR=2.5e-4; SLUG=isla-w768 ;;
    3|9)  MODEL=geotransolver_surface;  EXTRA="${GT_EXTRA}";                    LR=3e-3;   SLUG=gt-w256 ;;
    4|10) MODEL=geotransolver_surface;  EXTRA="${GT_EXTRA} model.n_hidden=512"; LR=3e-3;   SLUG=gt-w512 ;;
    5|11) MODEL=transolver_surface;     EXTRA="${T_EXTRA}";                     LR=3e-3;   SLUG=transolver-w256 ;;
    *) echo "bad array index $1" >&2; exit 2 ;;
  esac
  if [ "$1" -le 5 ]; then SEED=42; else SEED=43; fi
  CELLS=100000; EPOCHS=500
  RUN_ID="shl_med_${SLUG//-/_}_c100k_seed${SEED}"
  SLUG="${SLUG}-med-c100k-s${SEED}"
}
if [ "${1:-}" = "--slug" ]; then lane "$2"; echo "${SLUG}"; exit 0; fi
if [ "${1:-}" = "--manifest" ]; then
  for i in $(seq "${2:-0}" "${3:-11}"); do lane "$i"; printf "shl\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfalse\t\t\n" "$i" "${RUN_ID}" "${DATASET}" "${MODEL}" "${EXTRA}" "${LR}" "${SEED}"; done
  exit 0
fi
lane "${SLURM_ARRAY_TASK_ID}"
TAG="scalehl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
cd "${TASK}" && sbatch -J "${SLURM_JOB_NAME}" --array="${SLURM_ARRAY_TASK_ID}" --dependency=singleton "${SELF}" >> "${TASK}/sbatch_logs/scalehl_resubmit.log" 2>&1 || true
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
