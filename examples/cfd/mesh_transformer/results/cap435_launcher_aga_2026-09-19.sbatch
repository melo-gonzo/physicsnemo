#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.cap435
#SBATCH --time=03:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-21%22
#SBATCH --dependency=singleton
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
#SBATCH --open-mode=append
# CAP-435 (2026-09-19, prereg #sec-nb-cap435-prereg): how far do width and depth carry at 435 DrivAerML cars?
# ISO-COST's first 16 lanes found the cells axis flat from 100,000 to 200,000 cells and width worth 7% per doubling
# for ISLA and GeoTransolver alike. This campaign pushes width (ISLA 512/1024/1536; GeoTransolver 1024 at both rates;
# Transolver 512), depth (ISLA 6/24/48 layers at width 384; GeoTransolver 24 layers at 512) and slices (ISLA 1,024 at
# width 384), all at 100,000 cells per case (the scaling-study budget; cells are flat there), 500 epochs, bf16, seeds
# 42 and 43, scored in float32 at the 200,000-cell reporting protocol through the fleet pump (manifest group "cap").
# ISLA follows the width rule lr = 1e-3 x 192 / hidden. Job names follow the campaign convention <project>.<campaign>.<lane>:
#   cd $TASK && for i in $(seq 0 21); do sbatch -J isla.cap435.$(bash kernel/cap435_aga.sbatch --slug $i) --array=$i --dependency=singleton kernel/cap435_aga.sbatch; done
# Each link resubmits itself once (singleton chain, one queued successor per lane) and resumes from the newest intact
# checkpoint; the two-strike guard blocks a lane whose consecutive links fail with the same fingerprint.
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
CODE="${TASK}/code_lean"
RECIPE="${TASK}/recipe_globin"
SELF="${TASK}/kernel/cap435_aga.sbatch"
GT_EXTRA="forward_kwargs.global_embedding=global_data.U_inf_dir"
lane() {
  case "$1" in
    0|11)  MODEL=isla_surface_reference; EXTRA="+model.geo_kernel=fused model.hidden=512";                 LR=3.75e-4;  SLUG=isla-w512 ;;
    1|12)  MODEL=isla_surface_reference; EXTRA="+model.geo_kernel=fused model.hidden=1024";                LR=1.875e-4; SLUG=isla-w1024 ;;
    2|13)  MODEL=isla_surface_reference; EXTRA="+model.geo_kernel=fused model.hidden=1536";                LR=1.25e-4;  SLUG=isla-w1536 ;;
    3|14)  MODEL=isla_surface_reference; EXTRA="+model.geo_kernel=fused model.hidden=384 model.n_layers=6";  LR=5e-4;   SLUG=isla-w384-d6 ;;
    4|15)  MODEL=isla_surface_reference; EXTRA="+model.geo_kernel=fused model.hidden=384 model.n_layers=24"; LR=5e-4;   SLUG=isla-w384-d24 ;;
    5|16)  MODEL=isla_surface_reference; EXTRA="+model.geo_kernel=fused model.hidden=384 model.n_layers=48"; LR=5e-4;   SLUG=isla-w384-d48 ;;
    6|17)  MODEL=isla_surface_reference; EXTRA="+model.geo_kernel=fused model.hidden=384 model.n_slices=1024"; LR=5e-4; SLUG=isla-w384-s1024 ;;
    7|18)  MODEL=geotransolver_surface;  EXTRA="${GT_EXTRA} model.n_hidden=1024";                          LR=3e-3;     SLUG=gt-w1024-lr3e3 ;;
    8|19)  MODEL=geotransolver_surface;  EXTRA="${GT_EXTRA} model.n_hidden=1024";                          LR=1e-3;     SLUG=gt-w1024-lr1e3 ;;
    9|20)  MODEL=geotransolver_surface;  EXTRA="${GT_EXTRA} model.n_hidden=512 model.n_layers=24";         LR=3e-3;     SLUG=gt-w512-d24 ;;
    10|21) MODEL=transolver_surface;     EXTRA="forward_kwargs.fx.source=global_data.U_inf_dir model.n_hidden=512"; LR=3e-3; SLUG=transolver-w512 ;;
    *) echo "bad array index $1" >&2; exit 2 ;;
  esac
  if [ "$1" -le 10 ]; then SEED=42; else SEED=43; fi
  CELLS=100000; EPOCHS=500
  RUN_ID="cap_dr_${SLUG//-/_}_c100k_seed${SEED}"
  SLUG="${SLUG}-c100k-s${SEED}"
}
if [ "${1:-}" = "--slug" ]; then lane "$2"; echo "${SLUG}"; exit 0; fi
if [ "${1:-}" = "--manifest" ]; then
  for i in $(seq 0 21); do lane "$i"; printf "cap\t%s\t%s\tdrivaer_ml_surface\t%s\t%s\t%s\t%s\tfalse\t\t\n" "$i" "${RUN_ID}" "${MODEL}" "${EXTRA}" "${LR}" "${SEED}"; done
  exit 0
fi
lane "${SLURM_ARRAY_TASK_ID}"
TAG="cap435_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=900
export PYTHONPATH="${CODE}:${RECIPE}/src"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_${TAG}"' EXIT
RUN_DIR="${TASK}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "=== link start $(date -Is) job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME} node=$(hostname) lane=${SLURM_ARRAY_TASK_ID} run=${RUN_ID} model=${MODEL} extra='${EXTRA}' lr=${LR} seed=${SEED} cells=${CELLS} epochs=${EPOCHS}" >> "${RUN_DIR}/train.log"
if [ -f "${TASK}/BLOCKED_${RUN_ID}" ]; then echo "lane ${RUN_ID} BLOCKED; no-op"; exit 0; fi
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then printf "DONE\n" > "${TASK}/RUN_${TAG}"; exit 0; fi
# one queued successor per lane: same job name, singleton dependency
cd "${TASK}" && sbatch -J "${SLURM_JOB_NAME}" --array="${SLURM_ARRAY_TASK_ID}" --dependency=singleton "${SELF}" >> "${TASK}/sbatch_logs/cap435_resubmit.log" 2>&1 || true
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader
    sleep 300
  done
) >> "${TASK}/sbatch_logs/gpu_${TAG}.log" 2>&1 &
sampler_pid=$!
# checkpoint pre-flight (2026-09-18): move truncated checkpoints aside so the link resumes from an intact one
"${VENV}/bin/python" "${TASK}/kernel/ckpt_preflight.py" "${RUN_DIR}/checkpoints" >> "${RUN_DIR}/train.log" 2>&1 || true
# two-strike guard fingerprint (2026-09-19): only this link's log lines count
LOG_START=$(( $(wc -l < "${RUN_DIR}/train.log" 2>/dev/null || echo 0) + 1 ))
set +e
srun --overlap "${VENV}/bin/torchrun" \
  --standalone --nproc-per-node 4 \
  "${RECIPE}/src/train.py" \
  model="${MODEL}" ${EXTRA} \
  dataset=drivaer_ml_surface \
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
