#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.cl
#SBATCH --time=03:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
#SBATCH --open-mode=append
# TSV-driven lane launcher (2026-10-03, CHAR-LANES: the characterization campaign's dataset lanes under the three reference
# configurations chosen on DrivAerML Surface by REF-80: ISLA hidden 768 at 100,000 training cells (lr 2.5e-4, the width rule),
# GeoTransolver n_hidden 1024 at 100,000 (lr 1e-3), Transolver n_hidden 1024 at 100,000 (lr 1e-3); two seeds; augmentation
# off for every architecture. Datasets: DrivAerML Volume, AirfRANS interior, Shift-SUV, Tensile2D, GINOT lug, DeepJEB++
# (DrivAerML Surface and the 210-wing rung are the r80 frontier lanes). Preregistration: book notebook #sec-nb-char-prereg.
# idx group run_id dataset model extra lr seed cells code epochs. Chain-ahead singleton links of 3h59m; two-strike BLOCKED
# guard; checkpoint pre-flight; per-node GPU sampler. Copied from kernel/char/r80_lanes.sbatch with the epoch count read
# from the TSV (r80 hard-codes 500; DeepJEB++ runs the equal-epochs protocol of the jb lanes, 40 at the full clean split).
# Exposes --slug i and --manifest a b for kernel/lane_launch.sh.
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RECIPE="${TASK}/recipe_v8"
SELF="${TASK}/kernel/char/cl_lanes.sbatch"
LANES="${TASK}/kernel/char/cl_lanes.tsv"
lane() {
  local row; row=$(awk -F"\t" -v i="$1" 'NR>1 && $1==i {print; exit}' "${LANES}")
  [ -z "${row}" ] && { echo "bad array index $1" >&2; exit 2; }
  IFS=$'\t' read -r IDX GROUP RUN_ID DATASET MODEL EXTRA LR SEED CELLS CODE EPOCHS <<<"${row}"
  [ "${EXTRA}" = "-" ] && EXTRA=""
  SLUG="${RUN_ID#${GROUP}_}"; SLUG="${SLUG//_/-}"
}
if [ "${1:-}" = "--slug" ]; then lane "$2"; echo "${SLUG}"; exit 0; fi
if [ "${1:-}" = "--manifest" ]; then
  # The fleet manifest has no epochs column: record the count in the extra field (the scorer strips training.num_epochs).
  for i in $(seq "${2:-0}" "${3:-0}"); do lane "$i"; printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfalse\t\t\n" "${GROUP}" "$i" "${RUN_ID}" "${DATASET}" "${MODEL}" "${EXTRA:+${EXTRA} }training.num_epochs=${EPOCHS}" "${LR}" "${SEED}"; done
  exit 0
fi
lane "${SLURM_ARRAY_TASK_ID}"
CODEDIR="${TASK}/${CODE}"
TAG="cl${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=3600
export PYTHONPATH="${CODEDIR}:${RECIPE}/src"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_${TAG}"' EXIT
RUN_DIR="${TASK}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "=== link start $(date -Is) job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME} node=$(hostname) lane=${SLURM_ARRAY_TASK_ID} run=${RUN_ID} code=${CODEDIR} dataset=${DATASET} model=${MODEL} extra='${EXTRA}' lr=${LR} seed=${SEED} cells=${CELLS} epochs=${EPOCHS}" >> "${RUN_DIR}/train.log"
if [ -f "${TASK}/BLOCKED_${RUN_ID}" ]; then echo "lane ${RUN_ID} BLOCKED; no-op"; exit 0; fi
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then printf "DONE\n" > "${TASK}/RUN_${TAG}"; exit 0; fi
cd "${TASK}" && sbatch -J "${SLURM_JOB_NAME}" --array="${SLURM_ARRAY_TASK_ID}" --dependency=singleton "${SELF}" >> "${TASK}/sbatch_logs/rd_resubmit.log" 2>&1 || true
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
