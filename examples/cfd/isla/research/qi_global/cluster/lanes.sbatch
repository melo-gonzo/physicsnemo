#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.global
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
# QI-GLOBAL lane (2026-10-05): AirfRANS, read-only ISLA with the global boundary read (++model.kernel_read_radius=null) and
# its references (interacting, read-only, truncated read). Report: examples/cfd/isla/research/qi_global/
# REPORT.md on psharpe/isla-qi-global. Adapted from the QI-KERNEL launcher (agents/2026-10-03-isla-qi-kernel).
# One lane per array index, read from kernel/lanes.tsv (idx group run_id dataset model extra lr seed cells code epochs).
# Chain-ahead singleton links of 3h59: each link submits its successor under the same job name, then resumes training
# from the newest intact checkpoint; a lane whose log says "Training completed" exits at once. Two identical failures
# on consecutive links write status/BLOCKED_<run> and stop the chain. Adapted from the mt2-stage0 kernel/v18 launcher.
# Submit one lane:  cd $TASK && sbatch -J isla.global.<slug> --array=<idx> kernel/lanes.sbatch   (kernel/submit.sh does all)
set -euo pipefail
ulimit -s 8192
ulimit -c 0
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RECIPE="${TASK}/recipe_g"
SELF="${TASK}/kernel/lanes.sbatch"
LANES="${TASK}/kernel/lanes.tsv"
lane() {
  local row; row=$(awk -F"\t" -v i="$1" 'NR>1 && $1==i {print; exit}' "${LANES}")
  [ -z "${row}" ] && { echo "bad array index $1" >&2; exit 2; }
  IFS=$'\t' read -r IDX GROUP RUN_ID DATASET MODEL EXTRA LR SEED CELLS CODE EPOCHS <<<"${row}"
  [ "${EXTRA}" = "-" ] && EXTRA=""
  SLUG="${RUN_ID#${GROUP}_}"; SLUG="${SLUG//_/-}"
}
if [ "${1:-}" = "--slug" ]; then lane "$2"; echo "${SLUG}"; exit 0; fi
lane "${SLURM_ARRAY_TASK_ID}"
CODEDIR="${TASK}/${CODE}"
TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=3600
export PYTHONPATH="${CODEDIR}:${RECIPE}/src"
### torch.compile / Triton caches stay node-local (/tmp defaults): a Lustre cache made the first compiled step 6 min, not 1.
mkdir -p "${TASK}/status"
RUN_DIR="${TASK}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
link_start=$SECONDS
### Exit marker, and a guard against a runaway chain: three consecutive links of a lane that end within 300 s
### without the lane being complete write status/BLOCKED_<run>, which stops every later link.
on_exit() {
  local rc=$?
  printf "rc=%s\n" "$rc" > "${TASK}/status/STATUS_${TAG}"
  if [ "$((SECONDS - link_start))" -lt 300 ] && ! grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then
    local n=$(( $(cat "${TASK}/status/FASTFAIL_${RUN_ID}" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "${TASK}/status/FASTFAIL_${RUN_ID}"
    if [ "$n" -ge 3 ]; then echo "three consecutive links ended within 300 s" > "${TASK}/status/BLOCKED_${RUN_ID}"; fi
  else
    rm -f "${TASK}/status/FASTFAIL_${RUN_ID}"
  fi
}
trap on_exit EXIT
printf '[%s] QI-GLOBAL link start: job=%s name=%s node=%s restart=%s lane=%s run=%s\n' "$(date -Is)" "${SLURM_JOB_ID}" \
  "${SLURM_JOB_NAME}" "$(hostname)" "${SLURM_RESTART_COUNT:-0}" "${SLURM_ARRAY_TASK_ID}" "${RUN_ID}"
printf '  code=%s (snapshot %s) recipe=%s venv=%s\n' "${CODEDIR}" "$(cat "${CODEDIR}/SNAPSHOT_HEAD" 2>/dev/null)" "${RECIPE}" "${VENV}"
printf '  dataset=%s model=%s extra=%q lr=%s seed=%s cells=%s epochs=%s precision=bfloat16\n' "${DATASET}" "${MODEL}" "${EXTRA}" "${LR}" "${SEED}" "${CELLS}" "${EPOCHS}"
printf '  train log: %s/train.log  checkpoints: %s/checkpoints\n' "${RUN_DIR}" "${RUN_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  gpu /' || echo "  nvidia-smi failed" >&2
echo "=== link start $(date -Is) job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME} node=$(hostname) lane=${SLURM_ARRAY_TASK_ID} run=${RUN_ID} code=${CODEDIR} dataset=${DATASET} model=${MODEL} extra='${EXTRA}' lr=${LR} seed=${SEED} cells=${CELLS} epochs=${EPOCHS}" >> "${RUN_DIR}/train.log"
if [ -f "${TASK}/status/BLOCKED_${RUN_ID}" ]; then echo "lane ${RUN_ID} BLOCKED ($(cat "${TASK}/status/BLOCKED_${RUN_ID}")); no-op"; exit 0; fi
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then echo "lane ${RUN_ID} already complete; no-op"; printf "DONE\n" > "${TASK}/status/RUN_${TAG}"; exit 0; fi
cd "${TASK}" && sbatch -J "${SLURM_JOB_NAME}" --array="${SLURM_ARRAY_TASK_ID}" --dependency=singleton "${SELF}" >> "${TASK}/sbatch_logs/resubmit.log" 2>&1 || true
echo "  successor link submitted (see sbatch_logs/resubmit.log)"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader
    sleep 300
  done
) >> "${TASK}/sbatch_logs/gpu_${TAG}.log" 2>&1 &
sampler_pid=$!
"${VENV}/bin/python" "${TASK}/kernel/ckpt_preflight.py" "${RUN_DIR}/checkpoints" >> "${RUN_DIR}/train.log" 2>&1 || true
latest=$( (ls "${RUN_DIR}/checkpoints" 2>/dev/null || true) | grep -E '^checkpoint\.0\.[0-9]+\.pt$' | sed -E 's/checkpoint\.0\.([0-9]+)\.pt/\1/' | sort -n | tail -1 || true)
echo "  resuming from epoch ${latest:-0} (newest intact trainer checkpoint)"
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
last_epoch=$(tail -n +"${LOG_START}" "${RUN_DIR}/train.log" 2>/dev/null | grep -E '^Epoch \[[0-9]+/[0-9]+\]' | tail -1 || true)
printf '[%s] trainer exited rc=%s after %s s; last epoch line: %s\n' "$(date -Is)" "${rc}" "$((SECONDS - link_start))" "${last_epoch:-none}"
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then echo "lane ${RUN_ID} DONE"; printf "DONE\n" > "${TASK}/status/RUN_${TAG}"; exit 0; fi
if [ "${rc}" -ne 0 ]; then
  fp=$(tail -n +"${LOG_START}" "${RUN_DIR}/train.log" 2>/dev/null | grep -hE "step failed at epoch|Error executing job|^[A-Za-z_.]*Error|Could not override|is not in struct|out of memory" | grep -v ChildFailedError | tail -1 || true)
  echo "failure fingerprint: ${fp:-<none: wall-time kill or signal>}" >&2
  prev=$(cat "${RUN_DIR}/.last_failure" 2>/dev/null || true)
  printf "%s\n" "${fp}" > "${RUN_DIR}/.last_failure"
  if [ -n "${fp}" ] && [ "${fp}" = "${prev}" ]; then
    printf "identical failure on consecutive links: %s\n" "${fp}" > "${TASK}/status/BLOCKED_${RUN_ID}"
    echo "lane ${RUN_ID} BLOCKED: identical failure on consecutive links" >&2
    exit "${rc}"
  fi
fi
exit "${rc}"
