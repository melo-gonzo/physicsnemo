#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.routenorm
#SBATCH --time=03:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-29%30
#SBATCH --dependency=singleton
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
#SBATCH --open-mode=append
# ROUTE-NORM (2026-09-20, prereg #sec-nb-routenorm-prereg): does bounding the routing logits (cosine routing at a fixed
# temperature) remove the width- and depth-dependent learning-rate ceiling that ROUTING-SAT measured as total routing
# collapse? Code snapshot code_routenorm (ISLA with routing_norm). DrivAerML 435 cars at 100k cells: cosine routing at
# widths 192/384/768 at the rule rate and at the rates that collapse linear routing (384 at 1e-3, 768 at 5e-4, 768 at
# 1e-3); HiLift medium rung at 100k: cosine 192 and 384 at 1e-3 and the linear 384 at 1e-3 control. Two seeds, 500
# epochs, bf16. Manifest group "rn"; job names isla.routenorm.<slug>.
set -euo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
CODE="${TASK}/code_routenorm"
RECIPE="${TASK}/recipe_globin"
SELF="${TASK}/kernel/routenorm_aga.sbatch"
COS="+model.geo_kernel=fused +model.routing_norm=cosine"
LIN="+model.geo_kernel=fused +model.routing_norm=linear"
HL="model.out_scalars=3 model.out_vectors=2 +training.val_interval=25"
lane() {
  case "$1" in
    0|9)   DATASET=drivaer_ml_surface;      EXTRA="${COS}";                  LR=1e-3;   SLUG=cos-w192-lr1e-3 ;;
    1|10)  DATASET=drivaer_ml_surface;      EXTRA="${COS} model.hidden=384"; LR=5e-4;   SLUG=cos-w384-lr5e-4 ;;
    2|11)  DATASET=drivaer_ml_surface;      EXTRA="${COS} model.hidden=384"; LR=1e-3;   SLUG=cos-w384-lr1e-3 ;;
    3|12)  DATASET=drivaer_ml_surface;      EXTRA="${COS} model.hidden=768"; LR=2.5e-4; SLUG=cos-w768-lr2.5e-4 ;;
    4|13)  DATASET=drivaer_ml_surface;      EXTRA="${COS} model.hidden=768"; LR=5e-4;   SLUG=cos-w768-lr5e-4 ;;
    5|14)  DATASET=drivaer_ml_surface;      EXTRA="${COS} model.hidden=768"; LR=1e-3;   SLUG=cos-w768-lr1e-3 ;;
    6|15)  DATASET=highlift_surface_medium; EXTRA="${COS} ${HL}";                  LR=1e-3; SLUG=cos-w192-lr1e-3-med ;;
    7|16)  DATASET=highlift_surface_medium; EXTRA="${COS} ${HL} model.hidden=384"; LR=1e-3; SLUG=cos-w384-lr1e-3-med ;;
    8|17)  DATASET=highlift_surface_medium; EXTRA="${LIN} ${HL} model.hidden=384"; LR=1e-3; SLUG=lin-w384-lr1e-3-med ;;
    # ladder extension (2026-09-20, second reading): cosine ISLA at 1,024 and 1,536 wide at the rates the 768-wide lanes
    # survive, and GeoTransolver at 2,048 wide so the baseline is pushed as far.
    18|24) DATASET=drivaer_ml_surface;      EXTRA="${COS} model.hidden=1024"; LR=5e-4; SLUG=cos-w1024-lr5e-4 ;;
    19|25) DATASET=drivaer_ml_surface;      EXTRA="${COS} model.hidden=1024"; LR=1e-3; SLUG=cos-w1024-lr1e-3 ;;
    20|26) DATASET=drivaer_ml_surface;      EXTRA="${COS} model.hidden=1536"; LR=5e-4; SLUG=cos-w1536-lr5e-4 ;;
    21|27) DATASET=drivaer_ml_surface;      EXTRA="${COS} model.hidden=1536"; LR=1e-3; SLUG=cos-w1536-lr1e-3 ;;
    22|28) DATASET=drivaer_ml_surface;      EXTRA="forward_kwargs.global_embedding=global_data.U_inf_dir model.n_hidden=2048"; LR=1e-3; SLUG=gt-w2048-lr1e-3 ;;
    23|29) DATASET=drivaer_ml_surface;      EXTRA="forward_kwargs.global_embedding=global_data.U_inf_dir model.n_hidden=2048"; LR=3e-4; SLUG=gt-w2048-lr3e-4 ;;
    *) echo "bad array index $1" >&2; exit 2 ;;
  esac
  MODEL=isla_surface_reference
  case "${SLUG}" in gt-*) MODEL=geotransolver_surface ;; esac
  if [ "$1" -le 8 ] || { [ "$1" -ge 18 ] && [ "$1" -le 23 ]; }; then SEED=42; else SEED=43; fi
  CELLS=100000; EPOCHS=500
  RUN_ID="rn_${SLUG//-/_}_c100k_seed${SEED}"
  SLUG="${SLUG}-c100k-s${SEED}"
}
if [ "${1:-}" = "--slug" ]; then lane "$2"; echo "${SLUG}"; exit 0; fi
if [ "${1:-}" = "--manifest" ]; then
  for i in $(seq "${2:-0}" "${3:-29}"); do lane "$i"; printf "rn\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfalse\t\t\n" "$i" "${RUN_ID}" "${DATASET}" "${MODEL}" "${EXTRA}" "${LR}" "${SEED}"; done
  exit 0
fi
lane "${SLURM_ARRAY_TASK_ID}"
TAG="routenorm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PHYSICSNEMO_DIST_TIMEOUT_S=3600
export PYTHONPATH="${CODE}:${RECIPE}/src"
trap 'rc=$?; printf "rc=%s\n" "$rc" > "${TASK}/STATUS_${TAG}"' EXIT
RUN_DIR="${TASK}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
echo "=== link start $(date -Is) job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME} node=$(hostname) lane=${SLURM_ARRAY_TASK_ID} run=${RUN_ID} code=${CODE} dataset=${DATASET} model=${MODEL} extra='${EXTRA}' lr=${LR} seed=${SEED} cells=${CELLS} epochs=${EPOCHS}" >> "${RUN_DIR}/train.log"
if [ -f "${TASK}/BLOCKED_${RUN_ID}" ]; then echo "lane ${RUN_ID} BLOCKED; no-op"; exit 0; fi
if grep -q "Training completed" "${RUN_DIR}/train.log" 2>/dev/null; then printf "DONE\n" > "${TASK}/RUN_${TAG}"; exit 0; fi
cd "${TASK}" && sbatch -J "${SLURM_JOB_NAME}" --array="${SLURM_ARRAY_TASK_ID}" --dependency=singleton "${SELF}" >> "${TASK}/sbatch_logs/routenorm_resubmit.log" 2>&1 || true
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
