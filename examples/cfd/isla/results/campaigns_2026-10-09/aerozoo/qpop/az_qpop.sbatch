#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.dtt.azqpop
#SBATCH --time=02:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-06-isla-dtt/sbatch_logs/%x-%j.log
# ISLA-DTT QUERY-POP on AeroZoo: az_qpop.py on the full-split lanes (RUNS, space separated), one process per run, all
# runs at once on GPUs 0..3 (run i on GPU i mod 4); output ${OUT}/<run>/<condition>/metrics.jsonl and summary.json.
#   ARMS               isla | gt | dtt per run, space separated, same order as RUNS (default: from the run id)
#   AZ_QPOP_CASES      score only the first N validation cases (smoke test)
#   AZ_QPOP_EPOCH      load checkpoint.0.<N> instead of the latest (smoke test; implies ALLOW_UNFINISHED)
#   ALLOW_UNFINISHED=1 score lanes whose train.log has no "Training completed" (default: no-op on them)
#   QPOP_OUT           output root (default scoring/aerozoo/qpop)
set -uo pipefail
ulimit -s 8192
D=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-06-isla-dtt
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
S="${D}/scoring/aerozoo"
VENV="${VENV:-/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe-td14}"
OUT="${QPOP_OUT:-${S}/qpop}"; mkdir -p "${OUT}/logs"
RUNS="${RUNS:-az_full_isla_seed42 az_full_isla_seed43 az_full_gt_seed42 az_full_gt_seed43 az_full_dtt_seed42 az_full_dtt_seed43}"
read -r -a RUN_LIST <<<"${RUNS}"
read -r -a ARM_LIST <<<"${ARMS:-}"
if [ "${#ARM_LIST[@]}" -ne 0 ] && [ "${#ARM_LIST[@]}" -ne "${#RUN_LIST[@]}" ]; then echo "ARMS has ${#ARM_LIST[@]} entries for ${#RUN_LIST[@]} runs" >&2; exit 2; fi
FLAGS=""; [ -n "${AZ_QPOP_CASES:-}" ] && FLAGS+=" --n-cases ${AZ_QPOP_CASES}"; [ -n "${AZ_QPOP_EPOCH:-}" ] && FLAGS+=" --epoch ${AZ_QPOP_EPOCH}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
### Kernel caches in this task dir, not the 10 GB home (ISLA's fused Triton kernel and the per-block torch.compile).
export TRITON_CACHE_DIR="${S}/.cache/triton" TORCHINDUCTOR_CACHE_DIR="${S}/.cache/inductor"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
cd "${T}/recipe_v9"
run() {
  local RUN="$1" ARM="$2" GPU="$3"
  ### The lane's own library snapshot: code_dtt16 for the tree-read lanes, code_v26 for ISLA and GeoTransolver.
  local CODE="${T}/code_v26"; [ "${ARM}" = "dtt" ] && CODE="${D}/code_dtt16"
  if [ -z "${AZ_QPOP_EPOCH:-}" ] && [ -z "${ALLOW_UNFINISHED:-}" ] && ! grep -q "Training completed" "${T}/runs/${RUN}/train.log" 2>/dev/null; then
    echo "=== NO-OP run=${RUN}: not finished"; return 0
  fi
  echo "=== START $(date -Is) run=${RUN} arm=${ARM} gpu=${GPU} code=${CODE} flags='${FLAGS}'"
  env PYTHONPATH="${CODE}:${T}/recipe_v9/src" CUDA_VISIBLE_DEVICES="${GPU}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${VENV}/bin/python" "${S}/az_qpop.py" "${RUN}" "${OUT}/${RUN}" "${ARM}" ${FLAGS} > "${OUT}/logs/${RUN}.log" 2>&1 \
    && echo "=== DONE $(date -Is) run=${RUN}" || echo "=== FAILED rc=$? $(date -Is) run=${RUN}"
}
for i in "${!RUN_LIST[@]}"; do
  RUN="${RUN_LIST[$i]}"
  ARM="${ARM_LIST[$i]:-}"
  if [ -z "${ARM}" ]; then case "${RUN}" in *_dtt_*) ARM=dtt ;; *_gt_*) ARM=gt ;; *) ARM=isla ;; esac; fi
  run "${RUN}" "${ARM}" $((i % 4)) &
done
wait
echo "AZ-QPOP-TASK-DONE ${SLURM_JOB_ID}"
