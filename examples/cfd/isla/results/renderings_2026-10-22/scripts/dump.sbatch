#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.render090.dump
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=nvl72d015-T12,nvl72d058-T08,nvl72d052-T01
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-22/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-22/sbatch_logs/%x-%j.log
# RENDER-090 (2026-10-22): truth and predictions of the book's rendered cases from the current standings' checkpoints.
# renderings_2026-10-18/scripts/dump.sbatch with this lane's folder (book_figs_2026-10-22), the job name, FIG_REPLAY (column
# replay of fig_rows.tsv: dump_wrap.py replays the scorer's reads so the case's random draw is the scorer's), the rows of ONLY
# dealt round-robin over the four GPUs, and rows whose ckpt_root is REUSE:<row> skipped (their checkpoints are no longer on LHR;
# the 2026-10-18 dump of that row, the same run and case, stands in). Each row is its scorer's inference command (score_rows.tsv
# of its scorer: dataset, model, extra, code and recipe snapshots, cells, float32, infer_split=val, the final checkpoint and the
# run's own norm_stats.pt), one GPU per row. Inference only; snapshots are read, never written.
#   ONLY=0,1,2 sbatch dump.sbatch
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
F="${TASK}/book_figs_2026-10-22"; S="${F}/scripts"; ROWS="${ROWS:-${S}/fig_rows.tsv}"; ONLY="${ONLY:-}"
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
export TRITON_CACHE_DIR=/tmp/$USER/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor
cd "${TASK}"
unit() {
  local ROW="$1" gpu="$2"
  local IDX TAG RUN ROOT PROTO DATASET MODEL EXTRA CODE RECIPE CELLS SAMPLES REPLAY SCORER
  IFS=$'\t' read -r IDX TAG RUN ROOT PROTO DATASET MODEL EXTRA CODE RECIPE CELLS SAMPLES REPLAY SCORER <<<"${ROW}"
  [ "${EXTRA}" = "-" ] && EXTRA=""
  case "${ROOT}" in REUSE:*) echo "=== row ${IDX} SKIP: ${ROOT} (2026-10-18 dump)"; return 0 ;; esac
  local CK="${TASK}/runs"
  local DS="${DATASET}"; [ "${PROTO}" != "native" ] && DS="${DATASET}_${PROTO}"
  local OUT="${F}/dump/${TAG}/${PROTO}"; mkdir -p "${OUT}"
  [ -f "${OUT}/.done" ] && return 0
  if ! grep -q "Training completed" "${CK}/${RUN}/train.log" 2>/dev/null; then
    echo "=== row ${IDX} NO-OP $(date -Is): no 'Training completed' in runs/${RUN}/train.log" | tee -a "${OUT}/dump.log"; return 0
  fi
  echo "=== row ${IDX} START $(date -Is) job=${SLURM_JOB_ID} gpu=${gpu} run=${RUN} protocol=${PROTO} dataset=${DS} model=${MODEL} extra='${EXTRA}' cells=${CELLS} code=${CODE} recipe=${RECIPE} venv=${VENV} ckpt=${CK} samples=${SAMPLES} replay=${REPLAY}" | tee -a "${OUT}/dump.log"
  timeout 6600 env FIG_SAMPLES="${SAMPLES}" FIG_REPLAY="${REPLAY}" FIG_OUT="${OUT}" PYTHONPATH="${TASK}/${CODE}:${TASK}/${RECIPE}/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${VENV}/bin/python" "${S}/dump_wrap.py" --config-path "${TASK}/${RECIPE}/conf" --config-name infer model="${MODEL}" ${EXTRA} dataset="${DS}" run_id="${RUN}" infer_split=val +save_predictions=true \
    sampling_resolution=${CELLS} precision=float32 checkpoint_dir="${CK}" output_dir="${OUT}" >> "${OUT}/dump.log" 2>&1 \
    && touch "${OUT}/.done" || echo "=== row ${IDX} FAILED rc=$?" | tee -a "${OUT}/dump.log"
  echo "=== row ${IDX} END $(date -Is) $(ls ${OUT}/*.npz 2>/dev/null | wc -l) npz" | tee -a "${OUT}/dump.log"
}
mapfile -t SEL < <(tail -n +2 "${ROWS}" | while IFS= read -r ROW; do
  IDX=$(cut -f1 <<<"${ROW}")
  if [ -n "${ONLY}" ] && ! grep -qw "${IDX}" <<<"${ONLY//,/ }"; then continue; fi
  printf '%s\n' "${ROW}"
done)
worker() {
  local g="$1" k
  for ((k = g; k < ${#SEL[@]}; k += 4)); do unit "${SEL[k]}" "${g}"; done
}
for g in 0 1 2 3; do worker "${g}" & done
wait
echo "RENDER090-DUMP-DONE $(date -Is) npz=$(find ${F}/dump -name '*.npz' | wc -l)"
