#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.fig.dump
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-18/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-18/sbatch_logs/%x-%j.log
# FIG lane (ISLA book overhaul, 2026-10-18): truth and predictions of chosen validation samples, for the book's renderings.
# Each row of fig_rows.tsv is the CHAR / A2-RERUN scorer's inference command for that run (kernel/char/protocol_eval.sbatch,
# kernel/a2r/score.sbatch, kernel/airfrans_u/score.sbatch: recipe src/infer.py, float32, infer_split=val, the lane's
# dataset yaml or its _posed twin, the final checkpoint, the scorer's cell count, one GPU per row), run through dump_wrap.py,
# which restricts the sampler to the row's samples and writes .npz instead of .pdmsh. Inference only.
# ckpt_root F = this dir's runs/ (final .mdlus + norm_stats.pt copied read-only from AGA / JHB), R = $T/runs.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
F="${TASK}/book_figs_2026-10-18"; S="${F}/scripts"; ROWS="${ROWS:-${S}/fig_rows.tsv}"; ONLY="${ONLY:-}"
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
export TRITON_CACHE_DIR=/tmp/$USER/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor
cd "${TASK}"
unit() {
  local ROW="$1" gpu="$2"
  local IDX TAG RUN ROOT PROTO DATASET MODEL EXTRA CODE RECIPE CELLS SAMPLES
  IFS=$'\t' read -r IDX TAG RUN ROOT PROTO DATASET MODEL EXTRA CODE RECIPE CELLS SAMPLES <<<"${ROW}"
  [ "${EXTRA}" = "-" ] && EXTRA=""
  local CK="${TASK}/runs"; [ "${ROOT}" = "F" ] && CK="${F}/runs"
  local DS="${DATASET}"; [ "${PROTO}" != "native" ] && DS="${DATASET}_${PROTO}"
  local PYV="${VENV}"; case "${CODE}" in code_v2[2-9]|code_v[3-9]*) PYV="${VENV}-td14" ;; esac
  local OUT="${F}/dump/${TAG}/${PROTO}"; mkdir -p "${OUT}"
  [ -f "${OUT}/.done" ] && return 0
  echo "=== row ${IDX} START $(date -Is) job=${SLURM_JOB_ID} gpu=${gpu} run=${RUN} protocol=${PROTO} dataset=${DS} model=${MODEL} extra='${EXTRA}' cells=${CELLS} code=${CODE} recipe=${RECIPE} venv=${PYV} ckpt=${CK} samples=${SAMPLES}" | tee -a "${OUT}/dump.log"
  timeout 3000 env FIG_SAMPLES="${SAMPLES}" FIG_OUT="${OUT}" PYTHONPATH="${TASK}/${CODE}:${TASK}/${RECIPE}/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${PYV}/bin/python" "${S}/dump_wrap.py" --config-path "${TASK}/${RECIPE}/conf" --config-name infer model="${MODEL}" ${EXTRA} dataset="${DS}" run_id="${RUN}" infer_split=val +save_predictions=true \
    sampling_resolution=${CELLS} precision=float32 checkpoint_dir="${CK}" output_dir="${OUT}" >> "${OUT}/dump.log" 2>&1 \
    && touch "${OUT}/.done" || echo "=== row ${IDX} FAILED rc=$?" | tee -a "${OUT}/dump.log"
  echo "=== row ${IDX} END $(date -Is) $(ls ${OUT}/*.npz 2>/dev/null | wc -l) npz" | tee -a "${OUT}/dump.log"
}
worker() {
  local g="$1"
  tail -n +2 "${ROWS}" | while IFS= read -r ROW; do
    local IDX; IDX=$(cut -f1 <<<"${ROW}")
    if [ -n "${ONLY}" ] && ! grep -qw "${IDX}" <<<"${ONLY//,/ }"; then continue; fi
    [ $((IDX % 4)) -eq "${g}" ] && unit "${ROW}" "${g}"
  done
}
for g in 0 1 2 3; do worker "${g}" & done
wait
echo "FIG-DUMP-DONE $(date -Is) npz=$(find ${F}/dump -name '*.npz' | wc -l)"
