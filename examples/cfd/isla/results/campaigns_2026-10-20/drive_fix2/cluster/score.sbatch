#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.ud2.score
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/score_%A_%a.log
#SBATCH -e sbatch_logs/score_%A_%a.log
#SBATCH --open-mode=append
# DRIVE-FIX2 (2026-10-19): COPY of kernel/r80_ud/score.sbatch (md5 d5d16eef), itself a copy of kernel/a2r/score.sbatch (md5 8e238045), for the ten DRIVE-FIX2 unit-drive lanes (kernel/cl_ud and kernel/r80_ud2): default root kernel/cl_ud (rows score_rows.tsv here), output tree evals_ud2, job name isla.ud2.score. Nothing else changed.
# A2-RERUN scorer (2026-10-17): a COPY of kernel/char/protocol_eval.sbatch (md5 812b810b..., the CHAR-PROTOCOLS scorer that
# produced results/char/*.json) for the H-RESID A2 reruns. Same inference command (recipe src/infer.py, float32, infer_split=val,
# the lane's dataset yaml or its <dataset>_area / <dataset>_posed twin, the final checkpoint, one GPU per row, four rows per
# array task). Differences, none of which touches the inference command: the rows table carries dataset, model, extra, code
# snapshot, recipe snapshot, cell count and venv explicitly (no manifest / run-prefix lookup), and the root and output tree are
# this directory's (A2R_ROOT) and evals_a2r/<run_id>/<protocol>/ (EVAL_ROOT).
# A2R_TASK (default the LHR/AGA campaign root) selects the task root, e.g. the JHB task dir; the venv is the rows table's last column.
# Rows: score_rows.tsv (idx run_id protocol dataset model extra code recipe cells venv).
#   sbatch --array=<blocks> score.sbatch     (block b = rows 4b..4b+3)
set -uo pipefail
ulimit -s 8192
TASK="${A2R_TASK:-/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0}"   # JHB: sbatch --export=ALL,A2R_TASK=$J ...
A2R_ROOT="${A2R_ROOT:-${TASK}/kernel/cl_ud}"
ROWS="${SCORE_ROWS:-${A2R_ROOT}/score_rows.tsv}"
ROOT="${EVAL_ROOT:-${TASK}/evals_ud2}"; mkdir -p "${ROOT}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
cd "${TASK}"
unit() {
  local idx="$1" gpu="$2"
  local ROW; ROW=$(awk -F"\t" -v i="${idx}" 'NR>1 && $1==i {print; exit}' "${ROWS}")
  [ -z "${ROW}" ] && return 0
  local RUN PROTO DATASET MODEL EXTRA CODE RECIPE_DIR CELLS PYV
  IFS=$'\t' read -r _ RUN PROTO DATASET MODEL EXTRA CODE RECIPE_DIR CELLS PYV <<<"${ROW}"
  [ "${EXTRA}" = "-" ] && EXTRA=""
  CODE="${TASK}/${CODE}"; RECIPE_DIR="${TASK}/${RECIPE_DIR}"
  local OUT="${ROOT}/${RUN}/${PROTO}"
  [ -f "${OUT}/.done" ] && return 0
  mkdir -p "${OUT}"
  if [ "$(grep -c START "${OUT}.log" 2>/dev/null)" -ge 3 ]; then
    echo "gave up after 3 failed scoring attempts; see ${OUT}.log" > "${OUT}/.done"; return 0
  fi
  if ! grep -q "Training completed" "${TASK}/runs/${RUN}/train.log" 2>/dev/null; then
    echo "=== lane ${idx} NO-OP $(date -Is) run=${RUN} protocol=${PROTO}: no 'Training completed' in runs/${RUN}/train.log" >> "${OUT}.log"; return 0
  fi
  local DS="${DATASET}"; case "${PROTO}" in native) ;; area|posed) DS="${DATASET}_${PROTO}" ;; *) echo "=== lane ${idx} FAILED: unknown protocol '${PROTO}'" >> "${OUT}.log"; return 0 ;; esac
  if [ ! -f "${RECIPE_DIR}/datasets/${DS}.yaml" ]; then echo "=== lane ${idx} FAILED $(date -Is) run=${RUN} protocol=${PROTO}: ${RECIPE_DIR}/datasets/${DS}.yaml missing" >> "${OUT}.log"; return 0; fi
  printf "%s\n" "${CODE}" > "${OUT}/SNAPSHOT_USED"; printf "%s\n" "${DS}" > "${OUT}/DATASET_USED"
  EXTRA=$(sed -E "s/(^| )training\.num_epochs=[^ ]+//g" <<<"${EXTRA}")
  echo "=== lane ${idx} START $(date -Is) job=${SLURM_JOB_ID} run=${RUN} protocol=${PROTO} dataset=${DS} model=${MODEL} extra='${EXTRA}' cells=${CELLS} gpu=${gpu} code=${CODE} recipe=${RECIPE_DIR} venv=${PYV}" >> "${OUT}.log"
  timeout 9000 env PYTHONPATH="${CODE}:${RECIPE_DIR}/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${PYV}/bin/python" "${RECIPE_DIR}/src/infer.py" model="${MODEL}" ${EXTRA} dataset="${DS}" run_id="${RUN}" infer_split=val +save_predictions=false \
    sampling_resolution=${CELLS} precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${OUT}" >> "${OUT}.log" 2>&1 \
    && touch "${OUT}/.done" || echo "=== ${RUN} ${PROTO} FAILED rc=$?" >> "${OUT}.log"
  if [ ! -f "${OUT}/.done" ] && grep -q "Inference summary is inconsistent" "${OUT}.log" && grep -q '"phase": "infer_step"' "${OUT}/${RUN}/metrics.jsonl" 2>/dev/null; then
    echo "summary refused by infer.py (2-D *_z_l2 expected key); per-sample rows complete" > "${OUT}/.done"
  fi
  rm -rf "${OUT}"/*/predictions
}
first=$((SLURM_ARRAY_TASK_ID * 4))
for g in 0 1 2 3; do ( unit $((first + g)) "${g}" ) & done
wait
echo "A2R-SCORE-TASK-DONE ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} rows ${first}-$((first + 3)) done=$(find "${ROOT}" -name .done | wc -l)"
