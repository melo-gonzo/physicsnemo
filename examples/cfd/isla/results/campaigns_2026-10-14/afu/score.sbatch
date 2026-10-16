#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.afu.score
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/airfrans_u/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/airfrans_u/sbatch_logs/%x-%j.log
#SBATCH --open-mode=append
# AF-SCORE (LHR, 2026-10-15): the CHAR-PROTOCOLS scorer (kernel/char/protocol_eval.sbatch, the AGA copy that scored the
# reference lanes cl_r_af_* in rows 330-377 on 2026-10-04/05) for the AF-UNIFORM lanes afu_* of kernel/airfrans_u/lanes.tsv.
# Per protocol: float32 inference of the lane's final checkpoint (the latest in runs/<run>/checkpoints, epoch 500) on the
# validation split (full_test, 200 cases) at sampling_resolution=100000, with the lane's model and extra overrides and
#   native  the lane's own dataset yaml (airfrans_uvolume or airfrans_uvolume_2d)
#   area    <dataset>_area.yaml
#   posed   <dataset>_posed.yaml   every case rigidly rotated and translated by a deterministic per-case pose
# on recipe_v8 and the lane's code snapshot (afu_r_* code_v20 with .venv-recipe, afu_dtt_* code_dtt16 with
# .venv-recipe-td14). The infer.py call, the overrides, the seed (infer.py's set_seed(training.seed) = 42 for every run),
# the three-strike rule, the 2-D summary rule and the predictions cleanup are the reference scorer's. What differs is
# only the packing: one job per run, its three protocols on GPUs 0-2 (the reference packed four table rows per task).
# Run: sbatch -J isla.afu.score.<slug> --export=ALL,AFU_SCORE_RUN=<run_id> score.sbatch (kernel/airfrans_u/score_submit.sh).
# Output: ${AFU_SCORE_ROOT:-$T/evals_afu}/<run>/<protocol>/<run>/metrics.jsonl, a .done marker in <run>/<protocol>/,
# SNAPSHOT_USED and DATASET_USED beside it, and the merged log <run>/<protocol>.log.
# An unfinished lane (no "Training completed" in runs/<run>/train.log) is a no-op (logged, no .done; a later launch retries).
# Old checkpoints on the uniform twins (AF-SCORE decomposition, 2026-10-15): the block-window reference lanes cl_r_af_*
# (final checkpoints copied from AGA into $T/runs_from_aga, code_v20) and ISLA-DTT af_tg1pf_* (agents/2026-10-06-isla-dtt/runs,
# code_dtt16 on td14) are scored the same way from kernel/airfrans_u/oldckpt_lanes.tsv, whose dataset column is the lane's
# uniform twin (airfrans_uvolume{,_2d}), into $T/evals_afu_oldckpt; afu_* runs are unaffected.
# SMOKE TESTS ONLY: AFU_SCORE_CKPT=<checkpoints dir> evaluates that checkpoint copy and skips the completion check;
# AFU_SCORE_LANES / AFU_SCORE_ROOT redirect a smoke to its own lane table and output tree.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
AFU="${TASK}/kernel/airfrans_u"
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RECIPE_DIR="${TASK}/recipe_v8"
RUN="${AFU_SCORE_RUN:?set AFU_SCORE_RUN=<run_id>}"
### Per-run checkpoint tree, lane table and output tree.
RUNS="${TASK}/runs"; DEF_LANES="${AFU}/lanes.tsv"; DEF_ROOT="${TASK}/evals_afu"
case "${RUN}" in
  cl_r_af_*) RUNS="${TASK}/runs_from_aga" ;;
  af_tg1pf_*) RUNS=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-06-isla-dtt/runs ;;
esac
case "${RUN}" in cl_r_af_*|af_tg1pf_*) DEF_LANES="${AFU}/oldckpt_lanes.tsv"; DEF_ROOT="${TASK}/evals_afu_oldckpt" ;; esac
LANES="${AFU_SCORE_LANES:-${DEF_LANES}}"; ROOT="${AFU_SCORE_ROOT:-${DEF_ROOT}}"; CKPT="${AFU_SCORE_CKPT:-}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
mkdir -p "${ROOT}"
cd "${AFU}"   # Hydra's outputs/ go to kernel/airfrans_u/outputs
unit() {
  local PROTO="$1" gpu="$2"
  local OUT="${ROOT}/${RUN}/${PROTO}"
  [ -f "${OUT}/.done" ] && return 0
  mkdir -p "${OUT}"
  ### A row that has already failed three times fails deterministically; stop rescoring it (fleet scorer rule, 2026-09-29).
  if [ "$(grep -c START "${OUT}.log" 2>/dev/null)" -ge 3 ]; then
    echo "gave up after 3 failed scoring attempts; see ${OUT}.log" > "${OUT}/.done"; return 0
  fi
  ### Native checkpoints only: an unfinished lane is a no-op (no .done, so the next launch retries).
  if [ -z "${CKPT}" ] && ! grep -q "Training completed" "${RUNS}/${RUN}/train.log" 2>/dev/null; then
    echo "=== ${RUN} NO-OP $(date -Is) protocol=${PROTO}: no 'Training completed' in ${RUNS}/${RUN}/train.log" >> "${OUT}.log"; return 0
  fi
  local LANE; LANE=$(awk -F"\t" -v r="${RUN}" 'NR>1 && $3==r {print; exit}' "${LANES}")
  if [ -z "${LANE}" ]; then echo "=== ${RUN} FAILED $(date -Is): not in ${LANES}" >> "${OUT}.log"; return 0; fi
  local DATASET MODEL EXTRA LCELLS LCODE
  DATASET=$(cut -f4 <<<"${LANE}"); MODEL=$(cut -f5 <<<"${LANE}"); EXTRA=$(cut -f6 <<<"${LANE}"); [ "${EXTRA}" = "-" ] && EXTRA=""
  LCELLS=$(cut -f9 <<<"${LANE}"); LCODE=$(cut -f10 <<<"${LANE}")
  ### Code snapshot and cell count: the reference scorer's rule for the lanes these rerun (cl_r_af_*: code_v20 + recipe_v8,
  ### 100,000; ISLA-DTT af_tg1pf_*: code_dtt16 on the td14 venv + recipe_v8, 100,000), checked against the lane table.
  local CODE PYV="${VENV}" CELLS=100000
  case "${RUN}" in
    afu_r_*|cl_r_af_*) CODE="${TASK}/code_v20" ;;
    afu_dtt_*|af_tg1pf_*) CODE="${TASK}/code_dtt16"; PYV="${VENV}-td14" ;;
    *) echo "=== ${RUN} FAILED $(date -Is): not an afu_r_* / afu_dtt_* / cl_r_af_* / af_tg1pf_* run" >> "${OUT}.log"; return 0 ;;
  esac
  if [ "${TASK}/${LCODE}" != "${CODE}" ] || [ "${LCELLS}" != "${CELLS}" ]; then
    echo "=== ${RUN} FAILED $(date -Is): lane table says code=${LCODE} cells=${LCELLS}, scorer maps ${CODE} ${CELLS}" >> "${OUT}.log"; return 0
  fi
  local DS="${DATASET}"; case "${PROTO}" in native) ;; area|posed) DS="${DATASET}_${PROTO}" ;; *) echo "=== ${RUN} FAILED $(date -Is): unknown protocol '${PROTO}'" >> "${OUT}.log"; return 0 ;; esac
  if [ ! -f "${RECIPE_DIR}/datasets/${DS}.yaml" ]; then echo "=== ${RUN} FAILED $(date -Is) protocol=${PROTO}: ${RECIPE_DIR}/datasets/${DS}.yaml missing" >> "${OUT}.log"; return 0; fi
  printf "%s\n" "${CODE}" > "${OUT}/SNAPSHOT_USED"; printf "%s\n" "${DS}" > "${OUT}/DATASET_USED"
  EXTRA=$(sed -E "s/(^| )training\.num_epochs=[^ ]+//g" <<<"${EXTRA}")   # training-only override; infer.py has no such key
  EXTRA=$(sed -E "s/(^| )model\.attention_type=/\1+model.attention_type=/g" <<<"${EXTRA}")   # verbatim from the reference scorer
  local CKPT_ARG=""; [ -n "${CKPT}" ] && CKPT_ARG="checkpoint_path=${CKPT}"
  echo "=== ${RUN} START $(date -Is) job=${SLURM_JOB_ID:-} protocol=${PROTO} dataset=${DS} model=${MODEL} extra='${EXTRA}' cells=${CELLS} gpu=${gpu} code=${CODE} venv=${PYV} checkpoint_dir=${RUNS} ${CKPT_ARG}" >> "${OUT}.log"
  timeout 6600 env PYTHONPATH="${CODE}:${RECIPE_DIR}/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${PYV}/bin/python" "${RECIPE_DIR}/src/infer.py" model="${MODEL}" ${EXTRA} dataset="${DS}" run_id="${RUN}" infer_split=val +save_predictions=false \
    sampling_resolution=${CELLS} precision=float32 checkpoint_dir="${RUNS}" ${CKPT_ARG} output_dir="${OUT}" >> "${OUT}.log" 2>&1 \
    && touch "${OUT}/.done" || echo "=== ${RUN} ${PROTO} FAILED rc=$?" >> "${OUT}.log"
  ### Two-dimensional datasets: an infer.py that refuses its own summary after every per-sample row is on disk (the
  ### reference scorer's rule; code_v20 writes the 2-D summary) leaves the per-sample rows as the measurement. Mark done.
  if [ ! -f "${OUT}/.done" ] && grep -q "Inference summary is inconsistent" "${OUT}.log" && grep -q '"phase": "infer_step"' "${OUT}/${RUN}/metrics.jsonl" 2>/dev/null; then
    echo "summary refused by infer.py (2-D *_z_l2 expected key); per-sample rows complete" > "${OUT}/.done"
  fi
  rm -rf "${OUT}"/*/predictions   # per-sample DomainMesh predictions: file-count quota; metrics.jsonl is what every artifact reads
}
g=0
for p in native area posed; do ( unit "${p}" "${g}" ) & g=$((g + 1)); done
wait
echo "AFU-SCORE-DONE ${SLURM_JOB_ID:-} run=${RUN} root=${ROOT} done=$(find "${ROOT}/${RUN}" -name .done 2>/dev/null | wc -l)/3"
