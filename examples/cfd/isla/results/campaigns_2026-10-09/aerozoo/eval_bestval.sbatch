#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.az.bestval-eval
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
#SBATCH --open-mode=append
# AEROZOO diagnostic scoring of a CHOSEN checkpoint per lane (2026-10-11): every ISLA lane's validation loss reaches its minimum at
# epoch 30-60 and then rises two- to four-fold and stays there (the DeepJEB++ signature), so the final checkpoint the protocol scores
# is not the lane's best model. This scores the checkpoint directory ${CKPT_ROOT}/<run>/checkpoints (a directory of symlinks to the
# chosen epoch's checkpoint.0.E.pt, <Model>.0.E.mdlus and norm_stats.pt, built by hand) on the VALIDATION side, float32, the training
# draw, exactly like the fleet scorer, into ${OUT_ROOT}/<run>/<run>/metrics.jsonl. It is a labeled diagnostic, never the pre-registered
# standing: the test side is not touched here.
#   CKPT_ROOT (default $T/runs_bestval), OUT_ROOT (default $T/p200k_fp32_bestval), LANES_TSV; array index = lanes.tsv idx (one lane per
#   task, GPU 0; four lanes per node when --array lists four consecutive rows is not attempted: one lane per node keeps it simple).
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe-td14
RECIPE="${TASK}/recipe_v9"
LANES="${LANES_TSV:-${TASK}/kernel/aerozoo/lanes.tsv}"
### BESTVAL_SIDE=val (default) scores the validation side into p200k_fp32_bestval; BESTVAL_SIDE=test scores the test side into
### p200k_fp32_bestval_test (the best-checkpoint table's protocol: the epoch is chosen by the trainer's validation loss, the score is
### read on the test side, for every arm alike).
SIDE="${BESTVAL_SIDE:-val}"
CKPT_ROOT="${CKPT_ROOT:-${TASK}/runs_bestval}"; OUT_ROOT="${OUT_ROOT:-${TASK}/p200k_fp32_bestval$([ "${SIDE}" = test ] && echo _test)}"; mkdir -p "${OUT_ROOT}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
export TRITON_CACHE_DIR="${TASK}/.triton_cache" TORCHINDUCTOR_CACHE_DIR="${TASK}/.inductor_cache"
cd "${RECIPE}"
unit() {
  local idx="$1" gpu="$2"
  local row; row=$(awk -F"\t" -v i="${idx}" 'NR>1 && $1==i {print; exit}' "${LANES}")
  [ -z "${row}" ] && return 0
  local IDX GROUP RUN DATASET MODEL EXTRA LR SEED CELLS CODEV EPOCHS
  IFS=$'\t' read -r IDX GROUP RUN DATASET MODEL EXTRA LR SEED CELLS CODEV EPOCHS <<<"${row}"
  [ "${EXTRA}" = "-" ] && EXTRA=""
  local OUT="${OUT_ROOT}/${RUN}"
  [ -f "${OUT}/.done" ] && { echo "lane ${idx} ${RUN}: already scored"; return 0; }
  [ -d "${CKPT_ROOT}/${RUN}/checkpoints" ] || { echo "lane ${idx} ${RUN}: no chosen checkpoint under ${CKPT_ROOT}; skipped"; return 0; }
  local CODE="${TASK}/${CODEV}"; export PYTHONPATH="${CODE}:${RECIPE}/src"
  mkdir -p "${OUT}"; printf "%s\n" "${CODE}" > "${OUT}/SNAPSHOT_USED"; ls -l "${CKPT_ROOT}/${RUN}/checkpoints" > "${OUT}/CHECKPOINT_USED"
  EXTRA=$(sed -E "s/(^| )training\.num_epochs=[^ ]+//g; s/(^| )\+training\.[^ ]+//g" <<<"${EXTRA}")
  ### BESTVAL_SIDE=test: the same selection as the test-side scorer eval_test.sbatch: the lane override val_split=<split>.val
  ### becomes <split>.test and infer.py reads infer_split=test. (Fixed 2026-10-13: before this, SIDE only changed OUT_ROOT and the
  ### test-side job re-scored the validation split; those outputs were deleted.)
  local INFER_SPLIT=val
  if [ "${SIDE}" = test ]; then
    EXTRA=$(sed -E "s/val_split=([A-Za-z0-9_]+)\.val/val_split=\1.test/" <<<"${EXTRA}")
    grep -q "val_split=[A-Za-z0-9_]*\.test" <<<"${EXTRA}" || { echo "lane ${idx} ${RUN}: no <split>.val override to turn into .test; refusing" >> "${OUT}.log"; return 0; }
    INFER_SPLIT=test
  fi
  echo "=== lane ${idx} START $(date -Is) run=${RUN} ckpt=${CKPT_ROOT}/${RUN} extra='${EXTRA}' gpu=${gpu} code=${CODE}" >> "${OUT}.log"
  timeout 6600 env CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((24000 + SLURM_JOB_ID % 20000 + gpu)) \
    "${VENV}/bin/python" "${RECIPE}/src/infer.py" model="${MODEL}" ${EXTRA} dataset="${DATASET}" run_id="${RUN}" infer_split=${INFER_SPLIT} +save_predictions=false \
    sampling_resolution="${CELLS}" precision=float32 checkpoint_dir="${CKPT_ROOT}" output_dir="${OUT}" >> "${OUT}.log" 2>&1 \
    && touch "${OUT}/.done" || echo "=== ${RUN} FAILED rc=$?" >> "${OUT}.log"
  rm -rf "${OUT}"/*/predictions
}
### One job, up to four lanes in parallel on the node's four GPUs: ROWS="1 4 5" names the lanes.tsv rows (the short queue counts every
### array task against its 20-job cap, so a single multi-lane job is used instead of an array).
g=0
for idx in ${ROWS:?set ROWS to the lanes.tsv row indices}; do ( unit "${idx}" "${g}" ) & g=$((g + 1)); done
wait
echo "AZ-BESTVAL-EVAL-DONE ${SLURM_JOB_ID} rows='${ROWS}' done=$(find "${OUT_ROOT}" -name .done | wc -l)"
