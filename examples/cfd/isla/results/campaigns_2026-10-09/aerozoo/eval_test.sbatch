#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.az.test-eval
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
# AEROZOO test-side scoring (2026-10-10; #sec-nb-aerozoo-prereg: "model selection on the validation side only; the test
# side scored once"). For every lane of kernel/aerozoo/lanes.tsv whose training finished and whose test score is absent,
# float32 inference at the training draw (200,000 interior points, 50,000 faces per boundary type) on the TEST part of the
# lane's own split (`val_split=<split>.val` in the lane's overrides becomes `<split>.test`; for the held-out splits that
# part is the held-out family). Four lanes per array task, one GPU each, exactly like the fleet scorer
# (kernel/p200k_fleet/p200k_eval_aga.sbatch), which keeps scoring the validation sides. Idempotent through
# p200k_fp32_test/<run>/.done; a lane that is not finished is skipped and picked up by a later submission.
#   submit: bash kernel/aerozoo/submit.sh test        (array 0-3 covers the sixteen rows; LANE_QOS / LANE_TIME honored)
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe-td14
CODE="${TASK}/code_v26"; RECIPE="${TASK}/recipe_v9"
LANES="${LANES_TSV:-${TASK}/kernel/aerozoo/lanes.tsv}"
ROOT="${TASK}/p200k_fp32_test"; mkdir -p "${ROOT}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
export PYTHONPATH="${CODE}:${RECIPE}/src"
export TRITON_CACHE_DIR="${TASK}/.triton_cache" TORCHINDUCTOR_CACHE_DIR="${TASK}/.inductor_cache"
cd "${RECIPE}"
unit() {
  local idx="$1" gpu="$2"
  local row; row=$(awk -F"\t" -v i="${idx}" 'NR>1 && $1==i {print; exit}' "${LANES}")
  [ -z "${row}" ] && return 0
  local IDX GROUP RUN DATASET MODEL EXTRA LR SEED CELLS CODEV EPOCHS
  IFS=$'\t' read -r IDX GROUP RUN DATASET MODEL EXTRA LR SEED CELLS CODEV EPOCHS <<<"${row}"
  [ "${EXTRA}" = "-" ] && EXTRA=""
  local OUT="${ROOT}/${RUN}"
  [ -f "${OUT}/.done" ] && { echo "lane ${idx} ${RUN}: test side already scored"; return 0; }
  grep -q "Training completed" "${TASK}/runs/${RUN}/train.log" 2>/dev/null || { echo "lane ${idx} ${RUN}: training not finished; skipped"; return 0; }
  if [ "$(grep -c START "${OUT}.log" 2>/dev/null)" -ge 3 ]; then
    mkdir -p "${OUT}"; echo "gave up after 3 failed scoring attempts; see ${OUT}.log" > "${OUT}/.done"; return 0
  fi
  ### The lane's own snapshot (the tsv's code column): code_v26 for ISLA and GeoTransolver, code_dtt16 (a symlink to the Barnes-Hut
  ### fork's extracted snapshot, psharpe/isla-dtt 6094c4e9d) for the ISLA-DTT lanes; the recipe is recipe_v9 for all.
  local CODE="${TASK}/${CODEV}"; export PYTHONPATH="${CODE}:${RECIPE}/src"
  mkdir -p "${OUT}"; printf "%s\n" "${CODE}" > "${OUT}/SNAPSHOT_USED"
  ### training-only overrides out; the validation part becomes the test part of the same split.
  EXTRA=$(sed -E "s/(^| )training\.num_epochs=[^ ]+//g; s/(^| )\+training\.[^ ]+//g" <<<"${EXTRA}")
  EXTRA=$(sed -E "s/val_split=([A-Za-z0-9_]+)\.val/val_split=\1.test/" <<<"${EXTRA}")
  grep -q "val_split=[A-Za-z0-9_]*\.test" <<<"${EXTRA}" || { echo "lane ${idx} ${RUN}: no <split>.val override to turn into .test; refusing" >> "${OUT}.log"; return 0; }
  echo "=== lane ${idx} START $(date -Is) run=${RUN} dataset=${DATASET} model=${MODEL} extra='${EXTRA}' gpu=${gpu} code=${CODE}" >> "${OUT}.log"
  timeout 6600 env CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((24000 + SLURM_JOB_ID % 20000 + gpu)) \
    "${VENV}/bin/python" "${RECIPE}/src/infer.py" model="${MODEL}" ${EXTRA} dataset="${DATASET}" run_id="${RUN}" infer_split=test +save_predictions=false \
    sampling_resolution="${CELLS}" precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${OUT}" >> "${OUT}.log" 2>&1 \
    && touch "${OUT}/.done" || echo "=== ${RUN} FAILED rc=$?" >> "${OUT}.log"
  rm -rf "${OUT}"/*/predictions
}
first=$((SLURM_ARRAY_TASK_ID * 4))
for g in 0 1 2 3; do ( unit $((first + g)) "${g}" ) & done
wait
echo "AZ-TEST-EVAL-TASK-DONE ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} lanes ${first}-$((first + 3)) done=$(find "${ROOT}" -name .done | wc -l)"
