#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.h090.savepred
#SBATCH --time=00:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/h090/spatial/savepred-%j.log
# HILIFT-090 spatial probe (2026-10-20): re-run the HiLift native scoring of four arms (A2 ISLA seeds 42/43, GeoTransolver and
# Transolver unit-drive seed 42) exactly as their score rows (kernel/a2r_hl/score_rows.tsv, kernel/udrv_hl/score_rows.tsv:
# code_a2r / recipe_a2r_hl, float32, infer_split=val, 200k cells, epoch-500 checkpoints), but KEEPING the per-sample
# predictions (++save_predictions=true) and logging the reader's block draws (infer_wrap.py), one arm per GPU.
# Output: kernel/h090/spatial/out/<run>/ (metrics.jsonl + predictions/*.pdmsh), blocks/<run>.<pid>. Deleted after reduction.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
SP=$TASK/kernel/h090/spatial
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
echo "=== savepred start $(date -Is) job=${SLURM_JOB_ID} node=$(hostname)"
mkdir -p $SP/out $SP/blocks
cd $TASK
unit() {
  local rows="$1" run="$2" gpu="$3"
  local ROW; ROW=$(awk -F"\t" -v r="${run}" 'NR>1 && $2==r && $3=="native" {print; exit}' "${rows}")
  local RUN PROTO DATASET MODEL EXTRA CODE RECIPE_DIR CELLS PYV
  IFS=$'\t' read -r _ RUN PROTO DATASET MODEL EXTRA CODE RECIPE_DIR CELLS PYV <<<"${ROW}"
  [ "${EXTRA}" = "-" ] && EXTRA=""
  EXTRA=$(sed -E "s/(^| )training\.num_epochs=[^ ]+//g" <<<"${EXTRA}")
  local OUT=$SP/out/${RUN}; mkdir -p $OUT
  echo "=== ${RUN} START $(date -Is) model=${MODEL} extra='${EXTRA}' code=${CODE} recipe=${RECIPE_DIR} cells=${CELLS} gpu=${gpu}"
  BLOCKLOG=$SP/blocks/${RUN} timeout 3000 env PYTHONPATH="${TASK}/${CODE}:${TASK}/${RECIPE_DIR}/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${PYV}/bin/python" $SP/infer_wrap.py "${TASK}/${RECIPE_DIR}/src/infer.py" model="${MODEL}" ${EXTRA} dataset="${DATASET}" run_id="${RUN}" infer_split=val ++save_predictions=true \
    sampling_resolution=${CELLS} precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${OUT}" > $SP/out/${RUN}.log 2>&1
  echo "=== ${RUN} END rc=$? $(date -Is) samples=$(grep -c infer_step ${OUT}/${RUN}/metrics.jsonl 2>/dev/null)"
}
unit $TASK/kernel/a2r_hl/score_rows.tsv a2r_hl_isla_seed42 0 &
unit $TASK/kernel/a2r_hl/score_rows.tsv a2r_hl_isla_seed43 1 &
unit $TASK/kernel/udrv_hl/score_rows.tsv udrv_hl_gt_seed42 2 &
unit $TASK/kernel/udrv_hl/score_rows.tsv udrv_hl_t_seed42 3 &
wait
du -sh $SP/out; echo "=== savepred end $(date -Is)"
