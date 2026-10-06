#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.lugprec
#SBATCH --time=01:30:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/lug_prec_%A_%a.log
#SBATCH -e sbatch_logs/lug_prec_%A_%a.log
# LUG-PREC diagnostic (2026-10-05): score the lug campaign lanes in bfloat16 (the training precision) with the same scorer
# that produced the float32 record, to separate a precision sensitivity of the nearest-boundary-sample geometry from a
# configuration effect. Output: lug_prec/<precision>/<run>/<run>/metrics.jsonl. Four lanes per array task.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
CODE="${TASK}/code_v20"; RECIPE_DIR="${TASK}/recipe_v8"; PREC="${LUG_PREC:-bfloat16}"
LANES=(cl_lug_isla_seed42:isla_lug:768 cl_lug_isla_seed43:isla_lug:768 cl_r_lug_isla_seed42:isla_lug:512 cl_r_lug_isla_seed43:isla_lug:512 cl_lug_gt_seed42:geotransolver_lug:1024 cl_r_lug_gt_seed42:geotransolver_lug:256 cl_lug_t_seed42:transolver_lug:1024 cl_r_lug_t_seed42:transolver_lug:512)
unit() {
  local spec="$1" gpu="$2"; IFS=: read -r RUN MODEL W <<<"$spec"
  local EXTRA="train_split=train val_split=test"
  case "$MODEL" in isla_lug) EXTRA="$EXTRA model.hidden=$W"; [ "$W" = 512 ] && EXTRA="$EXTRA model.n_slices=512" ;;
                   geotransolver_lug) EXTRA="$EXTRA model.n_hidden=$W" ;;
                   transolver_lug) EXTRA="$EXTRA model.n_hidden=$W"; [ "$W" = 512 ] && EXTRA="$EXTRA model.n_layers=16" ;; esac
  local OUT="${TASK}/lug_prec/${PREC}/${RUN}"; mkdir -p "$OUT"
  echo "=== START $(date -Is) run=$RUN extra=$EXTRA prec=$PREC" >> "$OUT.log"
  cd "$RECIPE_DIR" && timeout 5000 env PYTHONPATH="${CODE}:${RECIPE_DIR}/src" CUDA_VISIBLE_DEVICES="$gpu" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${VENV}/bin/python" "${RECIPE_DIR}/src/infer.py" model="$MODEL" $EXTRA dataset=ginot_lug_volume run_id="$RUN" infer_split=val +save_predictions=false \
    sampling_resolution=200000 precision="$PREC" checkpoint_dir="${TASK}/runs" output_dir="$OUT" >> "$OUT.log" 2>&1 && touch "$OUT/.done" || echo "=== $RUN FAILED rc=$?" >> "$OUT.log"
  rm -rf "$OUT"/*/predictions
}
first=$((SLURM_ARRAY_TASK_ID * 4))
for g in 0 1 2 3; do i=$((first + g)); [ $i -lt ${#LANES[@]} ] && ( unit "${LANES[$i]}" "$g" ) & done
wait; echo "LUG-PREC-TASK-DONE"
