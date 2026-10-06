#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.global.prec
#SBATCH --time=01:00:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
# QI-GLOBAL bf16 re-scoring (2026-10-05): infer.py exactly as eval.sbatch but at precision=${PREC:-bfloat16}, for the units in
# ${UNITS_FILE:-eval/prec_units.tsv} (run|code|ckpt dir|model|dataset|cells|extra), four at a time. No training.
set -uo pipefail
ulimit -s 8192; ulimit -c 0
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RECIPE=${TASK}/recipe_g
PREC=${PREC:-bfloat16}
OUT=${TASK}/eval_prec/${PREC}/score; mkdir -p ${OUT}
export PATH=/usr/bin:/bin:/cm/local/apps/slurm/current/bin OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
mapfile -t UNITS < "${TASK}/${UNITS_FILE:-eval/prec_units.tsv}"
echo "[$(date -Is)] prec=${PREC} job ${SLURM_JOB_ID} on $(hostname): ${#UNITS[@]} units"
cd ${RECIPE}
g=0
for u in "${UNITS[@]}"; do
  IFS='|' read -r RUN CODE CK MODEL DATASET CELLS EXTRA <<<"${u}"
  ( env PYTHONPATH=${CODE}:${RECIPE}/src CUDA_VISIBLE_DEVICES=${g} RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((24000 + SLURM_JOB_ID % 20000 + g)) \
      timeout 1500 ${VENV}/bin/python ${RECIPE}/src/infer.py model=${MODEL} ${EXTRA} dataset=${DATASET} run_id=${RUN} \
      infer_split=val +save_predictions=false sampling_resolution=${CELLS} precision=${PREC} checkpoint_dir=${CK} output_dir=${OUT} > ${OUT}/${RUN}.log 2>&1
    echo "[$(date -Is)] ${RUN} rc=$?" ) &
  g=$((g + 1)); if [ ${g} -eq 4 ]; then wait; g=0; fi
done
wait
echo "[$(date -Is)] done"
