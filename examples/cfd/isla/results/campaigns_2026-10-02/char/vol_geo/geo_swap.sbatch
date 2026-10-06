#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.volgeo
#SBATCH --time=02:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/char/vol_geo/logs/volgeo_%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/char/vol_geo/logs/volgeo_%j.log
# VOL-GEO (2026-10-05): geo_swap.py on cl_r_vol_isla_seed42 (GPU 0) and _seed43 (GPU 1), one process each.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
OUT="${TASK}/kernel/char/vol_geo"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
cd "${TASK}/recipe_v8"
run() {
  local RUN="$1" GPU="$2"
  echo "=== START $(date -Is) run=${RUN} gpu=${GPU}"
  env PYTHONPATH="${TASK}/code_v20:${TASK}/recipe_v8/src" CUDA_VISIBLE_DEVICES="${GPU}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${VENV}/bin/python" "${OUT}/geo_swap.py" "${RUN}" "${OUT}/${RUN}" > "${OUT}/logs/${RUN}.log" 2>&1 \
    && echo "=== DONE $(date -Is) run=${RUN}" || echo "=== FAILED rc=$? $(date -Is) run=${RUN}"
}
run cl_r_vol_isla_seed42 0 &
run cl_r_vol_isla_seed43 1 &
wait
echo "VOLGEO-TASK-DONE ${SLURM_JOB_ID}"
