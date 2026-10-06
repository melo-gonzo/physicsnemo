#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.jdprobe
#SBATCH --time=01:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/char/jd_probe/sbatch_%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/char/jd_probe/sbatch_%j.log
# JEB-PLATEAU routing probe (2026-10-05): slice-routing statistics of cs_jd_r1 (plateaued) and cs_jd_rate (healthy)
# at epochs 4 and 5 (the checkpoints the smoke lanes keep), code_v20 checkpoints, float32, first 4 test_clean cases at 200k cells. One run per GPU.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
export PYTHONPATH="${TASK}/code_v20:${TASK}/recipe_v8/src"
OUT="${TASK}/kernel/char/jd_probe"
cd "${TASK}/recipe_v8"
unit() {
  local run="$1" gpu="$2"
  echo "=== ${run} START $(date -Is)" >> "${OUT}/${run}.log"
  env CUDA_VISIBLE_DEVICES="$gpu" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${VENV}/bin/python" "${OUT}/routing_probe_v20.py" --run "$run" --epochs 4 5 --k 4 --cells 200000 --out-dir "${OUT}" \
    >> "${OUT}/${run}.log" 2>&1 && echo "=== ${run} DONE $(date -Is)" >> "${OUT}/${run}.log" || echo "=== ${run} FAILED rc=$?" >> "${OUT}/${run}.log"
}
unit cs_jd_r1 0 &
unit cs_jd_rate 1 &
wait
echo "JD-PROBE-DONE $(date -Is)"
