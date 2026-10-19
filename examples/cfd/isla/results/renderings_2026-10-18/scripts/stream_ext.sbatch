#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.fig.stream
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-18/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-18/sbatch_logs/%x-%j.log
# FIG lane (ISLA book overhaul, 2026-10-18): extend the H-RESID Instrument-1 stream series (LHR jobs 224693, 240803, 240804)
# to the full 250-epoch schedule, and add the raw-read seed 43. Same probe (an unmodified copy of
# kernel/aerozoo/hresid/hresid_probe.py, md5 514f3c6b...), same 4 full.val cases and draws, same venv and recipe_v9,
# each arm on its own code snapshot (raw code_v26, A code_v29, A2 code_v30), as in the original jobs. Inference only.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe-td14
RECIPE="${TASK}/recipe_v9"
F="${TASK}/book_figs_2026-10-18"; S="${F}/stream"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
export TRITON_CACHE_DIR=/tmp/$USER/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${S}/out"
cd "${RECIPE}"
LATE="99,149,199,249"
run_arm() {
  local CODE="$1" PLAN="$2"
  echo "=== ARM START $(date -Is) job=${SLURM_JOB_ID} node=$(hostname) code=${CODE} plan=${PLAN}"
  env PYTHONPATH="${TASK}/${CODE}:${RECIPE}/src" CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${VENV}/bin/python" "${S}/hresid_probe.py" --plan "${PLAN}" --k 4 --cells 200000 --dataset aerozoo_uvolume \
    --precision bfloat16 --out-dir "${S}/out"
  echo "=== ARM END rc=$? $(date -Is) code=${CODE}"
}
run_arm code_v30 "azu_full_islabn2_seed42=${LATE};azu_full_islabn2_seed43=${LATE}"
run_arm code_v26 "azu_full_isla_seed42=${LATE};azu_full_isla_seed43=1,9,19,29,39,43,49,51,55,59,69,85,${LATE}"
run_arm code_v29 "azu_full_islabn_seed42=${LATE};azu_full_islabn_seed43=${LATE}"
echo "=== END $(date -Is) json=$(find "${S}/out" -name "e*.json" | wc -l)"
