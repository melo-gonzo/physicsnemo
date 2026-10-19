#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.fig.cost
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-18/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-18/sbatch_logs/%x-%j.log
# FIG lane (ISLA book overhaul, 2026-10-18): training-step time and peak memory vs token count, reference configurations,
# each on its own GPU of one GB300 node (one process per GPU; the four arms run concurrently on separate GPUs), each under
# the code + recipe snapshot its reference lanes trained with (bench_step.py). Microbenchmark only, no training run.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
F="${TASK}/book_figs_2026-10-18"; S="${F}/scripts"; O="${F}/cost"; mkdir -p "${O}"
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
cd "${TASK}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
arm() {
  local gpu="$1" label="$2" code="$3" recipe="$4" model="$5" ds="$6" extra="$7" tokens="$8"
  echo "=== ARM START $(date -Is) gpu=${gpu} ${label} code=${code} recipe=${recipe} model=${model} dataset=${ds} extra='${extra}' tokens=${tokens}"
  env TRITON_CACHE_DIR=/tmp/$USER/triton_${gpu} TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor_${gpu} \
    PYTHONPATH="${TASK}/${code}:${TASK}/${recipe}/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    timeout 6600 "${VENV}/bin/python" "${S}/bench_step.py" --label "${label}" --recipe-dir "${TASK}/${recipe}" --model "${model}" \
    --dataset "${ds}" --extra "${extra}" --tokens "${tokens}" --out "${O}/${label}.json" > "${O}/${label}.log" 2>&1
  echo "=== ARM END rc=$? $(date -Is) ${label}"; grep "N=" "${O}/${label}.log"
}
ST=25000,50000,100000,200000,400000,800000
arm 0 isla_surface_512x12 code_a2r recipe_a2r isla_surface drivaer_ml_surface "model.hidden=512 model.n_slices=512 model.compile_blocks=true" "${ST}" &
arm 1 geotransolver_surface_256 code_v20 recipe_v8 geotransolver_surface drivaer_ml_surface "" "${ST}" &
arm 2 transolver_surface_512x16 code_v20 recipe_v8 transolver_surface drivaer_ml_surface "model.n_hidden=512 model.n_layers=16" "${ST}" &
( arm 3 isla_volume_512x12 code_a2r recipe_a2r isla_volume drivaer_ml_volume_reference "model.hidden=512 model.n_slices=512 model.compile_blocks=true" "100000,200000,400000"
  arm 3 isla_dtt_volume_512x12 code_a2r recipe_a2r isla_volume drivaer_ml_volume_reference "model.hidden=512 model.n_slices=512 compile=false model.compile_blocks=true ++model.query_independent=true ++model.kernel_read=mean ++model.kernel_read_radius=null ++model.kernel_read_theta=1.0" "100000,200000,400000" ) &
wait
echo "FIG-COST-DONE $(date -Is)"
