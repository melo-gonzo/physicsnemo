#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.render090.cost
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=nvl72d015-T12,nvl72d058-T08,nvl72d052-T01
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-22/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-22/sbatch_logs/%x-%j.log
# COST-090 (RENDER-090 lane, 2026-10-22): renderings_2026-10-18/scripts/cost.sbatch for the current standing's models, same
# method and token counts: training-step time and peak memory against token count, each arm on its own GPU of one GB300 node
# (the four GPU streams concurrent), each under the code + recipe snapshot its standing lanes trained with (bench_step_090.py).
# ISLA 0.9.0 and ISLA-DTT 0.9.0: code_cmpaf_fix (code_r090 with per-block compile at Inductor's mix-order reduction off, as
# psharpe/isla now compiles every block) on recipe_r090, the r090 lanes' overrides (width 512, 512 slices, compile_blocks,
# whole-model compile off, as their resolved_config.yaml records), bf16; ISLA-DTT is query_independent=true, its tree read at
# TREE_READ_THETA = 1. The tail arm isla_surface_090_codeR090 is the same ISLA on code_r090, the snapshot the r090 DrivAerML
# lanes trained on (mix-order reduction at Inductor's default). GeoTransolver 256 and Transolver 512 x 16: the fair-drive
# standing arms' overrides (U_inf_dir; DRIVE-FIX2 score_rows.tsv), code_v20 on recipe_v8. Microbenchmark only, no training run.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
F="${TASK}/book_figs_2026-10-22"; S="${F}/scripts"; O="${F}/cost"; mkdir -p "${O}"
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
    timeout 6600 "${VENV}/bin/python" "${S}/bench_step_090.py" --label "${label}" --recipe-dir "${TASK}/${recipe}" --model "${model}" \
    --dataset "${ds}" --extra "${extra}" --tokens "${tokens}" --out "${O}/${label}.json" > "${O}/${label}.log" 2>&1
  echo "=== ARM END rc=$? $(date -Is) ${label}"; grep "N=" "${O}/${label}.log"
}
ST=25000,50000,100000,200000,400000,800000
I5="model.hidden=512 model.n_slices=512 model.compile_blocks=true compile=false"
arm 0 isla_surface_090 code_cmpaf_fix recipe_r090 isla_surface drivaer_ml_surface "${I5}" "${ST}" &
( arm 1 geotransolver_surface_256 code_v20 recipe_v8 geotransolver_surface drivaer_ml_surface "forward_kwargs.global_embedding=global_data.U_inf_dir" "${ST}"
  arm 1 isla_surface_090_codeR090 code_r090 recipe_r090 isla_surface drivaer_ml_surface "${I5}" "${ST}" ) &
arm 2 transolver_surface_512x16 code_v20 recipe_v8 transolver_surface drivaer_ml_surface "model.n_hidden=512 model.n_layers=16 forward_kwargs.fx.source=global_data.U_inf_dir" "${ST}" &
( arm 3 isla_volume_090 code_cmpaf_fix recipe_r090 isla_volume drivaer_ml_volume_reference "${I5}" "100000,200000,400000"
  arm 3 isla_dtt_volume_090 code_cmpaf_fix recipe_r090 isla_volume drivaer_ml_volume_reference "${I5} ++model.query_independent=true" "100000,200000,400000" ) &
wait
echo "RENDER090-COST-DONE $(date -Is)"
