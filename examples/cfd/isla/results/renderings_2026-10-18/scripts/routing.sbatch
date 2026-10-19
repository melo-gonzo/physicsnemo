#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.fig.routing
#SBATCH --time=00:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-18/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-18/sbatch_logs/%x-%j.log
# FIG lane (2026-10-18): trained routing of the A2 reference ISLA checkpoints (slices_probe_a2.py), one validation sample
# each (the median-ISLA-error sample of the renderings), float32, one GPU per run. Inference only.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
F="${TASK}/book_figs_2026-10-18"; S="${F}/scripts"; O="${F}/routing"; mkdir -p "${O}"
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
cd "${TASK}"
probe() {
  local gpu="$1" tag="$2" code="$3" recipe="$4" ck="$5"; shift 5
  echo "=== START $(date -Is) gpu=${gpu} ${tag} code=${code} recipe=${recipe} args=$*"
  env PYTHONPATH="${TASK}/${code}:${TASK}/${recipe}/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    timeout 3000 "${VENV}/bin/python" "${S}/slices_probe_a2.py" --recipe-dir "${TASK}/${recipe}" --checkpoint-dir "${ck}" --out "${O}" "$@" \
    > "${O}/${tag}.log" 2>&1
  echo "=== END rc=$? $(date -Is) ${tag}"; grep -v "Warning\|warn" "${O}/${tag}.log" | tail -n 16
}
I5="model.hidden=512 model.n_slices=512 model.compile_blocks=true"
probe 0 dr code_a2r recipe_a2r "${F}/runs" --run a2r_dr_isla_seed42 --model isla_surface --dataset drivaer_ml_surface --cells 200000 --sample 79 --extra "${I5}" --keep 200000 &
probe 1 af code_a2s recipe_v11 "${TASK}/runs" --run afu_a2s_isla_seed42 --model isla_volume_airfrans --dataset airfrans_uvolume_2d --cells 100000 --sample 215 --extra "train_split=full_train val_split=full_test ${I5}" &
probe 2 lug code_a2r recipe_a2r "${TASK}/runs" --run a2r_lug_isla_seed42 --model isla_lug --dataset ginot_lug_volume --cells 200000 --sample 633 --extra "train_split=train val_split=test ${I5}" &
probe 3 t2d code_a2r recipe_a2r "${TASK}/runs" --run a2r_t2dz_isla_seed42 --model isla_tensile2d --dataset plaid_tensile2d_2d --cells 200000 --sample 238 --extra "train_split=train_400 val_split=val_100 ${I5}" &
wait
echo "FIG-ROUTING-DONE $(date -Is)"
