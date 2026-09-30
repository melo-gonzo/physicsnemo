#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.viz.slices
#SBATCH --time=01:30:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# SLICES probe (2026-09-22): the trained routing of four completed reference-configuration ISLA runs on one validation case each.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONPATH="${TASK}/code_ref:${TASK}/recipe_globin/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((25000 + SLURM_JOB_ID % 20000))
OUT="${TASK}/probes/slices_2026-09-22"
P="${TASK}/kernel/viz/slices_probe.py"
cd "${TASK}/recipe_globin"
timeout 25m "${VENV}/bin/python" "$P" --run ref_isla_r1_dr435_seed42 --model isla_surface_ref --dataset drivaer_ml_surface \
  --train-split train --val-split val --cells 200000 --out "$OUT"; echo "rc_dr=$?"
timeout 20m "${VENV}/bin/python" "$P" --run so_t2d_isla_r1_frame_n64_seed42 --model isla_tensile2d_frame2 --dataset plaid_tensile2d_frame \
  --train-split train_64 --val-split val_100 --cells 20000 --out "$OUT"; echo "rc_t2d=$?"
timeout 20m "${VENV}/bin/python" "$P" --run ref_isla_r1_af_seed42 --model isla_volume_airfrans_ref --dataset airfrans_volume \
  --train-split full_train --val-split full_test --cells 100000 --out "$OUT"; echo "rc_af=$?"
timeout 20m "${VENV}/bin/python" "$P" --run so_lug_isla_r1_n300_seed43 --model isla_lug --dataset ginot_lug_volume \
  --train-split train_n300 --val-split test --cells 20000 --out "$OUT"; echo "rc_lug=$?"
echo "SLICES-DONE"
