#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.unified.diag
#SBATCH --time=00:40:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# UNIFIED-TOKENS test (a): the two volume-trained provided-geometry ISLA seeds evaluated at the surface cell centroids
# of the 48 validation cars (200,000-cell draw), one GPU per seed, in parallel.
set -uo pipefail
export T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
export RECIPE=$T/recipe_v3
export PYTHONPATH=$T/code_v3:$T/recipe_v3/src
export UV_CACHE_DIR=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/.uvcache
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
cd $T/recipe_v3/src
mkdir -p $T/unified_eval
trap 'echo "rc=$?" > $T/unified_eval/STATUS_diag_$SLURM_JOB_ID' EXIT
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29750 CUDA_VISIBLE_DEVICES=0 $PY $T/kernel/unified/unified_diag_2026-09-27.py > $T/unified_eval/diag_seed42.log 2>&1
tail -n 6 $T/unified_eval/diag_seed42.log
echo UNIFIED-DIAG-DONE
