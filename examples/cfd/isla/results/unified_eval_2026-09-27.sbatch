#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.unified.eval
#SBATCH --time=02:00:00
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
trap 'echo "rc=$?" > $T/unified_eval/STATUS_$SLURM_JOB_ID' EXIT
for s in 42 43; do
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29700 + s)) CUDA_VISIBLE_DEVICES=$((s - 42)) $PY $T/kernel/unified/unified_eval_2026-09-27.py --run sg_drvol_provided_seed$s --seed $s --out $T/unified_eval --flip-normals --tag _flip \
    > $T/unified_eval/sg_drvol_provided_seed${s}_flip.log 2>&1 &
  RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29710 + s)) CUDA_VISIBLE_DEVICES=$((s - 40)) $PY $T/kernel/unified/unified_eval_2026-09-27.py --run sg_drvol_provided_seed$s --seed $s --out $T/unified_eval --flip-normals --sdf 2.5e-7 --tag _flip_sdfeps \
    > $T/unified_eval/sg_drvol_provided_seed${s}_flip_sdfeps.log 2>&1 &
done
wait
grep -h SUMMARY $T/unified_eval/sg_drvol_provided_seed4?_flip*.log
echo UNIFIED-EVAL-DONE
