#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.global.prof
#SBATCH --time=00:40:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH -o sbatch_logs/%x-%j.log
# QI-GLOBAL: kernel timings of the fused global read at the protocol size (synthetic 2-D case), block-size variants,
# and a model-level profile (synthetic, protocol width), one GPU each.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
CODE=${CODE:-code_g2}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PYTHONPATH="$TASK/$CODE:$TASK/code_test" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $TASK/stats
echo "[$(date -Is)] prof on $(hostname) code $CODE $(cat $TASK/$CODE/SNAPSHOT_HEAD)"
( CUDA_VISIBLE_DEVICES=0 $VENV/bin/python prof_global.py ) > prof_${CODE}_g0.log 2>&1 &
( CUDA_VISIBLE_DEVICES=1 $VENV/bin/python model_bench.py 100000 global all prof ) > prof_${CODE}_g1.log 2>&1 &
( CUDA_VISIBLE_DEVICES=2 $VENV/bin/python model_bench.py 100000 ro all prof ) > prof_${CODE}_g2.log 2>&1 &
wait
echo "[$(date -Is)] done"
