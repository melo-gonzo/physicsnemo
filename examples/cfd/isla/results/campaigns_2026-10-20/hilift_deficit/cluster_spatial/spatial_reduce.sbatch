#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.h090.spred
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -t 00:20:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/h090/spatial/spred-%j.log
# HILIFT-090 spatial reduction (CPU): spatial_reduce.py over savepred.sbatch's outputs. Read-only on the dataset.
G=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae
SP=$G/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/h090/spatial
echo "=== start $(date -Is) job=$SLURM_JOB_ID node=$(hostname) n=${N:-0}"
OMP_NUM_THREADS=1 $G/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python $SP/spatial_reduce.py $SP $G/datasets/PhysicsNeMo-HighLiftAeroML-medium $SP/face_counts.txt 6
echo "=== end $(date -Is) rc=$?"
