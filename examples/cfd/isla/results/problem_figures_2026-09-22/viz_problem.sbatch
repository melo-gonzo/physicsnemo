#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.viz.problem
#SBATCH --time=01:30:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# Problem-figure artifacts (2026-09-22): one representative sample per dataset reduced to what the book plots.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONPATH="${TASK}/code_ref:${TASK}/recipe_globin/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd "${TASK}/kernel/viz"
for ds in shift_suv airfrans tensile2d lug olbc_poisson olbc_elasticity hilift; do
  timeout 20m "${VENV}/bin/python" problem_figures.py --out "${TASK}/probes/problem_figures_2026-09-22" --only $ds; echo "rc_$ds=$?"
done
echo "PROBLEM-DONE rc=$?"
