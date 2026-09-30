#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.sg.verify
#SBATCH --time=00:30:00
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# INTERNAL-GEOMETRY (2026-09-23): import and one forward pass of code_v3 + recipe_v3 before the sg lanes launch
# (login-node Python stalls, so this runs as a cpu-partition batch job).
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PYTHONPATH="${TASK}/code_v3:${TASK}/recipe_v3/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
cd "${TASK}"
"${VENV}/bin/python" kernel/sg/verify_v3.py "${TASK}/recipe_v3"
echo "VERIFY-RC=$?"
