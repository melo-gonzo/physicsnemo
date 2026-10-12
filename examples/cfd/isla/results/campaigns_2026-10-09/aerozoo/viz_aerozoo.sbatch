#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.viz.aerozoo
#SBATCH --time=00:59:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# AeroZoo v1.2 figure data (2026-10-10): the problem-figure case (problem_figures.py) and the six sample cases
# (sample_figures.py), read from the release root and reduced to what the book plots. CPU partition only: the
# extraction reads memmapped DomainMesh files and never needs a GPU. Submit from ${TASK} (the log path is relative):
#   cd ${TASK} && sbatch kernel/viz/viz_aerozoo.sbatch
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe-td14
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONPATH="${TASK}/code_v26:${TASK}/recipe_v9/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd "${TASK}/kernel/viz"
"${VENV}/bin/python" problem_figures.py --out "${TASK}/probes/problem_figures_2026-09-22" --only aerozoo; echo "rc_problem=$?"
"${VENV}/bin/python" sample_figures.py --out "${TASK}/viz_samples/out" --only aerozoo; echo "rc_samples=$?"
echo "AEROZOO-VIZ-DONE"
