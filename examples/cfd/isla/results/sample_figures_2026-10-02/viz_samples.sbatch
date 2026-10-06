#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.viz.samples
#SBATCH --time=03:00:00
#SBATCH -p cpu
#SBATCH -q cpu-normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --array=0-6
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
# Sample-figure artifacts (2026-10-02): several samples per dataset reduced to what the book plots. One array task per
# dataset, CPU partition only (the extraction reads memmapped DomainMesh files and never needs a GPU).
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONPATH="${TASK}/code_v18"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
cd "${TASK}/viz_samples"
DATASETS=(drivaer hilift airfrans shift_suv tensile2d lug deepjeb)
ds=${DATASETS[$SLURM_ARRAY_TASK_ID]}
"${VENV}/bin/python" sample_figures.py --out "${TASK}/viz_samples/out" --only "$ds"; echo "rc_$ds=$?"
echo "SAMPLES-DONE $ds"
