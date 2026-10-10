#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J verify-v26
#SBATCH -p batch
#SBATCH -q short
#SBATCH --time=01:30:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
#SBATCH --open-mode=append
# Snapshot verification for code_v26 (psharpe/isla 73fd89e5f: query geometry nearest only, AeroZoo recipe support) and
# recipe_v9 (the recipe archive of the same commit + recipe_v8's cluster-only datasets and model configs + dataset_paths
# with the aerozoo roots), under the td14 venv. Keeps the tracebacks (-rA) so a failure is legible in the log.
set -u
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe-td14
export PYTHONPATH="${TASK}/code_v26:${TASK}/recipe_v9/src"
export PYTHONUNBUFFERED=1
trap 'echo "rc=$?" > ${TASK}/STATUS_verify_v26_${SLURM_JOB_ID}' EXIT
echo "[$(date -Is)] verify code_v26 + recipe_v9 on $(hostname); head $(cat ${TASK}/code_v26/SNAPSHOT_HEAD)"
cd "${TASK}/code_v26"
env -u SLURM_PROCID -u SLURM_NPROCS -u SLURM_LOCALID "${VENV}/bin/python" -m pytest test/experimental/models/isla -q -rA -p no:cacheprovider 2>&1 | tail -n 40
echo "=== ISLA suite rc=${PIPESTATUS[0]}"
cd "${TASK}/recipe_v9"
env -u SLURM_PROCID -u SLURM_NPROCS -u SLURM_LOCALID "${VENV}/bin/python" -m pytest tests -q -rA -p no:cacheprovider 2>&1 | tail -n 60
echo "=== recipe suite rc=${PIPESTATUS[0]}"
echo "[$(date -Is)] done"
