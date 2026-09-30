#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J olbc.convert
#SBATCH --time=03:00:00
#SBATCH -p cpu
#SBATCH -q cpu-free
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-15
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
# OLBC conversion (2026-09-22): 8 shards per set; array 0-7 = poisson-square-bc5, 8-15 = elasticity-circlehollow-m1.
# Idempotent (existing files are skipped), so a cpu-free preemption/requeue just resumes. Finalize + verify is a separate job.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
U=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe
VENV=$U/physicsnemo-mesh-transformer/.venv-recipe
export PYTHONPATH="${TASK}/code_ref" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
NSH=8
if [ "${SLURM_ARRAY_TASK_ID}" -lt $NSH ]; then SET=poisson; RAWD=poisson-square-bc5; SHARD=${SLURM_ARRAY_TASK_ID};
else SET=elasticity; RAWD=elasticity-circlehollow-m1; SHARD=$(( SLURM_ARRAY_TASK_ID - NSH )); fi
OUT=$U/datasets/olbc_${SET}_pdmsh
mkdir -p "$OUT"
echo "=== start $(date -Is) job=$SLURM_JOB_ID node=$(hostname) set=$SET shard=$SHARD/$NSH out=$OUT"
cd "${TASK}"
"${VENV}/bin/python" kernel/olbc_convert.py "$U/datasets/olbc_raw/$RAWD" "$OUT" --set $SET --shard $SHARD --nshards $NSH
echo "=== end rc=$? $(date -Is)"
