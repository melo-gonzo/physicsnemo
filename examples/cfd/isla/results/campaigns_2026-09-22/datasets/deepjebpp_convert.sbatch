#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J jeb.convert
#SBATCH --time=06:00:00
#SBATCH -p cpu
#SBATCH -q cpu-free
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-31
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
# DeepJEB++ conversion (2026-09-22): 32 shards over the 15,360 brackets (4 files each). Idempotent (existing files are skipped),
# so a cpu-free preemption/requeue just resumes. Finalize + verify + write-lock is a separate job (deepjebpp_finalize.sbatch).
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
U=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe
VENV=$U/physicsnemo-mesh-transformer/.venv-recipe
export PYTHONPATH="${TASK}/code_ref" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
NSH=32
OUT=$U/datasets/deepjebpp_pdmsh
echo "=== start $(date -Is) job=$SLURM_JOB_ID node=$(hostname) shard=$SLURM_ARRAY_TASK_ID/$NSH out=$OUT"
cd "${TASK}"
"${VENV}/bin/python" kernel/deepjebpp_convert.py "$OUT/deepjebpp_raw" "$OUT" --shard "$SLURM_ARRAY_TASK_ID" --nshards $NSH
echo "=== end rc=$? $(date -Is)"
