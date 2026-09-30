#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J jeb.rescore
#SBATCH --time=03:00:00
#SBATCH -p cpu
#SBATCH -q cpu-free
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-31
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
# DeepJEB++ rescore (2026-09-22): load back every converted file of the shard and rebuild its stats record. Needed because the
# cpu-free requeues of jeb.convert (829391) re-ran shards that then reported "exists" (no sums) for the files a killed run had
# already written, leaving _stats over 17,320 of the 55,296 train files. Read-only on the dataset; writes one json per shard.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
U=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe
VENV=$U/physicsnemo-mesh-transformer/.venv-recipe
export PYTHONPATH="${TASK}/code_ref" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
OUT=$U/datasets/deepjebpp_pdmsh
echo "=== start $(date -Is) job=$SLURM_JOB_ID node=$(hostname) shard=$SLURM_ARRAY_TASK_ID/32"
cd "${TASK}"
"${VENV}/bin/python" kernel/deepjebpp_convert.py "$OUT/deepjebpp_raw" "$OUT" --rescore --shard "$SLURM_ARRAY_TASK_ID" --nshards 32
echo "=== end rc=$? $(date -Is)"
