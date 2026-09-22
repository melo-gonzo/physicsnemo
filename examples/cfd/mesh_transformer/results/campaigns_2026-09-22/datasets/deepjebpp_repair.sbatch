#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J jeb.repair
#SBATCH --time=03:00:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# DeepJEB++ repair (2026-09-22): scan all 61,440 expected files for completeness (four meta.json files) and re-convert the
# incomplete ones. Needed because cpu-free kills of jeb.convert (829391) mid-save left at least one partial .pdmsh
# (415-242-IS16_dia: no _tensordict/global_data/meta.json) that the plain existence check took for finished and that killed
# both smoke lanes (830052) in validation. The converter now publishes atomically (save to .tmp, rename). Non-preemptible QoS.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
U=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe
VENV=$U/physicsnemo-mesh-transformer/.venv-recipe
export PYTHONPATH="${TASK}/code_ref" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
OUT=$U/datasets/deepjebpp_pdmsh
echo "=== start $(date -Is) job=$SLURM_JOB_ID node=$(hostname)"
cd "${TASK}"
chmod u+w "$OUT"
"${VENV}/bin/python" kernel/deepjebpp_convert.py "$OUT/deepjebpp_raw" "$OUT" --repair
echo "=== end rc=$? $(date -Is)"
