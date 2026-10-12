#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.az.ckpt-drift
#SBATCH --time=03:00:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# AEROZOO checkpoint drift probe (2026-10-11): which weights move at the epoch 43-50 training transition of the ISLA lanes.
# CPU partition: torch.load of the kept odd-epoch checkpoints only, no GPU. Outputs kernel/aerozoo/probe/drift_<run>.{json,txt}.
set -uo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe-td14
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 PYTHONPATH="${T}/code_v26"
cd "${T}/kernel/aerozoo/probe"
odd() { seq "$1" 2 "$2" | tr '\n' ' '; }
run() {  # run epochs...
  local r="$1"; shift
  echo "=== ${r} $(date -Is)"
  "${VENV}/bin/python" ckpt_drift.py --run "${r}" --epochs "$@" --out "drift_${r}.json" > "drift_${r}.txt" 2>&1 || echo "=== ${r} FAILED rc=$?"
  tail -n 3 "drift_${r}.txt"
}
run az_full_isla_seed43  $(odd 29 69)
run az_full_gt_seed43    $(odd 29 69)
run az_nonbox_isla_seed42 $(odd 29 61)
run az_full_dtt_seed43   $(odd 45 83)
run az_full_isla_seed42  $(odd 29 61)
echo "AZ-CKPT-DRIFT-DONE ${SLURM_JOB_ID}"
