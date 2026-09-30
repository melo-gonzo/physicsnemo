#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J costsweep-ref
#SBATCH --time=03:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/costsweep_ref_%j.log
#SBATCH -e sbatch_logs/costsweep_ref_%j.log
#SBATCH --open-mode=append
# COST-SWEEP-REF (2026-09-22): step time and peak memory vs token count, 2.5k to 600k tokens, six arms (see
# kernel/cost_ref/bench_tokens_ref.py), one GB300 (cuda:0), bf16, batch 1, under the code_ref snapshot.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
KS=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-09-13-isla-kernel-study
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8
export TORCHINDUCTOR_CACHE_DIR="${TASK}/inductor_cache_ref" TRITON_CACHE_DIR="${TASK}/triton_cache_ref"
mkdir -p "${TASK}/results" "${TASK}/sbatch_logs"
cd "${TASK}"
echo "=== $(date -u +%FT%TZ) on $(hostname) job ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
PYTHONPATH="${TASK}/code_ref:${KS}/study" timeout 14000 "${VENV}/bin/python" -u kernel/cost_ref/bench_tokens_ref.py results/cost_sweep_ref_gb300_bf16.json
echo "=== rc=$? $(date -u +%FT%TZ)"
touch results/COST_SWEEP_REF_DONE
