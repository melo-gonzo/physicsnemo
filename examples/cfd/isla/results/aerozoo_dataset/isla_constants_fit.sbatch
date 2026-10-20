#!/bin/bash
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -t 01:30:00
#SBATCH -c 128
#SBATCH --mem=850G
#SBATCH -J isla.scales.fit
#SBATCH -A coreai_modulus_cae
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/isla_scales/fit/fit-%j.log
# ISLA-SCALES round 2: the per-case scale scan for the refit (isla_constants_fit.py), with the model configs in the
# job directory. usage: sbatch isla_constants_fit.sbatch <code dir> <tag> aerozoo <shard> <n_shards> | <code dir> <tag> airfrans
set -euo pipefail
K=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/isla_scales
TD=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-15-tensordict-015/td0152
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
DS=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets
CODE="$1"; TAG="$2"; shift 2
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH="$K/$CODE:$TD" CUDA_VISIBLE_DEVICES=""
cd "$K/fit"
echo "=== fit $* start $(date -Is) job=$SLURM_JOB_ID code=$CODE tag=$TAG"
if [ "$1" = aerozoo ]; then
  "$VENV/bin/python" isla_constants_fit.py aerozoo "$DS/aerozoo_v1.2_shuf_pdmsh" isla_volume_aerozoo.yaml "${TAG}_az_$2.jsonl" "$2" "$3" 100
else
  "$VENV/bin/python" isla_constants_fit.py airfrans "$DS/airfrans_pdmsh" isla_volume_airfrans.yaml "${TAG}_af.jsonl" 64
fi
echo "=== fit end $(date -Is)"
