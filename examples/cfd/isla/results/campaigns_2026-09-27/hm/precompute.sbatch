#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J hm.precompute
#SBATCH -p cpu
#SBATCH -q cpu-normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=23:00:00
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# HARMONIC precompute (2026-09-27): usage sbatch kernel/hm/precompute.sbatch <olbc|airfrans> [extra args e.g. --limit 3]
set -uo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
D=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets
export UV_CACHE_DIR=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/.uvcache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # Grace: BLAS thread storms hang scipy imports
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
trap 'echo "rc=$?" > $T/STATUS_hm_$SLURM_JOB_ID' EXIT
WHICH=$1; shift
echo "=== start $(date -Is) which=$WHICH extra=$*"
if [ "$WHICH" = olbc ]; then
  $PY $T/recipe_v4/src/precompute_harmonic.py --src $D/olbc_poisson_pdmsh --dst $D/olbc_poisson_hm_pdmsh --boundary outer \
    --label-fields bc_dirichlet bc_neumann bc_robin --node-tag-fields tag_dirichlet tag_neumann tag_robin \
    --value-field bc_value_g --node-value-field bc_value_node --workers 60 "$@"
elif [ "$WHICH" = airfrans ]; then
  $PY $T/recipe_v4/src/precompute_harmonic.py --src $D/airfrans_pdmsh --dst $D/airfrans_hm_pdmsh --boundary airfoil \
    --wall-from-sdf --pattern 'airFoil2D_SST_*' --workers 48 "$@"
elif [ "$WHICH" = smoke ]; then
  $PY $T/recipe_v4/src/precompute_harmonic.py --src $D/olbc_poisson_pdmsh --dst $D/_hm_smoke/olbc --boundary outer \
    --label-fields bc_dirichlet bc_neumann bc_robin --node-tag-fields tag_dirichlet tag_neumann tag_robin \
    --value-field bc_value_g --node-value-field bc_value_node --workers 2 --limit 3 --overwrite
  $PY $T/recipe_v4/src/precompute_harmonic.py --src $D/airfrans_pdmsh --dst $D/_hm_smoke/af --boundary airfoil \
    --wall-from-sdf --pattern 'airFoil2D_SST_*' --workers 1 --limit 1 --overwrite
fi
echo "=== end $(date -Is)"
