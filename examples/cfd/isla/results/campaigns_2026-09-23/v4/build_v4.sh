#!/bin/bash
# Build the code_v4 snapshot on AGA (run ON aga2, in $T): code_v3 + the ISLA package overlay of local HEAD (model.py,
# geo_kernel.py with the K + P' <= 3 fused kernel, nn_kernel.py; __init__.py compared, not copied unless it differs).
# recipe_v4 = recipe_v3 unchanged (cp -a), so every recipe_v3 config runs; the pump gets a v4_* case. Never copies whole
# recipe files from the local tree. Inputs are staged under kernel/v4/overlay by the deploy script.
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
cd "$T"
[ -d code_v4 ] || cp -a code_v3 code_v4
[ -d recipe_v4 ] || cp -a recipe_v3 recipe_v4
D=code_v4/physicsnemo/experimental/models/isla
for f in model.py geo_kernel.py nn_kernel.py; do cp kernel/v4/overlay/$f $D/$f; done
cmp -s $D/__init__.py kernel/v4/overlay/__init__.py && echo "init unchanged" || { cp kernel/v4/overlay/__init__.py $D/__init__.py; echo "init updated"; }
echo "overlay source commit: $(cat kernel/v4/overlay/SOURCE_COMMIT)" > code_v4/OVERLAY_SOURCE
find code_v4 -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
$PY -m py_compile $D/model.py $D/geo_kernel.py $D/nn_kernel.py && echo compiled
f=kernel/p200k_fleet/p200k_eval_aga.sbatch
grep -q 'v4_\*' "$f" || sed -i 's#case "${RUN}" in sg_\*) CODE="${TASK}/code_v3"; RECIPE_DIR="${TASK}/recipe_v3" ;;#case "${RUN}" in v4_*) CODE="${TASK}/code_v4"; RECIPE_DIR="${TASK}/recipe_v4" ;; sg_*) CODE="${TASK}/code_v3"; RECIPE_DIR="${TASK}/recipe_v3" ;;#' "$f"
grep -n 'v4_' "$f" | cut -c1-160; bash -n "$f" && echo pump-syntax-ok
### GPU verification: one forward+backward of a two-direction model and a rank-1-input model under the fused kernel must
### match the eager region in float64 (the kernel tests, run on the venv's pytest against code_v4)
cat > kernel/v4/verify_v4.sbatch <<'SB'
#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.v4.verify
#SBATCH --time=00:20:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
export PYTHONPATH=$T/code_v4
cd $T/code_v4 && $PY -m pytest -q test/experimental/models/isla/test_isla_kernels.py test/experimental/models/isla/test_isla_interior_geometry.py -x 2>&1 | tail -3
echo "VERIFY-RC=${PIPESTATUS[0]}"
SB
mkdir -p sbatch_logs; sbatch --parsable kernel/v4/verify_v4.sbatch
