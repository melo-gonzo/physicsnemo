#!/bin/bash
# Build the code_v6 snapshot on AGA (run ON aga2, in $T): code_v5 + the ISLA package overlay of local HEAD 9b799ac4
# (geo_kernel.py with the MLP routing bias fused into the Triton kernel; model.py lifts the fused refusal of geo_bias=mlp)
# and the two updated test files. recipe_v3 unchanged. Pump case gbf_* -> code_v6/recipe_v3. Never whole-file recipe copies.
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
cd "$T"
[ -d code_v6 ] || cp -a code_v5 code_v6
D=code_v6/physicsnemo/experimental/models/isla
for f in model.py geo_kernel.py nn_kernel.py; do cp kernel/v6/overlay/$f $D/$f; done
cmp -s $D/__init__.py kernel/v6/overlay/__init__.py && echo "init unchanged" || { cp kernel/v6/overlay/__init__.py $D/__init__.py; echo "init updated"; }
for f in test_isla_kernels.py test_isla.py; do cp kernel/v6/tests/$f code_v6/test/experimental/models/isla/$f; done
echo "overlay source commit: $(cat kernel/v6/overlay/SOURCE_COMMIT)" > code_v6/OVERLAY_SOURCE
find code_v6 -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
$PY -m py_compile $D/model.py $D/geo_kernel.py $D/nn_kernel.py && echo compiled
f=kernel/p200k_fleet/p200k_eval_aga.sbatch
grep -q 'gbf_\*' "$f" || sed -i 's#case "${RUN}" in sg_\*_plane_\*|gb_\*) CODE="${TASK}/code_v5"; RECIPE_DIR="${TASK}/recipe_v3" ;;#case "${RUN}" in gbf_*) CODE="${TASK}/code_v6"; RECIPE_DIR="${TASK}/recipe_v3" ;; sg_*_plane_*|gb_*) CODE="${TASK}/code_v5"; RECIPE_DIR="${TASK}/recipe_v3" ;;#' "$f"
grep -q 'gbf_af_\*' "$f" || sed -i 's#|gb_af_\*) CELLS=100000 ;;#|gb_af_*|gbf_af_*) CELLS=100000 ;;#' "$f"
grep -c 'gbf_' "$f"; bash -n "$f" && echo pump-syntax-ok
mkdir -p sbatch_logs
cat > kernel/v6/verify_v6.sbatch <<'SB'
#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.v6.verify
#SBATCH --time=00:40:00
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
export PYTHONPATH=$T/code_v6
cd $T/code_v6 && $PY -m pytest -q test/experimental/models/isla/test_isla_kernels.py test/experimental/models/isla/test_isla_interior_geometry.py test/experimental/models/isla/test_isla.py -x 2>&1 | tail -3
echo "VERIFY-RC=${PIPESTATUS[0]}"
SB
sbatch --parsable kernel/v6/verify_v6.sbatch
