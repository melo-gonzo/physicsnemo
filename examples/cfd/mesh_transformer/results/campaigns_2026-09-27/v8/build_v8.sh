#!/bin/bash
# Build the code_v8 snapshot on AGA (run ON aga2, in $T): code_v7 + the ISLA package overlay of local HEAD aab5eef7
# (model.py with seed_directions / geometry_directions, FRAME-CHANNEL) and the updated tests. recipe_v4 unchanged. Pump case
# fc_* -> code_v8/recipe_v4; CELLS 100000 for fc_af_*. Never whole-file recipe copies.
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
cd "$T"
[ -d code_v8 ] || cp -a code_v7 code_v8
D=code_v8/physicsnemo/experimental/models/isla
for f in model.py geo_kernel.py nn_kernel.py; do cp kernel/v8/overlay/$f $D/$f; done
cmp -s $D/__init__.py kernel/v8/overlay/__init__.py && echo "init unchanged" || { cp kernel/v8/overlay/__init__.py $D/__init__.py; echo "init updated"; }
for f in test_isla_kernels.py test_isla.py; do cp kernel/v8/tests/$f code_v8/test/experimental/models/isla/$f; done
echo "overlay source commit: $(cat kernel/v8/overlay/SOURCE_COMMIT)" > code_v8/OVERLAY_SOURCE
find code_v8 -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
$PY -m py_compile $D/model.py $D/geo_kernel.py $D/nn_kernel.py && echo compiled
f=kernel/p200k_fleet/p200k_eval_aga.sbatch
grep -q 'fc_\*' "$f" || sed -i 's#case "${RUN}" in la_\*) #case "${RUN}" in fc_*) CODE="${TASK}/code_v8"; RECIPE_DIR="${TASK}/recipe_v4" ;; la_*) #' "$f"
grep -q 'fc_af_\*' "$f" || sed -i 's#|la_af_\*) CELLS=100000#|la_af_*|fc_af_*) CELLS=100000#' "$f"
grep -c 'fc_' "$f"; bash -n "$f" && echo pump-syntax-ok
mkdir -p sbatch_logs
cat > kernel/v8/verify_v8.sbatch <<'SB'
#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.v8.verify
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
export PYTHONPATH=$T/code_v8
cd $T/code_v8 && $PY -m pytest -q test/experimental/models/isla/test_isla_kernels.py test/experimental/models/isla/test_isla_interior_geometry.py test/experimental/models/isla/test_isla.py test/experimental/models/isla/test_isla_inputs.py -x 2>&1 | tail -3
echo "VERIFY-RC=${PIPESTATUS[0]}"
SB
sbatch --parsable kernel/v8/verify_v8.sbatch
