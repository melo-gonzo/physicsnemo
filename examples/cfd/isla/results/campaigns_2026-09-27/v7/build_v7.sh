#!/bin/bash
# Build the code_v7 snapshot on AGA (run ON aga2, in $T): code_v6 + the ISLA package overlay of local HEAD 912d1859
# (model.py with local_rho / local_k, the LOCAL-AGG channel) and the updated tests. recipe_v4 unchanged. Pump case
# la_* -> code_v7/recipe_v4. Never whole-file recipe copies.
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
cd "$T"
[ -d code_v7 ] || cp -a code_v6 code_v7
D=code_v7/physicsnemo/experimental/models/isla
for f in model.py geo_kernel.py nn_kernel.py; do cp kernel/v7/overlay/$f $D/$f; done
cmp -s $D/__init__.py kernel/v7/overlay/__init__.py && echo "init unchanged" || { cp kernel/v7/overlay/__init__.py $D/__init__.py; echo "init updated"; }
for f in test_isla_kernels.py test_isla.py; do cp kernel/v7/tests/$f code_v7/test/experimental/models/isla/$f; done
echo "overlay source commit: $(cat kernel/v7/overlay/SOURCE_COMMIT)" > code_v7/OVERLAY_SOURCE
find code_v7 -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
$PY -m py_compile $D/model.py $D/geo_kernel.py $D/nn_kernel.py && echo compiled
f=kernel/p200k_fleet/p200k_eval_aga.sbatch
grep -q 'la_\*' "$f" || sed -i 's#case "${RUN}" in gbf_\*) #case "${RUN}" in la_*) CODE="${TASK}/code_v7"; RECIPE_DIR="${TASK}/recipe_v4" ;; gbf_*) #' "$f"
grep -c 'la_' "$f"; bash -n "$f" && echo pump-syntax-ok
mkdir -p sbatch_logs
cat > kernel/v7/verify_v7.sbatch <<'SB'
#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.v7.verify
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
export PYTHONPATH=$T/code_v7
cd $T/code_v7 && $PY -m pytest -q test/experimental/models/isla/test_isla_kernels.py test/experimental/models/isla/test_isla_interior_geometry.py test/experimental/models/isla/test_isla.py test/experimental/models/isla/test_isla_inputs.py -x 2>&1 | tail -3
echo "VERIFY-RC=${PIPESTATUS[0]}"
SB
sbatch --parsable kernel/v7/verify_v7.sbatch
