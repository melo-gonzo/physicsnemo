#!/bin/bash
# Build the INTERNAL-GEOMETRY snapshot pair on the AGA cluster (run ON aga2, in $T):
#   code_v3   = code_v2 + the ISLA package overlay (model.py; __init__.py unchanged, same md5)
#   recipe_v3 = recipe_v2 + `patch` of the recipe src diff (output_normalize.py) and of the four sg model yamls
# Never copies whole recipe files from the local tree. Then patches the scorer pump for sg_* and submits the
# cpu-partition verification job. Inputs are staged under kernel/sg/ by launch.sh.
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
cd "$T"
[ -d code_v3 ] || cp -a code_v2 code_v3
[ -d recipe_v3 ] || cp -a recipe_v2 recipe_v3
### code overlay: the ISLA package only
cp kernel/sg/overlay/model.py code_v3/physicsnemo/experimental/models/isla/model.py
md5sum code_v3/physicsnemo/experimental/models/isla/model.py kernel/sg/overlay/model.py
cmp -s code_v3/physicsnemo/experimental/models/isla/__init__.py code_v2/physicsnemo/experimental/models/isla/__init__.py && echo "init unchanged"
find code_v3 -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
### recipe patches (idempotent: skip if already applied)
if ! grep -q "_fields_lookup" recipe_v3/src/output_normalize.py; then
  patch -p5 -d recipe_v3 < kernel/sg/recipe_src_v3.patch
fi
if [ ! -f recipe_v3/conf/model/isla_volume_ref_provided.yaml ]; then
  patch -p1 -d recipe_v3/conf/model < kernel/sg/recipe_conf_v3.patch
fi
grep -c "interior_geometry" recipe_v3/conf/model/isla_volume_ref_provided.yaml recipe_v3/conf/model/isla_volume_airfrans_ref_provided.yaml
grep -c "sdf" recipe_v3/conf/model/isla_volume_ref.yaml recipe_v3/conf/model/isla_volume_airfrans_ref.yaml
### scorer pump: sg_* lanes are scored under code_v3 + recipe_v3; the AirfRANS pair at 100k queries
f=kernel/p200k_fleet/p200k_eval_aga.sbatch
grep -q 'sg_\*' "$f" || sed -i 's#case "${RUN}" in v2_\*) CODE="${TASK}/code_v2"; RECIPE_DIR="${TASK}/recipe_v2" ;;#case "${RUN}" in sg_*) CODE="${TASK}/code_v3"; RECIPE_DIR="${TASK}/recipe_v3" ;; v2_*) CODE="${TASK}/code_v2"; RECIPE_DIR="${TASK}/recipe_v2" ;;#' "$f"
grep -q 'sg_af_\*' "$f" || sed -i 's#case "${RUN}" in af_\*|ref_\*_af_\*) CELLS=100000 ;;#case "${RUN}" in af_*|ref_*_af_*|sg_af_*) CELLS=100000 ;;#' "$f"
grep -n 'sg_' "$f" | cut -c1-200
bash -n "$f" && echo pump-syntax-ok
bash -n kernel/sg/lanes.sbatch && echo lanes-syntax-ok
mkdir -p sbatch_logs
sbatch --parsable kernel/sg/verify_v3.sbatch
