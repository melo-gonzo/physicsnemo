#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J forceref
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:59:00
#SBATCH -o sbatch_logs/%x-%j.log
# FORCEREF (2026-09-26): float32 inference at 200,000 cells on the DrivAerML Surface validation split for the four
# reference checkpoints (ISLA reference pair, GeoTransolver-512 pair), mirroring kernel/p200k_fleet/p200k_eval_aga.sbatch
# except that the per-sample DomainMesh predictions are kept (+save_predictions=true, physical geometry), then a CPU
# reduction that integrates lift and drag on the sampled cells (Horvitz-Thompson weights) and on the full surface.
# Submit from $TASK:  sbatch kernel/forceref/forceref_aga.sbatch
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
CODE="${TASK}/code_ref"; RECIPE_DIR="${TASK}/recipe_globin"   # SNAPSHOT_USED by the fleet scorer for all four ref_* runs
export PYTHONPATH="${CODE}:${RECIPE_DIR}/src"
ROOT="${TASK}/forceref"; mkdir -p "${ROOT}"
cd "${RECIPE_DIR}"

# run<TAB>model<TAB>extra  (kernel/ref/lanes.tsv rows 0, 2, 28, 29, verbatim)
LANES=$(cat <<'EOF'
ref_isla_r1_dr435_seed42	isla_surface_ref	-
ref_isla_r1_dr435_seed43	isla_surface_ref	-
ref_gt_w512_dr435_seed42	geotransolver_surface	forward_kwargs.global_embedding=global_data.U_inf_dir model.n_hidden=512
ref_gt_w512_dr435_seed43	geotransolver_surface	forward_kwargs.global_embedding=global_data.U_inf_dir model.n_hidden=512
EOF
)

unit() {
  local gpu="$1" RUN="$2" MODEL="$3" EXTRA="$4"; [ "${EXTRA}" = "-" ] && EXTRA=""
  local OUT="${ROOT}/${RUN}"
  [ -f "${OUT}/.done" ] && return 0
  mkdir -p "${OUT}"; printf "%s\n" "${CODE}" > "${OUT}/SNAPSHOT_USED"
  echo "=== START $(date -Is) run=${RUN} model=${MODEL} extra='${EXTRA}' gpu=${gpu} code=${CODE}" >> "${OUT}.log"
  timeout 9000 env CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((24000 + SLURM_JOB_ID % 20000 + gpu)) \
    "${VENV}/bin/python" "${RECIPE_DIR}/src/infer.py" model="${MODEL}" ${EXTRA} dataset=drivaer_ml_surface run_id="${RUN}" infer_split=val \
    +save_predictions=true rescale_geometry=true \
    sampling_resolution=200000 precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${OUT}" >> "${OUT}.log" 2>&1 \
    && touch "${OUT}/.done" || echo "=== ${RUN} FAILED rc=$?" >> "${OUT}.log"
}

g=0
while IFS=$'\t' read -r RUN MODEL EXTRA; do ( unit "${g}" "${RUN}" "${MODEL}" "${EXTRA}" ) & g=$((g + 1)); done <<<"${LANES}"
wait
echo "FORCEREF-INFER-DONE $(date -Is) done=$(find "${ROOT}" -name .done | wc -l)"

RUNS=$(cut -f1 <<<"${LANES}" | tr '\n' ' ')
"${VENV}/bin/python" "${TASK}/kernel/forceref/forceref_integrate.py" "${TASK}" ${RUNS} > "${ROOT}/integrate.log" 2>&1 \
  && echo "FORCEREF-INTEGRATE-DONE $(date -Is)" || echo "FORCEREF-INTEGRATE-FAILED rc=$?"
