#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.rp.bench
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-5
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
# REF-BOARD inference half (2026-09-23, #sec-nb-refboard-prereg): the consistency-benchmark draws of the four R1
# checkpoints (ref_isla_r1_dr435_seed{42,43}, ref_isla_r1_hl35_seed{42,43}) -- independent uniform, stratified
# (Morton systematic) and 10:1 front/back biased draws from the 320,000-cell pool, float32, validation split, the
# same protocol as bench/consistency_bench_aga.sbatch (no seed override: the recipe seed 42 for every draw). One
# array task per (checkpoint, count); its three draws run in parallel on GPUs 0-2. Tasks 0-3 at 200,000 cells (the
# R1 protocol); tasks 4-5 the cars at 100,000 cells, the count of the 192-wide originals and of the GeoTransolver
# rows of results/trainarea_dr_reduction_2026-09-20.json. Snapshot code_ref + the $HOME/rp_probes/recipe_rp copy of
# recipe_globin (six bench yamls and StratifiedSubsampleMesh added, nothing else). Metrics only (+save_predictions=
# false); outputs under $HOME/rp_probes/bench_evals (scratch is at its file-count quota).
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RP=/home/psharpe/rp_probes
RECIPE=$RP/recipe_rp
TIMEOUT_S=9600
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=600
export PYTHONPATH="${TASK}/code_ref:${RECIPE}/src"
[ -d "$RECIPE/src" ] || { echo "recipe copy $RECIPE not visible on $(hostname)"; exit 3; }

case "${SLURM_ARRAY_TASK_ID}" in
  0) RUN=ref_isla_r1_dr435_seed42; DS=drivaer_bench; COUNT=200000; EXTRA="" ;;
  1) RUN=ref_isla_r1_dr435_seed43; DS=drivaer_bench; COUNT=200000; EXTRA="" ;;
  2) RUN=ref_isla_r1_hl35_seed42; DS=highlift_bench_super_scarce; COUNT=200000; EXTRA="model.out_scalars=3 model.out_vectors=2" ;;
  3) RUN=ref_isla_r1_hl35_seed43; DS=highlift_bench_super_scarce; COUNT=200000; EXTRA="model.out_scalars=3 model.out_vectors=2" ;;
  4) RUN=ref_isla_r1_dr435_seed42; DS=drivaer_bench; COUNT=100000; EXTRA="" ;;
  5) RUN=ref_isla_r1_dr435_seed43; DS=drivaer_bench; COUNT=100000; EXTRA="" ;;
  *) echo "no lane"; exit 2 ;;
esac
grep -q "Training completed" "${TASK}/runs/${RUN}/train.log" || { echo "${RUN}: not trained"; exit 1; }

unit() {
  local sampler="$1" gpu="$2" out rc t0 t1
  out="${RP}/bench_evals/${RUN}/${sampler}_${COUNT}"
  if [ -f "${out}/.done" ]; then echo "${RUN} ${sampler}_${COUNT}: already done"; return 0; fi
  mkdir -p "${out}"
  echo "=== START $(date -Is) run=${RUN} dataset=${DS}_${sampler} count=${COUNT} gpu=${gpu} host=$(hostname)" >> "${out}.log"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  timeout "${TIMEOUT_S}" env CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$((21000 + SLURM_JOB_ID % 30000 + gpu)) \
    "${VENV}/bin/python" "${RECIPE}/src/infer.py" \
    model=isla_surface_ref ${EXTRA} dataset="${DS}_${sampler}" run_id="${RUN}" infer_split=val \
    sampling_resolution="${COUNT}" precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${out}" \
    +save_predictions=false >> "${out}.log" 2>&1
  rc=$?
  t1=$(date +%s)
  if grep -qE "skipping load|Could not find valid model file" "${out}.log"; then echo "CHECKPOINT LOAD FAILURE" >> "${out}.log"; rc=98; fi
  if [ "${rc}" -eq 0 ] && ! grep -q '"infer_summary"' "${out}/${RUN}/metrics.jsonl" 2>/dev/null; then echo "no infer_summary row" >> "${out}.log"; rc=97; fi
  echo "LANE-WALL run=${RUN} sampler=${sampler} count=${COUNT} gpu=${gpu} seconds=$((t1 - t0)) rc=${rc}" | tee -a "${out}.log" >> "${RP}/bench_evals/lane_walltimes.log"
  rm -rf "${out}/${RUN}/predictions"
  [ "${rc}" -eq 0 ] && touch "${out}/.done"
  return "${rc}"
}

cd "${RECIPE}"
pids=(); names=()
for pair in "unif:0" "strat:1" "biased:2"; do
  s="${pair%%:*}"; g="${pair##*:}"
  ( unit "$s" "$g" ) & pids+=($!); names+=("$s")
done
rc_all=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then echo "${RUN} ${names[$k]} OK"; else echo "${RUN} ${names[$k]} FAILED"; rc_all=1; fi
done
echo "RP-BENCH-TASK-DONE ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${RUN} ${COUNT} rc_all=${rc_all}"
exit "${rc_all}"
