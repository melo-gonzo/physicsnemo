#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.qi.eval
#SBATCH --time=02:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
#SBATCH --open-mode=append
# QI evaluation: readout 1 (the recipe's infer.py exactly as the fleet scorer runs it: float32, held-out split, training
# request) and readouts 2-5 (eval/qi_probe.py) for the lanes of kernel/lanes.tsv, one GPU per lane, four lanes per node.
# usage: cd $TASK && sbatch --array=0-1 eval/eval.sbatch     (array task k: lanes 4k..4k+3)
# env:   EVAL_DIR=eval_smoke (output root, default eval); LANES_SEL="0 2" explicit lane list for array task 0; SCORE_ONLY=1 / PROBE_ONLY=1; MAX_CASES=N, N_DETAIL=N (probe)
set -uo pipefail
ulimit -s 8192
ulimit -c 0
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-02-isla-independent
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RECIPE="${TASK}/recipe_qi"
LANES="${TASK}/kernel/lanes.tsv"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin" OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
OUT="${TASK}/${EVAL_DIR:-eval}"   # EVAL_DIR=eval_smoke for a smoke test
mkdir -p "${OUT}/score" "${OUT}/probe"
echo "[$(date -Is)] QI eval task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname); lanes from ${LANES}"
unit() {
  local idx="$1" gpu="$2" row RUN DATASET MODEL EXTRA CELLS CODE
  row=$(awk -F"\t" -v i="${idx}" 'NR>1 && $1==i {print; exit}' "${LANES}")
  [ -z "${row}" ] && return 0
  RUN=$(cut -f3 <<<"${row}"); DATASET=$(cut -f4 <<<"${row}"); MODEL=$(cut -f5 <<<"${row}"); EXTRA=$(cut -f6 <<<"${row}")
  CELLS=$(cut -f9 <<<"${row}"); CODE="${TASK}/$(cut -f10 <<<"${row}")"; [ "${EXTRA}" = "-" ] && EXTRA=""
  local INFER_EXTRA; INFER_EXTRA=$(sed -E "s/(^| )\+?training\.[^ ]+//g" <<<"${EXTRA}")
  local ENVV=(env PYTHONPATH="${CODE}:${RECIPE}/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((24000 + SLURM_JOB_ID % 20000 + gpu)))
  cd "${RECIPE}"
  if [ -z "${PROBE_ONLY:-}" ]; then
    echo "[$(date -Is)] lane ${idx} ${RUN}: infer.py (readout 1) gpu=${gpu} extra='${INFER_EXTRA}'"
    timeout 5400 "${ENVV[@]}" "${VENV}/bin/python" "${RECIPE}/src/infer.py" model="${MODEL}" ${INFER_EXTRA} dataset="${DATASET}" run_id="${RUN}" \
      infer_split=val +save_predictions=false sampling_resolution="${CELLS}" precision=float32 checkpoint_dir="${TASK}/runs" \
      output_dir="${OUT}/score" > "${OUT}/score/${RUN}.log" 2>&1 && echo "[$(date -Is)] lane ${idx} ${RUN}: infer done" \
      || echo "[$(date -Is)] lane ${idx} ${RUN}: infer FAILED rc=$? (see ${OUT}/score/${RUN}.log)"
  fi
  if [ -z "${SCORE_ONLY:-}" ]; then
    echo "[$(date -Is)] lane ${idx} ${RUN}: qi_probe.py (readouts 2-5) gpu=${gpu}"
    timeout 5400 "${ENVV[@]}" "${VENV}/bin/python" "${TASK}/eval/qi_probe.py" --model "${MODEL}" --dataset "${DATASET}" --run-id "${RUN}" \
      --cells "${CELLS}" --extra "${EXTRA}" --task "${TASK}" --out "${OUT}/probe/${RUN}.json" \
      --n-detail "${N_DETAIL:-4}" --max-cases "${MAX_CASES:-0}" > "${OUT}/probe/${RUN}.log" 2>&1 \
      && echo "[$(date -Is)] lane ${idx} ${RUN}: probe done" || echo "[$(date -Is)] lane ${idx} ${RUN}: probe FAILED rc=$? (see ${OUT}/probe/${RUN}.log)"
  fi
}
if [ -n "${LANES_SEL:-}" ]; then read -r -a sel <<<"${LANES_SEL}"; else sel=(); for g in 0 1 2 3; do sel+=( $((SLURM_ARRAY_TASK_ID * 4 + g)) ); done; fi
g=0
for i in "${sel[@]}"; do ( unit "${i}" "${g}" ) & g=$((g + 1)); done
wait
echo "[$(date -Is)] QI eval task done: lanes ${sel[*]}"
