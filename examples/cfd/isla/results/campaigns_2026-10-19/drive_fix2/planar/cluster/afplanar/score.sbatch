#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.afplanar
#SBATCH --time=01:59:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/afplanar/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/afplanar/sbatch_logs/%x-%j.log
#SBATCH --open-mode=append
# AF-PLANAR (DRIVE-FIX2 Part C, 2026-10-19): the in-plane velocity error of the AirfRANS baselines (embedded in 3-D, z = 0).
# A copy of the unit() of kernel/afu_ud/score.sbatch (b23f4507; itself kernel/airfrans_u/score.sbatch f629ffc5 plus the
# afu_ud_* mapping) with the same infer.py call, overrides, snapshot mapping, 100,000 cells and float32, except:
#   +save_predictions=true (the shared scorer passes false), output to $T/evals_afplanar/<run>/<protocol>/,
#   then planar.py recomputes velocity_l2 / _x / _y / _z and the in-plane velocity_xy_l2 per case from the saved
#   predictions, checks them against this run's and the shared scorer's per-case rows, writes planar_cases.csv and
#   planar_summary.json, and the predictions are deleted (inode quota).
# One protocol per job (AFP_PROTO=native|posed); runs in AFP_RUNS go to GPU (i mod 4), sequentially per GPU.
# Run: sbatch -J isla.afplanar.<tag> --export=ALL,AFP_RUNS="<run> ...",AFP_PROTO=native score.sbatch
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
AFP="${TASK}/kernel/afplanar"
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
ROOT="${TASK}/evals_afplanar"
PROTO="${AFP_PROTO:-native}"
read -r -a RUNLIST <<<"${AFP_RUNS:?set AFP_RUNS}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
mkdir -p "${ROOT}"
cd "${AFP}"   # Hydra's outputs/ go to kernel/afplanar/outputs
echo "AFPLANAR START $(date -Is) job=${SLURM_JOB_ID:-} host=$(hostname) proto=${PROTO} runs=${RUNLIST[*]}"
unit() {
  local RUN="$1" gpu="$2"
  local OUT="${ROOT}/${RUN}/${PROTO}"
  [ -f "${OUT}/.done" ] && return 0
  mkdir -p "${OUT}"
  local LANES CODE RECIPE_DIR SHROOT
  case "${RUN}" in
    afu_ud_*) LANES="${TASK}/kernel/afu_ud/lanes.tsv"; CODE="${TASK}/code_v20"; RECIPE_DIR="${TASK}/recipe_v8ud"; SHROOT="${TASK}/evals_afu_ud" ;;
    afu_r_*) LANES="${TASK}/kernel/airfrans_u/lanes.tsv"; CODE="${TASK}/code_v20"; RECIPE_DIR="${TASK}/recipe_v8"; SHROOT="${TASK}/evals_afu" ;;
    *) echo "=== ${RUN} FAILED $(date -Is): not afu_ud_* / afu_r_*" >> "${OUT}.log"; return 0 ;;
  esac
  grep -q "Training completed" "${TASK}/runs/${RUN}/train.log" || { echo "=== ${RUN} FAILED: training not completed" >> "${OUT}.log"; return 0; }
  local LANE; LANE=$(awk -F"\t" -v r="${RUN}" 'NR>1 && $3==r {print; exit}' "${LANES}")
  [ -n "${LANE}" ] || { echo "=== ${RUN} FAILED $(date -Is): not in ${LANES}" >> "${OUT}.log"; return 0; }
  local DATASET MODEL EXTRA LCELLS LCODE CELLS=100000
  DATASET=$(cut -f4 <<<"${LANE}"); MODEL=$(cut -f5 <<<"${LANE}"); EXTRA=$(cut -f6 <<<"${LANE}"); [ "${EXTRA}" = "-" ] && EXTRA=""
  LCELLS=$(cut -f9 <<<"${LANE}"); LCODE=$(cut -f10 <<<"${LANE}")
  if [ "${TASK}/${LCODE}" != "${CODE}" ] || [ "${LCELLS}" != "${CELLS}" ]; then
    echo "=== ${RUN} FAILED $(date -Is): lane table says code=${LCODE} cells=${LCELLS}" >> "${OUT}.log"; return 0
  fi
  local DS="${DATASET}"; [ "${PROTO}" = native ] || DS="${DATASET}_${PROTO}"
  printf "%s\n" "${CODE}" > "${OUT}/SNAPSHOT_USED"; printf "%s\n" "${DS}" > "${OUT}/DATASET_USED"
  EXTRA=$(sed -E "s/(^| )training\.num_epochs=[^ ]+//g" <<<"${EXTRA}")
  EXTRA=$(sed -E "s/(^| )model\.attention_type=/\1+model.attention_type=/g" <<<"${EXTRA}")
  echo "=== ${RUN} START $(date -Is) job=${SLURM_JOB_ID:-} protocol=${PROTO} dataset=${DS} model=${MODEL} extra='${EXTRA}' cells=${CELLS} gpu=${gpu} code=${CODE} recipe=${RECIPE_DIR}" >> "${OUT}.log"
  timeout 6000 env PYTHONPATH="${CODE}:${RECIPE_DIR}/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    "${VENV}/bin/python" "${RECIPE_DIR}/src/infer.py" model="${MODEL}" ${EXTRA} dataset="${DS}" run_id="${RUN}" infer_split=val +save_predictions=true \
    sampling_resolution=${CELLS} precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${OUT}" >> "${OUT}.log" 2>&1 \
    || echo "=== ${RUN} ${PROTO} infer rc=$?" >> "${OUT}.log"
  local SH="${SHROOT}/${RUN}/${PROTO}/${RUN}/metrics.jsonl"; [ -f "${SH}" ] || SH="-"
  echo "=== ${RUN} PLANAR $(date -Is) shared=${SH} npred=$(ls "${OUT}/${RUN}/predictions" 2>/dev/null | wc -l)" >> "${OUT}.log"
  env PYTHONPATH="${CODE}:${RECIPE_DIR}/src" CUDA_VISIBLE_DEVICES="" "${VENV}/bin/python" "${AFP}/planar.py" "${OUT}" "${RUN}" "${SH}" >> "${OUT}.log" 2>&1
  local rc=$?
  echo "=== ${RUN} PLANAR rc=${rc} $(date -Is)" >> "${OUT}.log"
  if [ "${rc}" -eq 0 ]; then rm -rf "${OUT}/${RUN}/predictions"; touch "${OUT}/.done"; fi   # a failed check keeps its dumps for diagnosis
}
for g in 0 1 2 3; do
  ( for ((i = g; i < ${#RUNLIST[@]}; i += 4)); do unit "${RUNLIST[$i]}" "${g}"; done ) &
done
wait
echo "AFPLANAR DONE $(date -Is) job=${SLURM_JOB_ID:-} done=$(for r in "${RUNLIST[@]}"; do [ -f "${ROOT}/${r}/${PROTO}/.done" ] && echo x; done | wc -l)/${#RUNLIST[@]}"
