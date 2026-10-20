#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.stream.af
#SBATCH --time=01:00:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/stream_af_2026-10-19/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/stream_af_2026-10-19/sbatch_logs/%x-%j.log
# STREAM-AF (2026-10-19): the H-RESID Instrument-1 stream probe (book_figs_2026-10-18/stream/hresid_probe.py, md5 514f3c6b...,
# here hresid_probe_af.py: recipe/manifest from env, model.compile_blocks dropped from the lane extra) on the AirfRANS lanes,
# each lane on its own code + recipe snapshot (kernel/airfrans_u/lanes.tsv and the lane's train.log), the training venv,
# bf16, 4 full_test cases (sampler order), 100,000 cells (the lanes' training cells). One lane per GPU. Inference only.
# usage: sbatch stream_af.sbatch "run:code:recipe run:code:recipe ..."  (at most 4)
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
S="${TASK}/stream_af_2026-10-19"
EPOCHS="1,9,19,29,39,43,49,59,69,99,149,199,249,299,349,399,449,499"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
export TRITON_CACHE_DIR=/tmp/$USER/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor
export PROBE_MANIFEST="${TASK}/kernel/airfrans_u/lanes.tsv"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${S}/out"
echo "=== START $(date -Is) job=${SLURM_JOB_ID} node=$(hostname) lanes=$1 md5=$(md5sum < ${S}/hresid_probe_af.py)"
g=0
for spec in $1; do
  IFS=: read RUN CODE RECIPE <<< "${spec}"
  ( cd "${TASK}/${RECIPE}" && env PROBE_RECIPE="${TASK}/${RECIPE}" PYTHONPATH="${TASK}/${CODE}:${TASK}/${RECIPE}/src" \
      CUDA_VISIBLE_DEVICES=$g RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + 17 * g)) \
      "${VENV}/bin/python" "${S}/hresid_probe_af.py" --plan "${RUN}=${EPOCHS}" --k 4 --cells 100000 \
      --dataset airfrans_uvolume_2d --precision bfloat16 --out-dir "${S}/out" > "${S}/sbatch_logs/${RUN}-${SLURM_JOB_ID}.log" 2>&1
    echo "=== LANE END rc=$? $(date -Is) ${RUN} code=${CODE} recipe=${RECIPE} gpu=$g" ) &
  g=$((g + 1))
done
wait
echo "=== END $(date -Is) json=$(find "${S}/out" -name "e*.json" | wc -l)"
