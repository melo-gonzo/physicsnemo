#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.stream.jeb
#SBATCH --time=00:40:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/stream_af_2026-10-19/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/stream_af_2026-10-19/sbatch_logs/%x-%j.log
# STREAM-AF, DeepJEB++ part (2026-10-19): the same probe (hresid_probe_af2.py = hresid_probe_af.py + PROBE_RUNS checkpoint root)
# on the raw-read lanes cl_r_jeb_isla_seed42/43 (AGA jobs 1017653/1017654; pruned to epochs 39, 40; copied read-only from
# AGA to stream_af_2026-10-19/ckpt) and the A2 lanes jbn2_isla_seed42/43 (LHR runs/). Both: code_v20[a2] + recipe_v8,
# .venv-recipe, bf16, 4 test_clean cases, 200,000 cells (the training count). One lane per GPU. Inference only.
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
S="${TASK}/stream_af_2026-10-19"; RECIPE="${TASK}/recipe_v8"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
export TRITON_CACHE_DIR=/tmp/$USER/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor
export PROBE_MANIFEST="${S}/lanes_jeb.tsv" PROBE_RECIPE="${RECIPE}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${S}/out_jeb"
echo "=== START $(date -Is) job=${SLURM_JOB_ID} node=$(hostname) md5=$(md5sum < ${S}/hresid_probe_af2.py)"
g=0
for spec in cl_r_jeb_isla_seed42:code_v20:${S}/ckpt:39,40 cl_r_jeb_isla_seed43:code_v20:${S}/ckpt:39,40 \
            jbn2_isla_seed42:code_v20a2:${TASK}/runs:1,9,19,29,39,40 jbn2_isla_seed43:code_v20a2:${TASK}/runs:1,9,19,29,39,40; do
  IFS=: read RUN CODE ROOT EPOCHS <<< "${spec}"
  ( cd "${RECIPE}" && env PROBE_RUNS="${ROOT}" PYTHONPATH="${TASK}/${CODE}:${RECIPE}/src" \
      CUDA_VISIBLE_DEVICES=$g RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + 17 * g)) \
      "${VENV}/bin/python" "${S}/hresid_probe_af2.py" --plan "${RUN}=${EPOCHS}" --k 4 --cells 200000 \
      --dataset deepjebpp_surface --precision bfloat16 --out-dir "${S}/out_jeb" > "${S}/sbatch_logs/${RUN}-${SLURM_JOB_ID}.log" 2>&1
    echo "=== LANE END rc=$? $(date -Is) ${RUN} code=${CODE} root=${ROOT} gpu=$g" ) &
  g=$((g + 1))
done
wait
echo "=== END $(date -Is) json=$(find "${S}/out_jeb" -name "e*.json" | wc -l)"
