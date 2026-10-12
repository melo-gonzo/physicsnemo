#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J aerozoo-shuffle
#SBATCH -p cpu
#SBATCH -q cpu-normal
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
#SBATCH --open-mode=append
# AEROZOO benchmark (2026-10-09): row-permuted copy of the AeroZoo v1.2 interiors (kernel/aerozoo/aerozoo_shuffle.py).
# 3,816 cases in sorted index order, 1,908 per array task in 48 parallel workers of 40 cases (about 100 s per case on
# Lustre, measured on the login node; the work is I/O bound, so the workers overlap):
#   sbatch --array=0-1 kernel/aerozoo/aerozoo_shuffle.sbatch
# (AGA's cpu QOS: cpu-short admits 4 submitted jobs per user, cpu-normal 2 nodes per user; two wide tasks fit both.)
# Resumable: a case with SHUFFLE.json is skipped; the per-worker log is shuffle_log/<start>-<stop>.jsonl under the copy.
set -u
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
SRC=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/wip_datasets/aerozoo-v1.2
DST=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets/aerozoo_v1.2_shuf_pdmsh
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
trap 'echo "rc=$?" > ${TASK}/STATUS_aerozoo_shuffle_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}' EXIT
PER=1908; WORKERS=48; SUB=40
START=$((SLURM_ARRAY_TASK_ID * PER)); END=$((START + PER))
echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID}: cases ${START}..${END} in ${WORKERS} workers on $(hostname)"
cd "${TASK}/kernel/aerozoo"
for w in $(seq 0 $((WORKERS - 1))); do
  S=$((START + w * SUB)); E=$((S + SUB)); [ "$E" -gt "$END" ] && E=$END
  [ "$S" -ge "$END" ] && break
  "${PY}" -I aerozoo_shuffle.py "${SRC}" "${DST}" --index "${SRC}/index.json" --start "${S}" --stop "${E}" > /dev/null 2>> "${TASK}/sbatch_logs/aerozoo-shuffle-workers.err" &
done
wait
n=$(cat ${DST}/shuffle_log/*.jsonl 2>/dev/null | grep -c '"status": "written"')
echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID} done; copies written so far (all tasks): ${n}"
