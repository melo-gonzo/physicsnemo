#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.az.outliers
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
#SBATCH --open-mode=append
# AEROZOO (2026-10-11): which training cases carry the loss spikes, and how many cases of the full split have
# pathological standardized targets. Step 1 runs the recipe pipeline on the cases outliers_repro.py identified
# (outliers_verify.py, per split's constants); step 2 streams every interior row of the full split's train, val
# and test cases through the numpy copy of the target chain (outliers_scan.py, resumable); step 3 reduces
# everything with the case labels and the per-case validation scores into outliers.json / outliers.txt.
set -u
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
ROOT=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets/aerozoo_v1.2_shuf_pdmsh
PROBE=${TASK}/kernel/aerozoo/probe
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe-td14/bin/python
export PYTHONPATH=${TASK}/code_v26:${TASK}/recipe_v9/src
export OMP_NUM_THREADS=8
cd "${PROBE}"
echo "[$(date -Is)] start on $(hostname), job ${SLURM_JOB_ID}"

CASES="case-0000000000000001731-793b575a97a8 case-0000000000000002407-54a195ddf459 case-0000000000000002344-d27ef3571bc5 case-0000000000000003741-0fba3a592cb2"
for split in nonbox full blobs; do
  # no -I here: isolated mode drops PYTHONPATH, and the recipe pipeline needs code_v26 and recipe_v9/src
  timeout 2400 nice -n 19 "${PY}" outliers_verify.py --split "${split}" --cases ${CASES} --out "verify_${split}.json" \
    > "verify_${split}.log" 2>&1 || echo "[$(date -Is)] verify ${split} rc=$?"
  echo "[$(date -Is)] verify ${split} done"
done

export OMP_NUM_THREADS=1
nice -n 19 "${PY}" -I outliers_scan.py "${ROOT}" "${TASK}/kernel/aerozoo/manifest.json" "${TASK}/kernel/aerozoo/stats/stats_full.train.json" \
  --parts full.train full.val full.test --out outliers_scan.jsonl --workers 16 2>&1
echo "[$(date -Is)] scan rc=$? ($(wc -l < outliers_scan.jsonl) case lines)"

nice -n 19 "${PY}" -I outliers_reduce.py --out-json outliers.json --out-txt outliers.txt 2>&1
echo "[$(date -Is)] reduce rc=$?"
