#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J azv2.pstats
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -c 48
#SBATCH --mem=160G
#SBATCH -o %x.%j.out
# AZ-V2-RECIPE (2026-10-19): per-split train-side statistics of datasets/aerozoo_puvolume*.yaml (aerozoo_pstats.py, CPU).
# Run from a directory holding a copy of aerozoo_pstats.py (LHR kernel/az_compliant/v2/recipe, job 286018). The yaml
# files were then written with `reduce` (the targets scale-only, 2026-10-19) and `fill` on the copied rows.
set -u
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
ROOT=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets/aerozoo_v1.2_shuf_pdmsh
MAN=$T/kernel/aerozoo/manifest.json
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
cd "$SLURM_SUBMIT_DIR"; mkdir -p rows
for part in full.train full.val full.test; do
  n=$("$PY" -I -c "import json; print(len(json.load(open('$MAN'))['$part']))"); per=$(( (n + 15) / 16 ))
  for w in $(seq 0 15); do
    S=$((w * per)); E=$((S + per)); [ "$S" -ge "$n" ] && break
    timeout 3300 "$PY" -I aerozoo_pstats.py boundary "$ROOT" "$MAN" "$part" --out rows/$part.$w.jsonl --start $S --stop $E >> rows/$part.$w.log 2>&1 &
  done
done
wait
echo "boundary rows: $(cat rows/*.jsonl | wc -l)"
for split in full family_non_box reynolds bank_blobs bank_zero_to_cad profile_crossflow profile_dual_inlet; do
  "$PY" -I aerozoo_pstats.py reduce --manifest "$MAN" --split $split.train --label-rows $T/kernel/aerozoo/stats/full.*.jsonl \
    --boundary-rows rows/*.jsonl --out pstats_$split.train.json
done
