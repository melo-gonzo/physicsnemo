#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.rp.territory
#SBATCH --time=03:00:00
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
# REF-BOARD inference half (2026-09-23): TERRITORY under R1. The same probe and draws as kernel/territory_probe_aga.sbatch
# and territory_wings_aga.sbatch (four validation cases; uniform draws with recipe seeds 11 and 22, the 10:1-biased draw
# with seed 33, so no two draws share the 320,000-cell parent sample; float32), on the R1 checkpoints:
#   GPU 0: cars, 100,000 expected cells (the original protocol; the GeoTransolver-256 dumps of 2026-09-21 are the comparison), seeds 42 and 43
#   GPU 1: cars, 200,000 expected cells (the R1 training protocol), seed 42
#   GPU 2: wings, 200,000 expected cells, seeds 42 and 43
# Snapshot code_ref + the $HOME/rp_probes/recipe_rp copy; dumps under $HOME/rp_probes/territory (scratch file-count quota).
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RP=/home/psharpe/rp_probes
RECIPE=$RP/recipe_rp
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export PYTHONPATH="${TASK}/code_ref:${RECIPE}/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1
[ -d "$RECIPE/src" ] || { echo "recipe copy $RECIPE not visible on $(hostname)"; exit 3; }
M=$RP/manifest.tsv
P=${TASK}/kernel/rp_territory_probe.py
PY=${VENV}/bin/python
cd "${RECIPE}"

draws() {  # draws <gpu> <outdir> <cells> <dataset stem> <runs...>
  local gpu="$1" out="$2" cells="$3" ds="$4"; shift 4
  local td="${out}/dumps"; mkdir -p "${td}"
  export CUDA_VISIBLE_DEVICES="${gpu}" MASTER_PORT=$((25000 + SLURM_JOB_ID % 20000 + gpu))
  timeout 9000 "$PY" "$P" --manifest "$M" --recipe-dir "$RECIPE" --out "$out" --k 4 --cells "$cells" --dataset "${ds}_unif"   --seed 11 --tag unif1  --territory "$td" "$@" >> "${out}.log" 2>&1 || return 1
  timeout 9000 "$PY" "$P" --manifest "$M" --recipe-dir "$RECIPE" --out "$out" --k 4 --cells "$cells" --dataset "${ds}_unif"   --seed 22 --tag unif2  --territory "$td" "$@" >> "${out}.log" 2>&1 || return 1
  timeout 9000 "$PY" "$P" --manifest "$M" --recipe-dir "$RECIPE" --out "$out" --k 4 --cells "$cells" --dataset "${ds}_biased" --seed 33 --tag biased --territory "$td" "$@" >> "${out}.log" 2>&1 || return 1
  echo "DRAWS-DONE ${out}"
}

( draws 0 "$RP/territory/cars100k" 100000 drivaer_bench ref_isla_r1_dr435_seed42 ref_isla_r1_dr435_seed43 ) & p0=$!
( draws 1 "$RP/territory/cars200k" 200000 drivaer_bench ref_isla_r1_dr435_seed42 ) & p1=$!
( draws 2 "$RP/territory/wings200k" 200000 highlift_bench_super_scarce ref_isla_r1_hl35_seed42 ref_isla_r1_hl35_seed43 ) & p2=$!
rc=0
wait $p0 || { echo "cars100k FAILED"; rc=1; }
wait $p1 || { echo "cars200k FAILED"; rc=1; }
wait $p2 || { echo "wings200k FAILED"; rc=1; }
echo "RP-TERRITORY-DONE rc=${rc}"
exit $rc
