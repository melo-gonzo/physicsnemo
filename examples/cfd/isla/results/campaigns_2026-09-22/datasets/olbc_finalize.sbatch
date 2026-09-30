#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J olbc.finalize
#SBATCH --time=00:30:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-1
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
# OLBC finalize (2026-09-22): manifest.json (+ _stats) from the shard results, load-back verification of 3 random cases per
# split, then the Lustre write-lock (chmod -R a-w). Array 0 = poisson, 1 = elasticity. Run with --dependency=afterok:<convert array>.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
U=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe
VENV=$U/physicsnemo-mesh-transformer/.venv-recipe
export PYTHONPATH="${TASK}/code_ref" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
case "${SLURM_ARRAY_TASK_ID}" in
  0) SET=poisson; RAWD=poisson-square-bc5 ;;
  1) SET=elasticity; RAWD=elasticity-circlehollow-m1 ;;
esac
OUT=$U/datasets/olbc_${SET}_pdmsh
echo "=== start $(date -Is) job=$SLURM_JOB_ID node=$(hostname) set=$SET"
cd "${TASK}"
"${VENV}/bin/python" kernel/olbc_convert.py "$U/datasets/olbc_raw/$RAWD" "$OUT" --set $SET --finalize || { echo "FINALIZE FAILED"; exit 1; }
CASES=$("${VENV}/bin/python" - <<EOF
import json, random
m = json.load(open("$OUT/manifest.json")); random.seed(0)
print(" ".join(random.sample(m["train"], 3) + random.sample(m["test"], 1)))
EOF
)
echo "verify cases: $CASES"
"${VENV}/bin/python" kernel/olbc_verify.py "$OUT" $CASES || { echo "VERIFY FAILED"; exit 1; }
echo "--- file count: $(find "$OUT" -name '*.pdmsh' | wc -l); du: $(du -sh "$OUT" | cut -f1)"
chmod -R a-w "$OUT" && echo "write-locked $OUT"
echo "=== FINALIZE-DONE rc=0 $(date -Is)"
