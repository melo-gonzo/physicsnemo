#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J jeb.finalize
#SBATCH --time=04:00:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# DeepJEB++ finalize (2026-09-22): manifest.json (+ _stats) from the shard results (original + rescore records), load-back
# verification of 5 random files (one per load case + one more), then the Lustre write-lock (chmod -R a-w) of the converted tree,
# parallel over the 61,440 case dirs (a serial chmod -R of the 341 GB tree exceeded a 1 h wall on the first attempt). The raw
# tree stays writable. Run with --dependency=afterany:<convert or rescore array>.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
U=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe
VENV=$U/physicsnemo-mesh-transformer/.venv-recipe
export PYTHONPATH="${TASK}/code_ref" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
OUT=$U/datasets/deepjebpp_pdmsh
echo "=== start $(date -Is) job=$SLURM_JOB_ID node=$(hostname)"
cd "${TASK}"
chmod u+w "$OUT" "$OUT"/manifest.json 2>/dev/null
"${VENV}/bin/python" kernel/deepjebpp_convert.py "$OUT/deepjebpp_raw" "$OUT" --finalize || { echo "FINALIZE FAILED"; exit 1; }
CASES=$("${VENV}/bin/python" - <<EOF
import json, random
m = json.load(open("$OUT/manifest.json")); random.seed(0)
print(" ".join([random.choice(m[f"train_{ld}"]) for ld in ("ver", "hor", "dia", "tor")] + [random.choice(m["test_tor"])]))
EOF
)
echo "verify cases: $CASES"
"${VENV}/bin/python" kernel/deepjebpp_verify.py "$OUT" $CASES || { echo "VERIFY FAILED"; exit 1; }
echo "--- file count: $(find "$OUT" -maxdepth 2 -name '*.pdmsh' | wc -l) $(date -Is)"
find "$OUT" -maxdepth 1 -mindepth 1 -type d ! -name deepjebpp_raw -print0 | xargs -0 -P 16 -n 256 chmod -R a-w
chmod a-w "$OUT"/manifest.json "$OUT"/convert_results_*.json
echo "write-locked converted tree under $OUT ($(find "$OUT" -maxdepth 1 -mindepth 1 -type d ! -name deepjebpp_raw ! -perm -u+w | wc -l) locked case dirs) $(date -Is)"
echo "=== FINALIZE-DONE rc=0 $(date -Is)"
