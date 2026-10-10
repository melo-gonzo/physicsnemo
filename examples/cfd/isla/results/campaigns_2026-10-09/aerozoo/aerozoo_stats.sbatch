#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J aerozoo-stats
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH --time=03:50:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
#SBATCH --open-mode=append
# AEROZOO benchmark (2026-10-09): target and inlet-global statistics for the dataset yaml (aerozoo_stats.py).
# Samples a seeded 20,000-row interior block of every released case (the three parts of `full` cover all 3,815 planned
# cases; the random rows are read from the release, whose storage order does not matter for a seeded random draw), 16
# workers per part, then reduces per training side: full.train, family_non_box.train, reynolds.train, bank_blobs.train.
set -u
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
ROOT=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/wip_datasets/aerozoo-v1.2
MAN=${TASK}/kernel/aerozoo/manifest.json
OUT=${TASK}/kernel/aerozoo/stats
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
trap 'echo "rc=$?" > ${TASK}/STATUS_aerozoo_stats_${SLURM_JOB_ID}' EXIT
mkdir -p "${OUT}"
cd "${TASK}/kernel/aerozoo"
echo "[$(date -Is)] sampling on $(hostname)"
for part in full.train full.val full.test; do
  n=$("${PY}" -I -c "import json,sys; print(len(json.load(open('${MAN}'))['${part}']))")
  per=$(( (n + 15) / 16 ))
  for w in $(seq 0 15); do
    S=$((w * per)); E=$((S + per)); [ "$E" -gt "$n" ] && E=$n
    [ "$S" -ge "$n" ] && break
    "${PY}" -I aerozoo_stats.py sample "${ROOT}" "${MAN}" "${part}" --out "${OUT}/${part}.${w}.jsonl" --start "${S}" --stop "${E}" \
      >> "${OUT}/sample_${part}.${w}.log" 2>&1 &
  done
done
wait
echo "[$(date -Is)] sampled $(cat ${OUT}/full.*.jsonl | wc -l) case lines"
for split in full family_non_box reynolds bank_blobs; do
  "${PY}" -I - "${MAN}" "${split}.train" "${OUT}" <<'EOF'
import json, sys, glob
man, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
cases = set(json.load(open(man))[key])
seen, n = set(), 0
with open(f"{out}/filtered_{key}.jsonl", "w") as f:
    for p in sorted(glob.glob(f"{out}/full.*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if r["case"] in cases and r["case"] not in seen:
                seen.add(r["case"]); f.write(line); n += 1
print(f"{key}: {n} of {len(cases)} cases sampled", file=sys.stderr)
EOF
  "${PY}" -I aerozoo_stats.py reduce "${OUT}/filtered_${split}.train.jsonl" --out "${OUT}/stats_${split}.train.json" > "${OUT}/yaml_${split}.train.txt" 2>> "${OUT}/reduce.log"
done
echo "[$(date -Is)] reduced; see ${OUT}/yaml_*.txt"
