#!/bin/bash
# AEROZOO lane submission (AGA). One singleton chain per lane, named isla.az.<slug> so chains never collide.
#   bash kernel/aerozoo/submit.sh smoke            # the two one-epoch smoke lanes (smoke.tsv, release root, no chain-ahead)
#   bash kernel/aerozoo/submit.sh lanes 0 3        # Phase 1 rows 0..3 of lanes.tsv (chain-ahead links in the short QOS)
#   LANE_QOS=normal LANE_TIME=03:59:00 bash kernel/aerozoo/submit.sh lanes 8 15   # held-out-split lanes in the normal QOS
# (short admits 20 submitted jobs per user and a chain holds two, so at most eight chains live in short; the rest go to normal.)
set -euo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
LANE_QOS="${LANE_QOS:-short}"; LANE_TIME="${LANE_TIME:-01:59:00}"
cd "${TASK}"
case "${1:-}" in
  smoke)
    TSV="${TASK}/kernel/aerozoo/smoke.tsv"
    for i in 0 1; do
      slug=$(LANES_TSV="${TSV}" bash kernel/aerozoo/lanes.sbatch --slug "$i")
      sbatch -J "isla.azsmoke.${slug}" --array="$i" --export=ALL,LANES_TSV="${TSV}",NO_CHAIN=1 kernel/aerozoo/lanes.sbatch
    done ;;
  lanes)
    TSV="${TASK}/kernel/aerozoo/lanes.tsv"
    for i in $(seq "$2" "$3"); do
      slug=$(LANES_TSV="${TSV}" bash kernel/aerozoo/lanes.sbatch --slug "$i")
      sbatch -J "isla.az.${slug}" --array="$i" --qos="${LANE_QOS}" --time="${LANE_TIME}" \
        --export=ALL,LANES_TSV="${TSV}",LANE_QOS="${LANE_QOS}",LANE_TIME="${LANE_TIME}" kernel/aerozoo/lanes.sbatch
    done ;;
  *) echo "usage: submit.sh smoke | lanes <first> <last>" >&2; exit 2 ;;
esac
squeue -u "$USER" -h -o '%i %j %T %q %M' | grep -E 'isla\.az'
