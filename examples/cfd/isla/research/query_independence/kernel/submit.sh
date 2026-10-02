#!/bin/bash
# Submit QI lanes <from>..<to> of kernel/lanes.tsv, one singleton chain each (job name isla.qi.<slug>); appends
# "idx run_id job_name first_job_id submitted_at" rows to kernel/submitted.tsv.   usage: kernel/submit.sh <from> <to>
set -euo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-02-isla-independent
cd "${TASK}"
mkdir -p sbatch_logs status
for i in $(seq "$1" "$2"); do
  slug=$(bash kernel/lanes.sbatch --slug "$i")
  run=$(awk -F"\t" -v i="$i" 'NR>1 && $1==i {print $3}' kernel/lanes.tsv)
  jid=$(sbatch --parsable -J "isla.qi.${slug}" --array="$i" --dependency=singleton kernel/lanes.sbatch)
  printf "%s\t%s\t%s\t%s\t%s\n" "$i" "${run}" "isla.qi.${slug}" "${jid%%_*}" "$(date -Is)" | tee -a kernel/submitted.tsv
done
