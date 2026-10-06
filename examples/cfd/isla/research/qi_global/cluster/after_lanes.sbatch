#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.global.after-lanes
#SBATCH --time=3-00:00:00
#SBATCH -p cpu
#SBATCH -q cpu-free
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -o sbatch_logs/%x-%j.log
# QI-GLOBAL follow-up (2026-10-05): polls every 10 min; whenever lanes of kernel/lanes.tsv say "Training completed",
# submits their scoring and probes (eval/eval.sbatch with LANES_SEL, up to four lanes per node, EVAL_DIR=eval) and
# their bf16 re-scoring (eval/prec.sbatch, eval_prec/bfloat16); once every lane is scored, runs the reduction
# (eval/qg_reduce.py) on the login-free CPU node. A lightweight CPU watcher; no GPU. Records every submission in
# kernel/submitted_eval.tsv and appends to STATUS.md.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global
AG=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
cd "${TASK}"
mkdir -p status eval_prec
echo "[$(date -Is)] watcher ${SLURM_JOB_ID} on $(hostname)"
reduce() {
  "${VENV}/bin/python" eval/qg_reduce.py --scores eval/score refs/qi_score refs/kq_score \
    --probes eval/probe refs/qi_probe refs/kq_probe --out eval/qg_reduction.json > eval/qg_reduction.txt 2>&1
  "${VENV}/bin/python" eval/qg_reduce.py --scores eval_prec/bfloat16/score refs/kq_score_bf16 \
    --out eval/qg_reduction_bf16.json > eval/qg_reduction_bf16.txt 2>&1
}
while true; do
  new=(); n_all=0; n_sub=0
  while IFS=$'\t' read -r idx grp run rest; do
    [ "${idx}" = "idx" ] && continue
    n_all=$((n_all + 1))
    if [ -f "status/EVAL_SUBMITTED_${run}" ]; then n_sub=$((n_sub + 1)); continue; fi
    if grep -q "Training completed" "runs/${run}/train.log" 2>/dev/null; then new+=("${idx}"); fi
    if [ -f "status/BLOCKED_${run}" ]; then echo "[$(date -Is)] BLOCKED ${run}: $(cat status/BLOCKED_${run})"; fi
  done < kernel/lanes.tsv
  if [ "${#new[@]}" -gt 0 ]; then
    for ((k = 0; k < ${#new[@]}; k += 4)); do
      sel="${new[*]:k:4}"
      eid=$(EVAL_DIR=eval N_DETAIL=4 LANES_SEL="${sel}" sbatch --parsable -J isla.global.eval --array=0 --export=ALL eval/eval.sbatch)
      : > eval/prec_units_${eid}.tsv
      for i in ${sel}; do
        IFS=$'\t' read -r _ _ run ds model extra _ _ cells code _ < <(awk -F"\t" -v i="${i}" 'NR>1 && $1==i' kernel/lanes.tsv)
        ex=$(sed -E "s/(^| )\+?training\.[^ ]+//g" <<<"${extra}")
        echo "${run}|${TASK}/${code}|${TASK}/runs|${model}|${ds}|${cells}|${ex}" >> eval/prec_units_${eid}.tsv
        touch "status/EVAL_SUBMITTED_${run}"
      done
      pid=$(UNITS_FILE=eval/prec_units_${eid}.tsv sbatch --parsable -q normal --time=01:30:00 --export=ALL eval/prec.sbatch)
      printf "%s\teval lanes %s\t%s\n%s\tbf16 lanes %s\t%s\n" "${eid}" "${sel}" "$(date -Is)" "${pid}" "${sel}" "$(date -Is)" | tee -a kernel/submitted_eval.tsv
      printf '\n## %s (automatic)\n- Lanes %s complete: scoring and probes %s, bf16 re-scoring %s.\n' "$(date -Is)" "${sel}" "${eid}" "${pid}" >> STATUS.md
    done
    n_sub=$((n_sub + ${#new[@]}))
  fi
  echo "[$(date -Is)] scoring submitted for ${n_sub}/${n_all} lanes"
  if [ "${n_sub}" -eq "${n_all}" ]; then break; fi
  sleep 600
done
### Wait for every eval / bf16 job to leave the queue, then reduce.
while squeue -h -u psharpe -o '%j' | grep -qE '^isla\.global\.(eval|prec)$'; do sleep 300; done
reduce
printf '\n## %s (automatic)\n- Every lane scored; reduction eval/qg_reduction.{json,txt} and eval/qg_reduction_bf16.{json,txt}.\n' "$(date -Is)" >> STATUS.md
echo "[$(date -Is)] done"
