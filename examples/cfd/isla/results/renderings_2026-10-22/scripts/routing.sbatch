#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.render090.routing
#SBATCH --time=00:59:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=nvl72d015-T12,nvl72d058-T08,nvl72d052-T01
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-22/sbatch_logs/%x-%j.log
#SBATCH -e /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/book_figs_2026-10-22/sbatch_logs/%x-%j.log
# ROUTING-090 (RENDER-090 lane, 2026-10-22): trained routing of the standing ISLA 0.9.0 checkpoints (slices_probe_090.py) on
# the cases of renderings_2026-10-18/routing, one GPU per run, float32, inference only. Each probe is its scorer's infer.py
# command (score_rows.tsv of kernel/r090, r090eag, r090eagaf) with model.compile_blocks=false compile=false, the case's
# reads replayed. PROBES selects the probes (dr, t2d now; lug, af once their eager-trained standings are scored).
#   PROBES=dr,t2d sbatch routing.sbatch
set -uo pipefail
ulimit -s 8192
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
F="${TASK}/book_figs_2026-10-22"; S="${F}/scripts"; O="${F}/routing"; mkdir -p "${O}"
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
PROBES="${PROBES:-dr,t2d}"
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PHYSICSNEMO_DIST_TIMEOUT_S=900
cd "${TASK}"
probe() {
  local gpu="$1" key="$2" run="$3" code="$4" model="$5" ds="$6" cells="$7" sample="$8" replay="$9" keep="${10}" extra="${11}"
  grep -qw "${key}" <<<"${PROBES//,/ }" || return 0
  local tag="${run}_s${sample}" out="${O}/${run}_s${sample}"; mkdir -p "${out}"
  if ! grep -q "Training completed" "${TASK}/runs/${run}/train.log" 2>/dev/null; then echo "=== ${key} NO-OP: ${run} not completed"; return 0; fi
  echo "=== START $(date -Is) gpu=${gpu} ${key} run=${run} code=${code} recipe=recipe_r090 model=${model} dataset=${ds} cells=${cells} sample=${sample} replay=${replay} extra='${extra}'"
  env FIG_SAMPLES="${sample}" FIG_REPLAY="${replay}" FIG_OUT="${out}" PROBE_TAG="${tag}" PROBE_KEEP="${keep}" \
    PYTHONPATH="${TASK}/${code}:${TASK}/recipe_r090/src" CUDA_VISIBLE_DEVICES="${gpu}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=$(/usr/bin/python3 -c "import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1])") \
    timeout 3000 "${VENV}/bin/python" "${S}/slices_probe_090.py" --config-path "${TASK}/recipe_r090/conf" --config-name infer \
    model="${model}" ${extra} model.compile_blocks=false compile=false dataset="${ds}" run_id="${run}" infer_split=val \
    +save_predictions=true sampling_resolution=${cells} precision=float32 checkpoint_dir="${TASK}/runs" output_dir="${out}" > "${out}.log" 2>&1
  echo "=== END rc=$? $(date -Is) ${key}"; grep -v "Warning\|warn" "${out}.log" | tail -n 16
}
I5="model.hidden=512 model.n_slices=512 model.compile_blocks=true"
EAG="${I5} compile=false model.compile_blocks=false"
probe 0 dr r090_dr_isla_seed42 code_r090 isla_surface drivaer_ml_surface 200000 00079 1 200000 "${I5}" &
probe 1 t2d r090eag_t2dz_isla_seed42 code_cmpaf_fix isla_tensile2d plaid_tensile2d_2d 200000 00238 0 60000 "train_split=train_400 val_split=val_100 +training.val_interval=10 ${EAG}" &
probe 2 lug r090eag_lug_isla_seed42 code_cmpaf_fix isla_lug ginot_lug_volume 200000 00633 0 60000 "train_split=train val_split=test +training.val_interval=10 ${EAG}" &
probe 3 af r090eag_af_isla_seed42 code_cmpaf_fix isla_volume_airfrans airfrans_uvolume_2d 100000 00215 1 60000 "train_split=full_train val_split=full_test +training.val_interval=10 ${EAG}" &
wait
echo "RENDER090-ROUTING-DONE $(date -Is)"
