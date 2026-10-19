#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.tset200k.setcheck
#SBATCH --time=01:30:00
#SBATCH -p cpu
#SBATCH -q cpu-short
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=400G
#SBATCH --ntasks-per-node=1
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/tset200k/setcheck-%j.log
# T-SCORESET (2026-10-19): scored-set identity check for the Transolver 512x16 re-scoring at 200,000 cells. Adapted from
# kernel/globe_verify/setcheck.sbatch (md5 dd166f0a); scoredset_check.py here is a byte copy of kernel/globe_verify's (md5
# e611a919). Same check() and the same fixed infer overrides ($S); outputs in kernel/tset200k/scoredset/. GROUP selects the arms:
#   dr  : DrivAerML Surface, Transolver (code_v20 + recipe_v8) at 200k, with and without fx.source=U_inf_dir; native all 48,
#         area/posed first 12. Compared with kernel/globe_verify/scoredset/{dr,dra,drp}_isla_a2r.jsonl (ISLA A2 at 200k).
#   suv : SHIFT-SUV, Transolver cl_r_suv_t (code_v20 + recipe_v8) vs ISLA A2 a2r_suv_isla (code_a2r + recipe_a2r), 200k;
#         native all cases, area/posed first 12.
#   vol : DrivAerML Volume, Transolver cl_r_vol_t (code_v20 + recipe_v8) vs ISLA A2 a2r_vol_isla (code_a2r + recipe_a2r,
#         LHR's copies; the A2 lanes trained and scored on JHB), 200k; native all cases, posed first 12.
#   sbatch --export=ALL,GROUP=<dr|suv|vol> setcheck.sbatch
set -uo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
VF="$T/kernel/tset200k"; OUT="$VF/scoredset"; mkdir -p "$OUT" "$VF/smoke_hydra"
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
export PATH="/usr/bin:/bin:/cm/local/apps/slurm/current/bin" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PHYSICSNEMO_DIST_TIMEOUT_S=900
cd "$VF/smoke_hydra"
S="infer_split=val +save_predictions=false precision=float32 run_id=check dataloader.pin_memory=false"   # pinning (host memory only) needs a GPU driver; it changes no value
check() {  # check <name> <code> <recipe> <max> <overrides...>
  local NAME="$1" CODE="$2" REC="$3" MAX="$4"; shift 4
  ( CUDA_VISIBLE_DEVICES= PYTHONPATH="$T/$CODE:$T/$REC/src" timeout 5000 "$VENV/bin/python" "$VF/scoredset_check.py" "$T/$REC" "$MAX" "$@" > "$OUT/$NAME.jsonl" 2> "$OUT/$NAME.err"
    echo "check $NAME rc=$? $(tail -1 "$OUT/$NAME.jsonl")" ) &
}
TM="model=transolver_surface model.n_hidden=512 model.n_layers=16"
UD="forward_kwargs.fx.source=global_data.U_inf_dir"
IS="model=isla_surface model.hidden=512 model.n_slices=512 model.compile_blocks=true"
case "${GROUP:-dr}" in
dr)
  check dr_t_c200k   code_v20 recipe_v8 0  $TM     dataset=drivaer_ml_surface       sampling_resolution=200000 $S
  check dr_tud_c200k code_v20 recipe_v8 0  $TM $UD dataset=drivaer_ml_surface       sampling_resolution=200000 $S
  check dra_t_c200k   code_v20 recipe_v8 12 $TM     dataset=drivaer_ml_surface_area  sampling_resolution=200000 $S
  check dra_tud_c200k code_v20 recipe_v8 12 $TM $UD dataset=drivaer_ml_surface_area  sampling_resolution=200000 $S
  check drp_t_c200k   code_v20 recipe_v8 12 $TM     dataset=drivaer_ml_surface_posed sampling_resolution=200000 $S
  check drp_tud_c200k code_v20 recipe_v8 12 $TM $UD dataset=drivaer_ml_surface_posed sampling_resolution=200000 $S
  ;;
suv)
  check suv_t_c200k    code_v20 recipe_v8  0  $TM dataset=shift_suv_surface        sampling_resolution=200000 $S
  check suv_isla_a2r   code_a2r recipe_a2r 0  $IS dataloader.prefetch_factor=12 dataset=shift_suv_surface sampling_resolution=200000 $S
  check suva_t_c200k   code_v20 recipe_v8  12 $TM dataset=shift_suv_surface_area   sampling_resolution=200000 $S
  check suva_isla_a2r  code_a2r recipe_a2r 12 $IS dataloader.prefetch_factor=12 dataset=shift_suv_surface_area sampling_resolution=200000 $S
  check suvp_t_c200k   code_v20 recipe_v8  12 $TM dataset=shift_suv_surface_posed  sampling_resolution=200000 $S
  check suvp_isla_a2r  code_a2r recipe_a2r 12 $IS dataloader.prefetch_factor=12 dataset=shift_suv_surface_posed sampling_resolution=200000 $S
  ;;
vol)
  check vol_t_c200k    code_v20 recipe_v8  0  model=transolver_volume model.n_hidden=512 model.n_layers=16 dataset=drivaer_ml_volume_reference sampling_resolution=200000 $S
  check vol_isla_a2r   code_a2r recipe_a2r 0  model=isla_volume model.hidden=512 model.n_slices=512 model.compile_blocks=true dataset=drivaer_ml_volume_reference sampling_resolution=200000 $S
  check volp_t_c200k   code_v20 recipe_v8  12 model=transolver_volume model.n_hidden=512 model.n_layers=16 dataset=drivaer_ml_volume_reference_posed sampling_resolution=200000 $S
  check volp_isla_a2r  code_a2r recipe_a2r 12 model=isla_volume model.hidden=512 model.n_slices=512 model.compile_blocks=true dataset=drivaer_ml_volume_reference_posed sampling_resolution=200000 $S
  ;;
esac
wait
echo "SETCHECK-DONE GROUP=${GROUP:-dr} $(date -Is)"
