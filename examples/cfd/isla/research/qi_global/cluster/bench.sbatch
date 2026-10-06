#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J isla.global.bench
#SBATCH --time=00:50:00
#SBATCH -p batch
#SBATCH -q short
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --segment=1
#SBATCH -o sbatch_logs/%x-%j.log
# QI-GLOBAL: training-step cost of the global read (kernel_read_radius=null) at the AirfRANS protocol request
# (bf16, compiled blocks, one GPU per configuration); bench_step.py copied from the kernel lane.
set -uo pipefail
TASK=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-05-isla-qi-global
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
RECIPE=$TASK/recipe_g
CODE=${CODE:-code_g1}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PYTHONPATH="$TASK/$CODE:$RECIPE/src" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $RECIPE
echo "[$(date -Is)] bench on $(hostname) code $CODE $(cat $TASK/$CODE/SNAPSHOT_HEAD)"
QI="++model.query_independent=true"
GL="++model.kernel_read=mean ++model.kernel_read_radius=null"
AF="--model isla_volume_airfrans --dataset airfrans_volume_2d --cells 100000"
AFX="train_split=full_train val_split=full_test"
OUT=$TASK/stats/bench_lhr_${CODE}.jsonl
run() { local gpu=$1; shift; CUDA_VISIBLE_DEVICES=$gpu MASTER_PORT=$((27000+gpu)) $VENV/bin/python $TASK/stats/bench_step.py "$@" --out $OUT; }
( run 0 $AF --label af_ro --extra "$AFX $QI"; run 0 $AF --label af_int --extra "$AFX" ) > $TASK/stats/bench_lhr_${CODE}_g0.log 2>&1 &
( run 1 $AF --label af_global_b12 --extra "$AFX $QI $GL" ) > $TASK/stats/bench_lhr_${CODE}_g1.log 2>&1 &
( run 2 $AF --label af_trunc_sum_b4 --extra "$AFX $QI ++model.kernel_read=sum ++model.kernel_read_radius=0.01 ++model.kernel_read_blocks=4" ) > $TASK/stats/bench_lhr_${CODE}_g2.log 2>&1 &
wait
echo "[$(date -Is)] done"
