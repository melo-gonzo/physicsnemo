#!/bin/bash
# FRAME-AF local mechanism rung: two arms sequentially on the laptop GPU (baseline, then + laboratory frame), seed 42,
# 120 local AirfRANS cases, 20,000 queries, 300 epochs, lr 1e-3 (the 192-wide rule), bf16, eager compile off.
set -uo pipefail
cd /home/psharpe/gh/physicsnemo/examples/cfd/external_aerodynamics/unified_external_aero_recipe
export PYTHONPATH=/home/psharpe/gh/physicsnemo:$PWD/src RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 PYTHONUNBUFFERED=1
OUT=/home/psharpe/data/runs_local
for arm in base frame; do
  if [ "$arm" = base ]; then MODEL=isla_volume_airfrans_local; DS=airfrans_volume_local; else MODEL=isla_volume_airfrans_local_frame; DS=airfrans_volume_local_frame; fi
  RUN=frameaf_${arm}_seed42
  echo "=== $RUN start $(date -Is)" >> $OUT/frameaf.log
  /home/psharpe/gh/physicsnemo/.venv/bin/python src/train.py model=$MODEL dataset=$DS train_split=local_train val_split=local_test \
    sampling_resolution=20000 run_id=$RUN output_dir=$OUT checkpoint_dir=$OUT training.seed=42 training.num_epochs=300 \
    +training.val_interval=10 training.optimizer.lr=1e-3 compile=false > $OUT/$RUN.log 2>&1
  echo "=== $RUN rc=$? $(date -Is)" >> $OUT/frameaf.log
done
echo FRAMEAF-DONE >> $OUT/frameaf.log
