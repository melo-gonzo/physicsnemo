#!/bin/bash
# relay AGA -> local -> LHR for the two r80_dr_t final checkpoints (T-SCORESET part B, 2026-10-19); datamover endpoints aga1dc, lhr1dc
set -euo pipefail
A=/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
L=$(dirname "$0")/tmp_relay
for s in 42 43; do
  r=r80_dr_t_h512_l16_c100k_seed$s
  mkdir -p $L/$r/checkpoints
  scp -q -o BatchMode=yes aga1dc:$A/runs/$r/checkpoints/{Transolver.0.500.mdlus,checkpoint.0.500.pt,norm_stats.pt} $L/$r/checkpoints/
  scp -q -o BatchMode=yes aga1dc:$A/runs/$r/{train.log,resolved_config.yaml} $L/$r/
done
(cd $L && find . -type f | sort | xargs md5sum) > $(dirname "$0")/md5_local.txt
timeout 120 ssh -o BatchMode=yes lhr1dc "mkdir -p $T/runs_from_aga"
for s in 42 43; do
  r=r80_dr_t_h512_l16_c100k_seed$s
  timeout 120 ssh -o BatchMode=yes lhr1dc "mkdir -p $T/runs_from_aga/$r/checkpoints"
  scp -q -o BatchMode=yes $L/$r/checkpoints/* lhr1dc:$T/runs_from_aga/$r/checkpoints/
  scp -q -o BatchMode=yes $L/$r/train.log $L/$r/resolved_config.yaml lhr1dc:$T/runs_from_aga/$r/
done
timeout 300 ssh -o BatchMode=yes lhr1dc "cd $T/runs_from_aga && find . -type f | sort | xargs md5sum" > $(dirname "$0")/md5_lhr.txt
echo RELAY-DONE
