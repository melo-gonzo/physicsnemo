# Supervisor decisions on RERUN-CONFIGS' open questions (ISLA Autoresearch 01a13c2c, 02:47 EDT 10-26)

The branch is merged: psharpe/isla c52237940 (RERUN-CONFIGS b7196f0d2 plus 99fcc75fb, the S101 fix in
isla_constants_fit.py). The worktree is removed, and LHR $T/kernel/rerunconfigs and $T/kernel/az7_ab are deleted
(4.3 GB, logged in scratch/isla-main-2026-10-14/prune-2026-10-24/rerunconfigs_prune_2026-10-26.log).
$T/kernel/az7_fit (4 MB) is kept.

1. **Area draws: rescore.** Score every model on one draw per area file. The standing GeoTransolver and Transolver
   checkpoints of SHIFT-SUV, and HL-DUP stage 2's HiLift GeoTransolver and Transolver (see 5), are rescored on the
   current area files at the rerun's scoring stage (SCORING.tsv PENDING rows, about 2 GPU-h).
   - The deeper fix would tie each transform's generator to the transform rather than to its list position. It
     changes every draw, so it waits until after the rerun.
2. **Cost: asked Peter at 02:47.** He was told 2,500-4,000 GPU-h. ISLA's 32 lanes are 3,160, and the AeroZoo
   seven-field baseline retrain adds about 2,500. Until he answers, LANES.tsv keeps the AeroZoo baseline lanes. They
   launch after the ISLA lanes, so a "no" costs nothing.
3. **HiLift ISLA training file: accepted** (highlift_surface_medium_dedup, as the baselines and the HL-DUP standing).
4. **AeroZoo scorer: accepted.** Apply az7/SPECSCORE_PROPOSAL.diff to a copy of specscore.py in this folder's lhr/
   before any seven-field scoring. AZ-RESCORE's copy stays as its record.
5. **HiLift baselines: dropped from training** (-140 GPU-h). FREEZE-PLAN decision (d) makes HL-DUP stage 2's
   retrained baselines HiLift's baseline standing. Their checkpoints are rescored on the new area file (1).
6. **DeepJEB++ splits: accepted** (train_clean/test_clean, per REF-090).
7. **$T/kernel/premerge:** the fork deleted it at the supervisor's request (01:18). Noted.
