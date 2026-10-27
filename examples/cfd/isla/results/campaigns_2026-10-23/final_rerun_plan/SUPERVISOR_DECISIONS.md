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
   seven-field baseline retrain adds about 2,500. **Peter's answer, about 07:32 EDT 10-26, verbatim: "yes, do it".**
   So the AeroZoo baselines retrain on the spec's seven fields: GeoTransolver on all 7 splits and Transolver on the
   full split, 2 seeds each. That is about 5,660 GPU-h in all, on about 13 nodes for 5.5 days. LANES.tsv keeps those
   lanes, and they launch after the ISLA lanes.
3. **HiLift ISLA training file: accepted** (highlift_surface_medium_dedup, as the baselines and the HL-DUP standing).
4. **AeroZoo scorer: accepted.** Apply az7/SPECSCORE_PROPOSAL.diff to a copy of specscore.py in this folder's lhr/
   before any seven-field scoring. AZ-RESCORE's copy stays as its record.
5. **HiLift baselines: dropped from training** (-140 GPU-h). FREEZE-PLAN decision (d) makes HL-DUP stage 2's
   retrained baselines HiLift's baseline standing. Their checkpoints are rescored on the new area file (1).
   Applied 09:00 EDT 10-26: LANES.tsv drops fin_hl_{gt,t}_seed4{2,3} (48 lanes now), and SCORING.tsv's 12 rows point
   at $T/runs/hldupx_hl_{gt,t}_seed4{2,3} (epoch 500). The pre-edit copies are *.before_hl_drop.tsv.
8. **Placement (RERUN-LAUNCH-PLAN, 38b0d1e0; scratch/isla-rerun-launch-2026-10-26/PLAN.md).** Everything runs on LHR
   (about 15 nodes for 5.5 days, ending about 11-03 23:00) unless Peter approves copying the shuffled AeroZoo copy
   (1.48 TB, 794k files) from AGA to JHB before AGA closes on 10-30. With the copy, all 30 AeroZoo lanes, ISLA and
   baselines together, run on JHB and end about 10-31 18:00; the other 18 stay on LHR. DrivAerML and HiLift stay on LHR
   either way. Asked Peter at 09:00 EDT 10-26. **Peter's answer, about 09:33 EDT 10-26, verbatim: "yes".**
   So the copy goes ahead: AGA dc rsync push to JHB's `$U/datasets/aerozoo_v1.2_shuf_pdmsh` (the same path as LHR),
   checked with `sha256sum -c` against LHR's SHA256SUMS and made read-only. A 0.5 GPU-h AeroZoo speed smoke on JHB
   follows, before AeroZoo's lanes are placed there (lane AZ-COPY-JHB).
   **Done (AZ-COPY-JHB, ~14:30 EDT 10-26):**
   - The copy took 2 h 30 min. All 794,452 files pass sha256 against LHR's list (log md5 889c1e78), and the copy is read-only.
   - JHB smokes with the lanes' loader flags: ISLA-DTT 0.812 s, GeoTransolver 0.554 s, against LHR 7-channel references of 0.732 / 0.446.
   - NCCL bandwidth, algorithm and transport are the same on both clusters, and bf16 GEMM throughput is equal.
   - ASSEMBLY-014's LHR smoke of the same 9-channel code (e89e77180) reads ISLA-DTT AeroZoo at 0.783 s. So most of the ISLA-DTT gap is the two added fields, and JHB's own penalty is about 4 %. GeoTransolver's is unmeasured on LHR.
   - **Supervisor's decision: placement C.** All 30 AeroZoo lanes, ISLA and baselines together, run on JHB with LANES.tsv's flags; the other 18 run on LHR.
   - Speed and cost figures in the book come from LHR runs, never from JHB lanes.
   - Watch the first JHB links' epoch times. If they are far slower than the smokes, as HiLift's were, move AeroZoo back to LHR before AGA closes.
6. **DeepJEB++ splits: accepted** (train_clean/test_clean, per REF-090).
7. **$T/kernel/premerge:** the fork deleted it at the supervisor's request (01:18). Noted.
