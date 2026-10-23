# SUV-XFAM jobs and provenance (LHR)

T = /scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0

## Pre-registration
- PREREG.md md5 fe3ae692242da9cb47880d3a4c0178ad (2026-10-22 14:14 UTC, sent to the supervisor).
- Approved 10:20 EDT with two floors; amended 14:20 UTC (section 6): md5 c0d0caef0275fbfed03433f7e4e27186.

## Snapshot additions (2026-10-22 ~14:25 UTC)
- recipe_r090/datasets/shift_suv_surface_fastback.yaml md5 b09692b737376bdcf3502b9e7e1d5da0 (from shift_suv_surface.yaml
  b3f6e2cd; one line: `train_datadir: ${dataset_paths.shift_suv_fastback_root}`). Line recorded in a new file
  recipe_r090/SNAPSHOT_ADDITIONS, because recipe_r090/MD5SUMS covers SNAPSHOT_HEAD (md5sum -c MD5SUMS still passes).
  Modes restored (dr-xr-xr-x; new file r--r--r--).
- recipe_v8/datasets/shift_suv_surface_fastback.yaml md5 c671222b54caf90a3adaf6331e38ce6c (from 30b4b462; same one line).
  Line appended to recipe_v8/SNAPSHOT_HEAD, as its 2026-10-14 entry did. recipe_v8 and its datasets/ were already
  writable (drwxr-xr-x) before this lane; left as found.
- dataset_paths: both snapshots have shift_suv_fastback_root -> .../PhysicsNeMo-ShiftSUV/fastback (recipe_v8 via the
  /home/psharpe/coreai_modulus_cae symlink). Manifests: fastback 796/99/99, estate 794/99/100, no missing runs.

## Launcher $T/kernel/suvx (LAUNCH_MD5S.txt)
- score.sbatch 8a3e12dc: kernel/r090/score.sbatch (a4a24f5e) plus a split column; job isla.suvx.score, short QoS, 1 h.
- score_rows.tsv 134c64a3: 18 rows (6 checkpoints x fb_test, fb_val, est_test), 200000 cells, float32 in the scorer.
- setcheck/setcheck.sbatch 551d9a8d with scoredset_check.py e611a919 (unchanged from kernel/r090fix/setcheck).
- Output: $T/evals_suvx/<run>/<split>/<run>/metrics.jsonl; logs $T/evals_suvx/<run>/<split>.log.

## Jobs
| job | what | submitted (UTC) | state |
|---|---|---|---|
| 333701 | isla.suvx.setcheck, cpu-short: 10 loader hashes (3 loaders x 3 splits + estate-val anchor) | 14:21 | COMPLETED 2:35; all digests match per split, anchor 983a2c64 reproduced |
| 333702_0 | isla.suvx.score smoke, rows 0-3 (ISLA/GT/T s42 fb_test, ISLA s42 est_test) | 14:21 | COMPLETED 3:35; epoch 500, 99/100 cases, float32 |
| 333789_1-4 | isla.suvx.score, rows 4-17 (%4) | 14:27 | COMPLETED 4:26, 5:27, 6:00, 3:38 |

Total 1,386 node-s x 4 GPUs = 1.54 GPU-h. Reduction: reduce.py and percase.py (copies in $T/kernel/suvx) ->
reduce_out.json, percase_out.json. Registry row ^isla\.suvx\. added 14:21 UTC and removed 14:40 UTC (backups
ACTIVE_EXPERIMENTS.tsv.bak, .bak2; a MECH-090 row added meanwhile by another lane was kept).

# Part B (SUV-XFAM-B, fastback-trained ISLA and Transolver; stage 1)

## Pre-registration
- PREREG_B.md md5 6e77d876d1ad7d9b51516d456baa6a2a (2026-10-22 14:55 UTC, sent to the supervisor for approval).
- Approved 15:00 UTC as written. Section 8 added 15:01 UTC (approval line; amendment before submission: expected
  dataset_size 995, not 996): md5 a1104d7c7cf28d78d3b426e32cd0ae33. smoke_check.py now 2263858e (expects 995).
- Registry row ^isla\.suvxb\. added 15:02 UTC (backup ACTIVE_EXPERIMENTS.tsv.bak3 here, cmp clean).

## Jobs (part B)
| job | what | submitted (UTC) | state |
|---|---|---|---|
| 334400 | isla.suvxb.setcheck, cpu-short: splitcheck x 4 (2 lanes + 2 estate anchors), scoredset_check x 8 | 15:02 | COMPLETED 2:55. Lanes read fastback/train (796) and fastback/val (99) under .../fastback, disjoint; anchors estate/train 794, estate/val 99; normalizer = declared constants. Digests = part A's for both loaders: fb_test 4241f134, fb_val 1bb14edd, est_test 24460c3d, est_val 983a2c64 |
| 334401 | isla.suvxb.smoke, short QoS, 4 GPUs: sxb_smoke_isla_seed42 then sxb_smoke_t_seed42, 2 epochs each | 15:02 | COMPLETED 7:38 (ISLA 4.5 min, T 3.1 min). smoke_check ALL PASS: config diff = dataset, run_id, num_epochs only; T fx.source=global_data.U_inf_dir; 199/25 per rank, dataset_size 995; finite; norm_stats = constants. Harmless: ISLA GPU sampler had no isla/sbatch_logs dir (submit.sh creates it for the lanes) |
| 334466 / 334467 | isla.suvxb.isla-seed42 / -seed43 (isla/submit.sh lanes 0 1), normal, 3:59 singleton chains | 15:12 | links s42: 334466 TIMEOUT, 334520 COMPLETED (epoch 500, Training completed), 337904 no-op. s43: 334467 TIMEOUT, 337781 CANCELLED by the supervisor (quota event, epoch 303), 341667 COMPLETED (epoch 500), 342597 no-op |
| 334468 / 334469 | isla.suvxb.t-seed42 / -seed43 (t/submit.sh 0 1), normal, 3:59 singleton chains | 15:12 | links: 334468/334469 TIMEOUT; 337834/337876 CANCELLED by the supervisor (quota event, epoch 229); 341783/341827 running at epoch 435 on 10-23 05:16 UTC, successors pending |
| 345738_0-1 | isla.suvxb.score, short QoS: rows 0-7 (sxb_isla_seed42/43 x fb_test, fb_val, est_test, est_val) | 10-23 05:16 | COMPLETED 4 min; all 8: epoch 500, n 99/99/100/99, requested split and dataset, float32, one START, no FAILED, .done (errors not yet read) |
| 345774_0 / 345772_1 | pending chain-ahead links of sxb_t_seed42 / 43 after Training completed (would no-op) | 10-23 | cancelled by this lane by explicit id, 06:53 UTC |
| 347052_2-3 | isla.suvxb.score, short QoS: rows 8-15 (sxb_t_seed42/43 x 4 splits) | 10-23 06:53 | COMPLETED ~7 min; all 8 pass the per-scoring checks |

Lane checks after training (smoke_check.py on the 4 real runs, 06:53 UTC): ALL PASS; config diff = dataset and run_id only;
T fx.source=global_data.U_inf_dir; last val epochs 497-499.

Reduction. reduce_b.py 25a1a87c (as registered) on lhr2 at 07:00 UTC -> reduce_b_out.json: checks_all_pass false, because
identity check (b) fails on est_val (1.28e-4 on case 00888 CMR, near zero; see PREREG_B section 8, amendment 2). Only the
identity and checks fields were opened before the amendment. Amendment 2 approved by the supervisor and written at
07:04 UTC: PREREG_B md5 a1104d7c -> 57d188c37a9bec8b05fcd1c9a01cafe6. The supervisor's own recomputation:
scratch/isla-main-2026-10-14/suvxb-identity-2026-10-23/identity_detail.py. Amended check identity_b.py
4b03096a905233355dda7be6446276c2 -> identity_b_out.json: pass on all four splits (max statistic fb_test 5.2e-7,
fb_val 5.1e-7, est_test 6.1e-7, est_val 5.5e-7; ids and order match; 4,752 / 4,752 / 4,800 / 4,752 rows).
Stage 1 read 07:05 UTC: RESULTS_B.md. GPU-h 147.5 (sacct).
Erratum added to RESULTS_B.md by the supervisor 2026-10-23 07:03 EDT (book second read): fastback-trained seeds agree
within 0.7 / 0.9 percent on fastback test / val, not 0.5; ISLA estate-test pressure spread 8.4 percent. No verdict
changes. RESULTS_B.md md5 a0e1d7c3 -> c6715009750f4ed84878ac896cceb2e1. Stage 1 is in the book (psharpe/isla 93d53aaac).

# Part B, stage 2 (GeoTransolver)
- Approved by the supervisor 2026-10-23 ~07:10 UTC (cap 125 GPU-h; exclude four bad nodes incl. nvl72d057-T13).
- PREREG_B section 9 written 07:12 UTC before any GeoTransolver smoke, lane or score: PREREG_B md5 57d188c3 ->
  118f36a0e86fe690414f8b3feaaa9c3b.
- Launcher $T/kernel/suvxb/gt by src/build_gt.sh (output kernel_b/build_gt_out.txt): gt/lanes.sbatch e44113e1
  (cl_ud copy, SELF / LANES -> gt, exclude 4 nodes), gt/lanes.tsv 413441c4 (sxb_gt_seed42/43), gt/submit.sh abd7dba8;
  smoke/lanes_smoke_gt.sbatch 8288e209, lanes_smoke_gt.tsv 4e42020f, smoke_gt.sbatch 1ec0a8a1, smoke_check.py d6570561
  (adds the gt arm and its global_embedding check); score/score_rows.tsv 55b4f6b3 (rows 16-23 appended; stage-1 copy
  score_rows.stage1.tsv 955c33b6). LAUNCH_MD5S.txt regenerated over files only.
- Registry row ^isla\.suvxb\. updated 07:13 UTC for stage 2 (backup ACTIVE_EXPERIMENTS.tsv.bak4, cmp clean).

| job | what | submitted (UTC) | state |
|---|---|---|---|
| 347451 | isla.suvxb.smoke-gt, short QoS: sxb_smoke_gt_seed42, 2 epochs, then smoke_check.py | 10-23 07:13 | COMPLETED 5 min; smoke_check ALL PASS (diff dataset, run_id, num_epochs; global_embedding=global_data.U_inf_dir; 199/25, 995; finite; constants). 2-epoch val losses visible (wss_l2 0.306) |
| 347553 / 347554 | isla.suvxb.gt-seed42 / -seed43 (gt/submit.sh 0 1), normal, 3:59 singleton chains | 10-23 07:19 | links 347553/347554, 347562/347563, 351818/351877 TIMEOUT; 354737/354775 COMPLETED (Training completed, epoch 500, ~21:00 UTC); pending no-op links 357433/357471 cancelled by this lane by explicit id 21:24 UTC |
| 358922_4-5 | isla.suvxb.score, short QoS: rows 16-23 (sxb_gt_seed42/43 x 4 splits) | 10-23 21:24 | COMPLETED ~5 min; 8 of 8 pass the per-scoring checks |

Stage 2 lane checks (smoke_check.py on sxb_gt_seed42/43, 21:24 UTC): ALL PASS; config diff = dataset and run_id only;
global_embedding=global_data.U_inf_dir; validation epochs 0-499 without gaps.
Stage 2 reduction: reduce_b2.py f7c66a5d (as registered) on lhr2 21:31 UTC -> reduce_b2_out.json: checks_all_pass true
(amended identity passes on all four splits, max 6.1e-7; registered statistic fails on est_val at 1.28e-4, the same
case 00888 CMR). Read 21:32 UTC: RESULTS_B.md sections 7-10; NOTEBOOK_DRAFT_SUVXB.md (stage-2 entry and chapter changes).
Stage 2 GPU-h 105.0 (sacct: training 52.2 + 52.1, smoke 0.3, scoring 0.4); cap 125. Part B total 252.5 GPU-h.

- Section 9 approved by the supervisor as written, 07:15 UTC (no amendment).
- reduce_b2.py f7c66a5d657ddc05f06b040dcc73ce61, written 07:20 UTC before any GeoTransolver stage-2 score; dry-run on
  a synthetic tree only (deleted). status_gt.sh b99d8eb8 (read by kernel_b/wait_lanes.sh with STATUS=status_gt.sh).

Events. (1) LHR ssh outage 10-22 ~19:16-20:15 UTC (site-wide key refusal); jobs unaffected. (2) LHR project quota full
10-22 ~00:38-00:51 UTC 10-23 (another session's copy): rank 0 of every running link died on Errno 122 and the other ranks
hung; the supervisor cancelled the three hung links (log $T/kernel/storage/quota_cancel_2026-10-22.log), and each
successor resumed from its newest checkpoint. Check 10-23 05:16 UTC: every lane's val_step epochs are continuous from 0
(ISLA 0-499 both seeds; T 0-434, 0-433), no BLOCKED marker, no non-finite value. No action by this lane.

## Launcher $T/kernel/suvxb (built 2026-10-22 ~14:52 UTC by src/build.sh; LAUNCH_MD5S.txt; build output in
   kernel_b/build_out.txt here; local sources kernel_b/)
- isla/: kernel/r090 lanes.sbatch + one `#SBATCH --exclude` line (cb589cb9), site.env and ckpt_preflight.py identical
  (4a6ed5fb, e34bd611), submit.sh with prefix isla.suvxb. (45414a0a); lanes.tsv 1af8c7d1 (sxb_isla_seed42/43).
- t/: kernel/cl_ud lanes.sbatch + exclude line, SELF / LANES -> kernel/suvxb/t (8ec2cf42); lanes.tsv 7cf42f8f
  (sxb_t_seed42/43); submit.sh 1c68f634 (from TASK, as the chain-ahead link does).
- smoke/: smoke.sbatch 85b147e4, lanes_smoke_t.sbatch d9bd6497 (t/lanes.sbatch without the chain-ahead line, smoke TSV),
  lanes_smoke_{isla,t}.tsv (lane 0, run sxb_smoke_<arm>_seed42, 2 epochs), smoke_check.py d5356585.
- score/: score.sbatch 7f65a794 (kernel/suvx's with job name and roots changed), score_rows.tsv 955c33b6 (16 rows).
- setcheck/: setcheck.sbatch 802bcd2b, splitcheck.py 31771eff, scoredset_check.py e611a919 (unchanged).
- status.sh aa51826d (lane status and anomalies; read by local kernel_b/wait_lanes.sh every 30 min) and reduce_b.py
  25a1a87c (written 15:13 UTC, before any fastback-trained score; dry-run on synthetic numbers only).
