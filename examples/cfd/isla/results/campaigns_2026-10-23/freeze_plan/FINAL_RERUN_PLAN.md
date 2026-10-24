# The final reference rerun, lane by lane (FREEZE-PLAN, 2026-10-24)

Written 2026-10-24 from 12:19 EDT by FREEZE-PLAN, a read-only planning lane. No job was run. The numbers come from
`plan_model.py` in this folder (it writes `plan_tables.md`). Allocated GPU-h come from `sacct_all.psv`, a sacct dump of
LHR jobs since 2026-10-18, summed with `agg.py`. "Measured" means a number read from a log or from sacct. "Estimated"
(EST) means a number scaled from measured step-time ratios, and its source is named.

## 1. What the rerun trains

Canonical ISLA on the nine datasets of the book's standings (08-standing.qmd at origin/psharpe/isla eba038f08, and
chapters 09-17). Two seeds, 42 and 43, except Tensile2d, whose standing has four seeds (42-45; REF-090
`t2d_standing.md`). Each dataset keeps its registered epochs, points and settings (REF-090 PREREG section 3; AZ PREREG_v2
section 4; HL-DUP PREREG). Every lane runs on one 4-GPU GB300 node (LHR's minimum), in bf16, with compile_blocks.

| dataset | form | seeds | points | epochs | steps/epoch | registered by |
|---|---|---|---|---|---|---|
| DrivAerML Volume, whole boundary | ISLA-DTT | 2 | 200k | 500 | 108 | REF-090 rows 8-9 |
| AeroZoo full and six held-out splits | ISLA-DTT | 2 each (14 lanes) | 200k | 250 | 529-760 | AZ PREREG_v2 section 4 |
| AirfRANS, with its far field | ISLA-DTT | 2 | 100k | 500 | 200 | REF-090 rows 12-15, AF-WHOLE |
| lug (GINOT) | ISLA-DTT | 2 | 20k (9,808 nodes) | 500 | 600 | REF-090 rows 4-5, 16-17 |
| Tensile2d | ISLA-DTT | 4 | 20k | 500 | 100 | REF-090 rows 2-3 and the seed 44/45 amendment |
| DrivAerML Surface, with the tunnel pieces | option A | 2 | 200k | 500 | 108 | REF-090 rows 6-7, SURFACE-BC |
| SHIFT-SUV, with its CFD box | option A | 2 | 200k | 500 | 198 | REF-090 rows 0-1, BC-SUPPLY |
| HiLift, deduplicated, with its far field | option A | 2 | 200k | 500 | 128 | HILIFT-090, HL-DUP, BC-SUPPLY |
| DeepJEB++ | option A | 2 | 200k | 40 | 12,451 | REF-090 rows 10-11 |

That is 32 ISLA lanes. Tensile2d's ISLA-DTT has never been trained, so its step time is the least certain number here.

## 2. The code scenarios

Supervisor's steer of 12:25 EDT: price S2 on DTT-FAST's latest measured head, and add S4. All of DTT-FAST is unmerged.

- **S1**: the merged head eba038f08 (0.11.0 with the fast geometry, the thin-cell fix and ALIGN8's token padding),
  with ISLA-DTT as the default interior form.
- **S2-pass2**: S1 with DTT-FAST's pass 2 (757a5a877), as first briefed. Kept for reference.
- **S2**: S1 with DTT-FAST at its latest measured state (**unmerged**). That is pass 3c (5e82a9b98, TABLE.md job
  368334) times the step rule (ab6f253c3: vehicle 200k 1,162 to 1,148 ms, PLAN.md 11:35) times the faster fused-read
  backward (fba6f5699: 0.953x the step at vehicle 200k, PLAN.md 12:12). Where those were measured on DrivAerML only,
  the AeroZoo and AirfRANS factors are EST: readbwd 0.97 on AeroZoo (PROFILE-2: its read premium is 93 ms, of which
  readbwd cuts the backward by 29 %), and 0.99 on AirfRANS. The rule changes nothing where one chunk already fits.
  For the lug, PROFILE-2 (job 368376) measured ISLA-DTT at 242 ms on 5e82a9b98, on one lug case. I added the
  thin-cell delta of 41 ms, which gives 0.28 s. readbwd's review (12:24) asks for the rule's coefficients to be refit
  in pass 4, because readbwd adds 3.5-4.9 GiB.
- **S3**: S2 plus DTT-POSE's forest (0.12.0). DTT-POSE measured -12 % on the DrivAerML Volume step against 0.11.0
  (DTT-POSE FINAL_REPORT.md: 1,540/1,525 against 1,751/1,728 ms, job 366948). I apply that to DrivAerML only. It was
  measured without DTT-FAST, and the two change the same forest, so the -12 % may not add to DTT-FAST's gain. No
  forest speed is measured on AeroZoo, AirfRANS or the lug, so S3 carries none there.
- **S3 assumes a port with no lane yet.** DTT-FAST's fast forest build and walk match the *old* forest bit for bit.
  DTT-POSE's FINAL_REPORT lists five changes they need for the new forest (FREEZE_DECISIONS.md section 2). If S3
  ships without that port, ISLA-DTT runs the reference build and walk. That gives back about 90 ms (forest build;
  DTT-FAST forest/RESULTS.md) plus 110 ms (walk; walk/REPORT.md) a step on DrivAerML at 200k, and about 150 ms on
  AeroZoo (EST from the 100k walk's 64 ms). So **unported S3 is slower than S2**: about +10 % on DrivAerML and
  +16 % on AeroZoo. My EST for that case is about 3,750-3,800 GPU-h in all, against S2's 3,380.

- **S4**: S2 with 128 slices instead of 512, as the supervisor asked. See section 3 for how I estimated it.
  SLICES-DRV (RESULTS.md, md5 b2fd11e2) read "no change" for interacting ISLA 0.9.0 on DrivAerML Volume: a 0.625x
  step on the same node, 0.83-0.86x memory, 0.70x GPU-h a lane. SLICES-AF read "no change" on AirfRANS at a 0.42x
  step. 128 slices has never been tested on ISLA-DTT, on a surface dataset, or on solid mechanics. Adopting it changes
  the reference configuration, so it needs Peter (FREEZE_DECISIONS.md section 8).

**Geometry at the merged head** (GEO-BAND PHASE0.md, job 368333, fast path, one call per training step):
- The lug's call rose from 36.7 to 77.7 ms (+112 %) with the thin-cell fix.
- DrivAerML whole boundary rose from 353.7 to 432.5 ms, AeroZoo from 83.0 to 110.1 ms, and AirfRANS from 30.1 to
  33.5 ms (noise).
- Against 0.9.0's geometry, the lug adds +69.5 ms a step and AeroZoo +91 ms. On the lug, ISLA's measured step at the
  head (220 ms) equals the 0.9.0 lane (152 ms) plus this delta, which supports adding the delta as a constant a step.
- The surface tasks make no geometry call (SURFACE-BC cost_table.md), so the geometry does not affect them.

## 3. Step times and how each was obtained

All times are per training step on 4 ranks, bf16, compiled blocks.

- **AeroZoo (measured, then scaled).** S1 is each split's measured 0.9.0 lane median (AZ-DECIDE `tables_final.md` Cost,
  1.20-1.34 s) plus the head's +91 ms of geometry. S2-pass2 multiplies that by DTT-FAST's pass-2 ratio at the head's
  geometry, (800 + 27) / (1,236 + 27) = 0.654. S2 multiplies it by (741 x 0.97 + 27) / (1,236 + 27) = 0.591.
- **DrivAerML Volume (estimated).** No ISLA-DTT lane was ever trained on this dataset. S1 is DTT-FAST's harness step at
  7213c70c5 on the whole boundary at 200k (2,089 ms, the mean of jobs 366717 and 367435), plus the thin-cell delta of
  79 ms. I multiplied that by 1.08, the lane-to-harness ratio measured on AeroZoo ISLA-DTT. The result is 2.34 s.
  S2-pass2 is 0.700 times S1, 1.64 s. S2 is 0.603 times S1, 1.41 s; that ratio is (1,305 x 0.988 x 0.953 + 79) /
  (2,089 + 79). S3 takes off another 12 %, which gives 1.24 s.
- **AirfRANS (measured, then scaled).** S1 is the measured r090fix_af_dtt step, 0.571 s, times the harness ratio of the
  whole boundary to the airfoil at 100k (1.005), plus 21 ms of geometry. That gives 0.595 s. S2-pass2 is 0.819 times
  S1, 0.487 s; S2 is 0.780 times S1, 0.464 s.
- **lug.** S1 is measured: r090fix_lug_dtt's 0.424 s plus 69.5 ms of geometry, 0.493 s. S2-pass2 is EST: I applied
  AirfRANS's pass-2 ratio to the part of the step outside the geometry, which gives 0.418 s. S2 is PROFILE-2's
  measured 242 ms plus 41 ms, times 0.99, which gives 0.280 s. That is one case, and the lug's step is host-bound.
- **Tensile2d (estimated).** The interacting step, 0.147 s, times the lug's ISLA-DTT ÷ ISLA ratio at the head (2.23),
  gives 0.328 s in S1. AirfRANS's ratio (1.53) would give 0.22 s. Later scenarios scale it like the lug.
- **128 slices (S4), estimated from the blocks' share of the step.**
  1. Block cost at 128 slices over 512, f, from SLICES-DRV. Interacting ISLA's step fell to 0.625x on DrivAerML
     Volume, and PROFILE-2 puts ISLA's slice blocks at 389.5 of about 593 ms there (104.9 + 284.6 device ms, forward
     plus backward, vehicle 200k). So f = 1 - 0.375 x 593 / 389.5 = 0.43. I applied the same f to the slice blocks
     and to the points' block reads. A point's read is over the slices, so it should scale like the blocks, but
     that is not measured.
  2. ISLA-DTT's block share, from PROFILE-2's `tables_368376.md` (slice blocks plus point block reads, F plus B,
     device ms, over the timed step):
     - DrivAerML whole boundary: 308.7 / 1,298 ms, 24 %;
     - AeroZoo: 326.6 / 743 ms, 44 %;
     - lug: 63.9 / 242 ms, 26 %. Its step is host-bound, so the device saving may not show in wall time.
     AirfRANS has no profile at the head; I took 40 % (EST).
  3. The S4 factor is 1 - (1 - f) x share:
     - DrivAerML Volume 0.86; AeroZoo 0.75; lug and Tensile2d 0.85; AirfRANS 0.77.
     - Option A surfaces: 0.54, with every token going through the blocks and no geometry call (EST share 80 %).
     - DeepJEB++: 0.69 (EST share 55 %, since its mean step is 1.7x its median).
  4. With f from 0.35 to 0.50 and the surface share from 0.60 to 0.85, S4's training plus overhead ranges from 2,355
     to 2,610 GPU-h.
- **Option A surfaces.** DrivAerML Surface: r090_dr_isla's 0.243 s times 1.03 for the tunnel tokens (SURFACE-BC section
  3, at aligned token counts). SHIFT-SUV: 0.243 s times 1.03 (EST; BC-SUPPLY did not time the box). HiLift: HL-DUP's
  deduplicated lanes measure 0.321-0.323 s; I added 3 % for the far field (EST). DeepJEB++: r090_jb_isla's median,
  0.132 s. DTT-FAST does not touch these four, so S1, S2 and S3 are the same; S4 applies the slice factor.
- **Allocated overhead** is each dataset's measured ratio of allocated GPU-h (sacct, all links) to the pure step
  GPU-h of its latest lanes. It covers validation, data loading, link restarts, compile warm-up and requeues:
  - 1.05: AirfRANS;
  - 1.09-1.14: SHIFT-SUV, Tensile2d, the DrivAerML tasks and the lug;
  - 1.12-1.21: AeroZoo, per split;
  - 1.34: HiLift (data loading);
  - 1.35: DeepJEB++ (its mean step, 0.228 s in the smoke, is far above its median).
- **Scoring** is from the score jobs' sacct:
  - REF-090: 12.1 GPU-h over 16 jobs, about 0.7 a lane;
  - AeroZoo: test-eval-isla, 50.7 GPU-h over 28 lanes, 1.8 a lane;
  - ISLA-DTT on DrivAerML Volume, AirfRANS and the lug: 1.0-2.5 a lane (EST).

Harness and lane steps disagree by -8 to +8 % (AeroZoo, AirfRANS), so read every total as ±10 %.

## 4. Per lane (per seed; full table in `plan_tables.md`)

| lane | seeds | step S1 / S2-pass2 / S2 / S3 / S4 [s] | overhead | allocated GPU-h S1 / S2-pass2 / S2 / S3 / S4 | scoring |
|---|---|---|---|---|---|
| DrivAerML Volume, ISLA-DTT | 2 | 2.34 / 1.64 / 1.41 / 1.24 / 1.22 (all EST) | 1.14 | 161 / 113 / 97 / 85 / 84 | 2.5 |
| AeroZoo full | 2 | 1.39 / 0.91 / 0.82 / 0.82 / 0.62 | 1.17 | 308 / 202 / 182 / 182 / 136 | 1.8 |
| AeroZoo nonbox | 2 | 1.29 / 0.85 / 0.76 / 0.76 / 0.57 | 1.13 | 243 / 159 / 144 / 144 / 108 | 1.8 |
| AeroZoo reynolds | 2 | 1.38 / 0.90 / 0.81 / 0.81 / 0.61 | 1.15 | 232 / 152 / 137 / 137 / 103 | 1.8 |
| AeroZoo blobs | 2 | 1.39 / 0.91 / 0.82 / 0.82 / 0.61 | 1.15 | 240 / 157 / 142 / 142 / 106 | 1.8 |
| AeroZoo bank_zero_to_cad | 2 | 1.40 / 0.92 / 0.83 / 0.83 / 0.62 | 1.21 | 357 / 234 / 211 / 211 / 158 | 1.8 |
| AeroZoo profile_crossflow | 2 | 1.43 / 0.93 / 0.84 / 0.84 / 0.63 | 1.12 | 314 / 206 / 185 / 185 / 139 | 1.8 |
| AeroZoo profile_dual_inlet | 2 | 1.43 / 0.94 / 0.84 / 0.84 / 0.63 | 1.13 | 324 / 212 / 191 / 191 / 143 | 1.8 |
| AirfRANS, ISLA-DTT | 2 | 0.595 / 0.487 / 0.464 / 0.464 / 0.358 | 1.05 | 69 / 57 / 54 / 54 / 42 | 1.5 |
| lug, ISLA-DTT | 2 | 0.493 / 0.418 / 0.280 / 0.280 / 0.238 | 1.15 | 189 / 160 / 107 / 107 / 91 | 1.0 |
| Tensile2d, ISLA-DTT (EST) | 4 | 0.328 / 0.268 / 0.186 / 0.186 / 0.158 | 1.11 | 20 / 17 / 12 / 12 / 10 | 0.4 |
| DrivAerML Surface, option A | 2 | 0.250 / = / = / = / 0.136 | 1.12 | 17 / 17 / 17 / 17 / 9 | 0.7 |
| SHIFT-SUV, option A | 2 | 0.250 (EST) / = / = / = / 0.136 | 1.09 | 30 / 30 / 30 / 30 / 16 | 0.7 |
| HiLift, option A | 2 | 0.332 / = / = / = / 0.180 | 1.34 | 32 / 32 / 32 / 32 / 17 | 0.7 |
| DeepJEB++, option A | 2 | 0.132 / = / = / = / 0.091 | 1.35 | 99 / 99 / 99 / 99 / 68 | 0.7 |

The table gives each lane's steps per epoch and epochs in section 1, and its training GPU-h (without overhead) in
`plan_tables.md`. Every lane uses 4 GPUs.
The step is the same for every seed. The S1 AeroZoo total, 4,036 GPU-h, is 7 % above the 0.9.0 lanes' measured
3,772 GPU-h, because of the head's geometry.

## 5. Certification first (LANE_RULES "Compiled regions")

Every dataset's compiled regions are certified before its lanes launch. The certification is block by block, on real
checkpoints, against float64 eager, per tensor group, with the one-ulp yardstick, and with a family of at least 7
samples for bf16.
- **Measured cost basis.** DTT-FAST's certification jobs 363857 and 364478 certified AeroZoo s42/s43, AirfRANS s42/s43
  and a random-initialised DrivAerML model, and timed them, in 15 minutes and 1.0 GPU-h each. The FLAG job 368799 ran
  7-sample one-ulp families in 0.9 GPU-h.
- **Estimate.** Nine datasets, at about 1 GPU-h each, with one retry, comes to about 25 GPU-h. With smokes of every
  config (REF-090's took 4.5 GPU-h for 9 configs), about 37 GPU-h in all. Under the rule of 1-2 short-QoS jobs at a
  time, this takes about one calendar day, so it needs the frozen code by 10-27 evening.
- **Gap: almost no real checkpoint loads into the final code.** 0.11.0 refuses 0.9.0 state dicts. DTT-POSE's DESIGN
  (section 9) proposes that 0.12.0 also refuse version-11 ISLA-DTT state dicts. The few 0.11.0 checkpoints, such as
  TIE-EXP's four, are interacting ISLA. No ISLA-DTT checkpoint was ever trained on DrivAerML Volume or Tensile2d at
  any version. So certification "on real checkpoints" of the rerun's code needs checkpoints made by
  that code. There are two ways, and the freeze must pick one:
  - (a) A short pre-training per dataset on 10-27/28, for example 10 epochs (EST about 30-40 GPU-h in all; AeroZoo
    full is about 7 of it), then certify on those. This adds about half a day.
  - (b) Launch the rerun on 10-29 and certify at each lane's first saved checkpoint (save_interval 2). A failure
    then restarts that dataset's lanes eager and loses their first hours.
  Under either, the ±1-ulp yardstick needs eager's own change, which is measured on the same checkpoints.
- **If certification fails** on AirfRANS, the lug or Tensile2d, those lanes train eager. That adds 282 GPU-h in S1, or
  179 in S2 and S3 (147 in S4), from the measured eager-to-compiled ratios (r090eag against r090fix: 1.66x for AirfRANS ISLA-DTT,
  1.41x for the lug, 1.42x for Tensile2d). The AirfRANS lanes also take about 1.7 times as long.

## 6. Totals and wall-clock on LHR

| scenario | ISLA training | allocated overhead | scoring | certification + smokes | **total GPU-h** | longest lane (elapsed) | calendar at 8 / 12 / 16 nodes | fewest nodes for 5.5 days |
|---|---|---|---|---|---|---|---|---|
| S1 (merged head) | 4,587 | 722 | 42 | 37 | **5,390** | 89 h (AeroZoo bank_zero_to_cad) | 8.5 / 5.7 / 4.3 days | 14 |
| S2-pass2 | 3,208 | 513 | 42 | 37 | **3,800** | 58 h | 5.7 / 3.8 / 3.2 days | 9 |
| S2 (DTT-FAST latest, unmerged) | 2,842 | 458 | 42 | 37 | **3,380** | 53 h | 5.1 / 3.4 / 2.8 days | 8 |
| S3 (S2 + forest, ported) | 2,822 | 455 | 42 | 37 | **3,360** | 53 h | 5.0 / 3.4 / 2.7 days | 8 |
| S4 (S2 + 128 slices) | 2,137 | 341 | 42 | 37 | **2,560** (2,440-2,690) | 40 h | 3.9 / 2.6 / 2.1 days | 6 |

- **How the calendar is computed.** Each lane holds one node. The calendar is the longest-first (LPT) makespan of the
  lanes' elapsed node-hours, times 1.15. That factor is the measured calendar ÷ elapsed of the chains (sacct, 1.00-1.25
  outside maintenance windows). Once there are enough nodes, the calendar equals the longest lane: from 16 nodes in S1
  and from about 20 in S2.
- **The interior forms cost most.** ISLA-DTT accounts for 4,955 GPU-h in S1 and 2,946 in S2; the four surface
  datasets for 355 (221 in S4). AeroZoo alone is 75 % of S1 and 70 % of S2.
- **Sensitivities:**
  - DTT-FAST's pieces still running (plan, fused basis backward, pass 4) could cut more. PROFILE-2 estimates that all
    seven ranked fixes would bring DrivAerML vehicle 200k to 1.22-1.27x interacting ISLA, against 1.53x at pass 3c.
  - eager fallback on AirfRANS, the lug and Tensile2d: +282 GPU-h in S1, +179 in S2 and S3, +147 in S4;
  - unported S3 (section 2): about +400 GPU-h over S2.
- **Not in the totals:** the SHIFT-SUV part B ISLA pair, if chapter 08's families section keeps its fastback-trained
  ISLA. The last such lanes, suvxb_isla, took 30-33 GPU-h each, so about 62 GPU-h for the pair.
- **Current load.** LHR ran 64 GPU nodes of this account at 12:1x today (54 normal, 10 short; squeue), so 8-16 nodes is
  a modest share. The partition showed 1,009 nodes allocated and 1 idle, so start latency is the risk, not the cap.

## 7. Baselines

Baselines keep their released code (LANE_RULES). A baseline lane must be retrained only when its inputs changed since
its standing was measured. BASELINE_PARITY.md (with REPORT.md section 7) gives two options.

**Option 0: baselines as released.** They read what their configs read today: points, a stored or computed SDF to the
body, labels, and on AeroZoo the merged boundary. Here is what each change since the standings does to them:
- **DrivAerML floor fix** (PlaceGroundUnderBody, c4419b299). It translates the named domain meshes only. The Volume
  baselines set `drop_in_file_boundaries: true` and never load those meshes, and the interior points and the vehicle
  SDF do not change. Their standings (clud_vol_gt, clud_vol_t, 10-19) stand. One static risk: the transform refuses
  missing names, and it runs first in `drivaer_ml_volume_reference*`. If a baseline config reads that file with the
  domain meshes dropped, it would fail at once. drv-ground says the baselines use `drivaer_ml_volume.yaml`, which got a
  comment only. A one-batch smoke settles it, but it matters only if a baseline is ever rerun.
- **HiLift deduplication.** It changes the training cases and the scored set for every architecture, so HiLift's
  GeoTransolver and Transolver must be retrained and rescored. HL-DUP stage 2 is doing exactly this now (jobs
  367681/367682 GT and 367686/367687 T, 500 epochs, the standing's settings, about 60 epochs an hour; expected
  10-24 evening). If its lanes are accepted as the HiLift baseline standing, the rerun needs none. Otherwise it needs
  about 140 GPU-h (GT about 37 and T about 33 a lane, projected from HL-DUP's rate).
- **The whole boundary.** In option 0 no baseline reads a new piece:
  - SHIFT-SUV's box, HiLift's far field and the DrivAerML tunnel pieces reach ISLA only;
  - AF-WHOLE changes only the `airfrans_uvolume_2d` family, while the AirfRANS baselines read `airfrans_uvolume_ud`;
  - AeroZoo's GeoTransolver already reads all four pieces.
- **Stand:** DrivAerML Volume and Surface, SHIFT-SUV, AirfRANS, AeroZoo (14 GT and 2 T lanes), the lug, Tensile2d and
  DeepJEB++. **Retrain:** HiLift GT and T, which are in flight under HL-DUP. Option 0 adds no GPU-h to the rerun.
  Chapter 08 notes that the logs of some old lanes are gone (Tensile2d, the lug and DeepJEB++ smoke-priced rows); that
  does not need a retrain.

**Option 1: the whole boundary given to the baselines through config, with no code change.** These are the routes
BASELINE_PARITY found, none of them run:
- GeoTransolver: one merged geometry mesh, built with HarmonizeBoundaryFields, then MergeBoundaries and a subsample;
- Transolver: per-point distances to each piece, with ComputeSDFFromBoundary per piece and a wider `embedding_dim`;
- GLOBE: the meshes by name. GLOBE has no standing row.

This changes the baselines' inputs, so every affected lane retrains. From the measured lanes:

| dataset | lanes | GPU-h per lane (source) | GPU-h |
|---|---|---|---|
| DrivAerML Volume | GT, T × 2 | 65.4, 22.1 (clud_vol, sacct) | 175 |
| AirfRANS | GT, T × 2 | 25.9, 34.7 (afud, sacct) | 121 |
| DrivAerML Surface | GT, T × 2 | about 26 (EST from cost_090's 0.389 s), 20.0 (r80ud, sacct) | 92 |
| SHIFT-SUV | GT, T × 2 | 50.8, 39.5 (clud_suv, sacct) | 181 |
| HiLift | GT, T × 2 | about 37, 33 (HL-DUP rate) | 140 (replaces option 0's) |
| lug | T × 2 (its GT already reads the labels) | about 34 (EST from 0.093 s) | 68 |
| AeroZoo | T full × 2 (it does not see the inlet) | 154 (azc_full_t, sacct) | 308 |
| **total** | | | **about 1,090, plus an unmeasured 10-30 % for the larger inputs** |

Option 1 also needs a design, configs and smokes for each new input, and it changes the comparison protocol, so it is
Peter's call. It does not fit before 10-27 without starting that design now. Adding its 24 lanes (+20 % for the
larger inputs) to S2 lengthens the calendar from 3.8 to 5.0 days at 12 nodes, from 3.2 to 3.8 at 16, and from 5.7 to
7.6 at 8 (S1: 5.7 to 6.7 at 12, 4.3 to 5.1 at 16).

## 8. Do the dates hold?

The window 10-29 to 11-04 is about 6 days. I keep half a day for scoring and reduction, which leaves 5.5 days of
training. For the minimum node counts below, I took the smallest count whose makespan × 1.15 is 5.5 days or less:

| scenario | 8 nodes | 12 nodes | 16 nodes | fewest nodes for 5.5 days |
|---|---|---|---|---|
| S1 | **does not hold** (8.5 d, ends about 11-06) | **marginal** (5.7 d) | holds (4.3 d) | 14 |
| S2-pass2 | marginal (5.7 d) | holds (3.8 d) | holds (3.2 d) | 9 |
| S2 | holds, with half a day to spare (5.1 d) | holds (3.4 d) | holds (2.8 d) | 8 |
| S3 | holds (5.0 d) | holds (3.4 d) | holds (2.7 d) | 8 |
| S4 | holds (3.9 d) | holds (2.6 d) | holds (2.1 d) | 6 |

- S1 (5,390 GPU-h) does not fit the 2,500-4,000 GPU-h that Peter was told.
- S2, S3 and S2-pass2 (3,360-3,800) fit it.
- S4 (about 2,560) is at its low end.

**What would make S1 hold without weakening a registered protocol:**
- 14-16 nodes from 10-29;
- or merge DTT-FAST, which is S2. It is an exact speed change, certified by equivalence, so it does not change a
  protocol. Merges are the supervisor's.
- or start the AeroZoo ISLA-DTT lanes first, longest first. They set the calendar (89 h elapsed, about 103 h of
  calendar). This works only if their code can freeze before the rest; DTT-POSE's forest touches them.

**128 slices (S4)** keeps seeds, epochs and points, but changes the reference configuration. It needs the no-harm
check in section 8a and Peter's word.

### 8a. A no-harm check for 128 slices on every form (priced at the supervisor's request; not launched)

I followed SLICES-DRV's precedent: 128-slice arms on 0.9.0 code, read against the existing 0.9.0 512-slice lanes with
SLICES-DRV's rule, two seeds each. That needs no new control lanes. The cost factors come from section 3 and are EST.

| arm | control (existing, 0.9.0) | GPU-h per lane | elapsed per lane | pair |
|---|---|---|---|---|
| AirfRANS ISLA-DTT, compiled | r090fix_af_dtt (66.5 GPU-h, 16.5 h) | 40-53 (factor 0.6-0.8) | 10-13 h | 80-106 |
| lug ISLA-DTT, compiled (solid mechanics) | r090fix_lug_dtt (153-171 GPU-h, 38-43 h) | 138-154 (factor 0.85-0.95) | 33-40 h | 276-308 |
| DrivAerML Surface, interacting ISLA (body only at 0.9.0) | r090_dr_isla (16.4 GPU-h, 4 h) | 9-11 (factor 0.54-0.65) | 2.2-2.6 h | 18-22 |
| smokes, scoring | | | | about 5 |
| **total** | | | | **about 380-440 GPU-h**, 6 lanes on 6 nodes |

**Earliest reads, if it starts at 15:00 EDT today.** I took calendar as 1.15x elapsed, plus about an hour to score and
read:
- DrivAerML Surface: about 19:00-20:00 10-24.
- AirfRANS: about 05:00-08:00 10-25.
- The lug: about 10-26 06:00-14:00. That is before the 10-27 freeze, but has little margin. The lug is the long pole.

**Variants:**
- AirfRANS plus DrivAerML Surface only: about 100-130 GPU-h, read by about 08:00 10-25. Solid mechanics stays
  untested.
- Add AeroZoo full ISLA-DTT against azc_full_dtt, where 70 % of the rerun's cost lies: about +430 GPU-h, read about
  10-27 10:00 or later. That is too late for a 10-27 freeze unless the freeze waits for it.
- Run on the rerun's own code instead, the DTT-FAST head merged onto eba038f08, with new 512 controls. That needs both
  arms (about 640 GPU-h) and a merged snapshot that does not exist yet. It reads about 10-25 evening at the earliest.

**Caveats:**
- A check on 0.9.0 does not test the final code, as with SLICES-DRV.
- A PREREG and the supervisor's approval must come before 15:00, which is tight.
- The saving it buys: S2 to S4 is about 820 GPU-h and 1.2 days at 8 nodes. With the full check (about 400 GPU-h), the
  net saving is about 400 GPU-h and 1.2 days.

**Options that would change a registered protocol, named but not recommended:**
- fewer AeroZoo splits, or one seed;
- fewer epochs (AeroZoo 250, the rest 500, DeepJEB++ 40);
- fewer points (200k / 100k / 20k);
- Tensile2d with two seeds instead of four. That saves about 40 GPU-h, and the standing has four.

**A scope option that keeps every protocol but changes what "canonical" means:** keep the 0.9.0 AeroZoo ISLA-DTT lanes
(azc_*_dtt) as AeroZoo's standing instead of retraining them on the final code. That saves about 2,400 GPU-h in S2
and 4,036 in S1, and it shortens the calendar to the lug and DrivAerML lanes (1.4 days in S2 and 2.3 in S1, at 8 nodes). The
standing would then mix code versions. That is Peter's call, and I do not recommend it here.

## 9. Order of work (dates from the lanes' own plans; see FREEZE_DECISIONS.md)

- **10-24 to 10-26:** the reads that set the code land: BC-SUPPLY (tonight), OUTLET-EXP (10-25 morning), DTT-KNOBS
  (10-25 to 10-26), AF-WHOLE (10-25), DTT-POSE-EXP (AirfRANS 10-25 evening, the lug 10-26) and GEO-BAND (10-25).
- **10-27, the scope freeze:** merge 0.12.0 and tag the commit. Run the full ISLA test directory on CPU, and on a GB300
  with every CUDA test (one short job, 60 minutes).
- **10-27 evening to 10-28:** certification on every dataset, then smokes of every config. That is about 37 GPU-h and
  one calendar day; with way (a) of section 5, add about 35 GPU-h and half a day. If 0.12.0 carries DTT-POSE's forest,
  porting the fast build and walk to it must also land here; no lane owns that yet.
- **10-29:** launch all 32 lanes, longest first: AeroZoo, then the lug, DrivAerML Volume, DeepJEB++ and the rest.
  Stagger the submissions as LANE_RULES says.
- **When training ends:**
  - S2 at 8 nodes: about 11-03.
  - S2 at 12 nodes: about 11-01 to 11-02.
  - S1 at 16 nodes: about 11-02 to 11-03.
  - S4 at 8 nodes: about 11-02.
  Score each lane as it ends.
- **11-03 to 11-05:** reduce, write the notebook drafts and render the book. That leaves 4 days before the 11-09 code
  freeze.

## 10. Main uncertainties

1. DrivAerML Volume ISLA-DTT has never been trained. Its step (2.34 s in S1) is a harness number with a calibration
   factor, and its memory at S1 is 190.6 GiB a GPU (harness). That fits a GB300, but it is the largest of any lane.
2. The Tensile2d ISLA-DTT step is borrowed from the lug's ratio.
3. Compiled certification without real checkpoints: DrivAerML Volume ISLA-DTT, Tensile2d ISLA-DTT, and the surfaces
   with their new pieces.
4. Whether DTT-POSE's -12 % adds to DTT-FAST's gain. The two change the same forest, and S3 needs the fast build
   and walk ported to the new forest, or it is slower than S2 (section 2).
5. Node availability: the cluster showed 1 idle node today.
