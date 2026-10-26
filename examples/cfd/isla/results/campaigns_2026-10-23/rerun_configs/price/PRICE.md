# PRICE: the final rerun re-priced from wall time per iteration (child of RERUN-CONFIGS)

Status: DONE (written 01:45 EDT 10-26). Read-only: no jobs, no commits. Only light reads on lhr2: `parse_runs.py` piped
over ssh, and one `sacct` dump.

## Summary
- **Only one family has a measurement on code at or after 9f279664c: the lug.** No full training lane has run on such
  code. Every lane since 10-24 (AF-CAMPAIGN, DTT-POSE-EXP, SLICE-NORMAL-EXP, SLICES-ALL, HL-DUP) runs on 0.9.0-based
  snapshots (code_r090, code_cmpaf_fix, code_surfbc).
  - The lug is PR-P2 job 390719, arm A2. Code: `code_prp2_s3_isla`, whose SNAPSHOT_HEAD is 9f279664c plus #2173's
    muon.py, with the graph off (the default). It ran 100 steps of the recipe's real 4-GPU loop. Its `iter_time_s`
    median is **268.2 ms**, and 277.2 ms with cold caches (389560).
  - Every other step is EST. Each one comes from the nearest measurement on near-final code and is named in the
    table.
- **Totals.**
  - ISLA, 32 lanes: **3,160 GPU-h**. That is below FREEZE-PLAN's S2 (3,380) and S1 (5,390), and inside the
    2,500-4,000 GPU-h Peter was told.
  - The 20 baseline lanes that must retrain add **2,641 GPU-h**. Of that, AeroZoo GeoTransolver takes 2,185 (7
    splits x 2 seeds) and Transolver full 313.
  - **All: 5,801 GPU-h, which is above the 4,000 GPU-h ceiling Peter was told.** FREEZE-PLAN's option 0 had no
    baseline retrains. The seven-field AeroZoo decision, which retrains the baselines, accounts for the excess.
- **Calendar** (LPT x1.15, as FREEZE-PLAN computes it):

  | lanes | 8 nodes | 12 nodes | 16 nodes | fewest nodes for 5.5 days | longest lane |
  |---|---|---|---|---|---|
  | ISLA only | 4.7 d | 3.1 d | 2.5 d | 7 | 48 h (AeroZoo bank_zero_to_cad) |
  | ISLA and baselines | 8.5 d | 5.7 d | 4.7 d | 13 | 48 h |

## Method
- **The step is the wall time per iteration, never the logged host step.** For the old lanes, `parse_runs.py` reads
  `metrics.jsonl`. Every training step there carries a wall-clock `ts`. The script takes the differences between
  consecutive steps inside an epoch, leaves out epoch 0 and the first epoch of every link, and gives the median and
  the mean. It also gives epoch wall / steps (epochs without validation) and `epoch_seconds` / steps where logged; the
  three agree within 1-2 %. On the near-final runs, the step is the recipe's `iter_time_s`, which PERF-RECIPE checked
  against epoch seconds / steps (within 0.2 ms).
  - By median, the host step reads short by 2-5 % on most lanes (up to 9 % on jbn2), 9 % on HiLift GeoTransolver, and
    13-15 % on the HiLift ISLA and Transolver lanes. The HiLift lanes are loader-bound with 4 workers: their wall mean is 0.42 s against a median of
    0.37 s. Table: `price_tables.md`, last section.
- **Overhead factor.** For each lane: sacct allocated GPU-h over all links (`sacct_1026.psv`, dumped 01:36 EDT), divided
  by 4 x steps x that lane's own wall median. The factor is re-measured here because the old factors were taken
  against the host step. It ranges from 1.04 (AirfRANS) to 1.29 (DeepJEB++). Applied to the old lanes, it gives back
  their sacct totals.
- **Scoring.**
  - AeroZoo ISLA-DTT: 6.1 GPU-h a lane. The 258 GPU-h AZ-RESCORE spent splits into baselines 32.1 for 16 lanes, ISLA
    and ISLA-DTT 182.8 for 28 lanes, and E1/LRE 34.3. I split the ISLA part by the 0.75 / 0.86 s chunk ratio.
  - AeroZoo GeoTransolver and Transolver: 2.0 a lane.
  - The AeroZoo total for 14 + 14 + 2 lanes is **117 GPU-h**.
  - All other scoring, and certification and smokes (37 GPU-h), as FREEZE-PLAN.

## Per lane (per seed, 4 GPUs; full rows with sources in `price.tsv`)

| lane | seeds | steps x epochs | step [s] | how | overhead | GPU-h per seed | FREEZE S1 / S2 per seed |
|---|---|---|---|---|---|---|---|
| DrivAerML Volume, ISLA-DTT | 2 | 108 x 500 | 1.242 | EST-n: PERF-RECIPE 384246 arm B `iter_time_s`, 4-GPU real loop, e44015436+perf-recipe | 1.12 | 83 | 161 / 97 |
| AeroZoo full, ISLA-DTT | 2 | 678 x 250 | 0.763 | EST: PERF-SURVEY 382020 one GPU at b86c7391d, (716 + 11 loader) ms x1.05 rank imbalance | 1.14 | 164 | 308 / 182 |
| AeroZoo, the 6 held-out splits | 2 each | 529-760 x 250 | 0.705-0.782 | EST: full x the 0.9.0 lanes' split/full wall ratio | 1.10-1.18 | 124-190 | 232-357 / 137-211 |
| AirfRANS, far field, ISLA-DTT | 2 | 200 x 500 | 0.301 | EST-n: FUSED-PAD 384802 one GPU, 284 ms compiled (ee1917561), x1.06 (the lug's 4-GPU / 1-GPU) | 1.04 | 35 | 69 / 54 |
| lug, ISLA-DTT | 2 | 600 x 500 | 0.268 | **Measured** (short run): PR-P2 390719, 9f279664c | 1.13 | 101 | 189 / 107 |
| Tensile2d, ISLA-DTT, eager | 4 | 100 x 500 | 0.312 | EST-n: FUSED-PAD 384802 eager 294 ms x1.06; compiled 0.233 s beside | 1.06 | 18 (compiled 13) | 20 / 12 (compiled) |
| DrivAerML Surface, option A | 2 | 108 x 500 | 0.255 | EST: r090_dr_isla wall 0.247 s x1.03 tunnel; PERF-SURVEY 246 ms at 0.13.0 agrees | 1.10 | 17 | 17 / 17 |
| SHIFT-SUV, option A, CFD box | 2 | 198 x 500 | 0.255 | EST: r090_suv_isla wall 0.248 s x1.03 box (as FREEZE-PLAN) | 1.07 | 30 | 30 / 30 |
| HiLift, option A, far field + plane | 2 | 127 x 500 | 0.382 | EST: hldupx_hl_isla wall 0.367 s x1.04 | 1.27 | 34 | 32 / 32 |
| DeepJEB++, option A | 2 | 12,451 x 40 | 0.138 | EST: r090_jb_isla wall 0.138 s | 1.29 | 99 | 99 / 99 |
| HiLift GeoTransolver / Transolver | 2 / 2 | 127 x 500 | 0.471 / 0.365 | EST: hldupx_hl_{gt,t} walls | 1.15 / 1.25 | 38 / 32 | (option 0: 0) |
| AeroZoo GeoTransolver, 7 splits | 2 each | 529-760 x 250 | 0.768-0.776 | EST: azc_*_gt walls | 1.08-1.15 | 124-184 | (0) |
| AeroZoo Transolver, full | 2 | 678 x 250 | 0.734 | EST: azc_full_t walls | 1.12 | 154 | (0) |

"EST-n" marks an EST measured on near-final code that is not an ancestor of 9f279664c:
- ee1917561 lacks only INT64-OFFSETS (+1 ms) relative to 9f279664c;
- e44015436 + perf-recipe lacks FUSED-PAD (bitwise on 3-D models with a global direction) and INT64.

**What could still move the steps.** Code changes from 9f279664c to bf71190af:
- #2171 loads the next batch during the backward. It could hide part of DrivAerML's ~100 ms of exposed loader time and
  HiLift's loader stalls. On main's baselines it measured -0.8 % (GeoTransolver) and -5.6 % (Transolver).
- DRAW-EXACT adds +1-3 ms per draw.
- The library sampling and frame are a move.
- #2172 is a compile-cache fix.
- #2173 is off by default. If enabled, it is -7 % on the lug.

Not merged: GEO-PORT's G2 measured -4.5 % on DrivAerML (its lane is stopped on a decision).

The seven AeroZoo fields add 2 output channels and 2 loss terms to every AeroZoo model. EST under 0.5 % of a 0.7 s
step, so not priced.

**Variants.**
- Tensile2d compiled, if CERT certifies it: -18 GPU-h.
- AirfRANS and the lug eager, if certification fails: +64.
- AeroZoo ISLA-DTT rank imbalance x1.00 / x1.12 instead of x1.05: -102 / +143.
- HiLift with 12 workers: overhead 1.27 toward about 1.1, about -10.
- Not in the totals, as in FREEZE-PLAN: the SHIFT-SUV part B pair, about 62 GPU-h.

## How sure, and what is left uncertain
- The totals are about ±10 %. The ISLA total is 3,160, between about 2,950 and 3,450.
- **Biggest single uncertainty: AeroZoo ISLA-DTT** (2,149 GPU-h, 68 % of ISLA).
  - The step comes from a one-GPU replay at 0.13.0.
  - On DrivAerML, the 4-GPU real loop with the flat-buffer all-reduce matched the one-GPU real loop within 2 %.
  - The x1.05 for ranks holding cases of different sizes is EST and unmeasured.
  - A cross-version route through FREEZE-PLAN's 1.08 lane-to-harness ratio gives about 0.80 s instead of 0.76 s.
- **Torch version.** All measurements ran on torch 2.12 (`.venv-recipe`). RERUN-ENV builds `isla-rerun-t213`
  (torch 2.13) from bf71190af's uv.lock. If the rerun runs there, no step has been measured on it.
- **Steps never trained.**
  - DrivAerML Volume ISLA-DTT (146.5 GiB peak at 200k);
  - Tensile2d ISLA-DTT;
  - SHIFT-SUV's box and HiLift's far field and plane. Their token factors are EST; HL-SYMPLANE's smoke step measured
    data loading.
- **Scoring is not re-measured.** AeroZoo ISLA-DTT scoring (6.1 a lane) was measured at 0.9.0. The final code's
  faster read would likely lower it.

## For the main session
- The rerun's cost depends on the baseline retrains. ISLA alone fits 2,500-4,000 GPU-h and 5.5 days at 7 nodes. With
  the 20 baseline lanes it needs 5,800 GPU-h and 13 nodes for 5.5 days, or 8.5 days at 8 nodes. This is not a change
  of protocol, but it exceeds what Peter was told, so the supervisor may want to tell him.
- Suggested memory note: "Since 10-25, cost plans use wall time per iteration (`iter_time_s`, or the `ts` differences
  in metrics.jsonl), with overhead factors re-measured against it (PRICE, 10-26)."

## Files
- `price.tsv`: one row per lane with every source.
- `price_tables.md`: the generated tables, plus the wall/host table of 53 old lanes.
- `price.py` (`python3 price.py`).
- `parse_runs.py`, run as `ssh lhr2 "cd $T/runs && <venv-td14 python> - <runs>" < parse_runs.py`.
- `runs_a.tsv` and `runs_az.tsv`: its outputs.
- `sacct_1026.psv`, from `sacct -X -u psharpe -S 2026-10-18T00:00 -P -n -o JobID,JobName%80,State,ElapsedRaw,AllocTRES%80,Start,End,QOS`.

Nothing on disk over 1 GiB, and nothing written on LHR.
