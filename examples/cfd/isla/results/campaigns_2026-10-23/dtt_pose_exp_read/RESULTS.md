# DTT-POSE-EXP results (read 2026-10-26 06:36 EDT)

Registered read: PREREG.md (md5 2f46a10c, with amendment A1). Reducer `lhr/dttposex/read_x.py` (10983a33, written before
any arm result); its output is `results/read/read_all.md` and `.json`. Snapshot checks and smokes: CHECKS.md. Jobs: JOBS.md.

- Arms: dttposex_lug_dtt_seed42/43 (fix-trained, compile_blocks) and dttposex_af_dtt_seed42/43 (eager) on `$T/code_dttposex`
  (code_cmpaf_fix + DTT-POSE's 0.9.0 port: tree_read 00d83cb9, model fc8d0fa9).
- Controls: r090fix_lug_dtt and r090eag_af_dtt on code_cmpaf_fix.
- Scorers: 401023 (lug) and 383802 (AirfRANS). They run float32, eager, at the final checkpoint, native and posed.
- Checks before reading, all passed:
  - every row is at epoch 500, with 600 lug cases and 200 AirfRANS cases;
  - the case ids equal the control's;
  - SNAPSHOT_USED, DATASET_USED and the extra are as the lane's;
  - SCOREPROOF, from inside each scoring process: tree_read md5 00d83cb9, every forest built with an origin, 6 classes
    on the lug and 4 on AirfRANS.

## Verdicts

| rule | dataset / field | arm s42 / s43 | control s42 / s43 | arm / control | D | 2 SE | seed-matched | verdict |
|---|---|---|---|---|---|---|---|---|
| N | lug von Mises | 0.002410 / 0.002103 | 0.002219 / 0.001834 | 1.113 | +0.107 | 0.234 | 1.086 / 1.146 | **no change**; flag none (the seeds overlap) |
| N, paired (A1.3) | lug von Mises | | | | | | 1.086 / 1.146 | **no paired label** (cost needs both > 1.10) |
| N | AirfRANS pressure | 0.012802 / 0.013464 | 0.012428 / 0.012389 | 1.058 | +0.057 | 0.051 | 1.030 / 1.087 | **no change**; **flag up** |
| N | AirfRANS velocity | 0.007368 / 0.007205 | 0.007373 / 0.006973 | 1.016 | +0.016 | 0.060 | 0.999 / 1.033 | **no change**; flag none |
| N | AirfRANS nut | 0.022876 / 0.021351 | 0.020711 / 0.020484 | 1.074 | +0.071 | 0.070 | 1.105 / 1.042 | **no change**; **flag up** |
| P | lug posed / native | 1.0001 / 1.0014 | (control 1.528 / 1.977) | | | | | **pass** (bound [1/1.02, 1.02]) |
| Q | bf16 forest = float32 forest | lug 600 / 600 bitwise on native and on posed, for each seed; AirfRANS 200 / 200 likewise | | | | | | **pass** |

**Adoption rule (A1):** no field of either dataset reads cost under N, the lug reads no cost (paired), and P and Q pass, so
**the registered condition to adopt is met.**

## What it means, beside the verdicts

- **The pose defect is gone in trained models.** The lug's posed / native ratio was 1.53 / 1.98 for the control and is
  1.0001 / 1.0014 for the arm. That is at the floor of an equivariant model, as TIE-EXP found (1.0024 / 1.0015), even on
  0.9.0's geometry. The arm's posed error is 0.643 of the control's.
- **The bf16 training forest is the scored one.** Q holds on every case, and in training on 3 of 3 smoke forwards.
- **On the lug there is a probable native loss of about 10 %, which the registered rule cannot resolve.**
  - Both seed-matched ratios are above 1: 1.086 and 1.146.
  - The arm beats the control in only 19.2 % of the 600 cases (seed means). The case median is 0.002069 against
    0.001866 (1.109).
  - The two-seed spread within each arm is 0.10-0.13 in log, so N's bar is 2 SE = 0.234 (a ratio of about 1.26).
    The paired rule's 1.10 threshold is missed by one seed (1.086).
  - Unlike TIE-EXP, where the pairs landed within 1 %, here the change alters the function at every step. The
    seed-matched ratios therefore carry the effect plus a trajectory divergence of unknown size.
  - Arm / interacting ISLA is 1.552 (seed-matched 1.648 / 1.455), against the control's 1.394 (1.517 / 1.270).
- **On AirfRANS there is a seed-resolved loss of about 6-7 % on pressure and nut, under the 10 % bar.** Velocity is level.
  - Arm / interacting ISLA: 1.052 / 1.146 / 1.440, against the control's 0.994 / 1.129 / 1.342.
  - Case medians favour the arm on pressure (0.00908 against 0.00959) and are level on nut. The case-mean excess sits
    in fewer cases (arm better in 50.5 % of cases on pressure and 46.5 % on nut).
- **DK-GAP regions** (`results/regions/regions.md`; gate passed: geometry equal to DK-GAP's, recompute 2.3e-7):
  - wake in-plane velocity: arm / control 0.999 [1.016, 0.982]; the gap to ISLA is 1.727 against 1.729, unchanged;
  - nut beyond one chord: 1.067 [1.064, 1.070] (gap 1.455 against 1.363);
  - near-wall nut: 1.207 [1.372, 1.050];
  - wake nut: 1.120 [1.152, 1.086].
  So the nut loss sits near the wall and in the wake. The far-field velocity gap is untouched.
- **Pairs per point** (float32, every scored case):
  - lug 114.9 against the control's 118.4 (-2.9 %);
  - AirfRANS 21.03 against 21.64 (-2.8 %).
- **Step time** (smoke): lug 0.420 against 0.425 s; AirfRANS 0.934 against 0.946 s.
- **Lane cost from sacct:** lug pair 290 GPU-h (control 324), AirfRANS pair 215 (control 221).

Why it might cost: the face classes regroup cells into 6 trees (lug) and 3 trees (AirfRANS) instead of 8 and 4, and the
banded box test opens more ties. Each point therefore reads somewhat coarser clusters (3 % fewer pairs). At theta 1 that
is a slightly cruder far read, consistent with DTT-KNOBS stage 1, where every coarsening of the read cost accuracy.
This is a hypothesis; it was not tested here.

## Predictions (PREREG section 7)

- S1-S6: (a) passed (with the frame note); (b) failed its bound and passed through A1.2's fallback (S2's 0.35 branch);
  (c), (d), G and the smokes passed.
- N1: lug no change, as predicted (0.85).
- N2: AirfRANS no change on all three fields (0.85).
- N3: no cost anywhere (0.90).
- F1: an up flag fired (p 0.30) and no down flag.
- P1 passed (0.85); Q1 passed (0.95); A1 adoption condition met (0.75).
- B1 (lug arm below the control) failed (p 0.55).
- B2: lug pairs 2.9 % below, just outside the predicted 3-8 % band (p 0.70).
- B3: step times within 5 % (0.70).

## Cost

About 508 GPU-h in all (sacct), under the 650 cap:

| item | GPU-h |
|---|---|
| lanes (38 links) | 504.9 |
| checks and smokes | 1.2 |
| scoring | 0.7 |
| Q probes | 0.5 |
| regions | 0.4 |

Two lug links were lost to the NFS Inductor cache race (0.9 GPU-h).

## For the book (draft; the supervisor applies it)

The change goes only once adopted: chapter 04's invariance section and the lug ISLA-DTT standing would replace the posed /
native 1.528 / 1.977 with 1.0001 / 1.0014 (this file). The native lug ISLA-DTT standing would move from 0.002027 to
0.002256 (gap to ISLA 1.394 -> 1.552), and AirfRANS ISLA-DTT from 0.012408 / 0.007173 / 0.020598 to 0.013133 / 0.007286 /
0.022114. Each number needs an "In flight" callout if the supervisor takes the lug signal to Peter first.
