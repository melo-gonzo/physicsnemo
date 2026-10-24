# NEARWALL contingency: the viscous softening asinh(d U / nu) under the whole boundary (pre-registration draft)

Lane NEARWALL (research-lane, supervisor 01a13c2c). Drafted 2026-10-24 about 09:20 EDT, at the supervisor's approval
of DESIGN.md (md5 1ff494b6, 08:43 EDT). **Not submitted, and not to be submitted unless the trigger of section 1
fires and the supervisor approves the submission and sets the cap.** Adoption of (ii) is Peter's whatever this reads:
it is an architecture change, and it makes the encoding depend on declared physics.

## 0. What was seen before drafting (disclosure)

- NEARWALL's own phase A: `RESULTS.md` (md5 43888a10), `DESIGN.md`, `REVIEW_DESIGN.md`, `GEOMETRY_NOTE.md`, the
  tables, and `out/`. These are probes only; nothing was trained or scored.
- AF-WHOLE's PREREG.md (with amendment), FACTS.md and JOBS.md, up to its line "09:02 EDT 10-24: all four lanes on
  their second links". **No AF-WHOLE or FLOOR-DRV score, validation curve or band reading.** I will open none before
  this file is frozen (supervisor, 08:43: do not open their results).
- The standings' published scores: AirfRANS (B-ISLA, in AF-WHOLE's PREREG section 6) and AeroZoo `full`
  (`examples/cfd/isla/results/char/aerozoo_090.json`).

## 1. Trigger and precondition

**Trigger.** AF-WHOLE's registered R2 near-wall signature, read on both seeds of W-ISLA against B-ISLA: for velocity or
nut, the W / B ratio of the pooled band e_R below 3.05e-4 chords is larger than that above 3.05e-4 chords on seed 42
and on seed 43 (AF-WHOLE P9's reading, per seed). This must hold on top of an R1 "harms" label for the same field.
- The trigger is read and reported by AF-WHOLE or the supervisor, not by me. If it does not fire, this file stays a
  draft and nothing runs.

**Precondition (before any training job).** The geometry kernel's float32 closest-point arithmetic, measured far from
the origin, must be on record: `GEOMETRY_NOTE.md` (handed to the geometry follow-up lane), or a rerun of
`harness/nwprec2.py` on the arm's own snapshot.
- Gate: the (ii) channel's compute rounding on AirfRANS with its far field stays at p99.9 <= 0.1 in the scored frames
  (native and posed; measured 0.0018 and 0.0098). If not, stop and report.
- Reported beside, not gating: the same at T = 0.1 L and T = L (measured 3.1 and 5.9 p99.9 with the current kernel).
  The frame-free precision trade is part of what goes to Peter.

## 2. The encoding under test, written out per dataset

(ii) is two codes, chosen by what the model declares:
- where the model declares a viscosity role and a velocity scale (in ISLA these come together with `reynolds`, which
  needs both): the second distance channel is asinh(s Re_L) = asinh(d U / nu), with Re_L = U L / nu from
  `_case_scales` (float64), the same U, L and nu as the Reynolds input z;
- everywhere else: the current asinh(s / 1e-6), as (i).

The first channel s = d / L is unchanged, as are the input width, parameters and checkpoint layout.

| dataset (interior task) | declares | channel under (ii) | note |
|---|---|---|---|
| AirfRANS (+ far field) | `nu` (role viscosity, set by the model config's SetGlobalField 1.56e-5 m^2/s; not in the case files), `U_inf` (velocity_scale), `reynolds` from the dataset file | asinh(d U / nu): first nodes at U d / nu of about 5 (asinh 2.3), the same with or without the far field | nu is a recipe constant already approved for the Reynolds input; no new constant is supplied |
| AeroZoo | `physics.kinematic_viscosity` (stored per case), inlet `bc_velocity` (velocity_scale), `reynolds` | asinh(d U / nu): eps moves from 1e-6 L to between 4.9e-9 and 1e-4 L, so **every case changes** | stage 2 |
| DrivAerML Volume (five pieces) | no viscosity role (the files' nu = 1.0 is a placeholder; supplying air's value is Peter's decision, not taken) | (i), unchanged | an arm is **degenerate**: the function is identical, so FLOOR-DRV has no (ii) arm |
| lug, Tensile2d | no viscosity or velocity (elasticity) | (i), unchanged | degenerate |
| surface tasks (DrivAerML Surface, SHIFT-SUV, HiLift, DeepJEB++) | - | no interior query channel | not affected |

**Snapshot.** `$T/code_nearwall_ii` = a copy of `$T/code_cmpaf_fix` (AF-WHOLE's code, model.py md5 45b24fb8). Only
model.py changes:
- `_case_scales` keeps `Re_L = U L / nu` beside z;
- `forward` passes it to `_forward_tokens`;
- the builtin line (code_cmpaf_fix model.py:1874) becomes `asinh(s * Re_L)` where Re_L is given, else
  `soft_log_length(s)`;
- `_forward_tensors` passes none, so it keeps (i).

It is built in a worktree on psharpe/isla-nearwall-ii, verified by md5 and recorded in SNAPSHOT_HEAD. Before
submission it must pass:
- the whole ISLA test directory on LHR cpu-short;
- two new tests: the channel's invariance to a consistent rescale of length and viscosity, and (i) where no viscosity
  is declared, checked by an assert in the smoke;
- a CPU forward on 2 real AirfRANS cases, checking that the seed's last column equals asinh(d U / nu) to 1e-6
  relative.

Recipe: `$T/recipe_afwhole`, unchanged.

## 3. Arms

**Stage 1 (AirfRANS with its far field, ISLA):**
- **V (new):** AF-WHOLE's W-ISLA lanes.tsv rows 0-1 (`isla_volume_airfrans`, `airfrans_uvolume_2d`, the same extra, lr,
  500 epochs, 100,000 points, bfloat16, seeds 42 and 43). Code `code_nearwall_ii` replaces `code_cmpaf_fix`; nothing
  else differs. Launcher `$T/kernel/nearwall_ii/`, with byte copies of kernel/afwhole's lanes.sbatch,
  ckpt_preflight.py and score.sbatch, and site.env with CODE changed. Job names `isla.nearwall.af-ii-seed4x`, the five
  bad nodes excluded.
- **Control:** AF-WHOLE's W-ISLA, `afwhole_af_isla_seed42/43` (run ids from AF-WHOLE's JOBS.md), its own scores. No new
  control runs.
- **Beside:** B-ISLA (`r090eag_af_isla_seed42/43`, airfoil only), to read how much of W's gap (ii) closes.
- ISLA-DTT is not an arm. Its tree read keeps an unsoftened log(r / |d_q|), and its query channel is the same one. The
  supervisor may add a W-DTT pair (about 220-280 GPU-h).

**Stage 2 (AeroZoo `full` split, ISLA), only if stage 1 reads "helps" on any field and no field "harms":**
- **V2 (new):** `azc_full_isla` (code_r090, recipe_r090, `isla_volume_aerozoo`, `aerozoo_uvolume`, train full.train,
  scored on the full split's test side, 250 epochs, 200,000 points), seeds 42 and 43. The code is `code_r090_ii`, the
  same one-function change on code_r090.
- **Control:** `azc_full_isla_seed42/43` (per seed: velocity 0.120565 / 0.115665, pressure 0.311146 / 0.296336,
  nut 0.411742 / 0.388146, tke 1.033045 / 0.965195, omega 0.370631 / 0.374114; 716 cases).
- The other six held-out splits are not run. Generalization across families is AZ-DECIDE's instrument; an adoption
  that needs it is priced in section 7.

## 4. Smoke, scoring, checks

- **Smoke** (one short-QoS 4-GPU job, 40 min, `isla.nearwall.smoke`): V seed 42 and W-ISLA seed 42 (AF-WHOLE's
  afwhole launcher row 0, NO_CHAIN, written to a smoke directory, never to AF-WHOLE's runs), 50 steps each on one node.
  - Passes on rc 0, finite losses, and a resolved config equal to W's except `run_id` and the code path.
  - The seed-column check: the last column of the qt.seed input equals asinh(d U / nu) on V, and soft_log_length(s) on
    W.
  - Cost: V / W median step and peak memory.
- **Scoring exactly as AF-WHOLE:** kernel/r090eagaf/score.sbatch unchanged, native, area (checked equal to native,
  then labelled degenerate) and posed, float32, full_test (200 cases), the epoch-500 checkpoint.
  - Near-wall dumps with SLICES-AF's nearwall.sbatch, reduced with AF-WHOLE's R2 decades and the 3.05e-4 chord
    split.
  - Checks before reading: `"epoch": 500`, 200 rows with W's sample ids, SNAPSHOT_USED `code_nearwall_ii`, the
    freestream guard (AF-WHOLE amendment 1), and the same scored points as W (coordinate hash).
  - A failed check makes the row invalid, and its rule "incomplete as registered".
- **Event guard** as AF-WHOLE's: validation loss above 4x its running minimum after epoch 10, a non-finite loss, BLOCKED,
  or a hung link. Then stop and tell the supervisor; never resubmit.

## 5. Rules (noise-aware, as DTT-POSE-EXP's PREREG section 6 and amendment A1.3)

CM is the scorer's case-mean relative L2, native. AirfRANS fields: pressure, in-plane velocity (never velocity_z), nut.
AeroZoo fields: velocity, pressure, nut, tke, omega. Slots: 2 V runs + 2 control runs per stage. Posed and the three
fields are readings of the same runs.

**R1 per field (V against the control):**
- `D = ln(mean_seeds CM(V) / mean_seeds CM(control))`, `SE = sqrt((s_V^2 + s_c^2) / 2)`, where s is a pair's two-seed
  sample sd (n - 1) of ln CM.
- **Helps** if D < ln 0.90 and |D| > 2 SE; **harms** if D > ln 1.10 and |D| > 2 SE; otherwise **no effect**.
- s_c on AirfRANS is W-ISLA's own, read when its scores exist. For reference, B-ISLA's is 0.0777 / 0.0190 / 0.0355
  (p / v / nut).
- AeroZoo `full` controls' s_c: velocity 0.0293, pressure 0.0345, nut 0.0417, tke 0.0480, omega 0.0066. So the ln 1.10
  bar binds on every field unless s_V is far larger.
- **Sub-bar flag:** up if V / control > 1.03 with both V seeds above both control seeds; down if < 0.97 with both
  below. It changes no label.
- **Paired reading (registered).** The change adds no parameters, so a seed gives the same initialization and data order
  in V and its control. With r_s = CM(V, s) / CM(control, s):
  - helps (paired) if r_42 < 1/1.10 and r_43 < 1/1.10;
  - harms (paired) if r_42 > 1.10 and r_43 > 1.10;
  - otherwise no paired label.
  - A harms (paired) counts as harms.

**R2 near wall (AirfRANS, descriptive labels):** per field and AF-WHOLE's decades, and for the split below / above
3.05e-4 chords, the R1 rule applied to each run's pooled band e_R ("band helps / harms / no effect").
- Beside it, the share of W's band gap to B that V closes: `1 - ln(e_R(V) / e_R(B)) / ln(e_R(W) / e_R(B))`, per seed
  and pooled.

**R3 pose (AirfRANS):** posed / native of CM per field and seed lies within 0.002 of one, or at the standing's level
(AF-WHOLE R3's wording). Otherwise "pose broken", reported as a defect.

**R4 cost:** V / control smoke median step and peak memory; the lanes' GPU-h.

**What a reading leads to (none is a decision of mine):**
- Stage 1 "helps" on a field with no field "harms": stage 2 goes to the supervisor for approval.
- Stages 1 and 2 with no "harms": the supervisor takes (ii) to Peter, with the numbers, the two-code rule, the
  recipe-set nu on AirfRANS and the precision trade far from the origin.
- Any "harms": (ii) is not proposed. The near-wall harm then needs another cause search (the residual suspects in
  AF-WHOLE section 7).
- Lost or invalid arm or row: "incomplete as registered", escalated, no improvised verdict.

## 6. Predictions (mine, written before any AF-WHOLE or FLOOR-DRV score, conditional on the trigger having fired)

| id | prediction | p |
|---|---|---|
| C1 | Smoke passes on its first run | 0.85 |
| C2 | Stage 1 velocity: helps / no effect / harms | 0.30 / 0.62 / 0.08 |
| C3 | Stage 1 nut: helps / no effect / harms | 0.35 / 0.57 / 0.08 |
| C4 | Stage 1 pressure: helps / no effect / harms | 0.12 / 0.78 / 0.10 |
| C5 | R2: V / W of velocity or nut below 3.05e-4 chords < 0.90 on both seeds | 0.55 |
| C6 | R2: V closes at least half of W's gap to B below 3.05e-4 chords (pooled, velocity) | 0.40 |
| C7 | R3: pose holds or holds at the standing's level | 0.85 |
| C8 | R4: V / W median step within [0.98, 1.03] | 0.90 |
| C9 | Stage 2 (if run), any AeroZoo field harms | 0.20 |
| C10 | Stage 2 (if run), any AeroZoo field helps | 0.20 |

Why:
- (ii) restores the standing's log code for the 19.5 % of points that the far field moved into the linear part. If the
  near-wall harm is the encoding, V should recover most of it on velocity and nut, where the band sits.
- Pressure is set mostly 1e-3 to 0.1 chords out (CAP-AUDIT), in the log part either way, so it moves little.
- Seed spread (1-16 %) and the possibility that W's harm has another cause (the far field's 99.86 % share of the
  measure-weighted reads, the Reynolds input shift) keep "helps" well short of even.
- On AeroZoo every case's channel changes, over Re_L from 1e4 to 2e8. The trained standing already resolves its points
  in the log part (A1: 0.01 % linear), so a gain is unlikely, and a loss is possible where nu / U falls below float32's
  floor on the high-Re cases (A3: up to 3.3 already in the data frame).

## 7. Cost, cap and schedule

- **Stage 1:**
  - V pair: the standing's 89 GPU-h per lane at 1.0-1.25x for the far field (AF-WHOLE's projection), so 180-225 GPU-h.
  - Smoke at most 2.7 GPU-h; scoring about 1 GPU-h (6 rows); near-wall dumps about 1 GPU-h.
  - Snapshot tests: 2 cpu-short jobs. Total **about 185-230 GPU-h**.
- **Stage 2:** azc_full_isla took 115 train GPU-h and 122 allocated per seed (AZ-DECIDE tables_partial.md), so the pair
  is about **245-250 GPU-h**, plus scoring about 2 GPU-h. All seven held-out splits would be about 1,600 GPU-h (not
  proposed).
- **Schedule:**
  - AF-WHOLE's W-ISLA reads about 10-25 03:00-08:00 EDT.
  - If the trigger fires and the submission is approved that morning, stage 1 ends about 10-26 08:00-14:00 EDT (22-28
    h of wall plus queue gaps), and its scores an hour later, before the scope freeze (about 10-27).
  - Stage 2 would end about 10-27 midday, at the freeze.
  - A record branch psharpe/isla-nearwall-ii is made only when the submission is approved (`git commit -s`, hooks on,
    plain push, never merged).
- **Registry row** `^isla\.nearwall\.` re-added at the first job, expected_end an ISO date, and removed after the last.
