# E1-AZ result: HEADNORM-090 against the 0.9.0 CONTROL on compliant AeroZoo family_non_box (2026-10-22)

Registration: PREREG_AZ.md (md5 5f18f7e0b0d9ba104c5ffe2ef0b81c90), which completes E1's rule (PREREG.md) with this pair.
Read by lane AZ-DECIDE; verdict confirmed by the supervisor at 08:00 EDT 2026-10-22.

## Verdict, as registered: **REJECT**

- **Reject branch 1 fires on pressure.** The seed-mean ratio is 1.287, and both seed-matched ratios (1.339 and 1.178) are
  each ≥ 1.10. Velocity, 1.047 (seed-matched 1.022 / 1.081), does not fire it.
- **Reject branch 2 does not fire:** G14 = 1.0037, below 1.05.
- (a') fails on pressure (a seed-mean ratio ≥ 1.10). (b1)-(b3) hold on AeroZoo, as on AirfRANS and DeepJEB++.
- **Consequence (registered):** HEAD-NORM is not adopted into the next reference, and ISLA 0.9.0 stays the reference.
  psharpe/isla-headnorm-090 (fd2a91179) stays pushed and unmerged, as a candidate. Only Peter can override this.

| field | CONTROL seed mean (42 / 43) | HEADNORM seed mean (42 / 43) | ratio | seed-matched 42 / 43 | |
|---|---|---|---|---|---|
| velocity (headline) | 0.53322 (0.61284 / 0.45360) | 0.55844 (0.62658 / 0.49030) | 1.0473 | 1.022 / 1.081 | within seed spread |
| pressure (headline) | 2.7742 (3.7600 / 1.7884) | 3.5701 (5.0330 / 2.1072) | **1.2869** | **1.339 / 1.178** | reject clause; seed ranges overlap |
| nut (recorded) | 0.90230 | 0.92957 | 1.0302 | 0.997 / 1.066 | within seed spread |
| tke (recorded) | 1.2639 | 1.3167 | 1.0418 | 0.923 / 1.171 | within seed spread |
| omega (recorded) | 1.3150 | 1.3408 | 1.0196 | 0.829 / 1.374 | |

AeroZoo instrument: (b1) no HEADNORM collapse (min support 3.44e5 of 4.0e5 tokens, max weight 1e-5): holds. (b2) HEADNORM
final seed-logit p99.9 0.595 (0.550 / 0.640), against CONTROL 25.0 (15.76 / 34.26), ratio 0.024: holds. (b3) CONTROL growth
33.1 and 103.9: holds.

**Checks.** Test side of family_non_box, 1,074 cases in all four score files; `config` epoch 250 in all four;
SNAPSHOT_USED code_hn090 (HEADNORM) and code_r090 (CONTROL); ISLA.0.250.mdlus present in both HEADNORM runs. Probe logs:
`probing [1, 9, 29, 49, 99, 149, 249, 250]`, ISLA 0.10.0 / 0.9.0, HFRAME-PROBE-DONE in both units.

**Evidence.**
- Reducer: `scratch/isla-az-decide-2026-10-21/reduce_az.py`, md5 43774c8f, a byte copy of `reduce_az.py` beside this file.
- Its output: `scratch/isla-az-decide-2026-10-21/RESULTS_E1AZ_raw.txt`, md5 645d7cbd.
- Jobs (LHR):
  - Training: 306216 / 306217 plus chain links.
  - Scorer 329031: a fixed copy of the registered scorer, `$T/kernel/azdecide/eval_hn` (it differs only in a writable Hydra
    cwd, per-unit caches, QoS and time limit, because the registered copy `cd`s into the read-only recipe snapshot, where
    Hydra cannot start). Scored into `$T/p200k_fp32_test_hn090az`.
  - The CONTROL scores: 321552, through `eval_azi`.
  - Probe 329032: the registered `hframe_probe.sbatch`.
  - Norm-gain check 331663 (2d below).

## Beside the verdict (post hoc; not part of the rule)

**a. No skill on the deciding field.** Both arms do worse than predicting zero on nonbox pressure: physical relative L2 2.77
(CONTROL) and 3.57 (HEADNORM). The failure is broad across cases, not carried by a few. CONTROL's per-case median is 1.51
and its p90 6.3, and the 5 worst cases carry 4 % of the mean (`scratch/isla-az-decide-2026-10-21/percase_partial.md`).
GeoTransolver reads 1.17 there, with a per-case median of exactly 1.00, i.e. it predicts about zero pressure. So the reject
rests on a field where ISLA has no skill. Velocity, where ISLA has skill (0.53), reads 1.047, within seed spread.

**b. Seeds.** Within each arm the two seeds differ by about 2× in pressure (CONTROL 3.76 / 1.79; HEADNORM 5.03 / 2.11).
But seed 42 is the worse seed in both arms, so seed pairing removes most of that spread, and both paired ratios still
exceed 1.10. The registered both-seeds clause therefore fires on a consistent effect, even though the arms' unpaired seed
ranges overlap.

**c. Head-frame growth does not cause the nonbox pressure failure.** HEAD-NORM removes the growth on AeroZoo (final p99.9
0.55 / 0.64 against 15.8 / 34.3), and nonbox pressure gets worse, not better. This is passed to the per-case diagnosis.

**d. HEAD-NORM's head frame ends near-uniform on AeroZoo** (from the probe JSONs; a finding, not a rule outcome).
- At epoch 250, HEADNORM's minimum seed support is 3.44e5 and 3.46e5 points of 4.0e5 tokens. Its support fraction is 0.93
  / 0.90, the same as at epoch 1 (0.92). Its seed-logit spread is 0.076 / 0.096 and its largest weight 1e-5.
- CONTROL's minimum support is 2.3e4 / 4.7e3 at epoch 250 (1.9e4 / 3.3e3 at its sharpest). Its support fraction is 0.32 /
  0.50 and its logit spread 1.73 / 2.46.
- So CONTROL routes sharply only because its raw stream grew (mean token norm 208 → 3331, 186 → 3010). With the stream
  normalized, the frame barely learns to route.
- Instrument (b) checks only the collapse side, not this uniform side.
- **Correction to the expected mechanism.** HEADNORM's frame input fell from 22.48 at epoch 1 to 5.36-5.90 from epoch 99 on.
  That is not norm_out's gain shrinking: the gain's RMS stays at 0.94-0.95 (0.99 at epoch 1; job 331663), so 22.63 × RMS
  gain is 21.3-21.6.
- ‖γ ⊙ x̂ + β‖ falls below √d · RMS(γ) only when the normalized token concentrates its energy in channels with small gain.
  So the likely cause is a few outlier channels that dominate the grown stream and carry a small gain. That is an
  inference; the activations were not inspected.
- CONTROL's norm_out gain is the same (0.95-0.96), but CONTROL's frame does not read norm_out.

**e. G14 (non-binding).** As registered, 1.0037. That uses the 6 AirfRANS ratios from arms that both trained and were
scored under ISLA's compiled per-block path (COMPILE-AF).
- With E1's eager AirfRANS re-scores (native 0.978 / 1.009 / 0.957, posed 0.978 / 1.009 / 0.957,
  `scratch/isla-compile-af-2026-10-21/RESULTS.md` lines 139-140), G14 = 1.0030. Both arms were still trained on the
  compiled path.
- The 12 known slots carry about 5 independent numbers: AirfRANS native equals posed, and DeepJEB++ native equals area
  equals posed. G14 ≤ 1.01 holds whenever r_v · r_p ≤ 1.471. Over the 7 independent slots (eager AF 3, DeepJEB++ 2,
  AeroZoo 2), the geometric mean is 1.028.

**f. Registered predictions (PREREG_AZ §7) against the outcome.**

| # | prediction | p | outcome | |
|---|---|---|---|---|
| A1 | velocity ratio in [0.95, 1.05] | 0.70 | 1.047 | holds (narrowly) |
| A2 | pressure ratio in [0.90, 1.10] | 0.70 | 1.287 | fails |
| A3 | neither reject clause fires | 0.92 | branch 1 fires | fails |
| A4 | G14 ≤ 1.01 | 0.95 | 1.0037 | holds |
| A5 | (a') holds | 0.82 | pressure ≥ 1.10 | fails |
| I1 | HEADNORM's frame input stays at 22.6 × RMS(norm_out gain) and never follows the stream | 0.95 | 5.4-5.9 against 21.3-21.6; the stream grows to 2.6e3-3.9e3 while the input stays bounded | fails as written (the "never follows the stream" half holds) |
| I2 | HEADNORM final p99.9 in [0.2, 5], no collapse | 0.85 | 0.55 / 0.64, no collapse | holds |
| I3 | CONTROL final p99.9 in [8, 80] | 0.80 | 15.8 / 34.3 | holds |
| I4 | (b2) holds | 0.93 | 0.024× | holds |
| I5 | (b3) holds | 0.99 | growth 33 / 104 | holds |
| V | ADOPT 0.78, PETER DECIDES 0.18, REJECT 0.04 | – | REJECT | the 0.04 outcome |

**Instrument validity.** On 0.9.0 the hooks-on and hooks-off forwards of the first case differ by at most 2.2e-3 m/s in
velocity (7e-6 in pressure); on 0.10.0 they differ by 0.
- STREAM-AUDIT (scratch/isla-stream-audit-2026-10-21/REPORT.md) measured two bf16 forwards without hooks on the same
  AeroZoo cases: they differ by 1.83e-3, against 1.85e-3 hooks on/off. So the gap is GPU nondeterminism, and the hooks do
  not perturb the forward.
- STREAM-AUDIT also found that the 0.9.0 CONTROL's head frame collapses to an argmax on DeepJEB++ from epoch 13 (seed_maxw
  > 0.5 at epochs 31 and 35; p99.9 from about 4 to 2.3-3.0e3). So the defect E1 fixes is present in the 0.9.0 reference.
- On AeroZoo at 0.9.0 the stream grows 17× (non_box) to 29× (full).
