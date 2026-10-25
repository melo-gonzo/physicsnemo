# OUTLET-EXP results: the outlet pressure gauge by construction, trained (2026-10-25)

Written 2026-10-25 04:35 EDT by research-lane OUTLET-EXP. Registration: `PREREG.md` (md5 69283aa5824dd3155ff659afe15775a3;
sections 4 and 8, the 10-23 erratum). Tables: `tables.md` (md5 1bdbde00, from `reduce_outlet.py` a3ab1635). Jobs and
evidence: `JOBS.md`. Every rule below is read as registered.

## Verdicts as registered

- **M1 (event guard): pass.** All 2,588 scored B cases (2 x 578 crossflow, 2 x 716 full) have a mechanism row, and on every
  one the measure-weighted mean of B's shifted pressure over the probes equals the prescribed 0 Pa mean: largest
  |residual| / scale 1.4e-6 (tolerance 1e-5), at most 1.1e-6 of the case's RMS p; the shift is one constant; the probe set
  is never empty. **B is valid.**
- **PREREG section 8 declaration checks: pass.** B's four training resolved configs and four scorer Hydra configs show
  `dirichlet: pressure`; A's four and four show none; every B mechanism unit printed the declared level before reading a
  case, and every A unit printed none. SNAPSHOT_USED code_outlet on all B rows. Epoch 250, test side, 578 / 716 cases, the
  same sample ids as A's.
- **Event guard (validation loss > 2 x running minimum): not hit.** Largest ratio 1.367 (full seed 43); crossflow 1.059 / 1.073.
- **R1: no change on profile_crossflow, no change on full.**
- **Cost check: fails on cost as the code stands.** B's lanes' median step is 1.50-1.51 times A's (allowed 1.25).
- **Adoption:** the first clause (helps on crossflow, harms nowhere) is not met. The reading is "helps nowhere, harms
  nowhere", the clause for adoption on principle after a cost check, and the cost check fails. Not adopted by the rule;
  sent to the supervisor as a decision (registered: the lane does not take it).
- **Decision (the supervisor's, 2026-10-25 04:33 EDT): B is not adopted.** Reasons given: R1 reads no change on both
  splits; the near-outlet error shows the gauge is not crossflow's main error; the cost check fails at 1.5 times the step.
  B is not kept as an opt-in in the canonical code (Peter's rule: "No parallel old implementation: options are
  scaffolds"); it lives only on the unmerged branch `psharpe/isla-outlet-gauge` at 9250072af, as the record. No follow-up
  now; a bounded-probe design with more crossflow seeds is noted as a candidate after the reruns.

## R1 (median over test cases per seed; float32, 200,000 interior points, native protocol)

| split | arm | median p_L2 s42 / s43 | median v_L2 s42 / s43 |
|---|---|---|---|
| profile_crossflow | A (ISLA 0.9.0) | 3.0246 / 1.8880 | 1.5117 / 1.4054 |
| profile_crossflow | B (outlet level declared) | 1.3903 / 0.5184 | 1.1786 / 1.0092 |
| full | A | 0.3000 / 0.2706 | 0.1091 / 0.1029 |
| full | B | 0.2876 / 0.2735 | 0.1085 / 0.1081 |

| split | field | D = ln(B/A) | exp(D) | 2 SE | helps bar (D < -0.163 and \|D\| > 2 SE) | harms bar | per-seed B/A (42 / 43), beside |
|---|---|---|---|---|---|---|---|
| profile_crossflow | pressure | -0.945 | 0.389 | 1.093 | first met, second not (0.945 < 1.093) | no | 0.460 / 0.275 |
| profile_crossflow | velocity | -0.288 | 0.750 | 0.171 | (no helps clause on velocity) | no | 0.780 / 0.718 |
| full | pressure | -0.017 | 0.983 | 0.115 | no | no | 0.959 / 1.011 |
| full | velocity | +0.021 | 1.021 | 0.058 | – | no | 0.994 / 1.050 |

Two-seed standard deviations of ln(median p_L2): crossflow s_A 0.333, s_B 0.698; full s_A 0.073, s_B 0.036. On crossflow the
noise floor comes from both arms' seed gaps (A 60 %, B 2.7x between seeds).

## Beside the verdict (descriptive, not registered readings)

- On crossflow, both seeds' B/A pressure ratios (0.46, 0.27) and both velocity ratios (0.78, 0.72) are below 0.85, and B's
  seed-mean pressure is 0.39 of A's. The registered rule does not read this as "helps" because B's own two seeds differ
  by 2.7x. The direction is consistent on both seeds and both fields; the size is not resolved by two seeds.
- B's crossflow median pressure L2 is 1.39 / 0.52: seed 43 is below 1 (better than predicting zero), seed 42 is not. A's was
  3.02 / 1.89; GeoTransolver's is about 1.0 (AZ-DECIDE).
- M2, the near-outlet band (median over all cases of the mean predicted gauge pressure on the 1 % of scored points
  nearest the outlet, per RMS p): crossflow A 0.983 / 1.712, B 0.651 / 0.890, truth 0.017; full A 0.0057 / 0.0055, B
  0.0036 / 0.0036, truth 0.0011. The construction holds the outlet tokens' weighted mean exactly, yet the interior points
  next to the outlet stay far from the truth on crossflow: the remaining error is not the gauge.
- The constant the construction adds (median, per RMS p): crossflow 1.77 / 0.94, full 0.23 / 0.56. B's head leaves its own
  level free, as designed (the review found its pressure bias gets almost no gradient).
- The outlet was kept whole by the draw on 122 / 121 of 578 crossflow cases and 362 / 358 of 716 full cases; the probes'
  measure equals the outlet's area to 1e-5 (median ratio 1.0000).
- Mechanism driver against the scorer: per-case pressure_l2 within 1.5e-4 (B) and 6.8e-4 (A), velocity within 5.4e-4.

## Cost

| run | links | median step, all links (s) | median after each link's first 50 steps | B / A |
|---|---|---|---|---|
| A crossflow s42 / s43 | 8 / 8 | 0.5410 / 0.5401 | 0.5409 / 0.5401 | – |
| A full s42 / s43 | 8 / 8 | 0.5407 / 0.5388 | 0.5406 / 0.5388 | – |
| B crossflow s42 / s43 | 12 / 12 | 0.8142 / 0.8164 | 0.8141 / 0.8163 | 1.505 / 1.512 |
| B full s42 / s43 | 12 / 12 | 0.8115 / 0.8138 | 0.8115 / 0.8137 | 1.501 / 1.510 |

From the lanes' own metrics.jsonl (`train_step` step_time_s below 10 s; `cluster/cost_check.py`, output `cost_raw.txt`).
Peak memory 161 against 130 GiB (smoke). The profile beside it (descriptive, `prof/PROFILE.md`, LHR 340909): B / A 1.40 per
step on one GPU; of B's extra time, the probe tokens through the blocks take 87 % (forward 21 %, backward 66 %), the
probes' geometry 7 %, dynamic-shape kernels 2 %, selection and shift 0.5 %. The median 50,000 probes are 12.5 % more tokens
but cost about 40 % more time. So the cost failure comes from how many probes the construction reads and how they go
through backward, not from compilation; the number of probes is an implementation choice (the outlet mean needs far
fewer than 50,000 quadrature points), not part of the construction.

GPU-h (sacct, allocated): lanes 724.0, mechanism driver 13.7 (A 4, B 9.7), scorer 9.9, smoke 2.3, profile 0.3: **750.2**,
under the 800 cap (supervisor 10-23 18:00 EDT; 750 from 10-22 19:05).

## Predictions (PREREG section 6)

| prediction | p | outcome |
|---|---|---|
| M1 passes | 0.95 | yes |
| M2: B's crossflow band |pred mean| <= 0.3 RMS p on both seeds | 0.7 | no (0.65 / 0.89) |
| crossflow helps / no change / harms | 0.30 / 0.60 / 0.10 | no change |
| full no change | 0.85 | yes |
| B's step within 25 % of A's | 0.85 | no (1.50) |
| first clause / on principle to the supervisor / neither | 0.25 / 0.50 / 0.25 | the on-principle clause, failing its cost check |

## What it means, and how sure

- The construction works as designed at scale: exact outlet level on every case, no training instability, no harm on
  `full` (pressure 0.98, velocity 1.02 of A). Certain within the registered checks.
- On crossflow, B's errors fell on both seeds and both fields, by a large factor on pressure, but the registered noise
  floor, set by two seeds per arm with a 2.7x gap in B, cannot call it. More seeds would settle it; the registered verdict
  stays "no change".
- The near-outlet band shows crossflow's remaining error is not the gauge (PDIAG's amplitude reading); the construction
  may still help by removing a free degree of freedom from training.
- Cost: +50 % per step as implemented, almost all from the 50,000 passive probes in the blocks' backward pass.

## Records

Scores: `scores_B/` (LHR `$T/p200k_fp32_test_outlet/`), A's in `../isla-az-decide-2026-10-21/scores/p200k_fp32_test_azc/`.
Mechanism rows: `mech/<run>/mech.jsonl` (LHR `$T/kernel/outlet/mech/out/`). Checkpoints: LHR `$T/runs/outlet_*`. Reducer
command: `nice -n 19 ../../.venv/bin/python reduce_outlet.py ../isla-az-decide-2026-10-21/scores/p200k_fp32_test_azc
scores_B --mech mech --out tables.md`. Jobs: smoke 338532, profile 340909, lanes (first links) 340874-340877 (crossflow
resubmitted as 342423 / 342424 after the 10-22 17:41 PDT storage-quota event), scorer 377298 / 377876 / 379227, mechanism
367953 (A) and 377299 / 377877 / 379228 (B).
