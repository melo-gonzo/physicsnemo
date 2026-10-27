Status: DONE. All eight lanes trained to epoch 500 and were scored. Every section-4 check passed. The reads are
written and a second reader checked them. Earlier lane report: FINAL_REPORT_2026-10-26_partial.md.

**What ran.** All jobs on LHR; details in JOBS.md.
- Lanes:
  - F: 384252 and 384253.
  - FN: 384263 and 384264.
  - FNC: 384401 and 384402.
  - FNΩ: 384437 and 384438.
  - Each lane ran 9 links. The supervisor's watcher moved the final links to short QoS. I cancelled each finished lane's
    trailing pending link by its id: 419281, 419492, 419512, 419507, 419495, 419650, 420540 and 420531.
- Scorer: 420308, 420523, 420734 and 420920.
- Near-wall dumps: 420310 and 420921.
- Reduce: 422383.
- Read: `harness/read_affix.py`, md5 b60f28d3, unchanged. It wrote read.md (md5 0f2882fc) and read.json.
- Event guard: no anomaly in 89 checks. The worst validation spike was 2.05x.
- The registry row `^isla\.affix\.` is removed. Backup: ACTIVE_EXPERIMENTS.tsv.bak4.
- GPU-h: **918.6 of the 1,000 cap** (sacct; checks/sacct_final.txt).

**Registered readings.** Sections 4.1-4.5 of RESULTS.md give the tables. "Paired" means both seeds' ratios r42 / r43
are below 1/1.10.

| step | pressure | velocity | nut |
|---|---|---|---|
| F / W-DTT | no effect 0.756 (paired 0.623 / 0.949, flag down) | no effect 0.828 (0.707 / 0.977, down) | **helps (paired)** 0.697 (0.561 / 0.899, down) |
| FN / F | no effect 1.102 (0.995 / 1.205) | no effect 1.090 (1.052 / 1.125) | **harms** 1.156 (R1; paired 1.148 / 1.162, flag up) |
| FNC / FN | **helps** 0.610 (\|D\|/2SE 2.06; 0.684 / 0.551, down) | **helps** 0.640 (2.58; 0.698 / 0.592, down) | **helps** 0.653 (2.18; 0.743 / 0.571, down) |
| FNΩ / FN | **helps (paired)** 0.802 (0.873 / 0.746, down) | **helps** 0.803 (1.13; 0.839 / 0.773, down) | **helps (paired)** 0.858 (0.827 / 0.887, down) |

- R2, by wall-distance band. It is descriptive and listed in read.md.
  - FN / F, split at 3.05e-4 chords:
    - below: 1.020 / 1.080 / 0.997, no effect;
    - above: 1.129 / 1.145 / 0.988, no effect.
  - FNC helps in every band on every field.
- R3, pose:
  - F holds.
  - FN holds.
  - FNC holds at the standing's level (pressure 1.0021).
  - FNΩ holds.
  - W-DTT was broken (1.53-2.52).
- R4, cost:
  - Smoke step against W-DTT's code: F 1.033, FN 1.040, FNC 1.057, FNΩ 1.019.
  - Lanes, GPU-h per seed: F 114.3 / 114.3, FN 113.8 / 113.8, FNC 114.7 / 115.2, FNΩ 113.4 / 113.9. W-DTT: 110.4 / 110.3.
- Beside, not a rule:
  - **FNC / B-DTT: 0.987 / 0.885 / 0.996.**
  - **FNΩ / B-DTT: 1.297 / 1.110 / 1.309.**
  - FNC / FNΩ: 0.761 / 0.797 / 0.761.

**Predictions** (RESULTS.md section 3):
- A2, A3, A9 (p 0.25) and A16 happened.
- A8 and A15 (p 0.20 and 0.18) did not happen.
- Three-way predictions: A4 and A6 came in on their mode. A5 (FN harms nut, p 0.08), A7 (FNC helps pressure, p 0.40) and
  A12-A14 (FNΩ helps, p 0.15-0.25) came in off their mode.

**What the registration maps these readings to.** None is adopted; adoption is Peter's.
- **F** restores pose and helps (paired) nut. This is evidence for DTT-POSE-EXP's forest on the whole boundary, with
  rule 1.
  - F0, DTT-POSE's forest without rule 1, was not run.
  - On AirfRANS, conditions and components coincide, so rule 1 and grouping by component are not told apart here.
  - Rule-1 grouping on datasets with several conditions is Peter's decision.
- **FN** harms nut, so (ii) is not proposed. The registration's words: "Any harms: that fix is not proposed."
- **FNC** helps every field and harms none, so C goes to Peter with its blast radius: every dataset with several
  conditions.
- **FNΩ** helps every field and harms none, so Ω goes to Peter with its narrower blast radius: ISLA-DTT's read on every
  interior dataset.
- Both FNC and FNΩ carry N, which harmed nut. Neither was measured on F alone.

**Confidence.** The reviewer recomputed R1, R3 and the beside rows independently and found every number right. I applied
its wording fixes; RESULTS.md 4.7 lists them.
- FNC's R1 clears 2 SE on all three fields.
- F's R1 against W-DTT is limited by W-DTT's own seed spread.
- One beside finding: the frozen instrument would mark only R2 incomplete if a near-wall check failed. Every near-wall
  check passed, so nothing changes.

**Rulings relied on:**
- the supervisor's A1 and A2;
- the cap raise to 1,000 GPU-h;
- LANE_RULES;
- Peter's whole-boundary ruling.

New for project memory: none.

**Left for the supervisor:**
- Pruning `$T/runs/affix_*`, 132 GB each and 1.06 TB in all. I recommend keeping only `ISLA.0.500.mdlus`,
  `checkpoint.0.500.pt` and `norm_stats.pt` per run.
- Applying NOTEBOOK_DRAFT.md and its chapter changes.
- Peter's decisions on C, Ω and rule 1.
- The AeroZoo check, which stays gated.

Nothing is still running.

**Paths:**
- RESULTS.md, sections 3 and 4.
- NOTEBOOK_DRAFT.md.
- REVIEW_SECOND_READ.md.
- read.md.
