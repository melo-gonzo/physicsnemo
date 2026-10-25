# Notebook draft: OUTLET-EXP result (for 18-notebook.qmd, #sec-nb-outlet-gauge), 2026-10-25

Draft by research-lane OUTLET-EXP, 2026-10-25 04:40 EDT. Source: `scratch/isla-outlet-exp-2026-10-22/RESULTS.md`,
`tables.md`, `JOBS.md`, `prof/PROFILE.md`. The supervisor appends, renders and commits; the lane edits no book file.

## Proposed new title of the entry

2026-10-22 to 2026-10-25 — The outlet pressure gauge: ISLA holds AeroZoo's prescribed outlet pressure by construction on
every test case; trained, it reads "no change" on `profile_crossflow` and `full` by the registered rule, though both
crossflow seeds improved (pressure 0.46 and 0.27 of ISLA 0.9.0's); it costs 1.5 times ISLA's step as implemented, and
it is not adopted {#sec-nb-outlet-gauge}

## Text to append to the entry (after its **Lanes.** paragraph)

**Result** (OUTLET-EXP RESULTS.md; PREREG md5 69283aa5 with the 10-23 erratum, read as registered; 750.2 GPU-h).

- **The mechanism holds on every case.** On all 2,588 scored test cases of the four B runs, the measure-weighted mean of
  B's pressure over the outlet's drawn tokens equals the prescribed 0 Pa to at most 1.4e-6 of the tolerance scale
  (registered tolerance 1e-5), so B is valid. The declaration is in every B training and scorer config and in no A config.
  No lane approached the event guard (largest validation-loss ratio 1.37).
- **R1 reads no change on both splits.**

| split | ISLA 0.9.0 (A) p / v, s42 / s43 | with the outlet level (B) p / v, s42 / s43 | seed-mean p ratio exp(D), 2 SE | verdict |
|---|---|---|---|---|
| `profile_crossflow` | 3.02 / 1.89; 1.51 / 1.41 | 1.39 / 0.52; 1.18 / 1.01 | 0.389, 2 SE 1.09 vs \|D\| 0.95 | no change |
| `full` | 0.300 / 0.271; 0.109 / 0.103 | 0.288 / 0.274; 0.109 / 0.108 | 0.983, 2 SE 0.115 | no change |

: Median relative L2 over the test cases per seed, float32, 200,000 interior points (OUTLET-EXP tables.md). {#tbl-nb-outlet-result}

  On crossflow both seeds' pressure ratios (0.46, 0.27) and velocity ratios (0.78, 0.72) are below 0.85, but B's two seeds
  differ by 2.7 times (A's by 1.6), so the registered noise floor (|D| > 2 SE) is not met. The direction is the same on
  both seeds and both fields; two seeds do not resolve the size.
- **The gauge was not crossflow's main error.** Next to the outlet, B's mean predicted pressure is still 0.65 and 0.89
  times the case's RMS pressure (A 0.98 and 1.71; truth 0.017), although the outlet tokens' mean is exact. On `full` the
  same band falls from 0.006 to 0.004 (truth 0.001).
- **Cost: fails on cost as the code stands.** B's lanes' median step is 1.50 to 1.51 times A's (0.81 against 0.54 s), against
  the registered 25 percent. A profile (descriptive) puts 87 percent of the extra time in the probe tokens through the blocks,
  two thirds of it in backward; the median 50,000 probes are 12.5 percent more tokens. Dynamic shapes cost 2 percent, the
  probes' geometry 7 percent. The probe count is an implementation choice, not part of the construction.
- **Adoption.** Not adopted by the rule: the first clause needs "helps" on crossflow; the on-principle clause (helps nowhere,
  harms nowhere) needs the cost check, which fails. **The supervisor's decision (2026-10-25 04:33 EDT): not adopted**,
  because R1 reads no change on both splits, the near-outlet error shows the gauge is not crossflow's main error, and the
  cost check fails at 1.5 times the step. It is not kept as an opt-in in the canonical code (Peter's rule: no parallel old
  implementation; options are scaffolds); it lives only on the unmerged branch `psharpe/isla-outlet-gauge` at 9250072af, as
  the record. No follow-up now; a bounded-probe design with more crossflow seeds is a candidate after the reruns.
- Predictions: M1 passes (0.95) yes; crossflow no change (0.60) yes; full no change (0.85) yes; B's crossflow band below
  0.3 RMS p (0.7) no; cost within 25 percent (0.85) no.

**Correction to this entry's Lanes paragraph:** the crossflow links 340874 and 340875 did not die on faulty nodes. The LHR
project share hit its 100 TB quota from about 17:38 to 17:51 PDT on 2026-10-22, and every link that started then died in
about 10 s, unable to open its log file. nvl72d035-T06 and -T14 are not bad nodes.

## Chapter changes

1. **13-aerozoo.qmd, the "In flight" callout after the crossflow paragraph** (lines 883-888 at b86c7391d): remove the
   callout and replace it with a plain paragraph:
   "ISLA trained with the outlet's prescribed gauge pressure held by construction (its pressure shifted by the one
   constant that gives the outlet's drawn cells, weighted by their measure, the prescribed mean) holds that mean on every
   test case. On `profile_crossflow` its median pressure error is 1.39 and 0.52 (seeds 42 and 43) against 3.02 and 1.89,
   and its velocity error 1.18 and 1.01 against 1.51 and 1.41; on `full` it is unchanged (0.288 and 0.274 against 0.300
   and 0.271). By the registered rule both splits read "no change": the two seeds disagree too much to separate the
   crossflow gain from seed noise. Next to the outlet its pressure is still 0.65 to 0.89 times the case's RMS pressure, so
   the gauge is not the main crossflow error. As implemented it costs 1.5 times ISLA's training step. It was not adopted
   and is not part of ISLA; it remains on the unmerged branch `psharpe/isla-outlet-gauge` as the record
   (@sec-nb-outlet-gauge)."
   Source: scratch/isla-outlet-exp-2026-10-22/RESULTS.md and tables.md.
2. **14-program-status.qmd, the OUTLET-GAUGE row** (line 720): status RUNNING becomes "DONE 2026-10-25 (OUTLET-EXP
   RESULTS.md; M1 exact on all 2,588 cases; R1 no change on `profile_crossflow` (exp(D) 0.39, within the noise floor 2 SE
   1.09; per seed 0.46 / 0.27) and `full` (0.98); cost check fails as the code stands, step 1.50 times; not adopted (the
   supervisor, 2026-10-25), kept only on the unmerged branch 9250072af; 750.2 GPU-h; @sec-nb-outlet-gauge)". Remove "lanes since 2026-10-22 evening; read
   about 2026-10-25 to 2026-10-26".
3. **18-notebook.qmd, #sec-nb-outlet-gauge**: the new title above; append the Result text; replace the Lanes paragraph's
   "died after 9 s on two faulty nodes and were resubmitted at 20:59 EDT as 342423 and 342424, with those nodes excluded"
   by "died after 9 s when the LHR project share hit its quota (about 17:38 to 17:51 PDT) and were resubmitted at 20:59
   EDT as 342423 and 342424"; in the cost paragraph, add that the cap was raised to 800 GPU-h at 18:00 EDT 2026-10-23 and
   750.2 were spent.
4. **Standings (08-standing.qmd, 13-aerozoo.qmd tables):** no change. B is not the reference; the AeroZoo ISLA row stays
   ISLA 0.9.0 (A).
5. No in-flight callout added; the one in item 1 is removed.
