Merge verdict: **BLOCK**. The numbers in sections 3 and 4 are right. Two pieces of wording in RESULTS.md need fixing: the scoring of A8 and A15, and one sentence in 4.6 about the FNΩ gap.

I recomputed everything independently from the metrics.jsonl `infer_summary` and `infer_step` rows, without using read_affix.py. I edited no file; this answer is the review.

## Findings, most severe first

1. **RESULTS.md section 3, rows A8 and A15, and the tally · P1 · two predictions are scored "held" when their event did not happen.**
   - A8 predicts "FNC / FN harms on some field" at p 0.20. A15 predicts "FNΩ / FN harms on some field" at p 0.18. No field harms in either step.
   - Elsewhere the scorecard uses "held" to mean the event happened: A11 is "failed" when it didn't (p 0.65), and A9 is "held (at p 0.25)" when it did.
   - So the tally "A2, A3, A8, A9, A15, A16 held" is inconsistent with itself.
   - Fix: score A8 and A15 as "did not happen (p 0.20 / 0.18 on it; 0.80 / 0.82 on the outcome)". The binary predictions whose event happened are then A2, A3, A9 and A16.

2. **RESULTS.md 4.6, the FNΩ bullet · P1 · "so Ω alone leaves most of the pressure and nut gap" is wrong twice.**
   - FNΩ is not Ω alone: it carries F and N as well.
   - "Most" contradicts the section's own 4.5. Against W-DTT's gap to B-DTT, FNΩ closes 0.61 of it on pressure and 0.58 on nut (my numbers: 0.608 and 0.578), so about 40 % is left. "Most" is true only against FN's remaining gap (54 % / 64 % left).
   - This understates Ω, whose adoption is Peter's call.
   - Fix: "FNΩ / B-DTT 1.297 / 1.110 / 1.309; it closes 0.61 / 0.76 / 0.58 of W-DTT's gap (descriptive), against FNC's 1.02 / 1.29 / 1.01."

3. **4.6, the F bullet · P2 · a caveat is missing.** The mapping is quoted correctly ("evidence for DTT-POSE-EXP's adoption on the whole boundary"). But F is DTT-POSE's forest *plus* rule 1, and F0 (DTT-POSE exactly) was not run (A1.2). PREREG §2 gave F0 only p 0.45 of pose holding, because 4.1-26.5 % of its posed pairs have no native match. The bullet separates rule 1 from per-component grouping, but not from DTT-POSE without rule 1. Add one clause saying so.

4. **4.6, the FNC bullet · P2 · "the gap to the airfoil-only standing closed".** This is a seed-mean ratio from a beside row, and no rule tests it. Say "seed means at or below B-DTT's (beside, not a rule)".

5. **harness/read_affix.py:182-187 · P2 · near-wall check failures are recorded after the validity flags are computed (line 129).** A failed points-hash or scorer-deviation check would mark only R2 incomplete. R1, R3 and the beside rows would still print. PREREG §4 lists the coordinate hash among the checks before reading, and says a failed check makes "the row invalid". The docstring narrows the hash to R2. This changes nothing here, because every check passed.

6. **RESULTS.md section 3, row A17 · P2 · the "fourth digit" verdict can't be settled.** It rests on medians the training log rounds to milliseconds (0.971 / 0.991). Those allow a true ratio anywhere from 0.9788 to 0.9808, which straddles the 0.98 bar. This row is outside the reads-scored table.

7. **RESULTS.md section 4 checks paragraph · P2 · a small rounding.** "Dumps reproduce the scorer … at most 1.9e-7" holds for the 8 arm dumps (1.87e-7). Across all 12 dumps it is 2.0e-7 (B-DTT seed 42 pressure, 1.97e-7).

## Each item

1. **Section 4 numbers against read.md and read.json: CONFIRMED.**
   - 4.1: all 12 rows match read.md token for token, checked by script.
   - 4.3 and 4.5 match. W-DTT / B-DTT is 1.943 / 1.531 / 1.893. The CM gap-closed shares match: F 0.42 / 0.44 / 0.57, FN 0.27 / 0.24 / 0.34, FNC 1.02 / 1.29 / 1.01, FNΩ 0.61 / 0.76 / 0.58.
   - Every range and label in the 4.2 bullets matches read.md. The claim "where W-DTT's own gap is small" holds: W/B velocity is 1.116 at 0.1-1 chord and 1.098 beyond 1 chord.
   - 4.4 matches smoke_check.json and JOBS.md: step ratios 1.033 / 1.040 / 1.057 / 1.019 against W-DTT's code; 1.007, 1.016 and 0.980 between nested arms; peak memory and lane GPU-h as stated.
   - Checks: read.md md5 0f2882fc, read.json 92342bb5, nearwall_affix.json a0e4212d, instrument b60f28d3. All 12 dumps have the points hash ab498a7b9722. The nut deviation is 0.078530 on every run, so it is a constant scale. wait.log has 89 checks. All 24 arm rows have SNAPSHOT_USED `code_affix_<arm>`.

2. **Independent recompute: CONFIRMED.** I used sd as |Δln| / √2, the PREREG §5 thresholds, and `infer_summary`, which equals the mean of the 200 `infer_step` rows to 5e-8. Every run has epoch 500, 200 cases, and the same sample ids as W-DTT.

   | step, field | D | SE | \|D\|/2SE | R1 | r42 / r43 | label, flag |
   |---|---|---|---|---|---|---|
   | FN/F nut | 0.1446 | 0.0600 | 1.206 | harms | 1.148 / 1.162 | harms, up |
   | FNC/FN pressure | −0.4949 | 0.1200 | 2.062 | helps | 0.684 / 0.551 | helps, down |
   | FNΩ/FN pressure | −0.2213 | 0.1262 | 0.876 | no effect | 0.873 / 0.746 | helps (paired), down |
   | F/W-DTT nut | −0.3609 | 0.2002 | 0.901 | no effect | 0.561 / 0.899 | helps (paired), down |

   - The other 8 native cells and all posed cells also match.
   - R3 for FNC: pressure 1.00213 / 1.00168 and velocity 1.00185 / 1.00172. Only pressure exceeds 0.002, and its worst (0.00213) is no farther than B-DTT's worst (0.00219), so "holds at the standing's level" is right.
   - Beside rows: FNC / B-DTT 0.9868 / 0.8850 / 0.9963; FNΩ / B-DTT 1.2973 / 1.1100 / 1.3093; FNC / FNΩ 0.761 / 0.797 / 0.761.
   - W-DTT's pose ratios are 1.53-2.52.

3. **Instrument against PREREG §§4-5: CONFIRMED, apart from finding 5.**
   - R1, the paired reading and the sub-bar flag match NEARWALL PREREG_CONTINGENCY:125.
   - R3 uses the measured B-DTT worst seed, consistent with the 1.0017 / 1.0022 the registration cites.
   - Its source md5 e0739632 is the frozen PREREG. A2 (fc299626) changes only the new section 10, so sections 4-6 are identical, checked by diff.
   - The nut dump is excluded from the scorer check on purpose, and RESULTS discloses it.

4. **Predictions scored from the reads: CONFIRMED** for A2, A3, A4, A5, A6, A7, A9, A12, A13, A14 and A16. A8 and A15 are wrong (finding 1).

5. **Section 4.6:** the mapping is stated as registered. FN is "not proposed". C and Ω "go to Peter" with their blast radius. Nothing recommends adoption. The nesting caveat is accurate: N is in both FNC and FNΩ, and neither F + C nor F + Ω was run. Wording problems are findings 2-4.

6. **Anything that should make a step "incomplete as registered": none found.**
   - The section 4 checks pass.
   - No event guard fired (worst validation spike 2.05x, under the 4x bar).
   - The A1.7 test failures fall in the declared E-C and E-Ω classes.
   - The F smoke step (1.033) is under the 1.05 cost rule.
   - The GPU-h total (918.6) is within both the 920 and the 1,000 cap.

{"verdict": "block"}