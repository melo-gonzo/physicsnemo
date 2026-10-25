# NOTEBOOK_DRAFT: AZ-RESCORE (AeroZoo under AeroZoo's own metrics)

Draft of 2026-10-25 06:42 EDT for the supervisor and the second reader; lanes never edit the book. Source of every number:
`scratch/isla-az-rescore-2026-10-24/RESULTS.md` and `AEROZOO_SPECMETRIC.json` (md5 c8fb1c82; proposed home
`results/char/aerozoo_090_specmetric.json`), from LHR jobs 376802, 376803, 376804, 376818, 378817 and 378820.

## Notebook entry (18-notebook)

**2026-10-25: AeroZoo rescored with AeroZoo's own interior metrics.** AZ-SPEC-AUDIT (10-24) found that no model of the AeroZoo
standing receives data the task specification forbids, but that the book scored with a different metric (M1): the per-case
relative L2, every cell counted equally, on 200,000 random cells, in float32. The specification (aerozoo
`docs/book/06-splits.qmd`, lines 282-298 and 429-472) scores the interior task with ε_U and ε_p: the volume-weighted RMS velocity
and pressure errors over every interior cell, in units of the case's U_ref and q_ref, in float64; with ε^N, the same within one
reference length of the body; reported as mean and median over all test cases and over the `mean_stable` ones. AZ-RESCORE rescored
all 48 AeroZoo checkpoints that way, without retraining: chunked inference over the full mesh (0.6 M to 24 M cells per case), each
lane on its own snapshot, and AeroZoo's reference functions run verbatim. The scorer reproduces the registered 200k relative L2
exactly on the registered sample, agrees with an independent float64 implementation to 2.4e-8, and its free-stream rows reproduce
AeroZoo's own `freestream-v1.2.csv` on all 5,969 test case-splits to 4.8e-9. Cost: 258 GPU-h.

What changes. The relative L2 over the full mesh equals the registered 200k number to 0.5 %, so the sample was not the issue; the
metric was. Three readings change. On `family_non_box` velocity GeoTransolver moves ahead of both forms of ISLA (ε_U 0.396 against
0.404 for ISLA-DTT and 0.418 for interacting ISLA). On pressure ISLA-DTT moves ahead of interacting ISLA on `family_non_box`
(0.706 against 0.720), and interacting ISLA moves ahead of ISLA-DTT on `reynolds` (0.371 against 0.401). Every other order holds.
On `full`, `bank_zero_to_cad`, `bank_blobs` and `reynolds`, both forms of ISLA beat GeoTransolver by 4 to 10 times on both fields. On
`profile_crossflow` and `family_non_box` both forms are worse than AeroZoo's free stream on velocity and worse than zero on
pressure. GeoTransolver is not better than the free stream on any split (best: crossflow velocity, 0.97 of the free stream's ε_U).
Its ε_p is the zero predictor's to within 3 %.

Registered readings, beside. §12(f) on the test sides, read under ε: still a close call (six-split mean R 0.958 velocity, 0.990
pressure; reynolds pressure 1.082), so the default interior form is unchanged. The default rests on Peter's rulings in any case.
LRE-AZ: "harms" on velocity under ε as under relL2 (1.218), so it is still not adopted. E1-AZ (HEAD-NORM): the registered REJECT stands.
On the spec's case-mean ε its pressure ratio is 1.064 (seeds 1.002, 1.157) and branch 1 would not fire, but it fires on case medians
(1.298). Any change of status is Peter's.

## Chapter 13 (13-aerozoo.qmd @ b86c7391d), changes

Numbers come from `results/char/aerozoo_090_specmetric.json` (the file above). Its arms are keyed "ISLA (interacting)", "ISLA-DTT",
"GeoTransolver", "Transolver", "E1-AZ HEADNORM" and "LRE-AZ B", with splits as in `aerozoo_090_final.json`. For each field and
case set (`all` or `stable`) it holds `mean`, `per_seed`, `median` (the seed mean of the per-seed medians, AeroZoo's convention),
`per_seed_median`, `median_of_case_seed_mean` (the book's convention until now) and `n_cases`. Fields: eps_U, eps_p, eps_U_near,
eps_p_near, eps_nut, eps_k, eps_omega, rel_* (full mesh), and reg_rel_velocity and reg_rel_pressure (the registered 200k
numbers). `references` holds `zero` and `freestream`.

1. **The primary numbers become the spec's.** `tbl-aerozoo-standing` reports ε_U and ε_p, mean [median] over the test cases and
   seed means, from the new file. The zero-predictor row is no longer the constant 1.0: it is per split (for example `full` 1.16
   [1.00] and 1.68 [0.110]). Add a free-stream row (AeroZoo's own calibration): the area-weighted mean prescribed inlet velocity
   everywhere and zero pressure, its ε_p equal to the zero predictor's. Add a second table, or a toggle, with ε^N_U and ε^N_p and
   with the mean_stable case set. The relative-L2 table stays beside, relabelled "relative L2, beside AeroZoo's metric
   (registered readout)". The registered R1 readout, `tbl-aerozoo-r1`, stays as registered on relL2. Beside it, add the same
   ratios on ε. GeoTransolver ÷ ISLA on ε_U: full 8.78, bank_zero_to_cad 4.56, blobs 8.68, nonbox **0.946**, crossflow 0.638,
   dual_inlet 1.20, reynolds 7.13. On ε_p: 10.0, 8.07, 9.21, 0.665, 0.517, 1.33, 4.71. Above one on 4 of the 6 held-out splits on
   both fields; relL2 has 5 of 6 on velocity.
2. **para-aerozoo-standing-lede.** Rewrite on ε. "GeoTransolver ÷ ISLA is above one on 4 of the six held-out splits" (was 5 on
   velocity). GeoTransolver leads on the velocity of `family_non_box` (0.396 [0.264] against 0.418 [0.334]) and
   `profile_crossflow` (0.827 [0.777] against 1.29 [1.31]). Where both forms of ISLA are above the zero predictor on ε_p:
   `family_non_box` (0.720 and 0.706 against 0.467) and `profile_crossflow` (5.01 and 4.74 against 2.59), the same two splits as
   on relL2. New sentence: "Under AeroZoo's metric the free stream is the baseline that matters: GeoTransolver's ε_U is 0.97 to
   1.90 times the free stream's and its ε_p 1.00 to 1.03 times the zero predictor's, so neither baseline has skill beyond the
   free stream on any split; ISLA's ε_U is 0.16 to 0.27 of the free stream's on `full`, `bank_zero_to_cad`, `bank_blobs` and
   `reynolds`, 0.77 to 0.83 on `profile_dual_inlet`, and 1.47 to 1.55 on `family_non_box` and `profile_crossflow`." The 200k /
   float32 sentence becomes: "Every model is scored on every interior cell of every test case (0.6 M to 24 M), with AeroZoo's
   reference functions, in float64, from its final checkpoint; inference runs in chunks of 200,000 points in float32
   (@sec-aerozoo-request)."
3. **@sec-dataset-aerozoo (line 155), T1.** Replace "the model predicts all five" with: "the release stores seven mean fields;
   the protocol predicts five of them, velocity, gauge pressure, eddy viscosity, $k$ and $\omega$, and drops the intermittency
   and $Re_{\theta t}$, so this is the interior task on five of its seven fields. AeroZoo's interior metrics read only velocity
   and pressure, so they are unaffected." Use "the interior task, five of the seven fields" wherever the task is named,
   including the lede.
4. **@sec-data-aerozoo, the Scoring bullet (line 393).** Replace with: "**Scoring.** AeroZoo's interior metrics, ε_U and ε_p, are the
   root-mean-square velocity and pressure errors over every interior cell, each weighted by its volume
   (`_target_quadrature_measure`) and in units of the case's `reference/speed` and `reference/dynamic_pressure`; ε^N is the same
   over the cells within `reference/length` of a body face. They are computed with the release's reference functions, in
   float64, and reported as mean and median over all test cases and over the `mean_stable` ones. These labels are read only to
   score, never as inputs. Beside them, as before, the per-case relative L2 of each physical field. Source:
   scratch/isla-az-rescore-2026-10-24/RESULTS.md." Keep the Loss bullet, and add (L1): "The loss is not volume-weighted;
   AeroZoo names the cell volume as the weight for the loss and the metrics, and using it in the loss is a modelling choice."
5. **R1, release and hash (@sec-data-aerozoo or @sec-aerozoo-measured).** Add: "The release is AeroZoo v1.2 (`index.json` sha256
   c8e9508e8a9d8291e207470e7e5ecf2d44135ea0793aebe58cdb614da734b51b), read from a copy whose interior rows are permuted. The split
   lists are `splits.json` intersected with the release's complete cases; every list matches it case for case, and every test
   side is scored whole (716 / 333 / 1,367 / 1,074 / 578 / 516 / 1,385 cases; mean_stable 515 / 234 / 917 / 692 / 173 / 459 /
   1,001)."
6. **tbl-aerozoo-inputs, I1.** Label the Transolver row "Transolver (a floor without the boundary conditions)". Add to the
   caption: "Transolver receives neither the prescribed values nor the condition types, so it does not take the posed problem;
   AeroZoo's rule counts such a result as answering a different question, and it is reported as a floor."
7. **I2, beside tbl-aerozoo-inputs.** Add: "GeoTransolver reads a uniform 50,000-face draw of each boundary, and both baselines'
   distance to the `no_slip` faces is computed from that draw; ISLA reads every face and draws its 200,000 tokens inside the
   model. This is less input than ISLA reads, and AeroZoo's rule allows it; in the rescore the draw is seeded per case, and
   another draw moves GeoTransolver's ε by at most 1.8e-3 relative."
8. **@sec-aerozoo-measured (para-aerozoo-measured, lines 417-419).** Replace "per-case relative L2 per field ... averaged over the
   cases" with the ε definition of item 4, and state the relative L2 as beside.
9. **@sec-aerozoo-form, S1.** Keep the registered read (relL2, close call), and add after its first paragraph: "This read uses
   the test sides, where AeroZoo reserves the validation part for choosing models and settings. It is reported as registered,
   and it does not choose the form: Peter ruled before any ISLA-DTT test score that a close call goes to ISLA-DTT on principle
   (10-23), and since 10-24 an interior model is ISLA-DTT by construction. Read under AeroZoo's metric on the same test sides, the
   bands give the same close call: R = 0.912 to 0.990 on ε_U (mean 0.958) and 0.940 to 1.082 on ε_p (mean 0.990, above 1.05 on
   `reynolds` only); on `full`, 0.978 and 0.959." Add the ε table beside tbl-aerozoo-form (READINGS.md, "spec eps_U / eps_p",
   with seeds). Under ε, interacting ISLA is ahead of ISLA-DTT on `reynolds` pressure (0.371 against 0.401, seeds 1.516 and 0.753,
   within the seed spread) and on `profile_dual_inlet` pressure (7.04 against 7.22).
10. **@sec-aerozoo-baselines.** Add: "Under AeroZoo's metric the comparison is with the free stream, not only with zero:
    GeoTransolver's velocity error is below the zero predictor's on every split but not below the free stream's on any (0.97 to
    1.90 of it), and its pressure error is the zero predictor's (1.00 to 1.03 of it); Transolver's velocity error is the zero
    predictor's (1.15 against 1.16 on `full`)."
11. **@sec-aerozoo-failures.** Restate on ε: `profile_crossflow` interacting ISLA ε_p 5.01 [5.10], ISLA-DTT 4.74 [4.79], against the
    zero predictor's 2.59 [1.97]; ε_U 1.29 and 1.26, above the zero predictor's 1.04 and the free stream's 0.855. On
    `family_non_box`: ε_p 0.720 [0.176] and 0.706 [0.293] against 0.467 [0.092]; ε_U 0.418 and 0.404 against the free stream's 0.270.
12. **@sec-aerozoo-request.** Add the measured partition dependence of the rescore: "Scored over the full mesh in chunks of 200,000
    points, interacting ISLA's per-case ε moves by at most 4.5e-4 relative when the cells are partitioned again with another seed
    (two `full` test cases), GeoTransolver's by 4.3e-6 and Transolver's by 3e-8; ISLA-DTT moves by up to 1.9e-4 between chunk sizes
    of 200,000, 1,000,000 and 2,000,000, which is float32 roundoff, since it is exact under chunking."
13. **para-aerozoo-reynolds-scale (LRE-AZ).** Add: "Under AeroZoo's metric the fixed scale harms velocity too (ε_U B ÷ A 1.218,
    seeds 1.249 and 1.190; ε^N_U 1.155), and pressure reads no change (ε_p 1.433, with a two-seed spread too large for the bar)."
14. **@sec-aerozoo-limits.** Replace "**Per-case relative errors.**" with: "**AeroZoo's metric weights the far field.** ε_U and ε_p
    weight each cell by its volume, so the large free-stream cells of an enclosure dominate them; ε^N restricts them to within one
    reference length of the body, and the per-case relative L2 beside them counts each cell equally." Add four bullets:
    - T1: "**Five of the seven fields.** The intermittency and $Re_{\theta t}$ are not predicted."
    - S1: "**The interior form was read on test sides.** §12(f)'s registered read used the test sides; the form rests on
      Peter's ruling, not on that read (@sec-aerozoo-form)."
    - S1b: "**A design-time look at test cases.** The per-case relative loss of every model was chosen from a magnitude check on
      22 cases, one from each split's test side, which AeroZoo's rule does not allow for a setting. The same loss went to every
      model, and the fact it rests on, targets spanning decades across cases, holds on `full.train` as well, so no model gains."
    - L1: "**The loss is not volume-weighted** (@sec-data-aerozoo)."
15. **Callouts.** No in-flight callout is added. The form callout after tbl-aerozoo-r1 is unchanged.

## Chapter 20 (20-limits.qmd @ b86c7391d), changes

1. **The tally (lines 57-81).** AeroZoo enters on `full` velocity, which is unchanged under ε (GeoTransolver ÷ ISLA 8.78; seeds
   0.693 / 0.696 against 0.0809 / 0.0773, so "leads"). If the tally moves to ε, `az_cell` reads `eps_U` and `eps_p` from the new
   file and no verdict changes. `az_pressure_clause` then prints GeoTransolver ÷ ISLA 10.0 and Transolver ÷ ISLA 9.97 on ε_p
   (relL2: 4.12 and 3.84). The clause "both baselines are about a zero predictor" stays true (ε_p 1.69 and 1.69 against the zero
   predictor's 1.68).
2. **@sec-oq-aerozoo (lines 754-776).** Restate the two failures on ε as in item 11 of chapter 13: `profile_crossflow` ε_p 5.01
   [5.10] (ISLA) and 4.74 [4.79] (ISLA-DTT) against the zero predictor's 2.59; `family_non_box` 0.720 [0.176] and 0.706 [0.293]
   against 0.467. Add: "On both splits ISLA's velocity error is also above AeroZoo's free stream (ε_U 1.29 against 0.855 and 0.418
   against 0.270), so the failure is not only in pressure."
3. **The cross-dataset paragraph (lines 522-538).** "on AeroZoo the measurement does not say that ISLA transfers better" holds more
   strongly under ε, because GeoTransolver now also leads on `family_non_box` velocity. Cite results/char/aerozoo_090_specmetric.json
   beside aerozoo_090_final.json.
4. **The open-question table (line 644).** Unchanged.

## Chapter 14 (program status), for the supervisor

E1-AZ's line (HEAD-NORM REJECT) stays. Beside it, add: "on AeroZoo's case-mean ε the pressure ratio is 1.064 and branch 1 would not
fire, but it fires on case medians (1.298); the verdict stands, and any change is Peter's (RESULTS.md §8)."
