Verdict: CONFIRMED WITH CORRECTIONS

DK-STAGE1-REVIEW of STAGE1.md (md5 844ebad1), 2026-10-24 02:39 EDT, read-only, no jobs. I read the REPORT, REVIEW and REVISION_NOTE files and the tables of sens/ cost/ cap/ gap/ order/ treeprec/ design/, plus JOBS.md, ATTRIB-090 PREREG section 8 and REF-090 PREREG:394.
What holds: almost every number traces to its source. The theta table, the pair counts, the forward times, the 400k fits, 2.3-3.4x, the coherence bound, the gap ratios and the treeprec shifts all match. The GPU-h table is right: 7.71 + 3.63 + 4.68 + 1.51 + 3.07 + 1.16 = 21.76, and each part equals its reviewer's sacct check (order's 0.07 overrun is disclosed). The arm sums are right (940 and 580), and so is 2 x 71 = 142, about 140. The lug's 240 per run is REF-090's 480 per pair. The treeprec noise claim, the lug bf16 masking, the nut wake share and the octant-tie cause are all carried in their corrected form. The memory lines are scoped as sens items 1-6 and treeprec item 1 require.

P1 (fix before STAGE1.md goes to the supervisor):
1. L4-7: "Each report was second-read ... and revised" and "All seven reports were confirmed with corrections" are not true yet. design/ has no REVIEW.md, and sens/ and treeprec/ are not yet revised (their REPORT.md equals REPORT_v1.md). Fix: "six reviewed (design/ not; its coherence bound is checked in cost/ and sens/); sens/ and treeprec/ are read through their REVIEW.md pending revision".
2. L11 (and L153 "every knob measured here is a trained constant"): the general claim survives in the headline although sens REVIEW item 6 narrowed it. Leaf is not shown to be trained in: on AirfRANS, leaf 2 and leaf 8 are inert (<= 0.1 %; l8 velocity 0.00737203 against 0.00737341 for the reference, sens_table.md). Fix: "On theta-1 / leaf-4-trained checkpoints theta acts as a trained constant; leaf 2 / 8 is inert on AirfRANS and costs up to +33 % on the lug."
3. L19-21: the capacity headline drops two qualifiers.
   - "The read itself is a small share of the step": the measurement is the 12 per-block reads alone, with the shared forest, walk and basis still running (cap REVIEW P1.1), and the lug's share is 26 %, which is not small.
   - "Not in the heads or the number of reads" contradicts the next sentence, which says the lug's first two reads are unused. It is also an inference-only ranking; cap REVIEW item 9 asks for "high as an inference ranking", with basis 8 at 2-4x on the lug.
   - Fix: "At inference, the slack is in the basis (on AirfRANS; on the lug basis 8 costs 1.9/3.6x) ... The 12 per-block reads alone (forest, walk and basis excluded; eager) are 9 % / 26 % of the step."
4. L22 "the read's settings do not explain it": only inference-time settings were tested. Arm 2 exists to test whether a retrained finer read closes the far-field gap, and sens REPORT:29 says that is untested. Fix: "no inference-time setting of the read explains it".
5. L106 "H1 is worse at every theta >= 0.5": order REVISION_NOTE (line 31) rejected this wording. The median H1/monopole is 0.96 and 0.83 at theta 0.5 and 0.98 at 0.75. Fix: "median worse at every theta >= 1 in every block (1.01-3.81x at 1); p95 worse at every theta >= 0.5".
6. L171, the K-B1 'why', "best of the fewer-read forms at inference": this is the withdrawn 'B1 beats last-k' claim in a new form. Last 8 and last 9 beat B1 on the subset (for example, last 9 zeroed is 19.0 against B1's 35.1, AirfRANS s42; cap REPORT:22). Fix: "best at inference among forms with <= 6 reads".

P2 (notes):
7. Arm costs, L164 and L170: "scaled by the measured forward times" does not give about 160 for K-THETA05.
   - The sens forward times are 485 ms at theta 0.5 against 484 at theta 1. cost/'s measured AirfRANS bf16 training steps at 100k points are 921 / 935 / 891 ms at theta 0.5 / 1 / 2 (cost_table.md, AirfRANS timing), so the arm is about 140 like the others.
   - K-EXACT's about 220 is the inference ratio 751/484 = 1.55. No training step was measured at theta 0. cost/'s AirfRANS memory fit gives about 73 GiB at about 1e8 pairs per block (it fits a GB300), and DrivAerML's 5.16 ms per million pairs predicts a similar 1.5x.
   - Its wall time is then about 27 h, not 18. Fix: cite cost/'s training steps, give 140 for arm 2 (totals about 920 / 560), and label 220 and its wall time as extrapolated.
8. L170, "theta 0.5 (in 2D the box test alone)" is not a property of 2D. The 2D bound makes coherence vacuous only at theta >= 0.765. At 0.5 the threshold is |mean n| > 0.875, while AirfRANS's smallest node is 0.872, so coherence can bind below 0.505 (sens_table.md, coherence section).
   - It held empirically: no node was refused at 0.5, and t0.5c1 equals t0.5 in metrics and pairs (sens). order/ shows the same counts, and cost/ gives 1.0x over all N.
   - The margin is 0.005, and the 800 training cases were not checked. (order REVIEW:12's "held value 0.5 is vacuous" is also wrong: 0.875 > 0.707.)
   - Fix: "on the AirfRANS test cases coherence refuses nothing at 0.5 (the margin is thin: it binds below 0.505)", or train it with coherence off, as in DESIGN's D1.
9. L170, "DK-GAP's candidate 2": gap REPORT:51 makes that candidate conditional on DK-SENS finding the far field sensitive to theta. It is not, at inference (sens REPORT:22-29). Say that the rationale is the ladder rung, with candidate 2 untested rather than triggered.
10. L27, L195, "builds and walks its forest in bf16": treeprec REPORT:10-13 and REVIEW:9-10 show that the coordinates, the boxes and the measure are bf16, while diameter_sq and the walk's box table are float32. Fix: "its forest is built from bf16-rounded coordinates under bf16 training".
11. L144, "Measured (job 364211, lug native)": that job is the lug bf16 run, and the diagnostic `normals @ canonical_frame` ran under its autocast (sens/lhr/probe_sens.py:259). The 84 % is a bf16 measurement; the float32 share was not measured. Say so.
12. L118 "The difference is smooth": gap REPORT:35-36 makes an exception for nut beyond 0.1 chord (roughness 0.076 / 0.112), which is where the nut gap sits. Add it.
13. L130: add gap's counterweight. C_f nut 0.121 could be about 40 % of log(1.342) = 0.294 (gap REPORT:45-46).
14. Rounding and ranges:
    - L44, L47 (sens REVIEW item 12): theta 0.5 pressure s43 should read 1.40, theta 1.5 pressure s43 1.35, and lug theta 1.5 s43 1.94.
    - L63: the held-out medians are 5-25 %, not 7-25 % (AirfRANS is 5 %).
    - L89: 8-50 should be 8-54 (cap REVISION_NOTE).
    - L93, L172: 6-8x is the mean-replaced cut; zeroed it is up to 9.2x, so cap says 6-9x.
    - L77: the leaf 16 / 8 step gains are cost's theta-3 rows. Say "at theta 3".
    - L96: last 8 also beats B1 (subset).
    - L105: seed 42's block 9 also needs theta < 0.0625 (1.35 %, order REPORT:10).

None of these changes stage 1's direction or the arm ranking. Items 1-6 are text fixes. Item 7 lowers the stage-2 ask by about 20 GPU-h and should be fixed before decision 1 is asked.

{"verdict": "block"}
