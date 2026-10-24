Status: DONE
Revised after REVIEW.md, 2026-10-24 02:38 EDT (the first version is `REPORT_v1.md`; changes in `REVISION_NOTE.md`).

**DK-SENS (stage 1a; read 2026-10-24 02:00 EDT).** Both checkpoints are tuned to the read they were trained with. Among the theta settings, error is lowest at theta 1 on both datasets, all fields, both seeds and both poses; leaf 2 and 8 tie leaf 4 on AirfRANS (within 0.1 %, at most 0.02 % below it), and on the posed lug leaf 8 reads 0.9996 / 0.991 of it. On native reads error rises as theta falls and as it grows, and the exact read (the limit theta → 0) is the worst setting of all; on the posed lug seed 43, θ 2 (2.003) and θ 2 at leaf 2 (2.120) are worse than the exact read (1.752). This mirrors E2: dense-trained checkpoints moved 2-20x when read through the tree, and these move 1.8-4.7x when read exactly (native). A cheaper read saves no eager forward time at these sizes.

**What ran** (LHR `$T/kernel/dttknobs_sens`, code_cmpaf_fix + recipe_r090, eager, float32 unless stated): smokes 362956, 362994, 363044, 364090; production 363160 (4 runs, native + posed, 18 settings); 364211 (AirfRANS by DK-GAP region; lug at bf16). **7.71 GPU-h of 10**; nothing running, no slot held, no commits. Jobs and commands: `JOBS.md`. **Gate passed before any other number:** every case matches the scorer's record to ≤ 5.4e-6; AirfRANS means match the standing to ≤ 1.8e-6; lug 0.00221933 / 0.00183431 (standing given to 4 digits); repeat floor d ≈ 3e-7. The leaf setting reached the trees: logged leaf size and max cells per leaf = 2 / 4 / 8. All numbers: `sens_table.md`, `sens.json`.

**Error / theta 1 (same weights, paired; [seed 42, seed 43]).** AirfRANS pressure / velocity / nut, native:
- θ 0.75: [1.07, 1.03] / [1.03, 1.05] / [1.04, 1.08]
- θ 0.5: [1.41, 1.40] / [1.18, 1.23] / [1.27, 1.44]
- θ 0.25: [2.07, 1.86] / [1.58, 1.73] / [3.17, 3.98]
- θ 0 (exact read): [2.71, 2.36] / [1.79, 2.04] / [3.54, 4.68]
- θ 1.25: [1.18, 1.20] / [1.04, 1.07] / [1.07, 1.07]; θ 1.5: [1.35, 1.35] / [1.10, 1.14] / [1.16, 1.17]; θ 2: [1.86, 1.91] / [1.30, 1.33] / [1.38, 1.44]

Lug von Mises: θ 0.75 [1.24, 1.29], 0.5 [1.97, 2.00], 0.25 [2.90, 2.53], 0 [4.62, 3.47]; θ 1.25 [1.39, 1.41], 1.5 [2.12, 1.94], 2 [3.20, 2.79]. At bf16 the same shape holds, smaller: θ 0.5 [1.44, 1.39], 0.75 [1.10, 1.10], 2 [2.11, 1.76]. The bf16 reference reproduces DK-GAP's bf16 read on the mean only (relative 2.5e-5 / 1.2e-5 for seeds 42 / 43); per case it differs by up to 1.0e-2 (seed 42; 2.6e-3 on seed 43; medians 1.6e-5 / 3.0e-5).

The prediction change against theta 1 grows the same way (AirfRANS nut d [seed 42, seed 43]: [0.0088, 0.0100] at θ 0.75, [0.017, 0.021] at 0.5, [0.059, 0.077] at 0.25, [0.067, 0.092] at 0). Ratios to interacting ISLA, seed means:
- AirfRANS (r090eag_af_isla): 0.994 / 1.128 / 1.342 at θ 1, 1.044 / 1.173 / 1.424 at 0.75, 1.40 / 1.36 / 1.82 at 0.5, 2.52 / 2.16 / 5.51 at θ 0.
- Lug: 1.394 (r090fix_lug_isla) or 1.288 (r090eag_lug_isla) at θ 1; 2.76 / 2.55 at 0.5; 5.71 / 5.28 at θ 0. At bf16, 1.043 at θ 1 and 1.476 at 0.5.

Leaf size is inert at θ 1: leaf 2 and 8 change AirfRANS by ≤ 0.1 % (d ~1e-4). On the lug, leaf 8 changes it by ≤ 0.8 %, while leaf 2 adds +2 % on seed 42 and +33 % on seed 43.

**DK-GAP's regions (AirfRANS; `sens_regions.md`).** The far-field error does not move for θ ≥ 1 (≤ 1.2 % at 0.75 in the seed mean, ≤ 2 % per seed); it grows below that.
- Beyond one chord, every field reads 1.00× its theta-1 error at θ 1.25 to 2 (change ≤ 5 % of the regional error), and 1.004-1.012× at 0.75 in the seed mean (per seed up to 1.020, seed 43 pressure). My inference: far points already read only the coarsest clusters at θ 1.
- Below 0.75 it only grows: at θ 0.5 pressure 1.16×, velocity 1.11×, nut 1.08×; at θ 0 5.6×, 2.8×, 3.5×.
- The gap to ISLA beyond one chord (pressure / velocity / nut) is smallest at θ 1, 1.41 / 1.72 / 2.05, and rises to 1.64 / 1.91 / 2.22 at θ 0.5.
- Wake velocity against ISLA goes 1.97 → 2.05 → 2.22 (θ 1, 0.75, 0.5); wake nut goes 1.19 → 1.21 → 1.34.
- For θ > 1, the rise in pressure error sits near the wall (91-98 % of it within 0.01 chord).

So no inference-time theta narrows the wake or far-field gap. Whether training at a smaller theta would (DK-GAP's candidate 2) is untested here: a checkpoint co-adapts to its own read.

**Cost.** Pairs per point fall as theta grows and rise as it falls (exact-cell share 0.11 on AirfRANS, 0.08 on the lug at θ 1):
- AirfRANS: 21.6 at θ 1, 15.0 at θ 2, 34.7 at 0.5, 60.7 at 0.25, 1,007 at θ 0.
- Lug: 118, 54, 300, 791, 6,525 in the same order.

Forward time is flat for every θ > 0: AirfRANS 475-498 ms and lug 195-252 ms, against 751 ms and 353 ms at θ 0. What is timed: the median over warm cases (the first scored case of each part dropped) of CUDA events around the eager model forward (compile off), so it includes the forest build, the walk and the probe's own diagnostics (an extra canonical frame, an eigh, float64 node means, host syncs); collate and the host-to-device copy are outside it. The diagnostics do not depend on theta, so the flatness probably holds without them; compiled or larger-case timing was not measured. Peak memory (same bracket, so it includes the diagnostics too): AirfRANS 5.1 GiB (20.5 at θ 0); lug 1.47 GiB at θ 1, 1.39 at θ 2, 2.50 at 0.5, 21.7 at θ 0. A larger theta or a smaller leaf saves pairs and up to 0.08 GiB native (0.21 GiB on the posed lug at θ 2), at a cost in error of 4-220 % for θ 1.25-2 and up to 33 % for leaf 2 (none on AirfRANS). Leaf 8 costs pairs and memory (AirfRANS 24.5 vs 21.6 pairs/pt, 5.15 vs 5.11 GiB; lug 132.5 vs 118.4, posed 1.88 vs 1.63 GiB). A smaller theta costs pairs, memory and error at once.

**Coherence (the coordinator's add-on).** The vacuity claim holds in the code run: at θ ≥ 1 no node is refused by coherence on either dataset. The smallest node |mean n| is 0.872 (AirfRANS) and 0.705 (lug), so in these data the test binds only below θ 0.505 and 0.768.
- AirfRANS: holding coherence at its theta-1 value changes nothing at θ 0.5 or 0.75 (identical metrics and pairs). At θ 0.25 it carries 19-44 % of the rise across fields and seeds: pressure 1.67 instead of 2.07 on seed 42, 1.70 instead of 1.86 on seed 43.
- Lug native: coherence's share is ≤ 8 % (θ 0.25 seed 42: 2.756 instead of 2.901; θ 0.5: 1.950 instead of 1.968; pairs 299.5 against 300.3; 0.75 identical). On the posed lug it is larger (θ 0.5: 1.40 instead of 1.58 on seed 42).
- So the rise below θ 1 comes mostly from the box criterion.

**Posed against native.**
- AirfRANS: posed equals native to within 0.25 % at every setting (ratios 0.9995-1.0025), with the same pairs (pairs ratio 1.000; 1.001 at leaf 8 and θ 2 leaf 8). Not bitwise: 0.5-14 % of cases agree to 1e-4.
- Lug: they differ at every θ > 0 (posed / native 1.53 and 1.98 at θ 1, ranging 1.01-1.98 across settings, with 1.17-1.41× the pairs) and match only at θ 0 (1.0000 / 1.0001).
- Cause, partly inferred. Measured only in job 364211 (lug bf16, native pose; not in the fp32 production 363160 or in `sens.json`): 84 % of the lug's cells have a normal component below 1e-5 in the canonical frame (median over cases 0.8384, range 0.7565-0.9108, the same on both seeds). `normal_octants` assigns octants by a `> 0` test on normals @ frame (float32 in the fp32 reads where posed ≠ native was measured), so these cells already sit on the sign of a rounding residue in the native pose; that they flip once the case is posed, building a different forest (1.29× the pairs at θ 1), is inferred: no posed octant labels were compared and no tie-break was tried. The frame's own skews are near-ties too (1.2e-6 against the 1e-9 sign-tie threshold), so the frame is not excluded. Unconfirmed without a tie-break test.

**How sure.** High on the direction: every setting moves both seeds the same way, and the effects are paired against a 3e-7 floor. Effect sizes differ between seeds by up to about 30 % on native reads (AirfRANS nut, the lug's θ 0 and leaf 2 reads) and by more on the posed lug (θ 0: 3.02 vs 1.75; θ 0.25: 2.02 vs 1.39), so read the per-seed values, not the means. The bf16 shape is solid; its sizes are compressed by the bf16 read-out noise DK-GAP found. The far-field conclusion rests on both seeds (≥ 1 chord: 1.00 / 1.00 for θ ≥ 1; growth below 0.75 in both). This agrees with PROBES-090's `out_theta0.5`, `out_theta0.25` and `out_theta0` on the bug-trained r090_af_dtt, which moves further from its scorer record as theta falls (seed 42, median per-case |relative difference| of pressure against the scorer, `reduced_full.md`: 0.31 / 0.99 / 1.41; unsigned, so it shows change, not a direction).

**Rulings relied on:** COMMON_RULES (inference only; LHR only; constants set in-process on a copy; scorer protocol and gate; short-QoS slots). **For project memory:** a theta-1/leaf-4-trained ISLA-DTT checkpoint is most accurate read at theta 1; no inference-time theta in [0, 2] (0, 0.25, 0.5, 0.75, 1.25, 1.5, 2) or leaf in {2, 8} improves it (AirfRANS, lug): on AirfRANS leaf 2 and 8 tie within 0.1 %; on the lug leaf 2 is worse (+2-33 %) and leaf 8 ties within 0.8 %, except that it reads 0.991 on the posed lug seed 43. The exact read is not a better reference for such a checkpoint (1.8-4.7× worse on native reads). Checkpoints trained at another theta or leaf were not tested. Also, the lug's posed ≠ native likely comes from octant ties of axis-aligned normals (frame near-ties not excluded); unconfirmed without a tie-break test.

**Left.** Nothing is running. Open questions for stage 2: whether training at a smaller theta closes the far-field gap (needs a retrain), and, for the lug, a tie-break test to confirm the octant-tie cause, then a tie-robust octant split (a code change).

**Stays on disk:** LHR `$K/out` (67 MB) and `$K/out_v3` (24 MB); the local folder (scripts, tables, JSON) is about 2 MB. No worktree was made; the local raw copies are deleted.
