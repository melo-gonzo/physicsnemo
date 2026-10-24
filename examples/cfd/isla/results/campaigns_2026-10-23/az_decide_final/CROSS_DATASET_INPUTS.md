# Cross-dataset inputs for PREREG_v2 §13(b): ISLA-DTT ÷ interacting ISLA beside the AeroZoo §12(f) verdict

Gathered 2026-10-21 ~17:20 EDT by a read-only lane (AZ-DECIDE). No AeroZoo `azc_*_isla_*` / `azc_*_dtt_*` score or
validation file and nothing under `scratch/isla-az-decide-2026-10-21/scores/` was opened. PREREG_v2.md md5
ebd75524aa08dc4f1b01bacea8ac8cc7 (matches). Paths are relative to `scratch/` unless they start with `examples/`.
R = ISLA-DTT ÷ ISLA, seed means (seeds 42, 43), relative L2, float32. Per-seed R is seed-paired (s42 ÷ s42, s43 ÷ s43).
"Compiled" = the original scorer (`compile_blocks=true`, under the per-block compile fault COMPILE-AF found); "eager" =
`compile=false model.compile_blocks=false` re-score of the same checkpoints.

## Table

| row | dataset / code | metric | R eager (s42, s43) | R compiled (s42, s43) | label | source |
|---|---|---|---|---|---|---|
| 1 | AirfRANS, ISLA 0.9.0 (code_r090), r090_af_dtt ÷ r090_af_isla, 200 full_test cases, 100k pts, epoch 500, 512/512, bf16 training | p / v / nut | **1.181 / 1.154 / 1.433** (p 1.139, 1.228; v 1.143, 1.166; nut 1.440, 1.427) | 1.163 / 1.147 / 1.396 (p 1.126, 1.204; v 1.132, 1.163; nut 1.353, 1.441) | eager function of weights trained under the compiled-path defect; every DTT seed above every ISLA seed on every field, both paths | compiled: isla-ref090-2026-10-20/RESULTS_partial.md:79-81; eager: isla-compile-af-2026-10-21/RESULTS.md:60-62, impact/IMPACT.md:47; per seed: impact/IMPACT_tables.md:243-272 (native rows) |
| 1-fix | AirfRANS, 0.9.0 retrained under the fix (r090fix_af_isla / r090fix_af_dtt, code_cmpaf_fix, eager scorer) | p / v / nut | **does not exist yet** | – | training on LHR, paused by the maintenance | isla-ref090-2026-10-20/r090_final_tables.md:106 ("REF-090-FIX airfrans: not scored yet"); STATUS.md:1,13 (09:25 EDT) |
| 2a | AirfRANS, A2 era, E3b (code_anc, recipe_anc), arm D ÷ arm A, 512/512, 3.75e-4, same 200 cases / 100k pts as REF-090 | p / v / nut | **0.754 / 0.889 / 0.995** (p 0.767, 0.742; v 0.925, 0.854; nut 1.023, 0.971) | 0.742 / 0.887 / 0.988 (p 0.750, 0.734; v 0.920, 0.856; nut 1.012, 0.968) | A2-era code (code_anc); trained under the defect; D below A on both seeds for p and v, seeds interleave on nut | eager: isla-compile-af-2026-10-21/RESULTS.md:67, impact/IMPACT.md:48; per seed: impact/IMPACT_tables.md:87-122; config: isla-review-2026-10-21/numbers/REPORT.md:91-93 |
| 2b | AirfRANS, A2 era, afu_dtt (code_dtt16 + recipe_v8, **384 wide / 256 slices, 5e-4**) ÷ A2 ISLA afu_a2s_isla (512/512, 3.75e-4) | p / v / nut | **0.971 / 1.210 / 1.146** (p 0.924, 1.021; v 1.175, 1.246; nut 1.129, 1.166); afu_dtt eager 0.01752 / 0.01150 / 0.02560 ÷ afu_a2s eager 0.01804 / 0.00951 / 0.02233 | 1.109 / 1.258 / 1.213 (p 1.001, 1.225; v 1.216, 1.303; nut 1.195, 1.234); 0.01996 / 0.0119 / 0.02706 ÷ 0.01799 / 0.00946 / 0.02230 | different code versions on the two sides and unequal size/rate; trained under the defect; eager p within seed spread (seeds interleave), v and nut every DTT seed above | isla-compile-af-2026-10-21/RESULTS.md:144-154, impact/IMPACT.md:91-92; per seed: rescore_fu_cmp.txt (afu_dtt_seed42/43 native), impact/IMPACT_tables.md:33-50; config: isla-kwarg-audit-2026-10-17/REPORT.md:97-98 |
| 3 | GINOT lug, 0.9.0, r090_lug_dtt ÷ r090_lug_isla | von Mises | **not scored** (no R exists on either path) | – | r090_lug_dtt cancelled at epoch 235-237 of 500 by the supervisor (01:09 EDT 10-21, compiled-path defect), checkpoints kept, never scored; its replacement r090fix_lug_dtt (rows 0-1) is training | isla-ref090-2026-10-20/JOBS.md:121, PREREG.md:306-308, r090_final_tables.md:116, STATUS.md:13 |
| 3-ctx | lug ISLA 0.9.0 alone (for scale) | von Mises | eager 0.0018933 (0.0017804, 0.0020062) | 0.0019044 (0.0018163, 0.0019925) | – | r090_final_tables.md:62,101 |
| 4 | QUERY-POP-090, AirfRANS 0.9.0 + DrivAerML Volume 0.9.0 (eager only) | r = error on fixed scored set with altered query population ÷ native population | see notes (ISLA material/severe; ISLA-DTT roundoff in all 10 modes) | none (eager only) | AirfRANS rows: eager function of weights trained under the compiled-path defect; Volume unaffected | isla-az-decide-2026-10-21/QUERYPOP090_FOR_12F.md:16-31 |
| 5 | ATTRIB-090 R5 (where the AirfRANS order reversed) | ρ_CONT = DTT-CONT ÷ E2 CONT, pressure | **not read** | – | arms att2c af-dtt / af-cont training, interrupted by the maintenance | isla-attrib090-2026-10-21/RESULTS.md (only §1 Tensile2d, lines 1-44); STATUS.md:13-15; rule PREREG.md:98-101 |
| 6 | DrivAerML Volume, pre-A2: ISLA-DTT code_dtt16 (forest in own frame, θ = 1; branch commit 6094c4e9d and later; dtt_vol_tg1pf, job 152583) ÷ ISLA char reference (results/char/drivaerml_volume.json), both 512/512, 3.75e-4, raw reads, 48 val cars | p / v / nut | 1.12 / 1.11 / 1.12 (from seed means 0.0443 / 0.0707 / 0.0982 ÷ 0.0394 / 0.0635 / 0.0876 = 1.124 / 1.113 / 1.121; per seed DTT 0.0446 / 0.0728 / 0.0997 and 0.0439 / 0.0685 / 0.0967, ISLA 0.0387 / 0.0619 / 0.0868 and 0.0400 / 0.0651 / 0.0884) | same (Volume uses the fused geometry path, unaffected; checked at 0.9.0, not on code_dtt16) | **different code version** (pre-A2); every DTT seed above every ISLA seed on every field; not run at 0.9.0 | examples/cfd/isla/book/09-drivaerml.qmd:754-760 (tbl-interior-read-only + caption); examples/cfd/isla/research/dtt/REPORT.md:501-516, :20; isla-kwarg-audit-2026-10-17/REPORT.md:93; fused-path: isla-compile-af-2026-10-21/RESULTS.md:48 |
| 7 | AirfRANS cost at 0.9.0, ISLA-DTT vs ISLA (training, 4 GB300, bf16, 100k cells, 200 steps/epoch × 500) | median step; peak memory; wall; GPU-h | step 0.565 / 0.563 s vs 0.385 / 0.384 s (lanes' own metrics.jsonl) = **1.47×**; JOBS.md trainer medians 0.56 vs 0.38 s (smoke 0.58 vs 0.39); peak 39.4 vs 35.1 GiB (smoke only) = 1.12×; wall 16.6 / 16.5 h vs 11.1 / 11.2 h; GPU-h ≈ 66 vs ≈ 45 per lane (smoke estimate 71 vs 51) | – | compiled-path (pre-fix) lanes; post-fix smoke for AirfRANS ISLA only: 0.397 s, 35.1 GiB | isla-review-2026-10-21/scout/FACTS.md:14-15; isla-ref090-2026-10-20/JOBS.md:76-79, 129; PREREG.md:204-205 |

## Notes

**Row 1.** Eager seed means: ISLA 0.011940 / 0.0064229 / 0.015589, ISLA-DTT 0.014095 / 0.0074133 / 0.022345 (compiled
0.011857 / 0.0064361 / 0.015810 and 0.013787 / 0.0073839 / 0.022064). The nut move 1.396 → 1.433 is mostly
r090_af_dtt seed 42 (nut eager ÷ old 1.050, the largest single-cell move; IMPACT.md:38): compiled per-seed nut R
1.353 / 1.441 becomes 1.440 / 1.427 eager. Applied to AirfRANS alone, §12(f)'s bands read "ISLA clearly" on both paths
(PREREG_v2.md:266-268). r090fix: `grep -rl r090fix scratch/` finds launch/plan files only; no retrained AirfRANS score
exists in scratch (ref090 STATUS.md:1 ETA "about a day after the maintenance ends").

**Row 2.** E3b has an eager version (0.754 / 0.889 / 0.995; COMPILE-AF re-scored all ten E3b runs). E3b arm D seed
means compiled 0.013614 / 0.008613 / 0.022308 (= RESULTS_partial.md:92's "arm D 0.01361 / 0.00861 / 0.02231"), eager
0.013953 / 0.008678 / 0.022461; arm A compiled 0.018347 / 0.009708 / 0.022572, eager 0.018495 / 0.009766 / 0.022569.
The 2b comparison is not a matched pair: afu_dtt is 384/256 at 5e-4 on code_dtt16 against 512/512 at 3.75e-4 A2 ISLA
(kwarg audit REPORT.md:97-98, "unequal size"). Eager, afu_dtt's "worse than A2 ISLA" holds on v and nut, not p
(RESULTS.md:153).

**Row 4 (QUERY-POP-090, eager, seed means p / v / nut; s42, s43 in the source).** AirfRANS ISLA 0.9.0: half 1.11 / 1.12
/ 1.14 MATERIAL; 4 chunks 1.33 / 1.36 / 1.40 MATERIAL; sub1000 4.7 / 5.0 / 5.7 SEVERE; sub1 85 / 91 / 118 SEVERE; +50 %
same-law 0.95 / 0.95 / 0.95 not material; far-field 136 / 62 / 133 SEVERE; wall-heavy 20 / 30 / 22 SEVERE.
AirfRANS ISLA-DTT 0.9.0: roundoff in every mode (|r − 1| ≤ 4.3e-6, median d 2.6e-7 to 8.2e-7, both seeds).
DrivAerML Volume ISLA 0.9.0: half 1.01 / 1.01 / 1.02, chunks4 1.03 / 1.03 / 1.15 (not material), sub1000 4.5 / 4.0 / 5.5,
sub1 21 / 13 / 18, far 20 / 16 / 14, wall 6.2 / 4.5 / 4.5 (SEVERE). No 0.9.0 ISLA-DTT exists on Volume. GEO1
(pre-0.9.0) AirfRANS is similar to 0.9.0 ISLA. The A2 rung is "incomplete as registered". AeroZoo rows: not in the file
(QUERYPOP090_FOR_12F.md:4-5, 22-32).

**Row 5.** R5 has not been read: ATTRIB-090 RESULTS.md holds only Tensile2d (§1: R4, the CONT → 0.9.0 step is not real;
QEMB 1.223 the one solid effect). The R5 arms (DTT-CONT and the CONT partner, retrained under the fix) and the
r090fix AirfRANS bases they wait on are still training.

**Row 6.** Same pre-A2 report, other context: on the airfoil (old block draws, 384/256 both) the forest read 0.74 /
0.83 / 1.03 of ISLA (research/dtt/REPORT.md:517; book 14-program-status.qmd:563); AF-UNIFORM (DTT 384/256 code_dtt16
vs ISLA 512/512 code_v20) 0.96 / 0.99 / 1.12 (kwarg audit REPORT.md:97). Pre-A2 DTT cost on the car: 1.72 s per step,
194 GB per rank, 3.7× the read-only floor (REPORT.md:518).

**Row 7.** §13(b) asks for cost "from the lanes' own logs". Step time and wall come from the r090_af lanes'
metrics.jsonl (review scout); peak memory is only reported from the smokes (JOBS.md:79: "Memory as in the smokes"); the
GPU-h figures are wall × 4 (scout) or the smoke formula (PREREG §9a). Lug for comparison (smokes / trainer medians):
ISLA 0.15 s, ISLA-DTT 0.42-0.44 s (≈ 2.8×), peak 8.5 vs 9.9 GiB, GPU-h 63 vs 166 (smoke estimate). AirfRANS ISLA-DTT is
cheaper relative to ISLA (1.47×) than on AeroZoo (1.5-2.0× per §12(f) edit ii). All AirfRANS figures are pre-fix lanes.

**Principled case, as §13(b) words it (PREREG_v2.md:289-290).**
- For ISLA-DTT: exact query independence (QUERY-POP-090 measures its size: roundoff for DTT against 1.1-1.4× for half or
  chunked queries and 5-136× for small or companion-heavy queries for ISLA on AirfRANS).
- Against: the canonical frame's axis choice on near-symmetric bodies, its constants, and θ; θ's error vanishes as
  θ → 0, which the knob rule credits. (PROBES-090 beside it: ISLA-DTT moves 1-6 % under exact refinement through its
  θ = 1 tree read, where interacting ISLA is invariant; isla-probes090-2026-10-21/RESULTS.md:143-145.)

## Contradictions and caveats found in the sources

1. The label "eager function of weights trained under the defect" is PROBES-090's wording (QUERYPOP090_FOR_12F.md:16-17;
   isla-probes090-2026-10-21/RESULTS.md:8-9). COMPILE-AF's RESULTS.md §7 and IMPACT.md do not use it; they say the
   re-scores are "a statement about inference only: the training path was also perturbed" (RESULTS.md:75-77). Same
   meaning, different words.
2. The kwarg audit (REPORT.md:98) divides afu_dtt by a different A2 ISLA partner (H-RESID P5, 0.0179 / 0.0095 / 0.0213)
   and gets 1.12 / 1.25 / 1.27; §13(e)'s partner afu_a2s (nut 0.02230) gives 1.109 / 1.258 / 1.213 compiled. Use afu_a2s.
3. Three "pre-0.9.0 AirfRANS ISLA-DTT ÷ ISLA" numbers circulate and differ by protocol and size: 0.74 / 0.83 / 1.03
   (pre-A2 block draws, 384/256), 0.96 / 0.99 / 1.12 (AF-UNIFORM, unequal sizes), 0.742 / 0.887 / 0.988 (E3b, matched
   512/512, the one §13 cites). None contradicts §13, but they should not be mixed in the table.
4. The pre-A2 Volume 1.12 / 1.11 / 1.12 is hard-coded in the book's prose (scout isla-book-overhaul-2026-10-18/scouts/
   s1-method-protocol-use.md:470 flags "no artifact"), but 09-drivaerml.qmd's table caption gives the per-seed values
   and cites research/dtt/REPORT.md:510-516, which reproduce it (1.124 / 1.113 / 1.121).
5. No number quoted in the task contradicts its source: 1.163 / 1.147 / 1.396, 1.181 / 1.154 / 1.433, 0.742 / 0.887 /
   0.988, 0.01996 / 0.0119 / 0.02706, 0.01799 / 0.00946 / 0.02230, 0.01752 / 0.01150 / 0.02560, 0.971 / 1.210 / 1.146,
   0.38 / 0.56 s all recompute from the per-seed values to the quoted digits.

## Updates (AZ-DECIDE, 2026-10-22; from the supervisor's 07:10 and 10:10 notes; these supersede the rows above)

- **Row 1-fix (now exists).** Fix-trained ISLA-DTT ÷ fix-trained ISLA on AirfRANS, both eager-scored on the same 200 cases:
  **1.035 / 1.126 / 1.381** (seed-paired p 0.981 / 1.097, v 1.105 / 1.148, nut 1.264 / 1.501). Label: fix-trained. Sources:
  r090fix_af_isla in isla-ref090-2026-10-20/r090_final_tables.md:106; r090fix_af_dtt (scorer 330338) in
  isla-attrib090-2026-10-21/RESULTS.md §3 row B (0.01265 / 0.00713 / 0.02149). The AirfRANS standing will be the
  eager-trained r090eag rows (about 10-23); use them if scored by the read.
- **Beside row 1: what the compiled-path bug cost.** Fix-trained ÷ bug-trained, both eager: ISLA-DTT 0.898 / 0.961 / 0.962,
  ISLA 1.024 / 0.985 / 0.998. So row 1's 1.181 on pressure overstated ISLA's lead; with the fix it is 1.035.
- **Row 5, ATTRIB-090 R5 (now read).** ρ_CONT = DTT-CONT ÷ CONT = 0.895 on pressure (≤ 0.95), so the reversal happened in
  the 0.9.0 bundle: at CONT, ISLA-DTT led ISLA by about 10 % on pressure. Both forms gained from the bundle, ISLA more:
  0.9.0 ÷ CONT is 0.778 / 0.739 / 0.709 (ISLA) and 0.901 / 0.856 / 0.977 (ISLA-DTT). NOSC removes the scales and also feeds
  the raw, unstandardized speed, so ISLA's gain lies in the scales or the input standardization; ATTRIB-090's NOSC-STD
  separates them. DTT-NOSC (about 10-22 20:00 EDT) says whether ISLA-DTT's smaller gain comes from the same place. Source:
  isla-attrib090-2026-10-21/RESULTS.md §3.
- **AeroZoo exposure to the compiled-path fault: not settled** (reviewer cert.md item 2: ISLA's compiled-vs-eager gap on
  AeroZoo is 4.9e-4, above roundoff). A diagnostic (cmpdiag/, sub-lane 26432b9c) runs before the §12(f) read.
- **Principled weighing, a defect on ISLA-DTT's side:** tree_read.py:178 assigns normal octants with `> 0`, which is not
  antipodal for zero components, so the forest partition is not flip-symmetric on axis-aligned faces (supervisor verified
  the line). ATTRIB-090 is testing whether interacting ISLA is exactly flip-invariant (by negating weights); add it when it
  lands.
- **New row (context): AF-EXTRAP, AirfRANS Reynolds extrapolation** (penalty = reynolds-test error ÷ in-distribution
  error): ISLA-DTT about 2.4 / 2.4 / 2.6 against ISLA 3.2 / 3.1 / 2.9; absolute reynolds-test errors 0.0335 / 0.0175 /
  0.0590 against 0.0389 / 0.0195 / 0.0457 (provisional). Label: fix-trained; data-size control pending. Source:
  AFX_FOR_13.md in this folder (when it lands).
