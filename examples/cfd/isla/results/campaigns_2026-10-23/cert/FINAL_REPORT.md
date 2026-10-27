# CERT final report (2026-10-27, written 04:03 EDT)

Status: DONE. Of the nine families, only DeepJEB++ certifies on the registered Part 2 reading.

**What ran.**
- Code: psharpe/isla-014 8097ed00f, torch 2.13, through md5-verified LHR snapshots `$T/kernel/cert1027/code_8097` and
  `recipe_8097`.
- Plan: CERT_PLAN.md, with Part 1 approved with amendments A1 (the routing gradients are their own role) and A2 (the
  vol checkpoint), and Part 2 (section 9) approved with amendment 9a (joint calibration).
- Part 1 jobs: 417520, 417521, 417677, 417678, 417694.
- Smokes: 417914, 417915, 417933.
- Part 2 jobs: 418147, 418182, 418614, 418615.
- CPU dry runs: 417167-417169, 417425.
- **16.8 GPU-h** of the 30 cap. No commits; the scratch folder is git-ignored. Details are in JOBS.md and RESULTS.md.

## Verdicts

| family | checkpoint (final epoch) | Part 1 (registered) | Part 2 (null-calibrated, confirmatory) | max d(comp64, ref) |
|---|---|---|---|---|
| AeroZoo (az) | azc_full_dtt_seed42 (250) | not certified | not certified: G2' R 1.016 > 1.004 (fwd9.params c1) | 3.8e-14 |
| AirfRANS (af) | afwhole_af_dtt_seed42 (500) | not certified | not certified: G2' N 3 > 1, R 1.030 > 1.003 (fwd11/fwd5 routing) | 7.2e-13 |
| DrivAerML Volume (vol) | r090_vol_isla_seed42 (500) | not certified | not certified: G1' (6.45 > 3.45), G2' N 12, R 1.60 (blocks 3-4 routing) | 1.3e-13 |
| lug | r090fix_lug_dtt_seed42 (500) | not certified | not certified: G2' R 2.005 (fwd2.routing c1) | 7.5e-15 |
| Tensile2d (t2d) | r090fix_t2dz_isla_seed42 (500) | not certified | not certified: G2' R 11.4 (fwd2.routing c0), G3' 2 groups (1.5e-12) | 1.5e-12 |
| DrivAerML Surface (drs) | slicenormalx_dra_ctl_seed42 (500) | not certified | not certified: G1' N1 3 > 2, G2' R 1.165 (sfwd0.routing) | 5.8e-14 |
| HiLift (hl) | hldupx_hl_isla_seed42 (500) | not certified | not certified: G1' R1 2.64 > 2.32, G2' R 1.331 (sfwd0.routing) | 4.4e-14 |
| SHIFT-SUV (suv) | r090_suv_isla_seed42 (500) | not certified | not certified: G2' N 8, R 1.257 (sfwd0/2 routing), G3' 3 groups (up to 3.3e-12) | 3.3e-12 |
| DeepJEB++ (djb) | r090_jb_isla_seed42 (40) | not certified | **certified** (G1' 1 / 1.49 within 4 / 2.24; G2' 0 / 0.987; G3' 0) | 8.0e-15 |

**Keys.** Every family loads every `blocks.*` key. On every checkpoint the 12 `blocks.*.geo_logit.bias` keys are
unexpected; on the ISLA-DTT runs (az, af, lug) so is `kernel_basis.net.4.bias`. Both were removed in 0.14.0. What stays
at its initial weights:
- az: `head.weight` and `head.bias` (a 9-output checkpoint head against the 7-field config's 11);
- vol: `kernel_basis.*`, `kernel_reads.*`, and the inlet, no_slip, outlet and slip seeds. FLOOR-DRV's arm F also lacks
  the kernel-read keys (dry run 417425), so A2 kept r090_vol_isla_seed42;
- t2d: `kernel_basis.*` and `kernel_reads.*`;
- hl: the inlet, outlet and symmetry seeds;
- suv: the floor, inlet, outlet, roof and side_walls seeds;
- af, lug, drs and djb: nothing.

The excesses are systematic. With fresh seeds and fresh compile processes, Part 2 repeats Part 1's worst groups. In
those routing gradients Inductor's rounding lands further from float64 than eager's, while in 25-111 other groups per
family the compiled path is closer to float64 than eager. The size of the excess varies widely: 1.02-1.03x on az and
af, 1.1-1.3x on the three surfaces, 1.6x on vol, 2.0x on the lug and 11x on t2d.

The gates' own false-failure rate is in `out/cert/NULLCHECK_part2.md`. G2' passes eager against eager 93-100% of the
time. G1' passes only 77-100% (mean 87%), because its thresholds are discrete. So the combined null false-failure rate
is about 10-15%, not the 10% intended. This affects only drs, hl and vol, and each of them also fails G2'.

The SUV and t2d float64 flags sit in routing groups whose own float64 one-ulp change is 0.3-2e-12. I read them as
amplified reordering roundoff, not a bug. No code fix is indicated.

## Timing (4 ranks, 20-step smokes, `iter_time_s` median over steps 5-19, s)

| | vol | az full | af | lug | t2d | drs | suv | hl | djb |
|---|---|---|---|---|---|---|---|---|---|
| compiled | 1.155 | 0.846 | 0.332 | 0.285 | 0.277 | 0.238 | 0.242 | 0.240 | 0.118 |
| eager | 1.359 | 1.072 | 0.428 | 0.363 | 0.369 | 0.371 | 0.375 | 0.365 | 0.178 |
| saving | 15% | 21% | 22% | 22% | 25% | 36% | 35% | 34% | 34% |

## LANES.tsv proposal (the supervisor edits LANES.tsv; I did not touch it)

On the registered Part 2 verdict:
- **switch to `model.compile_blocks=false`**: fin_vol_dtt (2 lanes), the 14 fin_az_*_dtt lanes, fin_af_dtt (2),
  fin_lug_dtt (2), fin_drs_isla (2), fin_suv_isla (2) and fin_hl_isla (2);
- **keep compiled**: fin_djb_isla (2);
- **keep eager**: fin_t2d_dtt (4).

Priced on the smokes with PRICE's steps and overhead factors (`out/PRICE_CERT.md`), ISLA training rises from 3,216 to
4,031 GPU-h, **+815 GPU-h (+25%)**. The increase by family:

| family | extra GPU-h |
|---|---|
| AeroZoo | +635 |
| lug | +59 |
| SHIFT-SUV | +31 |
| vol | +27 |
| HiLift | +22 |
| af | +22 |
| drs | +18 |

**Alternative, for the supervisor's judgment, not a verdict.** az and af fail by 1.6% and 3% in at most three routing
groups, and are closer to float64 than eager in 68-111 groups. Keeping those two compiled would save about 657 of the
815 GPU-h. That would need a ruling that a systematic excess of a few percent is acceptable. The registered gate,
being one-sided, says it is not.

## For project memory
- Inductor's compiled block region systematically worsens a few bf16 routing-gradient groups and improves many
  others. Every gate of the form "no group worse than eager's own one-ulp spread" fails it. The size of the excess,
  not its presence, is what separates the families (from 1.02x on az to 11x on t2d).
- An eager-against-eager null is needed for any per-group equivalence gate. Part 1's gate failed eager itself.
- The 4-rank compiled savings at 8097ed00f are 15-36%. The PRICE estimates for AeroZoo understated the step by 11%.

## Left and on disk
- Nothing is running. The registry row is removed. The LHR per-job outputs are deleted after an md5 check.
- On LHR, `$T/kernel/cert1027/` holds 65 MB: the read-only snapshots, the tools, the queue files and the job logs. The
  snapshots can be reused to re-certify a family if an AirfRANS piece is adopted.
- Locally, this folder holds 77 MB: gzipped jsonl in `out/` (part 1, part 2, dry run), the smoke metrics and logs, and
  the summaries.
- No worktree was made.
