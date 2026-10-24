Status: DONE
Revised after REVIEW.md, 2026-10-24 01:02 EDT (original: `REPORT_v1.md`; changes line by line: `REVISION_NOTE.md`).

**DK-GAP (stage 1d; read 2026-10-24 00:20 EDT).** ISLA-DTT's gap to interacting ISLA is a fit gap, not a generalization
gap. On the lug the float32 gap (1.39) is real under the standing's protocol and sits below the bf16 read-out noise, so
at bf16 the forms read level (1.043); whether bf16 training causes it is untested. On AirfRANS the velocity gap is a
smooth bias of the wake and far field in both seeds; where the nut gap sits rests mostly on seed 43.

**What ran** (LHR `$T/kernel/dttknobs_gap`, code_cmpaf_fix + recipe_r090, eager): smokes 362963 and 363025; production
363147 (float32, the scorer's protocol, test and train); 363156 (precision=bfloat16, test); cpu-short 363365 (roughness).
**1.51 GPU-h of 5**; nothing running, no slot held, no commits (scratch only); `JOBS.md`. **Gate passed:** every per-run
standing score reproduced (AirfRANS max rel 1.8e-6 for ISLA-DTT, 3.5e-6 for ISLA; lug 0.00221933 / 0.00183431 against
0.002219 / 0.001834; the lug ISLA has no per-seed standing, only its mean 0.0014537, matched), each case matches the
scorer's record (≤ 8.4e-6), the four runs of a family score the same targets (hash). `rebuild_gap_json.py` rebuilds
`gap.json` from its four parts byte for byte (md5 44b84474).

**Results** (ISLA-DTT ÷ ISLA of the two-seed means [seed 42, seed 43]; `gap_table.md`, full tables beside it, everything
in `gap.json`; the revision's numbers from `check_revision.py`):
- **Fit, not generalization.** On the first 200 / 600 training cases the ratios are AirfRANS 0.987 / 1.083 / 1.413
  (test 0.994 / 1.129 / 1.342) and lug 1.434 (test 1.394). The AirfRANS subset covers Re 2.0-3.0e6 only (test 2.0-6.0e6):
  the test's low-Re quartile (0.996 / 1.128 / 1.179) agrees, but the subset is not representative.
- **Lug: below the bf16 read-out noise.** bf16 1.043 [1.096, 0.989] against float32 1.394 [1.518, 1.270] on the same
  weights. The bf16 read-out error (~0.0033, twice the float32 errors) compresses any gap this size: ISLA's per-case bf16
  excess added in quadrature to ISLA-DTT's float32 error predicts 1.114 / 1.048 (observed 1.096 / 0.989). So the level
  bf16 read is mostly arithmetic, not evidence that bf16 training causes the gap; at bf16 ISLA-DTT is still worse in 73 %
  of cases (median 1.06) and on the surface (1.11 [1.19, 1.03]). In float32 the gap covers 96 % of the cases and sits on
  the surface nodes (1.51×, 90 % of it; interior 1.03–1.07×), 1.27–1.31× by the hole, loaded and free parts (fixed 1.16
  [1.33, 0.99]), alike in the elastic and plastic zones; noise-like from node to node. Confound: the lug runs trained with
  `compile_blocks: true` at 20,000 cells, the probes ran eager at 200,000; the in-training validation also reads 1.04.
- **AirfRANS: wake and far field.** Velocity is 1.97× in the wake (1.95 / 1.99) and 1.72× beyond one chord. Nut is 2.05×
  beyond one chord (1.48 / 2.47; ISLA-DTT's own seeds differ there by 1.72×). The regions split squared error, which
  weights high-error cases: there seed 42's nut ratio is 1.03 (squared gap 0.010), seed 43's 1.33 (0.123), so "the wake
  carries 94 % of the nut gap" rests on seed 43 (wake nut 1.02 / 1.35); seed 42's per-case 1.32 sits in its low-error
  cases. Near the wall (d < 0.01 chord) nut is 1.20× in both seeds but ≤ 5 % of the gap (pressure 0.98, velocity 1.00).
  The difference is smooth (roughness ≤ 0.03: a coherent bias, not per-point tree noise) except nut beyond 0.1 chord
  (0.076, 0.112), where its gap sits. It holds at bf16 (0.954 / 1.074 / 1.292) and on the training cases.
- **Per case.** The ratio falls where ISLA's own error is large (AirfRANS nut 1.61 at low angle of attack to 1.09 above
  8.9°; lug 1.64 at low load to 1.15 at high load), so ISLA-DTT has a roughly additive error floor. The training curves
  (`curves.md`, monitoring only) stop at the StepLR schedule (learning rate 3.75e-8 after epoch 400) for both forms.

**The mirrors** (ATTRIB-090 RESULTS.md §3c, md5 e01a94de, read from the file). Neither DTT-SDF (1.040 / 1.043 / 1.009
÷ B) nor DTT-QEMB (0.983 / 0.994 / 0.949) moves any field. C_f = +0.131 / +0.047 / +0.121 stays unattributed (ψ ≤ 0.34
for the stored SDF and ≤ 0.77 for QEMB on pressure; σ 0.042 / 0.029 / 0.064). My reading, not a measurement: the 0.9.0
gap is mostly not a bundle effect (nil on pressure, where C_f is largest; on velocity and nut a fit bias of far points,
which no bundle piece targets). But C_f is internal to ISLA-DTT (fix-trained, DTT-NOSC against DTT-CONT), and C_f nut
0.121 could be ~40 % of log(1.342) = 0.294. I expect DTT-NOSC-STD to read no effect (p ≈ 0.6).

**Ranked stage-2 candidates** (evidence in `HYPOTHESES.md` §Stage-2):
1. Train the lug pair in float32 (or with a float32 read path): tests whether bf16 training causes the float32 gap;
   p ≈ 0.55 that it closes it.
2. Train AirfRANS at a smaller θ (0.5, or the exact read), if DK-SENS shows the far field is sensitive to θ: p ≈ 0.35.
3. Interior coupling that stays query independent: the slices also written by a request-free interior quadrature.
   A larger build, and it needs a design with no free constant.
4. DTT-NOSC-STD: closes the bundle thread, but cannot explain where the gap sits.
5. Read width: deferred to DK-CAP.
Not worth a retrain: near-wall changes (≤ 5 % of any field's gap), or anything aimed at generalization. Confidence: high
for fit against generalization and for where the velocity gap sits (two seeds, two precisions, both splits); medium for
nut (above 1 beyond one chord in both seeds, its wake share from seed 43); the causes behind 1-3 are not separated here.

**For project memory:** the lug's float32 errors (~0.0015-0.002) sit below its bf16 read-out noise (~0.0033), so a bf16
read compresses a float32 form gap there toward 1 and cannot by itself show what bf16 training did. **Stays on disk:**
`$T/kernel/dttknobs_gap/out` (4.1 GiB, float32 predictions and geometry) and `out_bf16` (1.3 GiB), kept for DK-SENS's
and stage 2's regional comparisons; delete them when stage 2 is planned. The local folder is about 1 MB.
