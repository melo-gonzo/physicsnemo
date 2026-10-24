# DK-GAP: where ISLA-DTT trails interacting ISLA (summary; read 2026-10-24 00:20 EDT; revised after REVIEW.md 2026-10-24 01:02 EDT)

Runs: AirfRANS r090eag_af_{dtt,isla}_seed{42,43} (eager-trained), lug r090fix_lug_{dtt,isla}_seed{42,43} (fix-trained).
All numbers: the scorer's per-case relative L2 (MetricCalculator, training space), every config row epoch 500. Ratio =
ISLA-DTT / ISLA of the two-seed means; [s42, s43] = seed-matched ratios. Full tables: `gap_table_fp32.md` (float32, the
scorer's protocol: job 363147), `gap_table_bf16.md` (precision=bfloat16, job 363156), `rough.md` (cpu job 363365),
`curves.md` (monitoring only). Everything in `gap.json` (md5 44b84474; rebuilt byte for byte from its four parts by
`rebuild_gap_json.py`). Scripts: `probe_gap.py`, `probe_gap.sbatch`, `probe_gap_prec.sbatch`, `reduce_gap.py`,
`roughness.py`, `curves.py`, `rebuild_gap_json.py`; `check_revision.py` recomputes the seed-matched values below.

**Gate (COMMON_RULES), passed before any other number was read:** at θ 1, leaf 4, the float32 harness reproduces every
per-run standing score (AirfRANS DTT 0.012428 / 0.0123887, 0.00737341 / 0.00697252, 0.0207112 / 0.0204842: max rel 1.8e-6;
AirfRANS ISLA max rel 3.5e-6; lug DTT 0.00221933 / 0.00183431 against the standing's 0.002219 / 0.001834, its four-digit
rounding; the lug ISLA has no per-seed standing: its two-seed mean 0.00145367 against 0.0014537), each case against
the scorer's record (max rel 3.7e-6 AirfRANS DTT, 0 AirfRANS ISLA, 8.4e-6 lug), the per-case metric recomputed from the
saved float32 arrays (max rel 2.4e-7), and the four runs of a family score the same points and targets (hash, every case).

## 1. Fit against generalization (test = the scored set; train = the first 200 / 600 training cases, same protocol)

The AirfRANS train subset covers Re 2.0-3.0e6 only (the test set 2.0-6.0e6). The test's low-Re quartile (Re 2.04-2.96e6)
reads 0.996 / 1.128 / 1.179, so the train ratios still match or exceed the test's at matched Re; the subset is not
representative of the test's Re range.

| | test fp32 | train fp32 | test bf16 | bf16-trained curves (train / val) |
|---|---|---|---|---|
| AirfRANS pressure | 0.994 [0.944, 1.050] | 0.987 [0.934, 1.052] | 0.954 [0.904, 1.009] | 0.93 / 0.97 |
| AirfRANS velocity | 1.129 [1.145, 1.112] | 1.083 [1.100, 1.065] | 1.074 [1.081, 1.067] | 1.06 / 1.08 |
| AirfRANS nut | 1.342 [1.316, 1.369] | 1.413 [1.456, 1.361] | 1.292 [1.200, 1.391] | 1.33 / 1.30 |
| lug von Mises | 1.394 [1.518, 1.270] | 1.434 [1.517, 1.347] | **1.043 [1.096, 0.989]** | 1.03 / 1.04 |

The gap is on the training cases as much as on the test cases: fit, not generalization, on both datasets. On the lug
the float32 gap (1.394; ISLA-DTT's own seeds differ by 0.83) is real under the standing's protocol and sits below the
bf16 read-out noise (~0.0033, twice the float32 errors), so at bf16 the forms read level (1.043). That is mostly
arithmetic: adding ISLA's per-case bf16 excess in quadrature to ISLA-DTT's float32 error predicts bf16 ratios 1.114 /
1.048 (observed 1.096 / 0.989); whether bf16 training causes the float32 gap is untested. At bf16 ISLA-DTT is still worse
in 73 % of cases (per-case median 1.06). bf16 / fp32 error: ISLA 2.29 / 2.33, ISLA-DTT 1.65 / 1.81 (lug); AirfRANS
1.09-1.39 for both. The lug runs trained with `compile_blocks: true` at 20,000 sampled cells (`resolved_config.yaml`);
the probes read eager at 200,000; the in-training validation on the test split (bf16, 20,000) also reads 1.04.

## 2. Per case (test, fp32)

| | per-case ratio q10 / q50 / q90 | DTT worse in | top-10 % of cases' share of the gap | where larger |
|---|---|---|---|---|
| AirfRANS pressure | 0.70 / 0.97 / 1.38 | 48 % | (no net gap) | - |
| AirfRANS velocity | 0.85 / 1.04 / 1.43 | 58 % | 0.88 | aoa -0.3..4.8°: 1.28; Re flat (1.10-1.15) |
| AirfRANS nut | 0.91 / 1.34 / 2.13 | 82 % | 0.52 | low aoa: 1.61 (< -0.5°) falling to 1.09 (> 8.9°); Re flat-ish |
| lug von Mises | 1.15 / 1.48 / 1.93 | 96 % | 0.22 (spread) | low load 1.64 -> high load 1.15; low plastic fraction 1.70 -> 1.11 |

The ratio falls where ISLA's own error is large (Spearman of the log ratio with ISLA's error: lug -0.62, AirfRANS nut
-0.25): ISLA-DTT carries a roughly additive error floor rather than a proportional penalty.

## 3. Per region (test, fp32; pooled RMS ratio [s42, s43]; share of the squared-error gap of the seed means)

AirfRANS seed noise (s43 / s42 of each form's pooled region error) is 0.87-1.16 in most cells, but ISLA-DTT's own seeds differ by
1.72 beyond one chord, 1.91 for nut below the 90th percentile and 1.24 for nut in the wake (ISLA 0.94-1.05 there), all
in the nut column. The regions split squared error, which weights the high-error cases: in seed 42 the 20 cases with
ISLA's largest nut error carry 87 % of its squared error and read 0.98.

| AirfRANS (chord units) | points | pressure | velocity | nut |
|---|---|---|---|---|
| all points | 100 % | [0.95, 1.06] | [1.18, 1.16] | [1.03, 1.33]; squared gap 0.010 / 0.123 |
| d < 0.01 (near wall) | 47 % | 0.98 [0.92, 1.04]; -2.19 (DTT better) | 1.00 [1.02, 0.98]; +0.01 | 1.20 [1.19, 1.21]; +0.05 |
| wake strip (d >= 0.01) | 10 % | 1.44 [1.44, 1.45]; +1.36 | **1.97 [1.95, 1.99]; +0.60** | 1.19 [1.02, 1.35]; +0.94 (95 % of it from s43) |
| outer (rest) | 43 % | 1.12 [1.08, 1.16]; +1.83 | 1.35 [1.32, 1.38]; +0.39 | 1.04 [1.01, 1.07]; +0.01 |
| d >= 1 chord | 8 % | 1.41 [1.35, 1.49] | 1.72 [1.66, 1.79] | **2.05 [1.48, 2.47]** |
| nut below the case's 90th percentile | 90 % | 1.00 [0.94, 1.06] | 1.12 [1.13, 1.11] | 2.19 [1.48, 2.67] |

| lug | nodes | von Mises fp32 (ratio; gap share) | bf16 |
|---|---|---|---|
| on the surface (d < 1e-6 L) | 34 % | **1.51 [1.69, 1.30]; +0.90** | 1.11 [1.19, 1.03]; +1.63 |
| interior, 0.03-0.1 L | 46 % | 1.03; +0.05 | 0.96 |
| interior, >= 0.1 L | 20 % | 1.07; +0.04 | 1.00 |
| nearest part fixed / hole / loaded / free | 5 / 5 / 5 / 85 % | 1.16 [1.33, 0.99] / 1.29 / 1.31 / 1.27 | 1.07 / 1.04 / 1.03 / 1.02 |
| elastic / above yield | 96 / 4 % | 1.24 / 1.24 | 1.03 / 1.05 |

AirfRANS: the velocity gap sits away from the airfoil, in the wake and the far field, in both seeds, where ISLA-DTT's far
points read the boundary through θ = 1 clusters and interacting ISLA's points share the slices. Nut is above 1 beyond
one chord in both seeds, but its wake share rests on seed 43 (wake nut 1.02 / 1.35; seed 43 carries 92 % of the pooled
nut gap). Near the wall nut is 1.20x in both seeds but carries only 5 % of the nut gap (the 1e-3..1e-2 band 1.25 / 1.22),
so near-wall resolution explains little of it. The same pattern holds at bf16 and on the training cases. Lug: the gap is
on the surface nodes, 1.27-1.31 near the hole, loaded and free parts (fixed 1.16 [1.33, 0.99]), and the same in the
elastic and plastic zones.

## 4. Roughness of the difference (`rough.md`; 0 = the same at neighbouring points, 1 = uncorrelated)

AirfRANS: the ISLA-DTT minus ISLA prediction is smooth (0.003-0.03) in every region except nut beyond 0.1 chord outside
the wake (0.076 at 0.1-1, 0.112 beyond one chord), which is where the nut gap sits; still smooth relative to 1: a
coherent bias of the wake and far field, not per-point noise from each point's own cluster list. Lug: both forms' errors and
their difference are rough at node scale (0.66-0.88, against the target's 0.03-0.05): noise-like, as a sub-bf16 floor.
