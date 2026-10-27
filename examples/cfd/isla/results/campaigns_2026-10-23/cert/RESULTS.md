# CERT results: the compiled path (compile_blocks) of every ISLA family on the staged 0.14.0 code

Written 2026-10-27 04:00 EDT. Code: psharpe/isla-014 8097ed00f, with the LHR snapshots `code_8097` and `recipe_8097`
(md5-verified, see JOBS.md). Torch 2.13.0+cu130, GB300. Plan: CERT_PLAN.md. Sections 1-6 are Part 1, sections 7-8
its amendments, and section 9 / 9a Part 2. The supervisor approved each part before its jobs ran. Every process
exited with rc 0.

## 1. Part 1 (registered reading, seeds 0 and 1): no family certified

Jobs 417520, 417521, 417677, 417678 and 417694. Full tables: `out/cert/SUMMARY_part1.md`. The gate is pooled groups
over 4 roles (out / in / routing / params) and 2 cases. G1 is the fp32 rule per process, d(comp) <= d(eager) + the
mean one-ulp change. G2 is the bf16 family rule: the median of 16 compiled samples <= the maximum of 16 eager samples.
G3 is float64 <= 1e-12. Each cell gives the failing groups.

| family | G1 fp32 (p0, p1) | G2 bf16 (largest compiled median / eager max) | G3 f64 | groups below the eager min (more accurate) |
|---|---|---|---|---|
| az | 1, 2 | 1 (1.02x) | 0 | 111 |
| af | 3, 3 | 3 (1.02x) | 0 | 68 |
| vol | 4, 4 | 15 (1.61x) | 0 | 67 |
| lug | 0, 0 | 1 (1.94x) | 0 | 96 |
| t2d | 3, 2 | 4 (9.78x) | 1 | 99 |
| drs | 3, 3 | 2 (1.13x) | 0 | 61 |
| hl | 2, 2 | 2 (1.20x) | 0 | 42 |
| suv | 1, 1 | 5 (1.27x) | 2 | 25 |
| djb | 1, 1 | 0 (0.98x) | 0 | 50 |

- **Null calibration** (`out/cert/NULLCAL_part1.md`, beside, does not gate): this gate fails eager judged against
  itself.
  - G1: an eager one-ulp draw standing in for the compiled run fails 0.1-15 groups on average. 12-100% of draws fail at
    least one group.
  - G2: eager against eager, 8 samples against 8, fails 0-11 groups, at up to 1.18x (af), 1.51x (djb) and 2.37x (t2d).
- The supervisor ruled at 02:55 EDT: the Part 1 verdict stands, it draws neither "all eager" nor a post-hoc pass, and a
  confirmatory, null-calibrated Part 2 runs on fresh data.
- Per tensor (FUSED-PAD's / C4's reading, beside): G1 fails 13-34 of 600 or 1,152 groups per process in fp32 and
  4-34 in bf16. Float64 above 1e-12 per tensor: af 1 (1.1e-12), drs 1, hl 1, t2d 2, suv 5-6 (up to 8.5e-12).

## 2. Speed: the 4-rank smokes (417914, 417915, 417933)

20 train and 3 val steps, bf16, LANES overrides, `iter_time_s` median over steps 5-19. The runs went compiled, eager,
compiled, each with a cold cache. Table: `out/smoke/SMOKE_TABLE.md`.

| family | compiled (c.3) s | eager (e.2) s | first compiled run (c.1) s | saving | peak GiB compiled / eager |
|---|---|---|---|---|---|
| DrivAerML Volume (vol) | 1.1547 | 1.3585 | 1.1882 | 15.0% | 142.4 / 169.8 |
| AeroZoo full (az) | 0.8461 | 1.0717 | 1.0041 | 21.1% | 131.7 / 159.2 |
| AirfRANS (af) | 0.3317 | 0.4276 | 0.3386 | 22.4% | 27.0 / 31.9 |
| lug | 0.2848 | 0.3626 | 0.2885 | 21.5% | 7.6 / 8.9 |
| Tensile2d (t2d) | 0.2769 | 0.3689 | 0.2834 | 24.9% | 4.3 / 4.8 |
| DrivAerML Surface (drs) | 0.2379 | 0.3712 | 0.3496 | 35.9% | 66.0 / 84.8 |
| SHIFT-SUV (suv) | 0.2421 | 0.3746 | 0.2534 | 35.4% | 65.8 / 84.5 |
| HiLift (hl) | 0.2397 | 0.3653 | 0.2386 | 34.4% | 61.9 / 79.5 |
| DeepJEB++ (djb) | 0.1183 | 0.1784 | 0.1201 | 33.7% | 17.1 / 21.7 |

Every family saves at least 10%, so all 9 entered Part 2, Tensile2d included. Two of the first compiled runs read
slower: drs c.1 (page cache) and az c.1.

## 3. Part 2 (confirmatory, null-calibrated, seeds 2 and 3): only DeepJEB++ certifies

Jobs 418147, 418182, 418614 and 418615. Each family ran 2 new cold processes, giving 16 compiled and 32 eager draws.
Full output: `out/part2/PART2_VERDICTS.md` and `.json`. G2' compares the compiled N and R (medians over 2,000
reference subsets) with the joint 95% thresholds (cN, cR) of 2,000 eager 16+16 splits. G1' compares the compiled
maxima over the two processes with the joint 95% thresholds over 256 eager stand-in pairs. G3' is
d(comp64) <= max(1e-12, eager's float64 one-ulp change).

| family | G1' fp32: N1 / R1 (cN1 / cR1) | G2' bf16: N / R (cN / cR) | G3' failing | verdict | groups above the eager max, largest first |
|---|---|---|---|---|---|
| az | 2 / 1.45 (4 / 2.16) | 1 / 1.016 (1 / 1.004) | 0 | not certified (G2' R) | fwd9.params c1 1.016 |
| af | 3 / 2.81 (5 / 3.04) | 3 / 1.030 (1 / 1.003) | 0 | not certified (G2') | fwd11.routing c1 1.027, fwd5.routing c0 1.015, fwd5.routing c1 1.014 |
| vol | 4 / 6.45 (3 / 3.45) | 12 / 1.603 (1 / 1.009) | 0 | not certified (G1', G2') | fwd3.routing c1 1.60, fwd4.routing c1 1.50, fwd4.routing c0 1.23, fwd0.routing c1 1.22 |
| lug | 0 / 0.55 (19 / 2.89) | 1 / 2.005 (1 / 1.005) | 0 | not certified (G2' R) | fwd2.routing c1 2.005 |
| t2d | 3 / 1.53 (8 / 2.61) | 7 / 11.44 (1 / 1.002) | 2 | not certified (G2', G3') | fwd2.routing c0 11.4, fwd0.routing c0 2.95, fwd2.in c0 2.47 |
| drs | 3 / 1.70 (2 / 1.51) | 2 / 1.165 (1 / 1.000) | 0 | not certified (G1', G2') | sfwd0.routing c1 1.165, sfwd0.routing c0 1.098 |
| hl | 2 / 2.64 (3 / 2.32) | 2 / 1.331 (1 / 1.000) | 0 | not certified (G1' R, G2') | sfwd0.routing c0 1.331, sfwd0.routing c1 1.250 |
| suv | 1 / 1.56 (4 / 2.87) | 8 / 1.257 (1 / 1.000) | 3 | not certified (G2', G3') | sfwd2.routing c0 1.26, sfwd0.routing c1 1.20, sfwd2.routing c1 1.12, sfwd0.in c1 1.08 |
| djb | 1 / 1.49 (4 / 2.24) | 0 / 0.987 (1 / 1.000) | 0 | **certified** | none |

- **Float64 (G3').**
  - suv fails 3 groups: sfwd9.routing c0 at 1.85e-12 and 1.20e-12, and sfwd0.routing c1 at 3.33e-12 against eager's
    float64 one-ulp change of 2.09e-12.
  - t2d fails fwd0.routing c1 at 1.53e-12.
  - All of these are routing gradients that carry a float64 one-ulp change of 0.3-2e-12, which is reordering roundoff
    amplified by an ill-conditioned group. I see no sign of a code bug: max d(comp64, ref) is at most 1.3e-13 for every
    other family, and eager64 against the reference is at most 1.6e-14.
- **The excesses are systematic, not noise.** The same groups top the list in Part 1 and in Part 2, with fresh seeds
  and fresh compile processes:
  - af: fwd5, fwd10 and fwd11 routing;
  - az: fwd9.params c1;
  - lug: fwd2.routing c1, at 1.94x and then 2.00x;
  - surfaces: block 0 (and block 2) routing;
  - vol: blocks 3-4 routing;
  - t2d: blocks 0-2.
- Inductor's rounding of those routing gradients is further from float64 than eager's, while the compiled path is
  closer to float64 than eager in 25-111 other groups per family. How large the excess is varies by family:
  1.02-1.03x on az and af, 1.1-1.3x on the surfaces, 1.6x on vol, 2.0x on the lug and 11x on t2d.
- **Null calibration of the gates** (`out/cert/NULLCHECK_part2.md`): a sanity check on Part 1's eager records at
  reduced sizes, run before the Part 2 data arrived.
  - G2' passes eager against eager in 93-100% of repetitions, mean 97%.
  - G1' passes in 77-100% of stand-in pairs, mean 87%. Its thresholds are discrete: on the pair maxima the 95% level
    lands on the null's maximum, so a stand-in fails when it exceeds every null value of its own process (1 in n + 1
    per process). At Part 2's 16 draws per process I expect about 6-12% false failures.
  - The combined false-failure rate of G1' and G2' is therefore about 10-15%, against the 10% the amendment allowed for.
    Only drs, hl and vol fail on G1', and each of them also fails G2'.
- Beside, not gating:
  - G1' on bf16 passes on every family except t2d.
  - G2' on fp32 fails on 8 of 9 families. The fp32 eager family is so tight that any systematic rounding difference
    lies outside it, which is why fp32 is gated by G1'.
  - The per-tensor G2' (max-relative) falls outside its null on every family; djb is outside on R alone (1.92x).

## 4. Rerun pricing (`lhr/price_cert.py`, `out/PRICE_CERT.md`)

Training GPU-h for all seeds: 4 x steps x epochs x the smoke step x PRICE's overhead factor. The AeroZoo held-out
splits scale the full split's step by PRICE's split/full ratio.

- With LANES as it is (every ISLA lane compiled except t2d): 3,216 GPU-h.
- With only djb compiled: 4,031 GPU-h, **+815 GPU-h** (+25%).
- Each switch to eager costs:

  | family | extra GPU-h |
  |---|---|
  | AeroZoo, 7 splits | +635 |
  | lug | +59 |
  | vol | +27 |
  | af | +22 |
  | suv | +31 |
  | hl | +22 |
  | drs | +18 |

- t2d stays eager. Training it compiled would save 22 GPU-h, but it does not certify.
- The smoke steps differ from PRICE's estimates. AeroZoo compiled is 0.846 s against the estimated 0.763 s, and DeepJEB++
  is 0.118 s against 0.138 s.
