# Drafts for the supervisor (GLOBE-SCORE, DrivAerML; 2026-10-25)

Source: scratch/isla-globe-2026-10-19/RESULTS_DR.md and scores_dr/globe_dr_scores.json.

## Notebook entry (for 18-notebook.qmd)

## 2026-10-25 — GLOBE on DrivAerML Surface: at its published configuration it trails A2 ISLA about twofold, so the AeroZoo stage does not run {#sec-nb-globe-dr-scored}

**Question** (GLOBE-SCORE, pre-registered 2026-10-19 in
scratch/isla-globe-2026-10-19/PREREG.md, at Peter's request). Where does
GLOBE stand on DrivAerML Surface? It runs at the settings of its own
DrivAerML example: 256 × 3 Padé kernels, 8 latent scalars and 4 latent
vectors, 4 Legendre terms, θ = 1.5, 1.0M parameters, 80,000 faces per step
at a rate of 1e-2. Everything else follows the recipe protocol: Muon and
AdamW, StepLR, Huber loss, bf16 and 500 epochs. The reference lengths are
constant (L_ref = 1, δ_turb = 0.015), per Peter's reference-length ruling.
The comparison is with A2 ISLA and the fair-drive baselines. Scores are in
float32 on the 48 validation cars at ISLA's 200,000 cells.

| Arm | Pressure | Wall shear | GLOBE ÷ arm |
|---|---|---|---|
| GLOBE | 0.09782 (0.09701, 0.09863) | 0.12145 (0.12037, 0.12254) | 1 |
| A2 ISLA | 0.04452 | 0.07004 | 2.197 / 1.734 |
| GeoTransolver, fair drive | 0.04552 | 0.06962 | 2.149 / 1.744 |
| Transolver, fair drive | 0.04234 | 0.06782 | 2.310 / 1.791 |

Native relative L2, seed means; GLOBE's seeds 42 and 43 in parentheses.
Every per-seed ratio lies on the same side of one as its mean.

**Verdicts.** There is no adoption rule: the result adds a GLOBE column to
the DrivAerML Surface standing. The fallback did not trigger; the worst
validation was 1.016 times its running minimum. The AeroZoo stage requires
GLOBE within 1.10 of A2 ISLA on some headline field of either dataset. The
closest fields are AirfRANS pressure at 1.32 and DrivAerML wall shear at
1.73, so the trigger does not fire and the stage does not run. GLOBE-SCORE
is done.

**Beside the rules.**

- GLOBE is worse than ISLA 0.9.0 on all 48 cars on both fields; per car,
  the ratio runs from 1.50 to 3.19 on pressure and from 1.24 to 2.29 on
  wall shear.
- GLOBE underfits: its final training loss, 0.00175, is close to its
  validation loss and 6.7 times ISLA 0.9.0's under the same loss. Its
  1.0M parameters are its authors' choice; ISLA 0.9.0 has 69M.
- Posed ÷ native is 1.03 to 1.08, against float32 equality on AirfRANS.
  The DrivAerML model sums with a Barnes-Hut tree of axis-aligned boxes
  (θ = 1.5), where the AirfRANS model sums exactly (θ = 0). This is the
  likely cause; it was not isolated. GLOBE's posed score, 0.104 / 0.127,
  is still about ten times better than the baselines' posed scores.
- Area ÷ native is 1.010 / 1.021.
- Scored at Transolver's 100,000 cells, GLOBE is 6 percent worse than at
  200,000, because its boundary input is the scored subsample. GLOBE ÷
  fair-drive Transolver on that set is 2.457 / 1.912.
- Registered predictions: GLOBE below A2 ISLA (0.3 and 0.25) failed,
  posed within 1e-3 of native (0.9) failed, 100k within 3 percent of 200k
  (0.8) failed; area within 5 percent (0.75) and no fallback (0.7) held.
- As on AirfRANS, the recipe protocol departs from the example's (RAdam,
  plateau schedule, gradient clipping, its own loss and targets, ground
  planes). This tests GLOBE under the shared protocol, not the example's
  own numbers (scratch/isla-globe-2026-10-19/RESULTS_DR.md).

**ISLA 0.9.0.** The ISLA 0.9.0 DrivAerML Surface lanes (REF-090) read
0.04113 / 0.06622, so GLOBE ÷ ISLA 0.9.0 is 2.378 / 1.834
(scratch/isla-ref090-2026-10-20/RESULTS_partial.md).

**Next.** None for GLOBE-SCORE. Testing GLOBE under its own training
protocol, or at a larger width, would be a new piece of work.

## Ledger row (replaces the GLOBE-SCORE row in 14-program-status.qmd)

| GLOBE-SCORE | DONE 2026-10-25 (2026-10-20: AirfRANS, GLOBE ÷ A2 ISLA 1.32 / 2.26 / 1.71 on pressure / velocity / eddy viscosity, posed equal to native, @sec-nb-globe-af-scored; 2026-10-25: DrivAerML Surface, GLOBE ÷ A2 ISLA 2.20 / 1.73 on pressure / wall shear, both seeds above one, posed ÷ native 1.03-1.08; the AeroZoo trigger does not fire on either dataset, so that stage does not run; no fallback; @sec-nb-globe-dr-scored) | Where does GLOBE, at its authors' published configurations and under the recipe protocol, stand against ISLA and the fair-drive baselines? | Peter's request (2026-10-19); the reference-length ruling; DRIVE-FAIRNESS | No adoption rule: a GLOBE column on AirfRANS and DrivAerML Surface; the AeroZoo stage runs only if GLOBE is within 1.10 of ISLA on a headline field; one half-rate fallback | @sec-nb-globe-af-scored, @sec-nb-globe-dr-scored |

## For the book

GLOBE, at its authors' published configurations and under the shared recipe protocol, is a third comparison
architecture that trails ISLA on both datasets: by 1.3 to 2.3 times on AirfRANS and by 1.7 to 2.2 times on DrivAerML
Surface against A2 ISLA, and by 1.8 to 3.3 times against ISLA 0.9.0. It trails the fair-drive baselines by about the
same factor on DrivAerML, and on AirfRANS everywhere except velocity, where it beats them. Its boundary-integral form
gives it the posed robustness the attention baselines lack: posed equals native on AirfRANS, and on DrivAerML its posed
error is within 8 percent of native, about ten times below the baselines'. The accelerated tree summation on DrivAerML
costs that exactness. The AeroZoo held-out stage, which needed GLOBE within 1.10 of ISLA somewhere, does not run.
