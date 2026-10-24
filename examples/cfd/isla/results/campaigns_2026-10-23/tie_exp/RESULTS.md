# TIE-EXP results (MEDIAL-TIE DESIGN f; read 2026-10-24 01:43 EDT)

Registered read: PREREG.md section 4, with amendments A1-A9 (final md5 f0e70cb2). Reducer `launcher/read_tie.py` (de7f5223)
on local copies of the scorer and probe files in `read/`. Its output is `read/results.md` and `read/results.json`.

- Code under test: `$T/code_tie2` = B's snapshot code_cmpaf_fix plus 19fb0a754's geometry.py (f1810d97) and tree_read.py
  (c1d34bc1). The proof that only these two files differ is in `evidence/overlay_proof_tie2.txt`.
- B: r090eag_lug_isla and r090eag_t2dz_isla, seeds 42 and 43. They were eager-trained and eager-scored by kernel/r090eagaf and
  kernel/r090eag; the lug rows come from REF-090-CLOSE's scorers 343170 and 343179.
- TIE: tie_lug_isla and tie_t2dz_isla, seeds 42 and 43, scored by kernel/tie2 (jobs 345741 and 364216).
- Every row is at epoch 500, with 600 cases on the lug and 100 on Tensile2d. Every row is float32, fully eager (`compile=false
  model.compile_blocks=false`), at 200,000 cells.

## Verdicts

| rule | dataset | numbers | verdict |
|---|---|---|---|
| F | lug | TIE median d_F 1.38e-7 / 1.41e-7 (s42 / s43; max 4.1e-7; d_U 1.4-1.6); B beside: 1.40e-4 / 1.56e-4 | **exact flip symmetry restored** |
| F | Tensile2d | TIE d_F exactly 0 in every field, both seeds (d_U 2.5-175); B beside: displacement 8e-7, stress 1.4e-6, q 2.7-3.0e-5 | (same verdict, both datasets) |
| N | lug | von Mises: B 0.00138 / 0.00177 (mean 0.00157), TIE 0.00139 / 0.00177 (0.00158); r 1.004, log r +0.004; sigma = max(0.036, s_pool 0.176) = 0.176 | **no cost** |
| N | Tensile2d | displacement: B 0.00783 / 0.00629, TIE 0.00628 / 0.00806; r 1.016, log r +0.016; sigma = max(0.232, 0.166) = 0.232 | **no cost** |
| N (beside) | Tensile2d | stress r 1.034 (sigma 0.330), plastic strain r 0.991 (sigma 0.303) | no cost, no cost |
| P | lug | B posed / native 1.456 (e_B +0.456; per seed 1.246 / 1.619); TIE 1.0019 (e_TIE +0.0019; 1.0024 / 1.0015); e_TIE / e_B 0.004. Guard: log(1 + e_B) 0.376 > 2 x 0.036 | **the medial ties carry most of the lug's pose sensitivity** |
| P control | Tensile2d | displacement posed / native: B 0.9999, TIE 0.9999; abs(log ratio) 0.0000, within 2 sigma (0.464) | as predicted (not a verdict) |

**Decision rule (DESIGN f):** F passes and N is not "cost" on either dataset, so the rule's condition to **adopt** is met.
Adoption needs either section d's kernel or a per-mesh cache, because the search adds 26 % to the lug's step. P says that the
2026-10-18 note's other options are not needed for the lug's pose sensitivity.

Predictions (PREREG section 5): gate (i) passed (p 0.85) and F was restored (0.65). N was "no cost" on both datasets (0.70 for
the lug, 0.90 for Tensile2d). P came out "most" (0.40); the Tensile2d control fell within 2 sigma (0.95). Gate (ii) passed on
the lug at 1.4e-7 (0.70).

## Beside the verdicts (not registered rules)

- **Per-seed pairing on the lug.** TIE / B native is 1.008 for seed 42 and 1.002 for seed 43. The two arms share seeds and data
  order, so each TIE seed lands within 1 % of its B seed. The N rule's sigma (0.176) comes from the spread between seeds, which
  is 0.25 in log for both arms, not from the arm difference.
- **The pose excess is gone at every seed.** TIE's posed / native is 1.0024 and 1.0015, against B's 1.246 and 1.619. That is the
  float32 floor of an equivariant model, not noise, and the posed set rotates and translates every input (it is not degenerate).
- **A3's condition is not met as registered.** The FOLD-SIGN repeat against 19fb0a754 was recorded only in A9, after the first
  arm result. The transfer to the whole bundle is "recorded late (A9); the repeat ran before any arm existed". Fold sign: 0 sign
  changes on the lug (600 cases) and Tensile2d (100 cases), against both d3d9da49d and 19fb0a754 (LHR 341328 and 341486; rerun
  by GEO-AUDIT in 346922_10, 347401 and 347555). Token draw: every boundary cell is kept on both datasets (cell counts in gate2;
  the maximum is 8,568 against 200,000). ONWALL's rule (A8): gray-zone count 0. So on these two datasets, TIE stands for the
  whole geometry bundle.
- **Transfers measured before the arms:**
  - 34654e01d (TIE-FIX2): 0 uncertified rows and 0 bitwise differences in 5,143,196 rows, so the result is the same on these data.
  - GEO-BUNDLE's float64 cell normals: within A7's bound of 2.1e-5 at 190 lug nodes.
- **Supervisor-approved deviation (A7).** Gate (a)(1) and the float32 half of (a)(2) failed on the lug. At 190 of about 3.8M
  on-wall nodes |u32 - uref| reaches 2.1e-5, because the least-norm point is ill-conditioned at near-duplicate normals (the
  lug's split hex faces). The routine itself is exact on its inputs. This goes to the design agenda.
- **Blinding.** I opened none of the supervisor's blinded files before this read. B's lug rows were read for the first time here.

## What it cost

206.1 GPU-h in all (sacct; `read/sacct_tie.txt`), under the 220 GPU-h cap:

| item | GPU-h |
|---|---|
| arms (18 links; lug 2 x about 88, Tensile2d 2 x about 13) | 201.1 |
| gate and pre-step jobs (pregpu 1-3, gate2, transfer) | 2.6 |
| smokes | 0.9 |
| scoring | 1.0 |
| rule F probes | 0.4 |

The cpu-short jobs cost no GPU-h.

## Files

- Verdict tables: `read/results.md` and `read/results.json`.
- Gate: `gate2/gate_summary.txt`, `gate2/gate_a_summary.json`, `gate2/diag_transfer_summary.txt` and `gate2/DIAG_NODES.md`.
- Superseded code_tie records: `prestep/`.
- Jobs and md5s: `JOBS.md`. Launchers and probes: `launcher/` (local mirror of `$T/kernel/tie2`).

To reproduce the read: fetch the trees as in JOBS.md, then run `python3 launcher/read_tie.py read`.
