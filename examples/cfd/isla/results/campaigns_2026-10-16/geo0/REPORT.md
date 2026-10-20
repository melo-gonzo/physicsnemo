# GEO0: are ISLA checkpoints trained with full-mesh query geometry insensitive to how the boundary is sampled? (2026-10-16)

The pre-registration is `PREREG.md`. The draft, the amendment (21:16 -04:00) and the DrivAerML implementation note
(21:24) were all written before any redraw result existed. Jobs are listed in `JOBS.md`. Per-unit numbers are in
`RESULTS_TABLE.md` and `results.json` (both made by `reduce.py`; its verdicts are in `reduce.log`). md5s are in
`MD5S_LHR.txt` and `MD5S_LOCAL.txt`; the copy md5s are in `md5_{runs,snap}_{aga,lhr}.txt`.

**Scope label.** All runs are ISLA R1 (384 wide, 12 layers, 256 slices) on code_v3 + recipe_v3. AirfRANS is the 3-D
`airfrans_volume` set, whose interior is a windowed (contiguous solver-order) draw. The primary read compares each
`sg_*_provided` checkpoint with its `sg_*_internal` twin: same snapshot, data and training, and only the query geometry
differs. Comparisons with CHAR-AREA2 and CHAR (512/512 class, other snapshots) are context only.

## What ran
* **Copies.** The checkpoints, code_v3 and recipe_v3 went from AGA to LHR `$T/kernel/geo0/{runs_from_aga,snap_from_aga}`
  through the LHR data copier (`copy_from_aga.sh`). md5s match: 40 checkpoint files and 1,097 snapshot files.
* **Recipe copy.** `recipe_v3g0` is recipe_v3 plus three changes:
  * a v3 port of the CHAR-AREA2 `GuardedPoissonSubsampleMesh`;
  * on AirfRANS, a 1/pi fold of the stored `quadrature_measure`, which recipe_v3 reads directly;
  * the dataset twins `airfrans_volume_g0{a,u,v}` and `drivaer_ml_volume_reference_g1{a,u,v}`.
* **Jobs, LHR, short QOS.**

  | Job | Units | Elapsed | Outcome |
  |---|---|---|---|
  | 241664 (AirfRANS) | 16 | 7:39 | completed |
  | 241665 (DrivAerML) | 16 | 5:13 | native units fine; the 12 `g0` redraw units stopped in the guard on case 1 (below) |
  | 241802 (DrivAerML) | 12 | 6:59 | `g1` redraws, completed |

* **Why `g0` stopped.** DrivAerML has a 722-cell in-file boundary besides the vehicle, and the guard refused to pass
  it. `g1` lets small meshes pass through unchanged, as the CHAR area protocol does. Every `g1` unit then logged 48 of
  48 vehicle draws from the 1,000,000-cell pool (192.6k to 200.8k cells kept).
* **Unit counts.** All AirfRANS redraw units have 200 guard lines and 200 infer rows. The DrivAerML units have 48 infer
  rows each.
* **ENV check.** The rescored native matches the committed 2026-09-23 records exactly, on every run and field.

## Results (seed 42 / 43)
r = area / uniform; d = |uniform-B / uniform - 1|; c = uniform / native; R = area / native.

| Set | Geometry | Field | r | d | c | R |
|---|---|---|---|---|---|---|
| AirfRANS, 400 segments | provided | p | 1.449 / 1.381 | 0.030 / 0.040 | 2.20 / 2.35 | 3.18 / 3.24 |
| | | v | 1.984 / 1.805 | 0.017 / 0.029 | 1.62 / 1.67 | 3.22 / 3.02 |
| | | nut | 1.135 / 0.983 | 0.013 / 0.043 | 1.72 / 1.86 | 1.95 / 1.83 |
| | internal | p | 2.649 / 2.885 | 0.016 / 0.027 | 4.32 / 3.98 | 11.4 / 11.5 |
| | | v | 1.741 / 1.778 | 0.003 / 0.003 | 7.49 / 6.72 | 13.0 / 12.0 |
| | | nut | 0.978 / 1.011 | 0.042 / 0.043 | 4.72 / 4.57 | 4.62 / 4.62 |
| DrivAerML Volume | provided | p / v / nut | 0.999 / 0.998, 1.000 / 0.999, 0.999 / 1.000 | ≤ 0.0007 | 0.999–1.002 | **0.998 / 0.998, 0.999 / 0.999, 1.000 / 1.000** |
| | internal | p / v / nut | 10.7 / 11.2, 8.10 / 8.06, 8.66 / 9.19 | ≤ 0.003 | 0.998–1.006 | **10.7 / 11.1, 8.10 / 8.07, 8.71 / 9.17** |

Absolute errors at the AirfRANS area draw, provided against internal: p 0.102–0.108 against 0.369–0.413; v 0.049–0.051
against 0.253–0.259.

## Verdicts
* **(ENV): holds.** The rescored native equals the committed records to within 0.0000%.
* **(G0-DV): holds, decisively.** The provided R is 0.9975–1.0004 on every field and seed. It is also within the design
  study's sharper forecast, 1 + max(2d, 0.01). The matched internal twin is ×8.1–11.2 (**M-DV holds**). On DrivAerML
  Volume, therefore, the whole area-draw sensitivity is the nearest-token query geometry. Full-mesh geometry removes all
  of it.
* **(G0-AF) count clause: holds.** c is 1.6–2.4 on every field, against 4.0–7.5 for the internal twin.
* **(G0-AF) law clause: fails on velocity.** r on pressure is 1.45 / 1.38 (pass), but on velocity it is 1.98 / 1.81
  (bar 1.5), with d ≤ 0.03.
* **(M-AF): first clause holds, second fails.** The internal r exceeds 1 + max(0.05, d) on p and v. "Provided r − 1 <
  internal r − 1" holds on pressure (0.38–0.45 against 1.65–1.89) but fails on velocity (0.81–0.98 against 0.74–0.78).
* **Unregistered per-case diagnostic** (`diag_af_percase.txt`): the per-case area/uniform ratio on provided velocity has
  a median of 1.48 / 1.33 and exceeds 1 in 74–77% of cases, with a heavy tail (max 8–11). So the effect is broad, not
  carried by a few outliers.

## What it means, and how sure
* **The geometry is the mechanism wherever the boundary keeps enough tokens.** Full-mesh query geometry makes ISLA
  draw-invariant on DrivAerML (200k tokens) and removes most of the AirfRANS count effect and most of its pressure law
  effect. High confidence: the matched twins and the noise floors are 10–1,000x below the effects.
* **At 400 airfoil segments a law effect of about 1.8–2.0x on velocity remains that geometry does not touch.** Likely
  cause, not tested here: the area law has pi_min ≈ 1.3e-3, so it drops the refined leading-edge and trailing-edge segments and gives the few it keeps HT
  weights of up to about 800. The slice-state quadratures would then lose resolution where the flow changes fastest. That
  matches the design study's "slice states" residual, but larger than its 1.3–1.5 estimate.
* **Consequence for Step 1.** Bar (d) of the design study ("ca2a/ca2u ≤ 1.5 on p and v") is unlikely to be met on
  velocity by the exact-geometry change alone. Its bars (a), DV area/native, and (e), count ≤ 4x, are supported.
* **Caveats.** The checkpoints are an older, smaller class than the reference, and on AirfRANS the interior is windowed.

## Left on disk
* LHR `$T/kernel/geo0`: 2.2 GB. It holds `runs_from_aga` (2.2 GB of checkpoint copies, removable once no longer
  needed), `snap_from_aga`, `recipe_v3g0`, `evals` (2.7 MB) and the logs.
* Local, this directory: about 4 MB, including `evals_lhr` (metrics and logs, 3.5 MB). The `ref/` source copies were
  deleted.
* Nothing is still running.
