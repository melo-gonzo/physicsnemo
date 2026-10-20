# HILIFT-090 (2026-10-20): ISLA's high-angle-of-attack errors on HiLift, and ISLA 0.9.0 on HiLift

HiLift = `highlift_surface_medium`, 690 cases (510 train, 180 val), ISLA's 200,000-cell score set, float32, epoch 500.
ISLA = A2 (a2r_hl_isla, seeds 42/43); GT / T = the unit-direction GeoTransolver 256 / Transolver 512x16 (udrv_hl_*);
"old reference" = **r80_hl_isla_w512_s512** (CHAR, 0.0402 / 0.0411). That is the run behind the notebook's 0.0407 and A2 ÷ old =
1.062. `r80_hl_isla_ref` (0.0450) is a different run, the pre-REF-80 default size.

Detail files: `partA/RESULTS_A.md` (per-case analysis, against r80_hl_isla_ref), `partA/tables_a_w512s512.md` +
`partA/percase_w512s512.csv` (the same reduction against the right old reference), `partA_spatial/SPATIAL.md` (surface
split), `partB/PARTB.md`, `partC/PLAN.md`.

## Part A: where the high-AoA error comes from (ranked by share of the gap)

1. **A few post-stall "branch" cases, in which the model predicts stall but the CFD flow stays attached.** About 10 cases
   at AoA 14-22 carry about 100 % of ISLA's 10.7 % pressure deficit to GT. Without them, ISLA ÷ GT is 0.999 for
   pressure and 1.017 for wall shear (partA F1).
   - In these cases the per-case error is close to all or nothing (about 0.2-0.29 against 0.03-0.07).
   - The wrong prediction under-predicts the lift: on the wrong branch, every model's CL is 4-8 % low.
   - Two cases are systematic for ISLA, on both seeds and in both A2 and old: LHC002_AoA_18 and LHC132_AoA_22, where GT is
     right on both seeds. They carry 41 % of the ISLA−GT gap. Most other branch cases flip from seed to seed for every model.
   - Four cases fail for every model (LHC040_16, LHC087_18, LHC060_14, LHC118_14, all about 0.25).
2. **On the surface, the excess lies in the leading-edge suction peaks of the mid-to-outboard main element and slat**
   (saved predictions, 174 labeled cases; share of ISLA−GT excess squared pressure error, all cases / ISLA's worst 18):
   - by element: main 82 % / 81 %, slat 16 % / 16 %, flap 1 % / 0 %, body and other 1 % / 2 %;
   - by pressure coefficient: Cp < −4 (0.2-0.4 % of the area) 76 % / 77 %, and −4 < Cp < −2 another 18 %;
   - by span: 40-80 % of the semispan 95 %;
   - by flow: regions of reversed true flow (τ·Û∞ < 0, 3-5 % of the area) only 7 %.
   - At AoA < 16, ISLA and GT are level (e² 17.1 against 16.8 ×1e-4).
   - Wall shear shows the same pattern: main 80-84 %, slat 12-14 %, Cp < −4 47 %.
   So ISLA misplaces where and whether the outboard leading-edge suction collapses. It does not miss the separated
   regions themselves.
3. **Against Transolver, the deficit is a broad offset.** It is not concentrated at high AoA. ISLA is worse on 85 % of
   cases and ISLA ÷ T is 1.12-1.20 in every AoA 4-12 bin (partA F2). A second mechanism, separate from item 1.
4. **The A2 regression against the old reference is three AoA-16 cases.** A2 ÷ old = 1.062 reproduces. The AoA-16
   bin's ratio is 1.71 and carries 122 % of the gap. LHC153_16, LHC162_16 and LHC119_16 carry 40 %, 38 % and 34 %.
   - The old reference is right on both seeds there (0.04-0.14; CL −0.5 to −2.4 %). A2 is wrong on both seeds
     (0.17-0.29; CL −4 to −7 %).
   - In every other bin, A2 is 1-6 % *better* (AoA 4-14 bin ratios 0.94-0.99). Wall shear behaves the same way: 1.010
     overall, 1.36 at AoA 16.
   - So the regression is a systematic regime choice at the stall onset, not a diffuse accuracy loss.
   - This fits the slice-read hypothesis (A2's normalized reads), but this data cannot isolate the reads from the other
     code_v20 to code_a2r changes.
5. **Training coverage does not explain it.** The split is random by case. AoA 16-22 has 192 train cases (37.6 %,
   45-60 per AoA value); 123 of 173 train geometries have a high-AoA train case; 113 of 118 val geometries appear in
   train. Within AoA ≥ 16, ISLA ÷ GT does not correlate with AoA, same-geometry train exposure or area shares
   (|ρ| ≤ 0.11). No deflection table exists in the dataset or manifest (partA F4-F5).
6. **Posed ÷ native (1.003-1.009) is an additive +2 to 6e-4 on every case.** It shrinks in ratio as the error grows and
   does not depend on AoA. The old ISLA shows the same (1.008 / 1.005), so it predates A2 (partA F7). Untested candidate:
   the pose changes which near-degenerate cells `DropDegenerateCells` removes, or which cells are drawn.

Spatial provenance: LHR 299722 (re-scored the four arms natively with the predictions kept and the reader's block draws
logged; per-case metrics reproduce the scores) and the CPU jobs 299886/299915/300009 (`spatial_reduce.py`). Mapping to the
full mesh is exact: 41-115 dropped cells per case, max centroid offset deviation 2.8e-4 in, element area shares within 0.01
of the labels json. Wall-shear e equals the scorer's to 2e-8. For pressure, the scorer's per-case denominator differs
(e ratio 0.47-0.95, the same for every arm), so e² was rescaled to the scorer per case; shares are unaffected. The
predictions (27 GB) were deleted after the reduction (job 300189).

## Part B: ISLA 0.9.0 fits on HiLift; no decimated asset is needed

HiLift is a surface task. 0.9.0's full-resolution path (Warp BVH, in-model draw) runs only for interior prediction points.
The surface reader already hands the model 200k uniformly drawn cells; the on-disk order is pre-shuffled, as checked.
Measured on GB300 (bf16, 4 GPUs, lane launcher; jobs 299686 smoke, 299755 paired on one node):
- memory: 64.5 GiB allocated / 66.4 GB reserved (A2: 66.0 GB) of about 277 GB;
- warm step median: 0.338-0.340 s (A2: 0.345 s); epoch wall 66-68 s (A2: 64.5 s), so about 1.0-1.05x A2;
- compile on the first step: about 160 s;
- losses finite.
Face counts of the full wing surface: min 248.7 M, median 277.2 M, max 302.0 M triangles (geo_LHC132_AoA_14). That is
126-153 M vertices, about 8.3 GB per case. The STL exists for only 20 cases. A fallback asset design, should a HiLift
interior task need one, is in `partB/PARTB.md`.

## Part C: the HiLift 0.9.0 lanes are staged, not launched

`$T/code_h090` (75ac723d8 + td0152), `$T/recipe_h090` (with the -medium dataset path; recipe_r090's points at a path missing
on LHR), `$T/kernel/h090/` (a2r rows 8-9; only run_id and code changed). Launch: `cd $T/kernel/h090 && bash submit.sh lanes 0 1`.
Score with `sbatch --array=0-1 score.sbatch`. Cost: about 68-76 GPU-h of training plus about 4 GPU-h of scoring, about 10 h of
wall time. LHR had 7 idle nodes at 06:20 EDT. Expected result: parity with A2 (0.95-1.05), because the surface path is A2
plus inputs by name and the gauge; the three AoA-16 cases should stay wrong. Details: `partC/PLAN.md`.

## Facts for project memory (2026-10-20)

- HiLift's old reference ISLA is **r80_hl_isla_w512_s512** (CHAR; 0.0402 / 0.0411 pressure). `r80_hl_isla_ref` (0.0450)
  is a different, smaller run and is not the A2 partner's old reference.
- ISLA 0.9.0's **surface task needs no geometry asset**. The full-resolution boundary path runs only for interior
  prediction points. On HiLift the reader passes 200k uniformly drawn cells (the cell order is pre-shuffled).
- **174 of the 180 HiLift validation cases have element labels** (`geo_*_element_labels.{npz,json}`; 179 of the 690
  cases have them in all).

## Follow-up: the HiLift 0.9.0 lanes were launched (2026-10-20 06:41 EDT)

Pre-registered as HL090 (`PREREG.md`, md5 a057d998, also at `$T/kernel/h090/PREREG.md`), separate from REF-090's R1.
Job IDs, acceptance and the score command are in `JOBS.md`.

## Cost of this lane
GPU: 299686 (about 7 min x 4 GPUs), 299722 (16 min x 4), 299755 (14 min x 4), about 2.5 GPU-h. CPU: 4 short jobs.
