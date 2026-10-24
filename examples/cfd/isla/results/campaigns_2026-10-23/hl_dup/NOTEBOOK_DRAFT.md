# Notebook entry draft: HL-DUP, the duplicated root cells of HiLiftAeroML (2026-10-24)

Source: `scratch/isla-hl-dup-2026-10-24/RESULTS.md` (census section 1, exposure section 2, scores section 3, hole
section 4).

## Entry

**HiLift's wing data repeats its root.** An exact census of all 180 geometries found the same pattern on 266 of 268 cases.
The wing stores about 18.9 M extra copies of about 25,700 root cells, up to 492,000 copies of one triangle. They carry
46 % of the summed wing area (median) and, through the standing's pipeline, 42 % of the Horvitz-Thompson measure of the
200,000 drawn tokens. The scorer weights its errors by that same measure.

Rescoring the standing tells three things:

- Scored without the copies, every arm's pressure error rises by about 38 % (ISLA 0.0427 to 0.0589) and its wall-shear
  error by about 6 %, but baseline / ISLA stays at 0.91-0.92 on both fields. The defect does not explain ISLA's HiLift deficit.
- With the copies also removed from the input, a shift from training, ISLA loses 41 % on pressure and the baselines
  5-6 %. ISLA reads the cell measures that the copies distort; the baselines do not.
- ISLA's length unit moves by -23 % to +19 % per case without them.

A retrain on deduplicated data is proposed (PREREG, awaiting approval).

The symmetry plane's hole comes out right from the wing's footprint only on the clean LHC001 root (612,137 in² against
curator's 612,137 in², Hausdorff 0.011 in). On the five broken roots, every on-plane edge together covers only about
70 % of the outline, but the stored vertices cover all of it: about 2,950 on-plane vertices per root belong to no cell.
So the root triangles, not the root geometry, are missing. A closing of each case's own edges and vertices gives
611,842 in² (0.05 % short), but only at a closing scale above the vertex spacing; a robust rebuild of the root strip is
future work.

## Chapter changes

1. **HiLift chapter, data section.**
   - Add after the description of the wing surface: "The stored wing repeats about 25,700 root cells about 18.9 M times
     on 266 of 268 cases censused (all 180 geometries), so 42 % of the drawn boundary measure, and of the scorer's
     weights, sits on copies (HL-DUP, `scratch/isla-hl-dup-2026-10-24/RESULTS.md` sections 1-2)."
   - Add one "In flight" callout: "Deduplicated-data retrain of ISLA and both baselines proposed (HL-DUP stage 2); the
     HiLift numbers below are on the stored data."
2. **HiLift chapter, standing table** (RESULTS_HL090 numbers).
   - Keep the numbers; they are the standing as scored and are reproduced exactly (HL-DUP (a)).
   - Add a row or footnote "without the duplicated copies in the metric (HL-DUP (b))": ISLA p 0.0589 / wss 0.1027;
     GeoTransolver 0.0540 / 0.0947; Transolver 0.0544 / 0.0934. The ratios are 0.917 / 0.922 and 0.924 / 0.909.
     Source: `score_table.md`.
3. **HiLift chapter, discussion of ISLA's deficit.** Add: "The deficit is not an artefact of the duplicated cells in the
   metric (pressure and wall-shear ratios unchanged to 0.004), but ISLA, which reads cell measures, is far more sensitive to them than the
   baselines: removing them from the input costs ISLA 41 % against 5-6 % (a shift from training; HL-DUP (c))."
4. **Whole-boundary / BC-SIDECAR section** (if present). Replace "the root needs a repair, or the hole a different
   construction" with: "the footprint construction reproduces curator's hole on LHC001 to 1e-6 in area; on the broken
   roots the on-plane edges cover only about 70 % of the root outline, but the stored on-plane vertices cover all of
   it (about 2,950 per root belong to no cell), so the missing part is root triangles, not geometry; a closing from
   each case's own edges and vertices gives 611,842 in², 0.05 % short, and a robust rebuild is future work (HL-DUP
   section 4)."
5. **Standing summaries elsewhere** (overview chapter, if it quotes HiLift baseline / ISLA 0.91-0.92). No numeric
   change; add the same "In flight" pointer as in item 1, and only there, to keep one callout per number.

No superseded number is cut: the standing stays as scored until a retrain lands.
