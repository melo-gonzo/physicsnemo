# HL-DUP stage 1: revision after the second read (2026-10-24, about 08:25 EDT)

The second reader's report is `REVIEW_STAGE1.md` (lane-reviewer, read-only). The v1 files are kept as
`RESULTS_v1.md` (3ae5c54d), `NOTEBOOK_DRAFT_v1.md` (042a5159) and `UPSTREAM_DRAFT_v1.md` (35d6b178), each cmp-equal to
its file before the edits. The fixes:

1. **GeoTransolver (c) / (b) on wall shear** was given as +6 %; it is +1.7 % (0.09635 / 0.09473). RESULTS section 3
   now reads "GeoTransolver +5 % / +2 %, and Transolver +6 % / +2 %".
2. **"(b) raises every arm's error by about 37 %"** is true of pressure only. RESULTS Answer and the NOTEBOOK entry now
   say pressure about 38 %, wall shear about 6 %.
3. **The correlation sentence** (r = 0.33 with the measure share; -0.04 / -0.05 with the length unit) used c / a and
   had no script. It is re-derived against (b) in `corr_cb.py` (output `corr_cb.json`). For log c / b on pressure, r is
   0.09-0.11 (ISLA) and 0.06-0.12 (baselines) with the measure share, and -0.10 / -0.11 with ISLA's |log length-unit
   ratio|. RESULTS now says both are weak and explain neither which cases suffer. It also gives the median per-case
   c / b: 1.64 / 1.57 for ISLA and 1.06-1.17 for the baselines.
4. **"Ratios change by at most 0.004"** is now scoped to pressure and wall shear (seed means), with "at most 0.009 on any
   field and seed" added. Changed in RESULTS section 3 and NOTEBOOK chapter change 3.
5. **"Seed ratios within 0.04"**: the Transolver ratios differ by up to 0.065 between seeds (pressure 0.955 / 0.890).
   RESULTS now says GeoTransolver agrees within 0.004 and Transolver within 0.065, and that every seed's ratio is below
   1 under (a) and (b).
6. **"Open-edge closing leaves 31-41 in²"** is now "0-41 in² (0 on LHC030_AoA_4)".
7. **The hole wording.** "About 30 % of the outline is missing once the duplicates go; that 30 % is where the
   duplicated cells were; repair needs data the files do not hold" went beyond the evidence. New checks settle it:
   - all on-plane edges, counting shared and non-manifold ones, still cover only 69.4-69.6 % of the outline
     (`hl_hole5.py`);
   - the stored vertices cover all of it, and 2,937-2,959 on-plane vertices per broken root belong to no cell
     (`hl_hole6.py`, `hl_hole7.py`);
   - a closing from each case's own open edges and vertices gives 611,842 in², 0.05 % short of the reference, at
     1.5x the vertex-spacing scale (`hl_hole7.py`).
   Rewritten in RESULTS section 4 (and its cost line), NOTEBOOK (entry and chapter change 4) and UPSTREAM. UPSTREAM now
   reports the orphan vertices and suggests the root triangles were lost in conversion.

Also, from the reader's "fine" items:

- RESULTS section 3: the 1.5e-5 agreement is for the five headline fields (8.3e-5 over all 11 keys).
- NOTEBOOK: 490,000 is now 492,000.
- UPSTREAM: the 1.08 M shared-edge figure is attributed to the earlier 21-geometry check.

The reader found everything else consistent with the raw records. No number in score_table.md, census_summary.json or
the exposure table changed.
