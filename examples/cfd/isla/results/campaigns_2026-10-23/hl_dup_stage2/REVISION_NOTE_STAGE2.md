# HL-DUP stage 2: revision after the second read (2026-10-24, about 19:05 EDT)

The second reader's report is `REVIEW_STAGE2.md` (lane-reviewer on devin/claude-opus-5.5:high, read-only; md5
ff430239). The v1 files are kept as `RESULTS_STAGE2_v1.md` (ce9cf8c5) and `NOTEBOOK_DRAFT_STAGE2_v1.md` (4ad260fb),
each cmp-equal to its file before the edits. The reader confirmed every score, rule cell, prediction P1-P4, band,
case count, spread and exposure figure, and the "mixed" verdict. The fixes:

1. **"Lower on 116 of 180 cases" before stall** counted all cases. It is 66 of 84 before stall and 50 of 96 after
   (NOTEBOOK entry and chapter change 3; RESULTS now gives the split).
2. **"The 41 cases that store the source STL ... the same on every one"**: only 39 caps were read. Now "39 of 41"
   (NOTEBOOK entry and chapter change 5).
3. **"Both baselines lead mostly through post-stall cases"** holds for GeoTransolver only; Transolver's lead is
   pre-stall. RESULTS "What it means" now says so.
4. **Post-stall leads were seed means only.** GeoTransolver's sign holds on both seeds (pre-stall 1.050 / 1.016,
   post-stall 0.925 / 0.892). Transolver's post-stall seeds disagree (1.047, 0.913). Both files now say so.
5. **P5 was labelled "held"**, but the reducer, which tests all five fields, records it as not held, and the claim
   that stage 1 reported this ratio on pressure and wall shear is not in the record. It is now **undetermined**: in band
   on pressure, wall shear and density, below on velocity and temperature; the PREREG's 1.009 is the pressure value.
6. **"ISLA's gain over (b) is similar in both bands"**: it is larger before stall (0.898 of (b) against 0.936).
7. **"The baselines' lead fell to about 4-6 %"**: 3.4-6.4 %, now "about 3-6 %".
8. **"Baselines 0.94-0.97 of their (b)"** and "3-7 %" are pressure only, per seed 0.935-0.972. Wall shear is now named
   separately (baselines 1-3 %, ISLA 5-8 %).
9. **ISLA's seed-mean wall shear** was 0.09632, rounded from rounded seeds; the raw value is 0.096326, now 0.09633.
10. **Supersession of the stage-1 draft.** Only its data-section sentence (item 1, without the callout) survives.
    Items 2-5 are replaced, including the (b) row and the deficit sentence, which are cut if already applied. The new
    footnote under the table names the (b) figure, 0.0589, instead of calling every older number incomparable.
11. **"The crust rebuild failed at the trailing edges"**: that is the knock-out tests on the clean roots. The broken
    roots stopped at the first check (16-17 distinct vertices sharing an (x, z)). Both are now stated.

Conclusions narrowed, as the reader asked:

- **"The duplicates explained part of ISLA's deficit"** is now limited to what the rule grants: narrowed on wall
  shear, unresolved on pressure. The ratio movements are labelled beside the rule.
- **"The retrain lowered ISLA's error more than the baselines'"** is now firm on seed 42 only. On seed 43 ISLA's
  pressure margin over GeoTransolver is under 1 % (0.955 against 0.957).

No number in stage2_table.md or stage2_summary.json changed.
