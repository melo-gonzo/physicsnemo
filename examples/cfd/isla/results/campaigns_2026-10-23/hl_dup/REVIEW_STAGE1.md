# HL-DUP stage 1: second read (2026-10-24, read-only)

Recomputed from the raw records (275 census JSONs, expose.jsonl, 12 score files plus standing/, hole*.jsonl).
Everything not listed reproduces: census ranges and medians, pooled shares, coverage (180 + 88 = 268), exposure table,
(a)/(b)/(c) seed means and per-seed ratios, per-case win counts 53/54/9 and 22/20/10, epoch 500, footprint/curator
hole numbers, loose ends, non-manifold counts, and outline coverage.

## Discrepancies

1. RESULTS §3 (+5 % / +6 % for GeoTransolver): (c)/(b) on wall shear is **+1.7 %** (0.09635 / 0.09473), not +6 %. Pressure +4.7 % is fine. **wrong**
2. RESULTS Answer, "(b) raises every arm's error by about 37 %", followed by the pressure *and wall shear* ratios: b/a is 1.38 on pressure but **1.06 on wall shear**, 1.22 on velocity, 1.01 on density and 1.35 on temperature. Say "pressure error by about 38 % (wall shear 6 %)". The NOTEBOOK entry's first bullet repeats this. **wrong**
3. RESULTS §3, "correlates with the measure share (r = 0.33) but not with the length unit (r = -0.04 / -0.05)": no record or script holds these r values, and I cannot reproduce them against (b). With c/b, r(share) is 0.13 / 0.11 per ISLA seed (0.12 for the seed mean). Only c/a gives ~0.31 / 0.31, which breaks the section's own rule that (c) be compared with (b). The "not with the length unit" half holds only for |log gauge|. The signed gauge ratio correlates with the share at r = 0.875 and with c/a at 0.29. **wrong**: re-derive against (b) or drop the sentence.
4. RESULTS §3 and NOTEBOOK item 3, "changes the ratios by at most 0.004": this holds for pressure and wall shear (seed means 0.0037). On velocity and temperature the change is up to 0.0068 (seed means) and 0.0085 (per seed, T/ISLA temperature). Scope it to p and wss. **imprecise**
5. RESULTS §3 Confidence, "seed ratios within 0.04": T/ISLA pressure is 0.955 vs 0.890 across seeds, a 0.065 gap (density 0.954 vs 0.896). **imprecise**
6. RESULTS §4, "the open-edge closing leaves 31-41 in² at eps = h_max/2": true for four of the five. On LHC030_AoA_4 it leaves **0 in²** (eps 0.73 in; 4.3 in² at 2x). Write "0-41 in²". **imprecise**
7. RESULTS §3, "each per-case metric within 1.5e-5 relative": this holds for the 5 headline fields (1.45e-5). Over all 11 metric keys it is 8.3e-5 (ISLA s43, a component field). **fine** (say "headline fields")
8. NOTEBOOK, "up to 490,000 copies": the maximum is 491,701 (UPSTREAM correctly says 492,000). **fine**
9. UPSTREAM, "edges on y = 0 shared by up to 1.08 M cells": no local record holds this (it presumably comes from BC-SIDECAR). Cite the source. **fine**

## Conclusions

- (b) vs (a): this follows. Same predictions and same draw, and the ratios move by ≤0.004 on p and wss and ≤0.009 on any field, so the ranking is unchanged.
- "Compare (c) with (b)": this is justified, since both score only distinct cells. The text should apply it consistently, which item 3 does not.
- The hole verdict, as far as it goes (no construction from the deduplicated wing closes the five broken roots), follows from hole.jsonl and hole2.jsonl. Two claims do not:
  (i) "about 30 % of the outline is missing *once the duplicates go*" and "that 30 % is where the duplicated root cells
  were". Deduplication keeps one copy of every cell, so the set of on-plane edges is unchanged. hl_hole3.py measures
  coverage by *open* edges (count == 1) only. The uncovered 30 % may be present as shared or non-manifold edges (55-300
  per root), and no record ties it to the duplicate locations. (ii) "Repairing needs data the files do not hold" therefore
  goes beyond the evidence. Check coverage by *all* on-plane edges first. The same wording is in NOTEBOOK item 4 and the
  entry's hole paragraph, and in UPSTREAM's "about 30 % of the root outline is missing".
  "The root outline is the same on every case checked" is shown only for the ~70 % that is present.
- NOTEBOOK chapter changes: concrete (location, text, numbers, source) and consistent with RESULTS, apart from inheriting
  items 2, 4 and the hole wording above. Item 2's (b) numbers and ratios match score_table.md.

second-read: 7 fixes needed (items 1-6 and the hole wording)
