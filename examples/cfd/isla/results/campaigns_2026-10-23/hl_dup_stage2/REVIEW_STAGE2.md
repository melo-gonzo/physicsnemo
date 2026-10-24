I couldn't write `REVIEW_STAGE2.md` myself: I have no file-writing tool and the task forbids edits in that directory. The full review below is meant to be saved at that path.

# Second read: HL-DUP stage-2 write-up

I recomputed every number from `scores_stage2/*/metrics.jsonl`, `score_summary.json`, both `expose.jsonl` files and `out/crust/*`. Most numbers match the records:
- all 12 score rows, the four rule cells (G, the change and the seed ratios) and the "mixed" verdict;
- P1–P4;
- the AoA bands (84 and 96 cases) and the case counts (116 vs 54 against GeoTransolver, 35 vs 20 against Transolver);
- the seed spreads (5.8 %, 5.7 %, 2.1 %, 0.07 %), the posed/native ranges and the "0 identical" check;
- the exposure figures (0.773–1.868, median 1.136; val 0.772–1.186; 0.01–84 %, median 46 %; CV 14.6/12.6 vs 1.3/0.6 %; 3 of 3 draw checks);
- the old standing rows in chapter change 1, the md5s and the 209.6 GPU-h.

The verdict follows cell by cell. GeoTransolver pressure (+0.019) is inside the 0.03 band, so "narrows" fails. Three cells move by more than 0.03, so "unchanged" fails. What is left is "mixed".

## Discrepancies

1. **Wrong.** NOTEBOOK_DRAFT_STAGE2 chapter change 3 (and the Entry's third bullet) says "Before stall ISLA is 3 % ahead of GeoTransolver (lower on 116 of 180 cases)". 116 of 180 counts all cases. In the 4–12° band ISLA is lower on 66 of 84.
2. **Wrong.** The NOTEBOOK Entry's hole paragraph says "The 41 cases that store the source STL hold a closed root cap… the same on every one of them". Only 39 caps were read (`stl_cap_all.jsonl` has 39 rows, and HOLE_CRUST notes two incomplete STLs). It should say 39 of 41.
3. **Wrong (contradicts itself).** RESULTS ("What it means") says both baselines lead "mostly through post-stall cases". That is true for GeoTransolver (0.908 post-stall), but Transolver's lead is pre-stall (0.925 pre against 0.978 post), as RESULTS itself says in the AoA section.
4. **Imprecise.** The notebook's "post-stall both baselines still lead" and chapter change 3's "Transolver by 2 %" are seed-mean only. Post-stall T/ISLA is 1.047 on s42 and 0.913 on s43, so the seeds disagree. (GeoTransolver is consistent: 1.050/1.016 pre-stall, 0.925/0.892 post-stall.)
5. **Imprecise: P5 label.** RESULTS marks P5 **held**, but `stage2_summary.json` (made by the cited reducer) has P5 `held: false` because it tests all five fields. Also, "stage 1 reported this ratio on pressure and wall shear" is not in any local record: RESULTS.md never mentions posed, and PREREG cites 1.009 for pressure only.
   - Per field: pressure, wall shear and density are in band (density 1.0078/1.0076). Velocity (1.0031/1.0029) and temperature (1.0041/1.0041) are below it.
   - Suggested label: "undetermined: holds on p, τ and ρ; fails on u and T", or else align the reducer with the claim.
6. **Imprecise.** RESULTS says ISLA's dedup gain over (b) is "similar in both bands". Computed: 0.898 at 4–12° against 0.936 at 14–22°, a 10.2 % against a 6.5 % cut. The pre-stall gain is clearly larger.
7. **Imprecise.** RESULTS says the baselines' lead fell "to about 4–6 %". Computed from G: 3.4–6.4 %, so "about 3–6 %".
8. **Imprecise.** The notebook says baselines reached "0.94–0.97 of their (b)". Per seed it is 0.935–0.972 (seed means 0.946/0.969). Chapter change 3's "baselines' by 3–7 %" holds for pressure only; their wall shear fell 1–3 % (0.972–0.989). Name the field.
9. **Imprecise.** ISLA's seed-mean wall shear is given as 0.09632 in the RESULTS table and chapter change 1. The raw value is 0.096326, so 0.09633 (it was rounded from rounded seeds).
10. **Imprecise.** The supersession of the stage-1 draft has two gaps:
    - Stage-1 item 3 is an "Add:" to the deficit paragraph, which stage-2 change 3 then says to "Replace". It is unclear whether the stage-1 sentence survives, and if it does, it sits awkwardly beside "duplicates were part of the deficit".
    - Change 1 doesn't say what happens to stage-1 item 2's (b) row or footnote if it was already applied. Its new footnote ("numbers before 2026-10-24 weighted 42 % on duplicated root cells") would be false for those (b) numbers.
11. **Imprecise.** The notebook and chapter change 5 say the crust rebuild "failed… at the sharp trailing edges". The five broken roots actually failed the first check (16–17 outline points with the same (x, z)). The trailing-edge failures are in the knock-out tests.

## Conclusions that do not follow

- "The duplicates explained part of ISLA's HiLift deficit" (RESULTS) and "So the duplicates were part of ISLA's HiLift deficit" (notebook) amount to a "narrows" claim. The rule as registered grants that only on wall shear; on pressure it is unresolved, and the registration says no verdict is claimed. The claim should be limited to wall shear, or labelled beside-the-rule.
- The notebook's unqualified "the retrain lowered ISLA's error more than the baselines'" drops the RESULTS caveat. On s43, ISLA is at 0.955 of its (b) and GeoTransolver at 0.957 on pressure, so the claim is firm only on seed 42.
- "Tenfold more variation" is fine as a lower bound: the CV ratio is 11× on training cases and 21× on val cases.

second-read: 11 fixes needed