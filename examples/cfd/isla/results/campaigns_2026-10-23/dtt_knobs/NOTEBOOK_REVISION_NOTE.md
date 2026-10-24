Status: DONE

# NOTEBOOK_DRAFT_STAGE1.md revised after NOTEBOOK_REVIEW.md (DK-NOTEBOOK-REV, 2026-10-24 03:36 EDT)

The first version is saved as `NOTEBOOK_DRAFT_STAGE1.v1.md` (`cmp` identical before editing, md5 bcb3796d). All 12 review
items are applied to `NOTEBOOK_DRAFT_STAGE1.md`. Each one was checked against its source: STAGE1.md (md5 1a28dc7c), the
child REPORT.md files and their tables, CODE_FACTS.md, PREREG.md (md5 84a25280 confirmed), and the book at
origin/psharpe/isla 1e7f03d7d, read with `git show` / `git grep`. Every passage named below was found as quoted. I ran no
jobs, changed no code and did not edit the book. The entry is 117 lines (116 before): the added qualifiers fill what
the trims freed. The rest of the draft is unchanged.

## P1
1. **Pose invariance, 05 and 20.** Change 4 is now three edits. (a) is the bfloat16 paragraph, unchanged. (b) scopes 05's
   "And a rotation, reflection or translation ... gives the same forest" to "in float32 or float64". I wrote float64 as
   well because the test that sentence cites, `test_forest_and_every_list_are_the_same_in_every_pose_at_finite_theta`,
   runs with `D = torch.float64`. (c) adds the octant tie to 05's list of ties: "So can the octant split, at a normal
   component that is zero but for rounding in that frame". I did not write "at zero": the book code's `normal_octants`
   already gives a component at exactly zero the sign of the next nonzero one, so only rounding residues still tie
   (CODE_FACTS §2: 84 % of the lug's cells are below 1e-5). Change 7 gains a new (b) that makes the same two edits to 20's
   para-oq-dtt-frame ("is the same in every pose" and its list of ties). The old (b) is now (c), and the later reference
   to it was updated.
2. **In-flight callouts.** Change 9 adds the DTT-FAST callout (ledger row 700, RUNNING), in the review's text, directly
   after the cell `para-cost`. The callout also covers that cell's existing ISLA-DTT step and memory figures. Change 10
   adds a copy of the AF-WHOLE callout at 12:278 (row 697, RUNNING), word for word. The callout note before change 1 now
   says that two changes add a callout now and that change 8's waits for stage 2.
3. **Lug bfloat16.** In the entry and in change 11, the lug's forms read level "mostly because bfloat16 read-out noise
   (about 0.0033, twice the float32 errors) compresses the gap toward 1". Both now add that ISLA-DTT is still worse in
   73 % of cases and on the surface (1.11). Source: gap/REPORT.md L22-26 and STAGE1 L231-232. "only because" and "cannot
   show the gap" are gone.

## P2
4. **Projections labelled.** The heading, Result 2, changes 2, 8 and 9 now say which 400,000-point rows ran. At θ = 3, 2
   and 1.5 they ran; at θ = 1 the run went out of memory; θ = 1.25 is the fit's 285 GiB (cost_table.md). The 349k, 380k,
   431k and 462k counts are "projections" of "the memory fit (held-out error at most 0.4 %)", and in change 9 so is 565k.
5. **Change 11.** It now goes after "...while the standing's ISLA trained eagerly ({source}).", before "Which form ...
   is open". It defines bfloat16 at first use in 16, in 01's wording; git grep finds no bfloat16 in 16 now.
6. **Roughness.** The entry and change 10 now read "except eddy viscosity beyond 0.1 chord, where its gap sits"
   (gap/REPORT.md L35-36).
7. **Values.** Result 4 and change 5 now read "mostly through the clusters' mean values" (order/REPORT.md: exact values
   0.31-0.49, exact weights 0.84-1.04).
8. **Settings and leaf.** Change 5 names the seven θ settings (0, 0.25, 0.5, 0.75, 1.25, 1.5, 2; sens/REPORT.md L50) and
   reads "The leaf size matters less: ... on the lug leaf 2 costs 2 % and 33 % (seeds 42 and 43) and leaf 8 changes it
   by at most 0.8 %". For consistency I made two more changes of the same kind: change 1 now says "among the eight
   settings from theta 0 to 2", and change 8's section names the settings.
9. **Row 701.** Change 2 adds a fifth-column edit that records the gate as met: ISLA-DTT on AirfRANS within 1.8e-6
   relative; lug 0.00221933 / 0.00183431 against 0.002219 / 0.001834 (sens/ and gap/ REPORT.md).
10. **PREREG.** The Consequence paragraph now cites PREREG.md (md5 84a25280) where it cited s2plan/PLAN.md: about
    1,010 GPU-h without the exact read, and the exact read about 330 to 390 more, submitted only if its smoke prices it at
    300 or less (PREREG L243-260). Change 8's row now reads "about 760 to 820" where it read "not priced": 210 + 220 +
    330-390, from PREREG's table, eager, two seeds each. The gate, with P11 = 0.75 that it stops the exact read, is in the
    note to the supervisor.
11. **For the book.** It now says "error ratio" for 04 and 20, and the path to the draft itself is replaced by "(records
    copied to results/campaigns_2026-10-24/dttknobs/)", the pattern 18's entries use.
12. **Order and length.** "So the knob rule's test..." now comes right after "...when read exactly.". I cut Result 2's
    argsort limit, the 3.2x coupled-read detail and the 1,180 / 1,202 ms probe check; change 9 keeps the check, and
    change 3 and CODE_FACTS keep the rest. The 93-99 % node share is now "on the volume datasets".

## Left for the supervisor
- In 05, the first use of bfloat16 (line 1641, `node_means`) comes before its definition (1851). This was there before
  the draft and falls outside the review. Change 4 adds a use near the same place.
- Change 10's paragraph may be printed inside the cell `para-airfrans-dtt-reference`, which prints a second paragraph
  after its first. The callout must then go after the cell.
- Change 5's note on STYLE's knob rule, and change 7's possible DTT-POSE callout, are still the supervisor's to decide.

On disk: this note, `NOTEBOOK_DRAFT_STAGE1.md` (md5 293fbd0b) and `NOTEBOOK_DRAFT_STAGE1.v1.md`, all
small, in this git-ignored scratch folder.
