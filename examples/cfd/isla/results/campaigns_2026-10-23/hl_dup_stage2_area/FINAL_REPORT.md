Status: DONE

**What ran.**
- LHR job 375068 (`isla.hlarea.score`, 21:02-21:19 EDT): 1.09 GPU-h against a cap of 3. No commits and no book edits.
- The scorer is HL-DUP stage 2's `kernel/hldupx_score`, copied to `$T/kernel/hlarea_score`. I changed only the protocol, the run
  directory (`evals_hlarea`), the time limit, the job name and the bad-node exclusion.
- I made one more change: the two Transolver rows ran at the same time on GPUs 0-1 of the same node, to stay within the cap. The
  inference command is unchanged. If you count this as a harness change, say so.
- The area dataset file is new: `highlift_surface_medium_dedup_area.yaml`, in the read-only snapshots `$T/recipe_hlarea_{isla,base}`.
  It is the dedup file with CHAR-PROTOCOLS' area diff applied, and its content lines are identical to that diff.
- The run used the same deduplicating reader with the masks in `$T/hldup/masks` (every log says "fingerprint checked"), float32,
  epoch 500.
- Each score has the same 180 cases (and sample ids) as its native score.
- Not degenerate: per-case pressure differs from native on 180 of 180 cases for every run.
- I added the registry row before the job and removed it at 21:23. Two children ran, both on devin/claude-opus-5.5:high, and both
  finished: a cost child (`COST_REPORT.md`, `cost/`) and a read-only book scout (`BOOK_SCOUT.md`).

**Area protocol** (seed mean, s42 / s43; full table in `area_table.md`):

| arm | pressure | wall shear | area ÷ native (p) | baseline ÷ ISLA, p / τ |
|---|---|---|---|---|
| ISLA 0.9.0 | 0.05426 (0.05245, 0.05606) | 0.09612 (0.09417, 0.09807) | 0.989, 0.999 | — |
| GeoTransolver | 0.26088 (0.23886, 0.28289) | 0.35275 (0.33094, 0.37455) | 4.72, 5.48 | 4.81 (4.55, 5.05) / 3.67 (3.51, 3.82) |
| Transolver | 0.18938 (0.19495, 0.18381) | 0.29541 (0.31049, 0.28032) | 3.60, 3.58 | 3.49 (3.72, 3.28) / 3.07 (3.30, 2.86) |

**Cost** (the cost_090 rule, from the lanes' own logs; dedup with stored-wing in parentheses; `COST_REPORT.md`, `cost/cost_hldupx.json`):

| arch | params | step [s] | peak [GiB/GPU] | GPU-h per lane (s42 / s43) |
|---|---|---|---|---|
| ISLA | 68.8 M | 0.323 (0.341) | 64.6 (65.6) | 32.8 / 32.7 (32.4 / 32.6) |
| GeoTransolver 256 | 9.05 M | 0.431 (0.457) | 77.9 (77.9) | 38.5 / 38.1 (38.9 / 39.5) |
| Transolver 512×16 | 47.2 M | 0.323 (0.342) | 62.3 (62.3) | 32.5 / 32.2 (31.6 / 32.3) |

- The cost extractor, rerun on the stored lanes, reproduces cost_090.json exactly.
- Both sides ran bf16 on 4 GPUs with 4 ranks, ISLA on code_h090 with compiled blocks, so the comparison is like for like.
- The step is 5-6 % shorter because each draw keeps about 7 % fewer cells. GPU-h moves only 0.98-1.01×.

**What it means.**
- HiLift's third protocol is back on the deduplicated wing, and it reads as DrivAerML Surface and SHIFT-SUV do. ISLA's error does
  not change (0.989-0.999 of native). The baselines' errors rise 2.8-5.5× across fields, so ISLA leads them 3.1-4.8×; on the native
  mesh they lead ISLA by 3-6 %.
- The HiLift cost rows can move to the dedup lanes. That drops the stored-wing label in 08 and 19, and the cross-architecture cost
  ratios do not move.
- Confidence: firm. The two seeds agree within 15 % on every baseline ratio.

**md5s.**
- RESULTS.md 3e2efaff2a2dbaed9f93904ee92723f2
- NOTEBOOK_DRAFT.md 21c27372d43b2cee5ee77886b5ff4e3d
- Also area_table.md 784802a1, area_summary.json feb1aff2, cost_hldupx.json 313ddf9b.

**For you to decide.**
- Whether to apply NOTEBOOK_DRAFT's data and generator steps: copy the area scores into
  `results/campaigns_2026-10-23/hl_dup_stage2_scores`, add `area` to the dedup generator's PROTOS and to `standing.py`, and point
  `cost_090_2026-10-22.py`'s hilift map at the `hldupx_hl_*` lanes. I recommend yes.
- Turning on HiLift's area column also changes text outside 10, 08 and 19: 04:387/401/460, 07:690/709-710, and the counts in index,
  04 and 20, which go back to three tasks. The draft lists each one.
- Correction to record: HL-DUP's RESULTS_STAGE2 "Not done here" calls the area protocol "degenerate for ISLA". On HiLift it is not:
  it changes the scored cells for every arm, as CHAR-PROTOCOLS' audit of 2026-10-16 found.
- In the cost table, T s43's chain has a 9 s closing link, 371355, that the brief's job list omits. It is counted.

**Rulings relied on.** LANE_RULES scoring (float32, the scored set, the area protocol where defined, the non-degeneracy check);
Peter's "the book stays current" (2026-10-22); the supervisor's reading that the HL-DUP masks are allowed (2026-10-24). No new
rulings.

**Left / on disk.** Nothing is running. On LHR: `$T/kernel/hlarea_score`, `$T/recipe_hlarea_{isla,base}` (about 3.5 MB),
`$T/evals_hlarea`. Locally this folder is 1.6 MB.
