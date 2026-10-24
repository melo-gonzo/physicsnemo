# BC-SUPPLY: the book changes the branch makes

The lane did not edit the book. Every chapter's Python cells were run with MERGE-PREP's line-start-fence runner
(`scratch/isla-merge-prep-2026-10-24/probes/run_cells_all.py`), local CPU, nice 19, one chapter at a time, with
`book/run_book2.sh`:

- **base**: origin/psharpe/isla at eba038f08. It moved from 921472488 during the lane, and it already carries the book batch
  85689de4f on Peter's approval of the box and the far field.
- **merged**: psharpe/isla-bc-supply ee9237933 merged with eba038f08, as a temporary merge 8613ce640 (its temporary branch is
  deleted). Its tree, fab5bdffe, is the tree of the pushed branch head 0a706b148, so these results are the head's.

The run covered chapters 05, 06, 07, 10, 11, 20 and index. Outputs are `*_base.out` and `*_head.out` here (head is the
merged tree); the logs are `run_book_base_eba0.log` and `run_book_merged.log`. A first pass on 921472488 against the
unmerged branch (`v921/`) found the same changes. Chapters 07, 10, 11 and index print the same on both trees, but their
static prose still needs the edits listed below.

## Failing cells (merged tree; all pass on the base)

Chapter 20 fails 3 of its 36 cells. There is one cause: two of the cells fail only because the first one fails.

1. `para-limits-whole-boundary` (20-limits.qmd, about line 501) fails at `assert [d for n in body for d in WB_CFG[n]] ==
   ["HiLiftAeroML"], body`. `isla_surface_highlift` now reads `vehicle`, `inlet` and `outlet`, so no configuration reads
   the body alone and `body` is empty. The edit it needs:
   - add `"isla_surface_shift_suv": ["SHIFT-SUV"]` to `WB_CFG`;
   - drop the `body` assertion and the clause "and on HiLiftAeroML one mesh, the wing (...)";
   - list HiLiftAeroML (`vehicle`, `inlet`, `outlet`) and SHIFT-SUV (`vehicle`, `inlet`, `outlet`, `floor`, `side_walls`,
     `roof`) among the named-piece tasks;
   - change the bold lead from "read it on eight of the nine tasks, and SHIFT-SUV has none" to all nine tasks, and name
     HiLiftAeroML's symmetry plane as the one piece not read;
   - replace the SUV_DS assertion (`"domain_boundaries" not in SUV_DS`) with one that the first transform of
     `shift_suv_surface.yaml` is `AddBoxBoundaries`;
   - drop the sentences "SHIFT-SUV's files store the vehicle alone. The surface configuration reads the tunnel's four
     pieces beside it, and on these files the recipe's path resolver raises a `KeyError` ...; the recipe does not train
     ISLA on the vehicle alone. Peter approved building the rest of its boundary in the recipe". In their place, say that
     the recipe builds SHIFT-SUV's box and HiLiftAeroML's far field (`AddBoxBoundaries`, `AddHemisphereBoundaries`), and
     that `isla_surface_shift_suv` and `isla_surface_highlift` read them. `PRED` then gains "SHIFT-SUV names `vehicle`"
     from the configs.
2. `para-oq-far-field` fails with `NameError: name 'BA' is not defined`, because `BA` is defined in the failing cell above.
   It needs no edit of its own.
3. `para-oq-whole-boundary` fails with the same `NameError`. Its prose also needs the edits below.

## Printed values that change (cells that pass)

Every change in this table is generated from the configs, so none of them needs an edit.

| chapter, cell | base | merged |
|---|---|---|
| 05, the configurations' boundary table, `isla_surface_highlift` row | `vehicle` | `inlet`, `outlet`, `vehicle` |
| 05, the same table | no `isla_surface_shift_suv` row | 6 rows: `floor`, `inlet`, `outlet`, `roof`, `side_walls`, `vehicle` |
| 06, "Every one of the recipe's N ISLA model configurations" (2 places) | 8 | 9 |
| 06, the boundary-condition table and its caption | HiLift `vehicle`; no SHIFT-SUV | HiLift `inlet`, `outlet`, `vehicle`; SHIFT-SUV's six pieces; the caption adds "`isla_surface_shift_suv`: `vehicle`" |
| 06, the configuration table | no SHIFT-SUV row | a row for `isla_surface_shift_suv`: surface, 3, `pressure` (0) and `wss` (1), `U_inf_dir` (1), 0, whole model |

## Static prose that is no longer true

These passages are either in cells that pass or outside any cell.

- **20-limits.qmd, `para-oq-whole-boundary`** (about lines 1114-1160).
  - The "Not stored" item says "The recipe has no ISLA configuration for SHIFT-SUV, since its surface configuration reads
    the tunnel's four pieces, which these files lack". Replace it with: the recipe builds the box (`AddBoxBoundaries`, the
    first transform of the five SHIFT-SUV dataset files), and `isla_surface_shift_suv` reads it.
  - For HiLiftAeroML, add that the recipe builds the inlet and the outlet (`AddHemisphereBoundaries`, the first transform
    of `highlift_surface.yaml`) as curator built them: at radius 24,533.41796875 in (88.95 MAC), split by each case's stored
    `U_inf`. They match the far field stored on LHC001_AoA_10 and _12 bit for bit, in points and in oriented cells; only
    the order differs, since the stored order is curator's permutation. The 1,800 cases store 10 distinct `U_inf` vectors,
    one per angle of attack.
  - Keep the symmetry-plane text as it is.
  - In the "Peter approved ..." paragraph, put "the approved supply builds / takes" in the past tense, since it is merged.
  - The "In flight" callout: the configurations are now built; what stays in flight is the retraining and the root rebuild.
    Suggested text: "This question rests on the SHIFT-SUV and HiLiftAeroML standings, trained with the body alone.
    Canonical ISLA's configurations for both read the pieces above (psharpe/isla-bc-supply); the retrained standings and a
    rebuild of HiLiftAeroML's open root outlines from their stored vertices are in flight, and the ledger (@sec-ledger)
    tracks them."
  - The radius sentence "while the stored interiors reach {HL_MAC} chords" agrees with BC-SUPPLY's census and needs no
    change. On the 13 stored interiors that hold points (all from LHC001 and LHC002), the extent along +x, -x, +z and -z
    equals the radius to the float32 bit. Between those axes, the far field's own nodes reach 1.0188 times the radius
    (90.6 MAC), with 9,284 nodes beyond the radius on every case. Source:
    `scratch/isla-bc-supply-2026-10-24/out/hl_val_368914.jsonl`.
- **20-limits.qmd, lines 153-158** ("Not that every standing ...") says "and the recipe has no ISLA configuration for
  SHIFT-SUV; Peter approved building the missing pieces of both in the recipe". Replace it with: the recipe now builds
  the pieces and the configurations read them, while the standings still come from checkpoints trained with the body
  alone.
- **07-benchmark.qmd line 649**, the dataset table's SHIFT-SUV row, ISLA configuration column, says "none; Peter approved
  building the CFD box in the recipe (...)". Replace it with "`conf/model/isla_surface_shift_suv.yaml` (the standing's
  checkpoints were trained with `isla_surface.yaml` on the vehicle alone; @sec-oq-whole-boundary)".
  - Line 137, the model-input table's SHIFT-SUV row ("the vehicle alone, as the standing's checkpoints were trained"), is
    still true of the standing.
  - As DRV_DOM does for DrivAerML, that row could add that ISLA's configuration also reads the five box pieces. The same
    goes for the `hilift` row, with the inlet and the outlet.
- **10-hilift.qmd line 289** says "All but {HL_BUT} files store the wing alone, without the far field or the symmetry
  plane of the CFD domain, so every model here reads the body alone". Add after it:
  - the recipe builds the far field's inlet and outlet (`AddHemisphereBoundaries`), which canonical ISLA reads beside the
    deduplicated wing;
  - the symmetry plane is not built;
  - the baselines' inputs are unchanged, and the numbers here come from body-only checkpoints.
- **11-shift-suv.qmd lines 172 and 175** say "so every model here reads the body alone, and the recipe has no ISLA
  configuration for SHIFT-SUV ... raises a `KeyError`" and "Canonical ISLA's SHIFT-SUV configuration ... is being
  built". Replace them with: the recipe builds the box and `isla_surface_shift_suv` reads it; the numbers come from
  body-only checkpoints, and the retrain is in flight.
- **21-design.qmd line 283**, the design table's whole-boundary row, says "... nearly all of HiLiftAeroML's files store
  the wing alone, and SHIFT-SUV's store the vehicle alone, and no ISLA configuration of the recipe reads SH[IFT-SUV]
  ...". Replace it with: the recipe builds SHIFT-SUV's box and HiLiftAeroML's inlet and outlet as transforms, from the
  documented or stored domain constants and each case's stored freestream, and the configurations read them;
  HiLiftAeroML's symmetry plane is the one piece not read.
- **14-program-status.qmd** needs a BC-SUPPLY row. The text is the supervisor's ledger; the facts are in REPORT.md.

## Numbers for the book (sources in REPORT.md)

- **Cost per load.** The box transform takes 1.9-2.1 ms per load and the far field 3.5 ms (median of 50 calls, 8
  threads, LHR CPU).
- **Baseline inputs.** With the box, GeoTransolver's and Transolver's inputs and targets are bitwise unchanged on 60 of
  60 items (5 dataset files, 3 cases, augment off and on). With the far field they are unchanged on 24 of 24 items
  (3 dataset files).
- **Tokens.** SHIFT-SUV: 200,000 vehicle cells plus 4,332 box cells (inlet, outlet, floor and roof 722 each, side_walls
  1,444). HiLift: 93,364 deduplicated wing cells from a 100,000-cell draw, plus 3,904 far-field cells (inlet 1,952,
  outlet 1,952).
