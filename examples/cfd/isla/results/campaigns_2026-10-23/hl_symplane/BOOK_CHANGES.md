# HL-SYMPLANE: the book changes the branch makes

The lane did not edit the book. Every Python cell of chapters 05, 06, 07, 10, 11, 20, 21 and index was run with
MERGE-PREP's line-start-fence runner (`scratch/isla-merge-prep-2026-10-24/probes/run_cells_all.py`), on local CPU at nice 19, one
chapter at a time (`book/run_book.sh`, log `book/run_1424.log`), on two trees:
- **base**: origin/psharpe/isla at 142485ef0, which already carries the supervisor's book text for Peter's approval of the
  plane, as a `git archive` in `book/base1424/`.
- **merged**: the branch head bc4081e75 merged with 142485ef0. Its tree is 63d98ae91 (`git merge-tree`; it merges
  cleanly), archived in `book/merged/`.

Outputs are in `book/<chapter>_{base,head}.{out,err}`, where head is the merged tree. An earlier run against the branch's
own base b4b10d5a9 (`book/v_b4b1/`) found the same printed changes.

## Failing cells

None. Every cell passes on both trees (0 FAIL in all 16 outputs).

## Printed values that change

All three are generated from the configs, so none needs an edit.

| chapter, cell | base | merged |
|---|---|---|
| 05, the configurations' boundary table, `isla_surface_highlift` | `inlet`, `outlet`, `vehicle` | adds the row `symmetry`, none, 2, 0 |
| 06, the boundary-condition table | HiLift `inlet`, `outlet`, `vehicle` | adds `symmetry` (none), `boundaries.symmetry` |
| 20, `para-limits-whole-boundary`, the list of named pieces | HiLiftAeroML: `vehicle`, `inlet`, `outlet` | HiLiftAeroML: `vehicle`, `inlet`, `outlet`, `symmetry` |

## Prose that is no longer true, with replacement text (line numbers are 142485ef0's)

The sources for the new numbers are, in this folder:
- `FINAL_REPORT.md`;
- `out/census_summary.txt` (from the record `out/hl_symmetry_check_medium.json`, md5 bc2ed653);
- `out/curval_374370.jsonl`;
- `out/move_374330.jsonl`;
- `gb/smoke_summary.txt`.

A results folder such as `results/campaigns_2026-10-24/hl_symplane/` holding them would let the cells read them.

1. **10-hilift.qmd, line 289.**
   - Replace "The symmetry plane is not read. It is the far field's $y = 0$ disk with the closed root cap that the dataset's
     own source STL files hold cut out, applied to a case only after a check that the case's own root points lie on that
     cap, so that a case failing the check gets no plane (@sec-oq-whole-boundary)." with:
     "The dataset's reader builds the symmetry plane (`AddSymmetryPlane`): the far field's $y = 0$ disk with the closed
     root cap of the dataset's own source STL files cut out. A case gets it only after a check that the case's own stored
     root vertices and open root edges lie on the cap's outline, to within 0.134 in, a tolerance derived from the cap's
     chord sagitta and float32 storage. All 690 cases pass, the worst vertex lying 0.061 in from the outline
     (@sec-oq-whole-boundary)."
   - "canonical ISLA's configuration reads them beside the deduplicated wing" becomes "...reads the three pieces beside the
     deduplicated wing".
   - "retrains ISLA with the inlet and outlet" becomes "retrains ISLA with the inlet, the outlet and the symmetry plane".
2. **10-hilift.qmd, the "In flight" callout at lines 291-295** ("HiLiftAeroML's boundary here lacks its symmetry plane.
   ... is being built"). Remove it. The standing's own in-flight callout for the retrain, if any, stays.
3. **07-benchmark.qmd, `cell-tbl-model-inputs`, the `hilift` row** (line 135). ISLA's cell lists only "boundary
   cells; `U_inf_dir`", which was already stale after BC-SUPPLY. Give it the DrivAerML rows' clause: "; ISLA's
   configuration also reads the far field's `inlet` and `outlet` and the symmetry plane `symmetry`, built at load, named
   conditions without fields, predicting at the wing's cells alone, and the standing's checkpoints were trained with the
   wing alone (@sec-standing-geometry)".
4. **20-limits.qmd, `para-not-claimed`, the whole-boundary item** (lines 153-158). "the recipe builds SHIFT-SUV's box and
   HiLiftAeroML's inlet and outlet" becomes "the recipe builds SHIFT-SUV's box and HiLiftAeroML's inlet, outlet and
   symmetry plane".
5. **20-limits.qmd, `para-limits-whole-boundary`.**
   - The bold lead is a literal string, so the run does not change it. Cut its clause "and on HiLiftAeroML every piece
     but the symmetry plane": the configurations now read the whole boundary on all nine tasks.
   - "HiLiftAeroML's inlet and outlet (`AddHemisphereBoundaries`)" becomes "HiLiftAeroML's inlet and outlet
     (`AddHemisphereBoundaries`) and, in its reader, its symmetry plane (`AddSymmetryPlane`)".
   - Cut "HiLiftAeroML's symmetry plane is the one piece no configuration reads."
   - Optional: beside the FIRST_T assertion, assert that the `whole_surface_transforms` of `highlift_surface.yaml` name
     `AddSymmetryPlane`.
6. **20-limits.qmd, `para-oq-whole-boundary`, the "Not stored" item.** Its last sentence (lines 1075-1077) is "So
   HiLiftAeroML's symmetry plane is the far field's $y = 0$ disk with that cap cut out, built at load case by case, and a
   case that fails the check gets no plane; the configuration does not read it (In flight, below)." Replace it with:
   "So the reader builds HiLiftAeroML's symmetry plane, the far field's $y = 0$ disk with that cap cut out, for each case
   that passes the check. A case passes when its own on-plane vertices and open root edges lie on the cap's outline to
   within 0.134 in, the largest bound on the sagitta of any of the cap's chords plus float32 storage precision, derived
   from the cap alone. All 690 cases pass, and their verdicts are kept in one fingerprinted record. On every case the worst
   vertex lies 0.061 in from the outline, 0.46 of the tolerance. The worst open-edge midpoint lies 0.060 in from it on 689
   cases; on geo_LHC024_AoA_8 it lies 0.127 in, 0.94 of the tolerance, where two root edges of 20 in span a curved stretch
   of the outline. `isla_surface_highlift` reads the plane as `symmetry`, the paper's no-stress condition."
7. **20-limits.qmd, the paragraph that begins "Peter approved building the missing pieces"** (lines 1081-1101).
   - Replace "Its symmetry plane, the far field's $y = 0$ disk with the source STL's root cap cut out, each case checked
     against its own root points (above), is not built." with:
     "Its symmetry plane is built in the reader as the curation pipeline's generator builds one: the far field's equator,
     with the cap's outline as the hole, triangulated by the generator's own constrained Delaunay code, every normal into
     the fluid. On LHC001_AoA_10 and _12, the built plane has the stored plane's equator bit for bit. Its area differs from
     the stored plane's by 4.6e-9 of the whole. Its hole lies within 0.061 in of the stored hole, curator's walk of the CFD
     wing's own root, and holds 0.99999 of that hole's area."
   - The baselines' parenthesis gains ", 24 of 24 with the symmetry plane".
   - The data list gains this lane's report.
8. **20-limits.qmd, the "In flight" callout after that cell** (lines 1104-1108, "HiLiftAeroML's symmetry plane is not
   read. ... is being built"). Remove it.
9. **20-limits.qmd, the open-questions table, the whole-boundary row** (lines 623-625). "SHIFT-SUV's box from constants and
   HiLiftAeroML's inlet and outlet at the stored far-field radius" becomes "SHIFT-SUV's box from constants and
   HiLiftAeroML's inlet and outlet at the stored far-field radius and its symmetry plane from the source STL's root cap".
10. **21-design.qmd, line 283**, the row "The whole boundary of the domain (recipe)".
    - "so the recipe builds SHIFT-SUV's CFD box and HiLiftAeroML's inlet and outlet as transforms, from the documented or
      stored domain constants and each case's stored freestream" becomes "...and HiLiftAeroML's inlet, outlet and symmetry
      plane as transforms, from the documented or stored domain constants, each case's stored freestream and the
      dataset's source STL root cap".
    - Cut "HiLiftAeroML's symmetry plane is the one piece not read;".
11. **06-using-isla.qmd, about line 687.** "and the HiLiftAeroML one the wing, the only piece all but two of its files store"
    was already stale after BC-SUPPLY. It becomes "and the HiLiftAeroML one the wing, the far field's `inlet` and `outlet`
    and the symmetry plane `symmetry`, the last three built at load since all but two of its files store the wing alone".

## Ledger and notebook (the supervisor's)

- **14-program-status.qmd.** Add a row for HL-SYMPLANE:
  - status: DONE 2026-10-24;
  - result: 690 of 690 cases pass; psharpe/isla-hl-symplane at bc4081e75, pushed and not merged;
  - question: "Can HiLiftAeroML's symmetry plane be built at load for every case from the dataset's own source STL root
    cap, after a check of each case's own root?".
- **18-notebook.qmd.** A new entry on this lane's result. FINAL_REPORT.md holds the numbers and their sources.
- **The standing** is still that of checkpoints trained with the wing alone. The final reference retrain (@sec-ledger) now
  reads the wing, the inlet, the outlet and the symmetry plane.
