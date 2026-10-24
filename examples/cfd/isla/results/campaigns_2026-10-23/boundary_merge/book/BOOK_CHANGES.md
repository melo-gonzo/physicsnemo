# MERGE-PREP 2c: what the boundary merge changes in book chapters 05 and 06 (for the next book batch)

Checked 2026-10-24 about 05:20 EDT. Every Python cell of `05-forward-pass.qmd` (41) and `06-using-isla.qmd` (32), run in
order from the book directory with `probes/run_cells_all.py`, CPU, on two trees:
- base: origin/psharpe/isla 7213c70c5 (`trees/isla7213`, a git archive of physicsnemo, examples/cfd/isla and the recipe);
- merged: psharpe/isla-boundary-merge e94d91c86 (the worktree).

The night lane's `run_cells.py` cuts a cell at the first triple backtick, and `setup-arch-source` prints a fence inside an
f-string, so 13 cells of chapter 05 "failed" on both trees. `run_cells_all.py` ends a cell only at a fence on its own
line; with it, base passes 41/41 and 32/32. Outputs: `05-forward-pass_{isla,merged}.out`, `06-using-isla_{isla,merged,merged_patched}.out`;
per-cell diffs: `06_cell_diff.txt`.

## Chapter 05: 41/41 cells pass on the merged tree; two cells print something new

- `tbl-arch-conditions` (the recipe's ISLA configurations' boundary conditions): `isla_surface` lists five conditions,
  `inlet`, `no_slip`, `outlet`, `slip` and `vehicle` (was `vehicle` alone); `isla_volume` likewise five (was `vehicle`);
  `isla_volume_airfrans` gains `freestream` (none, 3, 0) beside the airfoil. The caption and any prose that says the
  surface or Volume configuration reads only the vehicle, or AirfRANS only the airfoil, should follow.
- `code-stage-1` (the listing of stage 1 of `forward`): gains the 14-line `out_rows` block (the prediction boundaries' token
  rows). The text around the listing should name `prediction_boundaries`.

## Chapter 06: 11 of 32 cells fail on the merged tree, all from one cause

The demos call surface models without `prediction_boundaries`, which the merge makes a required forward argument
(SURFACE-BC's design: required for a surface task, refused for an interior one). Four cells raise
`ValueError: this model was built with interior=False (a surface task) and predicts at the cells of named boundaries;
pass prediction_boundaries ...`: `using-forward-demo`, `using-conditions-demo`, `using-checkpoint-demo`,
`using-contracts-demo`. The other seven (`tbl-using-shapes`, `para-using-conditions`, `using-draw-demo`, `para-using-draw`,
`para-using-checkpoint`, `para-using-refusals`, `tbl-using-contracts`) fail on names those cells did not define.

The needed edits (a scratch copy with exactly these passes 32/32: `06-using-isla_patched.qmd`):
- `using-forward-demo` (line 230): `surface(boundary_meshes={"airfoil": airfoil}, prediction_boundaries=["airfoil"], ...)`.
- `using-conditions-demo` (lines 321-323): `prediction_boundaries=["jet", "wall"]` for `a` and `b`, `["wall"]` for `alone`;
  line 331 (the undeclared-name refusal): `prediction_boundaries=["flap"]`, so that the refusal shown is still the
  undeclared name. The comment on line 324 ("One Mesh at the given conditions' cells, sorted by name") becomes "at the
  prediction boundaries' cells".
- `using-checkpoint-demo` (line 973): `loaded(..., prediction_boundaries=["airfoil"], ...)`.
- `using-contracts-demo` (line 1252): `surface(boundary_meshes={"airfoil": mesh}, prediction_boundaries=["airfoil"], ...)`.
- The unexecuted inference snippet (line 1035): `model(boundary_meshes={"vehicle": vehicle}, ...)` needs the tunnel and
  `prediction_boundaries=["vehicle"]` for a model trained with `isla_surface` after the merge.
- With those edits, every printed number of those cells is unchanged (a surface call that names every given boundary
  is bitwise the 0.11.0 one: `probes/bitwise_merge_isla_vs_boundary_merge_fix.txt`, `bitwise_merge_7213_vs_e94d91c86.txt`).

Two cells pass on both trees and print different text:
- `tbl-using-conditions`: as chapter 05's table (`isla_surface` and `isla_volume` read `boundaries.inlet`, `.no_slip`,
  `.outlet`, `.slip`, `.vehicle`; `isla_volume_airfrans` adds `boundaries.freestream`).
- `para-using-scales`: "on AirfRANS's, $\nu_t/(UL)$ has the median 7.5e-5" becomes "1e-7", and "3 to 4 orders of
  magnitude off" becomes "3 to 7". Source: AF-WHOLE's constants refit (`datasets/airfrans_uvolume_2d.yaml`
  `output_scale.nut` 7.511e-05 -> 1.045e-07; the length unit L now spans the far field). The cell's `_e` formatter
  (`.2g`) prints 1.045e-07 as "1e-7"; "1.0e-7" may read better.

## Recheck on the 2d head (2b15490b2, both parts merged) against psharpe/isla 1b25eb899, 07:00 EDT

Ten chapters that build or run ISLA (03, 04, 05, 06, 09, 10, 12, 19, 20, index), one tree per run, `run_cells_all.py`;
per-cell diffs in `book_diff_1b25_vs_2b15490b2.txt`. Every chapter passes on both trees except 06 on the merged tree
(the same 11 cells, the same cause). The seven edits above still apply; only the `using-contracts-demo` call moved, from
line 1252 to line 1288. With them, 06 passes 32/32. The differences are the four cells above (05 `tbl-arch-conditions`,
`code-stage-1`; 06 `para-using-scales`, `tbl-using-conditions`) and one more:

- 20 `para-limits-whole-boundary` (`ch20_diff.txt`): "on five of the nine tasks its configuration reads the body
  alone" becomes "on one of the nine tasks", and the list of tasks that read named pieces gains the DrivAerML Volume
  task, the AirfRANS interior task, the DrivAerML Surface task and SHIFT-SUV. **Two parts of the new text are wrong**,
  and the cell needs a change, not only new numbers:
  - The cell's `WB_CFG` maps `isla_surface` to both the DrivAerML Surface task and SHIFT-SUV. After the merge
    `isla_surface` names the tunnel's four pieces, which SHIFT-SUV's files do not store (SURFACE-BC REPORT section 5,
    BOUNDARY-AUDIT). So SHIFT-SUV still reads the body alone, and the true count is two of nine (SHIFT-SUV and
    HiLiftAeroML). The same mapping means `isla_surface` cannot run on SHIFT-SUV as it stands; see the report.
  - "every standing on those tasks was trained that way" now refers to HiLiftAeroML only. The DrivAerML and AirfRANS
    standings were also trained on the body alone, so that sentence, and "configurations that read all of it are built
    and have no standing", need rewording. "A surface task predicts at every cell it is given ... a task form that names
    the boundaries it predicts at ... is built and has no standing" is now the canonical surface task (Peter's ruling of
    23:26 EDT 10-23), so that sentence needs rewording too.

## The token padding alone (17c9df5c0 against 7213c70c5), for the same batch

`book_diff_7213_vs_17c9df5c0.txt` (from `book2/`): 3 cells. 05 `code-interior` and `code-read-only` show the padding
lines (`_pad_tokens`, `n_real`). 06 `tbl-using-contracts` shows float64 residuals at roundoff (rotation 6.5e-15 / 2.0e-15
becomes 6.4e-15 / 2.2e-15; geometry x7 1.7e-15 / 6.5e-16 becomes 1.4e-15 / 7.0e-16; measures x5 1.2e-15 / 5.6e-16 becomes
1.3e-15 / 5.9e-16; split cells 1.4e-15 / 5.6e-16 becomes 1.4e-15 / 6.0e-16). The demo's 80-cell airfoil pads to 96 tokens.

No chapter text was edited.
