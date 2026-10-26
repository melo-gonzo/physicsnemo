# Book references the switch makes stale (not edited; for the supervisor)

`git grep` of examples/cfd/isla/book at 7f7e1c09a (the book is the same as at BASE 5c39a6b52) for isla_cumsum, _cumsum,
_systematic_positions, sampling.py, tree_read, canonical_frame, frame_coordinates, CANONICAL_SKEW_TIE and
pps_inclusion_probabilities. No hit for isla_cumsum, _cumsum or _systematic_positions.

## Break the render (import or listing of a deleted name)
| file:line | text | fix |
|---|---|---|
| 05-forward-pass.qmd:376 | `from physicsnemo.experimental.models.isla.sampling import sample_boundary_cells, pps_inclusion_probabilities` | import `pps_inclusion_probabilities` from `physicsnemo.mesh.sampling` (sample_boundary_cells stays in ISLA) |
| 05-forward-pass.qmd:471 | `listing("sampling.py", "pps_inclusion_probabilities")` | the function is no longer in ISLA's sampling.py: list it from `physicsnemo/mesh/sampling/systematic_sampling.py` (and `systematic_sample` beside it), or drop the listing |
| 20-limits.qmd:1185 | `from physicsnemo.experimental.models.isla.tree_read import CANONICAL_SKEW_TIE` | import from `physicsnemo.mesh.spatial._frame` |

## Stale wording (renders, names the old home)
| file:line | text | fix |
|---|---|---|
| 05-forward-pass.qmd:383 | `pi_d = pps_inclusion_probabilities(a_d.double(), torch.tensor(float(BUDGET_D), ...))` | works once :376 imports the library's function (it takes a tensor or float share) |
| 05-forward-pass.qmd:417 | "(`pps_inclusion_probabilities`) and $u$ the start" | name the library function (physicsnemo.mesh.sampling) |
| 05-forward-pass.qmd:470, :472 | `listing("sampling.py", "water_fill")`, `listing("sampling.py", "sample_boundary_cells")` | still valid; the listed sample_boundary_cells now calls the library and carries the zero-measure fill |
| 20-limits.qmd:1146 | "positive (`tree_read.canonical_frame`)" | `physicsnemo.mesh.spatial._frame.canonical_frame` |
| 18-notebook.qmd:43564 | "the boundary's canonical frame (`tree_read.canonical_frame` over every cell ..." | historical notebook entry: leave, or add a note that the frame moved to the library |

## Not stale
- 03-isla.qmd:1643 (`tree_read.py` in the table: the file stays), 05-forward-pass.qmd:1937 (`canonical_frame` by name),
  05-forward-pass.qmd:2269-2270 (tree_read listings of BoundaryForest.point_pairs and node_means), 06-using-isla.qmd:143
  (`TREE_READ_THETA` stays in tree_read), 18-notebook.qmd:36008/40499/41128/41186/43687/43896 (historical, by file name),
  20-limits.qmd:1190/1192 (`canonical_frame`, `CANONICAL_SKEW_TIE` by name; the value is unchanged),
  21-design.qmd:276 (test_isla_sampling.py's exact count, HT sums and spread tests are kept).

## For the book (behaviour)
- The interior draw now uses physicsnemo.mesh.sampling (#2174). Outside one edge it is bitwise the 0.14.0 DRAW-EXACT draw.
  The edge is a condition with zero-measure cells whose water-fill share is at least its count of positive-measure cells.
  There, every positive cell is now kept (pi = 1), and the zero-measure cells fill the rest of the share at weight 0.
  Before, every cell got pi = 1, the line ran past k, and cells at the end of the line were dropped. No real case in the
  probe reaches this edge: none of 38 cases over 5 datasets has a zero-measure cell.
