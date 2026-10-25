# ISLA-API 013: the book changes psharpe/isla-api-013 makes (ISLA 0.13.0)

The lane did not edit the book. The book's text is the same on the branch head and on origin/psharpe/isla fc6ed19d0
(`git diff fc6ed19d0 HEAD -- examples/cfd/isla` is empty), so every difference below comes from the code.

**How the cells ran.** MERGE-PREP's line-start-fence runner (`scratch/isla-merge-prep-2026-10-24/probes/run_cells_all.py`),
local CPU, nice 19, under the local_heavy lock, one chapter at a time (`book/run_book.sh`). Two trees:
- **base2**: fc6ed19d0 (git archive; the results folder symlinked, identical). Outputs `book/<chapter>_base2.out`.
- **head**: the branch at 7b60bdf86 (physicsnemo and test identical to the pushed head 7c88b2e1f, recipe differing by one
  test line). Outputs `book/<chapter>_head.out`.
- An earlier run on 5b70c8686 (`book/<chapter>_base.out`) covers 01, 02, 03, 13, 14 and 18. Those chapters print
  identically on the head, so base2 did not rerun them.

## Cells that break (30 cells in 4 chapters; every other chapter's cells pass and print identically)

"Root" marks a cell that breaks on the API itself. "Cascade" marks a cell that breaks only because a root cell defined
nothing.

| chapter | cell | error on head | cause |
|---|---|---|---|
| 05 | para-arch-reference | KeyError 'boundary_tokens' | root: reads the constructor's defaults by the old name |
| 05 | tbl-arch-conditions | TypeError: unexpected keyword 'reynolds' | root: builds ISLA with `reynolds=` |
| 05 | fig-arch-block-layout | KeyError 'reynolds' | root |
| 05 | para-arch-params | TypeError: unexpected keyword 'query_independent' | root: `ISLA(**kw, query_independent=False)` (interacting interior) beside ISLA-DTT |
| 05 | code-read-only | StopIteration in `listing(...)` | root: lists `ISLA._query_tokens` from `start="if self.query_independent:"`, which is gone |
| 05 | tbl-arch-switches | KeyError 'n_boundary_tokens' | root: its table of constructor switches is keyed by the old names |
| 06 | para-using-arguments | AssertionError: the grouped argument list (`dim`, `reynolds`, `boundary_tokens`, `query_independent`) does not match the signature | root |
| 06 | using-forward-demo | TypeError: unexpected keyword 'dim' | root |
| 06 | using-conditions-demo | TypeError: unexpected keyword 'dim' | root |
| 06 | using-draw-demo | TypeError: unexpected keyword 'boundary_tokens' | root |
| 06 | code-using-tensile2d | KeyError 'dim' | root: reads `model.dim` from isla_tensile2d.yaml (now `n_spatial_dims`) |
| 06 | tbl-using-configs | KeyError 'dim' | root: the same key, in the configurations table |
| 06 | using-checkpoint-demo | NameError 'surface' | cascade from using-forward-demo (it also asserts `args["query_independent"] is False`, line 1043, which would fail next) |
| 06 | tbl-using-shapes, para-using-conditions, para-using-draw, para-using-checkpoint, para-using-refusals, using-contracts-demo, tbl-using-contracts | NameError | cascade |
| 09 | setup-dtt-params | TypeError: unexpected keyword 'query_independent' | root: `_ISLA(**_kw, query_independent=False)` counts interacting ISLA's parameters |
| 09 | para-dtt-lede, make-tbl-interior-read-only, para-dtt-reading, make-tbl-interior-read-ladder, para-dtt-ladder, para-qpop | NameError | cascade |
| 19 | setup-cost-sweep | KeyError 'boundary_tokens' | root |
| 19 | para-cost-lede, tbl-cost-slopes | NameError 'VB' | cascade |

Fixes: rename `dim`, `boundary_tokens` to `n_spatial_dims`, `n_boundary_tokens` in the cells and in the config keys they read.
Drop `reynolds=` and the viscosity field from the cells' constructions, and declare `re_z: {rank: 0}` where a cell needs
the AeroZoo or AirfRANS global layout. The parameter counts of interacting interior ISLA (05 para-arch-params, 09
setup-dtt-params) can no longer be computed from the class. Take them from the 0.9.0 checkpoints the standings use, or
state them as numbers with their source. The interior standings stay interacting ISLA 0.9.0's until the final rerun, as the
book says. 05 code-read-only needs a new `start=` marker (the passive `q_logw` line).

## Cells that print differently but pass

| chapter | cell | change |
|---|---|---|
| 05 | code-boundary-draw | the listed source reads `self.n_boundary_tokens` |
| 05 | code-interior | the listed `_QueryTokens.__init__` has no `routing_mass` and no `logw` parameter |

Both are source listings, so they follow the code automatically. Check that the surrounding prose does not describe the
routing-mass parameter (it does: see BOOK_PROSE.md, 05).

## Prose naming the removed API as a default or a choice

The full census is in `BOOK_PROSE.md` (read-only child, 25bea52cb). It has 86 passages in section A, across chapters 03,
04, 05, 06, 07, 09, 12, 13, 15, 16, 20, 21, 22, index and the undated part of 14. Each row gives the quoted phrase, what is
wrong and a suggested replacement. Section B holds borderline phrases that name a past measurement by its old flag.
Section C counts the dated records in 14 and 18 and leaves them unchanged. Section D notes the code that stops the
render. The main kinds:
- `query_independent` as a flag or default (`query_independent=True`, "ISLA's default interior form", "`query_independent=False`
  builds interacting ISLA", the recipe's `query_independent: true`, the override `model.query_independent=false compile=true`);
- interacting ISLA offered as an interior option ("read-only (the default) or interacting points");
- `dim` as the constructor argument or config key (06, 15, 12); `boundary_tokens` (03, 04, 05, 06, 19);
- the model's Reynolds number, viscosity and density roles and `velocity_scale` (03 *Scales*, 05, 06, 12, 13). The new
  wording: the recipe computes `re_z` (ComputeBoundaryScales, ComputeLogReynolds in ISLA's two model configs), and a field
  `defines` a scale;
- renamed tests: `test_query_independent_interior_contracts` is now `test_interior_task_contracts`, and
  `test_boundary_tokens_is_validated` is now `test_n_boundary_tokens_is_validated`.

Prose that reports the interior standings as interacting ISLA 0.9.0's stays as it is.
