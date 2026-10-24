# DEFAULT-DTT: book changes for the merge of psharpe/isla-default-dtt

Branch head e10681b33 (= 9d27ae47b model, b0fd82da8 recipe, e1a6d7853 tests, merged with psharpe/isla 0d46b2f9b).
The supervisor applies these with the merge; no book file was edited by this lane.

## 1. Cells whose output or behaviour changes

How measured: `book/run_cells_all.py` (md5 202a05db, from scratch/isla-merge-prep-2026-10-24/probes, fences matched at
line start), CUDA hidden, nice 19, local CPU, one chapter at a time, on the merged base 0d46b2f9b
(`git archive` in base_tree/, PYTHONPATH=base_tree) and on the branch e10681b33 (PYTHONPATH=worktree). Chapters: 03, 04,
05, 06, 07, 09, 10, 12, 13, 19, 20, 21, index (every chapter whose cells build or import ISLA). All 348 cells pass on both
trees. Output differs in 3 chapters, 10 printed lines: `book/book_diff_0d46_vs_e106.txt`; raw outputs
`book/<chapter>_{base,branch}.out`. (The same 10 lines differed at eba038f08: `book/eba038f08/`.)

| chapter, cell | base -> branch | what is wrong | smallest edit |
|---|---|---|---|
| 05, the unlabelled cell after `para-arch-params` (L1069-1077: `INT_ = ISLA(**kw); DTT_ = ...`) | "in ISLA, the one routing-mass scalar `qt.logw`, **2,049** in all" -> "**1,214,352** in all" | `INT_` (kw has `interior=True`) is now ISLA-DTT, so the interacting count prints ISLA-DTT's | `INT_ = ISLA(**kw, query_independent=False); DTT_ = ISLA(**kw)`; text "in ISLA" -> "in interacting ISLA"; comment L1069-1070 "(ISLA)" -> "(interacting ISLA)" |
| 05, `tbl-arch-switches` (L2578) | row `query_independent` (`False`) -> (`None`) | correct now, but the row never says what `None` means | prefix `WHAT["query_independent"]` with "`None` resolves to `True` on the interior task and to `False` on the surface task; " |
| 06, `using-draw-demo` / `para-using-draw` (L564-591) | "two training-mode calls ... differ by **8.9e-03**" -> "**1.3e-02**" | nothing: the demo's interior model is now ISLA-DTT, and the sentence (two draws give different predictions) holds | none (the number re-renders). To keep the old number instead, add `query_independent=False` to the demo's `ISLA(...)` |
| 06, `tbl-using-configs` (L826-845) | the five interior rows: task "interior" -> "interior, read-only", compile "whole model" -> "per block" (aerozoo was already per block) | rows correct; the caption's reason for per-block is now wrong | caption L842-843 "(the AeroZoo configuration, whose four boundaries recompile the whole-model graph)" -> "(every interior configuration: ISLA-DTT's boundary read breaks the whole-model graph)" |
| 06, `para-using-kernel` (L1290-1335) | "The recipe's ISLA configurations do, all but `isla_volume_aerozoo`:" -> "all but `isla_lug`, `isla_tensile2d`, `isla_volume`, `isla_volume_aerozoo`, `isla_volume_airfrans`:" | true but backwards: whole-model compile is now the surface configs' | L1308 -> "**Compile the whole model on the surface task.** The recipe's surface configurations do ({', '.join(f'`{n}`' for n in _whole)}): ..." and L1327-1330 "It is the way to compile ISLA-DTT (`compile=false model.compile_blocks=true`), ..., and the AeroZoo configuration uses it because that dataset's four boundaries recompile the whole graph." -> "It is the way to compile ISLA-DTT, whose boundary read breaks the whole-model graph, and every interior configuration uses it (`compile: false`, `compile_blocks: true`); interacting ISLA on AeroZoo needs it too, since that dataset's four boundaries recompile the whole graph." |
| 09, `setup-dtt-params` (L809-815) | "The boundary read adds **1.2** million parameters to ISLA's **69** million" -> "adds **0.0** million parameters to ISLA's **70** million" | `N_ISLA = _n(_ISLA(**_kw))` with `interior=True` now counts ISLA-DTT | `_ISLA(**_kw, query_independent=False)` |

No other cell's output changes: 03, 04, 07, 10, 12, 13, 19, 20, 21 and index print identically. Cells that build
`ISLA(interior=True)` from the configs (05 L231 `tbl-arch-conditions`, L513 `fig-arch-block-layout`; 06 L242
`using-forward-demo`) now build ISLA-DTT and print the same.

## 2. In-flight callouts to remove on merge (the switch of the code defaults)

All four say the code still builds interacting ISLA by default and that the switch "is being built"; all become false at
merge (found by grep across every chapter; none outside these four):

| chapter | lines | where | first sentence |
|---|---|---|---|
| 03 | L1596-1598 | end of cell `para-isla-dtt-standing` | "The class builds an interior model as interacting ISLA unless `query_independent=True` is given, and so does the recipe." |
| 05 | L2604-2605 (`print()` + callout `print`) | end of cell `tbl-arch-switches` | "The class's and the recipe's interior defaults here are the code as it is: `query_independent` defaults to `False`, interacting ISLA." |
| 06 | L162-166 (markdown) | after the callout "Interacting or read-only prediction points", before `## What the model needs from your data` | same first sentence as 05 |
| 21 | L308-311 (`print()` + callout `print`) | end of cell `tbl-principled-ledger` | "The class and the recipe build an interior model as interacting ISLA unless `query_independent=True` is given." |

Also: the ledger row DEFAULT-DTT (14-program-status.qmd L718, "RUNNING") -> DONE, with this lane's numbers (FINAL_REPORT.md).

Stay (interior standings and other pending measurements, not the default): 05 L1885-1890 (forest precision retrain),
05 L1962-1967 (pose-matched forest retrain), 05 L2414-2419 (compiled-block certification in the final rerun), 06 L1138
(the ‡ mark of `tbl-isla-cost`, printed only while a cost lane trains), 04 L871 (posed ÷ native of the standings), 13 L881
(outlet gauge). The interior standings are of interacting ISLA and the text already says so (03 L1595).

## 3. Sentences of 03, 05, 06 and 21 that state the interior default

Full list with quotes, line numbers and edits: `BOOK_SENTENCES_v2.md` (against e10681b33; v1 `BOOK_SENTENCES.md` was
against eba038f08, and 0d46b2f9b already fixed 13 of its items). The ones that are false after the merge ([must]):

- 05 L1089-1090 (`sec-arch-modes`): "*interacting* prediction points (interacting ISLA, the class's default,
  @tbl-arch-switches)" -> move "the class's default on the interior task" to the read-only kind (ISLA-DTT).
- 05 L2359-2378 (`sec-arch-cost`): "The recipe compiles ISLA whole" -> "The recipe's surface configurations compile ISLA
  whole"; "ISLA-DTT is the exception ... so it compiles each slice block separately" -> "... so every interior configuration
  of the recipe compiles each slice block separately (`compile: false`, `compile_blocks: true`)"; "So does ISLA on AeroZoo
  ... and `isla_volume_aerozoo.yaml` compiles per block." -> "Interacting ISLA on AeroZoo would need the per-block compile
  too: ...".
- 06 L801-809 (recipe commands): the comment "interacting prediction points" on the `model=isla_volume` command becomes
  "read-only prediction points (ISLA-DTT), compiled block by block"; the second command
  (`model.query_independent=true compile=false model.compile_blocks=true`) is now the config's default: replace it with
  the interacting variant `model.query_independent=false` (add `compile=true model.compile_blocks=false` for its
  whole-model compile).
- The cell items of section 1 (05 `para-arch-params`, 06 `tbl-using-configs` caption, 06 `para-using-kernel`) and the four
  callouts of section 2.

[should] items (true but imply the old default or name "ISLA" for the interacting form), all in BOOK_SENTENCES_v2.md:
03 L1195 caption `fig-routing-other-tasks` ("ISLA at its reference configuration" for interacting checkpoints); 05 L1166,
L1674, L1807, L2578, L2601-2603; 06 L80, L149, L383-386, L879, L1079-1083; 21 L41, L49, L336-337.
Elsewhere, beyond the four chapters: only 09 `setup-dtt-params` (section 1) is false; the other v1 "elsewhere" items were
fixed by 0d46b2f9b.
