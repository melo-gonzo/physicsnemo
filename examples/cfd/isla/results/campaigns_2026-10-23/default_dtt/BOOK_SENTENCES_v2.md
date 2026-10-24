# Book sentences that state or imply the interior default, v2 (against HEAD e10681b33)

Scope: `examples/cfd/isla/book/{03-isla,05-forward-pass,06-using-isla,21-design}.qmd` in worktree
`physicsnemo-isla-default-dtt`, HEAD e10681b33. That HEAD merges 0d46b2f9b, the AeroZoo book batch, which already moved
the prose to "ISLA-DTT is ISLA's default interior form" and added "In flight" callouts saying that the switch of the code
defaults is being built. v1 was written against eba038f08 (the book at b0fd82da8 is the same). Line numbers below are from the
current files. "cell" = a string in a python cell (its label is given); "prose" = markdown; "caption" = a `#| fig-cap`, mermaid
or table caption.
**[must]** = false, or the rendered output is wrong, after the change; **[should]** = implies the old default; **[opt]** = true
as written, but could name the default.

What the change does (checked in the worktree): the signature default is `query_independent: bool | None = None`, and
`self.query_independent = self.interior if query_independent is None else bool(query_independent)` (model.py L809), so `None`
gives ISLA-DTT on the interior task and False on the surface task. Every interior config states `query_independent: true`,
`compile: false`, `compile_blocks: true`. The surface configs keep `compile: true`.

Status key for v1 items: **unchanged** (same text, new line), **fixed** (dropped), **reworded** (new text given, then
re-rated).

---

## 03-isla.qmd

| v1 | status | current |
|---|---|---|
| 1 | unchanged | L1371–1373 |
| 2 | **fixed**: L1476–1477 now "In interacting ISLA, the interior form with `query_independent=False`, …" | — |
| 3 | **fixed**: L1488–1489 now "ISLA-DTT (`query_independent=True`), ISLA's default interior form, makes the points **read-only**" | — |
| 4 | **fixed**: both halves of the `fig-isla-dtt` caption (L1505–1509) are now "interacting ISLA" / "ISLA-DTT, ISLA's default interior form" | — |
| 5 | **fixed**: cell rewritten (L1582–1595, see the new items below). The "open question (@sec-oq-dtt)" sentence is gone | — |
| 6 | **fixed**: the AeroZoo `S.in_flight_callout` ("…which decides the interior form") is removed. A new default-switch callout replaces it (see the In-flight section) | — |
| 7 | unchanged | L1195 |
| 8 | unchanged | L1642 |

1. **L1371–1373, prose (Idea 7 lead) [opt]**, unchanged: "On an interior task the prediction points join the boundary cells
   as tokens, and each one is told only how far it is from the wall and in which direction."
   → optional: "…join the boundary cells as tokens (read-only by default, below), and each one…".

2. **L1195–1196, caption `fig-routing-other-tasks` [should]**, unchanged: "ISLA at its reference configuration, the
   standing's eagerly trained checkpoint on each task, seed 42, …" (the AirfRANS interior and lug checkpoints
   `r090eag_*_isla_*` are interacting). After the change, ISLA at its reference configuration on an interior task means
   ISLA-DTT.
   → "Interacting ISLA (`query_independent=False`) at its reference configuration, the standing's eagerly trained
   checkpoint …".

3. **L1488–1489, prose [opt]** (v1#3 as fixed): "ISLA-DTT (`query_independent=True`), ISLA's default interior form, makes
   the points **read-only**" is true. Optional: "…ISLA's default interior form (what `interior=True` alone builds), …".

4. **L1582–1583, cell `para-isla-dtt-standing` [new; true]**: "**ISLA-DTT is ISLA's default interior form, on principle: its
   prediction at a point does not depend on which other points are queried.**" No edit.

5. **L1594, same cell [new; opt]**: "The interacting form stays in the class: the surface task uses it, and an interior task
   may choose it explicitly." True after the change. Optional: "…may choose it explicitly (`query_independent=False`)."
   L1595 ("The interior standings of Part IV are of interacting ISLA, and each says what ISLA-DTT measures there.") is true; no
   edit.

6. **L1642, table `tbl-isla-ideas` row 7 [opt]**, unchanged: "…its two distance channels; interacting or read-only points"
   → "…; read-only (default) or interacting points".

7. **L1596–1598, same cell, In-flight callout [must]**: delete (see the In-flight section).

Not affected: L63 comment, L195–198, L245, L1027, L1129 and the measurement sentences L1583–1593.

## 05-forward-pass.qmd

| v1 | status | current |
|---|---|---|
| 1 | unchanged | L100–103 |
| 2 | unchanged | L1069–1077 |
| 3 | **reworded**, still [must] | L1089–1096 |
| 4 | unchanged | L1105–1110 |
| 5 | unchanged | L1166–1167 |
| 6 | unchanged | L1674 |
| 7 | unchanged | L1807–1809 |
| 8 | **fixed**: L2035–2044 now "…; ISLA-DTT is ISLA's default interior form all the same, on principle.**" and "On AeroZoo's held-out splits, the test registered to choose between the forms, the read is a close call, and Peter's ruling…". The AeroZoo callout is removed | — |
| 9 | unchanged | L2359–2378 |
| 10 | **reworded**, now [should] | L2578 |
| 11 | unchanged | L2577 |
| 12 | unchanged | L2583 |
| 13 | unchanged | L2601–2603 |
| 14 | unchanged (no edit) | L2557–2562 |

1. **L100–103, cell `para-arch-reference` [opt]**, unchanged: "The class defaults are the reference configuration of ISLA:
   `hidden=…`, `n_layers=…`, `n_slices=…`, and on the interior task `boundary_tokens=…`". Consistent after the change.
   Optional: append "…, and on the interior task the read-only prediction points of ISLA-DTT".

2. **L1069–1077, cell (unlabelled, after `para-arch-reference`) [must, the output changes]**, unchanged. L1072 is
   `INT_ = ISLA(**kw); DTT_ = ISLA(**kw, query_independent=True)`, and `kw` contains `interior=True`, so `INT_` becomes
   ISLA-DTT. L1074–1075, "and, in ISLA, the one routing-mass scalar `qt.logw`, {cnt(INT_) - TOT:,} in all", would then print
   the DTT count under the interacting label, and `INT_.qt.logw` stops existing.
   → code: `INT_ = ISLA(**kw, query_independent=False); DTT_ = ISLA(**kw)`. Sentence: "…and, in interacting ISLA, the one
   routing-mass scalar `qt.logw`…". Comment L1069–1070: "(ISLA)" → "(interacting ISLA)".

3. **L1089–1090, prose (`sec-arch-modes`) [must]**, reworded. It now reads: "They come in two kinds: *interacting*
   prediction points (interacting ISLA, the class's default, @tbl-arch-switches) carry routing mass and write into the
   slices; *read-only* prediction points (ISLA-DTT, `query_independent=True`) carry none and read the boundary cells directly
   instead (@sec-arch-passive-decoder)." "the class's default" becomes false.
   → "*interacting* prediction points (interacting ISLA, `query_independent=False`) carry routing mass and write into the
   slices; *read-only* prediction points (ISLA-DTT, `query_independent=True`, the class's default on the interior task,
   @tbl-arch-switches) carry none and read…".
   The new sentences at L1092–1096 ("ISLA-DTT is ISLA's default interior form; Peter's ruling … The interacting form stays the
   surface task's, and an explicit choice on an interior task.") are true; no edit.

4. **L1109, mermaid caption `fig-isla-modes` [opt]**, unchanged: "interacting points (ISLA) carry a learned routing mass,
   read-only points (ISLA-DTT) carry none" → "read-only points (ISLA-DTT, the interior default) carry none …; interacting
   points (ISLA, `query_independent=False`) carry a learned routing mass".

5. **L1166–1167, prose [should]**, unchanged: "**Interacting prediction points: routing mass.** In ISLA the prediction points
   take part in every softmax over tokens." → "…In interacting ISLA (`query_independent=False`) the prediction points…".

6. **L1674, prose (`sec-arch-passive-decoder` lead) [should]**, unchanged: "With `query_independent=True` (ISLA-DTT) the
   prediction points are read-only" → "With `query_independent=True` (ISLA-DTT, the interior default) …".

7. **L1807–1809, prose (`sec-arch-tree-read`) [should]**, unchanged: "ISLA-DTT is ISLA with read-only prediction points, the
   global read after every layer and the forest at $\theta = 1$, all set by `query_independent=True`." → "…all set by
   `query_independent=True`, the interior default."

8. **L2359–2378, prose (`sec-arch-cost`) [must]**, unchanged:
   - L2359 "**The whole model compiles to one graph.**" → "**The surface model compiles whole to one graph.**" [should]
   - L2363 "The recipe compiles ISLA whole, `Mesh` inputs and output included, …" → "The recipe's surface configurations
     compile ISLA whole, …" [must]
   - L2372–2374 "ISLA-DTT is the exception. … so it compiles each slice block separately instead (`compile_blocks=True`,
     @sec-arch-switches)." → "ISLA-DTT, the interior default, is the exception. … so every interior configuration of the
     recipe compiles each slice block separately instead (`compile: false`, `compile_blocks: true`, @sec-arch-switches)."
     [must]
   - L2374–2378 "So does ISLA on AeroZoo: … so the whole-model graph recompiles case after case, and
     `isla_volume_aerozoo.yaml` compiles per block." → "Interacting ISLA on AeroZoo would need the per-block compile too: …
     the whole-model graph recompiles case after case." [must: the config now compiles per block because it is ISLA-DTT]

9. **L2578, cell `tbl-arch-switches`, `WHAT["query_independent"]` [should]**, reworded. It now reads: "interior task only:
   `True` is ISLA-DTT, … after every slice layer (@sec-arch-passive-decoder), ISLA's default interior form
   (@sec-aerozoo-form); `False` is interacting ISLA; rank-2 boundary fields are refused with `True`". The row's default
   column will print `` `None` ``, and nothing in the row explains it.
   → prefix: "`None` resolves to `True` on the interior task and to `False` on the surface task; interior task only: `True`
   is ISLA-DTT, …".

10. **L2577, `WHAT["interior"]` [opt]**, unchanged: "…which join the slice layers as tokens with their exact geometry…" →
    "…as tokens (read-only by default, `query_independent`) with their exact geometry…".

11. **L2583, `WHAT["compile_blocks"]` [opt]**, unchanged: "The way to compile ISLA-DTT, whose boundary read breaks the
    whole-model graph (@sec-arch-cost)" → "…breaks the whole-model graph, and so every interior recipe configuration's
    (@sec-arch-cost)".

12. **L2601–2603, `tbl-arch-switches` caption [should]**, unchanged: "…every other default is the surface task with no global
    inputs." → "…every other default is the surface task with no global inputs, except `query_independent`, whose `None`
    follows the task (ISLA-DTT on the interior task)."

13. **L2604–2605, same cell, In-flight callout [must]**: delete `print()` and the callout (see the In-flight section).

Not affected: L32, L231, L513, the L1186 code callout and its listings, L1956 ("interacting ISLA, which builds no forest"),
L2433 caption.

## 06-using-isla.qmd

| v1 | status | current |
|---|---|---|
| 1 | unchanged; rating lowered to [should] (true as written, says nothing of the form) | L80 |
| 2 | **fixed**: the callout "Interacting or read-only prediction points" (L127–136) now opens "ISLA's default interior form is ISLA-DTT (`query_independent=True`)…" and says to deploy *interacting* ISLA at its training count | — |
| 3 | unchanged | L149 |
| 4 | unchanged | L383–386 |
| 5 | unchanged | L543 |
| 6 | unchanged | L801–809 |
| 7 | unchanged | L831–845 |
| 8 | unchanged | L879 |
| 9 | **reworded**, now [should] | L1079–1083 |
| 10 | unchanged | L1303–1308 |
| 11 | unchanged | L1327–1330 |
| 12 | unchanged | L1417 |

1. **L80, prose (`sec-using-config`) [should]**: "and an interior model adds `interior=True`." → "and an interior model adds
   `interior=True`, which builds ISLA-DTT; `query_independent=False` builds interacting ISLA."

2. **L128, callout "Interacting or read-only prediction points" [opt]** (v1#2 as fixed): "ISLA's default interior form is
   ISLA-DTT (`query_independent=True`), whose…" is true. Optional: "…ISLA-DTT (`query_independent=True`, what `interior=True`
   alone builds)…".

3. **L149, cell `para-using-query-population` [should]**: "On the DrivAerML Volume task, the standing's ISLA asked for its
   scored points in four chunks…" → "…the standing's interacting ISLA asked for…".

4. **L383–386, prose (measures) [should]**: "…the routing mass of the $Q$ interior prediction points, $\omega + \log\sum_i
   w_i - \log Q$ with a learned $\omega$ (@sec-arch-modes), shifts by the same constant;" → "…the routing mass of
   interacting ISLA's $Q$ prediction points (ISLA-DTT's carry none), … shifts by the same constant;".

5. **L543, prose (Budget) [opt]**: "…so the draw has static shapes and compiles into the one whole-model graph." → "…has
   static shapes, so it compiles into a whole-model graph or into the per-block graphs."

6. **L801–809, recipe commands [must]**:
   - L801 comment "# DrivAerML Volume task: boundary -> interior, interacting prediction points. …" → "# DrivAerML Volume
     task: boundary -> interior, read-only prediction points (ISLA-DTT), compiled block by block. …"
   - L806–809 ("# The same interior task with read-only prediction points (ISLA-DTT): …" and
     `… model.query_independent=true compile=false model.compile_blocks=true`) is now the config's own default, so it is
     redundant. Delete it, or replace it with: "# The same interior task with interacting prediction points." and
     `python src/train.py model=isla_volume dataset=drivaer_ml_volume_reference model.query_independent=false`. The
     per-block compile still works for the interacting form; add `compile=true model.compile_blocks=false` only if the
     whole-model compile is wanted.

7. **L831–845, cell `tbl-using-configs` [must: the caption]**: the code is correct as is. All five interior rows will show
   "interior, read-only" and "per block"; the three surface rows show "whole model". Caption L842–843: "…or the per-block
   `compile_blocks` (the AeroZoo configuration, whose four boundaries recompile the whole-model graph)." → "…or the per-block
   `compile_blocks` (every interior configuration: ISLA-DTT's boundary read breaks the whole-model graph)." Optional label
   "interior, read-only (ISLA-DTT)".

8. **L879, cell `para-using-optimizer` [should]**: "…in an interior model, the bias of the prediction points' input layer and
   their routing mass) under AdamW." → "…the bias of the prediction points' input layer and, with interacting points, their
   routing mass)…".

9. **L1079–1083, prose (inference) [should]**, reworded. It now reads: "Deploy interacting ISLA at its training point count
   (200,000 …), because an interacting point's prediction depends on the joint request (@sec-interior-companion). A model
   built with `query_independent=True` (ISLA-DTT) has no such rule and is queried at any count, alone or in chunks." True,
   but it reads as if ISLA-DTT were opt-in.
   → "…The default interior model, ISLA-DTT (`query_independent=True`), has no such rule and is queried at any count, alone
   or in chunks."

10. **L1303–1308, cell `para-using-kernel` [must, the output changes]**: "**Compile the whole model.** The recipe's ISLA
    configurations do, all but {_rest}: `compile: true` …". `_rest` will list all five interior configurations.
    → "**Compile the whole model on the surface task.** The recipe's surface configurations do ({_whole}): `compile: true`
    …; the interior configurations compile per block (below)."

11. **L1327–1330, same cell [must]**: "It is the way to compile ISLA-DTT (`compile=false model.compile_blocks=true`), whose
    boundary read breaks the whole-model graph, and the AeroZoo configuration uses it because that dataset's four boundaries
    recompile the whole graph." → "It is the way to compile ISLA-DTT, whose boundary read breaks the whole-model graph, and
    every interior configuration uses it (`compile: false`, `compile_blocks: true`); interacting ISLA on AeroZoo needs it too,
    since that dataset's four boundaries recompile the whole graph." Keep "The standing's training runs use it on every
    dataset but Tensile2d" (historical).

12. **L1417, prose (pose check) [opt]**: "a read-only model (ISLA-DTT) can also be checked…" → "the default read-only model
    (ISLA-DTT) can also be checked…".

13. **L162–166, In-flight callout [must]**: delete (see the In-flight section).

New in 06: L159 ("What the two forms measure … @sec-aerozoo-form.") is neutral. The geometry-cost rewrite (L1145–1245) does
not touch the default. Not affected: L66, L113, L242/L249, L569/L576, L1128 cost caption.

## 21-design.qmd

| v1 | status | current |
|---|---|---|
| 1 | unchanged | L41 |
| 2 | unchanged | L49 |
| 3 | **fixed**: L50 now "interacting ISLA's interior tokens carry routing mass … / ISLA-DTT's carry none" | — |
| 4 | unchanged | L268 |
| 5 | **fixed**: L277–279 row now "Read-only prediction points on the interior task, interacting ones as an option", "`query_independent=True` (ISLA-DTT), the default interior form, makes interior tokens read-only; …", evidence "@sec-aerozoo-form" | — |
| 6 | unchanged | L336–337 |

1. **L41, cell `tbl-key-names` row "ISLA" [should]**: "`physicsnemo.experimental.models.ISLA` as it is, the class
   defaults" → "…, the class defaults; beside ISLA-DTT on an interior task, the interacting form (`query_independent=False`)".

2. **L49, row "ISLA-DTT" [should]**: "`query_independent=True`: prediction points are read-only…; interior tasks only" →
   "`query_independent=True`, the interior default: prediction points are read-only…; interior tasks only".

3. **L268, `tbl-principled-ledger` row "Prediction-point routing mass" [opt]**: "the prediction points' total routing weight
   is…" → "interacting prediction points' total routing weight is…".

4. **L277, same table (v1#5 as fixed) [opt]**: true. Optional: "`query_independent=True` (ISLA-DTT, what `interior=True`
   alone builds), the default interior form, …".

5. **L336–337, prose (`sec-principled-ask`) [should]**: "The class carries one construction and the read-only interior form
   as its only switch;" → "The class carries one construction and the interacting interior form (`query_independent=False`)
   as its only switch;".

6. **L308–311, cell `tbl-principled-ledger`, In-flight callout [must]**: delete `print()` and the callout (see the In-flight
   section).

Not affected: L164, L170, L189–192, L238, L253.

---

## In-flight callouts to remove on merge (the switch of the code defaults)

These are all the callouts of this kind in the book (grepped across `*.qmd`; none exist outside these four chapters). Each
says the code still builds interacting ISLA by default and that the switch "is being built". All of them become false at
merge.

| chapter | lines | where / id | first sentence |
|---|---|---|---|
| 03 | L1596–1598 | `print(...)` at the end of cell `para-isla-dtt-standing` (sec-isla-interior) | "The class builds an interior model as interacting ISLA unless `query_independent=True` is given, and so does the recipe." |
| 05 | L2604–2605 (`print()` + `print('::: {.callout-note title="In flight"}…')`) | end of cell `tbl-arch-switches` (sec-arch-switches) | "The class's and the recipe's interior defaults here are the code as it is: `query_independent` defaults to `False`, interacting ISLA." |
| 06 | L162–166 (markdown block) | after the callout "Interacting or read-only prediction points", before `## What the model needs from your data {#sec-using-inputs}` | "The class's and the recipe's interior defaults here are the code as it is: `query_independent` defaults to `False`, interacting ISLA." |
| 21 | L308–311 (`print()` + the callout `print`) | end of cell `tbl-principled-ledger` (sec-principled) | "The class and the recipe build an interior model as interacting ISLA unless `query_independent=True` is given." |

Each callout's second sentence: "The switch of both defaults to ISLA-DTT for the interior task is being built, and the ledger
(@sec-ledger) tracks it." In 06, deleting L162–166 leaves `:::` (L160) followed directly by a blank line and the next heading,
which is fine. On merge, also set the ledger row DEFAULT-DTT (14-program-status.qmd L718, now "RUNNING … merges with its own
book batch") to DONE.

### Stay: callouts on the interior standings and other pending measurements (not about the default)

| chapter | lines | id | first sentence | why it stays |
|---|---|---|---|---|
| 05 | L1885–1890 | markdown callout in `sec-arch-tree-read` (forest precision) | "This limit is that of the code as it is." (it continues: "ISLA-DTT retrained with it on the lug and the AirfRANS interior task is being measured") | ISLA-DTT retrain with the forest built outside autocast |
| 05 | L1962–1967 | markdown callout in `sec-arch-tree-read` (pose-matched forest) | "This limit is that of the code as it is." (it continues: "ISLA-DTT retrained with it … read for native accuracy by a registered rule, is being measured") | ISLA-DTT retrain with the posed forest |
| 05 | L2414–2419 | markdown callout in `sec-arch-cost` (compiled-block certification) | "These numbers compare the compiled blocks with eager." (it continues: "…is measured first in the reference rerun, on every dataset") | awaits the final reference rerun |
| 06 | L1138 (conditional footnote mark ‡ in `tbl-isla-cost`, not a callout block) | `tbl-isla-cost` | "‡ …: the lane is still training, priced on the epochs logged so far." | printed only while a cost lane is in flight |

03 and 21 have no standings callouts. The interior standings themselves are of interacting ISLA, and the text already says so
(03 L1595, "The interior standings of Part IV are of interacting ISLA…"). It needs no callout.

---

## Elsewhere (v1 items re-checked; not a new sweep)

- **04-invariances** L152–156: "In ISLA's interacting interior form, … ISLA-DTT (`query_independent=True`) makes its
  prediction points read-only" is still there and true. Optional: "ISLA-DTT, the interior default, …". New at L8–9: "(interacting
  ISLA's; ISLA-DTT, the default interior form, is independent of them)" is true. The L871 In-flight callout concerns the
  posed ÷ native ratios of the standings: **stay**.
- **09-drivaerml** L815 (was L796), cell `setup-dtt-params`: `N_ISLA, N_DTT = _n(_ISLA(**_kw)), _n(_ISLA(**_kw,
  query_independent=True))`, with `interior=True` in `_kw`, is **still [must]**: `N_ISLA == N_DTT` after the change. →
  `_ISLA(**_kw, query_independent=False)`. L828 (was L809) "ISLA-DTT is ISLA with `query_independent=True`": [opt] "…, the
  interior default". New at L840 and L959 ("ISLA's default interior form is ISLA-DTT (@sec-aerozoo-form)…"): true.
- **12-airfrans** L570–572 and L961–962: **fixed**. They now read "One planar dataset does not choose ISLA's interior form …
  ISLA-DTT is ISLA's default interior form, on principle" (L571–573) and "No choice of ISLA's interior form … ISLA's interior
  form is ISLA-DTT, on principle, after a close call…" (L962–964).
- **13-aerozoo** L425: "ISLA-DTT: the same configuration with read-only prediction points (`query_independent=True`)" is
  unchanged [opt]. L644 ("…decides … which interior form ISLA uses"): **fixed** (gone; `sec-aerozoo-form` at L700 states the
  ruling). New at L768–769: "On an interior task it stays available as an explicit choice, `query_independent=False`" is true
  after the change. The L881 callout (outlet gauge pressure) is unrelated: stay.
- **14-program-status** L205: **fixed** ("DONE (the default interior form, by Peter's ruling on AeroZoo's close call)"). L563
  (DTT row): **fixed**. The history row keeps "the interacting form stays the class default" but appends "2026-10-24: …
  ISLA-DTT is the default interior form by Peter's ruling…". **New action**: L718 DEFAULT-DTT "RUNNING" → DONE on merge.
- **20-limits** L401–406: **fixed** (now "ISLA-DTT is ISLA's default interior form: on AeroZoo's…"). The L605 open-question
  row and the `sec-oq-dtt` section (old L673–695): **fixed** (removed; no `@sec-oq-dtt` reference remains in the book,
  only `sec-oq-dtt-frame` and `sec-oq-dtt-theta`). New at L319, L1338 and L1362 ("ISLA-DTT, the default interior form"):
  true.
- **22-literature** L414: "ISLA-DTT (`query_independent=True`) makes interior prediction points passive" is unchanged
  [opt]. L457 ("Which interior form is the better default is an open question"): **fixed** (L452–455 now "…ISLA's default
  interior form, on principle (@sec-aerozoo-form)").
- **index** L73: **fixed** (L73–74, "…makes ISLA-DTT … ISLA's default interior form, on principle"). L221 key row "ISLA-DTT
  … ISLA's reference configuration with read-only prediction points (interior tasks)" is unchanged [opt]: "…(interior
  tasks; the interior default)".

## Residual risks

- The book imports the installed `physicsnemo` (`/home/psharpe/gh/physicsnemo`), not this worktree, and `CONF`/`RECIPE`
  derive from it. The cells whose output changes (05 L1069–1077 and `tbl-arch-switches`; 06 `tbl-using-configs` and
  `para-using-kernel`; 09 `setup-dtt-params`) show the new behaviour only when rendered against the changed class and
  configs. Rendering was not run; every edit above is a wording proposal.
- Naming: in Parts III–IV, an unqualified "ISLA" beside "ISLA-DTT" on interior tasks means the interacting form. Only the
  instances in the four chapters (plus the v1 Elsewhere items) are listed. Other chapters were not re-swept for an
  unqualified "ISLA at its reference configuration" on interior tasks.
- I lowered v1's 06#1 (L80) from [must] to [should], because the sentence is true as written; this is a judgement call.
