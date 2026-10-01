# Review: the ISLA book

*A reader's report for the owner. I have never seen this project. I read index.qmd and chapters 01–15 once, in order, as an expert in PDE numerics and SciML would. I cannot check numbers against `results/`, so I dispute no value; I flag only numbers whose scope is missing and places where the book states one quantity two ways.*

---

## A. Verdict

No: an uninitiated expert cannot follow this book on one pass, and the loss happens on the first page of content. The thesis table has grown to sixteen rows that mix three different ISLA configurations ("the reference configuration", "the earlier protocol's 192-wide ISLA", "the 24-layer depth variant"), three cell budgets (200,000, 100,000, "cells are flat above 100,000"), two instruments, and a vocabulary — *rung, the reporting protocol, with no option, the parameter match, interim, in flight* — none of which is defined until chapters 05 and 06. A reader who cannot parse the thesis table cannot calibrate anything that follows, and the front matter then spends two more sections (the names-and-colors key, "What this book does not claim") on material that belongs at the back. The second loss is chapter 07, which opens with seventy lines of setup and then a single computed paragraph of about 350 words carrying twenty numbers, every one of which is restated in the two tables immediately below it; chapters 08, 09 and 10 repeat that opening pattern. The third loss is cumulative: by chapter 10 the reader has met pose in four places, discretization in five, and the output-component clause in five, each time with slightly different scope, and by chapter 15 the entire standing is restated a third time in a new table and a new figure. The underlying disease is visible and is exactly what the owner describes: the book has stopped deleting. New evidence is appended beside old evidence rather than replacing it, migration state ("their reference re-runs are in flight") is documented inside documentation chapters, and every new dataset arrives with a full set of its own sections rather than a row in an existing table. The skeleton underneath is good — the arc, the figures, the honesty, chapter 13 — and almost all of the damage is removable by cutting and moving rather than rewriting.

---

## B. The structural problems

### B1. The front matter is four documents, and the thesis table is the evidence chapters in miniature

**Where:** `index.qmd`, "## The thesis {#sec-verdict}", "## Names, symbols and colors", "## What this book does not claim".

**Why it hurts.** The thesis paragraph states eleven standings in prose; `tbl-thesis` then restates the same eleven plus five more; chapters 07–10 state them a third time; chapter 15 a fourth. The table's caption is itself ~350 words and ends with a fifteen-artifact source list. Worse, four rows are *not* the shipped model — the pose, sampling-law and AirfRANS-extrapolation rows are the 192-wide lane, the capacity row is the 24-layer variant — so the reader's first impression of "what ISLA scores" is an average over configurations the book elsewhere tells them not to use. Two rows also carry program status in the front matter: "their reference re-runs are in flight, @sec-program-status" and "each await seeds or lanes in flight". A reader meeting the project for the first time is being handed a migration log.

**Fix.** One thesis table, one row per dataset *and task*, reference configuration only, eight rows: DrivAerML Surface (435 cars, both GeoTransolver widths in one row), DrivAerML Surface ladder, HiLiftAeroML, DrivAerML Volume, AirfRANS interior, the solids (lug + Tensile2d), the fixed-mesh sets, pose + sampling. Everything measured on a non-reference lane moves into the chapter that owns it. Cut the caption to five sentences and put the artifact list in a `results/` manifest the caption names once. Move "Names, symbols and colors" and `fig-key` to an appendix (the book uses direct labels, so the key is rarely needed). Fold "What this book does not claim" into chapter 11, which already says the same eight things.

### B2. Equivariance and pose live in four chapters; discretization in five

**Where:** pose — `03-invariances.qmd` §"Equivariant by construction, or invariant by canonicalization" (`tbl-pose`), `08` §`sec-pose-contract` (`tbl-pose-200k`), `08` §`sec-pose-prior` (`tbl-pose-prior`), plus `01` §"What generalization means here" and `02` §"Why ISLA has no center". Discretization — `02` idea 1, `03` Part B, `08` §§`sec-robustness-density`, `sec-consistency-bench`, `sec-training-law`, `sec-territories`, `13` §"The ledger", `15` §`sec-vs-novel-measure`.

**Why it hurts.** The reader never learns where the canonical statement is. Chapter 03 is nominated as "the centerpiece", yet its pose evidence (`tbl-pose`) is bfloat16 at 10,000 cells — the weakest instrument in the book — while the reporting-protocol pose standing is sixty pages later. Meanwhile chapter 08 contains a fourth pose section about an input the book *rejects* (`sec-pose-prior`), whose numbers are then re-used in chapter 07's wing discussion, re-stated in 13's ledger and again in 15. Each repetition carries slightly different scope, so the reader cannot tell whether they are seeing one result or four.

**Fix.** Chapter 03 defines both invariances, states the contract, names the instruments, and carries no table. Move `tbl-pose` (canonicalization control) into chapter 08 beside `tbl-pose-200k`, so one section holds: the contract, what canonicalization buys, the posed standing. Move the pose-prior diagnostic (`sec-pose-prior`) to appendix 13, where the ruling that rejects the input already lives; chapters 07 and 08 cite it in one sentence each.

### B3. Chapter 07 is three chapters; chapter 10 is two

**Where:** `07-surface-accuracy.qmd` — §"The DrivAerML Surface task", §"HiLiftAeroML", §"A second surface task: DeepJEB++ brackets". `10-boundary-to-interior.qmd` — §§"The DrivAerML Volume task"/"The second dataset: AirfRANS" against §§"A solid: PLAID Tensile2d", "A second solid: the GINOT bracket lug", "Boundary data on a fixed mesh: the OLBC sets".

**Why it hurts.** 07's first page promises cars and wings; 4,000 words later a bracket dataset with a *different* training design (equal optimizer steps, not equal epochs) arrives without warning, and its caveat ("these rows are equal-compute statements only") must be carried forward by the reader. 10's first page says "the task that asks for the field rather than the wall pressure"; the reader then meets three datasets where geometry is fixed or nearly so and every mesh node is a query — not the boundary-to-interior problem the chapter set up. Both chapters also host a section that belongs elsewhere: 07's §`sec-deficit-anatomy` is a cell-count/cost argument whose every number is the earlier protocol's, and 10's §`sec-olbc-ablations` is a design-decision ledger entry.

**Fix.** Split 07 into "Surface accuracy on DrivAerML" and "Data efficiency and the wing deficit". Split 10 into "Boundary to interior" (DrivAerML Volume, AirfRANS) and "Beyond external aerodynamics" (Tensile2d, lug, DeepJEB++, OLBC) — the latter absorbing 07's DeepJEB section. Move `sec-deficit-anatomy` into chapter 09 as two sentences; move `sec-olbc-ablations` to appendix 13.

### B4. Chapter 15 restates the whole book before it says anything new

**Where:** `15-isla-vs-geotransolver.qmd` §"Where ISLA is better and where it is worse" (`tbl-vs-standing`, `fig-vs-standing`), §"Ideas from the literature worth trying" (`tbl-vs-ideas`).

**Why it hurts.** `tbl-vs-standing` has 26 rows and is the thesis table plus the chapter standings, recomputed from the same artifacts — a third full statement of the evidence, and in places a *different* one (see D-15). `fig-vs-standing` then draws it twice more in two panels with different readouts, which breaks "one idea per figure". `tbl-vs-ideas` duplicates chapter 11's open-questions table and chapter 12's PLANNED rows. The genuinely new content — §"Novel contributions of ISLA beyond the state of the art" and the bibliography with its read/not-read honesty — is the best argument in the book for why anyone outside should care, and it is buried behind 2,000 words of restatement.

**Fix.** Delete §"Where ISLA is better and where it is worse" entirely; replace with one paragraph pointing at `tbl-thesis`. Delete `tbl-vs-ideas`; its rows belong in chapter 11's open-questions table (which already has the right format: question / what would settle it / bar). Retitle the appendix "ISLA against the literature" and keep (a)–(h), the ranked list and the bibliography.

### B5. Program status leaks into every documentation chapter

**Where:** `05` §"Configuration of the evidence lanes" ("**Every standing in the book is the reference configuration's … wherever its lane has been scored.** The rows that still carry the earlier protocol's ISLA … are labeled 'earlier protocol' … and the re-run of @sec-program-status re-measures them"); `10` ("A cell marked *training* has not landed", "the 2,048-case elasticity reference marked *interim* is the same configuration trained on an earlier code snapshot"); `06` ("**the one option in flight**, `slice_prototypes`"); `04`'s switches table ("the one option in flight"); `index` ("in flight"); `08` ("the price was measured before the ruling that set it (2026-09-27…)"); and roughly eighty `@sec-nb-*` citations across 01–11.

**Why it hurts.** This is the mechanism by which the book "gets worse as more things are added": every new lane that lands adds a label, a caveat and a pointer rather than replacing a number. The reader of a documentation chapter is being asked to hold a model of the project's work queue. It also breaks rule 1 directly — `08` carries a date, `13` carries five, and the chapters narrate retraction ("the one the class no longer carries", "was built, priced and removed", "The convention this replaced … is gone from the class", "The earlier routing probes … indexed the flat list of calls").

**Fix.** Three rules, mechanically applied. (1) A documentation chapter states only what has been measured under the reference configuration; a lane not yet re-run is either reported as the earlier protocol *in the table cell only* (no prose, no cross-reference to status) or the row is absent. (2) No cell says "training" or "interim" — an unlanded rung is not a row. (3) Documentation chapters cite `results/*.json` and other chapters; only chapter 12 cites `@sec-nb-*`. That one change removes several hundred cross-references and most of the history.

---

## C. Proposed table of contents

The arc of STYLE.md is right. The chapter *boundaries* are not. Minimal restructuring, same four parts:

1. **The problem and how we measure** — the boundary-to-field operator, its general interface, and the one metric with its instrument. *(01, unchanged bar the unkept promise in D-01.)*
2. **ISLA, one idea at a time** — why each part is there, on the airfoil; ends with the trained routing pictures as the payoff. *(02 + `sec-arch-routing-trained` from 04; standings removed.)*
3. **The two invariances** — what equivariance and discretization consistency mean, what each buys, what each does not, and the instruments that read them. *(03, with no results tables.)*
4. **The ISLA forward pass** — the stage-by-stage reference with every shape and identifier. *(04, minus the routing pictures and the erratum.)*
5. **Baselines and the fair-comparison protocol** — GeoTransolver and Transolver, what each sees, the datasets in one table, the nine rules. *(05; per-dataset figures to appendix B.)*
6. **Using ISLA** — one configuration, the inputs, the commands, the sampler, checkpoints, cost. *(06, now the only home for cost-per-step.)*
7. **Surface accuracy on DrivAerML** — the 435-car standing against both GeoTransolver widths, the training-car ladder and where the crossover sits.
8. **Data efficiency and the wing deficit** — the HiLiftAeroML ladders, what the deficit is and is not, and its anatomy.
9. **Discretization and pose** — the draw, the training law, refinement, slice territories, and the posed standing. *(08 + `tbl-pose`; pose-prior out.)*
10. **Capacity, scaling and cost** — cells, width, depth, the shared floor, the fitted law and the compute-optimal rule, the frontier.
11. **Boundary to interior** — DrivAerML Volume and AirfRANS, query tokens, query dependence, what does not work.
12. **Beyond external aerodynamics** — Tensile2d, the lug, DeepJEB++, the OLBC sets: what the general interface buys and where it trails.
13. **Limits and open questions** — one home for every scope statement and every open readout with its bar.
14. **Program status** — the tech tree and the ledger (the one chapter with dates).

Appendices: **A** Design decisions; **B** Datasets in detail; **C** ISLA against the literature; **D** Lab notebook.

---

## D. Chapter-by-chapter notes, by severity

### index.qmd

- **"The thesis", opening sentence.** "**ISLA is built around two invariances that no other slice-attention surrogate has, rigid-motion equivariance, which is exact, and consistency under a change of boundary discretization, which holds in the fine-mesh limit, and at its reference configuration it is more accurate than GeoTransolver at the recipe width on the DrivAerML Surface task in less memory.**" Sixty-one words, four clauses, two parentheticals rendered as commas; not quotable (rule 3). **Rewrite:** "**ISLA is a slice-attention surrogate that is exactly equivariant to rigid motions and consistent under a change of boundary discretization, and it pays nothing in accuracy for either.** The two invariances are stated in @sec-invariances. The standing against GeoTransolver and Transolver is @tbl-thesis: one row per dataset and task, all at the reporting protocol of @sec-protocol."
- **`tbl-thesis`, rows 5–8 and 11.** "(ISLA under the earlier protocol, 192 wide; the reference configuration's factors are in @sec-robustness-density)". The front matter sends the reader into chapter 08 to get the real number for a thesis row. **Fix:** carry the reference configuration's factor (1.01x at 200,000 cells) and drop the row's parenthetical; move the 192-wide factor to 08.
- **`tbl-thesis` caption.** "the pose, sampling-law and AirfRANS-extrapolation rows hold the earlier protocol's 192-wide ISLA (their reference re-runs are in flight, @sec-program-status)". Status in the front matter (rule 1). **Cut** the parenthetical.
- **Thesis paragraph then table.** The paragraph already says "ISLA leads GeoTransolver-256 by 7% at full data and from 109 training cars up, in 0.7x its memory", "On the HiLiftAeroML 35-wing rung GeoTransolver and Transolver both lead by a quarter", and nine more standings the table repeats. **Fix:** cut the paragraph to four sentences (what ISLA is, what it is compared against, where it leads, where it trails) and let the table carry the rest.
- **"the baselines" unnamed.** "the baselines' error is {…}% below ISLA's" (row 4 reading). Rule 4. **Rewrite:** "GeoTransolver-256 and Transolver-256 are both … below ISLA".
- **Key table, "deficit anatomy" row.** "the decomposition of the in-distribution gap into measured parts | @sec-deficit-anatomy". The target section is now "The anatomy of the cell-count gap", and chapter 07 says there is no in-distribution gap left at the protocol. The key defines a concept the book no longer carries. **Cut the row.**
- **"What this book does not claim".** Eight bullets that chapter 11 states again ("**Not that discretization consistency is an exact invariance**" ≈ 03's "Consistency is a fine-mesh-limit property, not an exact invariance"; "**Not that the HiLiftAeroML numbers are test-set numbers**" ≈ 11 §`sec-limits-validation`). **Move to 11**, leave one sentence in the front matter pointing there.
- **"How to read this book".** "Surface accuracy and data efficiency … (@sec-surface). Discretization and pose (@sec-robustness-density). Capacity, scaling and cost (@sec-shared-floor)." Two of the four evidence chapters are cited by a *mid-chapter section label*. **Fix:** give each chapter a chapter-level label and cite that.
- **Ratio direction.** "in 0.7x its memory" and (05) "0.7x the memory", (07) "0.85x step time", (09) "0.85x the step time", (06) "ISLA trains at 0.71x GeoTransolver-256's peak memory" all point ISLA ÷ baseline, against rule 3's "every ratio is baseline divided by ISLA". The same quantity also appears as "0.7x", "0.71x" and "29% less memory" (tbl-cost-200k caption) — three renderings, three precisions. **Fix:** one direction, one precision, computed, everywhere.

### 07-surface-accuracy.qmd

- **Chapter opening.** There is no prose before `setup-surface`; the first thing the reader sees is a computed paragraph beginning "**On the DrivAerML Surface task ISLA leads GeoTransolver at its recipe width by 7% at 435 training cars…**" and running ~350 words with roughly twenty numbers, every one repeated in `tbl-scoreboard-435` and `tbl-hilift-ladder` below. Rules 3 and 7. **Rewrite the lede to three sentences:** "**At full data on the DrivAerML Surface task ISLA leads GeoTransolver at its recipe width and ties its parameter match at half the memory; on every HiLiftAeroML rung both baselines lead.** The standings are @tbl-scoreboard-435 (435 cars, with cost), @tbl-scoreboard-200k (the training-car ladder) and @tbl-hilift-ladder (the wings). Protocol: @sec-protocol; every ratio is baseline ÷ ISLA."
- **§"The anatomy of the cell-count gap" (`sec-deficit-anatomy`).** Ends: "Every number in this paragraph is the earlier protocol's." An entire section of the flagship evidence chapter whose every number is from a configuration the book tells you not to use, about a gap that no longer exists at the protocol. **Move** two sentences of it to chapter 09 (§"Cells per case are flat above 100,000") and **cut** the rest; keep the deployment recommendation, which already lives in 06 §`sec-using-sampler`.
- **§"A second surface task: DeepJEB++ brackets".** "the ladder holds the optimizer-step budget fixed at about 130,000 steps while the training set grows … It is therefore an equal-compute comparison at each rung and not a learning curve." Good honesty, wrong chapter. **Move to the new chapter 12** with Tensile2d and the lug.
- **§`sec-de-result`, wing rungs.** The 210-case row uses reference-configuration lanes against GeoTransolver-512, while chapter 15's "210 random cases" row uses the earlier-protocol arms, giving two different values for one named quantity. **Fix:** 15 should not restate it at all (B4); if it must, it carries the same lane.
- **§`para-wing-reading`.** ~400 words, twelve inline computed values, four artifacts named in prose. **Fix:** promote the controls (width, depth, rate, cell count, GeoTransolver-512) into a five-row table "what does *not* move the wing deficit", and keep three sentences of prose.
- **§`sec-de-limits`, fourth bullet.** "the same error at the parameter match's 0.85x step time and half its memory" — ratio inverted (rule 3), and "half its memory" is a third rendering of 0.52x/1.9x. **Rewrite:** "at parity, with GeoTransolver-512 at 1.17x ISLA's step time and 1.9x its memory (@tbl-scoreboard-435)."
- **§`sec-de-limits`, last bullet.** "Regime extrapolation is unmeasured for ISLA. … The notebook carries the degradation ratios (@sec-nb-hl-sealed-test)". A limits bullet that offloads its number to the dated record. **Fix:** state the number here or state only the limit.
- **Keep:** `tbl-scoreboard-435` (accuracy, parameters, step, memory and ratio in one table) is the single best table in the book; `fig-drivaerml-ladder`; §`sec-de-controls`; the honesty bullets of §`sec-de-limits`.

### 10-boundary-to-interior.qmd

- **Chapter lede.** 300 computed words covering six datasets before the task has been defined (the task section follows it). Rule 3 (concrete before abstract; lead with one sentence). **Fix:** move §"The task" and `fig-interior-task` ahead of the lede, then a three-sentence lede.
- **§`sec-interior-query-tokens`.** "Every standing of this chapter is under it. The alternative, `geometric_mean`, gives each query the mean boundary log-weight plus a learned offset." Chapter 04 says of the same thing: "The convention this replaced … **is gone from the class**." One of the two is wrong; either way the reader is told about an option that does not exist. **Cut** the `geometric_mean` sentences from 10; keep 04's one-sentence statement of the convention and drop its history.
- **§`sec-interior-olbc`, `tbl-olbc-ladder`.** "A cell marked *training* has not landed; the 2,048-case elasticity reference marked *interim* is the same configuration trained on an earlier code snapshot and is replaced by the current-snapshot pair when it lands." Rule 1, in a results table, in two places. **Fix:** a rung that has not landed is not a row. Delete the 8,448 column until it exists.
- **§`sec-interior-olbc` lede.** One sentence carrying ten numbers, two attention-backend names and three parentheticals ("each against the stronger of GeoTransolver's two released attention backends at that rung ({p512[1]} at 512, {p2048[1]} at 2,048; {…} against physics attention at 2,048)"). **Fix:** two sentences; put the backend choice in the table caption, where it already is.
- **§`sec-olbc-ablations`.** "**Every ablation is an option added to the reference ISLA and read as arm ÷ reference on the same code snapshot; none is part of the model.**" A 600-word research note about options that are not in the model, inside an evidence chapter, using project jargon ("code snapshot"). **Move to appendix 13**, cite in one sentence.
- **`tbl-lug-ladder` caption.** "the 5e-4 pair at that rung sat on a loss plateau for 90 to 150 epochs and one seed escaped it late, and the preregistration prescribes that a plateau lane be reported and its half-rate rerun replace it in the ladder". Process narration in a results caption. **Rewrite:** "At 300 lugs ISLA is reported at 2.5e-4; at 5e-4 one seed did not train (row below)."
- **Keep:** §"The task" and its honest statement that "GeoTransolver's volume configuration is not [held to that form]"; the per-architecture "how each treats an interior point" table; §`sec-interior-negatives`.

### 03-invariances.qmd

- **"## Part A: equivariance and pose" / "## Part B: discretization".** Headings that name a position in the chapter, not a message. **Rewrite:** "## A rigid motion of the body must move the field with it" and "## The answer must not depend on how the surface was meshed".
- **`tbl-pose`.** Caption: "10,000 sampled cells per case, seed 42 / seed 43, scored in bfloat16". The centerpiece chapter's only quantitative table is at the exploration budget in the precision the book says flatters ISLA. **Move to chapter 08**, beside `tbl-pose-200k`, where the instrument and the protocol standing can be read together.
- **§"The properties a PDE surrogate must have", four-row table.** Restates 01 §"What generalization means here" and 02's ideas 1–2. **Cut to three sentences** naming the two properties this chapter owns and the two that every architecture has.
- **§"What the contract buys".** "The clause was found on PLAID Tensile2d (@sec-nb-solids-frame-interim)." Order-of-discovery narration plus a notebook citation (rule 1). **Rewrite:** "The clause binds on PLAID Tensile2d, whose targets are stress components (@sec-interior-tensile2d)."
- **Keep:** §`sec-pose`'s argument and its conclusion ("on this data ISLA's exact equivariance and canonicalized GeoTransolver deliver the same pose invariance. No accuracy claim in this book is therefore attributed to equivariance"); §"What the criterion accepts and what it does not"; the collapsed note on frame vs aggregation.

### 02-isla.qmd

- **§`sec-isla-position` vs the chapter's promise.** "What the design buys and costs in measured error is the business of the evidence chapters. This one carries no standings." Then, two paragraphs later: "It does not survive separated three-dimensional flow (@sec-interior-negatives)." And in idea 4: "Removing them from both roles is the largest single-factor effect measured (@sec-de-limits)." The chapter breaks its own rule twice. **Fix:** delete both clauses; the cross-references alone carry the point.
- **§`sec-isla-routing-forms`.** "The form it replaced, and the one the class no longer carries, is an affine map of $\tilde h_i$, which nothing bounds". Rule 1 (cut, never retract), and the same removed form is narrated again in 04 ("An unbounded affine logit was tested and removed"). **Rewrite:** "**ISLA routes on a bounded cosine, so no block can route every point into one slice.** An unbounded affine logit has no such bound: as the weights grow, each slice's softmax sharpens toward a single point… ." State the mechanism as a property of the two forms, not as a history; keep the collapsed table (it is the right place for it) and **cut the duplicate from 04**.
- **§`sec-isla-heads`, the component clause.** ~400 words of inline depth with evidence ("its validation error is 20 to 60 times GeoTransolver's (memorization); with the load axis alone every field is repaired except $\sigma_{12}$") in the middle of the six-idea throughline, and the same clause is stated again in 03, 08, 10, 13 and 15. **Fix:** keep three sentences inline (a component is a view of a tensor; declare the rank; the head expands in symmetrized products), move the rest into a collapsed note, and let chapter 10 own the measurement.
- **Typo, callout "Reflections and pseudotensors":** "The A head that mixed both kinds of basis vector in one output would be covariant under proper rotations only." Stray "The".
- **Keep:** ideas 1–5 and their four airfoil figures — this is the clearest teaching in the book; the contracts table; the "What is not a contract" warning.

### 04-architecture.qmd

- **Notation table, rendering bug.** The row for the global inputs ends `… $S = 1$ on AirfRANS) |\n| $v_{ip}$, $T_{iq}$ | …` — a literal `\n` inside the Markdown table. The row will not render. **Fix:** real line break.
- **§`sec-arch-routing-trained`.** Five figures and a computed paragraph showing trained routing on four datasets, inside the chapter that declares itself "the reference for *what* the class does". It is also the second home for "where the slices sit" (08 §`sec-territories` measures the same object under two draws). **Move** to chapter 02 as the payoff of idea 3, and have 08 cite it.
- **Collapsed note, end of §`sec-arch-routing-trained`.** "The earlier routing probes (@sec-nb-routingsat-erratum) indexed the flat list of calls, so their 'layer $i$' was block $\lfloor i/2 \rfloor$, pass $i \bmod 2$." The book correcting its own past instrument (rule 1). **Cut.**
- **Stage 1 and §`sec-arch-modes`.** "A routing without the bias was tested as the weights-off ablation and removed from the class"; "The convention this replaced … is gone from the class"; "the tangent-plane distance that was tested as a fix and removed from the class is in @sec-nb-…". Three removals narrated in a reference chapter. **Fix:** state what the class does; the ledger in 13 owns what it rejected.
- **Switches table, `slice_prototypes`.** "the one option in flight". **Rewrite:** "`static` routes against learned prototypes shared by every sample; `centered` synthesizes them from the sample's token states. The contracts hold under both." Status belongs to 12.
- **Keep:** `fig-isla-pass`, `fig-isla-layer`, `fig-isla-modes`, the notation table, `tbl-isla-params`, the switches table, §"What the pass never does". This is a genuinely good reference chapter.

### 05-protocol.qmd

- **§"Configuration of the evidence lanes".** The whole section (see B5). It is the licence under which three later chapters carry "earlier protocol" labels. **Fix:** replace with one sentence in §`sec-protocol`: "A table row measured at another width, rate or cell budget says so in the row."
- **Eight full-page dataset figures.** `fig-problem-drivaer`, `-hilift`, `-airfrans`, `-shift-suv`, `-tensile2d`, `-lug`, `-olbc-poisson`, `-olbc-elasticity`. They are excellent, and four of them introduce datasets whose results are four chapters away; one (SHIFT-SUV) introduces a dataset with **no ISLA result anywhere in the book**. Between "what ISLA is" and "what the evidence says" the reader crosses twelve pages of problem statements. **Fix:** keep DrivAerML and HiLiftAeroML here; move the other six to appendix 14 beside each dataset's conversion.
- **`tbl-architectures`.** ISLA's hidden width reads "see @sec-using-config". A table that defers its own cell. **Fix:** write 384.
- **§"Protocol", the force paragraph.** "On the DrivAerML Surface reference checkpoints the 200,000-cell sample alone misses the full-surface drag by 2.4% and the lift by 30% … (@sec-nb-forceref-verdict, one dataset)." A limit with four numbers, no table, a notebook citation, arriving after the protocol table. **Fix:** move to chapter 01 §`sec-metrics` (where forces are defined) or to chapter 13's limits; put the numbers on their own line.
- **Splits.** §"Splits" here says the split table "and the other datasets' splits are in @sec-splits" — which is in appendix 14. The protocol chapter cannot state its own splits. **Fix:** bring `tbl-hilift-splits` and `fig-hilift-splits` into 05; leave provenance in 14.
- **Keep:** §`sec-unit-direction-rule` with `tbl-projector` and the collapsed trained control — a model of how to present a protocol decision; the nine numbered rules; §`sec-review-instrument`; the "What the validation set is used for" warning.

### 08-discretization-and-pose.qmd

- **"R1" in captions.** `fig-territory-maps`: "Top row ISLA (reference configuration R1, seed 42)"; `fig-territory-agreement`: "ISLA is the reference configuration R1, seed 42"; `tbl-density`: "ISLA is the reference configuration trained at 200,000 cells". Rule 4 forbids "R1" outside the notebook and ledger. **Cut "R1"** in all three.
- **§`sec-pose-prior`.** "That refusal has a price on fixed-pose datasets, and the price was measured before the ruling that set it (2026-09-27; @sec-nb-ruling-dataset-frame-inputs)." A date and a ruling in a documentation chapter (rule 1), introducing a four-table-wide diagnostic of a rejected input. **Move the section to appendix 13**; in 08 keep one sentence: "Every input ISLA takes is a datum of the PDE problem; the dataset's axes are not, and what refusing them costs is in @sec-principled-decisions."
- **`tbl-density` caption.** "The 192-wide lanes' factors are in the notebook (@sec-nb-refboard-probes)." A number half-stated and then outsourced. **Cut the sentence.**
- **§`para-territory`.** ~350 words, twenty computed values, two preregistration bars narrated ("just past the 0.05 the probe preregistered as 'unchanged'", "several times the 0.02 preregistered"). The reader cannot hold it. **Fix:** the bars go in the caption; the paragraph keeps three findings (anchors move far less than centroids; the excess is carried by the deep blocks; it halves at 200,000 cells) with the numbers in `tbl-territory`.
- **Slice colour convention.** `fig-territory-maps` uses `plt.get_cmap("turbo", 256)` with a random permutation; chapter 04 uses `slice_colors(M)`, "a shuffled 20-hue qualitative map", which STYLE names as the convention for exactly this. Two colour schemes for one concept. **Fix:** use `slice_colors`.
- **Keep:** `fig-scoreboard-density` and `tbl-density`; the mechanism paragraph ("The sensitivity is the **frame, not the aggregation**"); `fig-training-law` (the filled/open paired marker is the clearest figure idea in the book); `tbl-scoreboard-training-law`.

### 09-capacity-scaling-cost.qmd

- **Opening paragraph.** Two sentences, 150 words, five claims including a four-way cost comparison. **Fix:** split into a one-sentence message ("**On the DrivAerML Surface task, cells stop paying at 100,000, width and depth pay until a shared floor near 0.040, and compute should go to cases below about 200 and to width above it.**") and a scope sentence.
- **Inline computed values in prose.** E.g. "GeoTransolver's width ladder improves by `{python} …` per doubling from 256 to 2,048 wide"; the SCALE-HL paragraph carries ten. Rule 3 says numbers live in tables or on their own line. **Fix:** the SCALE-HL paragraph becomes a four-row table "what reproduces on the wings and what does not"; the width/depth readings cite `fig-capacity`.
- **`fig-equal-budget` caption** is a single unfolded string where every other caption in the book uses a folded block, and it runs to ~180 words with three caveats. **Fix:** fold it and move the "ISLA's depth ladder logged no cost and is absent" caveat into the body, where it is a real limit on the frontier claim.
- **§`sec-cost-sweep` duplicates 06 §`sec-using-cost`** (same artifact, same three rows, two tables: `tbl-isla-cost` and `tbl-cost-200k`). **Fix:** one table. Put it in 06 (the operating manual) and have 09 cite it; 09 keeps `fig-cost-sweep`, which is about scaling, not about the shipped configuration.
- **Status.** "The shared floor is a DrivAerML measurement until the wing learning-curve lanes land (@sec-program-status)." **Rewrite:** "The shared floor is measured on DrivAerML; on the wings it is not established (@sec-limits)."
- **Keep:** `tbl-floor` + `fig-capacity`; the three numbered readings; the compute-optimal rule paragraph, including "*collect cases until about 200, then widen*" — the most useful sentence in the book for a practitioner; `fig-exponents-datasets`.

### 01-problem.qmd

- **§"What generalization means here".** "**Cross-family transfer**: zero-shot evaluation on a vehicle family different from the training family (@sec-shift-suv)." The cross-reference lands on a *dataset description*, and chapter 11 states "Whether ISLA transfers zero-shot across vehicle families is unmeasured: no ISLA lane exists for it at any budget." The chapter promises a measurement the book does not make. **Fix:** list five axes, and name cross-family transfer in the same bullet as unmeasured, citing @sec-limits-transfer.
- **§`sec-metrics`.** "Every bfloat16 number in the book is labeled as such (@sec-nb-fp32-instrument)." Notebook citation for a protocol fact. **Fix:** cite @sec-protocol.
- **Keep:** this is the best chapter in the book. `fig-problem-schematic`; `tbl-metric-illustration` (one error, two samplings, two metrics — exactly the right teaching device); the Green's-identity callout and its three consequences; §`sec-fidelity-columns`, which draws the contract/metric distinction the whole book rests on.

### 06-using-isla.qmd

- **§`sec-using-config`, "The option policy".** "**the one option in flight**, `slice_prototypes`: `static` (the default) or `centered`, sample-conditioned prototypes under their universalization test (@sec-nb-proto-cond-2-prereg)." A manual should not document a decision in progress. **Fix:** document the two values and their effect; drop "in flight" and the notebook reference.
- **§`sec-using-cost`.** The computed lede states five ratios and `tbl-isla-cost` immediately restates all five. Also, the paragraph after the lede begins with a leading space (" The step is memory-bound at every stage but the MLP"), which will render as a continuation of the code block's output rather than a new paragraph. **Fix:** cut the lede to its first sentence; fix the whitespace.
- **Ratio direction:** "ISLA trains at 0.71x GeoTransolver-256's peak memory and 1.11x its step time" — ISLA ÷ baseline (rule 3). **Rewrite** as baseline ÷ ISLA, matching the table's own columns.
- **Keep:** the whole chapter is the clearest in the book after 01. The `using-forward-demo` code cell, the four input gotchas, §`sec-using-lr`, the "How many cells" tip, §`sec-using-contracts` ("A residual far above these on your own data points at the inputs, not the model").

### 11-limits.qmd

- **§`sec-limits-data-efficiency` and §`sec-limits-validation`.** "Its limits … are stated beside the ladder in @sec-de-limits and @sec-de-controls and are not repeated here." Two sections whose body is a forwarding address. **Fix:** either this chapter owns the limits (and 07 carries one sentence each), or it does not exist as a chapter. I prefer the former: a reader looking for scope should find it in one place.
- **Open-questions table.** Overlaps `tbl-vs-ideas` in 15 and the PLANNED rows of 12. **Fix:** this table is the right format and should be the *only* one; delete 15's.
- **Keep:** §`sec-review-roads` ("What would change the recommendation") — a genuinely useful closing; the per-claim "what would move this" sentences.

### 12–15 (brief)

- **12.** The ledger caption says "Statuses are as of 2026-09-22" while rows read "DONE 2026-10-01" and "nixed 2026-09-30" — the caption is stale. The **REF-BOARD row is malformed**: it opens with two status cells ("RUNNING (wing rungs DONE 2026-09-27: GeoTransolver-512 ÷ ISLA 0.89 / 0.90 / 0.95 at 35 / 210 / 510 wings) | RUNNING (510-wing rung DONE…)"), so the row's columns are shifted by one and will not render against the header. **Number conflict:** that row gives GeoTransolver-512 ÷ ISLA = **0.89** at 35 wings, while the REF row of the same table gives "35 wings … GeoTransolver-512 0.1042 (0.1022 / 0.1061) ÷ R1 = **0.71**", and chapter 07's last ladder row computes 0.71. One quantity, two values, same table. Also: many rows are 200–300 words of verdict prose (FRAME-AF-X/Y/EF/G/H/K4, GEO-BIAS, LOCAL-AGG) that duplicate the notebook. **Fix:** rows carry status, question, bar and a link; verdict prose goes to the notebook.
- **13.** Excellent, and the one chapter I would change least. One defect: it carries dates ("cut from every main figure and table on 2026-09-18", "implemented 2026-09-22", "a ruling of 2026-09-27"), which rule 1 allows only in 12 and 18. **Fix:** delete the dates; the ledger rows stand without them.
- **14.** The reference the book needs. One misplacement: §"Scouted, deferred, and in conversion" is program status ("promoted; conversion in progress", "in generation") and belongs in 12.
- **15.** See B4. Additionally: `tbl-vs-standing`'s HiLiftAeroML "210 random cases" row uses a different lane from chapter 07's 210-case row, so the two chapters give different ratios for the same named rung; the caption's blanket "ISLA is the reference configuration … on every row except the HiLiftAeroML rungs other than 35 cases" is not enough to protect a reader who reads the row (rule 3: scope in the same sentence as the claim).

---

## E. Cut list

| File | Opening words of the passage | Why | Words saved |
|---|---|---|---|
| index.qmd | "## What this book does not claim … **Each claim in this book has a scope, and the list below names what sits outside it.**" | Chapter 11 says all eight, with more care | ~500 |
| index.qmd | "## Names, symbols and colors … **Every thing in this book has one name and every architecture one color.**" + `fig-key` | A glossary and a style legend before any content; move to an appendix | ~550 |
| index.qmd | Thesis table rows 5–8, 11, and the capacity row; caption's artifact list | Non-reference configurations and a 15-file source list in the front matter | ~600 |
| 07 | "### The anatomy of the cell-count gap … **Below 100,000 cells per case ISLA's error carries finite-sample noise…**" | Every number is the earlier protocol's, about a gap that no longer exists | ~400 |
| 07 | "Three things travel with @tbl-deepjeb-surface. First, the lead is largest where the data are scarcest…" | Restates the table it follows | ~180 |
| 03 | "**Four properties are commonly required of any discretization…**" (table) | Restates 01 and 02 | ~220 |
| 04 | "The earlier routing probes (@sec-nb-routingsat-erratum) indexed the flat list of calls…" | Corrects a past instrument | ~120 |
| 05 | "### Configuration of the evidence lanes … **Every standing in the book is the reference configuration's…**" | Documents a migration | ~280 |
| 05 | Problem figures and captions for Tensile2d, lug, OLBC Poisson, OLBC elasticity, SHIFT-SUV | Belong beside their conversions in 14 | ~900 (captions) |
| 06 | "**At the reporting protocol of 200,000 cells per case, ISLA trains at …x GeoTransolver-256's peak memory…**" | Restates `tbl-isla-cost` below it | ~130 |
| 08 | "## What a pose prior would buy, and why ISLA does not take it" (whole section) | Fourth home for pose; a diagnostic of a rejected input; move to 13 | ~750 |
| 09 | "The same ladders on the 510-wing HiLiftAeroML rung … carry two of these statements and overturn two." (as prose) | Becomes a four-row table | ~250 net |
| 10 | "### Ablations: per-type geometry channels on the fixed mesh" + `tbl-olbc-ablations` + the paragraph after | Options that are not in the model; move to 13 | ~700 |
| 11 | "### The HiLiftAeroML ladder … are stated beside the ladder in @sec-de-limits … and are not repeated here." and the twin below it | Sections that forward | ~250 |
| 15 | "## Where ISLA is better and where it is worse" through `fig-vs-standing` | Third full statement of the evidence | ~1,400 |
| 15 | "## Ideas from the literature worth trying" (table only; keep the two closing paragraphs) | Duplicates 11's open-questions table | ~750 |
| 12 | Verdict prose in FRAME-AF-X/Y/EF/G/H/K4/SIZE, GEO-BIAS, GEO-BIAS-K3, LOCAL-AGG-STENCIL, LOCAL-LPR rows | Notebook material in the ledger | ~2,000 |

Roughly 9,500 words, with no loss of evidence: every number cut is still computed somewhere the chapter points to.

---

## F. What is good and must be kept

- **Chapter 01 entire.** The three-part input contract, `fig-problem-schematic`, and `tbl-metric-illustration` (one prediction error, two samplings, two metrics) teach the book's central idea before any architecture appears. §`sec-fidelity-columns` — "A structural property an architecture claims is asserted to roundoff in float64 as a unit test, never averaged into a score" — is the distinction the whole argument rests on, and it is stated once, correctly.
- **Chapter 02's ideas 1–5 and their airfoil figures.** `fig-isla-quadrature` (two samplings, two means), `fig-isla-frame` (why a point statistic is a bad frame) and `fig-isla-relational` (six scalars, drawn and tabulated at one token) are the best "words, then picture, then symbols" sequence in the book. The "What is not a contract" warning is exemplary honesty.
- **Chapter 03 §`sec-pose`.** The admission that a canonicalized GeoTransolver is as pose-invariant as ISLA, and the explicit refusal to claim accuracy for equivariance, is the passage that will earn an expert's trust. Keep the three numbered consequences verbatim.
- **Chapter 04's reference machinery.** The notation table, `fig-isla-pass`, `fig-isla-layer`, `tbl-isla-params` (computed from the class at render time) and the switches table. The two collapsed identities ("Pool first, then project", "Rebuild the geometry in the backward pass") are exactly what belongs in a dropdown.
- **Chapter 05 §`sec-unit-direction-rule`.** A protocol decision argued from an initialization measurement, confirmed by a trained control in a collapsed note, with "It makes no accuracy claim about any architecture and appears nowhere else in the book." That is the template every other protocol decision should follow. Keep the nine rules and the "What the validation set is used for" warning.
- **Chapter 06 entire.** The only chapter whose job is unambiguous on its first line. The runnable two-dimensional demo, the four input gotchas (effective measures, unit directions, "a global direction must be a datum of the problem", magnitudes as scalars) and §`sec-using-contracts` are immediately useful.
- **`tbl-scoreboard-435` (07).** Accuracy, seeds, ratio, parameters, step time and memory in one table at one protocol. If one artifact of this book survives, make it this one.
- **`fig-training-law` and `tbl-scoreboard-training-law` (08).** The filled/open paired marker on one vertical line is the clearest single idea in the evidence, and the finding behind it (a baseline trained under the other sampling law is unusable) is the book's strongest result.
- **`tbl-floor` + `fig-capacity`, and the compute-optimal paragraph (09).** "Collect cases until about 200, then widen" is the sentence a practitioner will quote.
- **Chapter 10 §"The task" and §`sec-interior-negatives`.** Stating plainly that the two interior models do not receive the same inputs, and that two designed alternatives failed, is worth more than the standings around them.
- **Chapter 13 entire.** The design-decision ledger, and the two paragraphs that follow it ("**Two rows deserve a sentence each, because they are the ones where the principle cost the most**"), are the best writing in the book and the clearest statement of why the program exists. Do not touch it except to remove the dates.
- **Chapter 14's conventions section** and `tbl-data-files`: a reader can reproduce a lane from it.
- **Chapter 15 §"Novel contributions"** (a)–(h), each with "closest prior work / what is new / evidence / publication-worthy?", the ranked list, and a bibliography that marks what was read in full. This is the part of the book that argues ISLA's case to the field, and it should be the appendix's whole content.
- **Two conventions that are observed almost everywhere and must not be lost:** the "How to read this" clause in every figure and table caption (I found no results figure missing one), and seeds drawn as marks beside a mean rather than as error bars.
