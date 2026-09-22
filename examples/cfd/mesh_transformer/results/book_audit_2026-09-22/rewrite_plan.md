# Rewrite plan for the ISLA book (2026-09-22)

Peter's ruling (2026-09-22), verbatim where it matters: "the book should NOT be path-dependent - its content should
be a reflection of what IS, not how we got here"; "if we're not using linear routing why highlight it everywhere
else?"; "I penalize you for repetition"; "the story around discretization invariance is highly fragmented. And
translation/rotation equivariance also don't have a great centerpiece"; "distinguish DrivAerML Surface task vs
DrivAerML Volume task"; "get your story straight" (text vs figure on DrivAerML accuracy); "one reference set of
hyperparameters ... the various experiments are all deviations from that"; "I worry that you're baking in
aerodynamics-only notions of things like freestream velocity into your writing"; "a total audit and overhaul of the
book - not just edits here-and-there, but a rewrite that preserves the good while completely fixing the organization
and clarity"; more datasets incl. non-aerodynamics; AeroZoo mention.

Inputs: `structure_audit.md`, `figures_audit.md`, `datasets_scout.md` (this folder), `book/STYLE.md`, `book/CLAIMS.md`.

## 1. New chapter list (files in `book/`; labels of moved sections are KEPT so cross-references keep resolving)

| new file | title | built from | new/kept labels |
|---|---|---|---|
| `index.qmd` | ISLA: a surrogate architecture for steady boundary-driven PDEs | index (rewrite) | sec-verdict, sec-key, sec-how-to-read, sec-not-claimed |
| `01-problem.qmd` | The problem and how we measure | 01:1-203 (operator, general interface, generalization axes), 01:336-510 (metric, instrument, contracts-as-tests) | sec-problem, sec-pde-theory, sec-parametric, sec-part-eval, sec-metrics, sec-fidelity-columns |
| `02-isla.qmd` | ISLA, one idea at a time | 02 minus evidence numbers, minus variant history; routing forms stay as idea 3's second half with the linear ceiling table as a collapsed note | all sec-isla-* ; sec-isla-routing-forms; sec-isla-variants renamed in place to "Why ISLA has no center" (physical argument only) |
| `03-invariances.qmd` (NEW centerpiece) | The two invariances ISLA is built for | 01:512-561 (four properties, discretization principle), 04:541-622 (pose: equivariant by construction vs canonicalization), 06:166-292 (the four sampling draws), 17:22-34 (criterion) | sec-four-properties, sec-principle, sec-pose, sec-sampling-laws; new sec-invariances, sec-equivariance-centerpiece, sec-discretization-centerpiece |
| `04-architecture.qmd` | The ISLA forward pass | 07 (cut 07:558-561, 07:642-644; fix switch table: routing_norm cosine default, linear alternative; drop total_measure row content beyond definition) | all sec-arch-* |
| `05-protocol.qmd` | Baselines, datasets and the fair-comparison protocol | 04:1-282 (baselines, what each sees, unit-direction rule stated generally, nine rules, instrument, resource matching), 01:205-334 (datasets, splits), 01:563-601 (protocol table) | sec-baselines, sec-part-baseline, sec-fair-comparison, sec-review-instrument, sec-resource-matching, sec-datasets, sec-benchmarks (rename heading "DrivAerML" with Surface/Volume tasks), sec-shift-suv, sec-splits, sec-protocol, sec-lr-protocol (one paragraph: each architecture at its own best rate) |
| `06-using-isla.qmd` | Using ISLA | 09 rewritten for the reference configuration; 02:735-753 cost merged into sec-using-cost | all sec-using-* ; sec-isla-cost -> anchor kept inside 06 |
| `07-surface-accuracy.qmd` | Surface accuracy and data efficiency: DrivAerML Surface and HiLiftAeroML | 03:78-168 tables, 06:51-114 (full data, deficit anatomy), 05 (ladders, controls, strata, limits), 06:1017-1079 only as one limits sentence | sec-scoreboard-200k, sec-full-data, sec-full-data-accuracy, sec-deficit-anatomy, sec-data-efficiency, sec-de-* ; new sec-surface |
| `08-discretization-and-pose.qmd` | Discretization and pose: the evidence | 06:294-360 (density), 06:362-473 (refinement bench; keep only if re-run at protocol, else one paragraph + notebook pointer), 06:475-564 (training law), 06:566-667 (territories), 06:669-710 (pose at 200k) | sec-robustness-density, sec-consistency-bench, sec-training-law, sec-territories, sec-pose-contract, sec-scoreboard-consistency, sec-scoreboard-training-law, sec-scoreboard-pose (anchors kept as aliases at the top of the corresponding sections) |
| `09-capacity-scaling-cost.qmd` | Capacity, scaling laws and cost | 06:722-1015 (frontier, floor, equal budget, capacity ladders), 04:284-538 (cells ladder, compute allocation, AirfRANS ladder), 03:361-499 (floor, cells, cost sweep) | sec-review-frontier, sec-shared-floor, sec-equal-budget, sec-compute-allocation, sec-scoreboard-floor, sec-scoreboard-tokens, sec-scoreboard-resources |
| `10-boundary-to-interior.qmd` | Boundary to interior | 08 with the cosine rows as the rows; DrivAerML Volume then AirfRANS; 06:712-720 cut | all sec-interior-* , sec-scoreboard-interior alias |
| `11-limits.qmd` | Limits, open questions and directions | 12 minus the established table (replaced by index thesis table), minus answered open questions, minus removed-variant findings (one sentence each + notebook pointer) | sec-limits, sec-review-honesty, sec-limits-*, sec-open-questions, sec-future, sec-review-roads; sec-established kept as an alias on the index thesis table |
| `12-program-status.qmd` | Program status (the one place with dates) | 13, statuses reconciled | sec-program-status, sec-tech-tree, sec-ledger, sec-provenance |
| `13-design-decisions.qmd` | Design decisions and the principle behind them | 17 with the cosine row fixed; rows without a current number carry principle only | sec-principled-* |
| `18-notebook.qmd` | Lab notebook (unchanged file name) | as is | sec-nb-* |

Deleted as files: `03-benchmarks.qmd` (tables move to 07/08/09/10), `04-baselines-and-fair-comparison.qmd`,
`05-data-efficiency.qmd`, `06-accuracy-and-robustness.qmd`, `07-architecture.qmd`, `08-boundary-to-interior.qmd`,
`09-using-isla.qmd`, `12-limits-and-directions.qmd`, `13-program-status.qmd`, `17-principled-decisions.qmd`
(their content lives in the new files above). Update `_quarto.yml` chapters accordingly. Label aliases: where a
retired label is still referenced, place an empty `::: {#sec-old-label}` `:::` div at the start of the new home
section (Quarto resolves the crossref to it).

## 2. Writing rules for every editor
1. Present tense, what IS. No dates in chapters (dates live in 12 and the notebook). No "was", "until", "before the
   change", "pending", "running", "in flight", "removed variant", "legacy", "interim", "superseded". If a measurement
   does not exist yet, the chapter does not talk about it; the status chapter does.
2. One home per concept. A chapter may cite another's result in one sentence with a cross-reference; it may not
   re-argue or re-tabulate it. No per-chapter protocol paragraph: one sentence "protocol: @sec-protocol".
3. Task names: "DrivAerML Surface task" / "DrivAerML Volume task" (or "the surface task on DrivAerML"); "HiLiftAeroML
   (35-wing rung)"; "AirfRANS (interior task)". Every cross-architecture sentence names dataset, task and rung.
4. Routing: cosine is ISLA. Linear routing appears only in 02 idea 3 (the two forms; why bounded logits train at width
   and depth; ceiling table collapsed) and in 04's switch table. Nowhere else as a row, a series or a caption.
5. General interface first: boundary cells (points, normals, measures) + K global directions + S global scalars +
   optional per-cell boundary scalars -> fields on the boundary or at interior queries. External aerodynamics is the
   benchmark instance (K = 1 far-field direction; S = 0 or 1). Mechanism text uses "along / against a global
   direction", not "upwind"; the unit-direction rule reads "directions are unit vectors, magnitudes are scalars".
6. Figures: every axis spans the range a reader expects or the figure says in its caption why not; seeds as dots or
   short bars; no dodge on categorical axes; legends carry the styles (fig-key covers every style used); no failed fit
   drawn as a fit; no hand-set bands; fits drawn only where accepted.
7. Numbers policy during the transition (until REF-BOARD lands): keep the current artifacts and numbers, but state
   ONCE, in 05-protocol.qmd under a short "Configuration of the evidence lanes" paragraph, that the surface-task lanes
   were trained with the previous routing form and 192 width and are being re-measured under the reference
   configuration; individual tables and captions do not narrate this. Where a cosine/reference artifact exists
   (35-wing 0.1408 / 0.1361, cars interior 1e-3, cars 384x24 at 200k, AirfRANS), it is the row.
8. Every quantitative claim is computed from its artifact in a cell (no numbers from memory): wire in
   `routenorm_p200k_2026-09-22.json`, `wingbench_200k_reduction_2026-09-19.json`, `territory_wings_2026-09-21.json`,
   `tokproto_reduction_2026-09-18.json` where the audits say they are unread.
9. Resolve the twelve contradictions in `structure_audit.md` §5 and `figures_audit.md` §2 explicitly; in particular
   the accuracy story on the DrivAerML Surface task: "parity within seed spread at 192 wide; the reference
   configuration is more accurate than GeoTransolver-256" is the sentence once REF lands; until then "parity within
   seed spread (0.998 on the trio pair, 0.99 on the iso-cost pair)".
10. STYLE.md governs prose and figures; CLAIMS.md governs what may be said; update both at the end of the pass.
11. Datasets: 05-protocol.qmd gets a "Datasets beyond external aerodynamics" paragraph naming the conversions in
    progress (PLAID Tensile2d, GINOT Bracket Lug, GNOT Heat2d) as the next benchmark instances and AeroZoo as the
    in-generation diversity benchmark, with no numbers; 11-limits.qmd lists them as the open axis.

## 3. Order of work
1. Editors compose the new files from the sources (read-only on the old files); old files stay until all new files
   exist. 2. Integration: delete old files, update `_quarto.yml`, add label aliases, render, fix crossrefs and
   contradictions, update STYLE.md/CLAIMS.md/notebook entry, commit. 3. As REF/REF-BOARD/BASELINE-GAPS lanes land,
   swap artifacts and remove the transition paragraph.
