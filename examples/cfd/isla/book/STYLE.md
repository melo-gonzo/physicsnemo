# Book style guide

The book is an instructive guide to the ISLA program in the manner of a
Drela user guide (TASOPT, XFOIL): it teaches the reader what the method is,
why each part is there, how to use it, and what the evidence says, with a
clean throughline and optional depth folded away. It is written for an
*uninitiated expert*: fluent in PDE numerics and machine learning, new to
this program. This file fixes the conventions every chapter follows.
`CLAIMS.md` remains the source of truth for numbers; this file governs
presentation.

## The arc

| Part | Chapters | The reader leaves knowing |
|---|---|---|
| Front matter | `index.qmd` | the thesis in one paragraph, ONE thesis table with dataset and task named per row, how to read the book |
| I The problem and the two invariances | `01-problem.qmd`, `02-isla.qmd`, `03-invariances.qmd`, `04-architecture.qmd` | the general operator interface and the metric, and that the surface task and the interior task are one problem with the prediction points placed differently; why each part of ISLA is there, one idea at a time, ending with the trained routing pictures; the two invariances (equivariance and pose; discretization) as contracts and instruments, followed by every pose and discretization measurement of the book (the one home for those tables, cited by the dataset chapters in one sentence each); exactly what the forward pass computes, stage by stage, each stage with its code listing extracted from the class (the source of truth for the architecture and its switches) |
| II Protocol and use | `05-protocol.qmd`, `06-using-isla.qmd` | the baselines and what each sees, the fair-comparison rules (cells, epochs, rates, unit directions, seeds, sealed splits, boundary conditions as boundary meshes, the stored signed distance to both architectures, no augmentation on any reference lane), the dataset roster and the file conventions and split rule common to every dataset; the one configuration, the inputs, the recipe, the inference sampler, checkpoints, and the book's one cost-per-step table |
| III Evidence, by dataset | `07-drivaerml.qmd`, `08-hilift.qmd`, `09-airfrans.qmd`, `10-shift-suv.qmd`, `11-tensile2d.qmd`, `12-lug.qmd`, `13-deepjeb.qmd`, `15-capacity-scaling-cost.qmd` | one chapter per dataset, every one in the same order: (a) the problem on sample cases (geometry, boundary conditions with their types and values, the global variables, the fields) and the diversity of the set; (b) what is measured (task, protocol, splits, cells); (c) the headline table of the three architectures under their reference configurations on the native mesh, with the uniform-point-cloud and rotated-and-translated columns read from the characterization artifact when it exists; (d) the ladders and controls that belong to that dataset; (e) one "what this dataset does not say" paragraph. Then the one chapter organized by question: capacity, scaling laws and cost |
| IV Limits and status | `16-limits.qmd`, `14-program-status.qmd` | every scope statement in one place, including what the book does not claim; what is running (the one chapter with dates) |
| Appendices | `17-design-decisions.qmd`, `19-literature.qmd`, `18-notebook.qmd` | the names and colors, the design-decision ledger, the pose-prior diagnostic and the ablations that are not in the model; which of ISLA's ideas are new against the literature, with a bibliography; the dated record |

File numbers follow chapter order except where a file is owned by another
process and keeps its name (`14-program-status.qmd`, `18-notebook.qmd`);
`_quarto.yml` is the order of record. Fixed-mesh datasets, on which the
geometry never varies, are unrealistic for engineering and are not
mentioned in the documentation chapters.

Chapters state current knowledge. History (what was believed before, what
was retracted) lives in the notebook and the status chapter only.

## Teaching pattern for every section

1. **Lead with the message** in one bold sentence a reader can quote.
2. **Words, then picture, then symbols.** Describe the mechanism physically,
   point to a figure, then write the equation. Never open with an equation.
3. **Concrete before abstract.** Use the running example (the Joukowski
   airfoil of Part I, `palette.joukowski()`) or one car/wing before the
   general statement.
4. **Define at first use, once per chapter.** A term is defined the first
   time a chapter uses it, even if an earlier chapter defined it.
5. **One idea per figure; a "How to read this" clause in every caption**
   stating what the figure does *not* show.
6. **Optional depth goes in a dropdown**, never inline:
   ```
   ::: {.callout-note collapse="true" title="Why the softmax runs over points, not slices"}
   …
   :::
   ```
   Use `callout-note` for derivations and detail, `callout-tip` for practical
   advice in Part III, `callout-warning` for a misreading to avoid. The
   throughline must read completely with every dropdown closed.
7. **Numbers live in tables or on their own line**, computed from the
   artifact (`#| output: asis`), with the artifact path in the caption.
   Prose carries magnitudes only when they change what the reader does.
8. **Every cross-architecture claim carries its scope** (dataset, rung,
   instrument) in the same sentence, and every ratio is *baseline ÷ ISLA*.

## Nomenclature (canonical names; use these and no synonyms)

| Say | Never say | Meaning |
|---|---|---|
| ISLA; the reference configuration (width 512, 12 layers, 512 slices, cosine routing, the centered sample-conditioned prototypes, the RMS length unit, the fused geometry kernel, learning rate 3.75e-4, 200,000 training cells; selected by REF-80, 2026-10-04) | MT2, MT, MeshTransformer2, "the mesh transformer"; "linear routing", "192 wide" or "384 wide" as if they described ISLA; "R1" in prose (a notebook and ledger label); "the reference" without naming the architecture | the definitive model as shipped: one origin-free frame, `scale_mode="rms_distance"`, the class defaults for everything but width and slice count (`hidden=512`, `n_slices=512`); the configuration is defined once in 05 with its selection rule (the most accurate under 80 GiB per GPU on DrivAerML Surface and the 210-wing rung, two seeds, ties within 3% to the cheaper) and stated for the user in 06, and never used as a name. Since 2026-09-20 the class has no frame variants |
| the baseline reference configurations: GeoTransolver width 256 (local width 32), learning rate 1e-3, 200,000 cells; Transolver width 512 with 16 layers at 100,000 cells, learning rate 1e-3 | "GeoTransolver at its recipe width", "the parameter match", "GeoTransolver-512" as a current arm | the two baselines' most accurate configurations under the same 80 GiB rule on the same two datasets; every current standing is these three configurations, and the chapters name them as written here |
| the earlier ISLA reference (384 wide, 12 layers, 256 slices, 5e-4); or in captions "ISLA at 384 wide, 12 layers, 256 slices, 5e-4" | "the reference configuration" for a 384-wide lane; an unlabeled 384-wide number | the configuration the ladders, the capacity and scaling studies, the trained-routing and slice-territory figures and the pose and sampling probes were run with; every row, caption and figure that holds one carries this label and is read for slopes, mechanisms and contracts, never as the standing of the reference configurations |
| the earlier protocol (192 wide, 1e-3) | "the old ISLA"; an unlabeled 192-wide number in a standing | the 192-wide, linear-routed lanes at 1e-3 that a reference-configuration re-run has not yet replaced; every table row or caption that still holds one carries this label (05 explains it once) |
| the 24-layer depth variant | "R2" in prose; "the deeper reference" | 384 wide, 24 layers, 5e-4; a lane of the depth ladder (15) and the cost sweep (06), within 1.5% of the 12-layer 384-wide lane on the cars and the 35-wing rung at twice the cost |
| the relative frame | "the reference frame", "the frame-free frame" ("reference frame" reads as a coordinate frame to a PDE reader) | ISLA's construction of position: no origin, positions enter only as point-to-anchor differences. The only frame the class has (the `frame_mode` flag was removed 2026-09-20) |
| the legacy centered variant (removed) | "the constant-gauge variant" in a main result; "prior reference"; "centered ISLA" | the centered frame with a constant length unit the program used before the relative frame; cut from every main figure and table on 2026-09-18 and removed from the class on 2026-09-20 (@sec-nb-frame-cleanup); appears only in collapsed diagnostic notes, labeled "removed variant" |
| form; one-car floor | | a structural choice of the architecture (as opposed to capacity or inputs); the converged error on a single training car (a capacity measure) |
| the sample-frame probe | "the probe instrument" alone | the 10,000-cell uniform and 10:1-biased DrivAerML evaluation draws, each model's frame taken from the drawn sample |
| $M$ slices, $S$ global scalars | $S$ for the slice count | symbol discipline: $M$ slices ($M = 512$ in the reference configuration, 256 the class default and the ladders' count); $S$ is reserved for global scalars |
| the similarity-gauge variant (removed) | "the similarity gauge" in a main result; "gauge ISLA" | the centered frame with a measure-weighted centroid and RMS radius; cut from every main figure and table on 2026-09-18 after SG-REF (within noise of ISLA, @sec-nb-sgref-verdict) and removed from the class on 2026-09-20; appears only in collapsed diagnostic notes, labeled; its result is phrased as "the price of not being told the body's center: at most about 2%, inside seed noise", never as recovering a deficit |
| the weights-off ablation | "uniform ISLA" | a removed diagnostic (uniform token measures); quoted only from the notebook, never adopted |
| the raw-coordinate diagnostic | | a removed diagnostic that fed absolute coordinates; breaks the contracts by design |
| GeoTransolver, Transolver, GeoTransolver-volume | "GeoTransolver, released"; "the released baselines"; "the baselines" without naming them; "GT"/"T" outside code | the definitive public models, never modified; no qualifier |
| GeoTransolver + ISLA's <feature> (ablation) | "research variant", "GeoTransolver, measure-weighted pooling" | a baseline with one or more of ISLA's features added to isolate an effect (e.g. "GeoTransolver + ISLA's measure-weighted pooling + HT-centered datapipe (ablation)"); never a variant of the public model |
| unit-direction protocol | "udrv" | every model receives the freestream as a unit vector |
| global vector inputs, global scalar inputs | "drive", "freestream feature" | the problem's global conditions; $K$ vectors, $S$ scalars |
| boundary cell / point / token | | one sampled surface element and its token |
| measure weight $w_i$ | "area", "quadrature weight" alone | the surface area a sampled cell stands for |
| slice, anchor | "cluster", "centroid" for slices | a soft group of tokens; its measure-weighted mean position and normal |
| relational invariants | "geo features" | the scalars describing a token relative to an anchor |
| routing / read-back | | softmax over points into slices / softmax over slices back to points |
| length unit $L$ | "gauge" in prose; "the total measure" in the present tense | the measure-weighted RMS pairwise distance of the sample (the only length unit in the code). "The total-measure unit" names a removed earlier option (the square root of the total measure); lanes trained before 2026-09-18 carry it and are labeled so where quoted |
| the DrivAerML Surface task, the DrivAerML Volume task; HiLiftAeroML (the $n$-wing rung); the AirfRANS interior task | "DrivAerML" alone in a sentence that compares architectures | dataset AND task on every cross-architecture sentence; "HiLift" allowed after first use in a chapter |
| the $n$-case rung | | a training-set size on the data-efficiency ladder |
| area-weighted relative L2, float32 instrument | "rel-L2" in prose | the metric and its reporting precision |
| the deficit anatomy | | the decomposition of the in-distribution gap into named parts |
| the reporting protocol (200,000 cells per case) | "the 200k runs" in prose; an unlabeled 10,000-cell number | every reported standing is scored at 200,000 sampled cells per case in float32, two seeds, 500 epochs; the reference configurations train at their own cell counts (ISLA and GeoTransolver 200,000, Transolver 100,000) and rates (ISLA 3.75e-4 by the width rule, the baselines 1e-3); the ladders' lanes carry their own rates per row (GeoTransolver-256 at the better of 1e-3 and 3e-3). Adopted 2026-09-18 |
| the exploration budget (10,000 cells per case) | "the reference resolution"; "the protocol" for a 10,000-cell number | the count the program explored at before 2026-09-18; every 10,000-cell figure, table and number carries this label |
| on the DrivAerML Surface task the three reference configurations tie within seed noise on the native mesh (GeoTransolver width 256 ÷ ISLA 1.03, Transolver width 512 with 16 layers ÷ ISLA 0.99 on pressure) and ISLA alone holds on the uniform point cloud and under rotation and translation | "ISLA leads on the cars"; "parity with the parameter match"; "ISLA leads GeoTransolver" without naming the configuration; a lead read off a 0.3 to 1% gap | results/char/drivaerml_surface.json; the ladders (384-wide ISLA against the 256-wide baselines, GeoTransolver at its better rate) are read for slopes and the 54-to-109-car crossover, never as the standing |
| on the 210-wing HiLiftAeroML rung ISLA leads both reference baselines on the native mesh (GeoTransolver width 256 ÷ ISLA 1.28, Transolver width 512 with 16 layers ÷ ISLA 1.07) | "the wing deficit" as a current standing; "both baselines lead on the wings" without naming the configurations | results/char/hilift_surface.json; the ladders against the parameter-matched GeoTransolver-512 (above the 80 GiB line) show the baseline ahead at 35, 210 and 510 wings and are labeled as such (08) |
| the ladder deficit against GeoTransolver-512 at the 35-wing rung is not count noise, width (to 384), depth or rate | "the wing gap is a budget artifact"; "the wing deficit" without naming GeoTransolver-512 and the rung | the controls table of 08, every row labeled with the configuration it holds (384 wide, 5e-4; GeoTransolver-512, 3e-3); results/ref_surface_2026-09-23.json, results/p200k_fleet_reduction_2026-09-19.json |

Symbols: $x_i, n_i, w_i$ inputs; $\hat g_k$ global unit vectors; $\sigma$
global scalars; $L$ length scale; $r_i = x_i / L$; $h_i$ token state;
$a_{is}$ routing weight; $z_s, m_s$ anchors; $\xi_{is}$ relational
invariants; $Z_s$ slice state; $\pi_{is}$ read-back weight; $\hat u_i$
radial basis vector. Definitions use $\equiv$.

## Color and mark semantics (from `palette.py`)

| Concept | Color | Marker / line | Notes |
|---|---|---|---|
| ISLA reference | `BLUE` | filled square, solid | |
| ISLA variants | `BLUE_LIGHT` | open diamond dashed (similarity-gauge variant) and open square dotted (legacy centered variant); both keys are used only inside collapsed diagnostic notes, never in a main figure | same hue, lighter |
| ISLA ablations | `BLUE_PALE`, hatched | | never adopted |
| GeoTransolver | `RED` | filled circle, solid | also GeoTransolver-volume |
| Transolver | `AMBER` | filled triangle, dashed | |
| baseline research variants | `RED_LIGHT` / `AMBER_LIGHT` | open marker, dotted | |
| slices, anchors | `VIOLET` | | concept figures only |
| queries, interior points | `AQUA` | | concept figures only |
| contract satisfied | `GREEN` | | |
| context (grids, reference lines, non-data) | grays | | never encodes a data distinction |
| signed fields | `DIVERGING` (blue–paper–red) | | unsigned fields: viridis |
| boundary-condition types (the dataset problem figures only) | `BC["value"]` plum (a value prescribed: Dirichlet, fixed, clamped), `BC["flux"]` burnt orange (a flux prescribed: Neumann, traction, applied pressure), `BC["robin"]` teal, `BC["free"]` gray | thick boundary stroke, direct label | global vectors (far-field direction, laboratory frame) are black arrows |
| dominant slice of a token (trained-routing figures) | `slice_colors(M)`: a shuffled 20-hue qualitative map | anchors: black-edged stars | labels only; two slices may share a hue; never an architecture hue's role |

In Mermaid diagrams: data boxes `fill:#f4f3ee,stroke:#898781`; fixed
geometric maps `fill:#ffffff,stroke:#2a78d6,stroke-dasharray:4 3`; learned
maps `fill:#e6f0fb,stroke:#2a78d6`; contracts `stroke:#008300`.

Use `STYLE[key]` / `series_kwargs(key)` / `bar_kwargs(key)` for every data
series so labels and colors never drift. Prefer direct labels (`label_line`)
to legends; when a legend is needed, its text is the canonical name.

## Figures and tables

- Every axis: quantity and unit in brackets, `[–]` for dimensionless.
- Bars start at zero; log axes only where the range spans a decade, with
  1–2–5 ticks written out.
- Multi-panel figures stack vertically and share the x-axis; no dual axes.
- Seeds are dots on bars or bands on lines; the seed mean is the mark.
- Table captions name the artifact; the `{#tbl-…}` label is on the caption line.
- A "How to read this" clause appears in every results figure caption.

## What the chapters do not do

- **Cut, never retract.** A claim now known wrong does not appear at all:
  not as "we previously believed", "retired", "reframed" or "superseded". A
  reader must never watch the book argue with itself. History is the
  notebook's job.
- **No dates, no chronology, no project slang.** No "as of 2026-09-05", no
  "wave", "round", "campaign", "addendum", "critic review", "before/after
  the fix". No node codes (A35b, W2-C, EQ-PRICE) or version tags in prose;
  say what the experiment measures ("the single-factor ablation at 35
  training cases"). Exceptions: the status chapter's tech tree and ledger,
  and the notebook.
- **No run identifiers** (`rf_dr_mt2_meas_seed42`) in prose; they appear
  in code cells and in captions' artifact paths only.
- **No numbers from memory.** Every number is computed from a file in
  `results/` (or `research/`) or quoted from `CLAIMS.md` with its source.
  Define the quantity before the number: what is measured, on which split,
  under what protocol, and which way a ratio points.
- **Resource-matched comparisons only.** A cross-architecture comparison
  states parameters, tokens per case and memory of both sides, or says in
  the same sentence where they differ and whether the difference could buy
  accuracy.
- **No jitter or dodge on categorical axes**; seeds and architectures are
  distinguished by marker, fill or line style.
- **No hedged verdicts** and no result made to look better than it is. If a
  comparison is unfair, fix it or drop it; do not caption around it.
- **Scope and limits sit beside the claim**, not only in the limits chapter.
- **Chapter rewrites do not edit `18-notebook.qmd` or `results/`.**
