# Book style guide

The book teaches ISLA the way a good monograph teaches a numerical method: what problem
it solves, how each part works and why it is there, how to run it, and what the
measurements say for it and against it. It is a guide, not a results log. This file
fixes the conventions every chapter follows. Re-read it before editing a chapter.

## The reader

A bright PhD student at a top university with deep applied mathematics and physics: PDEs
and their numerics, functional analysis, tensor calculus, quadrature, and optimization
theory (convexity, preconditioning, steepest descent in a chosen norm, KKT). Moderate
machine learning: knows gradient descent, MLPs and what a loss curve is; does **not**
necessarily know transformer internals (attention, softmax, LayerNorm, residual streams,
pre-LN blocks), optimizers beyond Adam, mixed precision, or what makes a GPU kernel fast
or slow.

So:

- Never explain a PDE concept, a norm, a quadrature rule or a gradient. Do explain, once
  and at first use, every piece of ML machinery: attention, softmax, LayerNorm, residual
  stream, MLP block, token, embedding, bf16, Muon, torch.compile, kernel fusion,
  activation recompute.
- Explain ML machinery through the reader's own mathematics: attention is a
  data-dependent kernel integral operator (a Nadaraya–Watson smoother); slice attention
  is a low-rank factorization through a learned soft partition (a restriction and a
  prolongation, as in algebraic multigrid); a measure-weighted softmax is a quadrature;
  Horvitz–Thompson weights are importance weights; a residual network is a product of
  near-identity maps; Muon is steepest descent in the spectral norm; the equivariant
  head is tensor calculus.
- Respect their intelligence: precise statements, real equations where they carry the
  idea, and honest scope. Do not hype and do not hedge.

## The arc

| Part | Chapters | The reader leaves knowing |
| --- | --- | --- |
| Front | `index.qmd` | the thesis in one paragraph; ISLA in one picture; the standing in one figure; the key of names and colors; three reading paths |
| I The problem and the landscape | `01-problem`, `02-landscape` | the boundary-to-field operator and how a surrogate is scored, on an airfoil with an exact solution; how neural PDE surrogates work (attention as an integral operator, slice attention, neural operators, graph and point networks, equivariant networks) and the two gaps ISLA fills |
| II How ISLA works | `03-isla`, `04-invariances`, `05-forward-pass` | each idea of ISLA with a picture on the airfoil; the two contracts (rigid motion; remeshing) as exact tests and as measurements on every dataset; the forward pass stage by stage with code, shapes, parameters and where the time and memory go on a GPU (the source of truth for the architecture) |
| III Using ISLA | `06-using-isla`, `07-benchmark` | the configuration, the input contract, commands, sampler, checkpoints and cost; the baselines, the fairness rules, the datasets and how every number is scored |
| IV Evidence | `08-standing`, `09`…`17` (one per dataset), `19-capacity` | the standing on one page, then each dataset in the same order (a)–(e) below, then capacity, scaling and cost |
| V Limits and status | `20-limits`, `14-program-status` | what each claim does not say and what would settle each open question; the live ledger (the one chapter with dates) |
| Appendices | `21-design`, `22-literature`, `18-notebook` | names, symbols, colors and the design decisions with their principle; what is new against the literature, with a bibliography; the dated lab notebook |

`_quarto.yml` is the order of record. `14-program-status.qmd` and `18-notebook.qmd`
belong to the program coordinator; chapter work never edits them.

Every dataset chapter follows the same order: (a) the problem on sample cases (geometry,
boundary conditions with their types, global inputs, fields) and what varies across the
set; (b) what is measured (task, splits, cells, fields); (c) the standing of the three
reference configurations, with a picture of the predictions on one held-out case; (d)
the studies that belong to this dataset; (e) one "What this dataset does not say"
section.

## Teaching pattern for every section

1. **Lead with the message** in one bold sentence a reader can quote. One per section,
   not per paragraph.
2. **Words, then picture, then symbols.** Describe the mechanism physically, point to a
   figure, then write the equation, sandwiched between a symbol table or a sentence
   tying symbols to code and a sentence on what it means. Never open a section with an
   equation.
3. **Concrete before abstract.** The running examples are the Joukowski airfoil
   (`palette.joukowski()`, every concept figure) and one DrivAerML car (every rendering
   of the real thing).
4. **Define at first use, once per chapter**, even if an earlier chapter defined it.
   Each chapter opens with a *Before this chapter:* line naming what it assumes.
5. **One idea per figure; every caption ends with a "How to read this" clause** that
   says what the figure does *not* show.
6. **Optional depth goes in a collapsed callout**: `::: {.callout-note collapse="true"
   title="Optional: …"}`. `callout-tip` for practical advice, `callout-warning` for a
   misreading to avoid. The throughline must read completely with every callout closed,
   and no callout holds something the throughline needs.
7. **Numbers live in tables or on their own line**, computed from the artifact as the
   page renders (`#| output: asis`), the artifact path in the caption. Prose carries a
   magnitude only when it changes what the reader concludes, and then computed inline,
   never typed.
8. **Every cross-architecture claim carries its scope** (dataset and task, size,
   protocol) in the same sentence. Every ratio is *baseline ÷ ISLA*.
9. **Small delights are part of the job**: a margin note that answers the question the
   reader is about to ask, a worked number, an analogy that lands, a figure that rewards
   a second look. Not jokes, not decoration.

## Prose

Follow the writing skill (`~/.pi/agent/skills/writing/SKILL.md`, the Declaw job) and run
its scanner on every chapter you write. In particular: short declarative sentences;
concrete nouns; no throat-clearing ("It is worth noting", "Importantly"); no "not just
X, but Y"; no decorative triplets; no "robust", "seamless", "leverage", "crucial",
"delve"; no rhetorical questions as section openers; no em-dash chains. Prefer the
active voice with a named actor ("the routing softmax normalizes over points"). Use
American spelling.

## Nomenclature (use these and no synonyms)

| Say | Never say | Meaning |
| --- | --- | --- |
| ISLA | MT2, MeshTransformer2, "the mesh transformer", "our model" | `physicsnemo.experimental.models.ISLA` as it is: the class defaults |
| the reference configuration (of an architecture) | "the reference" alone; "R1"; "the final configs" | the one configuration each architecture is scored at: ISLA width 512, 12 layers, 512 slices, learning rate 3.75e-4, 200,000 training cells; GeoTransolver width 256 (local width 32), learning rate 1e-3, 200,000 cells; Transolver width 512, 16 layers, learning rate 1e-3, 100,000 cells. Selected once by the rule in 07 (the most accurate under 80 GiB per GPU on two datasets, ties within 3% to the cheaper) |
| the normalized slice reads; "the slice sublayer reads the normalized state" | "A2", "H-RESID", "the fix" | the slice sublayer's routing, pooling and broadcast read `norm_assign(h)`; the raw stream enters only the residual identity and the MLP's LayerNorm. Part of ISLA, not an option |
| the raw-read variant (ablation) | "old ISLA", "pre-A2" | ISLA with the pooling and broadcast reading the raw stream; appears only as an ablation that explains the normalized reads (03), and in the one-sentence note on any standing whose ISLA checkpoints are of this variant (`standing.DATASETS[...].reference is False`) |
| the nearest-token form (`standing.FORM_TAG`); in a caption: "ISLA checkpoints of the nearest-token form (@sec-checkpoint-form)" | "old ISLA", "the A2 checkpoints", an untagged trained-ISLA figure | an earlier form of ISLA that some trained-ISLA figures (renderings, trained routing) and some study instruments are of; the standings are ISLA's own: the normalized slice reads, one boundary schema shared by all conditions, boundary tokens drawn by the datapipe, wall distance and direction from the nearest sampled boundary token, on AirfRANS, Tensile2d and the lug the stored signed distance as a prediction-point input, a length unit from the given cells' centroids alone, and no declared scales; defined once in 07 (`#sec-checkpoint-form`, `standing.FORM_NOTE`); every caption of a figure or table from such checkpoints carries the tag, and a figure that newer checkpoints will replace also carries the In-flight callout (`palette.rendering_in_flight`) |
| ISLA-DTT | "DTT ISLA", "the tree model", "read-only ISLA" without naming it | `interior=True` (ISLA's interior form): prediction points are passive readers of the slices and read every boundary cell through a Barnes–Hut forest; interior tasks only |
| interacting prediction points / read-only prediction points | "queries" without the adjective when it matters | interacting ISLA's interior tokens carry routing mass and write into the slices / ISLA-DTT's carry none |
| GeoTransolver, Transolver, GeoTransolver-volume | "GT", "T", "the released baselines", "the baselines" without naming them | the public models, never modified |
| GeoTransolver + ISLA's \<feature\> (ablation) | "research variant" | a baseline with one of ISLA's features added to isolate an effect |
| the study configuration tag: "Instrument: ISLA at \<width\> wide, \<slices\> slices, \<rate\>, raw slice reads" (in the caption) | an unlabeled smaller-ISLA number | a study run on a smaller ISLA than the reference configuration (scaling ladders, sampling probes); defined once in 07 (`#sec-study-configurations`); read for slopes and mechanisms, never as a standing |
| the $n$-case rung (of a data-efficiency ladder) | "rung" undefined | a training-set size of a data-efficiency ladder, defined in 07 |
| "Instrument: ISLA at \<w\> wide, \<M\> slices, \<rate\>, linear routing, raw slice reads" | | the tag variant for study lanes that used the removed unbounded routing score (07, `sec-study-configurations`) |
| slice, anchor | "cluster", "centroid" | a soft group of tokens; its measure-weighted mean position $z_s$ and mean normal $m_s$ |
| routing / read-back | | softmax over points into slices / softmax over slices back to points |
| relational invariants | "geo features" | the scalars describing a token relative to an anchor |
| the dataset's axes; raw coordinates; components along the dataset's axes | "laboratory frame", "laboratory axes", "laboratory coordinates", "laboratory components", "lab frame" | the fixed $x, y, z$ axes a dataset stores its meshes in (DrivAerML: $x$ along the car, $z$ up), as opposed to a frame the body carries or the slice anchors; defined in 01's pose bullet. A tree aligned with them is "a tree whose boxes are aligned with the dataset's axes" |
| length unit $L$ | "gauge" in prose | the measure-weighted RMS pairwise distance of the full-resolution boundary, each cell with its own second moment, computed before the token draw |
| measure weight $w_i$ | "area" alone, "quadrature weight" alone | the measure (area, length) a sampled cell stands for, Horvitz–Thompson-corrected under a non-uniform draw |
| global vector inputs, global scalar inputs | "drive", "freestream feature" | the problem's global conditions; $K$ unit directions $\hat g_k$, $S$ scalars $\sigma$ |
| the DrivAerML Surface task, the DrivAerML Volume task; HiLiftAeroML (the $n$-wing rung); the AirfRANS interior task; SHIFT-SUV; AeroZoo; PLAID Tensile2d; the GINOT bracket lug; DeepJEB++ | "DrivAerML" alone in a cross-architecture sentence | dataset and task on every cross-architecture sentence; short names after first use in a chapter |
| native mesh / uniform point cloud / rotated and translated (posed) | "area protocol", "ca2" | the three scorings of one checkpoint in every standing artifact |
| the scoring protocol | "the 200k runs" | 200,000 sampled cells per case, float32, final checkpoints, two seeds (42, 43), per-case area-weighted relative L2 averaged over cases |
| area-weighted relative L2 | "rel-L2" in prose | the metric (01) |
| level / leads / trails | "ties" for a separated pair, "beats", "wins", "crushes" | the seed rule: a lead needs every seed of one arm below every seed of the other and a seed-mean gap of at least 3%; else level (`standing.Standing.verdict`) |
| precise failure terms: "the validation loss rises 3x and does not recover", "routes every token to one slice" | "collapsed", "blew up", "diverged" without the measurement | |

Symbols: $x_i, n_i, w_i$ inputs; $\hat g_k$ global unit vectors; $\sigma$ global
scalars; $L$ length unit; $r_i \equiv x_i / L$; $h_i$ token state; $\tilde h_i \equiv
\mathrm{LN}(h_i)$; $a_{is}$ routing weight; $z_s, m_s$ anchors; $\xi_{is}$ relational
invariants; $Z_s$ slice state; $\pi_{is}$ read-back weight; $\hat u_i$ radial basis
vector; $M$ slices ($M = 512$); $N$ boundary tokens; $Q$ prediction points; $d$ spatial
dimension. Definitions use $\equiv$. $S$ is reserved for global scalars, never the slice
count.

## Color and mark semantics (from `palette.py`)

| Concept | Color | Marker / line |
| --- | --- | --- |
| ISLA | `BLUE` | filled square, solid |
| ISLA-DTT | `BLUE_LIGHT` | open square, dash-dot |
| ISLA ablations (raw-read variant, weights off) | `BLUE_PALE`, hatched | open square, dotted |
| GeoTransolver (and -volume) | `RED` | filled circle, solid |
| Transolver | `AMBER` | filled triangle, dashed |
| slices, anchors | `VIOLET` (concept figures); `slice_colors(M)` labels in trained-routing figures, anchors as black-edged stars | |
| prediction points, interior | `AQUA` | |
| contract satisfied | `GREEN` | |
| context (grids, reference lines) | grays | never encodes a data distinction |
| signed fields, errors | `DIVERGING` (blue–paper–red) | |
| unsigned fields | viridis | |
| boundary-condition kinds | `BC["value"]`, `BC["flux"]`, `BC["robin"]`, `BC["free"]` | thick boundary stroke, direct label |

Use `STYLE[key]`, `series_kwargs(key)`, `bar_kwargs(key)` for every data series. Prefer
direct labels (`label_line`) to legends. Mermaid: data boxes
`fill:#f4f3ee,stroke:#898781`; fixed geometric maps
`fill:#ffffff,stroke:#2a78d6,stroke-dasharray:4 3`; learned maps
`fill:#e6f0fb,stroke:#2a78d6`; contracts `stroke:#008300`.

## Shared code

- `palette.py`: colors, `STYLE`, house rcParams, plotting helpers, the running airfoil,
  and the dataset figure loaders (`load_problem`, `load_samples`, `part`, `bc_segments`,
  `field_scatter`, `small_colorbar`, `bc_legend`).
- `standing.py`: the one reader of the standing artifacts. `standing.DATASETS` names the
  artifact per dataset; `Standing(key)` gives `err`, `seeds`, `ratio`, `shift`,
  `verdict`, `table`. No chapter opens a `results/char/` standing file itself, and no
  chapter pastes helpers that these modules hold.

## Figures and tables

- Every axis: quantity and unit in brackets, `[–]` for dimensionless.
- Bars start at zero; log axes only where the range spans a decade, with 1–2–5 ticks
  written out.
- Multi-panel figures stack vertically and share the x-axis; no dual axes.
- Seeds are dots on bars or bands on lines; the seed mean is the mark.
- Renderings of fields (`figures/renderings/`): one named camera per dataset, a shared
  color scale across truth and models, the diverging map for errors, the colorbar
  labeled with the quantity and unit, the case id and each model's error on that case in
  the caption.
- Table captions name the artifact; the `{#tbl-…}` label is on the caption line.

## What the chapters do not do

- **State what is; cut, never retract.** No "before/now/previously/will/no
  longer/retired/ superseded/reframed". A claim now known wrong does not appear at all.
  History lives in the notebook and the ledger only, and chapters do not cite notebook
  entries (`@sec-nb-…`).
- **No dates, no status, no project slang.** No "as of", "wave", "round", "campaign",
  "rerun", "pending", "running"; no node codes (A35b, REF-80, H-RESID) or version tags
  in prose. Say what the experiment measures. A pending decision is written as an open
  question in 20 (with what would settle it), never as status. Exceptions: 14 and 18,
  and the in-flight callout below.
- **The book stays current; in flight is labelled.** Every chapter shows the current
  reference model's measurements as results land. A result that a newer measurement
  replaces is cut from the chapter (its history stays in 14 and 18). Where a chapter
  shows a number that a running experiment will replace, or a question that one will
  decide, put one callout beside it:

  ```markdown
  ::: {.callout-note title="In flight"}
  These numbers come from <what they are>. <What replaces them> is being
  measured, and the ledger (@sec-ledger) tracks it.
  :::
  ```

  Say what is shown and what will replace it, with no dates, run identifiers or node
  codes. When the result lands, update the numbers and remove the callout. Anything
  known wrong is cut, not labelled.

  The supervisor reads Peter's rule of 2026-10-22, "(and possibly labeled "in-flight"
  if you're planning on re-running them / re-assessing)", to cover planned reruns as
  well. A callout may also mark a number that a rerun planned in the ledger (state
  PLANNED) will replace, but only where a measured result already shows that the number
  will change: the lug standing's posed ÷ native, for example, which ISLA retrained with
  the class's tie rule already reads near one. The 0.9.0 standings in general carry no
  callouts, because 07's `sec-standing-geometry` states once that they are 0.9.0's.
- **No run identifiers** in prose; they appear in code cells and in caption artifact
  paths only.
- **No numbers from memory.** Every number is computed from a file in `results/` (or
  `research/`). Define the quantity before the number: what is measured, on which split,
  under what protocol, which way a ratio points.
- **Registered verdicts as registered.** Where a pre-registered test decided something,
  report its outcome as it was registered; do not re-word a failed bar as a pass.
- **Resource-matched comparisons only.** A cross-architecture comparison states
  parameters, tokens per case and memory of both sides, or says where they differ.
- **No hedged verdicts** and no result made to look better than it is. Scope and limits
  sit beside the claim, not only in 20.
- **A property exact in the limit of a performance knob is credited.** Where a property
  holds exactly as a knob that trades accuracy for speed goes to zero, with an error
  that converges to zero, and the model is run where that error is small, the book
  states the property as the model's. A Barnes–Hut opening angle $\theta$ is the case in
  point: GLOBE is rotation-equivariant and query-independent, and ISLA-DTT
  pose-invariant, although at a finite $\theta$ a rotation or a different request
  regroups their clusters. The chapter names the knob once, briefly, as an approximation
  that vanishes as $\theta \to 0$, not as a failure of the property, and keeps any
  deviation it measures at the working $\theta$ as the size of that approximation. A
  property broken with no convergent knob is still stated as a limit: float32
  tie-breaking on tied meshes, a non-equivariant network fed a canonicalized input,
  whose output jumps by $O(1)$ at a tie of the frame, and the query-population
  dependence of interacting ISLA.
- **Chapter work never edits `18-notebook.qmd`, `14-program-status.qmd` or `results/`
  files that already exist.** New result files (a reduction, a rendering) are added with
  their script.
