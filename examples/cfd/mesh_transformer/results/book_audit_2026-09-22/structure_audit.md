# Structure and repetition audit (fresh-context reviewer, read-only, 2026-09-22)

Scope read: STYLE.md, CLAIMS.md (head), index, 01–09, 12, 13, 17 in full; 18-notebook grepped only. Paths are under
`examples/cfd/mesh_transformer/book/`; line numbers refer to the tree at commit 557701af.

**Headline diagnosis.** The book is scoreboard-first (index thesis table → 03 scoreboard → 05/06/08 evidence → 12
limits table → 13 ledger → 17 ledger), and every layer re-states the same eight results with the same numbers.
Parity/0.998 is stated ~14 times, the 10:1-bias factors (1.07x / 8.7x / 5.3x) ~11 times, the training-law factors
~12 times, the linear-routing ceiling story ~9 times, the wing-deficit rungs ~9 times. The two conceptual
centerpieces Peter asked about (discretization invariance; rotation/translation equivariance) have no single home:
each is split across 01 (principle), 02 (mechanism), 03 (row), 06 (three sections), 12 (row), 17 (ledger).

## 1. Concept map (HOME = natural single home; dup = restates without new evidence)

### 1a. Equivariance and pose
- HOME for mechanism: 02-isla.qmd:194-305 (idea 2, frame without origin), 02:532-595 (idea 6, equivariant head),
  02:659-733 (contracts as tests).
- HOME for pose evidence: 04:541-622 (canonicalization control, tbl-pose) merged with 06:669-710 (200k numbers); the
  two sections make the same argument with the same closing sentence.
- Dups: 07:598-619 ("what the pass never does" restates 02 contracts); 03:229-242; index:42-51, index:116,
  index:317-319; 02:767 design-table row; 12:36, 12:121-127; 17:61-62 ledger (ok as ledger).

### 1b. Discretization / sampling invariance (most fragmented concept)
- Principle HOME: 01:525-561 (but already quotes results). Mechanism HOME: 02:132-192 (idea 1; 02:182-192 quotes
  training-law numbers). Instrument HOME: 06:166-292 (fig-sampling-laws).
- Evidence: 06:294-360 (10:1 probe), 06:362-473 (refinement/consistency, 10k checkpoints), 06:475-564 (training law),
  06:566-667 (territories).
- Pure duplicates drawn from the same JSON: 03:244-330 (fig-scoreboard-density + tbl) = 06 tbl-density;
  03:332-359 (tbl-scoreboard-training-law) = fig-training-law, prose nearly verbatim with 06:477-486; 05:181-195.
- Summaries ×6 in index (45-51, 57-66, 117-118, 159-172, 320-327), ×4 in 12 (37-38, 113-120, 226, 239-249),
  17:22-34/60/79-88/105-111, 02:768-769, 06:1-25, 06:759-766.
- Recommendation: one chapter "Discretization: the surface, not the mesh": principle → mechanism → four draws →
  inference-draw evidence → training-law evidence → territories → what is not claimed.

### 1c. Learning-rate ceiling / routing collapse / cosine routing
- Told in full in four places: 02:383-422, 06:116-164, 09:133-221, 04:624-657; echoes at 03:87-92, 03:132-133,
  03:182-192, 03:371-374, 08:123-127, 08:241-248, 12:39, 12:208, 17:65, index:290, index:328-332, 02:771.
- Recommendation: one subsection of the architecture chapter (the two forms; why bounded logits train at
  width/depth; rate-ceiling table as a collapsed note) plus one Part III paragraph on the starting rate. The ceiling is
  a property of the alternative form and needs no results-chapter section.

### 1d. Capacity floor and scaling laws
- HOME: 06:768-1015. Misplaced: 04:284-538 (cells ladder, scaling law, AirfRANS ladder live in the protocol chapter).
- Dups: 03:361-383, 03:385-414 (same JSON as 04 tbl-cells-ladder), 02:735-753, 05:271-278, index:62-66/115,
  12:41-43/224.

### 1e. Interior task
- HOME: 08. Dups: 03:170-227 (same rows as 08 tbl-interior-200k); 08's own lede, 08:194-205 and 08:308-327 state the
  ratios three times; index:119/333-335; 12:40/146-168/263-269; 04:205-215; 17:66-67; 02:699-713 vs 07:558-561 vs
  08:181-191 on query_mass.

### 1f. Wing deficit
- HOME: 05. Dups: 03:136-168 (same three JSONs as tbl-hilift-ladder), 03:1-17, 03:78-93, 03:124-133, 06:109-112,
  12:55-76 (= 05:269-299), index:36-41/313, 12:34/57-60/233-235, 17:71.

### 1g. Fair-comparison protocol
- HOME: 04:217-269. Dups: 01:563-601 (same nine rules), 03:25-54, per-chapter protocol paragraphs 05:22-25/54-65,
  06:27-37, 08:120-137; unit-direction story ×5 (04:117-215, 05:67-73, 01:575, index:228, 12:34); 13:553-561.

### 1h. Cost / memory
- HOME for footprint table: 09:298-339. Evidence: 03:416-499 (fig-cost-sweep; the figure that contradicts the parity
  text). Dups: 02:735-753, 04:271-282, 06:722-738, 12:41/224, index:63-66.

## 2. Path-dependent passages (R = restate as present fact; C = cut; N = move to notebook/status)
index: 127-128 (C provenance), 290 (R), 328-332 (R, no dates), 225/246-248 removed-variant key rows (C).
01: 424-426 (R), 570 (R), 588-601 (N phrasing).
02: 229-242 total-measure callout (C), 385 date (R), 410-422 (N), 528-530 multi-head (C), 623-657 variant history
(R physical argument only; C callout), 673/685-691 (C), 699-713 (N), 762 (C), 771 (R).
03: 22-23/501-520 status ledger inside evidence (N), 51-54 (C), 91 (R), 112-113 (C), 132-133 (N/R), 166 (N), 182-192
(R cosine row only), 383 (N), 509-512 (N, stale).
04: 82-83 (R), 336-340 (N), 589-592 (R), 632-639 (R; mechanism to architecture chapter).
05: 62-65 (C), 98-100 (N), 175-177 (N), 201-204 (R or N), 226-232 empty section (C), 234-258 no-number section (N),
275-278 (N).
06: 29-32 and 1017-1079 removed-variant regime table (N), 107 (C), 118-123 stale (R), 124-126 10k flag, 160-161 stale
(R), 316-319 (C), 366-369/458 10k lanes (N or re-run), 718-719 placeholder (C), 828 stale (R), 1013-1015 (N).
07: 558-561 (N), 633 (C), 635 (R), 642-644 (N).
08: 123-127 (R), 186-191 (N), 220-224 linear rows (demote), 241-248 (C), 284-288/320-323 (N), 380-382 (N).
09: 37 wrong (linear) (R), 38 stale (R), 40 (C), 43-46 (C), 119-122 wrong under cosine (R to 1e-3), 135-137 (R invert),
200-209 (R), 250 (R).
12: 34 stale (N), 39 (R), 45-51/184-193 (N), 67-69 stale/contradicts 05 (R), 94-99/141-142/157-165 (N), 208/210 answered
open questions (C), 255-257 (R).
17: 65 wrong (linear "current default") (R), 74 wrong (R), 55-56 (R), 60-62/69 no-number rows (N or principle only),
117-126 (N).
13: 367 vs 536 and 283-286 inconsistent ROUTENORM status; 538 "as of 2026-09-21".

## 3. Linear routing outside 07 and 02:383-422
Cosine artifacts on disk: routenorm_cars_2026-09-21 (cars width ladder 100k), routenorm_extension_2026-09-21 (depth;
GT-2048; cars interior cosine 192 at 200k), routenorm_p200k_2026-09-22 (35-wing cosine 192/384 at 200k; cars
384x24 at 200k), scalelaw_drivaerml_2026-09-22, airfrans_reduction_2026-09-22, airfrans_scale_2026-09-22.
Not on disk: cosine at 200k for the 435-car surface row, four DrivAerML ladder rungs, pose, fixed-angle rung,
training-law lanes (COSINE-TWINS in flight), 4-geometry/210/21-geometry rungs, density/consistency/territory probes,
cost sweep, cost-logged ISO-COST lanes (memory frontier).
Linear-routed locations: index tbl-thesis rows 1-2, pose row, density/training-law rows, interior row (cosine exists),
fig-headline; 03: tbl-scoreboard-200k, tbl-scoreboard-200k-wings (35-case cosine exists), tbl-scoreboard-200k-interior
(linear row beside cosine), fig-scoreboard-density, tbl-density-100k, tbl-scoreboard-training-law, floor prose,
tbl-scoreboard-cells, fig-cost-sweep; 04: tbl-unit-vs-physical (10k), tbl-cells-ladder; 05: fig-hilift-data-efficiency,
tbl-hilift-ladder, training-law rung; 06: para-full-data-accuracy, para-anatomy-200k, LR robustness section,
tbl-density, fig-consistency-curves/tbl-consistency-40k (10k), fig-training-law, territory figures, para-pose,
tbl-consistency-matrix, tbl-floor row 1, fig-equal-budget/tbl-equal-budget (all linear, memory-frontier claim),
fig-capacity linear series and reading 2; 08: lede, tbl-interior-200k rows 2-4, 241-248, 308-327, tbl-interior-airfrans
linear rows; 09: config table, interior command, ceiling table; 12: established rows 1-7; 17: 65, 74.
Already possible today: cosine interior rows become the rows; quote 35-wing cosine 0.1408 / 0.78; drop linear width
ladder from fig-capacity/tbl-floor to a collapsed note; drop linear interior control rows. Everything else waits for
REF-BOARD.

## 4. DrivAerML Surface vs Volume ambiguities
index:22-24, 28-33, 52-56, 62-66, 142-145 (false for interior), 310-313; 01:218-229 (anchor sec-benchmarks
misleading); 03:1-17 (both tasks called "DrivAerML" in one sentence), 03:78-93, 361-368, 385-390; 04:206-209, 323-341;
05:260-267, 296 (stale "one dataset"); 06:1-6, 770; 12:35 vs 12:40 identical naming, 12:103-104 (false for interior
pressure); 17:60-61/68/71; 13:361/367/385/517-518. Adopt canonical names "DrivAerML Surface task" / "DrivAerML Volume
task" in STYLE.md and require one on every cross-architecture sentence.

## 5. Contradictions
1. Parity text (03:3-6, 06:3-8, index:142-144) vs fig-cost-sweep accuracy panel (ISLA-192 linear 0.0450/0.0448/
   0.0450/0.0452 vs GeoTransolver-256 0.0447/0.0445/0.0445 on a 0.040-0.050 log axis, GeoTransolver visibly below);
   consistent within seed noise but reads as a lead. Same in fig-equal-budget caption 06:835.
2. Cosine default stated (02:385, 07:635, 09:200-201, 12:39, 12:208, 13:367, index:328) vs linear default / pending
   (09:37-38, 09:135, 09:119-122, 06:118-123, 06:160-161, 06:828, 02:771, 03:510-511, 17:65, 17:74, 13:283-286, 13:536).
3. Wing rungs: 05:167-175 and 03:157-166 report 210 and 21-geometry; 12:34 and 12:67-69 say they are training.
4. "Every rung ... 27 to 32%" (05:3-5) omits the 210 (11%/4%) and 21-geometry (2%/5%) rungs; 05:171-174 calls the
   latter within seed noise.
5. 05:295-296 "interior lead on one dataset" vs 08/12:40 two datasets.
6. Interior row choice: index:87-91/119, 03:207-210, 12:263-267, 12:214 use the linear 5e-4 lane (0.87/1.42/0.98)
   though the cosine row (0.95/1.56/0.97) is on disk and is the default configuration.
7. index:124 "about 27% at 35 cases regardless of budget" vs cosine 0.78 (22%) at 200k.
8. 09:119-122 interior must train at 5e-4 vs 08:127 / 03:187-188 cosine trains at 1e-3.
9. Memory-frontier claim 12:41 and 03:377-381 holds only at or below 70 GB (06:932-936).
10. Territory verdict "slices stay put" (03:50-51, index:49-51) vs 06:655-661 "nearly, not exactly".
11. 04:98-102 Transolver 8 layers vs family "12 layers" at 04:30-38, 02:311.
12. 10k-anchored accuracy claims remain: 04:224-231 and 06:918-919 (19% from 10k to 200k), 06:124-137, 06:362-473,
    04:166-196.

## 6. Aerodynamics-only framing
index:228-229; 01:3-6, 164-181, 575; 02:29-37, 90-91, 113, 439-442 ("upwind or downwind"), 455, 471, 565;
04:117-215 (section title "The freestream must enter as a unit direction"; rule 04:198-204), 04:610-621 is the only
general statement; 07:262 ("upwind of it"); 09:4, 58-62, 81-87, 94-104 (general case collapsed); 03:31-32; 05:56, 67-73;
12:212, 223, 34; 08:338-340 (AirfRANS is the first S = 1 instance and should teach the general interface).
Recommendation: state the general interface once (boundary cells + K directions + S scalars + optional per-cell
boundary data); external aero as the benchmark instance with K = 1, S in {0, 1}; neutral words in mechanism text; the
unit-direction rule becomes "directions are unit vectors; magnitudes are scalars".

## 7. Proposed new outline (motivate -> architecture -> protocol -> evidence -> limits)
0 Front matter: thesis paragraph; ONE thesis table with the task named per row; visual key; how to read; what is not
  claimed (no repeated numbers).
1 The problem: boundary-driven steady PDEs and their surrogates (general operator interface; why external aero is the
  benchmark instance; metric and instrument).
2 ISLA, one idea at a time (ideas 1-6; routing forms inside idea 3, cosine default, linear as a collapsed note with the
  ceiling table; contracts table; no evidence numbers).
3 The two invariances ISLA is built for (NEW centerpiece): 3.1 equivariance and pose (contract; canonicalization
  argument; when no canonical frame exists); 3.2 discretization (principle; the four draws; what "consistent" means).
4 The forward pass (07).
5 Baselines, datasets and the protocol (04:1-282 + datasets from 01 + nine rules; unit-direction rule general).
6 Using ISLA (one configuration; commands; width rule as a starting rate; sampler; checkpoints; cost table).
7 Surface accuracy and data efficiency (DrivAerML Surface, HiLiftAeroML): one table per dataset, the ladders, the
  deficit anatomy, draw-vs-seed controls.
8 Discretization and pose: the evidence (inference draw; refinement bench re-run at protocol or dropped; training law;
  territories; pose at 200k in one place).
9 Capacity, scaling laws and cost (cells flat; width/depth ladders; frontier; fitted law; AirfRANS case axis; cost sweep
  without an accuracy panel or with parity bands).
10 Boundary to interior (cosine rows are the rows; DrivAerML Volume then AirfRANS; query independence; negatives).
11 Limits and open questions (limits only; open questions with bars minus answered ones).
A Program status (13); B Design decisions (17, cosine row fixed); C Lab notebook.
Cuts: 03-benchmarks.qmd as a chapter (its tables move to 7/8/9/10); 06:362-473 unless re-run; 06:1017-1079 to the
notebook; 12:22-51 replaced by the thesis table; per-chapter protocol paragraphs.
Before the rewrite can be fully present-tense: REF-BOARD lanes (435-car row, ladder, pose, fixed angle, training law,
remaining wing rungs, cost-logged ISO-COST lanes, density/consistency/territory probes on a reference checkpoint).
