# NOTEBOOK_DRAFT: HL-AREA (draft for the supervisor; the lane did not edit the book)

Source: `scratch/isla-hl-area-2026-10-24/RESULTS.md`, `area_table.md` (784802a1), `area_summary.json` (feb1aff2), `COST_REPORT.md`,
`cost/cost_hldupx.json` (313ddf9b), `cost/extract.jsonl` (3879a90c). Book HEAD read: 5b70c8686. Line numbers below are at that head.

## Notebook entry (18-notebook.qmd)

```markdown
## 2026-10-24 — HL-AREA: on the uniform point cloud ISLA's error on the deduplicated HiLift wing is unchanged (0.989 to 0.999 of native) while the baselines' pressure error rises 3.6 to 5.5 times; the deduplicated lanes price within 6% of the stored-wing lanes {#sec-nb-hl-area}

**Question.** HL-DUP stage 2 scored its six deduplicated-wing checkpoints (ISLA 0.9.0, GeoTransolver and Transolver
with the unit drive, seeds 42 and 43) on the native mesh and rotated and translated only, and the book priced HiLift from
the lanes trained on the wing as stored. What do the same checkpoints score on the uniform point cloud, and what did their
own training cost?

**Method.** The scorer was HL-DUP stage 2's, with only these changes: the protocol, the run directory, the time limit, the
bad nodes excluded, and the Transolver rows run beside two others on the same node to stay within a 3 GPU-h cap. It used
the same inference command: float32, the 180 validation cases, the final checkpoint. The point cloud is CHAR-PROTOCOLS'
twin applied to the deduplicated dataset file. The reader draws a 1,000,000-cell block and drops the extra copies (about
934,000 cells kept). The area-proportional draw with exact Horvitz–Thompson weights then takes about 200,000 cells. The
cost is the book's rule (cost_090), from the six lanes' own training logs: the median step without compile steps, the
peak memory, and GPU-h from the scheduler.

**Checks.** Every row was at epoch 500 and scored the same 180 cases as native. The protocol is not degenerate: the
per-case pressure differs from native on 180 of 180 cases for every run. The cost extractor, rerun on the stored-wing
lanes, reproduces cost_090.json's per-seed values exactly.

**Result.** Seed means, seeds 42 / 43 in parentheses:

| arm | pressure, point cloud | wall shear, point cloud | point cloud ÷ native, pressure |
|---|---|---|---|
| ISLA 0.9.0 | 0.0543 (0.0525, 0.0561) | 0.0961 (0.0942, 0.0981) | 0.989, 0.999 |
| GeoTransolver 256 | 0.261 (0.239, 0.283) | 0.353 (0.331, 0.375) | 4.72, 5.48 |
| Transolver 512 × 16 | 0.189 (0.195, 0.184) | 0.295 (0.310, 0.280) | 3.60, 3.58 |

Baseline ÷ ISLA on the point cloud is 4.81 (pressure) and 3.67 (wall shear) for GeoTransolver, and 3.49 and 3.07 for
Transolver. The per-seed ratios run from 2.86 to 5.05. On the native mesh the same ratios are 0.936 / 0.957 and
0.966 / 0.954. ISLA's point-cloud error is 0.1 to 1.1% below its native error on every field and seed.

Cost, seed means, deduplicated (stored wing):

| | step [s] | peak [GiB per GPU] | GPU-h per lane |
|---|---|---|---|
| ISLA | 0.323 (0.341) | 64.6 (65.6) | 32.8 (32.5) |
| GeoTransolver | 0.431 (0.457) | 77.9 (77.9) | 38.3 (39.2) |
| Transolver | 0.323 (0.342) | 62.3 (62.3) | 32.3 (31.9) |

Each step is 5 to 6% shorter on every architecture because each draw keeps about 7% fewer cells. Loading and validation
fill the rest of each epoch, so GPU-h per lane moves only 0.98 to 1.01 times.

**Reading.** HiLiftAeroML has a point-cloud standing again, and it reads as DrivAerML Surface and SHIFT-SUV do. ISLA's
prediction does not depend on how the surface was sampled. Both baselines read the mesher's cell density, and their
errors rise 2.8 to 5.5 times across the five fields. Firm: the two seeds agree within 15% on every baseline ratio. The cost table's
cross-architecture ratios do not change.

Records: scratch/isla-hl-area-2026-10-24/ (RESULTS.md, area_table.md, COST_REPORT.md, cost/); LHR job 375068, 1.09 GPU-h.
```

## Chapter changes

### Data and generators (the supervisor applies these, then the chapters follow)

1. **Score records.** Copy `scores_area/<run>/area/<run>/metrics.jsonl` and `DATASET_USED` for the six runs into
   `examples/cfd/isla/results/campaigns_2026-10-23/hl_dup_stage2_scores/scores_stage2/<run>/area/<run>/`. Add their md5s to that
   folder's RECORDS.md / records_md5.txt. The metrics md5s are in RESULTS.md.
2. **`results/char/hilift_surface_090_dedup_2026-10-24.py`.**
   - :62-67: `PROTOS = {"native": "native mesh", "area": "uniform point cloud", "posed": "rotated + translated"}`. This is the label
     the stored-wing file used.
   - :94: accept `cfg["dataset"] == DATASET + {"native": "", "posed": "_posed", "area": "_area"}[protocol]`.
   - Docstring :23 and the `"protocol"` text: replace "Native mesh and rotated + translated; the uniform point cloud was not scored
     on this data." with "Native mesh, uniform point cloud (a 1,000,000-cell draw without its extra copies, then the
     area-proportional draw of about 200,000 cells with exact Horvitz–Thompson weights; HL-AREA, LHR job 375068) and rotated +
     translated."
   - Rerun it to regenerate `hilift_surface_090_dedup.json`. `case_level`, `registered` and `bands` read native only and do not
     change.
3. **`book/standing.py:169-181`.** Set the hilift `Dataset` protocols to `("native", "area", "posed")`. In the comment, replace
   "native and posed (the uniform point cloud was not scored on this data)" with "native, on the uniform point cloud (HL-AREA) and
   posed".
4. **Cost.**
   - Copy `cost/extract.jsonl` (six lines, hldupx_hl_*) to `results/campaigns_2026-10-24/hl_area_cost/extract.jsonl`, with a README
     that names `extract_cost.py` (9b87878e, unchanged) and the source logs.
   - In `results/char/cost_090_2026-10-22.py`, read that file beside `EXTRACT` (:48) and change the map at :69-72 to
     `"hilift": {"ISLA": "hldupx_hl_isla", "GeoTransolver": "hldupx_hl_gt", "Transolver": "hldupx_hl_t"}`. Then rerun it.
   - ISLA's note stays "trained on code_h090, before the compile fix (the compiled blocks ran Inductor's mix-order reduction)": the
     dedup ISLA lanes ran code_h090 with `model.compile_blocks=true`, as the stored ones did. 08's BEFORE list is unchanged.
   - Expected values (seed means): ISLA 0.3230 s, 64.61 GiB, 68,817,006 parameters. GeoTransolver 0.4307 s, 77.86 GiB, 9,047,301.
     Transolver 0.3228 s, 62.26 GiB, 47,239,305. All bfloat16, 4 GPUs, 4 ranks.

### Chapter 10 (10-hilift.qmd)

- **:350-351 (para-hilift-measured).** Replace "Each checkpoint is scored two ways, as on DrivAerML (@sec-drivaerml-scorings): on
  the native mesh, and rotated and translated with $\\hat g_1$ rotated alike. The uniform point cloud was not scored on the
  deduplicated wing." with "Each checkpoint is scored three ways, as on DrivAerML (@sec-drivaerml-scorings): on the native mesh, on
  the uniform point cloud, and rotated and translated with $\\hat g_1$ rotated alike."
- **:426-431 (para-de-lede).**
  - In the bold lead, change "; rotated and translated, ISLA leads both by a wide margin.**" to "; on the uniform point cloud and
    rotated and translated, ISLA leads both by a wide margin.**".
  - After the native sentence, insert
    `f"On the uniform point cloud, baseline ÷ ISLA is {span('area')[0]} to {span('area')[1]} on the two fields; rotated and translated, "`.
    This replaces the opening "Rotated and translated, " of the next sentence. The values print as 3.1 to 4.8.
- **:437-438 (tbl-hilift-protocols).** The area column appears by itself. Extend the caption: "…the distinct cells of 200,000 drawn
  per case (on the uniform point cloud, about 200,000 drawn by area from the distinct cells of a 1,000,000-cell draw)".
- **:452-462 (fig-hilift-shifts caption).**
  - "under the two scorings of one checkpoint" → "under the three scorings of one checkpoint".
  - "on the rotated row ISLA's marker stays put and the baselines' move right" → "on the point-cloud and rotated rows ISLA's marker
    stays put and the baselines' move right".
  - `protocol_dots` loops over `ds.protocols`, so the area row appears by itself.
- **:469-487 (para-hilift-protocols-reading).**
  - "Two readings." → "Three readings."
  - Add before the closing sentences:
    `f"**Third, on the uniform point cloud ISLA's error is {-100*max(adev):.1f}% to {-100*min(adev):.1f}% below its native error, while each baseline's is {min(bsh):.1f} to {max(bsh):.1f} times its own**, as on the DrivAerML Surface task and SHIFT-SUV: the baselines read the mesher's cell density, ISLA does not. "`
  - Define `adev = [p / n - 1 for f in HL_F for p, n in zip(HL.seeds("ISLA", f, "area"), HL.seeds("ISLA", f))]` and
    `bsh = [HL.shift(b, f, "area") for b in S.BASELINES for f in HL_F]`, using the existing `Standing.shift`.
  - Expected print: 0.1% to 1.1% below, and 3.2 to 5.1 times. HL_F is pressure and wall shear; the seed means' shifts are GT
    5.11 / 3.83 and T 3.59 / 3.21.
- No In-flight callout to add or remove. The symmetry-plane callout at :297-298 is unrelated.

### Chapter 08 (08-standing.qmd)

- **:86-89.** Cut "; the HiLiftAeroML standing, on the deduplicated wing, is not scored on it". The sentence ends at "…and changes
  no model's input anywhere else."
- **:93-98 (fig-standing-shifts caption).** "on the surface tasks where that changes the scored set and the standing scores it" →
  "on the three surface tasks where that changes the scored set". `SURF` (:127) picks HiLift up again, and the "{len(SURF)} surface
  tasks" count reads 3.
- **:314-315 (tbl-standing-cost caption).** Cut the sentence "The HiLiftAeroML lanes priced are the same configurations trained on
  the wing as stored; its standing is of the deduplicated wing, whose lanes drop the copies from each draw and are not priced
  here." Recommended replacement, as a scope statement:
  `f"The {S.DATASETS['hilift'].short} lanes train on the deduplicated wing, so each draw keeps only the distinct cells of the 200,000 drawn (Transolver's of 100,000). "`
- **Table cells after the regeneration.** Step: ISLA 0.32 (was 0.34), GeoTransolver 0.43 (0.46), Transolver 0.32 (0.34). Peak memory
  (.0f): ISLA 65 (66), GeoTransolver 78, Transolver 62. Parameters unchanged.

### Chapter 19 (19-capacity.qmd)

- **:681-682 (tbl-ref-cost caption).** Cut "The HiLiftAeroML lanes priced are the same configurations trained on the wing as
  stored; its errors are of the deduplicated wing, whose lanes drop the copies from each draw and are not priced here." Use the same
  replacement sentence as 08 ("…its errors are of the deduplicated wing" is then true of both the price and the error).
- **Table cells after the regeneration.** Peak (.1f): ISLA 64.6 (was 65.6), GeoTransolver 77.9, Transolver 62.3. Step (.2f): 0.32 /
  0.43 / 0.32 (was 0.34 / 0.46 / 0.34). Error and baseline ÷ ISLA come from the dedup standing already and are unchanged.
- **Unchanged.** `ref_point` (:48-50) reads only n_parameters, which deduplication does not change. The scaling stars (`HL_STORED`,
  :44-46, :739, :756) stay on the wing as stored, like the ladders.

### Tied to the same flag (outside 10/08/19; they read wrong once the area column exists)

- **04-invariances.qmd.**
  - :387: cut "; HiLiftAeroML's, on the deduplicated wing, scores the native mesh and the rotated and translated cases only".
  - :399-401 caption: "on the surface tasks where that protocol resamples the scored cells and the standing scores it (the
    HiLiftAeroML standing, on the deduplicated wing, does not)" → "on the surface tasks where that protocol resamples the scored
    cells".
  - :459-460: cut "and the standing scores it; the HiLiftAeroML standing, on the deduplicated wing, is not scored on it".
- **07-benchmark.qmd.**
  - :690: cut "(not scored on HiLiftAeroML's deduplicated wing)".
  - :709-710: "the DrivAerML Surface task and SHIFT-SUV appear as point-cloud standings (@fig-contract-shifts); the HiLiftAeroML
    standing, on the deduplicated wing, was not scored on the point cloud." → "the DrivAerML Surface task, HiLiftAeroML and SHIFT-SUV
    appear as point-cloud standings (@fig-contract-shifts)."
- **index.qmd:95.** `len(SURF)` returns to 3 by itself. Check the sentence still reads.
- **20 (surface paragraph) and 04's point-cloud count.** HL-DUP's book lane changed these to "2 tasks". They follow `SURF` or need
  "two" → "three". Check them on render.
- **14-program-status.qmd.** Add a ledger row for HL-AREA: DONE 2026-10-24. Point-cloud scoring of the six deduplicated
  checkpoints and their cost from their own logs; LHR 375068, 1.09 GPU-h; scratch/isla-hl-area-2026-10-24/.
