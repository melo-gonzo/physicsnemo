Status: DONE. HL-DUP stage 2 is read as registered, and the verdict is **mixed**, so no verdict is claimed. Under
the supervisor's amendment the deduplication is canonical on principle, and these rows set the book's HiLift standing.

**What ran** (LHR; `JOBS.md`):

- Six lanes, 500 epochs each, about 4 links each, with no event, alarm or BLOCKED marker. ISLA seed 43 lost its first
  link to the Triton-cache race; the chain recovered on its own.
- Scoring 371516 and 372745, plus the probe on the training cases, 373091.
- The hole prototype (`HOLE_CRUST.md`).
- Stage-2 GPU: 209.6 of the 320 cap.

**The new HiLift standing** (native, seed means, deduplicated wing, metric over distinct cells; `RESULTS_STAGE2.md`,
`stage2_table.md`):

| arm | pressure | wall shear | baseline / ISLA (p / wss) | stage-1 (b) reference |
|---|---|---|---|---|
| ISLA 0.9.0 | 0.0546 (0.0530, 0.0561) | 0.0963 | | |
| GeoTransolver, unit drive | 0.0511 | 0.0921 | 0.936 / 0.957 | 0.917 / 0.922 |
| Transolver, unit drive | 0.0527 | 0.0919 | 0.966 / 0.954 | 0.924 / 0.909 |

**The rule, cell by cell.**

- Wall shear narrows for both baselines, on both seeds.
- GeoTransolver pressure moved +0.019, inside the 0.03 band.
- Transolver pressure: the seeds disagree (1.022, 0.914).

**Predictions.** P1 held, P2 failed (GeoTransolver 5.4 % better than its (b)), P3 failed, P4 held, and P5 is
undetermined: in band on pressure, wall shear and density, below it on velocity and temperature.

**What it means.**

- Every ratio moved toward 1. Both baselines still lead in the mean: GeoTransolver through post-stall cases,
  Transolver through pre-stall ones.
- ISLA is ahead of GeoTransolver before stall on both seeds, and lower on 66 of 84 pre-stall cases.
- ISLA's seed spread rose to 5.8 % on pressure, from 0.07 %.
- With the copies, ISLA's length unit varied 14.6 % across training cases; without them, 1.3 %.
- The old 0.0427 standing is not comparable; the like-for-like old ISLA figure is 0.0589.
- Confidence: firm on wall shear; not firm on the pressure gap to Transolver.

**Second read.** A lane-reviewer on devin/claude-opus-5.5:high (`REVIEW_STAGE2.md`) found 11 fixes. All are applied
(`REVISION_NOTE_STAGE2.md`), and the v1 files are kept.

**Hole.** The constrained NN-crust rebuild (`DESIGN_HOLE.md`) fails its own validation:

- knock-outs: 14 of 40 random runs exact, and the transferred gap patterns fail the transfer check;
- the broken roots fail its first check;
- every failure was caught by a check, so no wrong loop passed.

Side finding: the source STL (41 cases stored, 39 read) holds the same closed root cap, equal to curator's hole to
0.06 in. Whether it may stand for the other cases is your decision with Peter.

**md5s.**

| file | md5 |
|---|---|
| RESULTS_STAGE2.md | 4d9fe204add4b8495e8b549d4063c04f |
| NOTEBOOK_DRAFT_STAGE2.md | 9512b28d5bc7a6f6ded64ac1fbeefe20 |
| REVISION_NOTE_STAGE2.md | 91947532fa10cb88768927a7b09da87f |
| HOLE_CRUST.md | 64cc1c9b050c0498c0f2ae434ed45baa |
| PREREG.md | 072b6b1c3ac1b59df729eadda9f04e40 |

**Rulings relied on.**

- The amendment of 06:44: deduplication canonical, and the masks a read-only cache.
- Never write psharpe/isla-hl-dedup; BC-SUPPLY merges it.
- Peter's rule that subagents launch on Devin.
- New for memory: the masks must stay read-only while BC-SUPPLY's canonical recipe reads them.

**Left.** Nothing is running. Both registry rows are removed, and no monitor remains.

**On disk.**

- LHR: `$T/hldup/masks` (23 GB, read-only; canonical HiLift reads it), `masks_main` (3.1 GB), `$T/evals_hldupx`
  (small), the run dirs `$T/runs/hldupx_hl_*` with their checkpoints, the snapshots `$T/recipe_hldup{,x}_*`, and
  `$T/kernel/hldupx_*`.
- Local: the scratch folder, about 20 MB.
