Status: DONE. Branch `psharpe/isla-bc-supply` (head 0a706b148, pushed) brings SHIFT-SUV's box, HiLift's inlet and outlet
(without the symmetry plane) and the deduplicated wing into canonical ISLA. It already contains `psharpe/isla` eba038f08, and
every final suite passes on it, with 0 CUDA failures. I did not merge it into `psharpe/isla`.

Details are in `REPORT.md` and the log in `JOBS.md`. Paths below are relative to `scratch/isla-bc-supply-2026-10-24/`.

**What ran** (all on LHR; job names isla.bcsupply.*):
- cpu-short: 368832, 368833, 368849, 368881, 368882, 368914, 368927 (I cancelled it: too slow), 369235, 369258, 369270,
  369519, 369520.
- GB300, short QoS: 368918 (smokes and tests) and 369518 (the final CUDA tests).
- Commits:
  - merges 6707e866a (bc-sidecar, clean), 63d6a2ec0 (hl-dedup, two conflicted files resolved with both sides kept) and
    0a706b148 (psharpe/isla eba038f08, clean, done on your 12:25 steer);
  - e7667d24b and ee9237933, my own.
- Compute: 2.55 GPU-h (cap 3); 104.4 core-h allocated (91.8 of them on the GB300 nodes' CPUs).

**HiLift's far field is a recipe transform, not 690 sidecars.** You approved this at 10:42 EDT.
- Curator's inlet and outlet depend only on the radius, the mesh divisions and `U_inf`, and the 1,800 cases store only 10
  distinct `U_inf` vectors, one per angle of attack.
- `AddHemisphereBoundaries` vendors curator ee6aeda's arithmetic. It runs first, at 24533.41796875 in, and costs 3.5 ms per
  load.
- Its validation found no failures (`out/hl_val_368914.jsonl`):
  - it equals the far field stored on LHC001_AoA_10 and _12 bit for bit, in the full and -medium copies;
  - it equals curator's own code for all 10 vectors, on CPU and on the GB300 node;
  - the census covers every case;
  - the radius equals the x and z extent of all 13 stored interiors that hold points, to the bit.
- I deleted the sidecar scaffolds (the reader option and the `_whole` files), as your condition 6 asked.
- `isla_surface_highlift` reads the deduplicated wing, the inlet and the outlet, and predicts at the wing. Its comment says why
  no symmetry plane is read.

**Baseline inputs are unchanged.** GeoTransolver's and Transolver's forward_kwargs and targets are bitwise equal before and
after: SHIFT-SUV 60 of 60 (5 files × 3 cases × 2 models × augment off and on), HiLift 24 of 24.

**The box and the far field move with the body.** On every SHIFT-SUV file (posed included) and every HiLift file (HL-DUP's
posed file included), a single similarity maps the added pieces and the body together (`out/move_369235.jsonl`). The box
costs 2.0 ms per load.

**Final suites on 0a706b148:**
- GB300, every CUDA test: 527 collected, 526 passed, 1 skipped, 0 failed (369518). To fit the remaining 0.91 GPU-h, the job
  split the tests into 8 time-balanced shards on one node, every test in exactly one, and finished in 6:55.
- The ISLA test directory on CPU: 397 passed, 0 failed.
- The recipe tests: 579 passed and 2 failed, the `test_campaign_c_hooks` Slurm TypeError the base also has. The real-data
  far-field tests ran.
- Hooks: pass on every changed file.

**Smokes on the GB300** (on ee9237933, whose model code is the head's; all losses finite; `gb/smoke_summary.txt`):
- SHIFT-SUV: 217 against 169 ms per step (1.29x), 67.6 against 64.9 GB, 204,332 against 200,000 tokens.
- HiLift: 32.2 against 31.0 GB, 97,268 against 93,364 tokens. Its step times measure data loading, so they say nothing about the
  far field's cost.

**The book** (`book/BOOK_CHANGES.md`, run on the head's own tree):
- Chapter 20 fails 3 cells, all from one cause: an assertion that HiLift reads the body alone.
- Chapters 05 and 06 print generated table changes, which need no edit.
- Prose needs edits in 07, 10, 11, 20 and 21; the file gives the text for each.

**What this means for the program.** Canonical ISLA now reads the whole stored or buildable boundary on all nine tasks;
HiLift's symmetry plane is the one piece missing. I am confident in the far field and the box, since both reproduce the stored
or documented domain bit for bit. The standings still come from checkpoints trained with the body alone. There is no overlap
with DEFAULT-DTT's files.

**Rulings I relied on:**
- Peter's ruling on the whole boundary (10-23, 17:08);
- his approval of the supply (10-24, 10:21);
- your choice of a transform over sidecars (10-24, 10:42).

**For project memory:** a domain boundary built from dataset constants and stored case globals is a recipe transform, not a
per-case cache.

**Left for you:**
- merge the branch together with the book batch;
- add `AddHemisphereBoundaries` to HL-DUP's cluster-only `highlift_surface_medium_dedup{,_posed}.yaml` when the canonical
  HiLift runs start (`lhr/make_cluster_files.py` does it);
- schedule the retraining of the standings.

Nothing is running, and the registry row is removed.

**On disk:** the worktree `/home/psharpe/gh/physicsnemo-isla-bc-sidecar` (clean); this folder (about 3 MB); on LHR,
`$T/bcsupply`, `$T/kernel/bcsupply`, the read-only snapshots `$T/recipe_bcsupply_{b921,b63d,e766,ee92}` and
`$T/code_bcsupply_0a70` (57 MB). Nothing is over 1 GiB.
