Status: DONE
Revised after REVIEW.md, 2026-10-24 02:07 EDT (all fourteen items; first version: REPORT_v1.md; changes: REVISION_NOTE.md).

**DK-COST: cost model of ISLA-DTT's boundary read (2026-10-24 01:30 EDT).** At 400k points on DrivAerML Volume (whole boundary), a GB300 (fit limit 274 GiB of its 276.6 GiB) fits theta 3 (any leaf, 1.83-2.0 s), theta 2 (leaf 2-8, 2.07-2.15 s) and theta 1.5 (leaf 4 measured at 2.7 s / 273 GiB; leaf 2 predicted at 2.3 s / 272 GiB; both within 2 GiB of the limit). Nothing at theta <= 1.25 fits at any leaf.

**What ran.** LHR cpu-short jobs 363138, 363201, 363251, 363377 and 363403 (smoke and debug); GPU jobs 363435 (48 min) and 364009 (6 min), normal QoS. GPU-h: 3.63 of 5. No short-QoS slot taken, no commits, nothing running.
- Code: `$T/code_geobundle2` (tree_read.py md5 57891d30) with `$T/recipe_r090_fd`, each dataset's ISLA config with query_independent=true.
- **Random weights (seed 0), no checkpoint.** Pairs depend only on geometry, theta and leaf. Time and memory are bf16 training steps (forward, a mean-square loss on every output, backward, Muon + AdamW), median of 5-8 steps.
- The query geometry is computed once per point set and excluded from step times.
- Commands and md5s: `JOBS.md`. Probe: `cost_probe.py`, `make_plans.py`. Fit: `fit_cost.py` writes `cost.json` and `cost_table.md`; raw rows are in `raw/` and `raw2/`.

**Checks.** The probe's pair count equals the model's own count on 39 of 40 checks; the 40th differs by 1 node pair in 41M. On DrivAerML these checks cover N 25k-100k tokens on case 0 only, not the training N of 200k (that load stopped at the INT_MAX limit, result 3). On DrivAerML at theta 1, leaf 4 the step comes close to the brief's: 1,180 ms / 132 GiB at 100k (brief: 1,202 / 129), 1,790 / 189 at 200k (1,833 / 183), and out of memory at 400k. It is not like-for-like:
- Reference (RENDER-090 job 336005, bench_step_090.py): `code_cmpaf_fix` (tree_read.py c99dbc37), model isla_volume, the car alone (16,647,247 cells), one train-loader sample, the recipe's LossCalculator and finite-gradient check, the query geometry inside each timed step.
- Probe: `code_geobundle2` (tree_read.py 57891d30: column products instead of the float64 frame GEMM, and a new octant tie rule), isla_volume_floor, the car (17,029,624 cells) plus 5,054 domain cells, val case run_112, a mean-square loss, no gradient check, the geometry memoized out of step_ms.
- Shared: bf16, Muon + AdamW, hidden 512, 512 slices, compile_blocks, 200k tokens, Q 100k / 200k.
At full size the synthetic case reproduces REPORT_dtt's scaling table: 211 / 232 / 250 / 201 / 191 pairs per point at 12k-192k cells, against 212 / 234 / 252 / 200 / 191. Its 261 came from another random draw of the same geometry.

**Model.**
- **Pairs per point,** box test only: A theta^-p leaf^-r (log2 N - c) for the cluster nodes, plus B theta^-q leaf^s for the near-field cells.
  - DrivAerML: A = 125, p = 0.91, r = 0.14, c = 9.8; B = 2.7, q = 1.5, s = 2.1. Other datasets: `cost_table.md`.
  - Fit error: median 4-12%, max 24-38% on the volume datasets; HiLift (surface) median 13%, max 39%.
  - Held-out largest N: DrivAerML 800k 7% median / 15% max, AeroZoo 380k 8% / 25%, AirfRANS 8k 5% / 17%, lug 470k 16% / 24%, synthetic 384k 25% / 36%, HiLift 400k 46% / 65%.
  - On the volume datasets the limit is a draw-to-draw sawtooth of about ±25% in N (for example, the synthetic case at 96k-192k cells), not the form. On HiLift the form itself fails: it has no term for the bf16 near-field cells (result 4), which grow with N.
- **Step time** (DrivAerML) = 203 ms + 5.16 ms per million pairs per block + 158 ms per 100k points + 184 ms per 100k tokens; fit error median 1.9%, max 6.2%. AeroZoo gives 5.23 ms per million pairs. Held-out Q = 400k: median 1.8%, max 15% (the row at 273 GiB ran 18% slower than predicted).
- **Peak memory** = 3.2 GiB + 0.265 GiB per million pairs per block + 35.0 GiB per 100k points + 36.0 GiB per 100k tokens. The per-pair term is 0.264-0.276 GiB per million on every dataset. Held-out error is at most 0.4%.
- **Tree time** (build, walk, plan, basis): 240-620 ms per DrivAerML step. That is DTT-FAST's forest and walk pieces; I did not profile it further.

**Results.**
1. **Coherence and box are separate terms.** At theta >= 0.75 coherence adds at most 2% pairs and rejects at most 1% of box-admitted clusters, except AeroZoo at 0.75: +4% / +5% / +9% pairs at N 200k / 100k / 380k (at most 0.9% rejected). This is consistent with DK-DESIGN's bound. At theta 0.5 the coupled read costs:
   - **DrivAerML:** 3.2x the box-only read at 200k tokens, leaf 4 (5,830 vs 1,851 pairs per point) and 3.9x at 800k; coherence rejects 17-23% of the box-admitted clusters at leaf 4 (up to 28% at leaf 16).
   - **Other datasets,** leaf 4 over all N: AeroZoo 1.06-1.71x, lug 1.00-1.40x, HiLift 1.04-1.08x (all leaves: 1.05-1.88x, 1.00-1.51x, 1.04-1.10x); AirfRANS and synthetic 1.0x.
   - **Coherence 0.5 at box 1:** 6.1x on DrivAerML at 200k tokens, leaf 4 (5,103 vs 833 pairs per point, 27.5% rejected); 4.2-6.1x over 25k-200k.
   - **DK-DESIGN's D2 (box 0.7, coherence off):** 1,237 pairs per point, 1.5x theta 1. At 100k points it runs at 1,389 ms / 143 GiB (measured with coherence 1.0, the same read: coherence is inactive above 0.919, and the coherence-1.0 and coherence-off counts are equal on all 12 DrivAerML rows that have both); at 200k, 2.16 s / 211 GiB (predicted). At most 293k points fit.
2. **Leaf size barely matters below 8 in pairs.** On DrivAerML at theta 1, against leaf 4: leaf 2 has -2% pairs, leaf 8 +13%, leaf 16 +56%. Leaf 2 has the fewest pairs (or ties) on every dataset. Cluster nodes are 93-99% of pairs at leaf 4 on the volume datasets. The DrivAerML step: leaf 2 is no faster (1,214 vs 1,180 ms at 100k points); at 200k, leaf 8 is +3% and leaf 16 +23%. Leaf 4 is cheapest or tied (within 0.3%) only on DrivAerML at theta <= 2. At theta 3, leaf 16 is faster by 9% on the lug (308 vs 339 ms), 6% on AirfRANS (874 vs 930) and 5.5% on AeroZoo (682 vs 722), and leaf 8 is 2-3% faster on DrivAerML (880 vs 910, 1,196 vs 1,218).
3. **400k points.** Projection table: `cost_table.md`, last section. Fits = predicted peak <= 274 GiB (272.9 ran; predicted 277-279 went out of memory).
   - **Fits (measured):** theta 3, leaf 4: 1,837 ms / 248 GiB; theta 2, leaf 4: 2,070 / 260; theta 2, leaf 8: 2,117 / 264; theta 1.5, leaf 4: 2,734 / 273 (at the limit). Theta 1.5, leaf 2 is predicted only (2,303 / 272).
   - **Does not fit:** theta 1.25 at any leaf (285 GiB predicted).
   - **Largest point counts that fit:** 349k at theta 1, 380k at 1.25, 431k at 2, 462k at 3.
   - **Against interacting ISLA:** 796 ms / 190 GiB measured at 400k (RENDER-090 job 336005, `isla_volume_090.json`; the reference configuration above, car alone). The settings that fit run 1,837-2,734 ms, 2.3-3.4x that time (3.4x at theta 1.5).
   - **What limits Q:** the blocks' 35 GiB per 100k points caps it near 565k even with no pairs. At theta 1 the read adds 22 GiB per 100k points. Halving the bytes per pair (a fused read, DTT-FAST) would fit theta 1 at about 430k.
   - **The walk's INT_MAX limit:** `point_pairs` fails past 2^31 - 1 pairs in one call (argsort, tree_read.py line 377). Observed in the coherence ladder (coherence 0.25, box 1, leaf 16) at N 200k tokens with 50k points; the same setting at N 100k held 1.456e9 pairs. Theta 0.5 at 400k points is projected past it (5,830 x 400k = 2.33e9).
4. **The forest is built in bfloat16.** `points @ frame` runs under autocast (tree_read.py lines 267 and 366), so the trained tree is a bf16 tree. A float32 scoring run reads a different tree: pairs per point differ by -2% to +9% on DrivAerML, -34% to +9% on AeroZoo, and +1% to +10% on AirfRANS. On the surface form (HiLift, the queries are the cells) bf16 adds a near-field cell term that theta does not shrink. It is 1.25x the float32 pairs at 50k cells and 2.4x at 400k (p99 23,000 pairs per point); float32 stays log-N (916 per point at 400k, theta 1). The current surface form costs 2,336 ms / 213 GiB at 200k cells, theta 1, and runs out of memory at 400k at theta 1, 2 and 3 (the only ones tried). Its exact query geometry for on-wall points took 570 s / 115 GiB per call at 50k cells (job 363435).

**How sure.** High for memory (≤0.4% held-out). High for the DrivAerML step: fit median 1.9%, max 6.2%; held-out Q = 400k median 1.8%, max 15% (the row at the memory limit); close to the brief's numbers in a different configuration (Checks). Medium for pairs extrapolated in N on the volume datasets (±25% draw sawtooth) and for steps within about 5 GiB of the memory limit. Low for HiLift pairs beyond 200k (held-out 46% / 65%). Step times leave out the query geometry: 3-13 s on the first call per point set on DrivAerML; BOUNDARY-AUDIT measured 2.4-2.8 s per step.

**Rulings relied on:** the whole PDE boundary (Peter, 2026-10-23), probe constants set only in the probe's process, GPU work on LHR, and reference code only. **For project memory:**
- ISLA-DTT's forest is bf16 under bf16 training, so a theta, leaf or precision probe should state which tree it reads.
- Coherence is inactive at theta >= 0.919 in 3D.
- The walk fails past 2^31 - 1 pairs per call (argsort): seen at coherence 0.25, box 1, leaf 16 with 200k tokens and 50k points on DrivAerML; projected for theta 0.5 at 400k points.

**Left.** Not measured: Q = 600k (out of memory at every setting tried), HiLift at 800k, AeroZoo's token-count ladder in timing, the DrivAerML model-own check at N 200k. On disk: this folder (about 7 MB), `$T/kernel/dttknobs_cost` (7.4 MB), no worktrees.
