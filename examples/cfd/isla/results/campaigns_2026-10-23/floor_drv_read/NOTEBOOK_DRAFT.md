# FLOOR-DRV notebook draft (for the supervisor; the lane never edits the book)

Written 2026-10-24 about 07:35 EDT, updated 15:01 EDT while job 355565 ran; filled by the reader lane 2026-10-26 about
05:35 EDT after the read. Source files: `RESULTS.md`, `PREREG.md`, `JOBS.md`, `tables/reduce_fd.md` (md5 eea5db7c),
`tables/resid_fd.md` (560eba98), `tables/residuals_fd.csv` (6eaf5138). Line numbers below are at psharpe/isla
fa1cab63d (2026-10-26 03:29 EDT); the book has moved since the first draft (fb4b23e0c).

## A. Chapter changes while in flight (superseded)

Part A (the in-flight ledger clause and the 20-limits "In flight" callout) is superseded by part C: the runs are done.

## B. Notebook entry

### FLOOR-DRV: one DrivAerML Volume car with the domain's whole boundary {#sec-nb-floor-drv}

**Given the ground, the slip walls, the inlet and the outlet, ISLA's far-field eddy-viscosity error on one DrivAerML
Volume car falls 2.4 times, to within 1.12-1.34 times Transolver's (registered reading "closes"), and its velocity
floor falls by 20-28 % ("helps") but stays 1.72-1.74 times Transolver's.**

The whole boundary is adopted by Peter's ruling of 2026-10-23; FLOOR-DRV measures its effect. Peter (17:08 EDT
10-23): "Seeing where the floor is through raw coordinates is a bug, not a feature. Even if there were an accuracy
*cost* to reading the whole boundary (not that I expect it), I would recommend this."

- **Design.** CAP-AUDIT's Part B protocol on run_171 (a fresh draw of 200,000 points every step, 20,000 steps, eager,
  float32, the corrected held-out set), two seeds per arm in one job on one node. Arm F reads the vehicle and the four
  domain meshes the case files store (`no_slip`, the ground under and behind the car; `slip`, the sides, the roof and
  the upstream ground; `inlet`; `outlet`), each by its name with no fields, since the files store none on them. C is
  Part B's ISLA rerun beside it. Transolver and GeoTransolver are Part B's runs; they read each point's coordinates, so
  they see the ground's height through z.
- **Checks.** On all 484 files the four meshes exist and every face points into the fluid. In every draw they move
  with the car by one scale and translation (within 0.004 mm). With them ISLA's length unit is 59.5 m instead of
  2.04 m, and 18 % of the points are nearer a domain wall than the car. All four runs completed 20,000 steps with
  F's five meshes and C's vehicle in their setup records, Part B's point hashes and the whole-draw held-out set.
- **Readings** (registered, PREREG section 4; seeds 42 / 43):
  - far-field eddy viscosity (more than 5 m from the car), held-out e_R: F 0.0130 / 0.0119, C 0.0310 / 0.0281,
    Transolver 0.0097 / 0.0107 (GeoTransolver 0.0115 / 0.0114); F / Transolver 1.34 / 1.12, F / C 0.42 / 0.42; label
    **closes** (bar: at most 1.5 times Transolver on both seeds);
  - velocity, held-out floor: F 0.0153 / 0.0149, C 0.0191 / 0.0207, Transolver 0.0088 / 0.0086; F / C 0.80 / 0.72,
    both F seeds below all five ISLA reference runs (0.0191-0.0207); label **helps** (bar: at most C / 1.2 on both
    seeds and below every reference). On seed 42 the margin is thin (0.80 against the bar 0.83).
- **Beside them** (descriptive):
  - held-out floors F / C: pressure 0.0092 / 0.0079 against 0.0098 / 0.0117 (F / C 0.95 / 0.68), eddy viscosity
    0.0144 / 0.0132 against 0.0208 / 0.0204 (0.69 / 0.65); F's floor sits 1.33-1.74 times the best baseline on the
    three fields, C's 1.66-2.39;
  - the share of the eddy-viscosity squared error beyond 5 m (0.7 % of the points) falls from 41 % (C) to 16 % (F);
  - F's error is below C's in every wall-distance band, from within 0.5 mm of the car to beyond 5 m, on all three
    fields and both seeds (36 of 36): the 29 times smaller s = d / L costs nothing near the wall. Velocity improves in
    every band (F / C 0.63-0.90); eddy viscosity most beyond 5 m (0.42) and least within 0.5 mm (0.82-0.96);
  - C reproduces Part B's ISLA: velocity 0.995 / 1.033 of it, eddy viscosity 0.87 / 0.93.
- **What it means.** On this car the missing boundary was a cause of ISLA's far-field eddy-viscosity miss: with the
  domain's whole boundary that miss closes to near Transolver's. It explains a part of the uniform velocity factor,
  not most of it: F's velocity floor is still 1.72-1.74 times Transolver's, and 1.4-1.9 times in every band, so the
  rest has another cause (CAP-AUDIT 15.10 points to depth). One car and two seeds per arm; the eddy-viscosity effect (2.4 times) is far above
  the 1.46 times spread of the five ISLA reference runs.
- **Cost.** With the four meshes ISLA 0.9.0's step took 1.58-1.61 times as long in the smoke and the first logs
  (5.27-5.29 s against 3.29-3.33 s), and **1.74-1.76 times over the full runs** (5.25 s against 3.01 s; C's step settled
  at 3.00 s after its first 1,000 steps, F's did not). Memory unchanged (234.4 GiB). Almost all of the added time is the
  Warp seed query of the closest-cell search (BOUNDARY-AUDIT, GEO-FAST). Job 355565 was requeued seven times (F
  resumed six times from its checkpoints, C three times, not bit-for-bit, F and C alike); **148.2 GPU-h** in all
  (146.6 for the job, by sacct), under the 150 GPU-h cap.
- **Predictions** (PREREG section 5): W1 nu_t "closes" (p 0.15) true; W2-W4 false; W6 velocity "helps" (p 0.12) true;
  W5, W7, W8 false; W9 C reproduces Part B within 10 % (0.8) true; W10 F's eddy-viscosity floor below C's (0.55) true;
  W11 F's far-field velocity error below C's (0.5) true. Both labels fell where I gave low probability.
- Data: scratch/isla-floor-drv-2026-10-23/RESULTS.md, tables/reduce_fd.md, tables/resid_fd.md, tables/residuals_fd.csv.

## C. Chapter changes after the read (branches chosen by the labels: nu_t "closes", velocity "helps")

1. **09-drivaerml.qmd, open questions** (line 962, "How much of ISLA's far-field error is the missing ground?"):
   nu_t reads "closes", so remove the bullet from the open questions and add one sentence to the Volume-task
   discussion: "Given the domain's four other boundary meshes, ISLA's far-field eddy-viscosity error on one car falls
   from 0.031 / 0.028 to 0.013 / 0.012 (Transolver 0.0097 / 0.0107; @sec-nb-floor-drv)." Line 290 also cites
   @sec-oq-far-field (see item 3).
2. **19-capacity.qmd, @sec-single-case-floor**:
   - the lead (line 445, "ISLA misses the eddy viscosity far from the car, whose level there follows the height above
     a floor ISLA is not given"): append "; given the floor and the domain's other walls, that miss closes to within
     1.12-1.34 times Transolver's (@sec-nb-floor-drv)". Velocity reads "helps", not "no effect", so instead of the
     "not a missing-input effect" sentence add: "Its velocity floor falls by 20-28 % with them but stays 1.72-1.74
     times Transolver's, so most of the uniform velocity factor has another cause."
   - the probes paragraph (lines 626-627): replace "Whether those four meshes, the domain's whole boundary, remove the
     far-field miss when given as boundary inputs is @sec-oq-far-field." with "Given as boundary inputs, those four
     meshes, the domain's whole boundary, bring the far-field error to 0.013 / 0.012, within 1.12-1.34 times
     Transolver's (@sec-nb-floor-drv)."
3. **20-limits.qmd, @sec-oq-far-field** (lines 1268-1330): nu_t reads "closes", so the section leaves the open
   questions. Move its text to @sec-limits-whole-boundary (line 695) as a measured effect: replace "What would settle
   it: ..." through "... (@sec-limits-whole-boundary)." with the result sentence ("With all four meshes added as named
   boundaries, two seeds against two without them, the far-field eddy-viscosity error falls from 0.031 / 0.028 to
   0.013 / 0.012, within 1.12-1.34 times Transolver's 0.0097 / 0.0107, and the velocity floor from 0.019 / 0.021 to
   0.015 / 0.015, still 1.72-1.74 times Transolver's; registered readings 'closes' and 'helps', @sec-nb-floor-drv"),
   keep the box, ground-offset, cost and geometry-seed sentences, and **remove the "In flight" callout** under it.
   - Remove its row from the open-questions table (line 908, "Does ISLA's far-field eddy-viscosity error ... (@sec-oq-far-field)").
   - Cost sentence: `fd_step` (from PREREG section 8) gives the smoke's 1.58x, which stays as the registered figure;
     add the full runs' 1.74-1.76x (F 5.25 s, C 3.01 s; RESULTS.md, Cost finding). `fd_hours` / `q_floor` give the
     planned 4 x 30 = 120 GPU-h; the job took 146.6 GPU-h (148.2 with smoke and check) after seven requeues.
   - Other @sec-oq-far-field references to repoint if the anchor goes (or keep the anchor on the moved text):
     09-drivaerml.qmd 290 and 962, 14-program-status.qmd 740, 19-capacity.qmd 627, 20-limits.qmd 742 and 1386,
     21-design.qmd 285 (18-notebook.qmd's dated entries stay as written).
4. **14-program-status.qmd, ledger**:
   - CAP-AUDIT row (line 685), FLOOR-DRV parenthetical: replace "its smoke measured ISLA's step 1.58 times longer with
     them, 5.27 against 3.33 s, so about 120 GPU-h on one 4-GPU node for about 30 h; submitted 2026-10-23, read about
     2026-10-24 evening" with "done 2026-10-26: nu_t far field closes, velocity helps, @sec-nb-floor-drv".
   - The FLOOR-DRV row exists (line 740, RUNNING); replace its status cell with:

     DONE 2026-10-26 (ISLA on one DrivAerML Volume case with the domain's four boundary meshes against the same run without them, two seeds each, CAP Part B's protocol: far-field eddy viscosity **closes** (e_R 0.0130 / 0.0119 against C's 0.0310 / 0.0281 and Transolver's 0.0097 / 0.0107), velocity **helps** (floor 0.0153 / 0.0149 against C's 0.0191 / 0.0207, Transolver 0.0088 / 0.0086); step 1.58 times as long in the smoke, 1.74-1.76 over the full runs; requeued seven times, 148.2 GPU-h; @sec-nb-floor-drv)

     and point its question cell's reference from @sec-oq-far-field to @sec-nb-floor-drv if item 3 removes the anchor.
     The other cells (question, origin "CAP-AUDIT; BOUNDARY-AUDIT", registration, data) stay.

## D. For project memory (proposed; the supervisor records)

- CAP-AUDIT's and FLOOR-DRV's wall-distance bands are in units of L_ref = 5 m (labels.py reads the sdf after
  NonDimensionalizeByMetadata): `[1, inf)` is beyond 5 m. The book already carries the erratum; scripts keep the
  band labels.
- drivaer_ml_volume_reference divides every coordinate by L_ref before CenterMesh. Any frame check against the raw
  files must fit a scale as well as a translation.
- ISLA 0.9.0's DrivAerML Volume step with the whole boundary: 1.58x in a 200-step smoke, 1.74-1.76x at steady state
  (C settles from 3.3 to 3.0 s after about 1,000 steps; F stays at 5.25 s). A step-time ratio read from a short smoke
  understates the steady-state ratio when the cheaper arm warms up. The added time is about 2.2 s per step.
