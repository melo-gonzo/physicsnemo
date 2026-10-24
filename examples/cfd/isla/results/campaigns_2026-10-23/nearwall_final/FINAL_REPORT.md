Status: DONE

NEARWALL (research-lane, supervisor 01a13c2c), 2026-10-24 07:40-09:16 EDT. Recommendation: keep the current
encoding, eps = 1e-6 L (approved by the supervisor at 08:43), and correct its documentation. The viscous softening is
drafted as a contingency PREREG and was not submitted.

**What ran (LHR).**
- GB300 short QoS: 367460, 367491, 367523 (exposure and the trained-layer response), 367615 and 367749 (precision, v1
  then v2), 367649 (DrivAerML's wall-only gauge). 1.0 GPU-h of the 6 GPU-h cap.
- cpu-short: 367917 (the whole ISLA test directory).
- Commit `9af2f110511474085985a3b2534b2c6413d7b364` on psharpe/isla-nearwall (off 1b25eb899, signed off, hooks passed,
  pushed, not merged; `git merge-tree` onto the current psharpe/isla is clean).
- Details: `JOBS.md`.

**Results** (`RESULTS.md`, `exposure_tables.md`, `response_tables.md`, `precision_tables_v2.md`, `out/`):
- **Exposure** (t = |d| / (1e-6 L); the channel is linear for t < 1; off-wall shares):
  - AirfRANS with its far field: L 0.42 to 305.3 chords, first nodes at t 3.8 to 0.0052, linear part 0 to 19.5 %.
    The softening sits at U d / nu of 610-1,830 (sublayer, buffer and start of the log layer).
  - DrivAerML Volume: L 2.1 to 59.5 m, first nodes at t 33 to 0.80, linear part 0 to 0.57 %.
  - AeroZoo (already whole): 0.01 %, on nodes 1e-9 to 1e-8 m off a wall, which are linear under the body's L too.
  - Lug and Tensile2d: none (first off-wall t of 2.6e4 and 7e3).
  - No 0.9.0 standing trained with a material share in the linear part (AeroZoo's 1e-4 was reported at 08:10).
- **Trained 0.9.0 input layers** (AirfRANS seeds 42 and 43, DrivAerML seed 42, AeroZoo seed 42):
  - They code log d linearly: |da| per decade of d is about 15 on AirfRANS and 17 on DrivAerML, flat over five
    decades.
  - With the whole-boundary L and the same weights, a factor 10 in d at AirfRANS's first node moves the input layer
    38x less, and the embedding's sensitivity per unit ln d falls 200x. That node is then resolved to about a factor
    2 in bfloat16.
  - DrivAerML is nearly unaffected (16.9 against 17.0).
  - This is resolution at fixed weights, not accuracy; only a retrain prices it.
- **Precision** (v2): rounding moves the current channel by at most 0.12 with the body anywhere in its domain.
  - Encodings that give AirfRANS's first nodes a log code (nu / U, a wall-only gauge, 1e-9 L) stay at or below 0.03
    in the scored frames, but reach 3-4 at 30 chords from the origin.
  - Most of that is the kernel's absolute-coordinate closest point (`GEOMETRY_NOTE.md`); the input's own storage
    rounding is about 0.44 p99.9.
- **Design** (`DESIGN.md` md5 1ff494b6, review `REVIEW_DESIGN.md`: sound with fixes, all applied): candidates (i)-(viii)
  judged on invariances, boundary-only input and forbidden constructs.
  - (i) is the only one defined identically on every dataset.
  - (ii) is undefined on DrivAerML, the lug and Tensile2d, and changes every AeroZoo case.
  - (iv) needs a wall declaration and barely moves DrivAerML (t 0.80 to 1.23).
  - (viii) breaks discretization invariance.
- **Docstring commit**: comments and docstrings only. ast.dump with docstrings stripped is identical for geometry.py
  and model.py. ISLA tests: 367 passed, 100 skipped (CUDA), 0 failed (`tests_isla_367917.log`).
- **Contingency**: `PREREG_CONTINGENCY.md` md5 26856eb9. Two codes chosen by what the model declares. Stage 1 is
  AirfRANS with its far field vs W-ISLA, 185-230 GPU-h; stage 2 is AeroZoo `full`, about 250 GPU-h. DrivAerML, lug
  and Tensile2d arms are degenerate. Predictions are included.

**Meaning and confidence.** The current encoding stands on principle: it is defined the same everywhere and is
robust to rounding wherever the body sits. Whether the lost near-wall log resolution costs AirfRANS accuracy is open; only AF-WHOLE prices it.
My predictions, made before any score:
- W-ISLA harms velocity or nut: 0.40;
- given a harm, it sits mainly below 3.05e-4 chords: 0.45;
- FLOOR-DRV shows a near-wall harm attributable to the encoding: 0.10.
The probe numbers are solid; the reading of resolution against accuracy is moderate.

**Rulings relied on.**
- Peter's whole-boundary ruling (2026-10-23 17:08) and his boundary-only input ruling (2026-10-17).
- No supplied viscosity: DrivAerML's nu = 1.0 was flagged and not used.
- The supervisor's approval at 08:43 (keep (i); draft the PREREG; the geometry note goes to the geometry lane).

**For project memory (suggested).**
- Under the whole boundary, ISLA's near-wall channel is linear below 1e-6 L: 19.5 % of AirfRANS points. Any change to it
  goes through PREREG_CONTINGENCY.md and Peter.
- ISLA's float32 wall distance degrades about ulp(T) when the body sits far from the origin (GEOMETRY_NOTE.md).

**Left and running.**
- Nothing is running.
- Open: the trigger read on AF-WHOLE (about 10-25 morning), the supervisor's merge of psharpe/isla-nearwall, and the
  geometry lane's check.
- On disk:
  - worktree `/home/psharpe/gh/physicsnemo-isla-nearwall` (1.1 GiB, pushed; remove it after merge or drop);
  - this folder (0.9 MB);
  - on LHR: `$T/nearwall` (69 MB) and the read-only snapshots `code_nearwall`, `recipe_nearwall`, `code_nearwall_doc`
    (78 MB).
