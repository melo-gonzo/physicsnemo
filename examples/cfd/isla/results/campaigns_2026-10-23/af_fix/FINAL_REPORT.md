Status: DONE

AF-FIX's probes and design are done, and the campaign is drafted but not submitted. The files: `DESIGN.md` (tasks 1-2
and the decisions), `PRESSURE_PROBE.md` (task 3), `PREREG_DRAFT.md` (task 4, md5 553e4714), `JOBS.md`, and the
children's `tie/REPORT_TIE.md` and `prec/REPORT_PREC.md`. A lane-reviewer (`REVIEW_DESIGN.md`) confirmed the numbers and
the probe's arithmetic. It blocked on four wording problems, all fixed, with its 16 notes.

**What ran.**
- LHR GPU short: 381565 and 381714 (read-mass probe) and 381746 (child PREC), 3.10 of the 6 GPU-h.
- cpu-short: 381569, 381862, and 13 jobs of child TIE.
- New read-only snapshots: `$T/affix/code_dttpose_38a79` and `$T/affix/recipe_013`.
- No commits. The registry row was added, then removed at 08:33.

**1. The tie** (measured, 200 full_test cases, posed as in the scorer):
- With faces, **0 far-field cells change group, in every case**; octants move 15.4 / 28.9 / 45.1 % (min / median / max).
  On the airfoil, faces move 1 cell in 3 cases, octants a median of 2.
- Faces still leave 4.1 / 15.8 / 26.5 % of posed pairs with no native match, mostly the airfoil's trees, which sit
  28 chords from the forest origin.
- Proposed **rule 1**: face trees per declared condition, each about its own centroid. Its counterfactual median is
  0.026 %, and on a single-condition boundary it equals DTT-POSE bitwise. Also proposed: a derived skew bound in place
  of `CANONICAL_SKEW_TIE`.

**2. The near-wall length** (measured on 0.13.0):
- 0.13.0 forms x - p in float64, so every candidate passes the §1 gate (p99.9 under 2e-7).
- On AirfRANS (first-node t / pose change p99.9):

  | candidate | t | pose change | verdict |
  |---|---|---|---|
  | ii, a viscous length | 8.3 | 0.018 | **recommended** |
  | F1, stored-frame floor | 9.1 | 8.4 | fails pose |
  | F2, floor about the centroid | 0.077 | small | still linear |
  | M, per-condition gauges | - | AeroZoo 31-35 | open pieces flip sign |

- **ii is `distance_softening: {nu: 1, U: -1}`**, opt-in, in the `units` grammar, with no fluid name in model.py. It
  leaves DrivAerML, the lug, Tensile2d and the surface tasks bitwise unchanged by construction, and changes AeroZoo only
  if AeroZoo declares it. Table: `DESIGN.md` §2.

**3. The pressure harm** (8 trained runs, 16 test cases each):
- *Measured:* W-DTT's slices average 99.75 % far field, and its anchors lie about 60 chords from the points. Its tree
  read gives the airfoil 10-12 % of the attention even 1e-6 chord off the wall; B-DTT gives 100 %.
- *Measured:* in W-ISLA the airfoil holds 0.09 % of the slice mass.
- *Inferred:* that this causes the pressure harm at 0.01-0.1 chord. The across-case test has no power, and reweighting
  at inference is off-distribution.
- Proposed **C**: per-condition quadratures with learned shares. Omega (a solid-angle prior on the read) is the
  narrower alternative.

**4. The campaign** (ISLA-DTT on AirfRANS with its far field):
- Nested arms F (forest with rule 1), FN (+ ii) and FNC (+ C), seeds 42 and 43, as 0.9.0 ports. The control is
  AF-WHOLE's W-DTT.
- About **700 GPU-h**. The lanes end about 10-26 21:00 to 10-27 05:00 EDT, at the scope freeze, if six lanes run at once.
- An AeroZoo no-harm check (about 545 GPU-h, about 68 h of wall) slips past the freeze in any case.

**For the program.** The faces settle the far-field tie, and ii is ready, with no precision objection left. C has a
measured mechanism but no causal test. I am confident in the numbers.

**Rulings relied on:** the whole boundary (Peter, 10-23); no fluid names in model.py and interior = ISLA-DTT (Peter,
10-24); LANE_RULES. Nothing new for project memory.

**Decisions needed**
- From you:
  - the campaign and its cap;
  - rule 1, a change to DTT-POSE's frozen forest, or F0, the forest as it is;
  - that FN supersedes NEARWALL's contingency, whose arm was interacting W-ISLA (removed in 0.13.0);
  - whether to add FNΩ (+220 GPU-h), and whether to run AeroZoo in parallel.
- From Peter:
  - whether to adopt ii, and whether AeroZoo declares it;
  - whether to adopt C, which changes every dataset with more than one condition;
  - whether rule 1 groups by declared conditions or by connected components.

**Side notes.** TIE's posed-interior-offset finding is refuted in the scorer's path (381862: the posed wall nodes are at
d = 0). The docstring of `soft_log_length` claims a 0.12 bound, but AeroZoo reaches 0.16-0.17 through band flips.

**Still running:** nothing. **On disk:** `$T/affix/` on LHR (231 MB) and this folder (129 MB, mostly `prec/out`).
