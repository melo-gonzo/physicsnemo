# Notebook draft: AF-WHOLE read (for 18-notebook.qmd; the supervisor appends, renders and commits)

## 2026-10-25 — AF-WHOLE read: with the AirfRANS far field, ISLA reads a harm on every field, ISLA-DTT a harm and a broken pose {#sec-nb-af-whole-read}

AF-WHOLE trained ISLA 0.9.0 and ISLA-DTT on full AirfRANS with the far field read as the named condition `freestream`. The
far field comes from the raw release's `_freestream.vtp`, geometry only, through `VtkBoundaryDomainMeshReader`
(psharpe/isla-whole-boundary 8e93f2307). The arms are two seeds each, eager-trained like the standings, and the read is
against r090eag_af_isla / r090eag_af_dtt (@sec-nb-af-whole-prereg). All checks and the supervisor's far-field guard
passed. The input is Peter's ruling (2026-10-23); this measures its effect.

**Result.**

| pair | pressure | velocity | eddy viscosity | pose (posed / native) |
|---|---|---|---|---|
| ISLA, far field / standing | harms, 1.35 | harms, 1.20 | harms, 1.28 | holds, 1.0001-1.0014 |
| ISLA-DTT, far field / standing | harms, 1.94 | harms, 1.53 | harms, 1.89 | **broken**, 1.53-2.52 |

ISLA's pressure is 0.0164 / 0.0173 against 0.0132 / 0.0118. The far field beats the standing on 8-16 % of the test cases.

**Where.**
- ISLA's velocity and eddy-viscosity harm sits at the wall: 1.43 and 1.12 in the first decade off it (1e-6 to 1e-5
  chords), falling to 1.01 and 1.00 beyond 0.1 chord. It is larger below the softening's knee, 1e-6 L = 3.05e-4 chords,
  than above it on both seeds (velocity 1.24 against 1.12). The far field makes L = 305 chords and puts 20 % of the points
  in the channel's linear part, so the trained model does not recover the log resolution NEARWALL found lost at fixed
  weights (@sec-oq-nearwall).
- ISLA's pressure harm does not sit at the wall: 1.01-1.07 there, 1.28 at 0.01-0.1 chord. One untested candidate: the far
  field holds 99.86 % of the boundary measure that ISLA's slices are weighted by.
- ISLA-DTT's harm is spread over every band (1.3-1.9).

**ISLA-DTT's pose: the octant tie, now on AirfRANS.**
- The far field's straight edges have normals on the dataset's axes, and so on the forest's frame, which the far field
  sets (eigenvalue gap 3.3 %) and the airfoil tilts by about 1e-6.
- Their small normal component is about 2e-6 (down to 1.7e-8), so the sign test that picks a cell's octant tree reads
  that tilt and float32 rounding.
- Posed against native, 16-45 % of the far-field cells (median 29 %, all 200 test cases) change octant, against a
  median of 2 airfoil cells.
- It is the lug's tie (05-forward-pass.qmd, the octant paragraph) at a larger scale. The frame itself does not tie: 0 sign
  flips of either axis on the 200 cases.

Cost: 397.5 of 520 GPU-h. Per step the far field costs nothing on AirfRANS (0.98x ISLA, 1.01x ISLA-DTT at the smoke).

**What it leaves open, for the supervisor and Peter.**
- NEARWALL's contingency trigger is met (a harm on velocity and eddy viscosity near the wall). The held viscous softening
  asinh(d U / nu), trained with only the channel changed and two seeds, is the registered next test.
- ISLA-DTT on AirfRANS with its far field needs an octant rule without exact ties, as on the lug, before its accuracy
  can be read apart from the tie.
- Whether the far-field runs replace the AirfRANS standings is the supervisor's call. Canonical ISLA reads the whole
  boundary, so its AirfRANS numbers are these.

Data: scratch/isla-af-whole-2026-10-23/RESULTS.md (sections 1-3), read.md, out/octant_pose_summary.txt,
out/frame_pose_summary.txt; jobs in JOBS.md.

## Chapter changes this result makes

Each row gives the place, the current text, the new text and its source.

1. **20-limits.qmd, `#sec-oq-nearwall`.**
   - Remove the In-flight callout ("ISLA trained on AirfRANS with its far field, read near the wall, is being measured").
   - In the paragraph, replace "Cost: the retrain is in flight; the viscous arm about ... GPU-hours" with: "The retrain
     read a harm near the wall (AF-WHOLE): velocity 1.43 and eddy viscosity 1.12 times the standing's error in the first
     decade off the airfoil, 1.24 and 1.15 below $10^{-6} L$ against 1.12 and 1.02 above it, on both seeds; so the
     viscous softening is the next test; it costs about ... GPU-hours."
   - Source: RESULTS.md section 2, scores/nearwall_isla.json.
2. **05-forward-pass.qmd, the distance channel paragraph (about line 1346).** Replace "whether a model trained with the
   whole boundary recovers it is open (@sec-oq-nearwall; ...)" with: "a model trained with the whole boundary does not
   recover it: its velocity error in the first decade off the wall is 1.43 times the standing's (@sec-oq-nearwall;
   AF-WHOLE RESULTS.md section 2)".
3. **05-forward-pass.qmd, the octant paragraph (about line 2040).**
   - Current: "On the AirfRANS interior task no normal has a component below $10^{-5}$, 0 to 8 cells change octant, and
     ...". This holds for the airfoil alone.
   - Add: "With its far field it does not: the far field's straight edges lie along the forest's axes, their small
     component is about $2 \times 10^{-6}$, posed against native 16 to 45% of the far-field cells (median 29%) change
     octant, and ISLA-DTT trained with the far field has a posed error 1.53 to 2.52 times its native error."
   - Source: RESULTS.md section 3, out/octant_pose.jsonl.
4. **20-limits.qmd.** Add to the ISLA-DTT pose limitation (where the lug's tie is listed) that AirfRANS with its far field
   shows the same tie at 1.5-2.5x, with the same source. Its fix and its test are the supervisor's.
5. **14-program-status.qmd, the AF-WHOLE row.**
   - Status: "DONE 2026-10-25 (397.5 of 520 GPU-h)".
   - Result: "with the far field ISLA harms pressure 1.35, velocity 1.20, eddy viscosity 1.28, pose holds; ISLA-DTT
     harms 1.94 / 1.53 / 1.89 and its pose breaks (1.5-2.5x) through the octant tie on the far field's straight edges;
     @sec-nb-af-whole-read".
6. **The AirfRANS standing values** (r090eag_af_isla, r090eag_af_dtt; chapters 03, 04, 05, 06 and the ledger read them).
   No change until the supervisor decides whether the far-field runs become the standings. If they do, the new values
   are RESULTS.md section 1. ISLA-DTT's should then carry an In-flight callout until its octant tie is fixed and
   retrained.
7. **06-using-isla.qmd, line 698** ("the AirfRANS one the airfoil and the far field `freestream`"): no change. The
   configuration is commit 8e93f2307, merged to psharpe/isla-whole-boundary at 4b772ec59 with DRV-GROUND.
