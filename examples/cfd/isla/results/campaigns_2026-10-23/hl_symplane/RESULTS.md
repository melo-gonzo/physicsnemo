# HL-SYMPLANE results (2026-10-24)

The branch is psharpe/isla-hl-symplane at bc4081e75, pushed and not merged. It merges cleanly with origin/psharpe/isla
142485ef0 (`git merge-tree`). Every job is in `JOBS.md`, and `out/sacct.txt` gives their accounting.

## The tolerance (src/symmetry.py `outline_tolerance`; record `tolerance`)

The cap is read from the first source STL in sorted order, `geo_LHC001_AoA_10/geo_LHC001_AoA_10.stl.pmsh`, in both
roots. It has 966 cells, and its key d3b5c68b is accepted. Its outline is one simple loop of 968 vertices. Distances are
measured in (x, z).

- **Storage precision.** u = spacing(float32(2583.33)) = 2.441e-4 in, and eps = sqrt(2) u = 3.453e-4 in. This bounds the
  change of a point-to-segment distance when the point and the segment's ends each move by half a spacing per coordinate.
- **Chord sagitta bound.** Between two cap vertices the section is taken as a convex arc whose end tangents lie within the
  end turnings, widened by their float32 uncertainty eps/L at each adjacent chord. The arc then lies in the tangent
  triangle, whose height is h_j = L_j / (cot b_start + cot b_end).
  - The largest bound is 0.134123 in, on the 1-in chord (2483.343, 725.646) to (2482.437, 725.224) near the flap. Its ends
    turn 11.8° and 20.5°, the same sign.
  - The 10 chords with an inflection use the same formula. Their bounds are far below that maximum.
  - A circle through three vertices would underestimate this chord's sagitta: it gives 0.035-0.045 in, against a measured
    CFD deviation of 0.061 in.
- **tau** = max h + eps = **0.1344687 in**.
- **The 0.011 in.** It is HL-DUP's distance from the broken roots' CFD vertices to LHC001's CFD outline. That compares two
  meshes about 10 times denser than the STL, and it is 0.08 tau. Against the STL outline the CFD vertices lie up to
  0.0613 in away, 0.46 tau. That is the 0.061 in Hausdorff distance HL-DUP found between the cap and curator's hole.

## Census (`out/census_summary.txt`; record `$T/hldup/symplane/hl_symmetry_check_medium.json`, md5 bc2ed653, read-only, MANIFEST)

| set | cases | pass | worst vertex | worst open-edge midpoint | on-plane vertices | open edges |
|---|---|---|---|---|---|---|
| -medium, every folder with a wing | 690 | 690 | 0.061264 in (0.456 tau), the same vertex on every case | 0.060126 in on 689; 0.126558 in (0.941 tau) on geo_LHC024_AoA_8 | 11,227-11,259 | 7,371-11,190 |
| canonical manifest (train 510 + val 180) | 690 (all in the record) | 690 | same | same | | |

- The cluster-only `highlift_surface_medium_dedup{,_posed}.yaml` read the same manifest on the same root.
- On LHC024_AoA_8 the 0.127 in comes from two root edges of about 20 in near the slat, from (124.3, 164.4) to
  (142.9, 155.5) (`out/edgeprobe_374259.log`). Their ends lie within 0.004 in of the outline, so the midpoint distance is
  the edges' own chord sagitta, not a different root.
- Cost: jobs 374043-374045, 3 × 8 cores × 0.26 h = 6.2 core-h. A case takes 30 s median and 54.5 s at most, with 8
  processes per node.

## Validation against curator's stored plane (`out/curval_374370.jsonl`, job 374370)

The plane comes from the recipe's own reader. On -medium it is built with the record; on the full root, with the exact
check at load. Both give the same plane bit for bit, and the plane is the same for every case.

| | ours | stored (LHC001_AoA_10; _12 has 1 point more and 1 cell fewer) |
|---|---|---|
| points / cells | 1,032 / 1,032 | 11,253 / 11,031 |
| area | 1,887,240,642.53 in² (+4.6e-9 relative) | 1,887,240,633.77 in² |
| hole (disk − plane) | 612,127.85 in² (0.9999857 of stored) | 612,136.62 in² |
| hole outline | 968 edges, every vertex of degree 2 | 10,935 edges, 11 vertices of degree 4 (curator's walk) |
| hole Hausdorff | 0.0613 in (stored→ours); 0.00085 in (ours→stored) | |
| equator | bitwise equal to the stored one (64 points) | |
| orientation | all 1,032 normals +y, into the fluid | all +y |

- The seam is checked on the built domain: inlet + outlet + symmetry, merged by coordinates, leave open exactly the 968
  hole edges, and no edge is used more than twice.
- Full and -medium copies give identical results.

## Moves with the body (`out/move_374330.jsonl`, job 374330)

One similarity maps the raw inlet, outlet and symmetry pieces and the wing onto the output. The check covers 8 samples:
highlift_surface with augment off and on, the dedup file, and the posed file (rotations 121° and 165°).

- Residuals are at most 1.5e-5 per piece and 2.6e-6 on the body, at an extent of 91-94.
- The test bound is 1e-5 × extent. BAD 0.

## Baseline identity (job 374371 against 373970)

GeoTransolver's and Transolver's collated forward_kwargs and targets were compared on the HiLift files (highlift_surface,
medium_dedup, medium_dedup_posed), 3 cases each, with augment off (and on for highlift_surface): **24 of 24 items are
bitwise equal**.

The cost per load:
- AddSymmetryPlane alone (the record lookup, two fingerprints and the vtkDelaunay2D plane) takes 12-13 ms.
- A reader read takes 414-514 ms with the plane and 408-429 ms without (noisy medians of 10).

## Tests

- Recipe tests on bc4081e75 (job 374461): 605 passed, 12 skipped and 2 failed. The failures are test_campaign_c_hooks,
  with the same E-lines as the base b4b10d5a9 (374311: 588 passed, 2 failed).
- The 3 real-data tests that the record agrees with the exact check ran: LHC001_AoA_10, and the broken roots
  LHC026_AoA_18 and LHC105_AoA_22.
- Nothing under physicsnemo/ or test/ changed, so the ISLA directory was not run.

## GB300 smoke (`gb/smoke_summary.txt`, job 374346, nvl72d019-T12, 0.27 GPU-h)

BC-SUPPLY's protocol: 20 steps, bfloat16, eager, geo_LHC002_AoA_18, 100,000 cells.

| arm | step median (ms) | peak mem (GB) | tokens |
|---|---|---|---|
| hl_body (base, wing alone) | 681.3 | 30.98 | 93,364 |
| hl_far (base; BC-SUPPLY's whole-boundary arm, which gave 32.23 GB and 97,268 tokens) | 682.0 | 32.23 | 97,268 |
| hl_far_hd (head; the plane built but not read) | 683.1 | 32.23 | 97,268 |
| **hl_sym (head, wing + inlet + outlet + symmetry)** | **682.3** | **32.51** | **98,300** (symmetry 1,032) |

- All losses are finite.
- hl_sym against hl_far: step 1.000×, memory 1.008×, tokens 1.0106×.
- As in BC-SUPPLY's run, the step time measures data loading; the minimum steps are 175-181 ms on every arm.
