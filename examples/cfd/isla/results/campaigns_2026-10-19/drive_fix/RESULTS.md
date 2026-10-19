# DRIVE-FIX results (scored 2026-10-19 11:43-11:52 EDT; reduced with reduce_drivefix.py -> drivefix_scores_2026-10-19.json)

All float32, final checkpoints (epoch 500). Seed values are 42 / 43; "mean" is the seed mean. The reduction is the
infer_summary row (all 18 files have one). AirfRANS "area" is identical to native.

## R1, AirfRANS AF-UNIFORM, baselines with the unit direction and the standardized speed (200 test airfoils)

| arm | native p | native v | native nut | posed p / v / nut |
|---|---|---|---|---|
| GeoTransolver afu_ud_gt | 0.01090 (0.01107, 0.01073) | 0.03998 (0.03973, 0.04022) | 0.01708 (0.01755, 0.01661) | 1.80 / 1.17 / 1.59 |
| Transolver afu_ud_t | 0.01090 (0.01068, 0.01112) | 0.02967 (0.02976, 0.02958) | 0.01782 (0.01767, 0.01796) | 1.27 / 1.23 / 1.85 |
| old raw-drive GeoTransolver | 0.0123 | 0.0365 | 0.0194 | |
| old raw-drive Transolver | 0.0135 | 0.0669 | 0.0212 | |
| ISLA A2 (afu_a2s) | 0.01799 | 0.00946 | 0.02230 | (equivariant) |

- New / old: GeoTransolver 0.886 / 1.095 / 0.880; Transolver 0.807 / 0.444 / 0.840.
- Baseline / ISLA A2, native: GeoTransolver 0.606 / 4.23 / 0.766; Transolver 0.606 / 3.14 / 0.799.
- Prediction check: the GeoTransolver velocity ratio was predicted to fall to 1.0-2.5 against the raw-read reference. It
  rose instead (0.0365 -> 0.0400), so that prediction failed. The Transolver prediction (1.2-4.0) holds at 3.14 against A2.
- Open check (AF-PLANAR): the baselines run AirfRANS embedded in 3-D. Their velocity_l2 includes the out-of-plane
  component (velocity_z_l2 is 20-280 relative to a near-zero truth norm). ISLA's 2-D set has no z component. A rough
  estimate from the per-component errors suggests z could account for about a quarter of the baselines' velocity error.

## R2, DrivAerML Surface CHAR, Transolver 512x16 with the unit direction (48 val cars, 100,000 cells)

| | native p | native wss | area p / wss | posed p / wss |
|---|---|---|---|---|
| r80ud_dr_t seed means | 0.04226 (0.04225, 0.04226) | 0.06760 (0.06741, 0.06779) | 0.345 / 0.422 | 1.29 / 1.35 |
| ratio to the reference 0.04368 / 0.06961 | 0.9674 | 0.9711 | | |

- Registered rule: both native seed means within [0.97, 1.03] x the reference. Pressure is at 0.967, outside the band;
  wall shear is inside. So the rule does not hold, and the reruns become the supervisor's decision.
- Against ISLA A2 (0.0445 / 0.0700, scored on 200k cells, a different scored set): 0.950 / 0.966.

## R3, lug CHAR, GeoTransolver with the boundary-condition labels in its geometry input (600 cases, whole part)

| | native von Mises | posed |
|---|---|---|
| lugbc_gt seed means | 0.00224 (0.00218, 0.00230) | 0.95 (0.937, 0.970) |
| old GeoTransolver without the labels | 0.00209 | |
| ISLA A2 | 0.001726 | |

- GeoTransolver / ISLA A2: 1.298 with the labels, against 1.212 without them. The prediction (0.95-1.35) holds.
