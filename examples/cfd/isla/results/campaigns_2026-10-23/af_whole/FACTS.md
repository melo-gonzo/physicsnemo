# AF-WHOLE pre-training facts (all 1,000 AirfRANS cases)

LHR cpu-short 361286 / 361287 (harness/facts.py) and 361288 (constants scan), 2026-10-23 20:44 EDT; summary by harness/summarize_facts.py. Lengths in chords. The table and its reading are PREREG.md section 3; the raw rows are out/facts_{a,b}.jsonl.

```
cases 1000 errors 0
n_airfoil: min 843 median 1017 max 1217
n_freestream: min 1563 median 1727 max 1923
chain_ok all True
free_into_fluid (fraction of cells): min 1 median 1 max 1
free_signed_area: min 1.428e+05 median 1.428e+05 max 1.428e+05
foil_signed_area: min -0.1365 median -0.08574 max -0.03414
foil geometric normals = stored: min 0.614 median 1 max 1
free_z [(0.0, 0.0)] interior_z [(0.0, 0.0)]
measure_airfoil: min 2.009 median 2.049 max 2.12
measure_freestream: min 1428 median 1428 max 1428
airfoil measure share: min 0.001405 median 0.001433 max 0.001482
airfoil token share: min 0.3504 median 0.3706 max 0.3876
tokens total: min 2406 median 2744 max 3140
L_airfoil (chords): min 0.4107 median 0.4206 max 0.4373
L_whole (chords): min 305.3 median 305.3 max 305.3
L_freestream (chords): min 305.5 median 305.5 max 305.5
L_whole / L_airfoil: min 698.1 median 725.8 max 743.5
1e-6 L_airfoil (chords): min 4.107e-07 median 4.206e-07 max 4.373e-07
1e-6 L_whole (chords): min 3.053e-04 median 3.053e-04 max 3.053e-04
d_min_pos (chords): min 1.092e-06 median 1.588e-06 max 1.717e-06
first-node height median (chords): min 1.906e-06 median 2.101e-06 max 2.112e-06
first-node height p01 (chords): min 1.097e-06 median 1.591e-06 max 1.720e-06
pooled first-node height sample: median 2.084e-06 p01 1.489e-06
first-node s / 1e-6 with L_airfoil: min 4.41 median 4.968 max 5.091
asinh at median first node, L_airfoil: min 2.19 median 2.306 max 2.33
linear-part fraction, all nodes, L_airfoil: min 0.005413 median 0.005631 max 0.005784
linear-part fraction, off-wall nodes, L_airfoil: min 0 median 0 max 0
fraction below 10 eps, L_airfoil: min 0.01292 median 0.01374 max 0.01523
first-node s / 1e-6 with L_whole: min 0.006244 median 0.006882 max 0.006919
asinh at median first node, L_whole: min 0.006244 median 0.006882 max 0.006919
linear-part fraction, all nodes, L_whole: min 0.1934 median 0.2017 max 0.2067
linear-part fraction, off-wall nodes, L_whole: min 0.189 median 0.1971 max 0.2021
fraction below 10 eps, L_whole: min 0.3633 median 0.3757 max 0.3851
n_wall_nodes / n_interior: min 0.005413 median 0.005631 max 0.005784
log10 Re airfoil: min 5.921 median 6.224 max 6.412
log10 Re whole: min 8.787 median 9.087 max 9.263
z shift = log10(L_whole / L_airfoil): min 2.844 median 2.861 max 2.871
z_LRE airfoil = log10(Re/1e6): min -0.079 median 0.2236 max 0.4122
z_LRE whole: min 2.787 median 3.087 max 3.263
frame_airfoil eigen gap (l1-l0)/l1: min 0.9392 median 0.9747 max 0.9958
frame_airfoil axis0 angle (deg): min 87.23 median 89.54 max 91.51
frame_airfoil |skew0|: min 6.031e-07 median 1.493e-01 max 4.380e-01
frame_airfoil |skew1|: min 6.875e-06 median 5.136e-03 max 1.709e-02
  frame_airfoil skew1 sign: + 33, - 967
frame_whole eigen gap (l1-l0)/l1: min 0.03303 median 0.03305 max 0.03309
frame_whole axis0 angle (deg): min -180 median -180 max 180
frame_whole |skew0|: min 1.844e-01 median 1.845e-01 max 1.846e-01
frame_whole |skew1|: min 5.388e-08 median 2.467e-05 max 7.834e-05
  frame_whole skew1 sign: + 724, - 276
frame_freestream eigen gap (l1-l0)/l1: min 0.03307 median 0.03309 max 0.03313
frame_freestream axis0 angle (deg): min -180 median -180 max 180
frame_freestream |skew0|: min 1.851e-01 median 1.851e-01 max 1.852e-01
frame_freestream |skew1|: min 9.276e-08 median 2.454e-05 max 7.855e-05
  frame_freestream skew1 sign: + 729, - 271
crop xmin: min -2.165 median -2.161 max -2.155
crop xmax: min 4.225 median 4.226 max 4.232
crop ymin: min -1.646 median -1.619 max -1.614
crop ymax: min 1.613 median 1.617 max 1.621
```

Refit ISLA constants with the far field (out/fits_afwhole*.json; committed as examples/cfd/isla/results/aerozoo_dataset/airfrans_whole_constants_fits.json):

| split | cases | reynolds mean | std | output_scale.nut |
|---|---|---|---|---|
| airfrans.full_train | 800 | 9.0652 | 0.13382 | 1.0450e-07 |
| airfrans.reynolds_train | 504 | 9.0838 | 0.06181 | 1.0380e-07 |
| airfrans.aoa_train | 804 | 9.0674 | 0.13213 | 1.0480e-07 |
| airfrans.full_train+full_test | 1000 | 9.0659 | 0.13326 | 1.0599e-07 |
| airfrans.full_train_s504 | 504 | 9.0660 | 0.13468 | 1.0664e-07 |
