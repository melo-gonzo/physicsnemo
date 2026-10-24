
## pressure_l2
- ref: 20k 0.001981 / 0.001986; 40k 0.0009643 / 0.0006194 (complete True, True); k_ref (s44) 0.487, seed mean 0.399
- w1024: k_top 0.476 (40k complete True, last 40000); k_ref / k_top 1.022 -> every ISLA needs more steps on this case; the 20k ladder measured budget, not capacity
- d24: k_top 0.332 (40k complete True, last 40000); k_ref / k_top 1.466 -> the larger rungs need more steps; capacity shows only at the larger budget
- s2048: k_top 0.379 (40k complete True, last 40000); k_ref / k_top 1.283 -> the larger rungs need more steps; capacity shows only at the larger budget
- baselines 40k (one seed): GeoTransolver 0.0003903 (complete True), Transolver 0.0004203 (complete True)
  - ISLA ref_40k_s44: 0.0009643; / GT 2.47, / T 2.29; DEFICIT (>= 2x best) (complete True)
  - ISLA ref_40k_s45: 0.0006194; / GT 1.59, / T 1.47; under 2x (complete True)
  - ISLA w1024_40k_s44: 0.0008016; / GT 2.05, / T 1.91; DEFICIT (>= 2x best) (complete True)
  - ISLA d24_40k_s44: 0.0006367; / GT 1.63, / T 1.51; under 2x (complete True)
  - ISLA s2048_40k_s44: 0.0007085; / GT 1.82, / T 1.69; under 2x (complete True)

## velocity_l2
- ref: 20k 0.001206 / 0.001217; 40k 0.0007998 / 0.0006464 (complete True, True); k_ref (s44) 0.663, seed mean 0.597
- w1024: k_top 0.645 (40k complete True, last 40000); k_ref / k_top 1.028 -> every ISLA needs more steps on this case; the 20k ladder measured budget, not capacity
- d24: k_top 0.542 (40k complete True, last 40000); k_ref / k_top 1.222 -> the larger rungs need more steps; capacity shows only at the larger budget
- s2048: k_top 0.510 (40k complete True, last 40000); k_ref / k_top 1.300 -> the larger rungs need more steps; capacity shows only at the larger budget
- baselines 40k (one seed): GeoTransolver 0.0004601 (complete True), Transolver 0.0003943 (complete True)
  - ISLA ref_40k_s44: 0.0007998; / GT 1.74, / T 2.03; DEFICIT (>= 2x best) (complete True)
  - ISLA ref_40k_s45: 0.0006464; / GT 1.40, / T 1.64; under 2x (complete True)
  - ISLA w1024_40k_s44: 0.0007522; / GT 1.63, / T 1.91; under 2x (complete True)
  - ISLA d24_40k_s44: 0.0006467; / GT 1.41, / T 1.64; under 2x (complete True)
  - ISLA s2048_40k_s44: 0.0006101; / GT 1.33, / T 1.55; under 2x (complete True)

## nut_l2
- ref: 20k 0.001738 / 0.001287; 40k 0.001131 / 0.0008346 (complete True, True); k_ref (s44) 0.651, seed mean 0.649
- w1024: k_top 0.703 (40k complete True, last 40000); k_ref / k_top 0.926 -> every ISLA needs more steps on this case; the 20k ladder measured budget, not capacity
- d24: k_top 0.660 (40k complete True, last 40000); k_ref / k_top 0.985 -> every ISLA needs more steps on this case; the 20k ladder measured budget, not capacity
- s2048: k_top 0.607 (40k complete True, last 40000); k_ref / k_top 1.073 -> every ISLA needs more steps on this case; the 20k ladder measured budget, not capacity
- baselines 40k (one seed): GeoTransolver 0.0006652 (complete True), Transolver 0.0004648 (complete True)
  - ISLA ref_40k_s44: 0.001131; / GT 1.70, / T 2.43; DEFICIT (>= 2x best) (complete True)
  - ISLA ref_40k_s45: 0.0008346; / GT 1.25, / T 1.80; under 2x (complete True)
  - ISLA w1024_40k_s44: 0.001147; / GT 1.72, / T 2.47; DEFICIT (>= 2x best) (complete True)
  - ISLA d24_40k_s44: 0.0008354; / GT 1.26, / T 1.80; under 2x (complete True)
  - ISLA s2048_40k_s44: 0.000929; / GT 1.40, / T 2.00; under 2x (complete True)
