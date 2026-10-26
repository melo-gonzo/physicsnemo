# SLICE-NORMAL-EXP results (in progress)

PREREG.md md5 4580615d (Parts A-F). Jobs and commands: JOBS.md. M = measured, E = estimated.

## 1. Smokes (all pass their registered items; M)

| arm | job | step control / arm (s) | ratio | peak alloc | m_s check (20 cases x 12 blocks) | notes |
|---|---|---|---|---|---|---|
| DR-A (a) | 374067 | 0.494 / 0.449 | 0.91 | 84.8 / 84.8 GiB | 240/240 bitwise, bf16 and fp32 | 'fused' everywhere |
| AF (a) | 374068 | 0.958 / 0.948 | 0.99 | 48.7-48.8 both | 240/240 | AF-WHOLE guard OK |
| LUG (a) | 374069 | 0.447 / 0.451 | 1.01 | 9.2-10.1 both | 240/240 | C4 fails for arm and control alike (§2) |
| DR-A-S | 374632 | 0.471 / 1.547 | **3.28** | | 240/240 | eager region forced (F3), against fused control |
| AF-S | 374633 | 0.954 / 0.983 | 1.03 | | 240/240 | guard OK |
| LUG-S | 374635 | 0.422 / 0.428 | 1.01 | | 240/240 | C4 reported (§2) |

Price of losing the fused kernel if (s) is adopted without a fused implementation (F3.2): DR-A surface step 3.28x
(0.471 -> 1.547 s at 205,054 tokens, eager region against the fused kernel, same node), about 33 -> 100 GPU-h per
DrivAerML Surface lane (E, from the (a) lanes' epoch rate).

## 2. Block-wise certification on the lug (C4; F1 ruling about 20:21 EDT)

`grad_geo_logit.bias` is excluded: softmax shift invariance makes it identically zero, so its relative distance is
roundoff over roundoff. Groups per run: 2 cases x 12 blocks x 25 groups x 2 precisions = 1,200. Fail = d(comp, f64) >
d(eager, f64) + allow, or fp64 compiled vs eager > 1e-12. Tables: `smoke/lug/cert_compare.txt`, `smoke/lugs/`.

| run | code | checkpoint | rule fails fp32 + bf16 | fp64 > 1e-12 | all failing groups | max excess fp32 / bf16 | max fp64 rel |
|---|---|---|---|---|---|---|---|
| (a) arm_cert (374069) | (a) | (a) 2-epoch | 43 + 12 = 55 | 18 | 66 | 5.6 / 11.7 | 1.5e-11 |
| (a) arm_ctlckpt (374069) | (a) | r090fix_lug_dtt_seed42 | 44 + 18 = 62 | 12 | 70 | 3.4 / 8.4 | 8.7e-12 |
| control (374069) | standing (code_cmpaf_fix) | r090fix_lug_dtt_seed42 | 55 + 11 = 66 | 33 | 86 | 3.5 / 3.4 | 4.1e-11 |
| (a)+(s) arm_cert (374635) | (a)+(s) | (a)+(s) 2-epoch | 11 + 3 = 14 | 0 | 14 | 3.8 / 3.7 | 4.3e-13 |
| (a)+(s) arm_ctlckpt (374635) | (a)+(s) | r090fix_lug_dtt_seed42 | 15 + 8 = 23 | 1 | 24 | 3.0 / 7.7 | 1.2e-12 |
| control (374635) | standing | r090fix_lug_dtt_seed42 | 56 + 10 = 66 | 33 | 86 | 3.0 / 3.5 | 4.1e-11 |

- **C4 fails for the standing's compiled region and the arm alike** (control 66 of 1,200 groups, arm 62 of 1,200, in the
  routing-path gradients; fp64 compiled against eager 1e-11 to 4e-11 relative), **so the rule cannot certify
  compile_blocks on the lug; the failure is not attributable to (a)** (supervisor's ruling, F1). The lug arms train
  compiled like the control.
- (a)+(s) certifies far better: 14-24 failing groups and fp64 at roundoff, against 66-70 for (a) and 86 for the
  standing, with fp64 up to 4e-11. The control's counts repeat in the two jobs (66 rule, 33 fp64, 86 in all), though the per-group excesses differ slightly.
- For the freeze: the lug's compiled training path fails the restated rule on the standing's own code; unless the final
  code certifies, the lug trains eager in the final rerun.

## 3. DR-A-S execution confound (F3.1; `smoke/dras/geo_exec_summary.txt`)

(a) code, latest checkpoint of slicenormalx_dra_m_seed42 at smoke time, 2 val cases x 12 blocks, geo_kernel 'eager'
against 'fused' on the same inputs; grad_geo_logit.bias excluded.

| precision | group | d(eager, fused) median / max | d(fused, f64) median / max |
|---|---|---|---|
| fp32 | output | 2.2e-7 / 2.3e-6 | 2.5e-6 / 7.2e-5 |
| fp32 | routing grads | 7.4e-7 / 2.0e-4 | 1.3e-5 / 1.1e-3 |
| fp32 | other grads | 4.0e-7 / 1.5e-6 | 4.0e-6 / 5.6e-5 |
| bf16 | output | 5.1e-3 / 9.7e-3 | 5.9e-3 / 1.3e-2 |
| bf16 | routing grads | 4.5e-3 / 1.9 | 1.5e-2 / 8.9 |
| bf16 | other grads | 2.5e-3 / 7.4e-3 | 4.1e-3 / 1.1e-2 |

Whole model RMS(eager - fused) / RMS(fused): fp32 8e-6 (pressure) to 1.2e-5 (wss); bf16 0.9-1.0 % (pressure), 1.8-1.9 %
(wss). The execution difference is at each precision's rounding level, the size of either path's own error.

## 4. Reads

### 4.1 DR-A, option (a) against the option-A control (read 03:52 EDT 10-25; `reading_DRA.md`, `reading_DRA.json`)

Scorer 379296 (eager, native and posed, 48 val cases, 200,000 cells); checks before reading pass (epoch 500, 48
infer_step rows with the control's sample ids, summary = case mean, START line names the row's code and dataset).
`read_slicenormalx.py` md5 711793fc.

| protocol | field | control s42, s43 | arm s42, s43 | arm ÷ control | D | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|
| native | pressure | 0.054298, 0.050429 | 0.064113, 0.058077 | **1.167** | +0.154 | 0.123 | 1.181, 1.152 | **harms** | up |
| native | wall shear | 0.082253, 0.078253 | 0.092831, 0.087052 | **1.121** | +0.114 | 0.081 | 1.129, 1.112 | **harms** | up |
| posed | pressure | | | 1.167 | | | | harms | up |
| posed | wall shear | | | 1.121 | | | | harms | up |

- The arm is worse on **every one of the 48 cases** (seed means), both fields; median case ratio 1.149 / 1.121.
- posed ÷ native is 1.0000-1.0001 for both arms (the posed protocol changes the input but not the error).
- **Under A3/A5 a "harms" blocks: option (a) is not a candidate for adoption** whatever AF and LUG read. A §5's fallback
  is (c), the smooth floor at bf16's rounding level (SLICE-NORMAL RESULTS §4); it is not tested here.
- Beside, not a registered comparison: the option-A control (eager-trained, tunnel read) scores 0.0524 / 0.0803 against
  the vehicle-only, compiled-trained 0.9.0 standing r090_dr_isla's 0.0412 / 0.0662 (REF-090); code, training path and
  boundary all differ, so this is not a measure of option A.

### 4.2 DR-A-S, (a)+(s), both ways (read 07:23 EDT 10-26; `reading_DRAS.md`, `reading_MDRA.md`)

Scorer 401974 (eager, native and posed, 48 cases); checks before reading pass. DR-A-S runs the eager geometry region
against fused controls and a fused (a) arm (PREREG F3); the measured execution difference is at the rounding level
(§3: whole model 1-2 % in bf16, 1e-5 in fp32), far below the effects below.

| reading | field | reference s42, s43 | (a)+(s) s42, s43 | ratio | D | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|
| registered: against the control | pressure | 0.054298, 0.050429 | 0.075845, 0.073091 | **1.422** | +0.352 | 0.083 | 1.397, 1.449 | **harms** | up |
| registered: against the control | wall shear | 0.082253, 0.078253 | 0.110592, 0.097596 | **1.297** | +0.260 | 0.135 | 1.345, 1.247 | **harms** | up |
| A6 marginal (beside): against (a) | pressure | 0.064113, 0.058077 | 0.075845, 0.073091 | **1.219** | +0.198 | 0.106 | 1.183, 1.259 | harms | up |
| A6 marginal (beside): against (a) | wall shear | 0.092831, 0.087052 | 0.110592, 0.097596 | **1.157** | +0.146 | 0.141 | 1.191, 1.121 | harms | up |

- (a)+(s) is worse than the control and than (a) alone on every one of the 48 cases, both fields (case medians 1.396 /
  1.293 against the control, 1.216 / 1.154 against (a)). Posed equals native to 1e-4.
- **Under A6 a "harms" blocks: (a)+(s) is not a candidate.** The marginal reading says (s) itself costs DR-A about 22 %
  on pressure and 16 % on wall shear on top of (a), beyond the execution confound's measured bound.

### 4.3 AF, option (a) against AF-WHOLE's afwhole_af_dtt (read 10:04 EDT 10-26; `reading_AF.md`)

Scorer 404046 (AF-WHOLE's scorer, eager, native / area / posed, 200 full_test cases); checks before reading pass, and
AF-WHOLE's far-field guard holds on every arm row (freestream in boundary_meshes and boundary_schemas, recipe_afwhole).
AF-WHOLE's control scores were opened only after its read was in (its RESULTS.md, 07:17 EDT 10-25; A2).

| field | control s42, s43 | arm s42, s43 | ratio | D | s_c, s_a | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|
| pressure | 0.028574, 0.019643 | 0.024797, 0.027103 | 1.076 | +0.074 | 0.265, 0.063 | 0.385 | 0.868, 1.380 | no change | none |
| velocity (in-plane) | 0.012181, 0.009786 | 0.011832, 0.013496 | 1.153 | +0.142 | 0.155, 0.093 | 0.255 | 0.971, 1.379 | no change | none |
| nut | 0.046560, 0.031442 | 0.042130, 0.054102 | 1.234 | +0.210 | 0.278, 0.177 | 0.466 | 0.905, 1.721 | no change | none |

- Registered labels: **no change on all three fields.** The control's own seeds sit 31-39 % apart (s_c 0.15-0.28;
  AF-WHOLE traced W-DTT's spread and its broken pose to a tie in the tree read's octant split), so 2 SE is 0.26-0.47 and
  the rule resolves only a large harm. The point estimates are 8-23 % worse; case medians 1.144 / 1.136 / 1.335; the arm
  beats the control on 26 / 16 / 24 % of cases. Read with low resolution, not as evidence of no effect.
- Posed (beside): 0.92 / 0.93 / 0.91, no change. posed ÷ native is 1.53-1.89 for the arm against 1.53-2.52 for the
  control (the octant-tie pose defect of this code, shared by both). Area differs from native in the sixth digit for both
  arms (AF-WHOLE: equal to 6e-8): degenerate, no claim.
- The (a) verdict stands from DR-A (§4.1): not a candidate.

### 4.4 AF-S, (a)+(s), both ways (read 10:54 EDT 10-26; `reading_AFS.md`, `reading_MAF.md`)

Scorer 404777 (as AF); checks and the far-field guard pass on every row.

| reading | field | reference s42, s43 | (a)+(s) s42, s43 | ratio | D | 2 SE | seed-matched | label | flag |
|---|---|---|---|---|---|---|---|---|---|
| registered: against the control | pressure | 0.028574, 0.019643 | 0.030579, 0.031812 | 1.294 | +0.258 | 0.377 | 1.070, 1.620 | no change | **up** |
| registered: against the control | velocity | 0.012181, 0.009786 | 0.013178, 0.013097 | 1.196 | +0.179 | 0.219 | 1.082, 1.338 | no change | **up** |
| registered: against the control | nut | 0.046560, 0.031442 | 0.061798, 0.051833 | 1.457 | +0.376 | 0.430 | 1.327, 1.649 | no change | **up** |
| A6 marginal: against (a) | pressure | 0.024797, 0.027103 | 0.030579, 0.031812 | **1.202** | +0.184 | 0.097 | 1.233, 1.174 | **harms** | up |
| A6 marginal: against (a) | velocity | 0.011832, 0.013496 | 0.013178, 0.013097 | 1.037 | +0.037 | 0.132 | 1.114, 0.970 | no change | none |
| A6 marginal: against (a) | nut | 0.042130, 0.054102 | 0.061798, 0.051833 | 1.181 | +0.166 | 0.306 | 1.467, 0.958 | no change | none |

- Registered: no field reads "harms", but **every field flags up** (both (a)+(s) seeds above both control seeds); the
  case medians are 1.334 / 1.143 / 1.320 and (a)+(s) beats the control on 12-19 % of cases. Posed: no change, no flag.
- Marginal (beside): (s) on top of (a) costs pressure 20 % (harms-sized, both seeds about 1.2, case median 1.166); on
  velocity and nut the seeds disagree (1.11 / 0.97, 1.47 / 0.96) and the case medians are 1.006 / 0.989.
- The (a)+(s) verdict stands from DR-A-S (§4.2): not a candidate.
