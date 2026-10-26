# AeroZoo v1.2: dataset metadata

These files describe the release. They hold no model result, and none of them is a model
input.

| file | what it holds | provenance |
| --- | --- | --- |
| `case_labels.json` | per case of the 3,816 released cases: the convergence census (`quality`), cell count, enclosure family, scenario and profile, body layout, geometry bank, target Reynolds number; `_meta`: counts per label | `case_labels.py`, run 2026-10-10 on the release's `index.json` and every case's `case-manifest.json`; moved here from `results/campaigns_2026-10-09/aerozoo/` |
| `case_labels.py` | the script that wrote it | moved with it |
| `target_scan.json` | the field scan over every interior row of the 3,815 cases of the `full` split: case counts per side (`scan`: train 2,713, validation 386, test 716), per-field counts of cases above the thresholds, the cases flagged, and the count of cases whose largest standardized ω exceeds 1,000 and 10,000 | `target_scan_extract.py`, run 2026-10-19 on `probe/outliers.json` and `probe/outliers_scan.jsonl` of `results/campaigns_2026-10-09/aerozoo/` before that tree was deleted; it keeps the dataset statistics and drops the blocks that read model runs |
| `target_scan_extract.py` | that extraction | |
| `isla_constants_fits.json` | ISLA's own dataset constants per split, fitted on the train parts: `reynolds` (mean and population std of log10(U L / nu)) and each output's `scale` (the median over the train cases of the per-case RMS of the target made dimensionless by ISLA's scales), for the seven AeroZoo splits and AirfRANS full_train; the values in the dataset files' `isla_constants` | `isla_constants_fit.py` (with `isla_constants_fit.sbatch`), LHR cpu-short jobs 295396-295399, 2026-10-19, on all 3,656 AeroZoo train cases and 800 AirfRANS cases, 0 errors |
| `isla_constants_fit.py`, `isla_constants_fit.sbatch` | that fit, through the model's own `_case_scales` | |
| `aerozoo_seven_field_constants_fits.json` | per split, fitted on its train part, the constants of the interior task's two added targets, `intermittency` and `retheta` (`solution.momentum_thickness_reynolds_number`): their `NormalizeMeshFields` std beside the five other fields' (refit, equal to the dataset files at 6 digits) and ISLA's `output_scale`; the values in `datasets/aerozoo_uvolume*.yaml` | `aerozoo_pstats.py targets` + `reduce` (LHR cpu-short job 397199) and `isla_constants_fit.py aerozoo_rms` + `aerozoo_rms_fit` (job 397200), 2026-10-26, every train case of the seven splits, 0 errors |
| `airfrans_extrap_constants_fits.json` | AirfRANS per split, fitted on each train side only: isla_constants, the nut `NormalizeMeshFields` stats (`nut_norm`: mean over cases of the per-case mean and std over every interior point) and the freestream-speed stats (`U_inf_mag`), for full_train (reproduces the full-split values), reynolds_train, aoa_train, and all 1,000 cases (`full_train+full_test`, a check only); the values in `airfrans_uvolume_{2d,ud}_{reynolds,aoa}*.yaml` | `isla_constants_fit.py airfrans` + `airfrans_fit` (AF-EXTRAP), LHR cpu-short job 309151, 2026-10-21, code_r090, 0 errors |

The scan makes each field dimensionless by the case's reference scales and standardizes
it by the `full` training side's constants. That is an analysis of the release that
reads its labels; the benchmark protocol (`../campaigns_2026-10-19/aerozoo_protocol/`)
gives no model a label.
