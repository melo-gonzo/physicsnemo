# AeroZoo v1.2: dataset metadata

These files describe the release. They hold no model result, and none of them is a model input.

| file | what it holds | provenance |
|---|---|---|
| `case_labels.json` | per case of the 3,816 released cases: the convergence census (`quality`), cell count, enclosure family, scenario and profile, body layout, geometry bank, target Reynolds number; `_meta`: counts per label | `case_labels.py`, run 2026-10-10 on the release's `index.json` and every case's `case-manifest.json`; moved here from `results/campaigns_2026-10-09/aerozoo/` |
| `case_labels.py` | the script that wrote it | moved with it |
| `target_scan.json` | the field scan over every interior row of the 3,815 cases of the `full` split: case counts per side (`scan`: train 2,713, validation 386, test 716), per-field counts of cases above the thresholds, the cases flagged, and the count of cases whose largest standardized ω exceeds 1,000 and 10,000 | `target_scan_extract.py`, run 2026-10-19 on `probe/outliers.json` and `probe/outliers_scan.jsonl` of `results/campaigns_2026-10-09/aerozoo/` before that tree was deleted; it keeps the dataset statistics and drops the blocks that read model runs |
| `target_scan_extract.py` | that extraction | |

The scan makes each field dimensionless by the case's reference scales and standardizes it by the `full` training side's
constants. That is an analysis of the release that reads its labels; the benchmark protocol (`../campaigns_2026-10-19/aerozoo_protocol/`)
gives no model a label.
