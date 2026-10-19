"""Extract the dataset-only part of the AeroZoo field scan (2026-10-19).

Run once, before results/campaigns_2026-10-09/aerozoo/ was deleted, on its probe/outliers.json (written by
probe/outliers_reduce.py, 2026-10-11) and probe/outliers_scan.jsonl (one line per case of the `full` split, written by
probe/outliers_scan.py over every interior row). Keeps the dataset statistics and drops the two blocks that read model
runs ("validation": per-case scores of label-input checkpoints; "spike_cases": loss spikes of label-input training runs).
Adds the count of cases of the scan whose largest standardized omega exceeds 1,000 and 10,000.

The scan's fields are made dimensionless by each case's reference scales (velocity / |U_inf|, p / q_inf, nu_t / (|U_inf| L_ref),
k / |U_inf|^2, omega L_ref / |U_inf|) and standardized by the `full` training side's constants: an analysis of the release that
uses its labels, not a model input.

usage: python target_scan_extract.py <outliers.json> <outliers_scan.jsonl> target_scan.json
"""
import json, sys

out_json, scan_jsonl, dst = sys.argv[1:4]
src = json.load(open(out_json))
rows = [json.loads(line) for line in open(scan_jsonl) if line.strip()]
omega = [r["fields"]["omega"]["max_abs_z"] for r in rows]
res = {"source": "results/campaigns_2026-10-09/aerozoo/probe/outliers.json and outliers_scan.jsonl (deleted 2026-10-19)",
       "thresholds": src["thresholds"], "scan": src["scan"],
       "omega_max_abs_z": {"n_cases": len(rows), "gt_1000": sum(v > 1000 for v in omega), "gt_10000": sum(v > 10000 for v in omega)}}
json.dump(res, open(dst, "w"), indent=1)
print(res["omega_max_abs_z"], {k: v["n_cases"] for k, v in res["scan"].items()})
