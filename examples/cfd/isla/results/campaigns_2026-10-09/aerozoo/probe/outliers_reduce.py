#!/usr/bin/env python
"""Reduce the AeroZoo outlier probe into outliers.json and outliers.txt (AEROZOO, 2026-10-11).

Joins outliers_repro.json (the case behind each logged loss spike), outliers_scan.jsonl (per-case standardized
target statistics over every interior row, full-split constants), verify_<split>.json (the recipe pipeline on the
identified cases), kernel/aerozoo/case_labels.json (bank, enclosure, quality, Reynolds) and the per-case
validation scores of the scored lanes (p200k_fp32/<lane>/<lane>/metrics.jsonl, phase infer_step).

usage: python outliers_reduce.py --out-json outliers.json --out-txt outliers.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TASK = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
)
PROBE = TASK / "kernel" / "aerozoo" / "probe"
FIELDS = ["velocity", "pressure", "nut", "tke", "omega"]
MEAN_THRESHOLD = 3.0
MAX_THRESHOLD = 100.0
SCORED_LANES = ["az_full_isla_seed43", "az_nonbox_isla_seed42", "az_nonbox_isla_seed43"]
CASE_RE = re.compile(r"(case-\d+-[0-9a-f]+)")


def label(labels: dict, case: str) -> dict:
    v = labels.get(case, {})
    return {k: v.get(k) for k in ("bank", "enclosure", "scenario", "profile", "quality", "reynolds", "log10_re", "cells")}


def g(x, spec: str = ".3g") -> str:
    """Format a number, or '?' when the label is missing."""
    return "?" if x is None else format(x, spec)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-txt", required=True)
    args = parser.parse_args(argv)

    labels = json.loads((TASK / "kernel/aerozoo/case_labels.json").read_text())
    manifest = json.loads((TASK / "kernel/aerozoo/manifest.json").read_text())
    membership = defaultdict(list)
    for key, cases in manifest.items():
        if key.split(".")[0] in ("full", "family_non_box", "bank_blobs", "reynolds"):
            for c in cases:
                membership[c].append(key)
    repro = json.loads((PROBE / "outliers_repro.json").read_text())
    scan_rows = {r["case"]: r for r in load_jsonl(PROBE / "outliers_scan.jsonl")}
    verify = {}
    for p in sorted(PROBE.glob("verify_*.json")):
        verify[p.stem.split("_", 1)[1]] = json.loads(p.read_text())

    lines: list[str] = []
    out: dict = {"thresholds": {"mean_abs_z": MEAN_THRESHOLD, "max_abs_z": MAX_THRESHOLD}}

    # 1. The spike cases.
    spikes: dict[str, dict] = {}
    for key, c in repro["consensus"].items():
        case = c["case"]
        if case is None:
            continue
        s = spikes.setdefault(case, {"classes": [], "labels": label(labels, case), "splits": membership.get(case, [])})
        losses = [
            h["loss"]
            for lane in repro["lanes"].values()
            for h in lane["hits"]
            if lane["train_split"] == c["train_split"] and h["class"] == c["loss_class"]
        ]
        s["classes"].append(
            {
                "train_split": c["train_split"],
                "loss_class": c["loss_class"],
                "n_spikes": c["n_spikes"],
                "logged_loss_min": min(losses),
                "logged_loss_max": max(losses),
                "lanes": sorted({lane for lane, v in repro["lanes"].items() if v["train_split"] == c["train_split"] and any(h["class"] == c["loss_class"] for h in v["hits"])}),
            }
        )
    for case, s in spikes.items():
        if case in scan_rows:
            s["scan_full_constants"] = {
                f: {k: scan_rows[case]["fields"][f][k] for k in ("mean_abs_z", "max_abs_z", "frac_abs_z_gt1", "n_abs_z_gt100")}
                for f in FIELDS
            }
            s["scan_full_constants"]["huber_zero_total"] = scan_rows[case]["huber_zero_total"]
            s["scan_full_constants"]["n_total"] = scan_rows[case]["n_total"]
        s["verify"] = {split: v["cases"].get(case) for split, v in verify.items() if case in v.get("cases", {})}
    out["spike_cases"] = spikes
    lines.append("AEROZOO loss spikes: the training cases behind them (outliers_repro.py; the logged loss is the 4-rank mean)")
    for case, s in spikes.items():
        lab = s["labels"]
        lines.append(f"  {case}: bank {lab['bank']}, enclosure {lab['enclosure']}, scenario {lab['scenario']}, profile {lab['profile']}, quality {lab['quality']}, Re {g(lab['reynolds'])}, {g(lab['cells'], ',')} cells")
        lines.append(f"    in: {', '.join(s['splits'])}")
        for c in s["classes"]:
            lines.append(f"    {c['train_split']}: {c['n_spikes']} spikes at logged loss {c['logged_loss_min']:.2f}-{c['logged_loss_max']:.2f} ({len(c['lanes'])} lanes)")
        if "scan_full_constants" in s:
            sc = s["scan_full_constants"]
            lines.append("    full-split constants, every row: " + "; ".join(f"{f} mean|z| {sc[f]['mean_abs_z']:.4g} max|z| {sc[f]['max_abs_z']:.4g}" for f in FIELDS) + f"; zero-pred loss {sc['huber_zero_total']:.4g}")
        for split, v in s["verify"].items():
            if v:
                lines.append(f"    recipe pipeline ({split} constants): zero-pred loss whole case {v['zero_pred_loss_whole_case']['total']:.4g}; a 200k block holding the worst row -> {v['zero_pred_loss_block_with_worst_row']['total']:.5g}, /4 ranks {v['logged_equivalent_block_over_4_ranks']:.5g}; " + "; ".join(f"{f} mean|z| {v['fields'][f]['mean_abs_z']:.4g} max|z| {v['fields'][f]['max_abs_z']:.4g}" for f in FIELDS))

    # 2. The scan of the full split.
    scan_summary = {}
    for part in ("full.train", "full.val", "full.test"):
        cases = sorted(manifest[part])
        rows = [scan_rows[c] for c in cases if c in scan_rows]
        flagged = []
        for r in rows:
            reasons = []
            for f in FIELDS:
                e = r["fields"][f]
                if e["mean_abs_z"] > MEAN_THRESHOLD:
                    reasons.append(f"{f} mean|z| {e['mean_abs_z']:.3g}")
                if e["max_abs_z"] > MAX_THRESHOLD:
                    reasons.append(f"{f} max|z| {e['max_abs_z']:.3g}")
            if reasons:
                flagged.append({"case": r["case"], "reasons": reasons, "huber_zero_total": r["huber_zero_total"], "n_total": r["n_total"], **label(labels, r["case"])})
        flagged.sort(key=lambda x: -x["huber_zero_total"])
        mean_flagged = [x for x in flagged if any("mean|z|" in s for s in x["reasons"])]
        max_only = [x for x in flagged if not any("mean|z|" in s for s in x["reasons"])]
        q_all = Counter(labels.get(c, {}).get("quality") for c in cases)
        q_flag = Counter(x["quality"] for x in flagged)
        q_mean = Counter(x["quality"] for x in mean_flagged)
        per_field_max = {f: sum(1 for r in rows if r["fields"][f]["max_abs_z"] > MAX_THRESHOLD) for f in FIELDS}
        per_field_mean = {f: sum(1 for r in rows if r["fields"][f]["mean_abs_z"] > MEAN_THRESHOLD) for f in FIELDS}
        scan_summary[part] = {
            "n_cases": len(cases),
            "n_scanned": len(rows),
            "quality_all": dict(q_all),
            "n_flagged": len(flagged),
            "quality_flagged": dict(q_flag),
            "n_flagged_by_mean": len(mean_flagged),
            "quality_flagged_by_mean": dict(q_mean),
            "n_flagged_by_max_only": len(max_only),
            "per_field_n_max_gt_threshold": per_field_max,
            "per_field_n_mean_gt_threshold": per_field_mean,
            "flagged": flagged,
        }
        lines.append("")
        lines.append(f"{part}: {len(rows)}/{len(cases)} cases scanned (every interior row, full-split constants); quality {dict(q_all)}")
        lines.append(f"  flagged (mean|z| > {MEAN_THRESHOLD:g} or max|z| > {MAX_THRESHOLD:g} on any field): {len(flagged)}, quality {dict(q_flag)}; by mean|z|: {len(mean_flagged)} ({dict(q_mean)}); by max|z| only: {len(max_only)}")
        lines.append(f"  per field, cases with max|z| > {MAX_THRESHOLD:g}: {per_field_max}; with mean|z| > {MEAN_THRESHOLD:g}: {per_field_mean}")
        for x in flagged[:25]:
            lines.append(f"    {x['case']}: zero-pred loss {x['huber_zero_total']:.3g}; {', '.join(x['reasons'])}; {x['bank']}/{x['enclosure']}/{x['quality']}, Re {g(x['reynolds'])}")
        if len(flagged) > 25:
            lines.append(f"    ... {len(flagged) - 25} more in outliers.json")
    out["scan"] = scan_summary

    # 3. The worst validation cases of the scored lanes.
    val = {}
    for lane in SCORED_LANES:
        path = TASK / "p200k_fp32" / lane / lane / "metrics.jsonl"
        if not path.exists():
            continue
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        config = next((r for r in records if r.get("phase") == "config"), {})
        steps = [r for r in records if r.get("phase") == "infer_step"]
        per_case = {}
        for r in steps:
            m = CASE_RE.search(r["sample_id"])
            case = m.group(1) if m else r["sample_id"]
            per_case[case] = r["metrics"]
        worst = {}
        for f in FIELDS:
            key = f"{f}_l2"
            ranked = sorted(per_case.items(), key=lambda kv: -kv[1].get(key, 0.0))[:10]
            worst[f] = [{"case": c, key: m.get(key), **label(labels, c)} for c, m in ranked]
        n_unstable = sum(1 for c in per_case if labels.get(c, {}).get("quality") == "mean_unstable")
        val[lane] = {"infer_split": config.get("infer_split"), "epoch": config.get("epoch"), "n_cases": len(per_case), "n_mean_unstable": n_unstable, "worst": worst}
        lines.append("")
        lines.append(f"{lane} ({config.get('infer_split')} at epoch {config.get('epoch')}): {len(per_case)} scored cases, {n_unstable} labeled mean_unstable; the 10 worst per field (relative L2):")
        for f in FIELDS:
            key = f"{f}_l2"
            lines.append(f"  {f}: " + "; ".join(f"{w['case'][5:24].lstrip('0') or '0'}..{w['case'][-4:]} {g(w[key])} ({w['quality']}, {w['bank']})" for w in worst[f]))
    out["validation"] = val

    Path(args.out_json).write_text(json.dumps(out, indent=1) + "\n")
    Path(args.out_txt).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
