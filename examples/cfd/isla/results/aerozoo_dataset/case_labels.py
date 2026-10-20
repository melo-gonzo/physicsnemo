#!/usr/bin/env python
"""Per-case labels of AeroZoo v1.2 for the benchmark's reductions (AEROZOO, 2026-10-10).

Usage::

    python case_labels.py <release_root> <out.json>

Reads ``<release_root>/index.json`` (``samples``: the released cases with their ``quality`` block) and every released
case's ``case-manifest.json`` (the generator's record of the scenario: enclosure family and profile, body layout, the
geometry bank, the target Reynolds number). Writes ``{case_id: {quality, cells, enclosure, scenario, profile,
body_layout, bank, reynolds, log10_re}}`` plus ``_meta`` (counts per label). The reduction stratifies the scored
cases by these labels (the two stability populations are reported separately per the pre-registration; the held-out
splits are read against the family they hold out).
"""

from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path


def main(root: str, out: str) -> None:
    root_p = Path(root)
    index = json.load(open(root_p / "index.json"))
    samples = index["samples"]
    ids = list(samples) if isinstance(samples, dict) else [s if isinstance(s, str) else s["case_id"] for s in samples]
    # The geometry bank is not a field of the case manifest (geometry.source names the STL file and the corpus hash); the
    # release's held-out-bank splits name it: a case in splits["bank_<name>_test"] belongs to bank <name>.
    splits = json.load(open(root_p / "splits" / "splits.json"))
    bank_of: dict[str, str] = {}
    for key, members in splits.items():
        m = re.match(r"^bank_(.+)_test$", key)
        if m:
            for cid in members:
                bank_of[cid] = m.group(1)
    labels: dict[str, dict] = {}
    geometry_keys: collections.Counter = collections.Counter()
    for i, cid in enumerate(sorted(ids)):
        rec = samples[cid] if isinstance(samples, dict) else {}
        shard = rec.get("shard")
        cand = [root_p / f"shard-{shard:03d}" / cid] if isinstance(shard, int) else []
        cand += sorted(root_p.glob(f"shard-*/{cid}"))
        case_dir = next((c for c in cand if (c / "case-manifest.json").exists()), None)
        q = rec.get("quality", {}) if isinstance(rec, dict) else {}
        lab = {"quality": q.get("solution_quality"), "cells": q.get("cells")}
        if case_dir is not None:
            m = json.load(open(case_dir / "case-manifest.json"))
            dom, sc, geo, phys = m.get("domain", {}), m.get("scenario", {}), m.get("geometry", {}), m.get("physics", {})
            geometry_keys.update(geo.keys())
            re_ = phys.get("target_reynolds_number")
            src = geo.get("source", {}) if isinstance(geo.get("source"), dict) else {}
            lab.update({
                "enclosure": dom.get("family"),
                "scenario": sc.get("family"),
                "profile": sc.get("profile"),
                "body_layout": geo.get("body_layout"),
                "bank": bank_of.get(cid),
                "source_file": src.get("relative_name"),
                "source_kind": src.get("kind"),
                "reynolds": re_,
                "log10_re": (math.log10(re_) if re_ else None),
            })
        labels[cid] = lab
        if i % 500 == 0:
            print(f"{i}/{len(ids)}", flush=True)
    counts = {k: dict(collections.Counter(str(v.get(k)) for v in labels.values()))
              for k in ("quality", "enclosure", "scenario", "profile", "body_layout", "bank")}
    counts["log10_re_decade"] = dict(collections.Counter(
        str(math.floor(v["log10_re"])) if v.get("log10_re") is not None else "None" for v in labels.values()))
    labels["_meta"] = {"release_root": root, "n_cases": len(ids), "counts": counts,
                       "geometry_keys_seen": dict(geometry_keys)}
    Path(out).write_text(json.dumps(labels, indent=1, default=str))
    print(json.dumps(labels["_meta"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
