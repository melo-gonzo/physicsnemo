"""Fill the statistics placeholders of datasets/aerozoo_volume.yaml from an aerozoo_stats.py `reduce` JSON.

The dataset yaml standardizes the five interior targets and the four inlet turbulence globals with the pooled moments of
the training side of the split in use (book notebook #sec-nb-aerozoo-prereg), so every split gets its own copy of the
yaml: this script writes one from the template, replacing the `"???"` / `???` placeholders and nothing else, and
records the provenance (split, cases, points) in a header comment.

usage: python fill_stats.py <template.yaml> <stats.json> <split_key> <out.yaml>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def fmt(x: float) -> str:
    return f"{x:.6g}"


def main(template: Path, stats_path: Path, split: str, out: Path) -> None:
    text = template.read_text()
    # A template may carry the "???" placeholders or the numbers of an earlier fill (then this rewrites them).
    text = re.sub(r"^# Statistics filled by fill_stats\.py.*\n", "", text, count=1, flags=re.M)
    stats = json.load(open(stats_path))
    t = stats["targets"]
    value = r'(?:"?\?\?\?"?|\[[^\]]*\]|[-+0-9.eE]+)'
    for name in ("velocity", "pressure", "nut", "tke", "omega"):
        s = t[name]
        if s["type"] == "vector":
            mean = "[" + ", ".join(fmt(v) for v in s["mean"]) + "]"
        else:
            mean = fmt(s["mean"])
        pattern = re.compile(rf'(\s{name}: \{{type: {s["type"]}, mean: ){value}(, std: ){value}(\}})')
        text, n = pattern.subn(rf"\g<1>{mean}\g<2>{fmt(s['std'])}\g<3>", text)
        if n != 1:
            raise SystemExit(f"{name}: expected one statistics line, found {n}")
    g = stats["inlet_globals"]
    names = ("inlet_tke", "inlet_omega", "inlet_intermittency", "inlet_retheta")
    means = "[" + ", ".join(fmt(g[k]["mean"]) for k in names) + "]"
    stds = "[" + ", ".join(fmt(g[k]["std"] if g[k]["std"] > 0 else 1.0) for k in names) + "]"
    ### the SplitGlobalVector block of the inlet turbulence: the only `mean:` / `std:` lines at that indentation that
    ### hold a placeholder or a four-element list
    text, n1 = re.subn(rf"(\n\s+mean: ){value}(\n\s+std: ){value}(\n\s+output_field: inlet_turbulence_std)",
                       rf"\g<1>{means}\g<2>{stds}\g<3>", text)
    if n1 != 1:
        raise SystemExit(f"inlet globals: expected one mean/std pair before output_field: inlet_turbulence_std, found {n1}")
    if "???" in text:
        raise SystemExit("a placeholder survived")
    header = (f"# Statistics filled by fill_stats.py from {stats_path.name}: split part {split}, {stats['n_cases']} cases, "
              f"{stats['n_points']:,} interior points (log10 Re mean {stats['log_re']['mean']:.3f}, std {stats['log_re']['std']:.3f}).\n")
    out.write_text(header + text)
    print(f"wrote {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), a[2], Path(a[3]))
