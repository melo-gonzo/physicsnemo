#!/usr/bin/env python
"""Statistics for the physical-input AeroZoo files, datasets/aerozoo_puvolume*.yaml (AZ-COMPLIANT v2, 2026-10-19).

Every statistic is a per-dataset constant fitted on the training side of a split; nothing is per case. Numpy only.

``boundary`` appends one JSON line per case of a manifest part: the kinematic viscosity and, for each of the four
boundaries, its face count and the face means of every prescribed field the pipeline harmonizes (``bc_*``), with the
boundary's token count min(F, 50,000) (SubsampleMesh's uniform draw, so the face mean is the expected token mean)::

    python aerozoo_pstats.py boundary <root> <manifest.json> full.train --out b0.jsonl --start 0 --stop 400

``fill`` writes a dataset file from the template (datasets/aerozoo_puvolume.yaml) with one split's statistics and
its label-era twin (the ``label_reference`` dataset)::

    python aerozoo_pstats.py fill aerozoo_puvolume.yaml pstats_reynolds.train.json aerozoo_uvolume_reynolds \
        aerozoo_puvolume_reynolds.yaml

``reduce`` pools, for one split part:

* the five targets in SI units, from the label-era per-case sample lines of ``aerozoo_stats.py sample`` (the same
  seeded 20,000 interior points per case). Those lines hold the sums of x / s and (x / s)^2 with the case's
  nondimensionalization scale s, and its scales, so the physical sums are s * sum and s^2 * sumsq: exact, with no
  re-read. Every target gets mean 0 and its RMS about 0 as the std (velocity: the isotropic one,
  sqrt(mean over components of E[u_c^2])), a scale-only standardization: the physical targets span decades
  across cases, and an offset would put a small case's whole field below the float32 / bf16 step of the offset
  (supervisor, 2026-10-19). The pooled mean is kept beside it as ``mean_measured``.
* ``nu_std``'s constants: the mean and std over the cases of log10 nu.
* the token standardization of the ``bc_*`` columns GeoTransolver reads (MergeBoundaries): pooled over the expected
  tokens of the four boundaries (zero where a boundary does not prescribe the field, as HarmonizeBoundaryFields
  fills it); scale-only like the targets, for the same reason (supervisor, 2026-10-19: a slow case's inlet k of
  ~1e-4 m^2/s^2 sat 1e-4 above a pooled-mean offset of 0.12, below bf16's step there), so every column has mean 0
  and its RMS about 0 as the std (vectors the isotropic one); an RMS of 0 (the gauge outlet pressure) is written
  as 1.0::

    python aerozoo_pstats.py reduce --manifest manifest.json --split full.train --label-rows stats/full.*.jsonl \
        --boundary-rows b*.jsonl --out pstats_full.train.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

TD = "_tensordict"
BOUNDARIES = ("inlet", "outlet", "no_slip", "slip")
TOKEN_CAP = 50000  # the SubsampleMesh n_cells of the dataset files
BC_FIELDS = {  # prescribed leaf -> the recipe name, as the dataset files rename it
    "dirichlet/velocity": "bc_velocity",
    "reverse_flow/velocity": "bc_velocity_reverse",
    "dirichlet/gauge_pressure": "bc_pressure",
    "dirichlet/turbulent_kinetic_energy": "bc_tke",
    "dirichlet/specific_dissipation_rate": "bc_omega",
    "dirichlet/intermittency": "bc_intermittency",
    "dirichlet/momentum_thickness_reynolds_number": "bc_retheta",
    "reverse_flow/turbulent_kinetic_energy": "bc_tke_reverse",
    "reverse_flow/specific_dissipation_rate": "bc_omega_reverse",
    "reverse_flow/intermittency": "bc_intermittency_reverse",
    "reverse_flow/momentum_thickness_reynolds_number": "bc_retheta_reverse",
}
VECTORS = ("bc_velocity", "bc_velocity_reverse", "velocity")
DTYPES = {
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "torch.int64": np.int64,
}


def _meta(td_dir: Path) -> dict:
    return json.loads((td_dir / "meta.json").read_text())


def _leaf(td_dir: Path, name: str) -> np.ndarray:
    meta = _meta(td_dir)[name]
    shape = tuple(meta["shape"])
    if math.prod(shape) == 0:
        return np.zeros(shape, np.float64)
    return np.memmap(
        td_dir / f"{name}.memmap", dtype=DTYPES[meta["dtype"]], mode="r", shape=shape
    )


def boundary_row(pdmsh: Path, case_id: str) -> dict:
    nu = float(_leaf(pdmsh / TD / "global_data" / "physics", "kinematic_viscosity"))
    row = {"case": case_id, "nu": nu, "boundaries": {}}
    for b in BOUNDARIES:
        bdir = pdmsh / TD / "boundaries" / b / TD
        n = int(_meta(bdir)["cells"]["shape"][0])
        entry = {"n_faces": n, "n_tokens": min(n, TOKEN_CAP), "mean": {}, "mean_sq": {}}
        for leaf, name in BC_FIELDS.items():
            group, field = leaf.split("/")
            gdir = bdir / "cell_data" / "prescribed" / group
            if n == 0 or not (gdir / "meta.json").exists() or field not in _meta(gdir):
                continue
            x = np.asarray(_leaf(gdir, field), dtype=np.float64).reshape(n, -1)
            entry["mean"][name] = x.mean(axis=0).tolist()
            entry["mean_sq"][name] = (x * x).mean(axis=0).tolist()
        row["boundaries"][b] = entry
    return row


def cmd_boundary(args) -> None:
    cases = sorted(json.loads(Path(args.manifest).read_text())[args.split])[
        args.start : args.stop
    ]
    out = Path(args.out)
    done = (
        {
            json.loads(line)["case"]
            for line in out.read_text().splitlines()
            if line.strip()
        }
        if out.exists()
        else set()
    )
    dirs = {
        c.name: c
        for shard in sorted(Path(args.root).glob("shard-*"))
        for c in shard.iterdir()
    }
    todo = [c for c in cases if c not in done]
    print(f"{args.split}[{args.start}:{args.stop}]: {len(todo)} to go", file=sys.stderr)
    with out.open("a") as f:
        for i, case in enumerate(todo, 1):
            f.write(json.dumps(boundary_row(dirs[case] / "domain.pdmsh", case)) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", file=sys.stderr)


def _rows(paths, cases: set[str]) -> dict[str, dict]:
    rows = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["case"] in cases:
                    rows[r["case"]] = r
    return rows


def _scale(name: str, s: dict) -> float:
    """The label-era nondimensionalization scale of a target (aerozoo_stats.py nondim): x_nd = x / scale."""
    U, L = s["U_inf_mag"], s["L_ref"]
    return {
        "velocity": U,
        "pressure": s["q_inf"],
        "nut": U * L,
        "tke": U * U,
        "omega": U / L,
    }[name]


def _moments(
    name: str,
    total: np.ndarray,
    total_sq: np.ndarray,
    n: float,
    scale_only: bool = False,
) -> dict:
    mean, mean_sq = total / n, total_sq / n
    if name in VECTORS:
        return {
            "type": "vector",
            "mean": [0.0] * len(mean),
            "std": float(np.sqrt(mean_sq.mean())),
            "mean_measured": mean.tolist(),
        }
    if scale_only:
        rms = float(np.sqrt(mean_sq[0]))
        return {
            "type": "scalar",
            "mean": 0.0,
            "std": rms if rms > 0 else 1.0,
            "mean_measured": float(mean[0]),
        }
    std = float(np.sqrt(max(mean_sq[0] - mean[0] ** 2, 0.0)))
    return {"type": "scalar", "mean": float(mean[0]), "std": std if std > 0 else 1.0}


def cmd_reduce(args) -> None:
    cases = set(json.loads(Path(args.manifest).read_text())[args.split])
    label = _rows(args.label_rows, cases)
    bnd = _rows(args.boundary_rows, cases)
    missing = (len(cases - set(label)), len(cases - set(bnd)))
    if any(missing):
        raise SystemExit(
            f"{args.split}: {missing[0]} cases without a label-era sample line, {missing[1]} without a boundary line"
        )
    stats = {"split": args.split, "n_cases": len(cases), "targets": {}, "bc": {}}
    n = sum(r["n"] for r in label.values())
    stats["n_points"] = n
    for name in ("velocity", "pressure", "nut", "tke", "omega"):
        total = sum(
            _scale(name, r["scales"]) * np.asarray(r["sum"][name])
            for r in label.values()
        )
        total_sq = sum(
            _scale(name, r["scales"]) ** 2 * np.asarray(r["sumsq"][name])
            for r in label.values()
        )
        stats["targets"][name] = _moments(name, total, total_sq, n, scale_only=True)
    log_nu = np.log10([r["nu"] for r in bnd.values()])
    stats["log10_nu"] = {
        "mean": float(log_nu.mean()),
        "std": float(log_nu.std()),
        "min": float(log_nu.min()),
        "max": float(log_nu.max()),
    }
    n_tok = sum(e["n_tokens"] for r in bnd.values() for e in r["boundaries"].values())
    stats["n_tokens"] = n_tok
    for name in BC_FIELDS.values():
        d = 3 if name in VECTORS else 1
        total, total_sq = np.zeros(d), np.zeros(d)
        for r in bnd.values():
            for e in r["boundaries"].values():
                if name in e["mean"]:
                    total += e["n_tokens"] * np.asarray(e["mean"][name])
                    total_sq += e["n_tokens"] * np.asarray(e["mean_sq"][name])
        stats["bc"][name] = _moments(name, total, total_sq, n_tok, scale_only=True)
    Path(args.out).write_text(json.dumps(stats, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: stats[k]
                for k in ("split", "n_cases", "n_points", "n_tokens", "log10_nu")
            }
        )
    )


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def cmd_fill(args) -> None:
    """Write a dataset file from the template with one split's statistics and its label-era twin."""
    text = Path(args.template).read_text()
    stats = json.loads(Path(args.stats).read_text())
    value = r"(?:\[[^\]]*\]|[-+0-9.eE]+)"
    for name, s in [*stats["targets"].items(), *stats["bc"].items()]:
        mean = (
            "[" + ", ".join(_fmt(v) for v in s["mean"]) + "]"
            if s["type"] == "vector"
            else _fmt(s["mean"])
        )
        pattern = re.compile(
            rf"(\s{name}: \{{type: {s['type']}, mean: ){value}(, std: ){value}(\}})"
        )
        text, n = pattern.subn(rf"\g<1>{mean}\g<2>{_fmt(s['std'])}\g<3>", text)
        if n != 1:
            raise SystemExit(f"{name}: expected one statistics line, found {n}")
    nu = stats["log10_nu"]
    text, n = re.subn(
        rf"(output_field: nu_std\n\s+log10: true\n\s+mean: ){value}(\n\s+std: ){value}",
        rf"\g<1>{_fmt(nu['mean'])}\g<2>{_fmt(nu['std'])}",
        text,
    )
    if n != 1:
        raise SystemExit("nu_std: expected one StandardizeGlobalLeaf block")
    text, n = re.subn(
        r"(\nlabel_reference:\n\s+dataset: )\S+", rf"\g<1>{args.label_dataset}", text
    )
    if n != 1:
        raise SystemExit("label_reference: expected one dataset line")
    header = (
        f"# Statistics filled by aerozoo_pstats.py fill from {Path(args.stats).name}: split part {stats['split']}, "
        f"{stats['n_cases']} cases, {stats['n_points']:,} interior points, {stats['n_tokens']:,} expected boundary "
        f"tokens (log10 nu mean {nu['mean']:.3f}, std {nu['std']:.3f}).\n"
    )
    lines = text.splitlines(keepends=True)
    if lines[0].startswith("# Statistics"):
        lines = lines[1:]
    Path(args.out).write_text(header + "".join(lines))
    print(f"wrote {args.out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("boundary")
    s.add_argument("root")
    s.add_argument("manifest")
    s.add_argument("split")
    s.add_argument("--out", required=True)
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--stop", type=int, default=None)
    s.set_defaults(func=cmd_boundary)
    r = sub.add_parser("reduce")
    r.add_argument("--manifest", required=True)
    r.add_argument("--split", required=True)
    r.add_argument("--label-rows", nargs="+", required=True)
    r.add_argument("--boundary-rows", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_reduce)
    f = sub.add_parser("fill")
    f.add_argument("template")
    f.add_argument("stats")
    f.add_argument(
        "label_dataset",
        help="the label-era file of the same split, e.g. aerozoo_uvolume_reynolds",
    )
    f.add_argument("out")
    f.set_defaults(func=cmd_fill)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
