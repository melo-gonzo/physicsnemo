#!/usr/bin/env python
"""Normalization statistics for datasets/aerozoo_volume.yaml, straight from the memmaps (no torch).

Two subcommands. ``sample`` reads, for the cases of one manifest split part, a seeded 20,000-point sample
of the interior plus the globals and the inlet's prescribed turbulence, applies the yaml's
non-dimensionalization (velocity / |U_inf|, gauge pressure / q_inf with q_inf = rho |U_inf|^2 / 2,
nu_t / (|U_inf| L_ref), k / |U_inf|^2, omega L_ref / |U_inf|) and appends one JSON line per case
(sums and sums of squares, so the pooled moments are exact). It is resumable (cases already in the output
are skipped) and parallel-safe by case ranges: give each range its own output file::

    python aerozoo_stats.py sample <root> <manifest.json> geometry_holdout.train --out part0.jsonl --start 0 --stop 800
    python aerozoo_stats.py sample <root> <manifest.json> geometry_holdout.train --out part1.jsonl --start 800 --stop 1600

``reduce`` pools the lines, writes a JSON with the pooled mean and std of the five targets (velocity: per
component mean, one isotropic std = sqrt(mean over components of the component variances), the recipe's
vector convention) and the per-case mean and std of the inlet globals and log10 Re, and prints the yaml
blocks to paste::

    python aerozoo_stats.py reduce part0.jsonl part1.jsonl --out stats_geometry_holdout_train.json

``<root>`` may be the release or the row-shuffled copy (the sample is a seeded random row draw, so the row
order does not matter). Runs on the CPU partition: numpy only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

TD = "_tensordict"
TARGETS = {  # yaml target name -> (interior solution field, nondim rule)
    "velocity": ("velocity", "velocity"),
    "pressure": ("gauge_pressure", "pressure"),
    "nut": ("turbulent_kinematic_viscosity", "kinematic_viscosity"),
    "tke": ("turbulent_kinetic_energy", "turbulent_kinetic_energy"),
    "omega": ("specific_dissipation_rate", "specific_dissipation_rate"),
}
INLET_GLOBALS = {  # yaml global name -> (inlet prescribed.dirichlet field, nondim rule)
    "inlet_tke": ("turbulent_kinetic_energy", "turbulent_kinetic_energy"),
    "inlet_omega": ("specific_dissipation_rate", "specific_dissipation_rate"),
    "inlet_intermittency": ("intermittency", "identity"),
    "inlet_retheta": ("momentum_thickness_reynolds_number", "identity"),
}
DTYPES = {
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "torch.int64": np.int64,
    "torch.int32": np.int32,
}


def leaf(td_dir: Path, name: str) -> np.ndarray:
    """A memmap leaf of a saved TensorDict directory, by its meta.json entry (zero-size leaves have no file)."""
    meta = json.loads((td_dir / "meta.json").read_text())[name]
    shape, dtype = tuple(meta["shape"]), DTYPES[meta["dtype"]]
    path = td_dir / f"{name}.memmap"
    if math.prod(shape) == 0:
        return np.zeros(shape, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def scales(pdmsh: Path) -> dict[str, float]:
    reference = pdmsh / TD / "global_data" / "reference"
    physics = pdmsh / TD / "global_data" / "physics"
    U_inf = np.asarray(leaf(reference, "velocity"), dtype=np.float64)
    U = float(np.linalg.norm(U_inf))
    rho = float(leaf(physics, "density"))
    L = float(leaf(reference, "length"))
    nu = float(leaf(physics, "kinematic_viscosity"))
    speed = float(leaf(reference, "speed"))
    return {
        "U_inf_mag": U,
        "q_inf": 0.5 * rho * U * U,
        "L_ref": L,
        "nu": nu,
        "speed": speed,
        "log_re": math.log10(speed * L / nu),
    }


def nondim(x: np.ndarray, rule: str, s: dict[str, float]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if rule == "velocity":
        return x / s["U_inf_mag"]
    if rule == "pressure":
        return x / s["q_inf"]  # p_inf = 0
    if rule == "kinematic_viscosity":
        return x / (s["U_inf_mag"] * s["L_ref"])
    if rule == "turbulent_kinetic_energy":
        return x / s["U_inf_mag"] ** 2
    if rule == "specific_dissipation_rate":
        return x * s["L_ref"] / s["U_inf_mag"]
    if rule == "identity":
        return x
    raise ValueError(rule)


def case_dirs(root: Path) -> dict[str, Path]:
    """case id -> case directory, from one scan of the shard directories."""
    out: dict[str, Path] = {}
    for shard in sorted(root.glob("shard-*")):
        for case in shard.iterdir():
            if (case / "domain.pdmsh").is_dir():
                out[case.name] = case
    return out


def sample_case(case_dir: Path, case_id: str, n_points: int, seed: int) -> dict:
    pdmsh = case_dir / "domain.pdmsh"
    s = scales(pdmsh)
    solution = pdmsh / TD / "interior" / TD / "point_data" / "solution"
    n_total = int(json.loads((solution / "meta.json").read_text())["shape"][0])
    rng = np.random.default_rng([seed, zlib.crc32(case_id.encode())])
    idx = np.sort(rng.choice(n_total, size=min(n_points, n_total), replace=False))
    row = {
        "case": case_id,
        "n_total": n_total,
        "n": int(idx.size),
        "sum": {},
        "sumsq": {},
        "globals": {},
        "scales": s,
    }
    for name, (field, rule) in TARGETS.items():
        x = nondim(leaf(solution, field)[idx], rule, s)
        row["sum"][name] = np.atleast_1d(x.sum(axis=0)).tolist()
        row["sumsq"][name] = np.atleast_1d((x * x).sum(axis=0)).tolist()
    inlet = (
        pdmsh
        / TD
        / "boundaries"
        / "inlet"
        / TD
        / "cell_data"
        / "prescribed"
        / "dirichlet"
    )
    for name, (field, rule) in INLET_GLOBALS.items():
        values = nondim(leaf(inlet, field), rule, s)
        row["globals"][name] = float(values.mean()) if values.size else 0.0
    row["globals"]["log_re"] = s["log_re"]
    return row


def cmd_sample(args) -> None:
    manifest = json.loads(Path(args.manifest).read_text())
    cases = sorted(manifest[args.split])[args.start : args.stop]
    done: set[str] = set()
    out = Path(args.out)
    if out.exists():
        done = {
            json.loads(line)["case"]
            for line in out.read_text().splitlines()
            if line.strip()
        }
    dirs = case_dirs(Path(args.root))
    missing = [c for c in cases if c not in dirs]
    if missing:
        raise SystemExit(
            f"{len(missing)} cases of {args.split} are not under {args.root}: {missing[:5]}"
        )
    todo = [c for c in cases if c not in done]
    print(
        f"{args.split}: {len(cases)} cases in [{args.start}, {args.stop}), {len(done)} done, {len(todo)} to go",
        file=sys.stderr,
    )
    with out.open("a") as f:
        for i, case in enumerate(todo, 1):
            row = sample_case(dirs[case], case, args.n_points, args.seed)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(
                    f"  {i}/{len(todo)} ({case}, {row['n_total']:,} points)",
                    file=sys.stderr,
                )


def pooled(rows: list[dict]) -> dict:
    n = sum(r["n"] for r in rows)
    stats: dict = {
        "n_points": n,
        "n_cases": len(rows),
        "targets": {},
        "inlet_globals": {},
        "log_re": {},
    }
    for name in TARGETS:
        total = np.sum([r["sum"][name] for r in rows], axis=0)
        total_sq = np.sum([r["sumsq"][name] for r in rows], axis=0)
        mean = total / n
        var = np.maximum(total_sq / n - mean * mean, 0.0)
        if name == "velocity":
            std = float(np.sqrt(var.mean()))  # isotropic: Frobenius / sqrt(3)
            stats["targets"][name] = {
                "type": "vector",
                "mean": mean.tolist(),
                "std": std,
                "std_per_component": np.sqrt(var).tolist(),
            }
        else:
            stats["targets"][name] = {
                "type": "scalar",
                "mean": float(mean[0]),
                "std": float(np.sqrt(var[0])),
            }
    for name in (*INLET_GLOBALS, "log_re"):
        values = np.array([r["globals"][name] for r in rows], dtype=np.float64)
        entry = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        if name == "log_re":
            stats["log_re"] = entry
        else:
            stats["inlet_globals"][name] = entry
    return stats


def fmt(x: float) -> str:
    return f"{x:.6g}"


def print_yaml(stats: dict) -> None:
    t = stats["targets"]
    print(
        "# NormalizeMeshFields block for datasets/aerozoo_volume.yaml "
        f"({stats['n_points']:,} points over {stats['n_cases']} cases)"
    )
    print("      fields:")
    v = t["velocity"]
    print(
        f"        velocity: {{type: vector, mean: [{', '.join(fmt(m) for m in v['mean'])}], std: {fmt(v['std'])}}}"
        f"   # per-component std {' / '.join(fmt(s) for s in v['std_per_component'])}; isotropic value"
    )
    for name in ("pressure", "nut", "tke", "omega"):
        print(
            f"        {name}: {{type: scalar, mean: {fmt(t[name]['mean'])}, std: {fmt(t[name]['std'])}}}"
        )
    g = stats["inlet_globals"]
    names = list(INLET_GLOBALS)
    stds = [g[n]["std"] if g[n]["std"] > 0 else 1.0 for n in names]
    print(
        "# SplitGlobalVector (inlet_turbulence) block; a std of 0 (constant over the split) is written as 1.0"
    )
    print(f"      mean: [{', '.join(fmt(g[n]['mean']) for n in names)}]")
    print(f"      std: [{', '.join(fmt(s) for s in stds)}]")
    lr = stats["log_re"]
    print(
        f"# log10 Re over the split: mean {fmt(lr['mean'])}, std {fmt(lr['std'])}, range [{fmt(lr['min'])}, {fmt(lr['max'])}]"
        " (the yaml standardizes with 5.5 / 1.5)"
    )


def cmd_reduce(args) -> None:
    rows: dict[str, dict] = {}
    for path in args.jsonl:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["case"]] = row  # the last line of a case wins (a rerun)
    if not rows:
        raise SystemExit("no case lines to reduce")
    stats = pooled(list(rows.values()))
    if args.out:
        Path(args.out).write_text(json.dumps(stats, indent=1) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    print_yaml(stats)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="append one JSON line per case of a split part")
    s.add_argument("root", help="the data root (release or shuffled copy)")
    s.add_argument("manifest", help="the recipe manifest.json (build_manifest.py)")
    s.add_argument("split", help="a manifest key, e.g. geometry_holdout.train")
    s.add_argument(
        "--out", required=True, help="the JSONL to append to (one file per case range)"
    )
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--stop", type=int, default=None)
    s.add_argument("--n-points", type=int, default=20000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)
    r = sub.add_parser(
        "reduce", help="pool JSONL files into mean / std and print the yaml blocks"
    )
    r.add_argument("jsonl", nargs="+")
    r.add_argument("--out", default=None, help="JSON file for the pooled statistics")
    r.set_defaults(func=cmd_reduce)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
