#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Statistics for the AeroZoo dataset files, datasets/aerozoo_uvolume*.yaml (physical inputs, 2026-10-19).

Every statistic is a per-dataset constant fitted on the training side of a split; nothing is per case. Numpy only.

``boundary`` appends one JSON line per case of a manifest part: the kinematic viscosity and, for each of the four
boundaries, its face count and the face means of every prescribed field the pipeline harmonizes (``bc_*``), with the
boundary's token count min(F, 50,000) (SubsampleMesh's uniform draw, so the face mean is the expected token mean)::

    python aerozoo_pstats.py boundary <root> <manifest.json> full.train --out b0.jsonl --start 0 --stop 400

``targets`` (2026-10-26, the seven interior fields of AeroZoo's interior task) appends, per case of a manifest part,
the per-case sample line of ``aerozoo_stats.py sample`` (the same seeded 20,000 release rows per case, the same
sums of x / s and (x / s)^2 with the case's scales s, a dimensionless field's s being 1) for all seven targets, read
from the row-shuffled copy: each case's SHUFFLE.json seed remakes its row permutation, so the release's rows are read
where the copy put them and the five fields' lines equal the 2026-10-09 lines of ``aerozoo_stats.py``::

    python aerozoo_pstats.py targets <shuffled root> <manifest.json> full.train --out t0.jsonl --start 0 --stop 400

``fill`` writes a dataset file from the template (datasets/aerozoo_uvolume.yaml) with one split's statistics::

    python aerozoo_pstats.py fill aerozoo_uvolume.yaml pstats_reynolds.train.json aerozoo_uvolume_reynolds.yaml

``reduce`` pools, for one split part:

* the seven targets in SI units (the two dimensionless ones as they are), from the per-case sample lines of
  ``targets`` (or of ``aerozoo_stats.py sample``, which held the first five; the same seeded 20,000 interior points
  per case; both nondimensionalize them by each case's reference scales). Those lines hold the sums of x / s and (x / s)^2 with the case's
  nondimensionalization scale s, and its scales, so the physical sums are s * sum and s^2 * sumsq: exact, with no
  re-read. Every target gets mean 0 and its RMS about 0 as the std (velocity: the isotropic one,
  sqrt(mean over components of E[u_c^2])), a scale-only standardization: the physical targets span decades
  across cases, and an offset would put a small case's whole field below the float32 / bf16 step of the offset
  (2026-10-19). The pooled mean is kept beside it as ``mean_measured``.
* ``nu_std``'s constants: the mean and std over the cases of log10 nu.
* the token standardization of the ``bc_*`` columns GeoTransolver reads (MergeBoundaries): pooled over the expected
  tokens of the four boundaries (zero where a boundary does not prescribe the field, as HarmonizeBoundaryFields
  fills it); scale-only like the targets, for the same reason (2026-10-19: a slow case's inlet k of
  ~1e-4 m^2/s^2 sat 1e-4 above a pooled-mean offset of 0.12, below bf16's step there), so every column has mean 0
  and its RMS about 0 as the std (vectors the isotropic one); an RMS of 0 (the gauge outlet pressure) is written
  as 1.0::

    python aerozoo_pstats.py reduce --manifest manifest.json --split full.train --target-rows stats/full.*.jsonl \
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
TARGETS = {  # recipe target name -> the interior solution field (the dataset files' RenameMeshFields)
    "velocity": "velocity",
    "pressure": "gauge_pressure",
    "nut": "turbulent_kinematic_viscosity",
    "tke": "turbulent_kinetic_energy",
    "omega": "specific_dissipation_rate",
    "intermittency": "intermittency",
    "retheta": "momentum_thickness_reynolds_number",
}
SAMPLE_POINTS = 20000  # aerozoo_stats.py sample's n_points and seed
SAMPLE_SEED = 0
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
    """One case's boundary line: nu and, per boundary, its face and token counts and the face means of every bc field."""
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
    """Append the boundary lines of a manifest part's case range (resumable: cases already written are skipped)."""
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


def _reference_scales(pdmsh: Path) -> dict[str, float]:
    """aerozoo_stats.py's ``scales``: the case's reference labels, for the fit only (never a model input)."""
    reference = pdmsh / TD / "global_data" / "reference"
    physics = pdmsh / TD / "global_data" / "physics"
    U = float(np.linalg.norm(np.asarray(_leaf(reference, "velocity"), np.float64)))
    rho = float(_leaf(physics, "density"))
    L = float(_leaf(reference, "length"))
    nu = float(_leaf(physics, "kinematic_viscosity"))
    speed = float(_leaf(reference, "speed"))
    return {
        "U_inf_mag": U,
        "q_inf": 0.5 * rho * U * U,
        "L_ref": L,
        "nu": nu,
        "speed": speed,
        "log_re": math.log10(speed * L / nu),
    }


def _nondim(name: str, x: np.ndarray, s: dict) -> np.ndarray:
    """aerozoo_stats.py's ``nondim``, operation for operation (the dimensionless fields as they are)."""
    x = np.asarray(x, dtype=np.float64)
    if name == "velocity":
        return x / s["U_inf_mag"]
    if name == "pressure":
        return x / s["q_inf"]
    if name == "nut":
        return x / (s["U_inf_mag"] * s["L_ref"])
    if name == "tke":
        return x / s["U_inf_mag"] ** 2
    if name == "omega":
        return x * s["L_ref"] / s["U_inf_mag"]
    return x


def target_row(case_dir: Path, case_id: str) -> dict:
    """One case's sample line, the release's rows read through the copy's permutation."""
    import hashlib
    import zlib

    pdmsh = case_dir / "domain.pdmsh"
    shuffle = json.loads((case_dir / "SHUFFLE.json").read_text())
    solution = pdmsh / TD / "interior" / TD / "point_data" / "solution"
    n_total = int(_meta(solution)["shape"][0])
    perm = np.random.default_rng(shuffle["seed"]).permutation(n_total)
    head = hashlib.sha256(perm[:4096].tobytes()).hexdigest()[:16]
    if n_total != shuffle["n_rows"] or head != shuffle["permutation_sha256_head"]:
        raise ValueError(f"{case_id}: the permutation does not match SHUFFLE.json")
    where = np.empty_like(perm)
    where[perm] = np.arange(n_total)  # the copy's row of each release row
    rng = np.random.default_rng([SAMPLE_SEED, zlib.crc32(case_id.encode())])
    idx = np.sort(rng.choice(n_total, size=min(SAMPLE_POINTS, n_total), replace=False))
    pos = where[idx]
    order = np.argsort(pos)
    s = _reference_scales(pdmsh)
    row = {
        "case": case_id,
        "n_total": n_total,
        "n": int(idx.size),
        "sum": {},
        "sumsq": {},
        "scales": s,
    }
    for name, field in TARGETS.items():
        values = np.empty((idx.size, *_leaf(solution, field).shape[1:]), np.float32)
        values[order] = _leaf(solution, field)[pos[order]]  # in the release's row order
        x = _nondim(name, values, s)
        row["sum"][name] = np.atleast_1d(x.sum(axis=0)).tolist()
        row["sumsq"][name] = np.atleast_1d((x * x).sum(axis=0)).tolist()
    return row


def cmd_targets(args) -> None:
    """Append the seven-field target sample lines of a manifest part's case range (resumable)."""
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
            f.write(json.dumps(target_row(dirs[case], case)) + "\n")
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
    """The nondimensionalization scale of a target (aerozoo_stats.py nondim): x_nd = x / scale."""
    U, L = s["U_inf_mag"], s["L_ref"]
    return {
        "velocity": U,
        "pressure": s["q_inf"],
        "nut": U * L,
        "tke": U * U,
        "omega": U / L,
        "intermittency": 1.0,
        "retheta": 1.0,
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
    """Pool one split part's target and boundary lines into its statistics JSON."""
    cases = set(json.loads(Path(args.manifest).read_text())[args.split])
    targets = _rows(args.target_rows, cases)
    bnd = _rows(args.boundary_rows, cases)
    missing = (len(cases - set(targets)), len(cases - set(bnd)))
    if any(missing):
        raise SystemExit(
            f"{args.split}: {missing[0]} cases without a target sample line, {missing[1]} without a boundary line"
        )
    stats = {"split": args.split, "n_cases": len(cases), "targets": {}, "bc": {}}
    n = sum(r["n"] for r in targets.values())
    stats["n_points"] = n
    for name in TARGETS:
        total = sum(
            _scale(name, r["scales"]) * np.asarray(r["sum"][name])
            for r in targets.values()
        )
        total_sq = sum(
            _scale(name, r["scales"]) ** 2 * np.asarray(r["sumsq"][name])
            for r in targets.values()
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
    """Write a dataset file from the template with one split's statistics."""
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
    """The command line: boundary, targets, reduce and fill."""
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
    t = sub.add_parser("targets")
    t.add_argument("root")
    t.add_argument("manifest")
    t.add_argument("split")
    t.add_argument("--out", required=True)
    t.add_argument("--start", type=int, default=0)
    t.add_argument("--stop", type=int, default=None)
    t.set_defaults(func=cmd_targets)
    r = sub.add_parser("reduce")
    r.add_argument("--manifest", required=True)
    r.add_argument("--split", required=True)
    r.add_argument("--target-rows", nargs="+", required=True)
    r.add_argument("--boundary-rows", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_reduce)
    f = sub.add_parser("fill")
    f.add_argument("template")
    f.add_argument("stats")
    f.add_argument("out")
    f.set_defaults(func=cmd_fill)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
