#!/usr/bin/env python
"""Per-case standardized-target statistics over EVERY interior row of AeroZoo v1.2 cases (AEROZOO, 2026-10-11).

For each case of the given manifest parts this reads the interior solution memmaps of the row-shuffled copy
(the layout aerozoo_stats.py and aerozoo_shuffle.py document: ``domain.pdmsh/_tensordict/interior/_tensordict/
point_data/solution/<field>.memmap`` with ``meta.json`` shapes), applies the dataset yaml's chain on the targets
(NonDimensionalizeByMetadata: U / |U_inf|, p / q_inf with q_inf = rho |U_inf|^2 / 2 and p_inf = 0,
nu_t / (|U_inf| L_ref), k / |U_inf|^2, omega L_ref / |U_inf|; then NormalizeMeshFields with the split's
constants, velocity with a per-component mean and one isotropic std) and accumulates, per target and per
velocity component, over all rows in chunks:

  mean |z|, max |z|, the fraction with |z| > 1, the count with |z| > 100, and the mean Huber(z, delta 1), which
  is the loss a zero prediction would give on that channel.

The recipe's total is the channel-normalized sum, (sum over 3 velocity components + 4 scalars) / 7 (loss.py:
``_vector_loss`` sums per-component means, ``LossCalculator`` divides by ``total_channels``), written as
``huber_zero_total``. One JSON line per case, resumable, parallel over cases with a process pool. numpy only.

usage: python outliers_scan.py <root> <manifest.json> <stats.json> --parts full.train full.val full.test \
           --out scan.jsonl --workers 16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
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
DTYPES = {
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "torch.int64": np.int64,
    "torch.int32": np.int32,
}
CHUNK = 2_000_000
HUBER_DELTA = 1.0


def leaf(td_dir: Path, name: str) -> np.ndarray:
    meta = json.loads((td_dir / "meta.json").read_text())[name]
    shape, dtype = tuple(meta["shape"]), DTYPES[meta["dtype"]]
    if math.prod(shape) == 0:
        return np.zeros(shape, dtype)
    return np.memmap(td_dir / f"{name}.memmap", dtype=dtype, mode="r", shape=shape)


def scales(pdmsh: Path) -> dict[str, float]:
    reference = pdmsh / TD / "global_data" / "reference"
    physics = pdmsh / TD / "global_data" / "physics"
    U_inf = np.asarray(leaf(reference, "velocity"), dtype=np.float64)
    U = float(np.linalg.norm(U_inf))
    rho = float(leaf(physics, "density"))
    L = float(leaf(reference, "length"))
    nu = float(leaf(physics, "kinematic_viscosity"))
    speed = float(leaf(reference, "speed"))
    return {"U_inf_mag": U, "q_inf": 0.5 * rho * U * U, "L_ref": L, "nu": nu, "speed": speed, "log_re": math.log10(speed * L / nu)}


def nondim(x: np.ndarray, rule: str, s: dict[str, float]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if rule == "velocity":
        return x / s["U_inf_mag"]
    if rule == "pressure":
        return x / s["q_inf"]
    if rule == "kinematic_viscosity":
        return x / (s["U_inf_mag"] * s["L_ref"])
    if rule == "turbulent_kinetic_energy":
        return x / s["U_inf_mag"] ** 2
    if rule == "specific_dissipation_rate":
        return x * s["L_ref"] / s["U_inf_mag"]
    raise ValueError(rule)


def huber(z: np.ndarray) -> np.ndarray:
    a = np.abs(z)
    return np.where(a <= HUBER_DELTA, 0.5 * z * z, HUBER_DELTA * (a - 0.5 * HUBER_DELTA))


def case_dirs(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for shard in sorted(root.glob("shard-*")):
        for case in shard.iterdir():
            if (case / "domain.pdmsh").is_dir():
                out[case.name] = case
    return out


def scan_case(job: tuple[str, str, dict]) -> dict:
    case_id, case_dir, stats = job
    t0 = time.perf_counter()
    pdmsh = Path(case_dir) / "domain.pdmsh"
    s = scales(pdmsh)
    solution = pdmsh / TD / "interior" / TD / "point_data" / "solution"
    n_total = int(json.loads((solution / "meta.json").read_text())["shape"][0])
    row: dict = {"case": case_id, "n_total": n_total, "scales": s, "fields": {}}
    huber_sum_channels = 0.0
    for name, (field, rule) in TARGETS.items():
        mean = np.asarray(stats[name]["mean"], dtype=np.float64)
        std = float(stats[name]["std"])
        raw = leaf(solution, field)
        ncomp = raw.shape[1] if raw.ndim == 2 else 1
        abs_sum = np.zeros(ncomp)
        sq_sum = np.zeros(ncomp)
        z_sum = np.zeros(ncomp)
        abs_max = np.zeros(ncomp)
        gt1 = np.zeros(ncomp, dtype=np.int64)
        gt100 = np.zeros(ncomp, dtype=np.int64)
        nonfinite = 0
        hub = np.zeros(ncomp)
        for start in range(0, n_total, CHUNK):
            x = nondim(raw[start : start + CHUNK], rule, s)
            if x.ndim == 1:
                x = x[:, None]
            z = (x - mean) / std  # NormalizeMeshFields (eps is negligible against these stds)
            finite = np.isfinite(z)
            if not finite.all():
                nonfinite += int((~finite).any(axis=1).sum())
                z = np.where(finite, z, 0.0)
            a = np.abs(z)
            abs_sum += a.sum(axis=0)
            sq_sum += (z * z).sum(axis=0)
            z_sum += z.sum(axis=0)
            abs_max = np.maximum(abs_max, a.max(axis=0))
            gt1 += (a > 1).sum(axis=0)
            gt100 += (a > 100).sum(axis=0)
            hub += huber(z).sum(axis=0)
        entry = {
            "mean_abs_z": float(abs_sum.mean() / n_total),
            "max_abs_z": float(abs_max.max()),
            "rms_z": float(np.sqrt(sq_sum.sum() / (n_total * ncomp))),
            "mean_z": (z_sum / n_total).tolist(),
            "frac_abs_z_gt1": float(gt1.sum() / (n_total * ncomp)),
            "n_abs_z_gt100": int(gt100.sum()),
            "n_nonfinite_rows": nonfinite,
            "huber_zero_per_channel": (hub / n_total).tolist(),
        }
        if ncomp > 1:
            entry["mean_abs_z_per_component"] = (abs_sum / n_total).tolist()
            entry["max_abs_z_per_component"] = abs_max.tolist()
        huber_sum_channels += float(hub.sum() / n_total)
        row["fields"][name] = entry
    row["huber_zero_total"] = huber_sum_channels / 7.0
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    parser.add_argument("manifest")
    parser.add_argument("stats", help="the split's stats JSON (aerozoo_stats.py reduce), e.g. stats/stats_full.train.json")
    parser.add_argument("--parts", nargs="+", required=True, help="manifest keys, e.g. full.train full.val full.test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cases", nargs="*", default=None, help="restrict to these case ids (a probe)")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    stats = json.loads(Path(args.stats).read_text())["targets"]
    cases: list[str] = []
    seen: set[str] = set()
    for part in args.parts:
        for c in sorted(manifest[part]):
            if c not in seen:
                seen.add(c)
                cases.append(c)
    if args.cases:
        cases = [c for c in args.cases]
    out = Path(args.out)
    done: set[str] = set()
    if out.exists():
        done = {json.loads(line)["case"] for line in out.read_text().splitlines() if line.strip()}
    dirs = case_dirs(Path(args.root))
    missing = [c for c in cases if c not in dirs]
    if missing:
        raise SystemExit(f"{len(missing)} cases are not under {args.root}: {missing[:5]}")
    todo = [c for c in cases if c not in done]
    print(f"{len(cases)} cases over {args.parts}, {len(done)} done, {len(todo)} to go, {args.workers} workers", file=sys.stderr, flush=True)
    jobs = [(c, str(dirs[c]), stats) for c in todo]
    t0 = time.perf_counter()
    rows_total = 0
    with out.open("a") as f, Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(scan_case, jobs, chunksize=1), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows_total += row["n_total"]
            if i % 25 == 0 or i == len(jobs):
                dt = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)} cases, {rows_total / 1e9:.2f} G rows, {dt / 60:.1f} min, {rows_total * 28 / 1e9 / dt:.2f} GB/s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
