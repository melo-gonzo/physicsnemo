"""AF-PLANAR (DRIVE-FIX2 Part C, 2026-10-19): in-plane velocity relative L2 from infer.py's saved predictions.

usage: python planar.py <out_dir> <run_id> <shared_metrics.jsonl or ->

<out_dir>/<run_id>/predictions/*.pdmsh are infer.py's per-case DomainMeshes (pred_velocity, true_velocity in physical
units, the case's global_data). infer.py scores velocity in training space, U / |U_inf| with |U_inf| = sqrt(sum U_inf^2)
(nondim.freestream_scales), unweighted (the AirfRANS interior carries no measure), with the library relative_l2
(eps = 1e-8 floors the summed squared target), over the flattened (N, C) field; the summary is the plain mean of the
200 per-case values. This script divides the saved fields by the same |U_inf| and recomputes, in float32 as infer.py
does, velocity_l2 (all C components), velocity_{x,y,z}_l2 and the in-plane velocity_xy_l2 (components 0, 1), and checks
the recomputed per-case velocity_l2 / _x / _y against this run's metrics.jsonl and the shared scorer's metrics.jsonl.
Writes <out_dir>/planar_cases.csv and <out_dir>/planar_summary.json. Exit 3 if the check fails.
"""

import csv
import json
import sys
from pathlib import Path

import torch

from physicsnemo.mesh.domain_mesh import DomainMesh
from physicsnemo.metrics.general.relative_error import relative_l2

TOL = 1e-5  # relative; float32 sums over 3e5 elements, GPU reduction order


def rel(p, t):
    return float(relative_l2(p.flatten(), t.flatten(), dim=-1, eps=1e-8))


def rows(path):
    """sample_id -> metrics of the last infer_step row per sample; plus the infer_summary metrics."""
    out, summ = {}, None
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") == "infer_step":
            out[r["sample_id"]] = r["metrics"]
        elif r.get("phase") == "infer_summary":
            summ = r["metrics"]
    return out, summ


def main():
    out_dir, run, shared = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    pred_dir = out_dir / run / "predictions"
    new_rows, new_summ = rows(out_dir / run / "metrics.jsonl")
    sh_rows, sh_summ = rows(shared) if shared != "-" else ({}, None)
    files = sorted(pred_dir.glob("*.pdmsh"))
    recs = []
    for f in files:
        sid = f.name[: -len(".pdmsh")]
        dm = DomainMesh.load(str(f))
        pd = dm.interior.point_data
        U_inf = dm.global_data["U_inf"].float()
        umag = (U_inf * U_inf).sum().sqrt()
        p = pd["pred_velocity"].float() / umag
        t = pd["true_velocity"].float() / umag
        C = p.shape[-1]
        r = {"sample_id": sid, "n_points": p.shape[0], "n_comp": C, "U_inf_mag": float(umag)}
        r["velocity_l2"] = rel(p, t)
        r["velocity_xy_l2"] = rel(p[:, :2], t[:, :2])
        for i, c in enumerate("xyz"[:C]):
            r[f"velocity_{c}_l2"] = rel(p[:, i], t[:, i])
        d64, t64 = (p - t).double(), t.double()
        e2 = (d64 * d64).sum(0)
        r["err2_share_z"] = float(e2[2] / e2.sum()) if C == 3 else 0.0
        r["true_z_rms"] = float(t64[:, 2].pow(2).mean().sqrt()) if C == 3 else 0.0
        r["true_z_absmax"] = float(t64[:, 2].abs().max()) if C == 3 else 0.0
        r["pred_z_rms"] = float(p.double()[:, 2].pow(2).mean().sqrt()) if C == 3 else 0.0
        for tag, src in (("new", new_rows), ("shared", sh_rows)):
            m = src.get(sid)
            r[f"velocity_l2_{tag}"] = m["velocity_l2"] if m else None
            r[f"reldiff_{tag}"] = (
                max(abs(r[k] - m[k]) / abs(m[k]) for k in ("velocity_l2", "velocity_x_l2", "velocity_y_l2"))
                if m else None
            )
        recs.append(r)
    with open(out_dir / "planar_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    n = len(recs)
    mean = lambda k: sum(r[k] for r in recs) / n  # noqa: E731
    keys = ["velocity_l2", "velocity_xy_l2", "velocity_x_l2", "velocity_y_l2"] + (
        ["velocity_z_l2", "err2_share_z", "true_z_rms", "pred_z_rms"] if recs[0]["n_comp"] == 3 else []
    )
    d_new = [r["reldiff_new"] for r in recs if r["reldiff_new"] is not None]
    d_sh = [r["reldiff_shared"] for r in recs if r["reldiff_shared"] is not None]
    summary = {
        "run": run,
        "n_cases": n,
        "n_matched_new": len(d_new),
        "n_matched_shared": len(d_sh),
        "mean": {k: mean(k) for k in keys},
        "true_z_absmax_max": max(r["true_z_absmax"] for r in recs),
        "max_reldiff_vs_new_rows": max(d_new) if d_new else None,
        "max_reldiff_vs_shared_rows": max(d_sh) if d_sh else None,
        "new_summary_velocity_l2": (new_summ or {}).get("velocity_l2"),
        "shared_summary_velocity_l2": (sh_summ or {}).get("velocity_l2"),
        "tol": TOL,
    }
    ok = n == 200 and len(d_new) == n and max(d_new) < TOL and (shared == "-" or (len(d_sh) == n and max(d_sh) < TOL))
    summary["check_ok"] = bool(ok)
    (out_dir / "planar_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
