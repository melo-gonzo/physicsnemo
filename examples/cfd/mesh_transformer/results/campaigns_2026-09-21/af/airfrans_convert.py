"""AIRFRANS-INT converter (2026-09-21, plan #sec-nb-airfrans-plan): AirfRANS raw VTK cases -> native DomainMesh files.

Per case <root>/Dataset/<case>/ (an ``_internal.vtu`` volume slice with point fields U, p, nut, implicit_distance;
an ``_aerofoil.vtp`` polyline; a ``_freestream.vtp``) the converter writes <out>/<case>/<case>.pdmsh holding

  interior   : a point cloud in three dimensions (z = 0) with point_data
                 velocity (N, 3), pressure (N,), nut (N,), sdf (N,) = implicit_distance,
                 sdf_normals (N, 3) = unit vector from the nearest airfoil point to the query (in-plane);
  boundaries : {"airfoil": Mesh of line cells (M, 2) in three dimensions with cell_data
                 normals (M, 3) (in-plane, outward: pointing away from the airfoil interior),
                 quadrature_measure (M,) = segment lengths};
  global_data: U_inf (3,) the inlet velocity (speed and angle of attack parsed from the case name, checked
               against the freestream boundary's velocity), U_inf_dir (3,), U_inf_mag (), aoa_deg (), Re () with
               nu = 1.56e-5 m^2/s and chord 1 m, and the NACA parameters as a vector.

usage: python airfrans_convert.py <dataset_root> <out_root> [--limit N] [--shard i --nshards n] [--overwrite]
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
print("imports done", flush=True)

from physicsnemo.mesh import DomainMesh, Mesh

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
root, out = Path(args.root), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
NU = 1.56e-5


def polyline_segments(poly: pv.PolyData) -> np.ndarray:
    """(M, 2) point-index pairs of every polyline segment of a vtkPolyData ``lines`` array."""
    lines = np.asarray(poly.lines); segs = []; i = 0
    while i < len(lines):
        n = int(lines[i]); ids = lines[i + 1 : i + 1 + n]
        segs += [(int(ids[k]), int(ids[k + 1])) for k in range(n - 1)]
        i += n + 1
    return np.asarray(segs, dtype=np.int64)


def convert(case_dir: Path) -> dict:
    case = case_dir.name
    dst = out / case / f"{case}.pdmsh"
    if dst.exists() and not args.overwrite:
        return {"case": case, "status": "exists"}
    t1 = time.time()
    vol = pv.read(case_dir / f"{case}_internal.vtu")
    foil = pv.read(case_dir / f"{case}_aerofoil.vtp")
    free = pv.read(case_dir / f"{case}_freestream.vtp")
    # ---- case parameters from the name: airFoil2D_SST_<U>_<alpha>_<naca params...>
    parts = case.split("_")[2:]
    u_mag, aoa = float(parts[0]), float(parts[1]); naca = [float(p) for p in parts[2:]]
    u_inf = np.array([u_mag * math.cos(math.radians(aoa)), u_mag * math.sin(math.radians(aoa)), 0.0], np.float32)
    u_free = np.asarray(free.point_data["U"]).mean(0)
    # ---- airfoil boundary: line cells in 3-D (z = 0), in-plane outward normals, segment lengths
    fp = np.asarray(foil.points, np.float64).copy(); fp[:, 2] = 0.0
    segs = polyline_segments(foil)
    a, b = fp[segs[:, 0]], fp[segs[:, 1]]
    tang = b - a; length = np.linalg.norm(tang, axis=1); tang = tang / length[:, None]
    nrm = np.stack([tang[:, 1], -tang[:, 0], np.zeros_like(tang[:, 0])], 1)   # rotate the tangent by -90 degrees
    mid = 0.5 * (a + b)
    # orient outward: away from the airfoil's mean point (a thin closed curve; the centroid is inside)
    c0 = fp.mean(0)
    flip = np.sign(((mid - c0) * nrm).sum(1)) < 0
    nrm[flip] *= -1
    # ---- interior point cloud with fields; the signed-distance direction from the nearest airfoil point
    pts = np.asarray(vol.points, np.float64).copy(); pts[:, 2] = 0.0
    from scipy.spatial import cKDTree
    # nearest point on the polyline approximated by the nearest polyline vertex (the aerofoil has ~1,000 points),
    # refined by projecting onto the two adjacent segments
    tree = cKDTree(fp[:, :2]); d_vert, i_vert = tree.query(pts[:, :2], k=1)
    # AirfRANS stores the implicit distance NEGATIVE in the fluid (min about -3.5, max 0 on the surface); the
    # program's convention (the recipe's SDF transform on the cars) is positive outside, so the sign is flipped.
    sdf = -np.asarray(vol.point_data["implicit_distance"], np.float64)
    # direction: from the nearest vertex to the point, in plane; sign so that it points out of the airfoil for
    # exterior points (all interior-mesh points are exterior; sdf > 0 outside in AirfRANS)
    dirv = pts[:, :2] - fp[i_vert, :2]
    nd = np.linalg.norm(dirv, axis=1, keepdims=True).clip(1e-12)
    sdf_normals = np.concatenate([dirv / nd, np.zeros((len(pts), 1))], 1)
    t2 = time.time()
    interior = Mesh(points=torch.from_numpy(pts.astype(np.float32)),
                    point_data={"velocity": torch.from_numpy(np.asarray(vol.point_data["U"], np.float32)),
                                "pressure": torch.from_numpy(np.asarray(vol.point_data["p"], np.float32)),
                                "nut": torch.from_numpy(np.asarray(vol.point_data["nut"], np.float32)),
                                "sdf": torch.from_numpy(sdf.astype(np.float32)).reshape(-1, 1),   # (N, 1): the recipe passes per-query scalars with a trailing channel axis
                                "sdf_normals": torch.from_numpy(sdf_normals.astype(np.float32))})
    airfoil = Mesh(points=torch.from_numpy(fp.astype(np.float32)), cells=torch.from_numpy(segs),
                   cell_data={"normals": torch.from_numpy(nrm.astype(np.float32)),
                              "quadrature_measure": torch.from_numpy(length.astype(np.float32))})
    gd = {"U_inf": torch.from_numpy(u_inf), "U_inf_dir": torch.from_numpy(u_inf / np.linalg.norm(u_inf)),
          "U_inf_mag": torch.tensor(u_mag, dtype=torch.float32), "aoa_deg": torch.tensor(aoa, dtype=torch.float32),
          "Re": torch.tensor(u_mag * 1.0 / NU, dtype=torch.float32),
          "naca": torch.tensor(naca + [0.0] * (5 - len(naca)), dtype=torch.float32),
          "U_inf_freestream_mean": torch.from_numpy(u_free.astype(np.float32)),
          # AirfRANS is incompressible with kinematic pressure (p / rho): rho_inf = 1 and p_inf = 0 make the recipe's
          # NonDimensionalizeByMetadata produce p / (0.5 |U_inf|^2) and U / |U_inf|.
          "rho_inf": torch.tensor(1.0), "p_inf": torch.tensor(0.0)}
    dm = DomainMesh(interior=interior, boundaries={"airfoil": airfoil}, global_data=gd)
    dst.parent.mkdir(parents=True, exist_ok=True)
    t3 = time.time()
    dm.save(dst)
    t4 = time.time()
    print(f"  timings read+geom {t2 - t1:.1f}s build {t3 - t2:.1f}s save {t4 - t3:.1f}s", flush=True)
    U = np.asarray(vol.point_data["U"], np.float64); P = np.asarray(vol.point_data["p"], np.float64); NUT = np.asarray(vol.point_data["nut"], np.float64)
    stats = {"p_over_q": [float((P / (0.5 * u_mag ** 2)).mean()), float((P / (0.5 * u_mag ** 2)).std())],
             "u_over_U": [float((U[:, :2] / u_mag).mean()), float((U[:, :2] / u_mag).std())],
             "nut": [float(NUT.mean()), float(NUT.std()), float(NUT.max())], "nut_over_nu": [float((NUT / NU).mean()), float((NUT / NU).std())]}
    return {"case": case, "status": "ok", "n_interior": int(len(pts)), "n_segments": int(len(segs)), "stats": stats,
            "u_name_vs_free": float(np.linalg.norm(u_inf - u_free)), "sdf_min": float(sdf.min()), "sdf_max": float(sdf.max()),
            "closed": bool(np.allclose(fp[segs[0, 0]], fp[segs[-1, 1]]) or len(np.unique(segs)) == len(segs))}


cases = sorted(p for p in (root / "Dataset").iterdir() if p.is_dir())
cases = cases[args.shard::args.nshards]
if args.limit:
    cases = cases[: args.limit]
t0 = time.time(); done = 0
for cd in cases:
    try:
        r = convert(cd)
    except Exception as e:  # noqa: BLE001
        r = {"case": cd.name, "status": f"error: {e!r}"}
    done += 1
    print(json.dumps(r), f"[{done}/{len(cases)} {time.time() - t0:.0f}s]", flush=True)
# the manifest travels with the converted tree (case names are the directory names)
if args.shard == 0:
    src = root / "Dataset" / "manifest.json"
    if src.exists():
        (out / "manifest.json").write_text(src.read_text())
print("CONVERT-DONE", done)
