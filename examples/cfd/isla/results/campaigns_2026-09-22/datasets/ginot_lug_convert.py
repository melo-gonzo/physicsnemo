"""GINOT PLASTIC_LUG converter (2026-09-22): Zenodo 15293036 ``PLASTIC_LUG.zip`` -> native DomainMesh files.

Raw layout (CC BY 4.0; Liu, Koric et al., "Geometry-Informed Neural Operator Transformer", data_scripts/PLASTIC_LUG/
data_process.py): ``LUG_node_S_PC.pkl`` = {"vertices": 3000 x (N_i, 3) float32 [m], "nodal_stress": 3000 x (N_i,) float32
von Mises stress [MPa] (1e-6 x the Abaqus Pa values), "points_cloud": 3000 x (B_i, 3) float32 = the nodes of the surface
extracted from the hex mesh}; ``LUG_cells.pkl`` = 3000 x (C_i, 9) int64 VTK hex8 rows [8, i0..i7] (the corner nodes of the
quadratic hexahedra, mid-edge nodes dropped); ``input_params.npy`` (3000, 4) float64 = [hole radius, length, thickness,
pressure] in [m, m, m, Pa]. Node counts 5k to 15k (mean 9.5k), surface nodes 2.2k to 4.3k. The lug lies along +x: the flat
end face x = x_min (the full thickness x width rectangle) is the clamped end; the far end is a semicircular nose around the
hole axis (parallel to z, z in [0, thickness]); the lug is 0.05 m wide (y in [-0.01, 0.04], hole axis at y = 0.015).
No split indices are published (the paper holds out 20%); this converter draws a seed-0 permutation: test = 600,
train = 2400, with nested train_n300 / train_n600 / train_n1200 ladders inside train.

Per sample the converter writes <out>/lug_<id>/lug_<id>.pdmsh with

  interior   : all mesh nodes as a point cloud with point_data von_mises (N,) [MPa], sdf (N, 1) = -(distance to the
                 surface) (positive outside the solid; nodes are inside or on it), sdf_normals (N, 3) = grad sdf (toward the
                 nearest surface point; the outward face normal for surface nodes);
  boundaries : {"surface": Mesh of triangles (M, 3) = the boundary quads of the hex mesh (faces owned by one hex) split
                 along a diagonal, cell_data normals (M, 3) outward (away from the owning hex's centroid), quadrature_measure
                 (M,) triangle areas, bc_fixed (M,) 1 on the clamped end face, bc_hole (M,) 1 on the hole's cylindrical
                 surface, bc_loaded (M,) 1 on the pressure-loaded part of the hole (LOADED_HALF below), bc_pressure (M,) =
                 pressure [MPa] on loaded cells else 0, traction (M, 3) = -pressure n [MPa] on loaded cells else 0};
  global_data: hole_radius, length, thickness [m], pressure [Pa], pressure_mpa [MPa], params (4,) = the raw
               input_params row, sample_id ().

usage: python ginot_lug_convert.py <raw_dir> <out_root> [--limit N] [--shard i --nshards n] [--overwrite]
       (the manifest and pooled statistics are written by shard 0 from the per-shard results once all shards finished:
        python ginot_lug_convert.py <raw_dir> <out_root> --manifest-only)
"""
from __future__ import annotations

import argparse, json, pickle, time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
print("imports done", flush=True)

from physicsnemo.mesh import DomainMesh, Mesh

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--manifest-only", action="store_true")
args = ap.parse_args()
raw, out = Path(args.raw), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# Which part of the hole carries the pressure. The same 3,000-design dataset is described in Geom-DeepONet (He, Koric et
# al. 2024, arXiv 2403.14788, "parameterized beam with a circular hole": length 80-120 mm, thickness 15-30 mm, hole radius
# 10-15 mm, pressure 50-100 MPa; E = 200 GPa, nu = 0.3, sigma_y = 380 MPa, H = 571.4 MPa): "The left end of the beam is
# held fixed. A pressure load is applied to the bottom half of the circular hole on the right end." Lower = y < hole axis.
LOADED_HALF = "lower"
HEX_FACES = ([0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])
N_TOTAL = 3000; N_TEST = 600


def splits() -> dict[str, list[int]]:
    perm = np.random.default_rng(0).permutation(N_TOTAL)
    test = sorted(perm[:N_TEST].tolist()); train = perm[N_TEST:]
    s = {"train": sorted(train.tolist()), "test": test}
    for n in (300, 600, 1200):
        s[f"train_n{n}"] = sorted(train[:n].tolist())     # nested: prefixes of the same permutation
    return s


def point_triangle_distance(p, a, b, c):
    """Exact distance from points p (K, 3) to triangles (a, b, c) (K, 3) each (Ericson, Real-Time Collision Detection)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = p - b; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cp = p - c; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    q = a.copy()
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    denom = 1.0 / np.where(np.abs(va + vb + vc) < 1e-30, 1e-30, va + vb + vc)
    v = vb * denom; w = vc * denom
    q = a + ab * v[:, None] + ac * w[:, None]                                      # face region
    m = (d1 <= 0) & (d2 <= 0); q[m] = a[m]                                           # vertex a
    m = (d3 >= 0) & (d4 <= d3); q[m] = b[m]                                          # vertex b
    m = (d6 >= 0) & (d5 <= d6); q[m] = c[m]                                          # vertex c
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0); t = (d1 / (d1 - d3 + 1e-30)); q[m] = (a + ab * t[:, None])[m]    # edge ab
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0); t = (d2 / (d2 - d6 + 1e-30)); q[m] = (a + ac * t[:, None])[m]    # edge ac
    m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0); t = (d4 - d3) / ((d4 - d3) + (d5 - d6) + 1e-30); q[m] = (b + (c - b) * t[:, None])[m]  # edge bc
    return np.linalg.norm(p - q, axis=1), q


def convert(sid: int, v: np.ndarray, s: np.ndarray, hexes: np.ndarray, prm: np.ndarray) -> dict:
    case = f"lug_{sid:04d}"
    dst = out / case / f"{case}.pdmsh"
    if dst.exists() and not args.overwrite:
        return {"case": case, "status": "exists"}
    t1 = time.time()
    v = v.astype(np.float64); n = len(v)
    hole_r, length, thick, pressure = (float(x) for x in prm)
    # ---- boundary quads: hex faces owned by exactly one hex; orient outward (away from the owning hex centroid)
    f = np.concatenate([hexes[:, q] for q in HEX_FACES]); owner = np.tile(np.arange(len(hexes)), 6)
    key = np.sort(f, 1); _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    keep = cnt[inv.ravel()] == 1; quads = f[keep]; owner = owner[keep]
    hc = v[hexes].mean(1)[owner]
    # split each quad along the 0-2 diagonal
    tri = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]); tri_owner_c = np.concatenate([hc, hc])
    A, B, C = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    cr = np.cross(B - A, C - A); area = 0.5 * np.linalg.norm(cr, axis=1); nrm = cr / (2 * area[:, None]).clip(1e-30)
    cen = (A + B + C) / 3
    flip = ((cen - tri_owner_c) * nrm).sum(1) < 0
    nrm[flip] *= -1; tri[flip] = tri[flip][:, [0, 2, 1]]
    # ---- BC labels from the geometry: clamped end face x = x_min; hole = faces whose centroid is at radius ~ hole_r from the
    # hole axis and whose outward normal points toward the axis
    xmin = v[:, 0].min(); eps = 1e-6
    bc_fixed = (np.abs(cen[:, 0] - xmin) < 1e-5).astype(np.float32)
    # hole axis: the "length" parameter is the distance from the clamped face to the hole axis (checked on the raw data:
    # x_min + length = x_max - R_nose with the nose a semicircle of the lug's half-width about the axis), the axis is at
    # mid-width; hole faces sit at radius hole_r from it with outward normals pointing toward the axis
    xa = xmin + length; ya = 0.5 * (v[:, 1].min() + v[:, 1].max())
    rr = np.hypot(cen[:, 0] - xa, cen[:, 1] - ya)
    inward = ((cen[:, :2] - np.array([xa, ya])) * nrm[:, :2]).sum(1) < 0
    curved = np.abs(nrm[:, 2]) < 0.5
    bc_hole = (curved & inward & (np.abs(rr - hole_r) < 0.1 * hole_r)).astype(np.float32)
    if bc_hole.sum() < 8:
        raise RuntimeError(f"hole detection failed: {int(bc_hole.sum())} faces")
    half = (cen[:, 1] < ya) if LOADED_HALF == "lower" else (cen[:, 1] > ya)
    bc_loaded = (bc_hole * half).astype(np.float32)
    p_mpa = pressure * 1e-6
    bc_pressure = bc_loaded * np.float32(p_mpa)
    traction = -p_mpa * nrm * bc_loaded[:, None]
    # ---- signed distance of every node to the surface: exact point-triangle distance over the k nearest triangles
    bnodes = np.unique(tri)
    on_bnd = np.zeros(n, bool); on_bnd[bnodes] = True
    tree = cKDTree(cen); k = 12
    _, nb = tree.query(v, k=k)
    P = np.repeat(v, k, 0); T = tri[nb.ravel()]
    d, q = point_triangle_distance(P, v[T[:, 0]], v[T[:, 1]], v[T[:, 2]])
    d = d.reshape(n, k); q = q.reshape(n, k, 3); j = d.argmin(1)
    dist = d[np.arange(n), j]; near = q[np.arange(n), j]
    sdf = -dist; sdf[on_bnd] = 0.0
    g = (near - v) / np.linalg.norm(near - v, axis=1, keepdims=True).clip(1e-30)
    vn = np.zeros((n, 3)); np.add.at(vn, tri[:, 0], nrm * area[:, None]); np.add.at(vn, tri[:, 1], nrm * area[:, None]); np.add.at(vn, tri[:, 2], nrm * area[:, None])
    vn = vn / np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-30)
    g[on_bnd] = vn[on_bnd]
    f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    interior = Mesh(points=f32(v), point_data={"von_mises": f32(s), "sdf": f32(sdf).reshape(-1, 1), "sdf_normals": f32(g)})
    binv = np.searchsorted(bnodes, tri)
    surface = Mesh(points=f32(v[bnodes]), cells=torch.from_numpy(binv.astype(np.int64)),
                   cell_data={"normals": f32(nrm), "quadrature_measure": f32(area), "bc_fixed": f32(bc_fixed), "bc_hole": f32(bc_hole),
                              "bc_loaded": f32(bc_loaded), "bc_pressure": f32(bc_pressure), "traction": f32(traction)})
    gd = {"hole_radius": torch.tensor(hole_r), "length": torch.tensor(length), "thickness": torch.tensor(thick),
          "pressure": torch.tensor(pressure), "pressure_mpa": torch.tensor(p_mpa), "params": f32(prm), "sample_id": torch.tensor(float(sid))}
    dm = DomainMesh(interior=interior, boundaries={"surface": surface}, global_data=gd)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dm.save(dst)
    # watertight check: every triangle edge is shared by exactly two triangles
    ek = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    closed = bool(np.all(np.unique(ek, axis=0, return_counts=True)[1] == 2))
    return {"case": case, "status": "ok", "n_nodes": int(n), "n_hex": int(len(hexes)), "n_tri": int(len(tri)), "n_bnodes": int(len(bnodes)),
            "n_fixed": int(bc_fixed.sum()), "n_hole": int(bc_hole.sum()), "n_loaded": int(bc_loaded.sum()),
            "area_total": float(area.sum()), "area_fixed": float((area * bc_fixed).sum()), "area_hole": float((area * bc_hole).sum()),
            "area_hole_expected": float(2 * np.pi * hole_r * thick), "area_fixed_expected": float(thick * (v[:, 1].max() - v[:, 1].min())),
            "hole_axis": [float(xa), float(ya)], "fit_r_spread": [float(rr[bc_hole > 0].min()), float(rr[bc_hole > 0].max())],
            "sdf_min": float(sdf.min()), "closed": closed,
            "params": [hole_r, length, thick, pressure], "stress": [float(s.mean()), float(s.max()), float(np.median(s))],
            "sums": {"von_mises": [float(s.sum()), float((s.astype(np.float64) ** 2).sum()), int(n)]}, "t": time.time() - t1}


def write_manifest():
    results = []
    for p in sorted(out.glob("convert_results_shard*.json")):
        results += json.loads(p.read_text())
    byid = {r["case"]: r for r in results if r.get("status") == "ok"}
    sp = splits(); name = lambda i: f"lug_{i:04d}"
    manifest = {k: [name(i) for i in v] for k, v in sp.items()}
    stats = {}
    for split in ("train", "train_n1200"):
        acc = np.zeros(3); prm = []; nn = []; miss = 0
        for c in manifest[split]:
            r = byid.get(c)
            if r is None:
                miss += 1; continue
            acc += r["sums"]["von_mises"]; prm.append(r["params"]); nn.append(r["n_nodes"])
        m = acc[0] / acc[2]
        prm = np.array(prm)
        st = {"von_mises": {"mean": m, "std": float(np.sqrt(max(acc[1] / acc[2] - m * m, 0)))}}
        for j, k in enumerate(["hole_radius", "length", "thickness", "pressure"]):
            st[k] = {"mean": float(prm[:, j].mean()), "std": float(prm[:, j].std()), "min": float(prm[:, j].min()), "max": float(prm[:, j].max())}
        st["pressure_mpa"] = {k: v * 1e-6 for k, v in st["pressure"].items()}
        st["_n_cases"] = len(nn); st["_missing"] = miss; st["_n_nodes_total"] = int(sum(nn)); st["_n_nodes_mean"] = float(np.mean(nn))
        stats[split] = st
    manifest["_stats"] = stats
    manifest["_note"] = ("No official split indices are published for PLASTIC_LUG (GINOT holds out 20%); train/test = seed-0 permutation "
                         "2400/600; train_n* are nested prefixes of the training permutation. Stress in MPa, pressure in Pa (pressure_mpa in MPa), "
                         f"lengths in m. bc_loaded = {LOADED_HALF} half of the hole (y {'<' if LOADED_HALF == 'lower' else '>'} axis); bc_fixed = end face x = x_min.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("MANIFEST-DONE", len(results), "ok", len(byid), json.dumps(stats, indent=1))


if args.manifest_only:
    write_manifest(); raise SystemExit

t0 = time.time()
data = pickle.load(open(raw / "LUG_node_S_PC.pkl", "rb"))
cells = pickle.load(open(raw / "LUG_cells.pkl", "rb"))
params = np.load(raw / "input_params.npy")
print(f"loaded raw in {time.time() - t0:.0f}s", flush=True)
ids = list(range(N_TOTAL))[args.shard::args.nshards]
if args.limit:
    ids = ids[: args.limit]
results = []; done = 0
for sid in ids:
    try:
        r = convert(sid, data["vertices"][sid], data["nodal_stress"][sid], np.asarray(cells[sid])[:, 1:], params[sid])
    except Exception as e:  # noqa: BLE001
        r = {"case": f"lug_{sid:04d}", "status": f"error: {e!r}"}
    results.append(r); done += 1
    print(json.dumps({k: v for k, v in r.items() if k != "sums"}), f"[{done}/{len(ids)} {time.time() - t0:.0f}s]", flush=True)
(out / f"convert_results_shard{args.shard}.json").write_text(json.dumps(results))
print("CONVERT-DONE", done)
if args.nshards == 1:
    write_manifest()
