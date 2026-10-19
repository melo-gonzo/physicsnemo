"""Tied nearest boundary samples on the two structured solid meshes (geometry only; no model).

ISLA gives every interior prediction point the signed distance and the direction to its nearest boundary sample
(``physicsnemo.experimental.models.isla.geometry.nearest_boundary_geometry``: an exact argmin of the difference-form
squared distance, float32 on CUDA). On a structured finite-element mesh many nodes sit at equal distance from two
boundary-cell centroids, so the argmin is decided by rounding and a rigid motion of the coordinates can change it.
This script measures that on the meshes the book draws (all nodes, all boundary cells, as converted):

* PLAID Tensile2d: results/problem_figures_2026-09-22/tensile2d.npz (sample_000001, val_100) and the six specimens of
  results/sample_figures_2026-10-02/tensile2d.npz, in the plane (z dropped, as RestrictToPlane does);
* GINOT bracket lug: results/problem_figures_2026-09-22/lug.npz (lug_0004, test).

Per mesh: the share of nodes whose two nearest boundary centroids are equally near to a relative gap below
1e-7 ... 1e-3 (distances in float64 on the float32 coordinates); among the nodes tied below 1e-4, the share whose two
candidates have normals within 1 degree (collinear or coplanar neighbours) and the share whose candidate normals differ
by more than 30 degrees; the share of nodes whose float32 nearest index changes under one generic rigid motion (a fixed
rotation, then a translation by one RMS node radius, both applied in float32, as the posed scoring does) and, on those
nodes, the angle between the direction to the old and to the new nearest centroid (both in the native frame); and the
nodes that lie on the boundary itself (mesh nodes at a boundary vertex): how many are tied, how many get the outside
sign (u . n >= 0, the tangent case counting as outside) and how many change the sign of their distance under the pose.

Per node of the two problem-figure meshes (tensile2d sample_000001, lug_0004), solids_ties_2026-10-18.npz holds the
relative gap, whether the nearest index changes under the pose, and the direction change in degrees (0 where it does not).

usage: python solids_ties_2026-10-18.py   (writes solids_ties_2026-10-18.{json,npz} beside this file; CPU, under a minute)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent
GAPS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
TIE = 1e-4  # the gap below which a node counts as tied in the summary fields


def two_nearest(q, c, chunk=512):
    """Exact two nearest centroids per node, distances in float64 on the float32 coordinates."""
    q64, c64 = q.astype(np.float64), c.astype(np.float64)
    idx, dist = [], []
    for s in range(0, len(q64), chunk):
        d2 = ((q64[s:s + chunk, None, :] - c64[None, :, :]) ** 2).sum(-1)
        i = np.argpartition(d2, 1, axis=1)[:, :2]
        v = np.take_along_axis(d2, i, 1)
        o = np.argsort(v, axis=1)
        idx.append(np.take_along_axis(i, o, 1)); dist.append(np.sqrt(np.take_along_axis(v, o, 1)))
    return np.concatenate(idx), np.concatenate(dist)


def nearest32(q, c, chunk=512):
    """The float32 argmin the model computes (difference-form squared distance, ties to the smallest index)."""
    q32, c32 = q.astype(np.float32), c.astype(np.float32)
    return np.concatenate([((q32[s:s + chunk, None, :] - c32[None, :, :]) ** 2).sum(-1, dtype=np.float32).argmin(1)
                           for s in range(0, len(q32), chunk)])


def rotation(dim, seed=7):
    """One fixed generic rotation: in the plane by a fixed angle, or a seeded random SO(3) matrix."""
    rng = np.random.default_rng(seed)
    if dim == 2:
        a = np.deg2rad(37.0)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    m, r = np.linalg.qr(rng.normal(size=(3, 3)))
    m = m * np.sign(np.diag(r))
    return m if np.linalg.det(m) > 0 else -m


def angle(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def side(q, c, n, i):
    """The sign the model gives the distance: + where the direction from the nearest centroid has u . n >= 0."""
    return np.where(((q - c[i]) * n[i]).sum(-1, dtype=np.float32) >= 0, 1, -1)


def measure(q, c, n, verts):
    """Tie and pose statistics of one mesh: nodes q (Q, d), boundary centroids c (N, d), outward unit normals n (N, d),
    boundary vertices verts (V, d) (a node within 1e-6 of the coordinate scale of a vertex lies on the boundary)."""
    n = (n / np.linalg.norm(n, axis=1, keepdims=True)).astype(np.float32)
    on = np.array([np.min(((verts - x) ** 2).sum(-1)) for x in q.astype(np.float64)]) < (1e-6 * np.abs(q).max()) ** 2
    idx, dist = two_nearest(q, c)
    gap = (dist[:, 1] - dist[:, 0]) / np.maximum(dist[:, 0], 1e-30)
    tied = gap < TIE
    nang = angle(n[idx[tied, 0]], n[idx[tied, 1]])
    i0 = nearest32(q, c)
    R = rotation(q.shape[1]).astype(np.float32)
    rms = float(np.sqrt(((q.astype(np.float64) - q.mean(0)) ** 2).sum(-1).mean()))
    t = np.zeros(q.shape[1], np.float32); t[0] = rms
    qp = (q.astype(np.float32) @ R.T + t).astype(np.float32); cp = (c.astype(np.float32) @ R.T + t).astype(np.float32)
    i1 = nearest32(qp, cp)
    moved = i0 != i1
    s0 = side(q.astype(np.float32), c.astype(np.float32), n, i0); s1 = side(qp, cp, (n @ R.T).astype(np.float32), i1)
    flip = s0 != s1
    uang = angle(q[moved] - c[i0[moved]], q[moved] - c[i1[moved]]) if moved.any() else np.zeros(0)
    # the relative gap that float32 rounding of the coordinates (unit roundoff 2^-24 at the coordinate's magnitude,
    # after the translation) can reorder: about 2^-24 |x| / d over the distance d to the nearest centroid
    g32 = 2.0 ** -24 * np.abs(qp).max(1) / np.maximum(dist[:, 0], 1e-30)
    dchange = np.zeros(len(q)); dchange[moved] = uang
    NODES[id(q)] = dict(gap=gap.astype(np.float32), moved=moved, direction_change_deg=dchange.astype(np.float32),
                        sign_flip=flip, on_boundary=on)
    return dict(
        n_nodes=int(len(q)), n_boundary_cells=int(len(c)),
        tie_fraction_by_rel_gap={f"{g:.0e}": float((gap < g).mean()) for g in GAPS},
        tied_fraction=float(tied.mean()),
        tied_with_parallel_normals_fraction=float((nang < 1.0).mean()) if tied.any() else 0.0,
        tied_with_normals_over_30deg_fraction=float((nang > 30.0).mean()) if tied.any() else 0.0,
        nearest_changes_under_pose_fraction=float(moved.mean()),
        nearest_changes_under_pose_nodes=int(moved.sum()),
        direction_change_deg_median=float(np.median(uang)) if moved.any() else 0.0,
        direction_change_over_30deg_fraction_of_nodes=float((uang > 30.0).sum() / len(q)),
        direction_change_over_30deg_nodes=int((uang > 30.0).sum()),
        float32_rounding_rel_gap_median=float(np.median(g32)),
        n_on_boundary=int(on.sum()),
        tied_on_boundary_nodes=int((tied & on).sum()), tied_interior_nodes=int((tied & ~on).sum()),
        nearest_changes_on_boundary_nodes=int((moved & on).sum()),
        direction_change_over_30deg_on_boundary_nodes=int(((dchange > 30) & on).sum()),
        on_boundary_outside_sign_nodes=int((s0[on] > 0).sum()), interior_outside_sign_nodes=int((s0[~on] > 0).sum()),
        sign_flips_under_pose_nodes=int(flip.sum()), sign_flips_on_boundary_nodes=int((flip & on).sum()),
    )


def planar(points):
    return points[:, :2]


def line_mesh(bp, bc):
    """Centroids and unit in-plane normals of line cells (normals perpendicular to the segment, sign irrelevant here)."""
    a, b = bp[bc[:, 0]], bp[bc[:, 1]]
    tvec = b - a
    nrm = np.stack([tvec[:, 1], -tvec[:, 0]], 1)
    return 0.5 * (a + b), nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


out = dict(protocol=__doc__.strip(), tensile2d={}, lug={})
NODES = {}

z = np.load(RES / "problem_figures_2026-09-22" / "tensile2d.npz")
meta = json.load(open(RES / "problem_figures_2026-09-22" / "tensile2d.json"))
c, _ = line_mesh(planar(z["bnd_points"]), z["bnd_cells"])
q = planar(z["int_points"]); out["tensile2d"][meta["case"]] = measure(q, c, planar(z["bnd_normals"]), planar(z["bnd_points"])); per_node = {"tensile2d": NODES[id(q)]}

z = np.load(RES / "sample_figures_2026-10-02" / "tensile2d.npz")
meta = json.load(open(RES / "sample_figures_2026-10-02" / "tensile2d.json"))
po, co, io = z["bnd_point_offsets"], z["bnd_cell_offsets"], z["int_offsets"]
for i, case in enumerate(meta["cases"]):
    bp = planar(z["bnd_points"][po[i]:po[i + 1]]); bc = z["bnd_cells"][co[i]:co[i + 1]]
    c, n = line_mesh(bp, bc)
    # these arrays carry no normals: orient the segment normals outward by the shoelace sign of each closed polyline
    a = bp[bc[:, 0]]; b = bp[bc[:, 1]]
    n = n * (1.0 if (a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]).sum() > 0 else -1.0)
    out["tensile2d"][case["case"]] = measure(planar(z["int_points"][io[i]:io[i + 1]]), c, n, bp)

z = np.load(RES / "problem_figures_2026-09-22" / "lug.npz")
meta = json.load(open(RES / "problem_figures_2026-09-22" / "lug.json"))
tri = z["bnd_points"][z["bnd_cells"]]
q = z["int_points"]; out["lug"][meta["case"]] = measure(q, tri.mean(1), z["bnd_normals"], z["bnd_points"]); per_node["lug"] = NODES[id(q)]
np.savez_compressed(HERE / "solids_ties_2026-10-18.npz", **{f"{ds}_{k}": v for ds, d in per_node.items() for k, v in d.items()})

json.dump(out, open(HERE / "solids_ties_2026-10-18.json", "w"), indent=1)
for ds in ("tensile2d", "lug"):
    for case, r in out[ds].items():
        print(ds, case, {k: (float(f'{v:.4g}') if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)},
              {k: round(v, 4) for k, v in r["tie_fraction_by_rel_gap"].items()})
