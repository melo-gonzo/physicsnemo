"""PLAID Tensile2d converter (2026-09-22): Hugging Face ``PLAID-datasets/Tensile2d`` parquet shards -> native DomainMesh files.

The dataset (Safran, CC-BY-SA-4.0; 2-D quasi-static elasto-viscoplastic tension, plane strain, small deformations) ships
702 samples as one binary column ``sample``: a pickled dict ``{"meshes": {0.0: <CGNS tree>}, "scalars": {...}}`` where the
CGNS tree is nested ``[name, value, children, label]`` lists. Reading it needs only pyarrow + numpy (plaid-lib is not
required). Per sample: an unstructured TRI_3 mesh (about 6k to 10k nodes, coordinates in [-1, 1]), nodal fields U1, U2
(displacement), q (cumulated plastic strain), sig11, sig22, sig12 (stress) on the 500 training samples ONLY (the 200 test
and 2 OOD samples ship geometry and input scalars but no output fields or output scalars), input scalars P (pressure on
the top edge; negative = tension) and p1..p5 (material parameters), output scalars max_von_mises, max_q, max_U2_top,
max_sig22_top, and nodal BC tags ``Top`` (loaded edge), ``Bottom`` (edge with U2 = 0) and ``BottomLeft`` (one node, U1 = 0).
Row index in the concatenated shards is the sample id used by the README's split lists.

Per sample the converter writes <out>/sample_<id>/sample_<id>.pdmsh (withheld samples: <out>/withheld/sample_<id>/...) with

  interior   : all mesh nodes as a point cloud embedded in 3-D (z = 0, the AirfRANS convention) with point_data
                 displacement (N, 3) = (U1, U2, 0), q (N,), sig11 (N,), sig22 (N,), sig12 (N,) [training samples],
                 sdf (N, 1) = -(distance to the outer boundary) (positive outside the solid; nodes are inside or on it),
                 sdf_normals (N, 3) = grad sdf (direction toward the nearest boundary point; the outward boundary normal
                 for nodes on the boundary), tag_top / tag_bottom / tag_bottomleft (N,) 0/1 nodal BC tags;
  boundaries : {"outer": Mesh of line cells (M, 2) = every triangle edge owned by exactly one triangle (the closed outer
                 polyline including any hole loops), cell_data normals (M, 3) outward (away from the owning triangle),
                 quadrature_measure (M,) segment lengths, bc_loaded (M,) 1 on Top-edge cells, bc_fixed (M,) 1 on
                 Bottom-edge cells, bc_pinned (M,) 1 on cells touching the BottomLeft node, bc_pressure (M,) = P on loaded
                 cells else 0, traction (M, 3) = -P n on loaded cells else 0 (so sig22 = -P on the top edge);
                 point_data tag_top / tag_bottom / tag_bottomleft (0/1 per boundary node)};
  global_data: P, p1..p5 () and params (6,) = [P, p1, ..., p5]; has_targets (); on training samples also
               max_von_mises, max_q, max_U2_top, max_sig22_top.

Units are the dataset's (MPa-scale stresses, dimensionless coordinates). The manifest maps the README's split names
(train_8 ... train_500, test, OOD) to case names and adds train_400 / val_100 (val_100 = 100 samples drawn with seed 0
from train_500 minus train_250, so the official nested ladder up to train_250 stays disjoint from val_100), plus
per-field mean/std over train_400 and train_500 for normalization.

usage: python plaid_tensile2d_convert.py <raw_dir> <out_root> [--limit N] [--overwrite]
"""
from __future__ import annotations

import argparse, json, re, time, pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
print("imports done", flush=True)

from physicsnemo.mesh import DomainMesh, Mesh

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
raw, out = Path(args.raw), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

IN_SCALARS = ["P", "p1", "p2", "p3", "p4", "p5"]
OUT_SCALARS = ["max_von_mises", "max_q", "max_U2_top", "max_sig22_top"]
FIELDS = ["U1", "U2", "q", "sig11", "sig22", "sig12"]


def cgns_child(node, name):
    for c in node[2]:
        if c[0] == name:
            return c
    raise KeyError(name)


def cgns_children(node, label):
    return [c for c in node[2] if c[3] == label]


def parse_sample(blob: bytes) -> dict:
    obj = pickle.loads(blob)
    tree = obj["meshes"][0.0]
    base = cgns_children(tree, "CGNSBase_t")[0]
    zone = cgns_children(base, "Zone_t")[0]
    gc = cgns_child(zone, "GridCoordinates")
    x = np.asarray(cgns_child(gc, "CoordinateX")[1], np.float64); y = np.asarray(cgns_child(gc, "CoordinateY")[1], np.float64)
    elems = cgns_children(zone, "Elements_t")
    assert len(elems) == 1 and int(elems[0][1][0]) == 5, [(e[0], e[1]) for e in elems]   # CGNS TRI_3 = 5
    tri = np.asarray(cgns_child(elems[0], "ElementConnectivity")[1], np.int64).reshape(-1, 3) - 1   # 1-based -> 0-based
    fields = {}
    for fs in cgns_children(zone, "FlowSolution_t"):
        if fs[0] == "PointData":
            for c in fs[2]:
                if c[3] == "DataArray_t":
                    fields[c[0]] = np.asarray(c[1], np.float64)
    tags = {}
    for bc in cgns_children(cgns_child(zone, "ZoneBC"), "BC_t"):
        tags[bc[0]] = np.asarray(cgns_child(bc, "PointList")[1], np.int64).ravel() - 1
    scalars = {str(k): float(v) for k, v in obj["scalars"].items()}
    return {"x": x, "y": y, "tri": tri, "fields": fields, "tags": tags, "scalars": scalars}


def boundary_edges(tri: np.ndarray):
    """(M, 2) edges owned by exactly one triangle and (M,) index of the opposite vertex in that triangle."""
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], 0)
    opp = np.concatenate([tri[:, 2], tri[:, 0], tri[:, 1]], 0)
    key = np.sort(e, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    keep = cnt[inv.ravel()] == 1
    return e[keep], opp[keep]


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """For points p (N, 2) and segments a->b (M, 2): (N,) min distance, (N, 2) nearest point, (N,) segment index."""
    ab = b - a; L2 = (ab ** 2).sum(1).clip(1e-30)
    dmin = np.full(len(p), np.inf); near = np.zeros((len(p), 2)); seg = np.zeros(len(p), np.int64)
    for s in range(0, len(p), 2048):   # chunk queries: (chunk, M, 2) intermediates
        q = p[s : s + 2048]
        t = (((q[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None]).clip(0.0, 1.0)
        c = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(q[:, None, :] - c, axis=2)
        j = d.argmin(1)
        dmin[s : s + 2048] = d[np.arange(len(q)), j]; near[s : s + 2048] = c[np.arange(len(q)), j]; seg[s : s + 2048] = j
    return dmin, near, seg


def convert(sid: int, blob: bytes) -> dict:
    case = f"sample_{sid:06d}"
    S = parse_sample(blob)
    has_targets = all(f in S["fields"] for f in FIELDS)
    dst = (out if has_targets else out / "withheld") / case / f"{case}.pdmsh"
    if dst.exists() and not args.overwrite:
        return {"case": case, "status": "exists", "has_targets": has_targets}
    t1 = time.time()
    xy = np.stack([S["x"], S["y"]], 1); n = len(xy)
    tri = S["tri"]
    edges, opp = boundary_edges(tri)
    a, b = xy[edges[:, 0]], xy[edges[:, 1]]
    tang = b - a; length = np.linalg.norm(tang, axis=1); tang = tang / length[:, None]
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
    mid = 0.5 * (a + b)
    flip = ((mid - xy[opp]) * nrm).sum(1) < 0          # outward = away from the owning triangle's opposite vertex
    nrm[flip] *= -1
    # ---- nodal tags (0/1) and per-cell BC labels
    tag = {k: np.zeros(n, np.float32) for k in ("Top", "Bottom", "BottomLeft")}
    for k in tag:
        tag[k][S["tags"][k]] = 1.0
    both = lambda k: (tag[k][edges[:, 0]] > 0) & (tag[k][edges[:, 1]] > 0)
    any_ = lambda k: (tag[k][edges[:, 0]] > 0) | (tag[k][edges[:, 1]] > 0)
    bc_loaded = both("Top").astype(np.float32); bc_fixed = both("Bottom").astype(np.float32); bc_pinned = any_("BottomLeft").astype(np.float32)
    P = S["scalars"]["P"]
    bc_pressure = bc_loaded * np.float32(P)
    traction2 = -P * nrm * bc_loaded[:, None]           # pressure P on the top edge: sigma n = -P n
    # ---- signed distance of every node to the outer boundary (nodes are inside or on the solid: sdf <= 0)
    dist, near, seg = point_segment_distance(xy, a, b)
    sdf = -dist
    # vertex-averaged outward normals for nodes that lie on the boundary (grad sdf is undefined there)
    vn = np.zeros((n, 2)); np.add.at(vn, edges[:, 0], nrm); np.add.at(vn, edges[:, 1], nrm)
    vnn = np.linalg.norm(vn, axis=1, keepdims=True)
    on_bnd = dist < 1e-9
    dirv = near - xy; dn = np.linalg.norm(dirv, axis=1, keepdims=True).clip(1e-30)
    g = dirv / dn                                        # grad(sdf) for interior nodes: toward the nearest boundary point
    g[on_bnd] = (vn / vnn.clip(1e-30))[on_bnd]
    isolated = on_bnd & (vnn[:, 0] < 1e-12)              # boundary node without incident boundary edge (should not happen)
    g[isolated] = nrm[seg[isolated]]
    to3 = lambda v: np.concatenate([v, np.zeros((len(v), 1))], 1).astype(np.float32)
    pd = {"sdf": torch.from_numpy(sdf.astype(np.float32)).reshape(-1, 1), "sdf_normals": torch.from_numpy(to3(g)),
          "tag_top": torch.from_numpy(tag["Top"]), "tag_bottom": torch.from_numpy(tag["Bottom"]), "tag_bottomleft": torch.from_numpy(tag["BottomLeft"])}
    if has_targets:
        F = S["fields"]
        pd["displacement"] = torch.from_numpy(to3(np.stack([F["U1"], F["U2"]], 1)))
        for k in ("q", "sig11", "sig22", "sig12"):
            pd[k] = torch.from_numpy(F[k].astype(np.float32))
    interior = Mesh(points=torch.from_numpy(to3(xy)), point_data=pd)
    # boundary mesh: compact the node set so its points are only the boundary nodes
    bnodes, inv = np.unique(edges, return_inverse=True); inv = inv.reshape(edges.shape)
    outer = Mesh(points=torch.from_numpy(to3(xy[bnodes])), cells=torch.from_numpy(inv.astype(np.int64)),
                 cell_data={"normals": torch.from_numpy(to3(nrm)), "quadrature_measure": torch.from_numpy(length.astype(np.float32)),
                            "bc_loaded": torch.from_numpy(bc_loaded), "bc_fixed": torch.from_numpy(bc_fixed), "bc_pinned": torch.from_numpy(bc_pinned),
                            "bc_pressure": torch.from_numpy(bc_pressure.astype(np.float32)), "traction": torch.from_numpy(to3(traction2))},
                 point_data={f"tag_{k.lower()}": torch.from_numpy(tag[k][bnodes]) for k in tag})
    gd = {k: torch.tensor(S["scalars"][k], dtype=torch.float32) for k in IN_SCALARS}
    gd["params"] = torch.tensor([S["scalars"][k] for k in IN_SCALARS], dtype=torch.float32)
    gd["has_targets"] = torch.tensor(float(has_targets))
    for k in OUT_SCALARS:
        if k in S["scalars"]:
            gd[k] = torch.tensor(S["scalars"][k], dtype=torch.float32)
    dm = DomainMesh(interior=interior, boundaries={"outer": outer}, global_data=gd)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dm.save(dst)
    # ---- checks and per-case statistics (sums for pooled mean/std over splits)
    deg = np.bincount(edges.ravel(), minlength=n)[bnodes]
    closed = bool(np.all(deg == 2))   # closed loops: every boundary node has exactly two incident boundary edges
    r = {"case": case, "status": "ok", "has_targets": has_targets, "n_nodes": int(n), "n_tri": int(len(tri)), "n_bedges": int(len(edges)),
         "n_loaded": int(bc_loaded.sum()), "n_fixed": int(bc_fixed.sum()), "n_pinned": int(bc_pinned.sum()), "closed": closed,
         "perimeter": float(length.sum()), "sdf_min": float(sdf.min()), "P": P, "scalars": S["scalars"],
         "bbox": [float(xy[:, 0].min()), float(xy[:, 0].max()), float(xy[:, 1].min()), float(xy[:, 1].max())],
         "loaded_len": float((length * bc_loaded).sum()), "fixed_len": float((length * bc_fixed).sum()), "t": time.time() - t1}
    if has_targets:
        F = S["fields"]
        r["sums"] = {k: [float(F[k].sum()), float((F[k] ** 2).sum()), int(n)] for k in FIELDS}
        r["bc_check"] = {"U2_bottom_absmax": float(np.abs(F["U2"][tag["Bottom"] > 0]).max()),
                         "U1_bottomleft_absmax": float(np.abs(F["U1"][tag["BottomLeft"] > 0]).max()),
                         "sig22_top_mean": float(F["sig22"][tag["Top"] > 0].mean())}
    return r


shards = sorted(raw.glob("all_samples-*.parquet"))
splits_txt = re.search(r"    split:\n(.*?)\n    task:", (raw / "README.md").read_text(), re.S).group(1)
splits: dict[str, list[int]] = {}; cur = None
for line in splits_txt.splitlines():
    s = line.strip()
    if s.endswith(":"):
        cur = s[:-1]; splits[cur] = []
    elif s.startswith("- "):
        splits[cur].append(int(s[2:]))
rng = np.random.default_rng(0)
pool = np.array(sorted(set(splits["train_500"]) - set(splits["train_250"])))
val100 = sorted(rng.permutation(pool)[:100].tolist())
splits["val_100"] = val100; splits["train_400"] = sorted(set(splits["train_500"]) - set(val100))
assert len(splits["train_400"]) == 400 and not set(splits["train_400"]) & set(val100)

results = []; sid = 0; t0 = time.time(); done = 0
for sh in shards:
    pf = pq.ParquetFile(sh)
    for batch in pf.iter_batches(batch_size=4, columns=["sample"]):   # small batches: a whole row group (~120 MB) trips the login node's memory cap
        col = batch.column("sample")
        for i in range(len(col)):
            if args.limit and done >= args.limit:
                break
            try:
                r = convert(sid, col[i].as_py())
            except Exception as e:  # noqa: BLE001
                r = {"case": f"sample_{sid:06d}", "status": f"error: {e!r}"}
            results.append(r); done += 1
            print(json.dumps({k: v for k, v in r.items() if k not in ("sums", "scalars")}), f"[{done} {time.time() - t0:.0f}s]", flush=True)
            sid += 1
print("n_rows", sid, "expected 702")

# ---- manifest: split -> case names; pooled per-field mean/std over the training splits
name = lambda i: f"sample_{i:06d}"
manifest = {k: [name(i) for i in v] for k, v in splits.items()}
byid = {r["case"]: r for r in results if r.get("status") == "ok"}
stats = {}
for split in ("train_400", "train_500"):
    acc = {k: np.zeros(3) for k in FIELDS}; sc = {k: [] for k in IN_SCALARS + OUT_SCALARS}; nn = []
    for c in manifest[split]:
        r = byid.get(c)
        if r is None or not r["has_targets"]:
            continue
        for k in FIELDS:
            acc[k] += r["sums"][k]
        for k in sc:
            sc[k].append(r["scalars"][k])
        nn.append(r["n_nodes"])
    st = {}
    for k in FIELDS:
        s, s2, cnt = acc[k]; m = s / cnt; st[k] = {"mean": m, "std": float(np.sqrt(max(s2 / cnt - m * m, 0.0)))}
    u = np.stack([acc["U1"], acc["U2"]]); m = u[:, 0] / u[:, 2]
    st["displacement"] = {"mean": [float(m[0]), float(m[1]), 0.0], "std": [float(np.sqrt(max(u[i, 1] / u[i, 2] - m[i] ** 2, 0))) for i in range(2)] + [0.0],
                          "std_scalar": float(np.sqrt(((u[:, 1] / u[:, 2] - m ** 2).sum()) / 2))}
    for k in sc:
        v = np.array(sc[k]); st[k] = {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}
    st["_n_cases"] = len(nn); st["_n_nodes_total"] = int(sum(nn)); st["_n_nodes_mean"] = float(np.mean(nn)) if nn else None
    stats[split] = st
manifest["_stats"] = stats
manifest["_note"] = ("Official PLAID splits (train_8..train_500 nested; test and OOD have no output fields, stored under withheld/). "
                     "val_100 = seed-0 draw of 100 from train_500 minus train_250; train_400 = train_500 minus val_100. "
                     "Stats are pooled over all nodes of the split's cases.")
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
(out / "convert_results.json").write_text(json.dumps(results))
print("CONVERT-DONE", done, json.dumps(stats, indent=1))
