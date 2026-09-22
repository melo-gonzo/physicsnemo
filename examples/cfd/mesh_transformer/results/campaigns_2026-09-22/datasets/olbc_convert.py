"""OLBC converter (2026-09-22): Zenodo NetCDF/HDF5 sets of Mousavi, Mishra & De Lorenzis, "Imposing Boundary Conditions on
Neural Operators via Learned Function Extensions" (arXiv 2602.04923; collection DOI 10.5281/zenodo.18377370; CC BY 4.0) ->
native DomainMesh files.

Two sets are converted, selected with --set:
  poisson     poisson-square-bc5 (record 18377514): -lap u = f on the square [-1, 1]^2 ("Mixed+": non-homogeneous Dirichlet,
              Neumann and Robin conditions on four random boundary segments; the source f is a random sinusoid that varies per
              sample). One fixed mesh of 16,598 nodes (1,120 on the boundary; linear FEM, FEniCSx).
  elasticity  elasticity-circlehollow-m1 (record 18377539): linear elasticity, Steel (lambda 115 GPa, mu 76.9 GPa; stresses
              non-dimensionalized by Sigma = 100 MPa, lengths by L = 1 m, displacements therefore dimensionless: 1e-4 = 0.1 mm)
              on the unit disc with two holes. The smallest hole is clamped (u = 0), the other hole is a free surface; per
              component the exterior circle is cut into four random segments carrying one free-surface segment, one prescribed
              displacement (Dirichlet) and two tractions (Neumann). One fixed mesh of 11,786 nodes (1,312 on the boundary;
              quadratic FEM, stored at the vertices).

Each file stores, per sample, the node coordinates (identical for every sample of a set), interior/sdf (>= 0 inside, 0 on the
boundary), interior/sdf_grad (points INTO the domain), the solution and, per component and BC type, the boundary node indices
with the BC value g (Robin also alpha, in alpha u + du/dn = g). No connectivity is stored; the four exterior segment-junction
nodes per component are listed under no BC type. The boundary loops are reconstructed here from the boundary node set
(sdf == 0) by 2-nearest-neighbour chaining, verified to give closed loops (every node of degree 2).

Per sample the converter writes <out>/<split>_<id>/<split>_<id>.pdmsh with

  interior   : all mesh nodes as a point cloud embedded in 3-D (z = 0) with point_data
                 sdf (N, 1) = -(file sdf)  (<= 0 inside: the program's positive-outside convention),
                 sdf_normals (N, 3) = unit(-(file sdf_grad))  (outward at the boundary, = grad of the stored sdf),
                 poisson:    u (N,), source (N,), tag_dirichlet / tag_neumann / tag_robin (N,) 0/1 nodal labels,
                             bc_value_node (N,) = g at labelled boundary nodes (0 elsewhere), bc_robin_alpha_node (N,);
                 elasticity: displacement (N, 3) = (u_x, u_y, 0), sig11 / sig22 / sig12 (N,) Cauchy stress components,
                             eps11 / eps22 / eps12 (N,) strain components, tag_dirichlet_x / tag_neumann_x /
                             tag_dirichlet_y / tag_neumann_y (N,), bc_disp_node_x / bc_disp_node_y (N,) prescribed
                             displacement at Dirichlet nodes (0 elsewhere), bc_trac_node_x / bc_trac_node_y (N,) traction
                             component at Neumann nodes (0 elsewhere);
  boundaries : {"outer": Mesh of line cells (M, 2) over the compacted boundary nodes (all loops, hole loops included),
                 cell_data normals (M, 3) outward from the solid (into a hole on hole loops; sign fixed by the file's
                 sdf_grad), quadrature_measure (M,) segment lengths, loop_id (M,) 0 = exterior, 1.. = holes by decreasing
                 node count, and per-cell BC labels/values: a cell takes the type shared by its two end nodes; the cell on
                 either side of a junction node takes the type of its labelled end node; values are the mean of the labelled
                 end nodes' g.
                 poisson:    bc_dirichlet, bc_neumann, bc_robin (M,) 0/1, bc_value_g (M,), bc_robin_alpha (M,) (0 off Robin);
                 elasticity: bc_dirichlet_x, bc_neumann_x, bc_disp_x, bc_trac_x and the same for _y (M,);
                 point_data: the nodal tags restricted to the boundary nodes};
  global_data: sample_id ().  (No per-sample scalar parameters exist in these sets: the material is fixed and the source is
               a field. The laboratory frame for the elasticity set is injected by the dataset yaml's SetDomainGlobalField.)

Boundary values are stored in the files' own (dimensionless) units. The elasticity Dirichlet values (~1e-4) and tractions
(~1e-1) differ by three orders of magnitude, which is why they are split into bc_disp_* / bc_trac_* (standardized in the
dataset yaml from the manifest _stats) instead of one bc_value_g.

Splits (manifest.json): train = the 8,448 samples of train.nc, test = the 256 of test.nc, train_n512 / train_n2048 = nested
prefixes of one seed-0 permutation of train. _stats: pooled per-field mean/std over all nodes (cells) of train.

usage: python olbc_convert.py <raw_dir> <out_root> --set {poisson,elasticity} [--shard I --nshards N] [--limit N] [--overwrite]
       python olbc_convert.py <raw_dir> <out_root> --set ... --finalize        # manifest.json from the shard result files
"""
from __future__ import annotations

import argparse, json, time, warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
print("imports done", flush=True)

from physicsnemo.mesh import DomainMesh, Mesh

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("out")
ap.add_argument("--set", required=True, choices=["poisson", "elasticity"])
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--finalize", action="store_true")
args = ap.parse_args()
raw, out = Path(args.raw), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
ELASTIC = args.set == "elasticity"
COMPS = ("0", "1") if ELASTIC else ("0",)
BCTYPES = ("dirichlet", "neumann") if ELASTIC else ("dirichlet", "neumann", "robin")
N_TRAIN_EXPECTED, N_TEST_EXPECTED = 8448, 256
# pooled-statistics fields (interior point_data) and boundary cell fields
if ELASTIC:
    FIELDS = ["u_x", "u_y", "sig11", "sig22", "sig12", "eps11", "eps22", "eps12"]
    BFIELDS = ["bc_disp_x", "bc_trac_x", "bc_disp_y", "bc_trac_y"]
else:
    FIELDS = ["u", "source"]
    BFIELDS = ["bc_value_g", "bc_robin_alpha"]


def to3(v: np.ndarray) -> np.ndarray:
    return np.concatenate([v, np.zeros((len(v), 1))], 1).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------------
# the one mesh of the set: coordinates, boundary node set, loops, cells, outward normals (from train.nc sample 0)
# --------------------------------------------------------------------------------------------------------------------
def build_mesh(f: h5py.File) -> dict:
    xy = np.stack([f["coordinates"][0, 0, 0], f["coordinates"][0, 0, 1]], 1).astype(np.float64)
    sdf_in = np.asarray(f["interior/sdf"][0, 0, 0], np.float64)
    grad_in = np.stack([f["interior/sdf_grad"][0, 0, 0], f["interior/sdf_grad"][0, 0, 1]], 1)
    n = len(xy)
    assert sdf_in.min() >= 0.0, "expected the file's sdf to be >= 0 inside"
    bidx = np.where(sdf_in == 0.0)[0]
    P = xy[bidx]
    # 2-nearest-neighbour chaining of the boundary nodes
    _, j = cKDTree(P).query(P, k=3)
    e = np.concatenate([np.stack([np.arange(len(P)), j[:, 1]], 1), np.stack([np.arange(len(P)), j[:, 2]], 1)], 0)
    e = np.unique(np.sort(e, 1), axis=0)
    deg = np.bincount(e.ravel(), minlength=len(P))
    assert np.all(deg == 2), f"boundary chaining failed: degree histogram {np.bincount(deg).tolist()}"
    # loops by union-find, ordered by walking
    parent = np.arange(len(P))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for a, b in e:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(len(P))])
    comps = sorted(np.unique(roots).tolist(), key=lambda r: -(roots == r).sum())   # exterior = most nodes
    # sanity: the loop with the most nodes must also have the largest extent
    ext = [np.ptp(P[roots == r], axis=0).max() for r in comps]
    assert np.argmax(ext) == 0, f"loop extents {ext} vs node counts {[(roots == r).sum() for r in comps]}"
    # walk every loop into an ordered polygon so that all its cells share one orientation; then fix the orientation
    # once per loop by a majority vote against the file's (inward) sdf gradient. A per-cell vote is unreliable at the
    # square's corners, where the stored gradient is ill-defined (two corner cells came out flipped in the first pass).
    adj = {}
    for a_, b_ in e:
        adj.setdefault(int(a_), []).append(int(b_)); adj.setdefault(int(b_), []).append(int(a_))
    cells_l, loop_l = [], []
    for li, r in enumerate(comps):
        start = int(np.where(roots == r)[0][0]); prev, cur = -1, start; order = [start]
        while True:
            nxt = [v for v in adj[cur] if v != prev][0]
            if nxt == start:
                break
            order.append(nxt); prev, cur = cur, nxt
        assert len(order) == int((roots == r).sum()), "loop walk did not visit every node once"
        order = np.array(order); ce_l = np.stack([order, np.roll(order, -1)], 1)
        t = P[ce_l[:, 1]] - P[ce_l[:, 0]]; n_l = np.stack([t[:, 1], -t[:, 0]], 1)
        vote = float(np.mean(np.sum(n_l * -(grad_in[bidx[ce_l[:, 0]]] + grad_in[bidx[ce_l[:, 1]]]), 1) > 0))
        if vote < 0.5:
            ce_l = ce_l[:, ::-1]
        assert abs(vote - 0.5) > 0.4, f"ambiguous loop orientation vote {vote}"
        cells_l.append(ce_l); loop_l.append(np.full(len(ce_l), li, np.int64))
    ce = np.concatenate(cells_l, 0); loop_id = np.concatenate(loop_l)
    a, b = P[ce[:, 0]], P[ce[:, 1]]
    tang = b - a; length = np.linalg.norm(tang, axis=1); tang = tang / length[:, None]
    nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)               # right-hand normal of the oriented loop = outward from the solid
    # per-loop closure: sum n L over a closed loop is zero
    closure = [float(np.linalg.norm((nrm[loop_id == li] * length[loop_id == li, None]).sum(0))) for li in range(len(comps))]
    perim = [float(length[loop_id == li].sum()) for li in range(len(comps))]
    # interior sdf normals: unit(-grad); ill-defined on the medial axis (|grad| small) -> keep direction, zero if degenerate
    g = -grad_in; gn = np.linalg.norm(g, axis=1, keepdims=True)
    g = np.where(gn > 1e-8, g / np.clip(gn, 1e-30, None), 0.0)
    # boundary nodes: use the vertex-averaged outward cell normal (consistent with the Tensile2d converter)
    vn = np.zeros((len(P), 2)); np.add.at(vn, ce[:, 0], nrm); np.add.at(vn, ce[:, 1], nrm)
    vnn = np.linalg.norm(vn, axis=1, keepdims=True)
    g[bidx] = vn / np.clip(vnn, 1e-30, None)
    # note the file's gradient agrees with the reconstructed outward normals?
    agree = float(np.mean((-(grad_in[bidx]) / np.clip(np.linalg.norm(grad_in[bidx], axis=1, keepdims=True), 1e-30, None) * g[bidx]).sum(1)))
    info = {"n_nodes": int(n), "n_boundary_nodes": int(len(bidx)), "n_cells": int(len(ce)), "n_loops": len(comps),
            "loop_nodes": [int((roots == r).sum()) for r in comps], "loop_perimeter": perim, "loop_closure": closure,
            "loop_centers": [P[roots == r].mean(0).round(4).tolist() for r in comps],
            "seg_len_min": float(length.min()), "seg_len_max": float(length.max()), "sdf_in_max": float(sdf_in.max()),
            "grad_dot_outward_mean_at_boundary": agree,
            "bbox": [float(xy[:, 0].min()), float(xy[:, 0].max()), float(xy[:, 1].min()), float(xy[:, 1].max())]}
    return {"xy": xy, "sdf": (-sdf_in).astype(np.float32), "sdf_normals": to3(g), "bidx": bidx, "cells": ce, "loop_id": loop_id,
            "normals": to3(nrm), "length": length.astype(np.float32), "info": info}


# --------------------------------------------------------------------------------------------------------------------
def read_bc(f: h5py.File, s: int, comp: str, n: int):
    """Per node: type flags (dict bc -> bool (n,)), value g (dict bc -> (n,) nan where absent), robin alpha (n,)."""
    flag, val = {}, {}
    alpha = np.full(n, np.nan)
    for bc in BCTYPES:
        idx = np.asarray(f[f"boundaries/{comp}/{bc}/indices"][s, 0, 0], np.int64)
        gv = np.asarray(f[f"boundaries/{comp}/{bc}/g"][s, 0, 0], np.float64)
        fl = np.zeros(n, bool); fl[idx] = True
        v = np.full(n, np.nan); v[idx] = gv
        flag[bc] = fl; val[bc] = v
        if bc == "robin":
            alpha[idx] = np.asarray(f[f"boundaries/{comp}/robin/alpha"][s, 0, 0], np.float64)
    return flag, val, alpha


def cell_labels(M: dict, flag: dict, val: dict, alpha: np.ndarray):
    """Per boundary cell: one-hot type (dict), value of that type, robin alpha; plus conflict / junction counters."""
    ce = M["bidx"][M["cells"]]                         # global node ids of the cell ends
    m = len(ce)
    types = np.full(m, -1, np.int64)
    both = np.stack([flag[bc][ce[:, 0]] & flag[bc][ce[:, 1]] for bc in BCTYPES], 1)
    anyf = np.stack([flag[bc][ce[:, 0]] | flag[bc][ce[:, 1]] for bc in BCTYPES], 1)
    # cells whose two ends share one type
    shared = both.any(1)
    types[shared] = both[shared].argmax(1)
    # junction cells (one end unlabelled): the labelled end's type; conflicting cells (two labelled ends, different types)
    rest = ~shared
    n_conflict = int((anyf[rest].sum(1) > 1).sum())
    types[rest] = anyf[rest].argmax(1)                # D > N > R priority for the (not expected) conflicts
    unl = ~anyf.any(1); n_unlabelled = int(unl.sum())
    types[unl] = -1                                   # no labelled end at all (not expected): left without a type, counted
    onehot = {bc: (types == k).astype(np.float32) for k, bc in enumerate(BCTYPES)}
    value = np.zeros(m, np.float64)
    for k, bc in enumerate(BCTYPES):
        sel = types == k
        value[sel] = np.nanmean(np.stack([val[bc][ce[sel, 0]], val[bc][ce[sel, 1]]], 1), 1)
    value = np.nan_to_num(value, nan=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # nanmean of two NaNs (cells off Robin) is the intended 0
        al = np.nan_to_num(np.nanmean(np.stack([alpha[ce[:, 0]], alpha[ce[:, 1]]], 1), 1), nan=0.0) * onehot.get("robin", 0.0)
    return onehot, value.astype(np.float32), al.astype(np.float32), n_conflict, n_unlabelled


def convert(f: h5py.File, split: str, s: int, M: dict) -> dict:
    case = f"{split}_{s:06d}"
    dst = out / case / f"{case}.pdmsh"
    if dst.exists() and not args.overwrite:
        return {"case": case, "status": "exists"}
    t1 = time.time()
    n = M["info"]["n_nodes"]
    xy = np.stack([f["coordinates"][s, 0, 0], f["coordinates"][s, 0, 1]], 1)
    if xy.shape != M["xy"].shape or not np.allclose(xy, M["xy"], atol=1e-12):
        raise RuntimeError("coordinates differ from the set's mesh")
    sdf_s = np.asarray(f["interior/sdf"][s, 0, 0], np.float64)
    if not np.allclose(sdf_s, -M["sdf"], atol=1e-9):
        raise RuntimeError("sdf differs from the set's mesh")
    pd = {"sdf": torch.from_numpy(M["sdf"]).reshape(-1, 1), "sdf_normals": torch.from_numpy(M["sdf_normals"])}
    cd = {"normals": torch.from_numpy(M["normals"]), "quadrature_measure": torch.from_numpy(M["length"]),
          "loop_id": torch.from_numpy(M["loop_id"].astype(np.float32))}
    bpd = {}
    sums = {}; nnan = 0; r = {"case": case, "status": "ok", "split": split, "n_conflict": 0, "n_unlabelled": 0}
    # ---- fields
    if ELASTIC:
        ux = np.asarray(f["interior/solution"][s, 0, 0], np.float64); uy = np.asarray(f["interior/solution"][s, 0, 1], np.float64)
        st = [np.asarray(f["interior/cauchystress"][s, 0, c], np.float64) for c in (0, 3, 1)]   # xx, yy, xy
        ep = [np.asarray(f["interior/strain"][s, 0, c], np.float64) for c in (0, 3, 1)]
        arrs = {"u_x": ux, "u_y": uy, "sig11": st[0], "sig22": st[1], "sig12": st[2], "eps11": ep[0], "eps22": ep[1], "eps12": ep[2]}
    else:
        arrs = {"u": np.asarray(f["interior/solution"][s, 0, 0], np.float64), "source": np.asarray(f["interior/source"][s, 0, 0], np.float64)}
    for k, v in arrs.items():
        nn = int(np.isnan(v).sum()); nnan += nn
        if nn:
            v = np.nan_to_num(v, nan=0.0); arrs[k] = v
        sums[k] = [float(v.sum()), float((v ** 2).sum()), int(n)]
        r[f"{k}_absmax"] = float(np.abs(v).max())
    if ELASTIC:
        pd["displacement"] = torch.from_numpy(to3(np.stack([arrs["u_x"], arrs["u_y"]], 1)))
        for k in ("sig11", "sig22", "sig12", "eps11", "eps22", "eps12"):
            pd[k] = torch.from_numpy(arrs[k].astype(np.float32))
    else:
        pd["u"] = torch.from_numpy(arrs["u"].astype(np.float32)); pd["source"] = torch.from_numpy(arrs["source"].astype(np.float32))
    # ---- boundary conditions per component
    for comp in COMPS:
        sfx = {"0": "_x", "1": "_y"}[comp] if ELASTIC else ""
        flag, val, alpha = read_bc(f, s, comp, n)
        onehot, value, al, ncf, nul = cell_labels(M, flag, val, alpha)
        r["n_conflict"] += ncf; r["n_unlabelled"] += nul
        for bc in BCTYPES:
            cd[f"bc_{bc}{sfx}"] = torch.from_numpy(onehot[bc])
            tag = flag[bc].astype(np.float32)
            pd[f"tag_{bc}{sfx}"] = torch.from_numpy(tag); bpd[f"tag_{bc}{sfx}"] = torch.from_numpy(tag[M["bidx"]])
            r[f"n_{bc}{sfx}"] = int(onehot[bc].sum()); r[f"len_{bc}{sfx}"] = float((onehot[bc] * M["length"]).sum())
        if ELASTIC:
            disp = value * onehot["dirichlet"]; trac = value * onehot["neumann"]
            cd[f"bc_disp{sfx}"] = torch.from_numpy(disp.astype(np.float32)); cd[f"bc_trac{sfx}"] = torch.from_numpy(trac.astype(np.float32))
            nd = np.nan_to_num(val["dirichlet"], nan=0.0).astype(np.float32); nt = np.nan_to_num(val["neumann"], nan=0.0).astype(np.float32)
            pd[f"bc_disp_node{sfx}"] = torch.from_numpy(nd); pd[f"bc_trac_node{sfx}"] = torch.from_numpy(nt)
            for k, v, sel in ((f"bc_disp{sfx}", disp, onehot["dirichlet"] > 0), (f"bc_trac{sfx}", trac, onehot["neumann"] > 0)):
                sums[k] = [float(v[sel].sum()), float((v[sel] ** 2).sum()), int(sel.sum())]   # over labelled cells only
        else:
            cd["bc_value_g"] = torch.from_numpy(value); cd["bc_robin_alpha"] = torch.from_numpy(al)
            node_val = np.zeros(n); node_alpha = np.nan_to_num(alpha, nan=0.0)
            for bc in BCTYPES:
                node_val[flag[bc]] = val[bc][flag[bc]]
            pd["bc_value_node"] = torch.from_numpy(node_val.astype(np.float32)); pd["bc_robin_alpha_node"] = torch.from_numpy(node_alpha.astype(np.float32))
            sums["bc_value_g"] = [float(value.sum()), float((value ** 2).sum()), int(len(value))]
            sel = onehot["robin"] > 0
            sums["bc_robin_alpha"] = [float(al[sel].sum()), float((al[sel] ** 2).sum()), int(sel.sum())]
    interior = Mesh(points=torch.from_numpy(to3(M["xy"])), point_data=pd)
    outer = Mesh(points=torch.from_numpy(to3(M["xy"][M["bidx"]])), cells=torch.from_numpy(M["cells"].astype(np.int64)),
                 cell_data=cd, point_data=bpd)
    gd = {"sample_id": torch.tensor(float(s), dtype=torch.float32)}
    dm = DomainMesh(interior=interior, boundaries={"outer": outer}, global_data=gd)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dm.save(dst)
    r.update({"n_nan_replaced": nnan, "sums": sums, "t": time.time() - t1})
    return r


# --------------------------------------------------------------------------------------------------------------------
def finalize():
    res = []
    for p in sorted(out.glob("convert_results_*.json")):
        res += json.load(open(p))
    byid = {r["case"]: r for r in res if r.get("status") == "ok"}
    err = [r for r in res if str(r.get("status", "")).startswith("error")]
    print("results", len(res), "ok", len(byid), "errors", len(err), err[:5])
    train = [f"train_{i:06d}" for i in range(N_TRAIN_EXPECTED)]; test = [f"test_{i:06d}" for i in range(N_TEST_EXPECTED)]
    missing = [c for c in train + test if c not in byid and not (out / c / f"{c}.pdmsh").exists()]
    print("missing files", len(missing), missing[:10])
    rng = np.random.default_rng(0); perm = rng.permutation(N_TRAIN_EXPECTED)
    manifest = {"train": train, "test": test,
                "train_n512": sorted(train[i] for i in perm[:512]), "train_n2048": sorted(train[i] for i in perm[:2048])}
    keys = FIELDS + BFIELDS
    acc = {k: np.zeros(3) for k in keys}; nn = 0; nconf = 0; nunl = 0; nnan = 0; absmax = {k: 0.0 for k in FIELDS}
    for c in train:
        r = byid.get(c)
        if r is None:
            continue
        for k in keys:
            if k in r["sums"]:
                acc[k] += r["sums"][k]
        nn += 1; nconf += r["n_conflict"]; nunl += r["n_unlabelled"]; nnan += r["n_nan_replaced"]
        for k in FIELDS:
            absmax[k] = max(absmax[k], r[f"{k}_absmax"])
    st = {}
    for k in keys:
        s, s2, cnt = acc[k]
        if cnt > 0:
            m = s / cnt; st[k] = {"mean": m, "std": float(np.sqrt(max(s2 / cnt - m * m, 0.0))), "count": int(cnt)}
    if ELASTIC:
        u = np.stack([acc["u_x"], acc["u_y"]]); m = u[:, 0] / u[:, 2]; var = u[:, 1] / u[:, 2] - m ** 2
        st["displacement"] = {"mean": [float(m[0]), float(m[1]), 0.0], "std": [float(np.sqrt(max(var[0], 0))), float(np.sqrt(max(var[1], 0))), 0.0],
                              "std_scalar": float(np.sqrt(max(var.sum(), 0) / 2))}
    st["_n_cases"] = nn; st["_n_conflict_cells"] = nconf; st["_n_unlabelled_cells"] = nunl; st["_n_nan_replaced"] = nnan
    st["_absmax"] = absmax
    manifest["_stats"] = {"train": st}
    manifest["_mesh"] = json.load(open(out / "mesh_info.json"))
    manifest["_note"] = (f"OLBC {'elasticity-circlehollow-m1 (Steel; record 18377539)' if ELASTIC else 'poisson-square-bc5 (record 18377514)'}, "
                         "CC BY 4.0. train = the 8448 samples of train.nc, test = the 256 of test.nc; train_n512 / train_n2048 are nested "
                         "prefixes of one seed-0 permutation of train. _stats pooled over all nodes (interior fields) or labelled boundary "
                         "cells (bc_*) of train. Fields in the files' dimensionless units.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("FINALIZE-DONE", json.dumps(st, indent=1))


if args.finalize:
    finalize(); raise SystemExit

files = {"train": h5py.File(raw / "train.nc", "r"), "test": h5py.File(raw / "test.nc", "r")}
n_train, n_test = int(files["train"].attrs["count"]), int(files["test"].attrs["count"])
print("counts", n_train, n_test, flush=True)
assert (n_train, n_test) == (N_TRAIN_EXPECTED, N_TEST_EXPECTED), (n_train, n_test)
M = build_mesh(files["train"])
print("mesh", json.dumps(M["info"]), flush=True)
if args.shard == 0:
    (out / "mesh_info.json").write_text(json.dumps(M["info"], indent=1))
# cross-check the test file shares the mesh
xy_t = np.stack([files["test"]["coordinates"][0, 0, 0], files["test"]["coordinates"][0, 0, 1]], 1)
assert np.allclose(xy_t, M["xy"]), "test.nc mesh differs from train.nc"

todo = [("train", i) for i in range(n_train)] + [("test", i) for i in range(n_test)]
todo = todo[args.shard::args.nshards]
if args.limit:
    todo = todo[: args.limit]
results = []; t0 = time.time()
for k, (split, s) in enumerate(todo):
    try:
        r = convert(files[split], split, s, M)
    except Exception as e:  # noqa: BLE001
        r = {"case": f"{split}_{s:06d}", "status": f"error: {e!r}"}
    results.append(r)
    print(json.dumps({kk: v for kk, v in r.items() if kk != "sums"}), f"[done {k + 1}/{len(todo)} {time.time() - t0:.0f}s]", flush=True)
(out / f"convert_results_{args.shard:02d}.json").write_text(json.dumps(results))
print("CONVERT-SHARD-DONE", args.shard, len(results), "errors", sum(str(r.get("status")).startswith("error") for r in results))
