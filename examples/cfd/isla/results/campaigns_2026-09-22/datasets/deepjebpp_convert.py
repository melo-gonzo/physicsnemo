"""DeepJEB++ converter (2026-09-22): Yoo, Jeong, Ra, Lee, Yang, Jeong & Kang (KAIST SmartDesignLab), "DeepJEB++: Foundation
Model-Driven Large-Scale 3D Engineering Dataset via 2D Latent Space Augmentation" (arXiv 2606.12994; Hugging Face dataset
KAIST-SmartDesignLab/DeepJEB-PP, revision main = v1.2 of 2026-09; ODC-By 1.0) -> native DomainMesh files, one per
(bracket, load case).

Source (README.txt / metadata.json of the release):
  15,360 TRELLIS-generated jet-engine brackets (SimJEB design space; case id "<base>-<sub>-diag_xz_mm_IS<aug>": <base>-<sub>
  is the SimJEB seed pair, IS00..IS18 the 2-D latent interpolation frame -> the brackets of one pair are a smooth family and
  must not straddle a split). Linear static elasticity, Ti-6Al-4V (E = 113,800 MPa, nu = 0.342), units mm / N / N*mm / MPa.
  Surface decimated to 25,000 vertices (pymeshfix-cleaned), tetgen "pq1.2/10Y" volume mesh, four bolt holes clamped (u = 0),
  the load applied on the lug/clevis bore ("total traction shared equally over lug nodes"). Load cases (name, type,
  direction, magnitude): ver force (0,0,1) 35,600 N; hor force (-1,0,0) 37,800 N; dia force (-0.669,0,0.743) 42,300 N;
  tor moment about (0,0,1) 565,000 N*mm (v1.1: axis corrected Y -> Z; the v1.0 torsion is retained only under git tag v1.0).
  3_fea_fields/<case>.npz: surface_points (N,3) mm, surface_faces (M,3), per load L_U (N,3) mm, L_vm (N,) MPa, L_maxu,
  L_p95vm (volume-wide scalars), R (lug bore radius). 2_boundary_conditions/<case>.npz: bolt_idx (clamped nodes, recovered
  as U == 0 in all four cases), lug_idx (loaded nodes), indices INTO surface_points of the 25k FEM surface (not the native
  ~50k .obj of 1_surface_meshes, which is therefore not used here). deepjebpp_labels.csv: mass_g, vol_mm3, R, per-load
  maxu / p95vm / maxvm. reproducibility/data_splits.csv (v1.2): the authors' pair-grouped and random train/test splits.

Per (bracket, load) the converter writes <out>/<bracket>_<load>/<bracket>_<load>.pdmsh, bracket = "<base>-<sub>-IS<aug>":

  boundaries  : {"surface": Mesh of the FEM surface triangles, faces re-wound so that the signed volume is positive (outward
                 normals; the winding of every face is flipped together, the input orientation is checked to be edge-
                 consistent and the count of inconsistent edges is recorded), degenerate (zero-area) faces dropped.
                 cell_data: normals (M,3) unit outward, quadrature_measure (M,) areas mm^2,
                   bc_fixed (M,) 1 on cells with >= 1 vertex in bolt_idx, bc_loaded (M,) 1 on cells with >= 1 vertex in lug_idx,
                   bc_load (M,3) traction MPa on loaded cells, 0 elsewhere:
                       force  cases: t = F d / A_loaded                              (sum_c A_c t_c = F d)
                       torque case : t = M (a x r_perp_c) / sum_loaded A |r_perp|^2  (sum_c A_c r_c x t_c = M a),
                                     r_perp = offset of the cell centroid from the axis a through the area-weighted centroid
                                     of the loaded cells; this is the traction field of a rigid twist about the lug axis,
                                     the natural per-cell representation of a moment,
                   bc_load_n (M,) = n . t and bc_load_mag (M,) = |t| (the two rotation-invariant scalars of the traction),
                   von_mises (M,) MPa and displacement (M,3) mm = mean of the three vertex values (the surface configs read
                   targets from cell_data; MeshToDomainMesh moves them to the cell centroids)};
  interior    : the FEM surface vertices as a point cloud with point_data von_mises (N,), displacement (N,3), tag_fixed,
                tag_loaded (N,) and normals (N,3) (area-weighted vertex normals) -- the vertex-resolution copy of the same
                fields for readers that want nodal data; unused by the surface configs;
  global_data : load_dir (3,) unit vector (force direction, or the moment axis), load_type (2,) one-hot [is_force, is_torque],
                load_magnitude () N or N*mm, load_case () 0..3 in the order ver/hor/dia/tor, bracket_id () int64 =
                base*100000 + sub*100 + aug, mass_g, vol_mm3, lug_radius_mm, label_maxu, label_p95vm, label_maxvm ()
                (volume-wide labels of this load case from the CSV; targets, never inputs).

Splits (manifest.json), all grouped by SEED PAIR (so a fortiori by bracket): test = the brackets of a seed-0 random 10 % of
the pairs (pairs drawn until >= 10 % of the brackets), train = the rest; train_n256 / train_n1024 / train_n4096 = nested
seed-0 prefixes of the train BRACKETS (files = 4 x that); per-load sub-splits <split>_<load>; the authors' splits under
official_pairgrouped_{train,test} and official_random_{train,test}. _stats: pooled per-field mean/std over the cells of
train (all four loads), plus _stats_by_load.

usage: python deepjebpp_convert.py <raw_dir> <out_root> [--shard I --nshards N] [--limit N] [--overwrite]
       python deepjebpp_convert.py <raw_dir> <out_root> --finalize
"""
from __future__ import annotations

import argparse, csv, json, os, re, shutil, stat, time
from pathlib import Path

import numpy as np
import torch
print("imports done", flush=True)

from physicsnemo.mesh import DomainMesh, Mesh

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("out")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--finalize", action="store_true")
ap.add_argument("--rescore", action="store_true", help="load back every written file of this shard and rebuild its stats record "
                "(convert_results_rescore_XX.json); needed because a cpu-free requeue re-runs a shard, which then reports 'exists' "
                "without sums for the files the killed run had written")
ap.add_argument("--no-torsion-patch", action="store_true", help="read tor_* from 3_fea_fields even if the v1.1 patch dir exists")
ap.add_argument("--repair", action="store_true", help="scan every expected file for completeness (the four meta.json files) and "
                "re-convert the incomplete ones (unlocking a write-locked case dir first); writes convert_results_repair.json")
ap.add_argument("--expect-cases", type=int, default=15360, help="number of brackets in the raw tree (override for synthetic tests only)")
args = ap.parse_args()
raw, out = Path(args.raw), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

LOADS = ("ver", "hor", "dia", "tor")
LOAD_SPEC = {  # from metadata.json v1.1
    "ver": ("force", (0.0, 0.0, 1.0), 35600.0),
    "hor": ("force", (-1.0, 0.0, 0.0), 37800.0),
    "dia": ("force", (-0.669, 0.0, 0.743), 42300.0),
    "tor": ("torque", (0.0, 0.0, 1.0), 565000.0),
}
N_CASES_EXPECTED = args.expect_cases
CASE_RE = re.compile(r"^(\d+)-(\d+)-diag_xz_mm_IS(\d+)$")
CELL_FIELDS = ["von_mises", "u_x", "u_y", "u_z", "area"]          # pooled over all cells
LOADED_FIELDS = ["bc_load_n", "bc_load_mag"]                       # pooled over loaded cells


def bracket_of(case: str) -> tuple[str, str, int]:
    m = CASE_RE.match(case)
    if not m:
        raise ValueError(f"unexpected case id {case!r}")
    base, sub, aug = m.groups()
    return f"{base}-{sub}-IS{aug}", f"{base}-{sub}", int(base) * 100000 + int(sub) * 100 + int(aug)


def index_files(sub: str, ext: str) -> dict[str, Path]:
    files = sorted((raw / sub).rglob(f"*{ext}"))
    d = {p.stem: p for p in files}
    assert len(d) == len(files), f"duplicate basenames under {sub}"
    return d


# --------------------------------------------------------------------------------------------------------------------
def oriented_surface(P: np.ndarray, F: np.ndarray) -> dict:
    """Outward-wound faces, unit normals, areas; degenerate faces dropped; orientation diagnostics."""
    F = F.astype(np.int64)
    n_in = len(F)
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    av = 0.5 * np.cross(b - a, c - a)                    # area vectors
    area = np.linalg.norm(av, axis=1)
    keep = area > 1e-9 * max(area.max(), 1e-30)
    n_degenerate = int((~keep).sum())
    F, av, area, a, b, c = F[keep], av[keep], area[keep], a[keep], b[keep], c[keep]
    vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    flipped = vol < 0
    if flipped:
        F = F[:, [0, 2, 1]]; av = -av; vol = -vol
    # edge consistency of the (global) winding: every directed edge once, its reverse once
    N = len(P)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    key = e[:, 0] * N + e[:, 1]; rkey = e[:, 1] * N + e[:, 0]
    uk, cnt = np.unique(key, return_counts=True)
    dup = int((cnt > 1).sum())                             # same directed edge twice = two faces disagree in winding
    pos = np.searchsorted(uk, rkey); pos[pos >= len(uk)] = 0
    has_rev = uk[pos] == rkey
    n_open = int((~has_rev).sum())                         # directed edges without a reverse partner = boundary edges
    nrm = av / area[:, None]
    closure = float(np.linalg.norm(av.sum(0)) / area.sum())
    return {"F": F, "normals": nrm.astype(np.float32), "area": area.astype(np.float32), "volume_mm3": vol, "flipped": flipped,
            "n_faces_in": n_in, "n_degenerate": n_degenerate, "n_dup_directed_edges": dup, "n_open_edges": n_open,
            "closure_rel": closure}


def vertex_normals(P: np.ndarray, F: np.ndarray, av_unit: np.ndarray, area: np.ndarray) -> np.ndarray:
    vn = np.zeros_like(P, dtype=np.float64)
    w = av_unit * area[:, None]
    for k in range(3):
        np.add.at(vn, F[:, k], w)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return np.where(n > 0, vn / np.clip(n, 1e-30, None), 0.0).astype(np.float32)


def traction(load: str, cen: np.ndarray, area: np.ndarray, loaded: np.ndarray) -> np.ndarray:
    kind, d, mag = LOAD_SPEC[load]
    d = np.asarray(d, np.float64); d = d / np.linalg.norm(d)
    t = np.zeros((len(cen), 3), np.float64)
    A = area[loaded].astype(np.float64)
    if kind == "force":
        t[loaded] = mag * d / A.sum()
    else:
        c0 = (cen[loaded] * A[:, None]).sum(0) / A.sum()
        r = cen[loaded] - c0; r_perp = r - np.outer(r @ d, d)
        t[loaded] = mag * np.cross(d, r_perp) / (A * (r_perp ** 2).sum(1)).sum()
    return t


# --------------------------------------------------------------------------------------------------------------------
def is_complete(name: str) -> bool:
    """A finished save has these four meta files (the last one written is the top-level meta.json)."""
    f = out / name / f"{name}.pdmsh"
    return all((f / p).is_file() for p in ("meta.json", "_tensordict/global_data/meta.json",
                                            "_tensordict/boundaries/surface/_tensordict/cell_data/meta.json",
                                            "_tensordict/interior/_tensordict/point_data/meta.json"))


def make_writable(path: Path) -> None:
    """Undo a write-lock (chmod -R a-w) on one case dir so it can be repaired."""
    for root_, dirs, files in os.walk(path):
        for n in dirs + files:
            p = Path(root_) / n; p.chmod(p.stat().st_mode | stat.S_IWUSR)
    path.chmod(path.stat().st_mode | stat.S_IWUSR)


def convert_bracket(case: str, fld_path: Path, bc_path: Path, tor_path: Path | None, label: dict, only: list[str] | None = None) -> list[dict]:
    bracket, pair, bid = bracket_of(case)
    todo = [ld for ld in (only or LOADS) if args.overwrite or only or not is_complete(f"{bracket}_{ld}")]
    if not todo:
        return [{"case": f"{bracket}_{ld}", "status": "exists"} for ld in LOADS]
    t1 = time.time()
    fld = np.load(fld_path); bc = np.load(bc_path)
    P = np.asarray(fld["surface_points"], np.float64); F0 = np.asarray(fld["surface_faces"])
    assert int(bc["n_surface"]) == len(P), ("n_surface mismatch", int(bc["n_surface"]), len(P))
    bolt = np.asarray(bc["bolt_idx"], np.int64); lug = np.asarray(bc["lug_idx"], np.int64)
    S = oriented_surface(P, F0); F = S["F"]
    cen = P[F].mean(1)
    tag_fixed = np.zeros(len(P), np.float32); tag_fixed[bolt] = 1.0
    tag_loaded = np.zeros(len(P), np.float32); tag_loaded[lug] = 1.0
    fixed_c = tag_fixed[F].max(1) > 0; loaded_c = tag_loaded[F].max(1) > 0
    n_both = int((fixed_c & loaded_c).sum())
    assert loaded_c.sum() > 0 and fixed_c.sum() > 0, "no loaded or no fixed cells"
    vn = vertex_normals(P, F, S["normals"], S["area"])
    tor_src = "main"
    tor = None
    if tor_path is not None and not args.no_torsion_patch:
        tor = np.load(tor_path)
        if not np.array_equal(np.asarray(tor["surface_points"]), np.asarray(fld["surface_points"])):
            raise RuntimeError("torsion patch surface_points differ from 3_fea_fields")
        tor_src = "patch"
    results = []
    for ld in todo:
        name = f"{bracket}_{ld}"; dst = out / name / f"{name}.pdmsh"
        src = tor if (ld == "tor" and tor is not None) else fld
        U = np.asarray(src[f"{ld}_U"], np.float64); vm = np.asarray(src[f"{ld}_vm"], np.float64)
        nnan = int(np.isnan(U).sum() + np.isnan(vm).sum())
        U = np.nan_to_num(U); vm = np.nan_to_num(vm)
        assert U.shape == (len(P), 3) and vm.shape == (len(P),), (U.shape, vm.shape)
        kind, d, mag = LOAD_SPEC[ld]
        t = traction(ld, cen, S["area"].astype(np.float64), loaded_c)
        tn = (t * S["normals"]).sum(1); tmag = np.linalg.norm(t, axis=1)
        vm_c = vm[F].mean(1); U_c = U[F].mean(1)
        # moment / force resultant check of the traction field
        dn = np.asarray(d) / np.linalg.norm(d); res_perp = 0.0
        if kind == "force":
            res = float(np.linalg.norm((t * S["area"][:, None]).sum(0) - mag * dn) / mag)
        else:
            A = S["area"][loaded_c].astype(np.float64); c0 = (cen[loaded_c] * A[:, None]).sum(0) / A.sum()
            mom = (np.cross(cen[loaded_c] - c0, t[loaded_c]) * A[:, None]).sum(0)
            res = float(abs(mom @ dn - mag) / mag)                       # axial moment: exact by construction
            res_perp = float(np.linalg.norm(mom - (mom @ dn) * dn) / mag)  # transverse moment of the twist on an asymmetric lug
        cd = {"normals": torch.from_numpy(S["normals"]), "quadrature_measure": torch.from_numpy(S["area"]),
              "bc_fixed": torch.from_numpy(fixed_c.astype(np.float32)), "bc_loaded": torch.from_numpy(loaded_c.astype(np.float32)),
              "bc_load": torch.from_numpy(t.astype(np.float32)), "bc_load_n": torch.from_numpy(tn.astype(np.float32)),
              "bc_load_mag": torch.from_numpy(tmag.astype(np.float32)),
              "von_mises": torch.from_numpy(vm_c.astype(np.float32)), "displacement": torch.from_numpy(U_c.astype(np.float32))}
        surface = Mesh(points=torch.from_numpy(P.astype(np.float32)), cells=torch.from_numpy(F), cell_data=cd)
        pd = {"von_mises": torch.from_numpy(vm.astype(np.float32)), "displacement": torch.from_numpy(U.astype(np.float32)),
              "tag_fixed": torch.from_numpy(tag_fixed), "tag_loaded": torch.from_numpy(tag_loaded), "normals": torch.from_numpy(vn)}
        interior = Mesh(points=torch.from_numpy(P.astype(np.float32)), point_data=pd)
        f32 = lambda v: torch.tensor(float(v), dtype=torch.float32)
        gd = {"load_dir": torch.tensor(np.asarray(d) / np.linalg.norm(d), dtype=torch.float32),
              "load_type": torch.tensor([1.0, 0.0] if kind == "force" else [0.0, 1.0], dtype=torch.float32),
              "load_magnitude": f32(mag), "load_case": f32(LOADS.index(ld)),
              "bracket_id": torch.tensor(bid, dtype=torch.int64),
              "mass_g": f32(label["mass_g"]), "vol_mm3": f32(label["vol_mm3"]), "lug_radius_mm": f32(label["R"]),
              "label_maxu": f32(label[f"{ld}_maxu"]), "label_p95vm": f32(label[f"{ld}_p95vm"]), "label_maxvm": f32(label[f"{ld}_maxvm"])}
        dm = DomainMesh(interior=interior, boundaries={"surface": surface}, global_data=gd)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")                 # atomic publish: a kill mid-save must not leave a file that
        if tmp.exists():                                       # the existence check would take for finished (829391 left one)
            shutil.rmtree(tmp)
        dm.save(tmp)
        if dst.exists():
            shutil.rmtree(dst)
        tmp.rename(dst)
        sums = {"von_mises": [float(vm_c.sum()), float((vm_c ** 2).sum()), len(vm_c)], "area": [float(S["area"].sum()), float((S["area"] ** 2).sum()), len(vm_c)]}
        for k, col in zip(("u_x", "u_y", "u_z"), U_c.T):
            sums[k] = [float(col.sum()), float((col ** 2).sum()), len(col)]
        for k, v in (("bc_load_n", tn[loaded_c]), ("bc_load_mag", tmag[loaded_c])):
            sums[k] = [float(v.sum()), float((v ** 2).sum()), int(loaded_c.sum())]
        results.append({"case": name, "status": "ok", "bracket": bracket, "pair": pair, "load": ld, "tor_src": tor_src if ld == "tor" else "-",
                        "n_points": len(P), "n_cells": len(F), "n_fixed_cells": int(fixed_c.sum()), "n_loaded_cells": int(loaded_c.sum()),
                        "n_fixed_nodes": len(bolt), "n_loaded_nodes": len(lug), "n_fixed_and_loaded_cells": n_both,
                        "flipped": bool(S["flipped"]), "n_degenerate": S["n_degenerate"], "n_dup_directed_edges": S["n_dup_directed_edges"],
                        "n_open_edges": S["n_open_edges"], "closure_rel": S["closure_rel"], "volume_mm3": S["volume_mm3"],
                        "extent_mm": float(np.ptp(P, axis=0).max()), "resultant_rel_err": res, "torque_transverse_rel": res_perp,
                        "vm_max": float(vm.max()), "u_absmax": float(np.abs(U).max()), "u_bolt_absmax": float(np.abs(U[bolt]).max()),
                        "n_nan_replaced": nnan, "sums": sums, "t": time.time() - t1})
    return results


def rescore_file(name: str, label: dict) -> dict:
    """The stats record of an already-written file, rebuilt from the file itself (no raw input needed)."""
    bracket, ld = name.rsplit("_", 1)
    dm = DomainMesh.load(out / name / f"{name}.pdmsh")
    B = dm.boundaries["surface"]; I = dm.interior
    P = B.points.double().numpy(); F = B.cells.numpy(); cd = {k: v.numpy() for k, v in B.cell_data.items()}
    area = cd["quadrature_measure"].astype(np.float64); n = cd["normals"].astype(np.float64); t = cd["bc_load"].astype(np.float64)
    fixed_c = cd["bc_fixed"] > 0; loaded_c = cd["bc_loaded"] > 0; cen = P[F].mean(1)
    kind, d, mag = LOAD_SPEC[ld]; dn = np.asarray(d) / np.linalg.norm(d); res_perp = 0.0
    if kind == "force":
        res = float(np.linalg.norm((t * area[:, None]).sum(0) - mag * dn) / mag)
    else:
        A = area[loaded_c]; c0 = (cen[loaded_c] * A[:, None]).sum(0) / A.sum()
        mom = (np.cross(cen[loaded_c] - c0, t[loaded_c]) * A[:, None]).sum(0)
        res = float(abs(mom @ dn - mag) / mag); res_perp = float(np.linalg.norm(mom - (mom @ dn) * dn) / mag)
    vm_c = cd["von_mises"].astype(np.float64); U_c = cd["displacement"].astype(np.float64)
    vm = I.point_data["von_mises"].numpy(); U = I.point_data["displacement"].numpy(); bolt = I.point_data["tag_fixed"].numpy() > 0
    sums = {"von_mises": [float(vm_c.sum()), float((vm_c ** 2).sum()), len(vm_c)], "area": [float(area.sum()), float((area ** 2).sum()), len(vm_c)]}
    for k, col in zip(("u_x", "u_y", "u_z"), U_c.T):
        sums[k] = [float(col.sum()), float((col ** 2).sum()), len(col)]
    tn = (t * n).sum(1); tmag = np.linalg.norm(t, axis=1)
    for k, v in (("bc_load_n", tn[loaded_c]), ("bc_load_mag", tmag[loaded_c])):
        sums[k] = [float(v.sum()), float((v ** 2).sum()), int(loaded_c.sum())]
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    return {"case": name, "status": "ok", "bracket": bracket, "pair": "-".join(bracket.split("-")[:2]), "load": ld, "rescored": True,
            "n_points": len(P), "n_cells": len(F), "n_fixed_cells": int(fixed_c.sum()), "n_loaded_cells": int(loaded_c.sum()),
            "n_fixed_nodes": int(bolt.sum()), "n_loaded_nodes": int((I.point_data["tag_loaded"].numpy() > 0).sum()),
            "n_fixed_and_loaded_cells": int((fixed_c & loaded_c).sum()), "closure_rel": float(np.linalg.norm((n * area[:, None]).sum(0)) / area.sum()),
            "volume_mm3": float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0), "extent_mm": float(np.ptp(P, axis=0).max()),
            "resultant_rel_err": res, "torque_transverse_rel": res_perp, "vm_max": float(vm.max()), "u_absmax": float(np.abs(U).max()),
            "u_bolt_absmax": float(np.abs(U[bolt]).max()) if bolt.any() else 0.0, "n_nan_replaced": 0, "sums": sums}


# --------------------------------------------------------------------------------------------------------------------
def finalize():
    orig, resc = [], []
    for p in sorted(out.glob("convert_results_*.json")):
        (resc if "rescore" in p.name else orig).extend(json.load(open(p)))
    byid = {r["case"]: r for r in resc if r.get("status") == "ok"}          # load-back records: complete coverage
    n_resc = len(byid)
    for r in orig:                                                          # original records carry the raw-input-only diagnostics
        if r.get("status") == "ok":
            byid[r["case"]] = {**byid.get(r["case"], {}), **r}
    err = [r for r in orig + resc if str(r.get("status", "")).startswith("error") and r["case"] not in byid]
    print("results orig", len(orig), "rescored ok", n_resc, "merged ok", len(byid), "unrepaired errors", len(err), err[:5])
    if err:
        raise SystemExit(f"{len(err)} files have an error record and no ok record; repair them (--repair) before finalizing")
    labels = list(csv.DictReader(open(raw / "deepjebpp_labels.csv")))
    cases = sorted(r["case"] for r in labels)
    assert len(cases) == N_CASES_EXPECTED, len(cases)
    br = {c: bracket_of(c) for c in cases}
    brackets = sorted(b for b, _, _ in br.values())
    pairs = sorted({p for _, p, _ in br.values()})
    pair_brackets = {p: sorted(b for b, pp, _ in br.values() if pp == p) for p in pairs}
    files = lambda bs: sorted(f"{b}_{ld}" for b in bs for ld in LOADS)
    missing = [f for f in files(brackets) if not (out / f / f"{f}.pdmsh").exists()]
    print("missing files", len(missing), missing[:10])
    # seed-0 pair-grouped 90/10
    rng = np.random.default_rng(0); perm = rng.permutation(len(pairs))
    test_b, i = [], 0
    while len(test_b) < 0.1 * len(brackets):
        test_b += pair_brackets[pairs[perm[i]]]; i += 1
    test_pairs = [pairs[perm[j]] for j in range(i)]
    test_b = sorted(test_b); train_b = sorted(set(brackets) - set(test_b))
    perm_t = np.random.default_rng(0).permutation(len(train_b))
    nested = {f"train_n{n}": sorted(train_b[j] for j in perm_t[:n]) for n in (256, 1024, 4096)}
    manifest = {"train": files(train_b), "test": files(test_b)}
    for k, bs in nested.items():
        manifest[k] = files(bs)
    for k, bs in [("train", train_b), ("test", test_b)] + list(nested.items()):
        for ld in LOADS:
            manifest[f"{k}_{ld}"] = sorted(f"{b}_{ld}" for b in bs)
    manifest["train_brackets"] = train_b; manifest["test_brackets"] = test_b; manifest["test_pairs"] = sorted(test_pairs)
    # authors' splits (v1.2 reproducibility/data_splits.csv)
    sp = raw / "reproducibility" / "data_splits.csv"
    if sp.exists():
        rows = list(csv.DictReader(open(sp)))
        for col, tag in (("split_pairgrouped", "pairgrouped"), ("split_random", "random")):
            for s in ("train", "test"):
                manifest[f"official_{tag}_{s}"] = files(br[r["case"]][0] for r in rows if r[col] == s)
    # stats over train (pooled) and by load
    def accumulate(names):
        acc = {k: np.zeros(3) for k in CELL_FIELDS + LOADED_FIELDS}; n = 0
        for c in names:
            r = byid.get(c)
            if r is None:
                continue
            n += 1
            for k in acc:
                acc[k] += r["sums"][k]
        st = {}
        for k, (s, s2, cnt) in acc.items():
            if cnt > 0:
                m = s / cnt; st[k] = {"mean": m, "std": float(np.sqrt(max(s2 / cnt - m * m, 0.0))), "count": int(cnt)}
        u = np.stack([acc["u_x"], acc["u_y"], acc["u_z"]]); m = u[:, 0] / u[:, 2]; var = u[:, 1] / u[:, 2] - m ** 2
        st["displacement"] = {"mean": m.tolist(), "std": np.sqrt(np.clip(var, 0, None)).tolist(), "std_scalar": float(np.sqrt(max(var.sum(), 0) / 3))}
        st["_n_files"] = n
        return st
    st = accumulate(manifest["train"])
    okr = [byid[c] for c in manifest["train"] if c in byid]
    raw_r = [r for r in okr if "flipped" in r]                             # records that saw the raw input
    diag = {"n_with_raw_diagnostics": len(raw_r), "n_flipped": sum(r["flipped"] for r in raw_r), "n_with_degenerate": sum(r["n_degenerate"] > 0 for r in raw_r),
            "n_with_dup_edges": sum(r["n_dup_directed_edges"] > 0 for r in raw_r), "n_with_open_edges": sum(r["n_open_edges"] > 0 for r in raw_r),
            "closure_rel_max": max(r["closure_rel"] for r in okr), "resultant_rel_err_max": max(r["resultant_rel_err"] for r in okr),
            "torque_transverse_rel_max": max(r.get("torque_transverse_rel", 0.0) for r in okr), "torque_transverse_rel_median": float(np.median([r.get("torque_transverse_rel", 0.0) for r in okr if r["load"] == "tor"])),
            "n_fixed_and_loaded_cells_total": sum(r["n_fixed_and_loaded_cells"] for r in okr), "n_nan_replaced": sum(r["n_nan_replaced"] for r in okr),
            "u_bolt_absmax": max(r["u_bolt_absmax"] for r in okr), "tor_src": sorted({r["tor_src"] for r in raw_r if r["load"] == "tor"}),
            "volume_mm3_min": min(r["volume_mm3"] for r in okr),
            "n_points": [min(r["n_points"] for r in okr), int(np.median([r["n_points"] for r in okr])), max(r["n_points"] for r in okr)],
            "n_cells": [min(r["n_cells"] for r in okr), int(np.median([r["n_cells"] for r in okr])), max(r["n_cells"] for r in okr)],
            "n_fixed_cells_median": int(np.median([r["n_fixed_cells"] for r in okr])), "n_loaded_cells_median": int(np.median([r["n_loaded_cells"] for r in okr])),
            "extent_mm": [min(r["extent_mm"] for r in okr), float(np.median([r["extent_mm"] for r in okr])), max(r["extent_mm"] for r in okr)],
            "vm_max_by_load": {ld: max(r["vm_max"] for r in okr if r["load"] == ld) for ld in LOADS},
            "u_absmax_by_load": {ld: max(r["u_absmax"] for r in okr if r["load"] == ld) for ld in LOADS}}
    st["_diagnostics"] = diag
    manifest["_stats"] = {"train": st}
    manifest["_stats_by_load"] = {ld: accumulate(manifest[f"train_{ld}"]) for ld in LOADS}
    manifest["_load_cases"] = {ld: {"type": k, "direction": list(d), "magnitude": m, "unit": "N" if k == "force" else "N*mm"} for ld, (k, d, m) in LOAD_SPEC.items()}
    manifest["_note"] = ("DeepJEB++ v1.1 fields (HF KAIST-SmartDesignLab/DeepJEB-PP main, ODC-By 1.0), 15,360 brackets x 4 load cases; dir name "
                         "<base>-<sub>-IS<aug>_<load>. Splits grouped by SimJEB seed pair (<base>-<sub>): test = seed-0 10% of brackets by whole "
                         "pairs, train = rest; train_n256/1024/4096 = nested seed-0 prefixes of the train brackets (files = 4x); <split>_<load> = "
                         "one load case; official_* = the authors' data_splits.csv. _stats pooled over train cells (von_mises MPa, displacement mm, "
                         "area mm^2) or loaded cells (bc_load_n, bc_load_mag MPa).")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("split sizes", {k: len(v) for k, v in manifest.items() if isinstance(v, list)})
    print("FINALIZE-DONE", json.dumps(st, indent=1))


if args.finalize:
    finalize(); raise SystemExit

labels = {r["case"]: r for r in csv.DictReader(open(raw / "deepjebpp_labels.csv"))}
if args.rescore:
    todo = sorted(labels)[args.shard::args.nshards]
    if args.limit:
        todo = todo[: args.limit]
    results = []; t0 = time.time()
    for k, case in enumerate(todo):
        for ld in LOADS:
            name = f"{bracket_of(case)[0]}_{ld}"
            try:
                results.append(rescore_file(name, labels[case]))
            except Exception as e:  # noqa: BLE001
                results.append({"case": name, "status": f"error: {e!r}"})
        if k % 50 == 0:
            print(json.dumps({kk: v for kk, v in results[-1].items() if kk != "sums"}), f"[rescored {k + 1}/{len(todo)} brackets {time.time() - t0:.0f}s]", flush=True)
    (out / f"convert_results_rescore_{args.shard:02d}.json").write_text(json.dumps(results))
    print("RESCORE-SHARD-DONE", args.shard, len(results), "errors", sum(str(r.get("status")).startswith("error") for r in results))
    raise SystemExit

fields = index_files("3_fea_fields", ".npz"); bcs = index_files("2_boundary_conditions", ".npz")
tors = index_files("3_fea_fields_torsion_z_v1.1", ".npz") if (raw / "3_fea_fields_torsion_z_v1.1").exists() else {}
if args.repair:
    broken = {}
    for case in sorted(labels):
        b = bracket_of(case)[0]
        bad = [ld for ld in LOADS if not is_complete(f"{b}_{ld}")]
        if bad:
            broken[case] = bad
    print("incomplete files:", sum(len(v) for v in broken.values()), {bracket_of(c)[0]: v for c, v in broken.items()}, flush=True)
    results = []
    for case, bad in broken.items():
        b = bracket_of(case)[0]
        for ld in bad:
            d = out / f"{b}_{ld}"
            if d.exists():
                make_writable(d); shutil.rmtree(d / f"{b}_{ld}.pdmsh", ignore_errors=True)
        try:
            rs = convert_bracket(case, fields[case], bcs[case], tors.get(case), labels[case], only=bad)
        except Exception as e:  # noqa: BLE001
            rs = [{"case": f"{b}_{ld}", "status": f"error: {e!r}"} for ld in bad]
        results += rs
        print(json.dumps([{kk: v for kk, v in r.items() if kk != "sums"} for r in rs]), flush=True)
    prev = out / "convert_results_repair.json"
    old = json.load(open(prev)) if prev.exists() else []
    prev.write_text(json.dumps(old + results))
    print("REPAIR-DONE", len(results), "errors", sum(str(r.get("status")).startswith("error") for r in results))
    raise SystemExit
print("counts fields", len(fields), "bc", len(bcs), "torsion patch", len(tors), "labels", len(labels), flush=True)
assert len(fields) == len(bcs) == len(labels) == N_CASES_EXPECTED, (len(fields), len(bcs), len(labels))
assert set(fields) == set(bcs) == set(labels)
if tors:
    assert set(tors) == set(fields), "torsion patch case set differs"
todo = sorted(fields)[args.shard::args.nshards]
if args.limit:
    todo = todo[: args.limit]
results = []; t0 = time.time(); n_ok = 0
for k, case in enumerate(todo):
    try:
        rs = convert_bracket(case, fields[case], bcs[case], tors.get(case), labels[case])
    except Exception as e:  # noqa: BLE001
        rs = [{"case": case, "status": f"error: {e!r}"}]
    results += rs; n_ok += sum(r.get("status") == "ok" for r in rs)
    if k % 20 == 0 or any(str(r.get("status")).startswith("error") for r in rs):
        print(json.dumps({kk: v for kk, v in rs[-1].items() if kk != "sums"}), f"[done {k + 1}/{len(todo)} brackets, {n_ok} files, {time.time() - t0:.0f}s]", flush=True)
(out / f"convert_results_{args.shard:02d}.json").write_text(json.dumps(results))
print("CONVERT-SHARD-DONE", args.shard, len(results), "errors", sum(str(r.get("status")).startswith("error") for r in results))
