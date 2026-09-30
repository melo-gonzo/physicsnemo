"""Problem-figure artifacts (2026-09-22): one representative sample per dataset, reduced to what the book's
"the problem" figures need (geometry, boundary-condition labels, one solution field), written as small .npz + .json
files under --out. Read-only on the datasets. Every dataset is loaded from its native DomainMesh files, so the
fields are the raw CFD/FEM outputs before any recipe normalization; nondimensional coefficients (Cp) are formed here
from the case's own global_data so the pictures carry physical meaning.

usage: python problem_figures.py --out <dir> [--only name,name]
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch
from physicsnemo.mesh import DomainMesh, Mesh

U = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe"
H = "/home/psharpe/coreai_modulus_cae/datasets"
HL = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-HighLiftAeroML"

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--only", default="")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
only = set(args.only.split(",")) if args.only else None
rng = np.random.default_rng(0)


def npf(t):
    return np.asarray(t.detach().cpu().numpy() if torch.is_tensor(t) else t)


def centroids(points, cells):
    return npf(points)[npf(cells)].mean(axis=1)


def sub(n, k):
    return np.sort(rng.choice(n, size=k, replace=False)) if n > k else np.arange(n)


def gd(dm, k, default=None):
    try:
        return npf(dm.global_data[k]).astype(np.float64)
    except Exception:
        return default


def write(name, arrays, meta):
    np.savez_compressed(out / f"{name}.npz", **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"wrote {name}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in arrays.items()), flush=True)


def ranges(**fields):
    return {k: [float(np.nanmin(v)), float(np.nanmax(v))] for k, v in fields.items()}


def want(name):
    return only is None or name in only


t0 = time.time()

# ----------------------------------------------------------------------------------------------------- DrivAerML
if want("drivaer"):
    case = "run_112"   # the first case of the recipe's 48-car validation split (manifest.json: val[0])
    dm = DomainMesh.load(f"{H}/PhysicsNeMo-DrivaerML/{case}/domain_{case}.pdmsh")
    U_inf = gd(dm, "U_inf"); rho = float(gd(dm, "rho_inf")); p_inf = float(gd(dm, "p_inf", np.array(0.0)))
    q_inf = 0.5 * rho * float(np.linalg.norm(U_inf)) ** 2
    veh = dm.boundaries["vehicle"]
    cen = centroids(veh.points, veh.cells)
    p = npf(veh.cell_data["pMeanTrim"]).reshape(-1)
    wss = npf(veh.cell_data["wallShearStressMeanTrim"])
    n_cells = cen.shape[0]
    idx = sub(n_cells, 150_000)
    bbox = np.stack([cen.min(0), cen.max(0)])
    # symmetry-plane band of the volume mesh (the Volume task's query domain), clipped to a window around the car
    pts = dm.interior.points
    n_int = pts.shape[0]
    ys = npf(pts[:, 1]).reshape(-1)
    band = np.abs(ys) < 0.01
    xs = npf(pts[:, 0]).reshape(-1); zs = npf(pts[:, 2]).reshape(-1)
    L = bbox[1, 0] - bbox[0, 0]
    win = band & (xs > bbox[0, 0] - 0.4 * L) & (xs < bbox[1, 0] + 1.0 * L) & (zs < bbox[1, 2] + 0.6 * L) & (zs > bbox[0, 2] - 0.05 * L)
    sel = np.flatnonzero(win)
    sel = sel[sub(sel.size, 80_000)]
    sel_t = torch.as_tensor(sel)
    sp = npf(pts[sel_t]); sU = npf(dm.interior.point_data["UMeanTrim"][sel_t]); spp = npf(dm.interior.point_data["pMeanTrim"][sel_t]).reshape(-1)
    int_bbox = np.stack([np.array([xs.min(), ys.min(), zs.min()]), np.array([xs.max(), ys.max(), zs.max()])])
    write("drivaer", dict(surf_points=cen[idx].astype(np.float32), surf_cp=((p[idx] - p_inf) / q_inf).astype(np.float32),
                          surf_cf=(np.linalg.norm(wss[idx], axis=1) / q_inf).astype(np.float32),
                          slice_points=sp.astype(np.float32), slice_cp=((spp - p_inf) / q_inf).astype(np.float32),
                          slice_umag=(np.linalg.norm(sU, axis=1) / np.linalg.norm(U_inf)).astype(np.float32),
                          U_inf=U_inf.astype(np.float32), surf_bbox=bbox.astype(np.float32), interior_bbox=int_bbox.astype(np.float32)),
          dict(dataset="DrivAerML", case=case, split="val", n_surface_cells=int(n_cells), n_surface_stored=int(idx.size),
               n_interior_points=int(n_int), n_slice_stored=int(sel.size), slice_band_m=0.01, units="m",
               U_inf=U_inf.tolist(), rho_inf=rho, p_inf=p_inf, L_ref=float(gd(dm, "L_ref", np.array(np.nan))),
               ranges=ranges(surf_cp=(p[idx] - p_inf) / q_inf, slice_cp=(spp - p_inf) / q_inf, slice_umag=np.linalg.norm(sU, axis=1) / np.linalg.norm(U_inf))))
    del dm, pts
    print("drivaer done", time.time() - t0, flush=True)

# --------------------------------------------------------------------------------------------------- HiLiftAeroML
if want("hilift"):
    man = json.load(open(f"{HL}/manifest.json"))
    key = "full_val" if "full_val" in man else sorted(k for k in man if "val" in k)[0]
    import glob
    case, path = None, None
    for cand in man[key]:                      # the first validation case whose files are present on this machine
        hits = [p for p in glob.glob(f"{HL}/{cand}/*.pdmsh") if not p.endswith(".stl.pmsh")]
        if hits:
            case, path = cand, hits[0]; break
    print("hilift case", case, path, flush=True)
    # the recipe's surface reader reads the boundary tensordict alone and merges the parent's global_data; do the same
    # (the full DomainMesh load fails on this machine's copy, whose interior is not complete for every case)
    from tensordict import TensorDict
    veh = Mesh.load(f"{path}/_tensordict/boundaries/vehicle")
    glob_td = TensorDict.load_memmap(f"{path}/_tensordict/global_data")
    class _G:  # minimal stand-in for gd()
        global_data = glob_td
    dm = _G()
    U_inf = gd(dm, "U_inf"); rho = float(gd(dm, "rho_inf")); p_inf = float(gd(dm, "p_inf"))
    q_inf = 0.5 * rho * float(np.linalg.norm(U_inf)) ** 2
    cen = centroids(veh.points, veh.cells) * 0.0254   # inch -> m
    P = npf(veh.cell_data["PROJ(AVG(P))"]).reshape(-1)
    n_cells = cen.shape[0]
    idx = sub(n_cells, 150_000)
    aoa = float(case.split("AoA_")[-1]) if "AoA_" in case else float("nan")
    write("hilift", dict(surf_points=cen[idx].astype(np.float32), surf_cp=((P[idx] - p_inf) / q_inf).astype(np.float32),
                         U_inf=U_inf.astype(np.float32), surf_bbox=np.stack([cen.min(0), cen.max(0)]).astype(np.float32)),
          dict(dataset="HiLiftAeroML", case=case, split=key, n_surface_cells=int(n_cells), n_surface_stored=int(idx.size),
               units="m (converted from inch)", U_inf=U_inf.tolist(), rho_inf=rho, p_inf=p_inf, aoa_deg=aoa,
               L_ref_inch=float(gd(dm, "L_ref", np.array(np.nan))), ranges=ranges(surf_cp=(P[idx] - p_inf) / q_inf)))
    del dm
    print("hilift done", time.time() - t0, flush=True)

# ------------------------------------------------------------------------------------------------------ SHIFT-SUV
if want("shift_suv"):
    case = "run_00001"
    dm = DomainMesh.load(f"{H}/PhysicsNeMo-ShiftSUV/fastback/{case}/domain.pdmsh")
    U_inf = gd(dm, "U_inf"); rho = float(gd(dm, "rho_inf")); p_inf = float(gd(dm, "p_inf"))
    q_inf = 0.5 * rho * float(np.linalg.norm(U_inf)) ** 2
    veh = dm.boundaries["vehicle"]
    cen = centroids(veh.points, veh.cells)
    p = npf(veh.cell_data["pressure_average"]).reshape(-1)
    n_cells = cen.shape[0]
    idx = sub(n_cells, 150_000)
    write("shift_suv", dict(surf_points=cen[idx].astype(np.float32), surf_cp=((p[idx] - p_inf) / q_inf).astype(np.float32),
                            U_inf=U_inf.astype(np.float32), surf_bbox=np.stack([cen.min(0), cen.max(0)]).astype(np.float32)),
          dict(dataset="SHIFT-SUV", family="fastback", case=case, n_surface_cells=int(n_cells), n_surface_stored=int(idx.size),
               units="m", U_inf=U_inf.tolist(), rho_inf=rho, p_inf=p_inf, ranges=ranges(surf_cp=(p[idx] - p_inf) / q_inf)))
    del dm
    print("shift done", time.time() - t0, flush=True)

# ------------------------------------------------------------------------------------------------------- AirfRANS
if want("airfrans"):
    man = json.load(open(f"{U}/datasets/airfrans_pdmsh/manifest.json"))
    case = man["full_test"][0]
    dm = DomainMesh.load(f"{U}/datasets/airfrans_pdmsh/{case}/{case}.pdmsh")
    af = dm.boundaries["airfoil"]
    bp = npf(af.points); bc = npf(af.cells); bn = npf(af.cell_data["normals"]); bw = npf(af.cell_data["quadrature_measure"])
    ip = npf(dm.interior.points); vel = npf(dm.interior.point_data["velocity"]); pr = npf(dm.interior.point_data["pressure"]).reshape(-1)
    nut = npf(dm.interior.point_data["nut"]).reshape(-1)
    Umag = float(gd(dm, "U_inf_mag")); Udir = gd(dm, "U_inf_dir")
    win = (ip[:, 0] > -0.75) & (ip[:, 0] < 2.0) & (np.abs(ip[:, 1]) < 0.9)
    sel = np.flatnonzero(win); sel = sel[sub(sel.size, 70_000)]
    write("airfrans", dict(bnd_points=bp.astype(np.float32), bnd_cells=bc.astype(np.int32), bnd_normals=bn.astype(np.float32), bnd_measure=bw.astype(np.float32),
                           int_points=ip[sel].astype(np.float32), int_cp=(pr[sel] / (0.5 * Umag ** 2)).astype(np.float32),
                           int_umag=(np.linalg.norm(vel[sel], axis=1) / Umag).astype(np.float32), int_nut=nut[sel].astype(np.float32),
                           U_inf_dir=Udir.astype(np.float32), interior_bbox=np.stack([ip.min(0), ip.max(0)]).astype(np.float32)),
          dict(dataset="AirfRANS", case=case, split="full_test", n_boundary_cells=int(bc.shape[0]), n_interior_points=int(ip.shape[0]),
               n_interior_stored=int(sel.size), units="m (chord 1)", U_inf_mag=Umag, aoa_deg=float(gd(dm, "aoa_deg")), Re=float(gd(dm, "Re")),
               ranges=ranges(int_cp=pr[sel] / (0.5 * Umag ** 2), int_umag=np.linalg.norm(vel[sel], axis=1) / Umag, int_nut=nut[sel])))
    print("airfrans done", time.time() - t0, flush=True)

# ---------------------------------------------------------------------------------------------------- Tensile2d
if want("tensile2d"):
    man = json.load(open(f"{U}/datasets/plaid_tensile2d_pdmsh/manifest.json"))
    case = man["val_100"][0]
    dm = DomainMesh.load(f"{U}/datasets/plaid_tensile2d_pdmsh/{case}/{case}.pdmsh")
    ob = dm.boundaries["outer"]
    arrays = dict(bnd_points=npf(ob.points).astype(np.float32), bnd_cells=npf(ob.cells).astype(np.int32), bnd_normals=npf(ob.cell_data["normals"]).astype(np.float32))
    for k in ("bc_loaded", "bc_fixed", "bc_pinned", "bc_pressure", "quadrature_measure"):
        arrays[k] = npf(ob.cell_data[k]).reshape(-1).astype(np.float32)
    arrays["int_points"] = npf(dm.interior.points).astype(np.float32)
    for k in ("sig11", "sig22", "sig12", "q"):
        arrays[k] = npf(dm.interior.point_data[k]).reshape(-1).astype(np.float32)
    arrays["displacement"] = npf(dm.interior.point_data["displacement"]).astype(np.float32)
    glob = {k: float(gd(dm, k)) for k in ("P", "p1", "p2", "p3", "p4", "p5")}
    write("tensile2d", arrays, dict(dataset="PLAID Tensile2d", case=case, split="val_100", n_boundary_cells=int(arrays["bnd_cells"].shape[0]),
                                    n_nodes=int(arrays["int_points"].shape[0]), units="dataset units (mm-scale lengths, MPa stresses)",
                                    globals=glob, ranges=ranges(sig22=arrays["sig22"], sig11=arrays["sig11"], sig12=arrays["sig12"],
                                                                umag=np.linalg.norm(arrays["displacement"], axis=1))))
    print("tensile2d done", time.time() - t0, flush=True)

# ---------------------------------------------------------------------------------------------------------- Lug
if want("lug"):
    man = json.load(open(f"{U}/datasets/ginot_lug_pdmsh/manifest.json"))
    case = man["test"][0]
    dm = DomainMesh.load(f"{U}/datasets/ginot_lug_pdmsh/{case}/{case}.pdmsh")
    sf = dm.boundaries["surface"]
    arrays = dict(bnd_points=npf(sf.points).astype(np.float32), bnd_cells=npf(sf.cells).astype(np.int32), bnd_normals=npf(sf.cell_data["normals"]).astype(np.float32))
    for k in ("bc_fixed", "bc_hole", "bc_loaded", "bc_pressure", "quadrature_measure"):
        arrays[k] = npf(sf.cell_data[k]).reshape(-1).astype(np.float32)
    arrays["int_points"] = npf(dm.interior.points).astype(np.float32)
    arrays["von_mises"] = npf(dm.interior.point_data["von_mises"]).reshape(-1).astype(np.float32)
    glob = {k: float(gd(dm, k)) for k in ("hole_radius", "length", "thickness", "pressure", "pressure_mpa")}
    write("lug", arrays, dict(dataset="GINOT Bracket Lug", case=case, split="test", n_boundary_cells=int(arrays["bnd_cells"].shape[0]),
                              n_nodes=int(arrays["int_points"].shape[0]), units="m, MPa", globals=glob, ranges=ranges(von_mises=arrays["von_mises"])))
    print("lug done", time.time() - t0, flush=True)

# --------------------------------------------------------------------------------------------------------- OLBC
for name, root, bkeys, ikeys in (
    ("olbc_poisson", f"{U}/datasets/olbc_poisson_pdmsh", ("bc_dirichlet", "bc_neumann", "bc_robin", "bc_value_g", "bc_robin_alpha", "loop_id", "quadrature_measure"), ("u", "source")),
    ("olbc_elasticity", f"{U}/datasets/olbc_elasticity_pdmsh", ("bc_dirichlet_x", "bc_neumann_x", "bc_disp_x", "bc_trac_x", "bc_dirichlet_y", "bc_neumann_y", "bc_disp_y", "bc_trac_y", "loop_id", "quadrature_measure"), ("sig11", "sig22", "sig12", "displacement")),
):
    if not want(name):
        continue
    man = json.load(open(f"{root}/manifest.json"))
    case = man["test"][0]
    dm = DomainMesh.load(f"{root}/{case}/{case}.pdmsh")
    ob = dm.boundaries["outer"]
    arrays = dict(bnd_points=npf(ob.points).astype(np.float32), bnd_cells=npf(ob.cells).astype(np.int32), bnd_normals=npf(ob.cell_data["normals"]).astype(np.float32))
    for k in bkeys:
        arrays[k] = npf(ob.cell_data[k]).astype(np.float32)
    arrays["int_points"] = npf(dm.interior.points).astype(np.float32)
    for k in ikeys:
        arrays[k] = npf(dm.interior.point_data[k]).astype(np.float32)
    r = {k: arrays[k] for k in ikeys if arrays[k].ndim == 1}
    if "displacement" in arrays:
        r["umag"] = np.linalg.norm(arrays["displacement"], axis=1)
    write(name, arrays, dict(dataset=f"OLBC {'poisson-square-bc5' if 'poisson' in name else 'elasticity-circlehollow-m1'}", case=case, split="test",
                             n_boundary_cells=int(arrays["bnd_cells"].shape[0]), n_nodes=int(arrays["int_points"].shape[0]),
                             units="dimensionless (file units)", ranges=ranges(**r)))
    print(name, "done", time.time() - t0, flush=True)
print("ALL DONE", time.time() - t0, flush=True)
