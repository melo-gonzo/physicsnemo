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

# ------------------------------------------------------------------------------------------------------- AeroZoo
# AeroZoo v1.2 (3,816 steady RANS cases, each a different body in its own enclosure at its own scale, Reynolds number
# and pose). Four boundaries by role: inlet (velocity and turbulence prescribed), outlet (gauge pressure 0), no_slip
# and slip. A body's surface is split into no-slip and slip bands in two thirds of the cases (annotation.body_id >= 1
# on either boundary); the enclosure walls carry body_id 0 (the ground of a ground-effect box, the symmetry plane of a
# hemisphere, the duct walls); the slip boundary has no faces in a seventh of the cases. The stored frame is a random
# rotation of the posed problem, so every reduction below is built from the data alone (the inlet direction and the
# body), never from the stored axes.
AZ = f"{U}/wip_datasets/aerozoo-v1.2"                             # the v1.2 release root (index, splits, case directories)
AZ_MAN = f"{U}/datasets/aerozoo_v1.2_shuf_pdmsh/manifest.json"   # the recipe's manifest: <split>.<part> keys (build_manifest.py)
AZ_ROLES = ("inlet", "outlet", "no_slip", "slip")


def aerozoo_rows(split_key="full.val"):
    """One metadata row per released case of a manifest split part, in manifest order: splits/cases.json (bank, enclosure
    family, body layout, source geometry) joined with the case's case-manifest.json (profile, Re, reference length)."""
    from tensordict import TensorDict  # noqa: F401  (imported here so the other entries do not need it)
    man = json.load(open(AZ_MAN))
    cases = json.load(open(f"{AZ}/splits/cases.json"))["cases"]
    if isinstance(cases, list):
        cases = {c["case_id"]: c for c in cases}
    rows = []
    for c in man[split_key]:
        row = dict(cases[c]); d = f"{AZ}/shard-{row['shard']:03d}/{c}"
        m = json.load(open(f"{d}/case-manifest.json"))
        row.update(path=f"{d}/domain.pdmsh", profile=m["scenario"]["profile"], Re=float(m["physics"]["target_reynolds_number"]),
                   L_ref=float(m["physics"]["reference_length_m"]), n_regions=len(m["geometry"].get("regions", [])),
                   n_secondary_inlets=len(m["scenario"].get("secondary_inlet_patches", [])))
        rows.append(row)
    return rows


def aerozoo_reduce(path, n_body, n_slice, n_band, n_patch=None, with_mesh=False):
    """One case reduced to what the figures need, in the stored (world) frame, plus the plane frame to project it with.

    The mid-span plane is defined from the data alone, so it is the same construction in every pose: it passes through
    the bodies' area-weighted centroid, contains the reference (primary inlet) velocity direction and is normal to the
    bodies' longest cross-flow axis (the principal axis of the body cells' centroids after removing their component along
    the flow); for a car that is the vertical symmetry plane, for a wing the mid-span chordwise section. A cell is "cut"
    by the plane when its centre lies within half its size of it (size = volume^(1/3) for a cell, area^(1/2) for a
    face), so the slice is the set of cells the plane passes through at every refinement level.
    Returns (arrays, meta): body cells of both roles (0 no-slip, 1 slip), the enclosure faces cut by the plane
    (roles 0 inlet, 1 outlet, 2 no-slip wall, 3 slip wall), optionally a sparse subsample of every enclosure patch
    and the body's triangles, the interior cells cut by the plane with Cp = gauge pressure / q_ref and |U| / U_ref, and
    ``frame`` (rows: flow direction, in-plane cross-flow axis, plane normal) about ``body_centroid``."""
    from tensordict import TensorDict
    gt = TensorDict.load_memmap(f"{path}/_tensordict/global_data")
    G = lambda *k: npf(gt[k]).astype(np.float64)
    U_ref = G("reference", "velocity"); u_hat = U_ref / np.linalg.norm(U_ref)
    speed = float(G("reference", "speed")); L_ref = float(G("reference", "length")); q_ref = float(G("reference", "dynamic_pressure"))
    rho = float(G("physics", "density")); nu = float(G("physics", "kinematic_viscosity"))
    try:
        secondary = G("reference", "secondary_inlet_velocities").reshape(-1, 3).tolist()
    except Exception:
        secondary = []
    body, wall, counts, inlet_bc = [], [], {}, {}
    for role in AZ_ROLES:
        bdir = f"{path}/_tensordict/boundaries/{role}"
        n = int(json.load(open(f"{bdir}/_tensordict/meta.json"))["cells"]["shape"][0]); counts[role] = n
        if n == 0:
            continue
        m = Mesh.load(bdir); pts = npf(m.points).astype(np.float64); cells = npf(m.cells)
        tri = pts[cells]; cen = tri.mean(axis=1)
        size = np.sqrt(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1))
        cp = npf(m.cell_data["solution", "gauge_pressure"]).reshape(-1) / q_ref
        bid = npf(m.cell_data["annotation", "body_id"]).reshape(-1) if role in ("no_slip", "slip") else np.zeros(n, np.int64)
        if role == "inlet":
            inlet_bc = {k: float(np.median(npf(m.cell_data["prescribed", "dirichlet", k])))
                        for k in ("turbulent_kinetic_energy", "specific_dissipation_rate", "intermittency", "momentum_thickness_reynolds_number")}
        on = bid >= 1
        body.append(dict(cen=cen[on], cp=cp[on], size=size[on], bid=bid[on], role=np.full(int(on.sum()), 0 if role == "no_slip" else 1), cells=cells[on], pts=pts))
        wall.append(dict(cen=cen[~on], size=size[~on], role=np.full(int((~on).sum()), AZ_ROLES.index(role))))
        del m, tri
    cat = lambda items, k: np.concatenate([it[k] for it in items], axis=0)
    bcen, bcp, bsize, bid, brole = (cat(body, k) for k in ("cen", "cp", "size", "bid", "role"))
    w = bsize ** 2
    c = (bcen * w[:, None]).sum(0) / w.sum()
    P = bcen - c; P = P - np.outer(P @ u_hat, u_hat)
    n_hat = np.linalg.eigh((P * w[:, None]).T @ P)[1][:, -1]
    n_hat -= (n_hat @ u_hat) * u_hat; n_hat /= np.linalg.norm(n_hat); e2 = np.cross(n_hat, u_hat)
    # interior: every cell the plane cuts, Cp and speed ratio there
    it = TensorDict.load_memmap(f"{path}/_tensordict/interior/_tensordict")
    pts = npf(it["points"]).astype(np.float64); n_int = pts.shape[0]
    ibb = np.stack([pts.min(0), pts.max(0)])
    # the in-plane cross-flow axis points away from the enclosure's walls (the ground of a ground-effect box, the symmetry
    # plane of a hemisphere: no-slip walls first, else slip walls), so that they sit below the body in the figures; with no
    # wall it points from the body toward the domain's centre
    walls = [np.concatenate([it["cen"][it["role"] == r] for it in wall]) for r in (2, 3)]
    walls = [wc for wc in walls if wc.shape[0]]
    anchor, anchor_above = (walls[0].mean(0), False) if walls else (0.5 * (ibb[0] + ibb[1]), True)
    if ((anchor - c) @ e2 > 0) != anchor_above:
        e2, n_hat = -e2, -n_hat
    size = np.cbrt(npf(it["point_data", "_target_quadrature_measure"]).astype(np.float64))
    sel = np.flatnonzero(np.abs((pts - c) @ n_hat) <= 0.5 * size); n_cut = int(sel.size)
    sel = sel[sub(sel.size, n_slice)]; sel_t = torch.as_tensor(sel)
    sp = pts[sel]; ssize = size[sel]
    sU = npf(it["point_data", "solution", "velocity"][sel_t]); spp = npf(it["point_data", "solution", "gauge_pressure"][sel_t]).reshape(-1)
    del it, pts, size
    # enclosure faces the plane cuts (the outline of the domain in the figure), per role
    wcen, wsize, wrole = (cat(wall, k) for k in ("cen", "size", "role"))
    band = np.flatnonzero(np.abs((wcen - c) @ n_hat) <= 0.6 * wsize)
    band = np.concatenate([band[wrole[band] == r][sub(int((wrole[band] == r).sum()), n_band)] for r in range(4)]) if band.size else band
    bsel = sub(bcen.shape[0], n_body)
    f32 = lambda a: np.asarray(a, np.float32)
    arrays = dict(body_points=f32(bcen[bsel]), body_cp=f32(bcp[bsel]), body_role=brole[bsel].astype(np.int8), body_id=bid[bsel].astype(np.int32), body_size=f32(bsize[bsel]),
                  band_points=f32(wcen[band]), band_role=wrole[band].astype(np.int8), band_size=f32(wsize[band]),
                  slice_points=f32(sp), slice_cp=f32(spp / q_ref), slice_umag=f32(np.linalg.norm(sU, axis=1) / speed), slice_size=f32(ssize),
                  frame=f32(np.stack([u_hat, e2, n_hat])), body_centroid=f32(c), interior_bbox=f32(ibb), body_bbox=f32(np.stack([bcen.min(0), bcen.max(0)])), U_ref=f32(U_ref))
    if n_patch:
        psel = np.concatenate([np.flatnonzero(wrole == r)[sub(int((wrole == r).sum()), n_patch)] for r in range(4)])
        arrays.update(patch_points=f32(wcen[psel]), patch_role=wrole[psel].astype(np.int8), patch_size=f32(wsize[psel]))
    if with_mesh:                                          # the body's triangles (re-indexed), for a solid rendering
        mp, mc, mr, off = [], [], [], 0
        for b in body:
            uniq, inv = np.unique(b["cells"], return_inverse=True)
            mp.append(b["pts"][uniq]); mc.append(inv.reshape(-1, 3) + off); mr.append(b["role"]); off += uniq.size
        arrays.update(body_mesh_points=f32(np.concatenate(mp)), body_mesh_cells=np.concatenate(mc).astype(np.int32), body_mesh_role=np.concatenate(mr).astype(np.int8))
    meta = dict(n_boundary_cells=int(sum(counts.values())), n_boundary_cells_by_role=counts, n_body_cells=int(bcen.shape[0]), n_body_stored=int(bsel.size),
                n_bodies=int(np.unique(bid).size), n_interior_points=int(n_int), n_slice_cut=n_cut, n_slice_stored=int(sel.size), n_band_stored=int(band.size),
                U_inf=U_ref.tolist(), U_ref=speed, L_ref=L_ref, q_ref=q_ref, rho=rho, nu=nu, Re=speed * L_ref / nu,
                body_acceleration=G("physics", "body_acceleration").tolist(), closure_model_id=int(G("physics", "closure_model_id")),
                secondary_inlet_velocities=secondary, inlet=inlet_bc, body_centroid=c.tolist(), frame=np.stack([u_hat, e2, n_hat]).tolist(),
                slice_rule="cells whose centre lies within half their size (volume^(1/3); area^(1/2) for faces) of the plane through the bodies' "
                           "area-weighted centroid that contains the inlet direction and is normal to the bodies' longest cross-flow axis",
                ranges=ranges(body_cp=bcp[bsel], slice_cp=spp / q_ref, slice_umag=np.linalg.norm(sU, axis=1) / speed))
    return arrays, meta


if want("aerozoo"):
    rows = aerozoo_rows("full.val")
    # the representative case: a procedural car in an opposed-hemisphere enclosure (a non-box enclosure, so the inlet and
    # outlet arcs, the symmetry plane and the body's two bands all appear in one plane), the one of those whose Reynolds
    # number is nearest the split's median log10 Re
    med = float(np.median(np.log10([r["Re"] for r in rows])))
    cand = [r for r in rows if r["bank"] == "procedural-cars" and r["family"] == "external_opposed_hemisphere"]
    row = min(cand, key=lambda r: (abs(np.log10(r["Re"]) - med), r["case_id"]))
    print("aerozoo case", row["case_id"], row["bank"], row["family"], row["profile"], "Re", row["Re"], flush=True)
    arrays, m = aerozoo_reduce(row["path"], n_body=80_000, n_slice=80_000, n_band=20_000, n_patch=12_000, with_mesh=True)
    write("aerozoo", arrays, dict(dataset="AeroZoo v1.2", case=row["case_id"], split="full.val", n_val=len(rows), bank=row["bank"], enclosure=row["family"],
                                  profile=row["profile"], body_layout=row["body_layout"], source_geometry=row["source_geometry"],
                                  n_surface_cells=m["n_boundary_cells"], units="m",
                                  selection=f"the procedural car in an opposed-hemisphere enclosure whose log10 Re is nearest the full.val median ({med:.2f}); "
                                            f"{len(cand)} candidates",
                                  description="steady transition-SST RANS (time-averaged) around one body in its own enclosure; inlet velocity and turbulence, "
                                              "outlet gauge pressure 0, no-slip and slip walls; the body is split into a no-slip and a slip band; density, "
                                              "viscosity, scale and pose vary per case",
                                  **m))
    print("aerozoo done", time.time() - t0, flush=True)
print("ALL DONE", time.time() - t0, flush=True)
