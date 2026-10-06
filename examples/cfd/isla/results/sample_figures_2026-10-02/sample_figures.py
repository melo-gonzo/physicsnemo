"""Sample-figure artifacts (2026-10-02): several samples per dataset, reduced to what the book's "several samples"
figures need (geometry, boundary-condition labels and values, the global parameters, one solution field per sample),
written as one small .npz + .json per dataset under --out. Read-only on the datasets; every case is loaded from its
native DomainMesh files, so the fields are the raw CFD/FEM outputs before any recipe normalization, and nondimensional
coefficients (Cp) are formed here from the case's own global_data.

The cases are chosen by a deterministic rule per dataset, recorded in the json under "selection", so that the figures
span the parameter ranges rather than showing six arbitrary neighbours:

  drivaer    six of the 48 validation cars at evenly spaced positions of the manifest's val list; the first and the
             last also carry a symmetry-plane band of the volume mesh (the Volume task's query domain)
  hilift     three geometries (min / median / max flap area share among the geometries that carry an element-label file; 15 do)
             at the lowest and the highest angle of attack (4 and 22 degrees)
  airfrans   six full_test cases: the four corners of the (Re, angle of attack) rectangle, nearest in normalized
             distance, and the two cases nearest the centre, one four-digit and one five-digit NACA section
  shift_suv  the first, middle and last validation case of each family (estate, fastback)
  tensile2d  six cases at the 0, 20, 40, 60, 80 and 100% quantiles of the waist width over all 500 cases
             (results/tensile2d_geometry_2026-10-02.json, shipped next to this script)
  lug        the test cases with the smallest and largest hole radius, thickness and pressure (six distinct cases)
  deepjeb    six clean test brackets at the 0, 20, 40, 60, 80 and 100% quantiles of bracket mass, each under one of the
             four load cases in turn (vertical, horizontal, diagonal force; torsion; then diagonal and torsion again)

usage: python sample_figures.py --out <dir> [--only name,name]
"""
from __future__ import annotations

import argparse, glob, json, os, time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from physicsnemo.mesh import DomainMesh, Mesh

U = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe"
H = "/home/psharpe/coreai_modulus_cae/datasets"
HL = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/PhysicsNeMo-HighLiftAeroML"
HERE = Path(__file__).resolve().parent

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


def glob_td(path):
    """The global_data TensorDict of a .pdmsh directory, read lazily (memmap)."""
    return TensorDict.load_memmap(f"{path}/_tensordict/global_data")


def g(td, k, default=None):
    try:
        return npf(td[k]).astype(np.float64)
    except Exception:
        return default


def write(name, arrays, meta):
    np.savez_compressed(out / f"{name}.npz", **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, default=float))
    mb = (out / f"{name}.npz").stat().st_size / 1e6
    print(f"wrote {name} ({mb:.1f} MB): " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in arrays.items()), flush=True)


def ranges(**fields):
    return {k: [float(np.nanmin(v)), float(np.nanmax(v))] for k, v in fields.items()}


def want(name):
    return only is None or name in only


def spread(n, k=6):
    """k evenly spaced indices of range(n), first and last included."""
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def quantile_picks(values, k=6):
    """Indices of the sorted values at the 0, 1/(k-1), ..., 100% quantiles (distinct)."""
    order = np.argsort(values)
    picks = [int(order[int(round(q * (len(order) - 1)))]) for q in np.linspace(0, 1, k)]
    seen, uniq = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def stack(items, key):
    return np.concatenate([it[key] for it in items], axis=0)


def offsets(items, key):
    return np.cumsum([0] + [len(it[key]) for it in items]).astype(np.int64)


t0 = time.time()

# ----------------------------------------------------------------------------------------------------- DrivAerML
if want("drivaer"):
    man = json.load(open(f"{H}/PhysicsNeMo-DrivaerML/manifest.json"))
    val = man["val"]
    picks = spread(len(val))
    cases = [val[i] for i in picks]
    with_slice = {cases[0], cases[-1]}
    per, slices, meta_cases = [], [], []
    for case in cases:
        path = f"{H}/PhysicsNeMo-DrivaerML/{case}/domain_{case}.pdmsh"
        gt = glob_td(path)
        U_inf = g(gt, "U_inf"); rho = float(g(gt, "rho_inf")); p_inf = float(g(gt, "p_inf", np.array(0.0)))
        q_inf = 0.5 * rho * float(np.linalg.norm(U_inf)) ** 2
        veh = Mesh.load(f"{path}/_tensordict/boundaries/vehicle")
        cen = centroids(veh.points, veh.cells)
        p = npf(veh.cell_data["pMeanTrim"]).reshape(-1)
        n_cells = cen.shape[0]
        idx = sub(n_cells, 60_000)
        bbox = np.stack([cen.min(0), cen.max(0)])
        per.append(dict(points=cen[idx].astype(np.float32), cp=((p[idx] - p_inf) / q_inf).astype(np.float32)))
        m = dict(case=case, val_index=int(val.index(case)), n_surface_cells=int(n_cells), n_surface_stored=int(idx.size),
                 bbox=bbox.tolist(), length=float(bbox[1, 0] - bbox[0, 0]), width=float(bbox[1, 1] - bbox[0, 1]),
                 height=float(bbox[1, 2] - bbox[0, 2]), U_inf=U_inf.tolist(), rho_inf=rho, p_inf=p_inf, has_slice=case in with_slice)
        del veh
        if case in with_slice:
            it = TensorDict.load_memmap(f"{path}/_tensordict/interior/_tensordict")
            pts = it["points"]
            xs = npf(pts[:, 0]).reshape(-1); ys = npf(pts[:, 1]).reshape(-1); zs = npf(pts[:, 2]).reshape(-1)
            L = bbox[1, 0] - bbox[0, 0]
            win = (np.abs(ys) < 0.01) & (xs > bbox[0, 0] - 0.4 * L) & (xs < bbox[1, 0] + 1.0 * L) & (zs < bbox[1, 2] + 0.6 * L) & (zs > bbox[0, 2] - 0.05 * L)
            sel = np.flatnonzero(win); sel = sel[sub(sel.size, 50_000)]
            sel_t = torch.as_tensor(sel)
            sp = npf(pts[sel_t]); sU = npf(it["point_data", "UMeanTrim"][sel_t])
            slices.append(dict(points=sp.astype(np.float32), umag=(np.linalg.norm(sU, axis=1) / np.linalg.norm(U_inf)).astype(np.float32), case=case))
            m["n_interior_points"] = int(xs.size); m["n_slice_stored"] = int(sel.size)
            del it, pts
        meta_cases.append(m)
        print("drivaer", case, time.time() - t0, flush=True)
    arrays = dict(surf_points=stack(per, "points"), surf_cp=stack(per, "cp"), surf_offsets=offsets(per, "points"),
                  slice_points=stack(slices, "points"), slice_umag=stack(slices, "umag"), slice_offsets=offsets(slices, "points"),
                  U_inf=np.array(meta_cases[0]["U_inf"], np.float32))
    write("drivaer", arrays, dict(dataset="DrivAerML", split="val", n_val=len(val), cases=meta_cases, slice_cases=[s["case"] for s in slices],
                                  selection="six of the 48 validation cars at evenly spaced positions (0, 9, 19, 28, 38, 47) of the manifest's val list; "
                                            "the first and the last carry a symmetry-plane band (|y| < 1 cm) of the volume mesh",
                                  slice_band_m=0.01, units="m", ranges=ranges(surf_cp=arrays["surf_cp"], slice_umag=arrays["slice_umag"])))
    print("drivaer done", time.time() - t0, flush=True)

# --------------------------------------------------------------------------------------------------- HiLiftAeroML
if want("hilift"):
    man = json.load(open(f"{HL}/manifest.json"))
    split_of = {}
    for k in ("full_train", "full_val", "full_test"):
        for c in man[k]:
            split_of[c] = k
    # the flap area share of every geometry, from the element-label files (one per case; the geometry is the same at every angle)
    shares = {}
    for d in sorted(glob.glob(f"{HL}/geo_LHC*_AoA_4")):
        geo = os.path.basename(d).split("_AoA_")[0]
        hits = glob.glob(f"{d}/*_element_labels.json")
        if hits:
            lab = json.load(open(hits[0]))
            shares[geo] = dict(flap=float(lab["area_share"]["flap"]), slat=float(lab["area_share"]["slat"]))
    geos = sorted(shares); flap = np.array([shares[k]["flap"] for k in geos])
    order = np.argsort(flap)
    chosen_geos = [geos[order[0]], geos[order[len(order) // 2]], geos[order[-1]]]
    aoas = [4, 22]
    per, meta_cases = [], []
    for geo in chosen_geos:
        for aoa in aoas:
            case = f"{geo}_AoA_{aoa}"
            hits = [p for p in glob.glob(f"{HL}/{case}/*.pdmsh") if not p.endswith(".stl.pmsh")]
            if not hits:
                print("hilift MISSING", case, flush=True); continue
            path = hits[0]
            gt = glob_td(path)
            U_inf = g(gt, "U_inf"); rho = float(g(gt, "rho_inf")); p_inf = float(g(gt, "p_inf"))
            q_inf = 0.5 * rho * float(np.linalg.norm(U_inf)) ** 2
            # the surface has ~280 million cells (20 GB); read only the sampled rows through raw memmaps of the tensordict files
            vd = f"{path}/_tensordict/boundaries/vehicle/_tensordict"
            vm = json.load(open(f"{vd}/meta.json")); cm = json.load(open(f"{vd}/cell_data/meta.json"))
            tdt = lambda s: dict(**{"torch.float32": np.float32, "torch.float64": np.float64, "torch.int64": np.int64, "torch.int32": np.int32})[s]
            cells = np.memmap(f"{vd}/cells.memmap", dtype=tdt(vm["cells"]["dtype"]), mode="r", shape=tuple(vm["cells"]["shape"]))
            points = np.memmap(f"{vd}/points.memmap", dtype=tdt(vm["points"]["dtype"]), mode="r", shape=tuple(vm["points"]["shape"]))
            Pm = np.memmap(f"{vd}/cell_data/PROJ(AVG(P)).memmap", dtype=tdt(cm["PROJ(AVG(P))"]["dtype"]), mode="r", shape=tuple(cm["PROJ(AVG(P))"]["shape"]))
            n_cells = cells.shape[0]
            idx = sub(n_cells, 60_000)
            tri = np.asarray(cells[idx]); cen = np.asarray(points[tri.reshape(-1)]).reshape(-1, 3, 3).mean(axis=1) * 0.0254   # inch -> m
            P = np.asarray(Pm[idx]).reshape(-1)
            per.append(dict(points=cen.astype(np.float32), cp=((P - p_inf) / q_inf).astype(np.float32), U=U_inf.astype(np.float32)))
            meta_cases.append(dict(case=case, geometry=geo, aoa_deg=float(aoa), split=split_of.get(case, "?"), n_surface_cells=int(n_cells),
                                   n_surface_stored=int(idx.size), U_inf=U_inf.tolist(), rho_inf=rho, p_inf=p_inf,
                                   flap_area_share=shares[geo]["flap"], slat_area_share=shares[geo]["slat"],
                                   bbox=np.stack([cen.min(0), cen.max(0)]).tolist()))
            del cells, points, Pm
            print("hilift", case, time.time() - t0, flush=True)
    arrays = dict(surf_points=stack(per, "points"), surf_cp=stack(per, "cp"), surf_offsets=offsets(per, "points"),
                  U_inf=np.stack([it["U"] for it in per]))
    write("hilift", arrays, dict(dataset="HiLiftAeroML", cases=meta_cases, geometries=chosen_geos, aoas_deg=aoas,
                                 selection=f"three geometries at the minimum, median and maximum flap area share among the {flap.size} geometries with an element-label file, "
                                           "each at the lowest and the highest angle of attack of the grid",
                                 flap_area_share_all=dict(min=float(flap.min()), median=float(np.median(flap)), max=float(flap.max()), n=int(flap.size)),
                                 units="m (converted from inch)", ranges=ranges(surf_cp=arrays["surf_cp"])))
    print("hilift done", time.time() - t0, flush=True)

# ------------------------------------------------------------------------------------------------------ SHIFT-SUV
if want("shift_suv"):
    per, meta_cases = [], []
    for fam in ("estate", "fastback"):
        man = json.load(open(f"{H}/PhysicsNeMo-ShiftSUV/{fam}/manifest.json"))
        val = man["val"]
        for i in spread(len(val), 3):
            case = val[i]
            path = f"{H}/PhysicsNeMo-ShiftSUV/{fam}/{case}/domain.pdmsh"
            gt = glob_td(path)
            U_inf = g(gt, "U_inf"); rho = float(g(gt, "rho_inf")); p_inf = float(g(gt, "p_inf"))
            q_inf = 0.5 * rho * float(np.linalg.norm(U_inf)) ** 2
            veh = Mesh.load(f"{path}/_tensordict/boundaries/vehicle")
            cen = centroids(veh.points, veh.cells)
            p = npf(veh.cell_data["pressure_average"]).reshape(-1)
            n_cells = cen.shape[0]
            idx = sub(n_cells, 60_000)
            bbox = np.stack([cen.min(0), cen.max(0)])
            per.append(dict(points=cen[idx].astype(np.float32), cp=((p[idx] - p_inf) / q_inf).astype(np.float32)))
            meta_cases.append(dict(case=case, family=fam, val_index=int(i), n_val=len(val), n_surface_cells=int(n_cells), n_surface_stored=int(idx.size),
                                   bbox=bbox.tolist(), length=float(bbox[1, 0] - bbox[0, 0]), width=float(bbox[1, 1] - bbox[0, 1]),
                                   height=float(bbox[1, 2] - bbox[0, 2]), U_inf=U_inf.tolist(), rho_inf=rho, p_inf=p_inf))
            del veh
            print("shift", fam, case, time.time() - t0, flush=True)
    arrays = dict(surf_points=stack(per, "points"), surf_cp=stack(per, "cp"), surf_offsets=offsets(per, "points"), U_inf=np.array(meta_cases[0]["U_inf"], np.float32))
    write("shift_suv", arrays, dict(dataset="SHIFT-SUV", cases=meta_cases,
                                    selection="the first, middle and last validation case of each family's manifest val list",
                                    units="m", ranges=ranges(surf_cp=arrays["surf_cp"])))
    print("shift done", time.time() - t0, flush=True)

# ------------------------------------------------------------------------------------------------------- AirfRANS
if want("airfrans"):
    man = json.load(open(f"{U}/datasets/airfrans_pdmsh/manifest.json"))
    test = man["full_test"]
    # the name encodes the inlet speed and the angle of attack, then the NACA parameters (three for four-digit, four for five-digit)
    toks = [c.split("_")[2:] for c in test]
    Uin = np.array([float(t[0]) for t in toks]); aoa = np.array([float(t[1]) for t in toks]); five = np.array([len(t) == 6 for t in toks])
    nu = 1.56e-5; Re = Uin / nu
    def nearest(re_t, aoa_t, mask=None):
        d = ((Re - re_t) / (Re.max() - Re.min())) ** 2 + ((aoa - aoa_t) / (aoa.max() - aoa.min())) ** 2
        if mask is not None:
            d = np.where(mask, d, np.inf)
        return int(np.argmin(d))
    rc, ac = 0.5 * (Re.min() + Re.max()), 0.5 * (aoa.min() + aoa.max())
    picks = [nearest(Re.min(), aoa.min()), nearest(Re.max(), aoa.min()), nearest(rc, ac, ~five), nearest(rc, ac, five),
             nearest(Re.min(), aoa.max()), nearest(Re.max(), aoa.max())]
    roles = ["low Re, low angle", "high Re, low angle", "centre, four-digit section", "centre, five-digit section", "low Re, high angle", "high Re, high angle"]
    per, meta_cases = [], []
    for i, role in zip(picks, roles):
        case = test[i]
        dm = DomainMesh.load(f"{U}/datasets/airfrans_pdmsh/{case}/{case}.pdmsh")
        af = dm.boundaries["airfoil"]
        bp = npf(af.points); bc = npf(af.cells)
        ip = npf(dm.interior.points); vel = npf(dm.interior.point_data["velocity"]); pr = npf(dm.interior.point_data["pressure"]).reshape(-1)
        Umag = float(g(dm.global_data, "U_inf_mag")); Udir = g(dm.global_data, "U_inf_dir")
        win = (ip[:, 0] > -0.5) & (ip[:, 0] < 1.6) & (np.abs(ip[:, 1]) < 0.6)
        sel = np.flatnonzero(win); sel = sel[sub(sel.size, 35_000)]
        per.append(dict(bp=bp.astype(np.float32), bc=bc.astype(np.int32), ip=ip[sel].astype(np.float32),
                        cp=(pr[sel] / (0.5 * Umag ** 2)).astype(np.float32), umag=(np.linalg.norm(vel[sel], axis=1) / Umag).astype(np.float32), udir=Udir.astype(np.float32)))
        meta_cases.append(dict(case=case, role=role, split="full_test", n_boundary_cells=int(bc.shape[0]), n_interior_points=int(ip.shape[0]), n_interior_stored=int(sel.size),
                               U_inf_mag=Umag, aoa_deg=float(g(dm.global_data, "aoa_deg")), Re=float(g(dm.global_data, "Re")), naca_five_digit=bool(five[i]),
                               naca_params=[float(x) for x in toks[i][2:]]))
        print("airfrans", case, time.time() - t0, flush=True)
    arrays = dict(bnd_points=stack(per, "bp"), bnd_cells=stack(per, "bc"), bnd_point_offsets=offsets(per, "bp"), bnd_cell_offsets=offsets(per, "bc"),
                  int_points=stack(per, "ip"), int_cp=stack(per, "cp"), int_umag=stack(per, "umag"), int_offsets=offsets(per, "ip"),
                  U_inf_dir=np.stack([it["udir"] for it in per]))
    write("airfrans", arrays, dict(dataset="AirfRANS", cases=meta_cases, split="full_test", n_test=len(test),
                                   selection="the four corners of the (Re, angle of attack) rectangle of the 200 full_test cases, nearest in normalized distance, "
                                             "and the two cases nearest its centre, one four-digit and one five-digit NACA section",
                                   test_ranges=dict(Re=[float(Re.min()), float(Re.max())], aoa_deg=[float(aoa.min()), float(aoa.max())], U_inf=[float(Uin.min()), float(Uin.max())]),
                                   units="m (chord 1)", ranges=ranges(int_cp=arrays["int_cp"], int_umag=arrays["int_umag"])))
    print("airfrans done", time.time() - t0, flush=True)

# ---------------------------------------------------------------------------------------------------- Tensile2d
if want("tensile2d"):
    G = json.load(open(HERE / "tensile2d_geometry_2026-10-02.json"))
    rows = [r for r in G["cases"] if "waist_width" in r]
    ww = np.array([r["waist_width"] for r in rows])
    picks = quantile_picks(ww)
    per, meta_cases = [], []
    for i in picks:
        case = rows[i]["case"]
        dm = DomainMesh.load(f"{U}/datasets/plaid_tensile2d_pdmsh/{case}/{case}.pdmsh")
        ob = dm.boundaries["outer"]
        it = dict(bp=npf(ob.points).astype(np.float32), bc=npf(ob.cells).astype(np.int32), ip=npf(dm.interior.points).astype(np.float32),
                  sig22=npf(dm.interior.point_data["sig22"]).reshape(-1).astype(np.float32))
        for k in ("bc_loaded", "bc_fixed", "bc_pinned", "bc_pressure"):
            it[k] = npf(ob.cell_data[k]).reshape(-1).astype(np.float32)
        per.append(it)
        glob = {k: float(g(dm.global_data, k)) for k in ("P", "p1", "p2", "p3", "p4", "p5")}
        meta_cases.append(dict(case=case, waist_width=float(ww[i]), n_boundary_cells=int(it["bc"].shape[0]), n_nodes=int(it["ip"].shape[0]), globals=glob))
        print("tensile2d", case, time.time() - t0, flush=True)
    arrays = dict(bnd_points=stack(per, "bp"), bnd_cells=stack(per, "bc"), bnd_point_offsets=offsets(per, "bp"), bnd_cell_offsets=offsets(per, "bc"),
                  int_points=stack(per, "ip"), sig22=stack(per, "sig22"), int_offsets=offsets(per, "ip"),
                  **{k: stack(per, k) for k in ("bc_loaded", "bc_fixed", "bc_pinned", "bc_pressure")})
    write("tensile2d", arrays, dict(dataset="PLAID Tensile2d", cases=meta_cases,
                                    selection="six cases at the 0, 20, 40, 60, 80 and 100% quantiles of the waist width over all 500 cases (results/tensile2d_geometry_2026-10-02.json)",
                                    units="dataset units (unit half-width; MPa stresses)", ranges=ranges(sig22=arrays["sig22"])))
    print("tensile2d done", time.time() - t0, flush=True)

# ---------------------------------------------------------------------------------------------------------- Lug
if want("lug"):
    man = json.load(open(f"{U}/datasets/ginot_lug_pdmsh/manifest.json"))
    test = man["test"]
    keys = ("hole_radius", "length", "thickness", "pressure_mpa")
    table = np.array([[float(g(glob_td(f"{U}/datasets/ginot_lug_pdmsh/{c}/{c}.pdmsh"), k)) for k in keys] for c in test])
    print("lug scanned", table.shape, time.time() - t0, flush=True)
    picks, roles = [], []
    for j, k in ((0, "hole_radius"), (2, "thickness"), (3, "pressure_mpa")):
        for which, idx in (("smallest", np.argsort(table[:, j])), ("largest", np.argsort(-table[:, j]))):
            for i in idx:
                if int(i) not in picks:
                    picks.append(int(i)); roles.append(f"{which} {k}"); break
    per, meta_cases = [], []
    for i, role in zip(picks, roles):
        case = test[i]
        dm = DomainMesh.load(f"{U}/datasets/ginot_lug_pdmsh/{case}/{case}.pdmsh")
        sf = dm.boundaries["surface"]
        it = dict(bp=npf(sf.points).astype(np.float32), bc=npf(sf.cells).astype(np.int32), ip=npf(dm.interior.points).astype(np.float32),
                  vm=npf(dm.interior.point_data["von_mises"]).reshape(-1).astype(np.float32))
        for k in ("bc_fixed", "bc_hole", "bc_loaded", "bc_pressure"):
            it[k] = npf(sf.cell_data[k]).reshape(-1).astype(np.float32)
        per.append(it)
        glob = {k: float(g(dm.global_data, k)) for k in ("hole_radius", "length", "thickness", "pressure", "pressure_mpa")}
        meta_cases.append(dict(case=case, role=role, split="test", n_boundary_cells=int(it["bc"].shape[0]), n_nodes=int(it["ip"].shape[0]), globals=glob))
        print("lug", case, role, time.time() - t0, flush=True)
    arrays = dict(bnd_points=stack(per, "bp"), bnd_cells=stack(per, "bc"), bnd_point_offsets=offsets(per, "bp"), bnd_cell_offsets=offsets(per, "bc"),
                  int_points=stack(per, "ip"), von_mises=stack(per, "vm"), int_offsets=offsets(per, "ip"),
                  **{k: stack(per, k) for k in ("bc_fixed", "bc_hole", "bc_loaded", "bc_pressure")})
    write("lug", arrays, dict(dataset="GINOT Bracket Lug", cases=meta_cases, split="test", n_test=len(test),
                              selection="the test cases with the smallest and the largest hole radius, thickness and pressure (six distinct cases)",
                              test_ranges={k: [float(table[:, j].min()), float(table[:, j].max())] for j, k in enumerate(keys)},
                              units="m (lengths), MPa (stress, pressure)", ranges=ranges(von_mises=arrays["von_mises"])))
    print("lug done", time.time() - t0, flush=True)

# ------------------------------------------------------------------------------------------------------ DeepJEB++
if want("deepjeb"):
    root = f"{U}/datasets/deepjebpp_pdmsh"
    man = json.load(open(f"{root}/manifest.json"))
    brackets = man["test_clean_brackets"]
    mass = np.array([float(g(glob_td(f"{root}/{b}_ver/{b}_ver.pdmsh"), "mass_g")) for b in brackets])
    print("deepjeb scanned", mass.shape, time.time() - t0, flush=True)
    picks = quantile_picks(mass)
    loads = ["ver", "hor", "dia", "tor", "dia", "tor"]
    per, meta_cases = [], []
    for i, load in zip(picks, loads):
        b = brackets[i]; case = f"{b}_{load}"
        path = f"{root}/{case}/{case}.pdmsh"
        sf = Mesh.load(f"{path}/_tensordict/boundaries/surface")
        gt = glob_td(path)
        cen = centroids(sf.points, sf.cells)
        it = dict(bp=npf(sf.points).astype(np.float32), bc=npf(sf.cells).astype(np.int32), cen=cen.astype(np.float32),
                  vm=npf(sf.cell_data["von_mises"]).reshape(-1).astype(np.float32),
                  umag=np.linalg.norm(npf(sf.cell_data["displacement"]), axis=1).astype(np.float32),
                  bc_fixed=npf(sf.cell_data["bc_fixed"]).reshape(-1).astype(np.float32), bc_loaded=npf(sf.cell_data["bc_loaded"]).reshape(-1).astype(np.float32),
                  bc_load_mag=npf(sf.cell_data["bc_load_mag"]).reshape(-1).astype(np.float32))
        per.append(it)
        lt = g(gt, "load_type")
        meta_cases.append(dict(case=case, bracket=b, load=load, load_case=man["_load_cases"][load], load_dir=g(gt, "load_dir").tolist(),
                               load_type_onehot=lt.tolist(), load_type="force" if lt[0] > 0.5 else "torque", load_magnitude=float(g(gt, "load_magnitude")),
                               mass_g=float(g(gt, "mass_g")), vol_mm3=float(g(gt, "vol_mm3")), lug_radius_mm=float(g(gt, "lug_radius_mm")),
                               n_cells=int(it["bc"].shape[0]), n_points=int(it["bp"].shape[0]), bbox_mm=np.stack([it["bp"].min(0), it["bp"].max(0)]).tolist(),
                               split="test_clean"))
        print("deepjeb", case, time.time() - t0, flush=True)
    arrays = dict(bnd_points=stack(per, "bp"), bnd_cells=stack(per, "bc"), bnd_point_offsets=offsets(per, "bp"), bnd_cell_offsets=offsets(per, "bc"),
                  centroids=stack(per, "cen"), von_mises=stack(per, "vm"), disp_mag=stack(per, "umag"),
                  **{k: stack(per, k) for k in ("bc_fixed", "bc_loaded", "bc_load_mag")})
    write("deepjeb", arrays, dict(dataset="DeepJEB++", cases=meta_cases, split="test_clean", n_test_brackets=len(brackets),
                                  selection="six clean test brackets at the 0, 20, 40, 60, 80 and 100% quantiles of bracket mass, each under one load case in turn "
                                            "(vertical, horizontal, diagonal force, torsion, diagonal, torsion)",
                                  test_mass_range_g=[float(mass.min()), float(mass.max())], load_cases=man["_load_cases"],
                                  units="mm (lengths), MPa (stress), mm (displacement)", ranges=ranges(von_mises=arrays["von_mises"], disp_mag=arrays["disp_mag"])))
    print("deepjeb done", time.time() - t0, flush=True)

print("ALL DONE", time.time() - t0, flush=True)
