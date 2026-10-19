"""Point-cloud renderings (FIG lane, 2026-10-18) for the datasets whose surface mesh is not on this machine: DeepJEB++
(bracket 60073, 611-092-IS06_hor) and SHIFT-SUV (run 00187), truth / ISLA / GeoTransolver / Transolver on the scored
points of one validation sample (dump.sbatch rows 9-11 and 18-20, LHR jobs 271987 and 275252).

Each scored point (a cell centroid of the surface; all three models see the same points, checked) is drawn as a small
shaded sphere, off-screen with VTK (pyvista, CPU llvmpipe), coloured by its value. Top row: the field, one shared scale
(viridis, 0.5-99.5th percentile of the truth). Bottom row: prediction minus truth, one shared symmetric scale (the
book's DIVERGING map, 99th percentile of the three errors). Panel titles carry the scorer's per-sample relative L2 of
that field (metrics.jsonl of the same run, normalized space, as in results/char).
DeepJEB++ coordinates are the pipeline's (centred, mm x 0.01); von Mises [MPa] and |u| [mm] as infer.py writes them.
SHIFT-SUV coordinates are infer.py's (divided by L_ref); c_p = (p - p_inf) / q_inf, q_inf = rho_inf |U_inf|^2 / 2.
usage: python render_cloud.py <dump_dir> <out_png_dir> <out_arrays_dir> [deepjebpp,shift_suv] [<deepjebpp surface dir>]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_lib as R  # noqa: E402
from render_2d import MODELS, load  # noqa: E402
from render_lib import palette  # noqa: E402


def panels(xyz, truth, preds, view, size, point_size, titles, lab, unit, out_png, suptitle, lo0=None, mesh=None):
    """mesh = (polydata, index of the scored point of each cell): draw the surface instead of the points."""
    lo, hi = np.percentile(truth, [0.5, 99.5])
    if lo0 is not None:
        lo = lo0
    errs = [p - truth for p in preds]
    v = float(np.percentile(np.abs(np.concatenate(errs)), 99))
    if mesh is None:
        draw = lambda f, cmap, lim: R.render_cloud(xyz, f, view, cmap, lim, size, point_size)  # noqa: E731
    else:
        draw = lambda f, cmap, lim: R.render(mesh[0], view, scalars=f[mesh[1]], cmap=cmap, clim=lim, size=size)  # noqa: E731
    top = [draw(f, palette.SEQUENTIAL, (lo, hi)) for f in [truth] + preds]
    bot = [draw(e, palette.DIVERGING, (-v, v)) for e in errs]
    allc = R.common_crop(top + bot)
    R.four_up(allc[:4], allc[4:], ["truth (solver)"] + titles, [t.split(" (")[0] + ": prediction $-$ truth" for t in titles],
            (palette.SEQUENTIAL, (lo, hi), f"{lab} {unit}"), (palette.DIVERGING, (-v, v), f"prediction $-$ truth {unit}"),
            out_png, suptitle)
    return {"clim": [float(lo), float(hi)], "err_clim": v}


def check_same_points(D):
    for k, _ in MODELS:
        assert np.allclose(D[k][0]["points"], D["isla"][0]["points"]), k


def deepjebpp(dump, out_png, out_arr, surface=None):
    """surface: the case's FEM surface (deepjebpp_pdmsh/<case>/<case>.pdmsh/_tensordict/boundaries/surface, copied from
    LHR); its cells map one to one onto the 50,024 scored centroids (x_mm = 100 x* + center), so each triangle shows its own
    value (vertex colours = the mean of the adjacent triangles)."""
    D = {k: load(dump, f"jeb_{k}") for k, _ in MODELS}
    check_same_points(D)
    xyz = D["isla"][0]["points"].astype(np.float64)
    sid = D["isla"][2]
    mesh, mesh_rec = None, None
    if surface is not None:
        from physicsnemo.mesh import Mesh

        m = Mesh.load(surface)
        V = (m.points.numpy().astype(np.float64) - D["isla"][0]["global.center"].astype(np.float64)) / 100.0
        F = m.cells.numpy().astype(np.int64)
        idx, dist = R.nn_map(xyz, np.arange(len(xyz)), V[F].mean(1))
        mesh = (R.polydata(V, F), idx)
        mesh_rec = {"surface": surface, "n_triangles": int(len(F)), "n_distinct_points_used": int(len(np.unique(idx))),
                    "centroid_match_distance_max_mm": float(100.0 * dist.max())}
    c = 0.5 * (xyz.min(0) + xyz.max(0))
    view = dict(position=(c[0] + 3.2, c[1] - 4.2, c[2] + 3.0), focal=tuple(c), up=(0, 0, 1), parallel=False, angle=24.0)
    rec = {"sample_id": sid, "source": R.provenance([(f"jeb_{k}", "native") for k, _ in MODELS]), "n_points_scored": int(len(xyz)),
           "load_dir": D["isla"][0]["global.load_dir"].tolist(), "load_magnitude": float(D["isla"][0]["global.load_magnitude"]),
           "per_sample_rel_l2": {f: {name: D[k][1]["metrics"][f] for k, name in MODELS} for f in ("von_mises_l2", "displacement_l2")},
           "views": {}, "mesh": mesh_rec}
    arrays = {"xyz": xyz.astype(np.float32)}
    for field, lab, unit, get, mkey in (
            ("von_mises", "von Mises stress", "[MPa]", lambda d, w: d[f"{w}_von_mises"], "von_mises_l2"),
            ("displacement", "displacement magnitude $|u|$", "[mm]", lambda d, w: np.linalg.norm(d[f"{w}_displacement"], axis=1),
             "displacement_l2")):
        truth = get(D["isla"][0], "true").astype(np.float64)
        preds = [get(D[k][0], "pred").astype(np.float64) for k, _ in MODELS]
        titles = [f"{name} (rel. L2 {D[k][1]['metrics'][mkey]:.3f})" for k, name in MODELS]
        rec["views"][field] = panels(xyz, truth, preds, view, (1500, 1300), 8.0, titles, lab, unit, f"{out_png}/deepjebpp_{field}.png",
                                     f"DeepJEB++ test bracket {sid.split('_')[0]} ({sid.split('_', 1)[1].rsplit('_', 2)[0]}, horizontal "
                                     f"load): the case of median ISLA von Mises error; {len(xyz):,} scored surface cells", lo0=0.0, mesh=mesh)
        arrays[f"{field}_true"] = truth.astype(np.float16)
        for (k, _), p in zip(MODELS, preds):
            arrays[f"{field}_{k}"] = p.astype(np.float16)
    np.savez_compressed(f"{out_arr}/deepjebpp_{sid.split('_')[0]}.npz", **arrays)
    return rec


def shift_suv(dump, out_png, out_arr):
    D = {k: load(dump, f"suv_{k}") for k, _ in MODELS}
    check_same_points(D)
    xyz = D["isla"][0]["points"].astype(np.float64)
    sid = D["isla"][2]
    g = D["isla"][0]
    U = np.asarray(g["global.U_inf"], np.float64)
    q = 0.5 * float(g["global.rho_inf"]) * float(U @ U)
    p_inf = float(g["global.p_inf"])
    c = 0.5 * (xyz.min(0) + xyz.max(0))
    view = dict(position=(c[0], c[1] - 10, c[2]), focal=tuple(c), up=(0, 0, 1), parallel=True, scale=0.21)
    rec = {"sample_id": sid, "source": R.provenance([(f"suv_{k}", "native") for k, _ in MODELS]), "n_points_scored": int(len(xyz)),
           "q_inf": q, "per_sample_rel_l2": {f: {name: D[k][1]["metrics"][f] for k, name in MODELS} for f in ("pressure_l2", "wss_l2")},
           "views": {}}
    left = xyz[:, 1] <= c[1] + 0.02  # the left half, so the far side's points do not show through gaps
    truth = (g["true_pressure"] - p_inf) / q
    preds = [(D[k][0]["pred_pressure"] - p_inf) / q for k, _ in MODELS]
    titles = [f"{name} (rel. L2 {D[k][1]['metrics']['pressure_l2']:.4f})" for k, name in MODELS]
    rec["views"]["cp_left_side"] = panels(xyz[left], truth[left], [p[left] for p in preds], view, (2400, 1000), 12.0, titles,
                                          "pressure coefficient $c_p$", "[–]", f"{out_png}/shift_suv_cp.png",
                                          f"SHIFT-SUV validation case {sid.split('_')[0]} (run 00187): the case of median ISLA pressure "
                                          f"error; left side, orthographic, {int(left.sum()):,} of {len(xyz):,} scored surface points")
    arrays = {"xyz": xyz.astype(np.float32), "cp_true": truth.astype(np.float16)}
    for (k, _), p in zip(MODELS, preds):
        arrays[f"cp_{k}"] = p.astype(np.float16)
    np.savez_compressed(f"{out_arr}/shift_suv_{sid.split('_')[0]}.npz", **arrays)
    return rec


if __name__ == "__main__":
    dump, out_png, out_arr = sys.argv[1:4]
    os.makedirs(out_png, exist_ok=True)
    os.makedirs(out_arr, exist_ok=True)
    which = sys.argv[4].split(",") if len(sys.argv) > 4 else ["deepjebpp", "shift_suv"]
    for w in which:
        extra = [sys.argv[5]] if (w == "deepjebpp" and len(sys.argv) > 5) else []
        rec = {"deepjebpp": deepjebpp, "shift_suv": shift_suv}[w](dump, out_png, out_arr, *extra)
        json.dump(rec, open(f"{out_arr}/render_cloud_{w}.json", "w"), indent=1)
        print(w, json.dumps(rec)[:600])
