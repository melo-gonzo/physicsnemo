"""2-D renderings (FIG lane, 2026-10-18): AirfRANS, Tensile2d and the lug's top face truth / ISLA / GeoTransolver / Transolver on the scored
points of one validation sample (dump.sbatch, LHR jobs 271987 and 275252), drawn with matplotlib on a Delaunay
triangulation of those points (triangles longer than a cut-off are masked, so holes such as the airfoil stay empty).
Top row: the field, one shared colour scale (viridis, 0.5-99.5th percentile of the truth). Bottom row: prediction minus
truth, one shared symmetric scale (the book's DIVERGING map, 99th percentile of the three errors). Panel titles carry the
scorer's per-sample relative L2 of that field (metrics.jsonl of the same run, normalized space, as in results/char).
Also writes the compact arrays the book can redraw from (float32 coordinates, float16 fields).
usage: python render_2d.py <dump_dir> <out_png_dir> <out_arrays_dir> [airfrans,tensile2d,lug,drivaerml_volume] [drivaerml_volume=<stl.pmsh>]
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_lib as R  # noqa: E402
from render_lib import palette  # noqa: E402

# Lug and Tensile2d dumps are in the pipeline's normalized units (their pipelines have no NonDimensionalizeByMetadata for
# infer.py to invert); the figures undo NormalizeMeshFields with the stats of ginot_lug_volume.yaml / plaid_tensile2d_2d.yaml.
TENSILE_SIG_STD = 37.837  # MPa, zero mean, shared by sig11, sig22, sig12
LUG_VM = (150.2, 92.4)  # MPa, mean and std of von_mises
WALL_SDF = 2e-4  # [chord] points this close to the airfoil count as wall points for the hole mask
MODELS = [("isla", "ISLA"), ("gt", "GeoTransolver"), ("t", "Transolver")]  # sizes: README run map


def load(dump, tag):
    f = glob.glob(f"{dump}/{tag}/native/*.npz")
    assert len(f) == 1, (tag, f)
    m = {}
    for line in open(glob.glob(f"{dump}/{tag}/native/*/metrics.jsonl")[0]):
        j = json.loads(line)
        if j.get("phase") == "infer_step":
            m = j
    return dict(np.load(f[0])), m, os.path.basename(f[0])[:-4]


def tri_of(xy, maxlen, hole_box=None, wall=None):
    """Delaunay triangulation with the long triangles masked (only inside hole_box = (x0, x1, y0, y1), if given)."""
    t = mtri.Triangulation(xy[:, 0], xy[:, 1])
    p = xy[t.triangles]
    e = np.max(np.linalg.norm(p - np.roll(p, 1, axis=1), axis=-1), axis=1)
    mask = e > maxlen
    if hole_box is not None:
        c = p.mean(1)
        mask &= (c[:, 0] > hole_box[0]) & (c[:, 0] < hole_box[1]) & (c[:, 1] > hole_box[2]) & (c[:, 1] < hole_box[3])
    if wall is not None:  # triangles whose three vertices are all wall points span the solid
        mask |= wall[t.triangles].all(1)
    t.set_mask(mask)
    return t


def panel_grid(xy, truth, preds, titles, label, unit, window, maxlen, out_png, suptitle, aspect_w=4.2, hole_box=None, wall=None,
               err_label="prediction $-$ truth", tri=None, overlay=None):
    T = tri if tri is not None else tri_of(xy, maxlen, hole_box, wall)
    lo, hi = np.percentile(truth, [0.5, 99.5])
    errs = [p - truth for p in preds]
    v = np.percentile(np.abs(np.concatenate(errs)), 99)
    fig, axes = plt.subplots(2, 4, figsize=(4 * aspect_w * 0.62 + 1.6, max(2 * aspect_w * 0.62 * (window[3] - window[2]) / (window[1] - window[0]) + 1.4, 4.4)),
                             constrained_layout=True)
    for ax in axes.ravel():
        ax.set_xlim(window[0], window[1]); ax.set_ylim(window[2], window[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    im = axes[0, 0].tripcolor(T, truth, shading="gouraud", cmap=palette.SEQUENTIAL, vmin=lo, vmax=hi, rasterized=True)
    axes[0, 0].set_title("truth (solver)\n", fontsize=10)
    for k, (p, ttl) in enumerate(zip(preds, titles)):
        axes[0, k + 1].tripcolor(T, p, shading="gouraud", cmap=palette.SEQUENTIAL, vmin=lo, vmax=hi, rasterized=True)
        axes[0, k + 1].set_title(ttl, fontsize=10)
        ime = axes[1, k + 1].tripcolor(T, errs[k], shading="gouraud", cmap=palette.DIVERGING, vmin=-v, vmax=v, rasterized=True)
        axes[1, k + 1].set_title(ttl.split("\n")[0] + ": error", fontsize=10)
    axes[1, 0].axis("off")
    if overlay is not None:
        for ax in list(axes[0]) + list(axes[1, 1:]):
            overlay(ax)
    cb = fig.colorbar(im, ax=axes[0, :], shrink=0.85, pad=0.01)
    cb.set_label(f"{label} {unit}")
    cb2 = fig.colorbar(ime, ax=axes[1, :], shrink=0.85, pad=0.01)
    cb2.set_label(f"{err_label} {unit}")
    fig.suptitle(suptitle, fontsize=10.5)
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def airfrans(dump, out_png, out_arr):
    D = {k: load(dump, f"af_{k}") for k, _ in MODELS}
    xy = D["isla"][0]["points"][:, :2].astype(np.float64)
    for k, _ in MODELS:
        assert np.array_equal(D[k][0]["points"][:, :2], D["isla"][0]["points"][:, :2]), k
    g = D["isla"][0]
    q = 0.5 * float(g["global.rho_inf"]) * float(g["global.U_inf_mag"]) ** 2
    U = float(g["global.U_inf_mag"])
    sid = D["isla"][2]
    window = (-0.25, 1.35, -0.40, 0.40)
    sel = (xy[:, 0] > window[0] - 0.3) & (xy[:, 0] < window[1] + 0.3) & (xy[:, 1] > window[2] - 0.3) & (xy[:, 1] < window[3] + 0.3)
    xs = xy[sel]
    rec = {"sample_id": sid, "source": R.provenance([(f"af_{k}", "native") for k, _ in MODELS]), "q_inf": q, "U_inf": U, "aoa_deg": float(g["global.aoa_deg"]), "Re": float(g["global.Re"]),
           "n_points_scored": int(len(xy)), "per_sample_rel_l2": {}}
    arrays = {"xy": xs.astype(np.float32)}
    for field, lab, unit, f in (("pressure", "$c_p$", "[–]", lambda d, w: d[f"{w}_pressure"] / q),
                                ("velocity", "$|u|/U_\\infty$", "[–]", lambda d, w: np.linalg.norm(d[f"{w}_velocity"][:, :2], axis=1) / U)):
        get = f
        truth = get(D["isla"][0], "true")[sel]
        preds = [get(D[k][0], "pred")[sel] for k, _ in MODELS]
        titles = [f"{name}\nrel. L2 {D[k][1]['metrics'][field + '_l2']:.4f}" for k, name in MODELS]
        arrays[f"{field}_true"] = truth.astype(np.float16)
        for (k, _), p in zip(MODELS, preds):
            arrays[f"{field}_{k}"] = p.astype(np.float16)
        rec["per_sample_rel_l2"][field] = {name: D[k][1]["metrics"][field + "_l2"] for k, name in MODELS}
        panel_grid(xs, truth, preds, titles, lab, unit, window, 0.05, f"{out_png}/airfrans_{field}.png",
                   f"AirfRANS validation sample {sid.split('_')[0]} (Re {rec['Re']:.2e}, angle of attack {rec['aoa_deg']:.2f}°): "
                   f"the case of median ISLA pressure error among the 200 test cases", aspect_w=4.2, hole_box=(-0.02, 1.02, -0.2, 0.2),
                   wall=np.abs(np.ravel(D["isla"][0]["interior.sdf"])[sel]) < WALL_SDF)
    np.savez_compressed(f"{out_arr}/airfrans_{sid.split('_')[0]}.npz", **arrays)
    return rec


def tensile(dump, out_png, out_arr):
    D = {k: load(dump, f"t2d_{k}") for k, _ in MODELS}
    xy = D["isla"][0]["points"][:, :2].astype(np.float64)
    for k, _ in MODELS:
        assert np.allclose(D[k][0]["points"][:, :2], D["isla"][0]["points"][:, :2]), k
    sid = D["isla"][2]

    def vm(d, w):  # plane-stress von Mises from the scored components (infer.py writes them normalized: x 37.837 MPa)
        s11, s22, s12 = (TENSILE_SIG_STD * d[f"{w}_{c}"].astype(np.float64) for c in ("sig11", "sig22", "sig12"))
        return np.sqrt(s11 ** 2 - s11 * s22 + s22 ** 2 + 3 * s12 ** 2)

    truth = vm(D["isla"][0], "true")
    preds = [vm(D[k][0], "pred") for k, _ in MODELS]
    titles = [f"{name}\nstress rel. L2 {D[k][1]['metrics']['stress_l2']:.4f}" for k, name in MODELS]
    lo, hi = xy.min(0), xy.max(0)
    pad = 0.03 * (hi - lo).max()
    window = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
    e = np.sort(np.linalg.norm(xy[mtri.Triangulation(xy[:, 0], xy[:, 1]).edges[:, 0]] - xy[mtri.Triangulation(xy[:, 0], xy[:, 1]).edges[:, 1]], axis=1))
    maxlen = 4 * e[len(e) // 2]
    panel_grid(xy, truth, preds, titles, "von Mises stress (from $\\sigma_{11}, \\sigma_{22}, \\sigma_{12}$)", "[MPa]", window,
               maxlen, f"{out_png}/tensile2d_stress.png",
               f"Tensile2d validation sample {sid.split('_')[0]}: the case of median ISLA stress error among the 100 validation specimens", aspect_w=4.2)
    arrays = {"xy": xy.astype(np.float32), "von_mises_true": truth.astype(np.float16)}
    for (k, _), p in zip(MODELS, preds):
        arrays[f"von_mises_{k}"] = p.astype(np.float16)
    np.savez_compressed(f"{out_arr}/tensile2d_{sid.split('_')[0]}.npz", **arrays)
    return {"sample_id": sid, "source": R.provenance([(f"t2d_{k}", "native") for k, _ in MODELS]), "n_points_scored": int(len(xy)),
            "per_sample_rel_l2": {f: {name: D[k][1]["metrics"][f] for k, name in MODELS} for f in ("stress_l2", "displacement_l2", "q_l2")}}


def lug(dump, out_png, out_arr):
    """The lug's top face (z = thickness, the 1,060 nodes of the top layer of the 12-layer hex mesh), von Mises in MPa."""
    D = {k: load(dump, f"lug_{k}") for k, _ in MODELS}
    P = D["isla"][0]["points"].astype(np.float64)
    for k, _ in MODELS:
        assert np.allclose(D[k][0]["points"], P), k
    sid = D["isla"][2]
    g = D["isla"][0]
    top = np.abs(P[:, 2] - P[:, 2].max()) < 1e-5
    xy = P[top, :2]
    vm = lambda d, w: LUG_VM[0] + LUG_VM[1] * d[f"{w}_von_mises"].astype(np.float64)  # noqa: E731
    truth = vm(g, "true")[top]
    preds = [vm(D[k][0], "pred")[top] for k, _ in MODELS]
    titles = [f"{name}\nrel. L2 {D[k][1]['metrics']['von_mises_l2']:.4f}" for k, name in MODELS]
    lo, hi = xy.min(0), xy.max(0)
    pad = 0.03 * (hi - lo).max()
    window = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
    t = mtri.Triangulation(xy[:, 0], xy[:, 1])
    e = np.sort(np.linalg.norm(xy[t.edges[:, 0]] - xy[t.edges[:, 1]], axis=1))
    panel_grid(xy, truth, preds, titles, "von Mises stress", "[MPa]", window, 3 * e[len(e) // 2], f"{out_png}/lug_von_mises.png",
               f"Lug test case {sid.split('_')[0]} (hole radius {1e3 * float(g['global.hole_radius']):.1f} mm, length "
               f"{1e3 * float(g['global.length']):.0f} mm, thickness {1e3 * float(g['global.thickness']):.1f} mm, pressure "
               f"{float(g['global.pressure_mpa']):.1f} MPa):\nthe case of median ISLA error among the 600 test lugs; top face",
               aspect_w=4.2, err_label="error")
    arrays = {"xy_top": xy.astype(np.float32), "von_mises_true": truth.astype(np.float16)}
    for (k, _), p in zip(MODELS, preds):
        arrays[f"von_mises_{k}"] = p.astype(np.float16)
    np.savez_compressed(f"{out_arr}/lug_{sid.split('_')[0]}.npz", **arrays)
    return {"sample_id": sid, "source": R.provenance([(f"lug_{k}", "native") for k, _ in MODELS]), "n_points_scored": int(len(P)),
            "n_points_top_face": int(top.sum()), "units": "MPa (von_mises = 150.2 + 92.4 x the dump's normalized value)",
            "params": {"hole_radius_m": float(g["global.hole_radius"]), "length_m": float(g["global.length"]),
                       "thickness_m": float(g["global.thickness"]), "pressure_mpa": float(g["global.pressure_mpa"])},
            "per_sample_rel_l2": {"von_mises_l2": {name: D[k][1]["metrics"]["von_mises_l2"] for k, name in MODELS}}}


def volume(dump, out_png, out_arr, stl, half_width=0.03):
    """DrivAerML Volume car 00033 (run 129): the scored volume points in the slab |y*| < half_width (x L_ref = 15 cm) about
    the symmetry plane, drawn in (x, z) on a Delaunay of the slab's points, with the body's cut at y* = 0 masked and drawn
    grey. infer.py writes the points centred and divided by L_ref = 5 m without storing the centre; the offset is solved
    by point-to-plane least squares so that the distance of the near-wall points (sdf < 2e-4) to the case's STL
    (drivaer_129_single_solid.stl.pmsh, copied from LHR) equals their sdf. A view of a thin slab, not an interpolation
    onto the plane."""
    import pyvista as pv
    from matplotlib.path import Path
    from physicsnemo.mesh import Mesh

    D = {k: load(dump, f"vol_{k}") for k, _ in MODELS}
    P = D["isla"][0]["points"].astype(np.float64)
    for k, _ in MODELS:
        assert np.allclose(D[k][0]["points"], P), k
    g = D["isla"][0]
    sid = D["isla"][2]
    L = float(g["global.L_ref"])
    U = float(np.linalg.norm(g["global.U_inf"]))
    q = 0.5 * float(g["global.rho_inf"]) * U ** 2
    sdf = np.ravel(g["interior.sdf"]).astype(np.float64)
    m = Mesh.load(stl)
    V = m.points.numpy().astype(np.float64)
    F = m.cells.numpy().astype(np.int64)
    body = pv.PolyData(V, np.hstack([np.full((len(F), 1), 3, np.int64), F]).ravel())
    near = (sdf < 2e-4) & (P[:, 2] > P[:, 2].min() + 0.005)  # near-wall points on the car, not the ground
    x = L * P[near]
    off = V[F].mean(1).mean(0) - x.mean(0)
    for _ in range(20):
        _, cp = body.find_closest_cell(x + off, return_closest_point=True)
        r = cp - (x + off)
        dist = np.linalg.norm(r, axis=1)
        off = off + np.linalg.lstsq(r / np.maximum(dist[:, None], 1e-12), dist - L * sdf[near], rcond=None)[0]
    fit = np.abs(dist - L * sdf[near])
    cut = body.slice(normal="y", origin=(0.0, off[1], 0.0)).strip(join=True)
    loops = []
    for c in range(cut.n_cells):
        pts = np.asarray(cut.get_cell(c).points)
        loops.append(np.c_[(pts[:, 0] - off[0]) / L, (pts[:, 2] - off[2]) / L])
    window = (-0.75, 1.15, P[:, 2].min(), 0.30)
    sel = (np.abs(P[:, 1]) < half_width) & (P[:, 0] > window[0] - 0.1) & (P[:, 0] < window[1] + 0.1) & (P[:, 2] < window[3] + 0.1)
    xz = P[sel][:, [0, 2]]
    t = mtri.Triangulation(xz[:, 0], xz[:, 1])
    c = xz[t.triangles].mean(1)
    inside = np.zeros(len(c), bool)
    for lp in loops:
        inside |= Path(lp).contains_points(c)
    t.set_mask(inside)

    def overlay(ax):
        for lp in loops:
            ax.fill(lp[:, 0], lp[:, 1], color=palette.BASELINE, lw=0.4, ec=palette.MUTED, zorder=3)

    rec = {"sample_id": sid, "source": R.provenance([(f"vol_{k}", "native") for k, _ in MODELS]), "n_points_scored": int(len(P)),
           "slab_half_width_m": half_width * L, "n_points_in_slab_window": int(sel.sum()), "q_inf": q, "U_inf": U, "stl": stl,
           "solved_centre_m": off.tolist(), "stl_fit_abs_sdf_residual_m": {"median": float(np.median(fit)), "p99": float(np.percentile(fit, 99))},
           "per_sample_rel_l2": {f: {name: D[k][1]["metrics"][f] for k, name in MODELS} for f in ("pressure_l2", "velocity_l2", "nut_l2")}}
    arrays = {"xz": xz.astype(np.float32)}
    for field, lab, f, mkey in (("pressure", "$c_p$", lambda d, w: d[f"{w}_pressure"].astype(np.float64) / q, "pressure_l2"),
                                ("velocity", "$|u|/U_\\infty$", lambda d, w: np.linalg.norm(d[f"{w}_velocity"].astype(np.float64), axis=1) / U,
                                 "velocity_l2")):
        truth = f(g, "true")[sel]
        preds = [f(D[k][0], "pred")[sel] for k, _ in MODELS]
        titles = [f"{name}\nrel. L2 {D[k][1]['metrics'][mkey]:.4f}" for k, name in MODELS]
        panel_grid(xz, truth, preds, titles, lab, "[–]", window, None, f"{out_png}/drivaerml_volume_{field}.png",
                   f"DrivAerML Volume validation car {sid.split('_')[0]} (run 129), the case of median ISLA pressure error among the 48 "
                   f"validation cars:\nthe {int(sel.sum()):,} scored points (of {len(P):,}) within {1e2 * half_width * L:.0f} cm of the "
                   f"symmetry plane, side view; coordinates / L_ref", aspect_w=4.2, tri=t, overlay=overlay)
        arrays[f"{field}_true"] = truth.astype(np.float16)
        for (k, _), p in zip(MODELS, preds):
            arrays[f"{field}_{k}"] = p.astype(np.float16)
    for n, lp in enumerate(loops):
        arrays[f"body_cut_{n}"] = lp.astype(np.float32)
    np.savez_compressed(f"{out_arr}/drivaerml_volume_{sid.split('_')[0]}.npz", **arrays)
    return rec

if __name__ == "__main__":
    dump, out_png, out_arr = sys.argv[1:4]
    os.makedirs(out_png, exist_ok=True); os.makedirs(out_arr, exist_ok=True)
    which = sys.argv[4].split(",") if len(sys.argv) > 4 else ["airfrans", "tensile2d", "lug"]  # drivaerml_volume needs its STL
    for w in which:
        extra = [a.split("=", 1)[1] for a in sys.argv[5:] if a.startswith(w + "=")]  # drivaerml_volume=<stl.pmsh>
        rec = {"airfrans": airfrans, "tensile2d": tensile, "lug": lug, "drivaerml_volume": volume}[w](dump, out_png, out_arr, *extra)
        print(w, json.dumps(rec)[:400])
        json.dump({w: rec}, open(f"{out_arr}/render_2d_{w}.json", "w"), indent=1)
