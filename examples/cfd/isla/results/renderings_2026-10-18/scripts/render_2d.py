"""2-D renderings (FIG lane, 2026-10-18): AirfRANS and Tensile2d truth / ISLA / GeoTransolver / Transolver on the scored
points of one validation sample (dump.sbatch, LHR jobs 271987 and 275252), drawn with matplotlib on a Delaunay
triangulation of those points (triangles longer than a cut-off are masked, so holes such as the airfoil stay empty).
Top row: the field, one shared colour scale (viridis, 0.5-99.5th percentile of the truth). Bottom row: prediction minus
truth, one shared symmetric scale (the book's DIVERGING map, 99th percentile of the three errors). Panel titles carry the
scorer's per-sample relative L2 of that field (metrics.jsonl of the same run, normalized space, as in results/char).
Also writes the compact arrays the book can redraw from (float32 coordinates, float16 fields).
usage: python render_2d.py <dump_dir> <out_png_dir> <out_arrays_dir>
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
from render_lib import palette  # noqa: E402

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


def panel_grid(xy, truth, preds, titles, label, unit, window, maxlen, out_png, suptitle, aspect_w=4.2, hole_box=None, wall=None):
    T = tri_of(xy, maxlen, hole_box, wall)
    lo, hi = np.percentile(truth, [0.5, 99.5])
    errs = [p - truth for p in preds]
    v = np.percentile(np.abs(np.concatenate(errs)), 99)
    fig, axes = plt.subplots(2, 4, figsize=(4 * aspect_w * 0.62 + 1.6, 2 * aspect_w * 0.62 * (window[3] - window[2]) / (window[1] - window[0]) + 1.4),
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
    cb = fig.colorbar(im, ax=axes[0, :], shrink=0.85, pad=0.01)
    cb.set_label(f"{label} {unit}")
    cb2 = fig.colorbar(ime, ax=axes[1, :], shrink=0.85, pad=0.01)
    cb2.set_label(f"prediction $-$ truth {unit}")
    fig.suptitle(suptitle, fontsize=10.5)
    fig.savefig(out_png, dpi=200, facecolor="white")
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
    rec = {"sample_id": sid, "q_inf": q, "U_inf": U, "aoa_deg": float(g["global.aoa_deg"]), "Re": float(g["global.Re"]),
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

    def vm(d, w):  # plane-stress von Mises from the scored components
        s11, s22, s12 = d[f"{w}_sig11"], d[f"{w}_sig22"], d[f"{w}_sig12"]
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
    return {"sample_id": sid, "n_points_scored": int(len(xy)),
            "per_sample_rel_l2": {f: {name: D[k][1]["metrics"][f] for k, name in MODELS} for f in ("stress_l2", "displacement_l2", "q_l2")}}


if __name__ == "__main__":
    dump, out_png, out_arr = sys.argv[1:4]
    os.makedirs(out_png, exist_ok=True); os.makedirs(out_arr, exist_ok=True)
    which = sys.argv[4].split(",") if len(sys.argv) > 4 else ["airfrans", "tensile2d"]
    res = {}
    for w in which:
        res[w] = {"airfrans": airfrans, "tensile2d": tensile}[w](dump, out_png, out_arr)
        print(w, json.dumps(res[w])[:400])
    json.dump(res, open(f"{out_arr}/render_2d_{'_'.join(which)}.json", "w"), indent=1)
