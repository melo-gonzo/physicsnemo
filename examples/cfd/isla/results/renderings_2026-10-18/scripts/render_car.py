"""Car-surface renderings (FIG lane, 2026-10-18): DrivAerML Surface truth / ISLA / GeoTransolver / Transolver for one
validation car, and the posed protocol mapped back to the body frame.

Geometry: the case's own STL (drivaer_<run>_single_solid.stl.pmsh, 753k triangles for run 171), rendered off-screen with
VTK (pyvista, CPU llvmpipe). Fields: each model's predictions live on the cells it was scored on (the scorer's draw of
200,000 cells; Transolver's 100,000); each STL triangle shows the value of the nearest scored cell centroid of that model
(the dataset pipeline centres the car, so centroids are shifted back by global.center). Truth panel: the truth on ISLA's
200,000 scored cells. Error panels: prediction minus truth evaluated at each model's own scored cells, then mapped the
same way. c_p = (p - p_inf) / q_inf and c_f = |tau_w| / q_inf with q_inf = rho_inf |U_inf|^2 / 2 from the case's
global data. Camera "left side" is orthographic, looking along +y.

Posed protocol: the scorer's drivaer_ml_surface_posed.yaml applies FixedRandomPose (keyed rotation uniform on SO(3) and a
translation of one RMS radius) before CenterMesh; dump_wrap.py records R and t. The posed cell centroids x' (centred)
map back to the body frame as x = R^T x' (the centring removes R c + t exactly: CenterMesh averages the same posed
points); the scalar c_p needs no rotation. The posed-minus-native panels compare each model's posed prediction (mapped
back) with its own native prediction on the native scored cells (nearest posed cell; the two draws are the same cells
when the reader draws the same indices, which the record checks).
usage: python render_car.py <dump_dir> <stl.pmsh> <out_png_dir> <out_arrays_dir> [pose]
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_lib as R  # noqa: E402
from render_lib import palette  # noqa: E402

MODELS = [("isla", "ISLA"), ("gt", "GeoTransolver 256"), ("t", "Transolver 512 x 16")]


def load(dump, tag, proto="native"):
    f = glob.glob(f"{dump}/{tag}/{proto}/*.npz")
    assert len(f) == 1, (tag, proto, f)
    m = None
    for line in open(glob.glob(f"{dump}/{tag}/{proto}/*/metrics.jsonl")[0]):
        j = json.loads(line)
        if j.get("phase") == "infer_step":
            m = j
    return dict(np.load(f[0])), m, os.path.basename(f[0])[:-4]


def load_stl(path):
    from physicsnemo.mesh import Mesh

    m = Mesh.load(path)
    return m.points.numpy().astype(np.float64), m.cells.numpy().astype(np.int64)


def coeffs(d, w):
    U = np.asarray(d["global.U_inf"], np.float64)
    q = 0.5 * float(d["global.rho_inf"]) * float(U @ U)
    p_inf = float(d.get("global.p_inf", 0.0))
    return {"cp": (d[f"{w}_pressure"] - p_inf) / q, "cf": np.linalg.norm(d[f"{w}_wss"], axis=1) / q}, q


VIEWS = {
    "left_side": lambda c, s: dict(position=(c[0], c[1] - 20, c[2]), focal=tuple(c), up=(0, 0, 1), parallel=True, scale=s),
    "top": lambda c, s: dict(position=(c[0], c[1], c[2] + 20), focal=tuple(c), up=(0, 1, 0), parallel=True, scale=s),
    "front_three_quarter": lambda c, s: dict(position=(c[0] - 6.5, c[1] - 6.0, c[2] + 3.2), focal=tuple(c), up=(0, 0, 1),
                                             parallel=False, angle=26.0),
}


def grid_figure(rows, row_labels, col_titles, cbars, out_png, suptitle, panel_w=4.6):
    """rows: list of lists of images (None = empty); cbars: per row (cmap, clim, label)."""
    h, w = rows[0][0].shape[:2] if rows[0][0] is not None else rows[0][1].shape[:2]
    ncol = max(len(r) for r in rows)
    fig = plt.figure(figsize=(panel_w * ncol + 1.2, panel_w * h / w * len(rows) + 0.9 + 0.35 * len(rows)))
    gs = fig.add_gridspec(len(rows), ncol + 1, width_ratios=[1] * ncol + [0.045], wspace=0.03, hspace=0.25)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            ax = fig.add_subplot(gs[i, j])
            ax.axis("off")
            if im is not None:
                ax.imshow(im, interpolation="lanczos")
                ax.set_title(col_titles[i][j], fontsize=10)
        cax = fig.add_subplot(gs[i, ncol])
        cmap, clim, label = cbars[i]
        sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(*clim), cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(label, fontsize=9.5)
        cb.ax.tick_params(labelsize=8.5)
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 + 0.15 * pos.height, pos.width, 0.7 * pos.height])
    fig.suptitle(suptitle, fontsize=10.5, y=0.995)
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    dump, stl, out_png, out_arr = sys.argv[1:5]
    do_pose = len(sys.argv) > 5 and sys.argv[5] == "pose"
    os.makedirs(out_png, exist_ok=True)
    os.makedirs(out_arr, exist_ok=True)
    V, F = load_stl(stl)
    poly = R.polydata(V, F)
    cent = V[F].mean(1)
    D = {k: load(dump, f"dr_{k}") for k, _ in MODELS}
    ref = D["isla"][0]
    sid = D["isla"][2]
    rec = {"sample_id": sid, "stl": stl, "n_stl_triangles": int(len(F)), "per_sample_rel_l2": {}, "nn_distance_m": {}}
    pts = {k: D[k][0]["points"].astype(np.float64) + D[k][0]["global.center"].astype(np.float64) for k, _ in MODELS}
    for k, _ in MODELS:
        _, dist = R.nn_map(pts[k], np.zeros(len(pts[k])), cent)
        rec["nn_distance_m"][k] = {"median": float(np.median(dist)), "p99": float(np.percentile(dist, 99))}
    true_c, q = coeffs(ref, "true")
    rec["q_inf"] = q
    c = 0.5 * (V.min(0) + V.max(0))
    side = VIEWS["left_side"](c, 0.80)
    size = (2400, 800)
    arrays = {}
    for field, lab in (("cp", "pressure coefficient $c_p$ [–]"), ("cf", "skin-friction coefficient $|\\tau_w|/q_\\infty$ [–]")):
        tv, _ = R.nn_map(pts["isla"], true_c[field], cent)
        preds, errs, titles, etitles = [], [], [], []
        mkey = "pressure_l2" if field == "cp" else "wss_l2"
        for k, name in MODELS:
            pc, _ = coeffs(D[k][0], "pred")
            tc, _ = coeffs(D[k][0], "true")
            preds.append(R.nn_map(pts[k], pc[field], cent)[0])
            errs.append(R.nn_map(pts[k], pc[field] - tc[field], cent)[0])
            l2 = D[k][1]["metrics"][mkey]
            rec["per_sample_rel_l2"].setdefault(mkey, {})[name] = l2
            titles.append(f"{name} (rel. L2 {l2:.4f})")
            etitles.append(f"{name}: prediction $-$ truth")
            arrays[f"{field}_{k}"] = pc[field].astype(np.float16)
            arrays[f"{field}_true_at_{k}"] = tc[field].astype(np.float16)
        clim = tuple(np.percentile(true_c[field], [0.5, 99.5]))
        if field == "cf":
            clim = (0.0, clim[1])
        ev = float(np.percentile(np.abs(np.concatenate(errs)), 99))
        ims = [R.render(poly, side, scalars=tv, cmap=palette.SEQUENTIAL, clim=clim, size=size)]
        ims += [R.render(poly, side, scalars=p, cmap=palette.SEQUENTIAL, clim=clim, size=size) for p in preds]
        eims = [R.render(poly, side, scalars=e, cmap=palette.DIVERGING, clim=(-ev, ev), size=size) for e in errs]
        allc = R.common_crop(ims + eims)
        ims, eims = allc[:4], allc[4:]
        grid_figure([ims[:2], ims[2:], [None, eims[0]], eims[1:]],
                    None, [["truth (solver)", titles[0]], titles[1:], ["", etitles[0]], etitles[1:]],
                    [(palette.SEQUENTIAL, clim, lab)] * 2 + [(palette.DIVERGING, (-ev, ev), "prediction $-$ truth [–]")] * 2,
                    f"{out_png}/drivaerml_surface_{field}.png",
                    f"DrivAerML Surface, validation car {sid.split('_')[0]} (run 171): the case of median ISLA pressure error "
                    f"among the 48 validation cars; left side, orthographic", panel_w=5.0)
    for k, _ in MODELS:
        arrays[f"xyz_{k}"] = pts[k].astype(np.float32)
    if do_pose:
        rec["pose"] = pose(dump, D, pts, poly, cent, side, size, out_png, arrays, sid)
    np.savez_compressed(f"{out_arr}/drivaerml_surface_{sid.split('_')[0]}.npz", **arrays)
    json.dump(rec, open(f"{out_arr}/render_car.json", "w"), indent=1)
    print(json.dumps(rec, indent=1)[:3000])


def pose(dump, D, pts, poly, cent, side, size, out_png, arrays, sid):
    out = {}
    diffs, ims_posed, titles = [], [], []
    for k, name in MODELS:
        P, mP, _ = load(dump, f"dr_{k}", "posed")
        Rm = P["pose_R"].astype(np.float64)
        x_body = P["points"].astype(np.float64) @ Rm + D[k][0]["global.center"].astype(np.float64)  # R^T x' (row vectors)
        same = len(P["points"]) == len(D[k][0]["points"])
        d_nn = R.nn_map(x_body, np.zeros(len(x_body)), pts[k])[1]
        cp_posed, _ = coeffs(P, "pred")
        cp_nat, _ = coeffs(D[k][0], "pred")
        cp_posed_on_native = R.nn_map(x_body, cp_posed["cp"], pts[k])[0]
        diff = cp_posed_on_native - cp_nat["cp"]
        out[name] = {"rel_l2_native": D[k][1]["metrics"]["pressure_l2"], "rel_l2_posed": mP["metrics"]["pressure_l2"],
                     "posed_minus_native_rms_cp": float(np.sqrt(np.mean(diff ** 2))),
                     "posed_minus_native_rel": float(np.linalg.norm(diff) / np.linalg.norm(cp_nat["cp"])),
                     "map_back_nn_distance_m_median": float(np.median(d_nn)), "map_back_nn_distance_m_max": float(d_nn.max()),
                     "same_count": bool(same), "pose_R": Rm.tolist(), "pose_t": np.asarray(P.get("pose_t", [])).tolist()}
        arrays[f"cp_posed_mapped_back_{k}"] = cp_posed_on_native.astype(np.float16)
        diffs.append(R.nn_map(pts[k], diff, cent)[0])
        ims_posed.append(R.nn_map(x_body, cp_posed["cp"], cent)[0])
        titles.append(name)
    true_c, _ = coeffs(D["isla"][0], "true")
    clim = tuple(np.percentile(true_c["cp"], [0.5, 99.5]))
    ev = float(np.percentile(np.abs(np.concatenate(diffs[1:])), 99))
    top = [R.render(poly, side, scalars=v, cmap=palette.SEQUENTIAL, clim=clim, size=size) for v in ims_posed]
    bot = [R.render(poly, side, scalars=v, cmap=palette.DIVERGING, clim=(-ev, ev), size=size) for v in diffs]
    allc = R.common_crop(top + bot)
    top, bot = allc[:3], allc[3:]
    t1 = [f"{n}: posed, mapped back (rel. L2 {out[n]['rel_l2_posed']:.4f})" for n in titles]
    t2 = [f"{n}: posed $-$ native (rms {out[n]['posed_minus_native_rms_cp']:.1e})" for n in titles]
    grid_figure([top, bot], None, [t1, t2],
                [(palette.SEQUENTIAL, clim, "pressure coefficient $c_p$ [–]"), (palette.DIVERGING, (-ev, ev), "posed $-$ native $c_p$ [–]")],
                f"{out_png}/drivaerml_surface_pose.png",
                f"DrivAerML Surface, car {sid.split('_')[0]}: each model's prediction on the rotated and translated car "
                f"(scorer's posed protocol), mapped back to the body frame; bottom row shares one scale", panel_w=4.6)
    return out


if __name__ == "__main__":
    main()
