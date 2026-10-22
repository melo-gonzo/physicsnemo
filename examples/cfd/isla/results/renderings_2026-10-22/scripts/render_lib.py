# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rendering helpers for the RENDER-090 lane (2026-10-22; renderings_2026-10-18/scripts/render_lib.py with the provenance of
this lane's dumps and the scorer-record check, scorer_checked): off-screen VTK (pyvista) surface renderings on the CPU, composed into
labeled multi-panel figures with matplotlib so every panel shares one colour scale and one colorbar with units.

Conventions (book/STYLE.md and the FIG brief): white background; fields in the book's SEQUENTIAL map (viridis), errors
in its DIVERGING map (blue-paper-red), symmetric about zero; one camera per dataset view, named, orthographic for the
engineering views; dominant slices in palette.slice_colors.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

BOOK = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "book")
)
sys.path.insert(0, BOOK)
import palette  # noqa: E402,F401

pv.OFF_SCREEN = True


def nn_map(src_pts, src_vals, dst_pts):
    """Value of the nearest source point at every destination point (and the distance to it)."""
    d, i = cKDTree(np.asarray(src_pts, np.float64)).query(
        np.asarray(dst_pts, np.float64), workers=4
    )
    return np.asarray(src_vals)[i], d


def polydata(points, tris):
    """A pyvista PolyData of triangles."""
    tris = np.asarray(tris, np.int64)
    faces = np.hstack([np.full((len(tris), 1), 3, np.int64), tris]).ravel()
    return pv.PolyData(np.asarray(points, np.float64), faces)


def render(
    poly,
    view,
    scalars=None,
    cmap=None,
    clim=None,
    rgb=None,
    size=(2000, 1100),
    points=None,
    point_rgb=None,
    point_size=2.0,
    extra=None,
):
    """One off-screen render; returns an (H, W, 3) uint8 image. view: dict(position, focal, up, parallel, scale)."""
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    kw = dict(
        show_scalar_bar=False,
        ambient=0.3,
        diffuse=0.75,
        specular=0.08,
        specular_power=20,
        smooth_shading=False,
    )
    if poly is not None:
        # Cell-data colouring renders blank (white) with lighting on this machine's software OpenGL (VTK 9.7.1,
        # llvmpipe); point data renders correctly, so per-cell values become per-vertex averages of the adjacent cells.
        if rgb is not None:
            poly = poly.copy()
            poly.cell_data["rgb"] = np.asarray(rgb, np.float64)
            poly = poly.cell_data_to_point_data()
            poly.point_data["rgb"] = np.clip(
                np.rint(poly.point_data["rgb"]), 0, 255
            ).astype(np.uint8)
            p.add_mesh(poly, scalars="rgb", rgb=True, preference="point", **kw)
        elif scalars is not None:
            poly = poly.copy()
            s = np.asarray(scalars, np.float64)
            if len(s) == poly.n_cells and len(s) != poly.n_points:
                poly.cell_data["f"] = s
                poly = poly.cell_data_to_point_data()
            else:
                poly.point_data["f"] = s
            p.add_mesh(
                poly, scalars="f", cmap=cmap, clim=clim, preference="point", **kw
            )
        else:
            p.add_mesh(poly, color="#d9d8d2", **kw)
    if points is not None:
        cloud = pv.PolyData(np.asarray(points, np.float64))
        if point_rgb is not None:
            cloud.point_data["rgb"] = np.asarray(point_rgb, np.uint8)
            p.add_mesh(
                cloud,
                scalars="rgb",
                rgb=True,
                point_size=point_size,
                render_points_as_spheres=True,
                show_scalar_bar=False,
                ambient=0.5,
                diffuse=0.6,
            )
        else:
            p.add_mesh(
                cloud,
                color="#3a3a3a",
                point_size=point_size,
                render_points_as_spheres=True,
                ambient=0.5,
            )
    if extra is not None:
        extra(p)
    p.camera_position = [
        tuple(view["position"]),
        tuple(view["focal"]),
        tuple(view["up"]),
    ]
    if view.get("parallel", False):
        p.enable_parallel_projection()
        p.camera.parallel_scale = view["scale"]
    else:
        p.camera.view_angle = view.get("angle", 30.0)
    p.reset_camera_clipping_range()
    p.enable_anti_aliasing("ssaa")
    img = p.screenshot(return_img=True)
    p.close()
    return np.asarray(img)[..., :3]


def render_cloud(points, values, view, cmap, clim, size=(1600, 1000), point_size=4.0):
    """Off-screen render of a coloured point cloud (spheres, point data, so it renders on llvmpipe); (H, W, 3) uint8."""
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    cloud = pv.PolyData(np.asarray(points, np.float64))
    cloud.point_data["f"] = np.asarray(values, np.float64)
    p.add_mesh(
        cloud,
        scalars="f",
        cmap=cmap,
        clim=clim,
        point_size=point_size,
        render_points_as_spheres=True,
        show_scalar_bar=False,
        ambient=0.35,
        diffuse=0.7,
        specular=0.05,
    )
    p.camera_position = [
        tuple(view["position"]),
        tuple(view["focal"]),
        tuple(view["up"]),
    ]
    if view.get("parallel", False):
        p.enable_parallel_projection()
        p.camera.parallel_scale = view["scale"]
    else:
        p.camera.view_angle = view.get("angle", 30.0)
    p.reset_camera_clipping_range()
    img = p.screenshot(return_img=True)
    p.close()
    return np.asarray(img)[..., :3]


def silhouette(points, view, size, radius):
    """Pixel mask of a point cloud's outline in an orthographic view (rasterized points, closed by `radius` pixels)."""
    from scipy.ndimage import distance_transform_edt as edt

    W, H = size
    f = np.asarray(view["focal"], np.float64)
    d = f - np.asarray(view["position"], np.float64)
    d /= np.linalg.norm(d)
    u = np.asarray(view["up"], np.float64)
    r = np.cross(d, u)
    p = np.asarray(points, np.float64) - f
    k = (H / 2) / view["scale"]
    px = np.rint(W / 2 + k * (p @ r)).astype(int)
    py = np.rint(H / 2 - k * (p @ u)).astype(int)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    m = np.zeros((H, W), bool)
    m[py[ok], px[ok]] = True
    grown = edt(~m) <= radius
    return edt(grown) > radius


def fill_holes(img, mask, thresh=250):
    """Give every background pixel inside `mask` the colour of the nearest drawn pixel (gaps between sparse points)."""
    from scipy.ndimage import distance_transform_edt as edt

    bg = (img >= thresh).all(-1)
    _, (iy, ix) = edt(bg, return_indices=True)
    out = img.copy()
    sel = bg & mask
    out[sel] = img[iy[sel], ix[sel]]
    return out


def common_crop(images, pad=12, thresh=250):
    """Crop a list of same-size images to the union of their non-white boxes (so panels keep one scale)."""
    m = np.zeros(images[0].shape[:2], bool)
    for im in images:
        m |= (im < thresh).any(-1)
    ys, xs = np.where(m)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, m.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, m.shape[1])
    return [im[y0:y1, x0:x1] for im in images]


def four_up(top, bot, ttop, tbot, cb_top, cb_bot, out_png, suptitle, panel_w=4.2):
    """2 x 4 panels: truth and the three predictions above, the three errors below; one colorbar per row."""
    h, w = top[0].shape[:2]
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4 * panel_w + 1.0, 2 * panel_w * h / w + 1.3))
    gs = fig.add_gridspec(
        2, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.03, hspace=0.18
    )
    for i, (ims, tts) in enumerate(((top, ttop), ([None] + bot, [""] + tbot))):
        for j, (im, tt) in enumerate(zip(ims, tts)):
            ax = fig.add_subplot(gs[i, j])
            ax.axis("off")
            if im is not None:
                ax.imshow(im, interpolation="lanczos")
                ax.set_title(tt, fontsize=10)
        cmap, lim, lab = (cb_top, cb_bot)[i]
        cax = fig.add_subplot(gs[i, 4])
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(*lim), cmap=cmap
            ),
            cax=cax,
        )
        cb.set_label(lab, fontsize=9.5)
        cb.ax.tick_params(labelsize=8.5)
        pos = cax.get_position()
        cax.set_position(
            [pos.x0, pos.y0 + 0.1 * pos.height, pos.width, 0.8 * pos.height]
        )
    fig.suptitle(suptitle, fontsize=10.5, y=0.995)
    fig.savefig(
        out_png, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05
    )
    plt.close(fig)


def sym_clim(*arrays, q=99.0):
    """A symmetric colour range at the q-th percentile of the absolute values."""
    v = np.percentile(np.abs(np.concatenate([np.ravel(a) for a in arrays])), q)
    return (-v, v)


def range_clim(*arrays, q=(0.5, 99.5)):
    """The 0.5-99.5th percentile range of the given arrays."""
    c = np.concatenate([np.ravel(a) for a in arrays])
    return tuple(np.percentile(c, q))


def rel_l2(pred, true):
    """Relative L2 error of pred against true over all entries."""
    pred = np.asarray(pred, np.float64).reshape(len(pred), -1)
    true = np.asarray(true, np.float64).reshape(len(true), -1)
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


# T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
# LHR dump job of each row of fig_rows.tsv (the "=== row N END ... 1 npz" lines of sbatch_logs/isla.render090.dump-<job>.log);
# a REUSE:<row> row is the 2026-10-18 dump of that row (same run, case and command; LHR job 275252), its checkpoint gone.
# (Jobs 336002-336004 ran only their first row: sbatch --export split ONLY at its commas; the rest ran in 336068-336070.)
DUMP_JOB = {
    0: 336002,
    4: 336003,
    23: 336004,
    1: 336068,
    2: 336068,
    3: 336068,
    5: 336068,
    18: 336069,
    19: 336069,
    21: 336069,
    22: 336069,
    7: 336070,
    8: 336070,
    9: 336070,
    13: 336070,
    15: 336070,
    20: 336070,
}
LHR_DUMP = "$T/book_figs_2026-10-22/dump"
OLD_DUMP = "$T/book_figs_2026-10-18/dump"
LOCAL_DUMP = "scratch/isla-render090-2026-10-22/dump"
# The standing's scorer records (metrics.jsonl, phase infer_step): LHR $T/<evals>/<run>/<protocol>/<run>/metrics.jsonl,
# local copies under SCORES; the CHAR record's (fig_rows scorer "char") in the repository.
SCORES = os.environ.get(
    "RENDER_SCORES",
    "/home/psharpe/gh/physicsnemo/scratch/isla-render090-2026-10-22/scores",
)
CHAR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "campaigns_2026-10-02",
        "char",
        "protocols",
    )
)
CHECKS = []


def _rows():
    return [
        ln.rstrip("\n").split("\t")
        for ln in open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_rows.tsv")
        )
    ][1:]


def _row(tag, proto):
    (r,) = [r for r in _rows() if r[1] == tag and r[4] == proto]
    return r


def scorer_record(tag, proto):
    """Local path and cited path of the scorer record of a fig_rows.tsv row."""
    r = _row(tag, proto)
    run, scorer = r[2], r[13]
    if scorer == "char":
        return (
            f"{CHAR}/{run}/{proto}/metrics.jsonl",
            f"results/campaigns_2026-10-02/char/protocols/{run}/{proto}/metrics.jsonl",
        )
    return (
        f"{SCORES}/{scorer}/{run}/{proto}/{run}/metrics.jsonl",
        f"$T/{scorer}/{run}/{proto}/{run}/metrics.jsonl",
    )


def scorer_checked(tag, proto, sid, dump_row):
    """The standing's scorer row of case ``sid`` (the numbers the captions print), checked against the dump's own row: every
    metric of the dump's row, its value in the scorer's row and their relative difference go to CHECKS."""
    path, cited = scorer_record(tag, proto)
    rows = [json.loads(ln) for ln in open(path)]
    epoch = [r for r in rows if r.get("phase") == "config"][-1]["epoch"]
    hit = [
        r for r in rows if r.get("phase") == "infer_step" and r.get("sample_id") == sid
    ]
    assert hit, (tag, proto, sid)
    # A record scored in two passes holds the case twice; the CHAR reducer reads the last row per case.
    s = hit[-1]["metrics"]
    d = (dump_row or {}).get("metrics", {})
    rel = {k: abs(d[k] - s[k]) / max(abs(s[k]), 1e-30) for k in d if k in s}
    CHECKS.append(
        {
            "tag": tag,
            "protocol": proto,
            "run_id": _row(tag, proto)[2],
            "sample_id": sid,
            "scorer_record": cited,
            "scorer_epoch": epoch,
            "scorer_rows_for_case": len(hit),
            "scorer_rows_agree": all(h["metrics"] == s for h in hit),
            "scorer": s,
            "dump": d,
            "rel_diff": rel,
            "max_rel_diff": max(rel.values()) if rel else None,
        }
    )
    return {**hit[-1], "metrics": s}


def case_rank(tag, proto, sid, key):
    """(rank, n, median): the case's rank (1 = lowest error) by ``key`` among the scorer record's cases, and the median."""
    rows = [json.loads(ln) for ln in open(scorer_record(tag, proto)[0])]
    v = {
        r["sample_id"]: r["metrics"][key]
        for r in rows
        if r.get("phase") == "infer_step"
    }  # last row per case
    order = sorted(v, key=v.get)
    return order.index(sid) + 1, len(order), float(np.median(list(v.values())))


def rank_phrase(tag, proto, sid, key, what):
    """The title phrase giving the case's rank by ISLA's error (case_rank)."""
    k, n, _ = case_rank(tag, proto, sid, key)
    return f"ISLA's {what} error on this case ranks {k} of {n} (1 = lowest)"


def provenance(tag_protos):
    """Per (tag, protocol): the fig_rows.tsv row, run id, LHR dump job and dump path, and the scorer record, for the jsons."""
    out = {}
    for tag, proto in tag_protos:
        r = _row(tag, proto)
        reuse = r[3].startswith("REUSE:")
        out[f"{tag}/{proto}"] = {
            "row": int(r[0]),
            "run_id": r[2],
            "dataset": r[5],
            "model": r[6],
            "code": r[8],
            "recipe": r[9],
            "cells": int(r[10]),
            "replay": r[12] == "1",
            "lhr_dump_job": 275252 if reuse else DUMP_JOB[int(r[0])],
            "dump": f"{OLD_DUMP if reuse else LHR_DUMP}/{tag}/{proto}/ (copy: {LOCAL_DUMP}/{tag}/{proto}/)",
            "reused_2026_10_18_row": int(r[3].split(":")[1]) if reuse else None,
            "scorer_record": scorer_record(tag, proto)[1],
        }
    return out
