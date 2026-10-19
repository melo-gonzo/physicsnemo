"""Rendering helpers for the FIG lane (2026-10-18): off-screen VTK (pyvista) surface renderings on the CPU, composed into
labeled multi-panel figures with matplotlib so every panel shares one colour scale and one colorbar with units.

Conventions (book/STYLE.md and the FIG brief): white background; fields in the book's SEQUENTIAL map (viridis), errors
in its DIVERGING map (blue-paper-red), symmetric about zero; one camera per dataset view, named, orthographic for the
engineering views; dominant slices in palette.slice_colors.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

BOOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "book"))
sys.path.insert(0, BOOK)
import palette  # noqa: E402,F401

pv.OFF_SCREEN = True


def nn_map(src_pts, src_vals, dst_pts):
    """Value of the nearest source point at every destination point (and the distance to it)."""
    d, i = cKDTree(np.asarray(src_pts, np.float64)).query(np.asarray(dst_pts, np.float64), workers=4)
    return np.asarray(src_vals)[i], d


def polydata(points, tris):
    tris = np.asarray(tris, np.int64)
    faces = np.hstack([np.full((len(tris), 1), 3, np.int64), tris]).ravel()
    return pv.PolyData(np.asarray(points, np.float64), faces)


def render(poly, view, scalars=None, cmap=None, clim=None, rgb=None, size=(2000, 1100), points=None, point_rgb=None,
           point_size=2.0, extra=None):
    """One off-screen render; returns an (H, W, 3) uint8 image. view: dict(position, focal, up, parallel, scale)."""
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    kw = dict(show_scalar_bar=False, ambient=0.3, diffuse=0.75, specular=0.08, specular_power=20, smooth_shading=False)
    if poly is not None:
        # Cell-data colouring renders blank (white) with lighting on this machine's software OpenGL (VTK 9.7.1,
        # llvmpipe); point data renders correctly, so per-cell values become per-vertex averages of the adjacent cells.
        if rgb is not None:
            poly = poly.copy()
            poly.cell_data["rgb"] = np.asarray(rgb, np.float64)
            poly = poly.cell_data_to_point_data()
            poly.point_data["rgb"] = np.clip(np.rint(poly.point_data["rgb"]), 0, 255).astype(np.uint8)
            p.add_mesh(poly, scalars="rgb", rgb=True, preference="point", **kw)
        elif scalars is not None:
            poly = poly.copy()
            s = np.asarray(scalars, np.float64)
            if len(s) == poly.n_cells and len(s) != poly.n_points:
                poly.cell_data["f"] = s
                poly = poly.cell_data_to_point_data()
            else:
                poly.point_data["f"] = s
            p.add_mesh(poly, scalars="f", cmap=cmap, clim=clim, preference="point", **kw)
        else:
            p.add_mesh(poly, color="#d9d8d2", **kw)
    if points is not None:
        cloud = pv.PolyData(np.asarray(points, np.float64))
        if point_rgb is not None:
            cloud.point_data["rgb"] = np.asarray(point_rgb, np.uint8)
            p.add_mesh(cloud, scalars="rgb", rgb=True, point_size=point_size, render_points_as_spheres=True,
                       show_scalar_bar=False, ambient=0.5, diffuse=0.6)
        else:
            p.add_mesh(cloud, color="#3a3a3a", point_size=point_size, render_points_as_spheres=True, ambient=0.5)
    if extra is not None:
        extra(p)
    p.camera_position = [tuple(view["position"]), tuple(view["focal"]), tuple(view["up"])]
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
    p.add_mesh(cloud, scalars="f", cmap=cmap, clim=clim, point_size=point_size, render_points_as_spheres=True,
               show_scalar_bar=False, ambient=0.35, diffuse=0.7, specular=0.05)
    p.camera_position = [tuple(view["position"]), tuple(view["focal"]), tuple(view["up"])]
    if view.get("parallel", False):
        p.enable_parallel_projection()
        p.camera.parallel_scale = view["scale"]
    else:
        p.camera.view_angle = view.get("angle", 30.0)
    p.reset_camera_clipping_range()
    img = p.screenshot(return_img=True)
    p.close()
    return np.asarray(img)[..., :3]


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
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.03, hspace=0.18)
    for i, (ims, tts) in enumerate(((top, ttop), ([None] + bot, [""] + tbot))):
        for j, (im, tt) in enumerate(zip(ims, tts)):
            ax = fig.add_subplot(gs[i, j])
            ax.axis("off")
            if im is not None:
                ax.imshow(im, interpolation="lanczos")
                ax.set_title(tt, fontsize=10)
        cmap, lim, lab = (cb_top, cb_bot)[i]
        cax = fig.add_subplot(gs[i, 4])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(*lim), cmap=cmap), cax=cax)
        cb.set_label(lab, fontsize=9.5)
        cb.ax.tick_params(labelsize=8.5)
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 + 0.1 * pos.height, pos.width, 0.8 * pos.height])
    fig.suptitle(suptitle, fontsize=10.5, y=0.995)
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def sym_clim(*arrays, q=99.0):
    v = np.percentile(np.abs(np.concatenate([np.ravel(a) for a in arrays])), q)
    return (-v, v)


def range_clim(*arrays, q=(0.5, 99.5)):
    c = np.concatenate([np.ravel(a) for a in arrays])
    return tuple(np.percentile(c, q))


def rel_l2(pred, true):
    pred = np.asarray(pred, np.float64).reshape(len(pred), -1)
    true = np.asarray(true, np.float64).reshape(len(true), -1)
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


# LHR dump job of each row of fig_rows.tsv (from the "=== row N END ... 1 npz" lines of sbatch_logs/isla.fig.dump-<job>.log):
# rows 6-9, 12, 15, 18 wrote their npz in job 271987, every other row in job 275252.
DUMP_JOB = {r: (271987 if r in (6, 7, 8, 9, 12, 15, 18) else 275252) for r in range(24)}
LHR_DUMP = "$T/book_figs_2026-10-18/dump"  # T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0


def provenance(tag_protos):
    """Per (tag, protocol): the fig_rows.tsv row, run id, LHR dump job and dump path, for the rendering jsons."""
    rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_rows.tsv"))][1:]
    out = {}
    for tag, proto in tag_protos:
        (r,) = [r for r in rows if r[1] == tag and r[4] == proto]
        out[f"{tag}/{proto}"] = {"row": int(r[0]), "run_id": r[2], "dataset": r[5], "model": r[6], "cells": int(r[10]),
                                 "lhr_dump_job": DUMP_JOB[int(r[0])], "dump": f"{LHR_DUMP}/{tag}/{proto}/ (copy: fig/dump/{tag}/{proto}/)"}
    return out
