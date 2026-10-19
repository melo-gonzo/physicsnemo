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
        if rgb is not None:
            poly = poly.copy()
            poly.cell_data["rgb"] = np.asarray(rgb, np.uint8)
            p.add_mesh(poly, scalars="rgb", rgb=True, preference="cell", **kw)
        elif scalars is not None:
            poly = poly.copy()
            poly.cell_data["f"] = np.asarray(scalars, np.float64)
            p.add_mesh(poly, scalars="f", cmap=cmap, clim=clim, preference="cell", **kw)
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
    p.enable_anti_aliasing("ssaa")
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
