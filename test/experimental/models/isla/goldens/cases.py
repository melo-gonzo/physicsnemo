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

"""The golden cases of ISLA's modules.

A case is a pair of functions: ``build()`` makes its inputs on the CPU (every random
draw through a seeded ``torch.Generator``; the tensors are recorded and hashed as
``in.*``, since some pass through library primitives and LAPACK), and ``run(inputs,
device, variant)`` calls the functions under test and returns every output and
gradient (``out.*``, ``grad.*``). ``variant`` names an implementation choice of the
device (the reference or the fast/fused kernels); the first CPU variant is the one
the committed golden stores, and every other variant of every device is compared to
that golden at the tolerances of ``test_isla_goldens.py``.

Every function under test is reached through :data:`TARGETS`, one line per function
naming the module that owns it, so that moving a function into the library is a
one-line change here (its golden must then still pass, bitwise in the A/B harness).
"""

import fnmatch
import functools
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

D = torch.float64
F = torch.float32
ISLA = "physicsnemo.experimental.models.isla"

#: The module that owns each function under test: ``"module:attribute"``, or a
#: bare module for the G-GEO spies (which patch that module's private search ops
#: to record their outputs). LIB-MOVE retargets these lines.
TARGETS = {
    # G-FRAME, G-SAMP (LIB-MOVE P4)
    "canonical_frame": f"{ISLA}.tree_read:canonical_frame",
    "frame_coordinates": f"{ISLA}.tree_read:frame_coordinates",
    "water_fill": f"{ISLA}.sampling:water_fill",
    "pps_inclusion_probabilities": f"{ISLA}.sampling:pps_inclusion_probabilities",
    "draw_inputs": f"{ISLA}.sampling:draw_inputs",
    "sample_boundary_cells": f"{ISLA}.sampling:sample_boundary_cells",
    # G-FOREST, G-WALK (LIB-MOVE P1, P2: the build and walk behind these seams)
    "BoundaryForest": f"{ISLA}.tree_read:BoundaryForest",
    "node_means": f"{ISLA}.tree_read:node_means",
    "fused_node_means": f"{ISLA}.fused_read:fused_node_means",
    # G-SEG (LIB-MOVE P5)
    "PairPlan": f"{ISLA}.segment_softmax:PairPlan",
    "segment_softmax_read": f"{ISLA}.segment_softmax:segment_softmax_read",
    # G-GEO (LIB-MOVE P6)
    "exact_boundary_geometry": f"{ISLA}.geometry:exact_boundary_geometry",
    "geometry_spies": f"{ISLA}.geometry",
    "canonical_cells_reference": f"{ISLA}.geometry:_canonical_cells",
    "closest_weights_reference": f"{ISLA}.geometry:_closest_weights",
    "zero_area": f"{ISLA}.geometry:_zero_area",
    "closest_weights_constants": f"{ISLA}.geometry",
    "run_start": f"{ISLA}.geometry_fast:run_start",
    "canonical_cells": f"{ISLA}.geometry_fast:canonical_cells",
    "closest_weights": f"{ISLA}.geometry_fast:closest_weights",
    # G-READ (stays in ISLA)
    "tree_read_walk": f"{ISLA}.tree_read:tree_read_walk",
    "walk_range_pairs": f"{ISLA}.tree_read:walk_range_pairs",
    "walk_points_range": f"{ISLA}.tree_read:walk_points_range",
    "walk_pair_counts": f"{ISLA}.tree_read:walk_pair_counts",
    "tree_pair_invariants": f"{ISLA}.tree_read:tree_pair_invariants",
    "kernel_geo_width": f"{ISLA}.tree_read:kernel_geo_width",
    "KernelBasis": f"{ISLA}.tree_read:_KernelBasis",
    "TreeKernelRead": f"{ISLA}.tree_read:_TreeKernelRead",
    "pair_invariants_fused": f"{ISLA}.fused_basis:pair_invariants",
    "fused_kernel_read": f"{ISLA}.fused_read:fused_kernel_read",
    # G-BLOCKS
    "SliceBlock": f"{ISLA}.blocks:_SliceBlock",
    "RoutingLogits": f"{ISLA}.blocks:_RoutingLogits",
    "softmax_over_points": f"{ISLA}.blocks:_softmax_over_points",
    "relational_invariants": f"{ISLA}.blocks:_relational_invariants",
    "geo_width": f"{ISLA}.blocks:_geo_width",
    "geo_region": f"{ISLA}.blocks:_geo_region",
    "fused_vectors": f"{ISLA}.blocks:_fused_vectors",
    "fused_geo_region": f"{ISLA}.geo_kernel:fused_geo_region",
    # G-HEADS
    "levi_civita": f"{ISLA}.heads:_levi_civita",
    "isotropic_tensors": f"{ISLA}.heads:_isotropic_tensors",
    "OutputLayout": f"{ISLA}.heads:_OutputLayout",
    "assemble_output": f"{ISLA}.heads:_assemble_output",
    "pseudovectors": f"{ISLA}.heads:_pseudovectors",
    "polar_complement": f"{ISLA}.heads:_polar_complement",
    # G-MODEL, G-PARTS (the spies patch the names the model module calls)
    "ISLA": f"{ISLA}.model:ISLA",
    "model_module": f"{ISLA}.model",
}


def target(name: str):
    """The function (or module) :data:`TARGETS` names, imported now."""
    path, _, attr = TARGETS[name].partition(":")
    mod = importlib.import_module(path)
    return getattr(mod, attr) if attr else mod


@dataclass(frozen=True)
class Case:
    """One golden case (see the module docstring).

    ``kept`` and ``weights`` name the draw's kept cells and Horvitz-Thompson weights
    (read by the A/B harness's rules), ``family`` the CUDA outputs that are known not
    to repeat bitwise run to run (float atomics), judged against their run-to-run
    family there (fnmatch patterns). ``tol`` maps fnmatch patterns of output names to
    a relative tolerance (of ``max|golden|``) against the golden that replaces the
    dtype's, on every device (each with its reason and measurement at the case);
    ``cuda_tol`` does so for the CUDA variants alone, ``None`` meaning not compared
    on CUDA. ``requires`` lists what the CUDA variants need beyond CUDA. ``samples``
    maps fnmatch patterns of float outputs to the number of their elements the
    committed golden stores, where fewer than the default keep it within its budget."""

    group: str
    name: str
    build: Callable[[], dict]
    run: Callable[[dict, str, str], dict]
    variants: dict = field(
        default_factory=lambda: {"cpu": ("reference",), "cuda": ("auto",)}
    )
    kept: tuple = ()
    weights: tuple = ()
    family: tuple = ()
    tol: dict = field(default_factory=dict)
    cuda_tol: dict = field(default_factory=dict)
    requires: tuple = ()
    samples: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.group}/{self.name}"

    def matches(self, patterns, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, p) for p in patterns)

    def n_samples(self, name: str) -> int:
        """How many elements of the float tensor ``name`` the committed golden stores
        (``samples`` maps fnmatch patterns to fewer than the default)."""
        from test.experimental.models.isla.goldens.record import SAMPLES

        return next(
            (k for p, k in self.samples.items() if fnmatch.fnmatchcase(name, p)),
            SAMPLES,
        )


CASES: list[Case] = []


def _case(group, name, build, run, **kw):
    CASES.append(Case(group, name, build, run, **kw))


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _to(inputs: dict, device, keys=None) -> dict:
    return {k: v.to(device) for k, v in inputs.items() if keys is None or k in keys}


### Inputs are built from correctly rounded operations only (+, -, *, /, one per
### torch op, and float64 Generator draws, cast once to float32), so they are the
### same bits on every platform: transcendental functions and even ``sqrt``, fused
### reductions (``normalize``), matrix products and LAPACK (QR) round differently on
### x86 and Grace (measured in jobs 377631 and 377864), and a discontinuous output
### then moves with them.


def _directions(g: torch.Generator, *shape: int) -> torch.Tensor:
    """Unit vectors ``shape`` (last axis the dimension) without a square root: the
    inverse stereographic projection ``(2x, |x|^2 - 1) / (|x|^2 + 1)`` of Gaussian
    ``x`` in one dimension fewer (unit to rounding)."""
    x = torch.randn(*shape[:-1], shape[-1] - 1, generator=g, dtype=D)
    s = x[..., 0] * x[..., 0]
    for k in range(1, x.shape[-1]):
        s = s + x[..., k] * x[..., k]
    return torch.cat([2 * x, (s - 1)[..., None]], -1) / (s + 1)[..., None]


def _unit(x: torch.Tensor) -> torch.Tensor:
    """``x`` normalized along its last axis (only for orientation decisions)."""
    sq = x[..., 0] * x[..., 0]
    for k in range(1, x.shape[-1]):
        sq = sq + x[..., k] * x[..., k]
    return x / sq.sqrt()[..., None]


def _apply(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``x @ R.T`` by elementwise operations."""
    cols = []
    for j in range(R.shape[0]):
        acc = x[..., 0] * R[j, 0]
        for k in range(1, R.shape[1]):
            acc = acc + x[..., k] * R[j, k]
        cols.append(acc)
    return torch.stack(cols, -1)


def _rotation(d: int) -> torch.Tensor:
    """A proper rotation in a generic pose with rational entries: from the
    quaternion (1, 2, 3, 4) in three dimensions, the 3-4-5 triangle in two."""
    if d == 2:
        return torch.tensor([[3.0, -4.0], [4.0, 3.0]], dtype=D) / 5
    m = [[-20.0, 4.0, 22.0], [20.0, -10.0, 20.0], [10.0, 28.0, 4.0]]
    return torch.tensor(m, dtype=D) / 30


def _cloud(n, d=3, seed=0, *, dtype=D, scale=None):
    """``_meshes.cloud`` (the same draws), with its unit vectors normalized by
    :func:`_unit` and float32 clouds drawn in float64 and rounded once."""
    g = _gen(seed)
    scale = (
        torch.arange(d, 0, -1, dtype=D)
        if scale is None
        else torch.as_tensor(scale, dtype=D)
    )
    pts = torch.randn(n, d, generator=g, dtype=D) * scale
    nrm = _directions(g, n, d)
    w = torch.rand(n, generator=g, dtype=D) + 0.5
    drv = _directions(g, d)
    return tuple(t.to(dtype) for t in (pts, nrm, w, drv))


def _cotangent(t: torch.Tensor, seed: int) -> torch.Tensor:
    """A fixed random cotangent for ``t`` (drawn in float64 on the CPU)."""
    return torch.randn(t.shape, generator=_gen(seed), dtype=D).to(t.device, t.dtype)


def _params(module: torch.nn.Module, seed: int, std: float = 0.3) -> dict:
    """Random values for every parameter of ``module`` (``param.<name>``), float64."""
    g = _gen(seed)
    return {
        f"param.{n}": torch.randn(p.shape, generator=g, dtype=D) * std
        for n, p in module.named_parameters()
    }


def _load(
    module: torch.nn.Module, inputs: dict, device, dtype, prefix="param."
) -> torch.nn.Module:
    module = module.to(device, dtype)
    with torch.no_grad():
        for n, p in module.named_parameters():
            p.copy_(inputs[prefix + n])
    return module


def _param_grads(module: torch.nn.Module, prefix="grad.param.", zero=()) -> dict:
    """Every parameter's gradient but those named in ``zero``: parameters whose
    gradient is zero in exact arithmetic (a constant that cancels in a softmax), so
    that what is computed is roundoff alone, which no golden can hold."""
    return {
        prefix + n: p.grad
        for n, p in module.named_parameters()
        if p.grad is not None and n not in zero
    }


# --------------------------------------------------------------------------- G-FRAME


def _frame_build(d, dtype, kind):
    def build():
        if kind == "cloud":
            pts, _, w, _ = _cloud(1500, d=d, seed=21, dtype=dtype)
        else:
            ### A symmetric body with distinct variances along the lab axes: every third
            ### moment vanishes, so each axis's sign is the tie branch's (CANONICAL_SKEW_TIE).
            lengths = torch.tensor([3.0, 2.0, 1.0][:d], dtype=dtype)
            axis = (torch.arange(5, dtype=dtype) - 2) / 2
            pts = torch.cartesian_prod(*[axis] * d) * lengths
            w = torch.ones(pts.shape[0], dtype=dtype)
        return {"in.points": pts, "in.measure": w}

    return build


def _frame_run(inputs, device, variant):
    x = _to(inputs, device)
    frame = target("canonical_frame")(x["in.points"], x["in.measure"])
    return {
        "out.frame": frame,
        "out.coords": target("frame_coordinates")(x["in.points"], frame),
    }


for _d in (2, 3):
    for _dt, _dn in ((D, "f64"), (F, "f32")):
        for _kind in ("cloud", "symmetric"):
            _case(
                "G-FRAME",
                f"frame_{_kind}_{_d}d_{_dn}",
                _frame_build(_d, _dt, _kind),
                _frame_run,
            )


# --------------------------------------------------------------------------- G-SAMP

_FILL_BUDGETS = (77, 300, 3540, 5000)


def _fill_build():
    return {"in.counts": torch.tensor([500.0, 40.0, 3000.0], dtype=D)}


def _fill_run(inputs, device, variant):
    counts = inputs["in.counts"].to(device)
    return {f"out.shares_{b}": target("water_fill")(counts, b) for b in _FILL_BUDGETS}


_case("G-SAMP", "water_fill", _fill_build, _fill_run)

_PPS_SHARES = (7.5, 77.3, 499.0, 600.0)


def _pps_build():
    _, _, w, _ = _cloud(500, seed=0)
    w[:5] *= 50.0  # five dominant cells, capped at pi = 1 for the larger shares
    return {"in.measures": w}


def _pps_run(inputs, device, variant):
    a = inputs["in.measures"].to(device)
    fn = target("pps_inclusion_probabilities")
    return {f"out.pi_{s}": fn(a, a.new_tensor(s)) for s in _PPS_SHARES}


_case("G-SAMP", "pps_inclusion_probabilities", _pps_build, _pps_run)

_DRAWS = {
    "3d": (
        (500, 40, 3000),
        3,
        ((77, 0.0), (300, 0.37), (300, 0.999), (3540, 0.5), (4000, 0.25)),
    ),
    "2d": ((300, 700), 2, ((128, 0.5), (999, 0.125))),
}


def _draw_build(sizes, d):
    def build():
        out = {}
        for i, n in enumerate(sizes):
            pts, _, w, _ = _cloud(n, d=d, seed=40 + i)
            out[f"in.centroids_{i}"], out[f"in.measures_{i}"] = pts + 2.0 * i, w
        return out

    return build


def _draw_run(sizes, draws):
    def run(inputs, device, variant):
        x = _to(inputs, device)
        m = [x[f"in.measures_{i}"] for i in range(len(sizes))]
        c = [x[f"in.centroids_{i}"] for i in range(len(sizes))]
        out = {}
        for budget, start in draws:
            kept, w = target("sample_boundary_cells")(
                m, c, budget, torch.tensor(start, dtype=D, device=device)
            )
            out[f"out.kept_{budget}_{start}"], out[f"out.weights_{budget}_{start}"] = (
                kept,
                w,
            )
        return out

    return run


#: How far every recorded draw's hits ``start + m`` keep from the cells' boundaries on
#: the line (``test_the_draw_goldens_keep_off_the_cell_boundaries``): far above the
#: float64 rounding of the line's sums (at most ``k`` ulps of ``k``, 5e-10 here), so a
#: change of their summation order (DRAW-EXACT's exact sums, CUDA's scan) keeps the
#: same cells. Measured smallest margins: 5.9e-5 (G-SAMP), see README.md.
DRAW_MARGIN = 1e-9


def draw_specs() -> list[tuple]:
    """Every draw a golden records: ``(name, measures, centroids, budget, start)``,
    the inputs of ``sample_boundary_cells`` (float64, on the CPU)."""
    specs = []
    for name, (sizes, d, draws) in _DRAWS.items():
        x = _draw_build(sizes, d)()
        m = [x[f"in.measures_{i}"] for i in range(len(sizes))]
        c = [x[f"in.centroids_{i}"] for i in range(len(sizes))]
        specs += [(f"{name}_{b}_{s}", m, c, b, s) for b, s in draws]
    ### The model's draw (G-MODEL's dtt_draw): its conditions in canonical order, the
    ### draw's inputs from the float32 meshes, the evaluation start.
    from physicsnemo.mesh import Mesh

    spec = _MODEL_SPECS["dtt_draw"](F)
    x = _model_build(spec)()
    draw = [
        target("draw_inputs")(
            Mesh(points=x[f"in.{n}.points"].to(F), cells=x[f"in.{n}.cells"])
        )
        for n in sorted(spec.meshes)
    ]
    budget = spec.cfg["n_boundary_tokens"]
    start = target("model_module").EVAL_START
    specs.append(
        ("model_dtt_draw", [a for a, _ in draw], [c for _, c in draw], budget, start)
    )
    return specs


for _name, (_sizes, _d, _draws) in _DRAWS.items():
    _case(
        "G-SAMP",
        f"sample_boundary_cells_{_name}",
        _draw_build(_sizes, _d),
        _draw_run(_sizes, _draws),
        kept=("out.kept_*",),
        weights=("out.weights_*",),
    )


def _draw_inputs_build():
    pts, _, w, _ = _cloud(900, seed=0)
    ### One triangle per cell around each point (vertex mean = the point), with the
    ### cloud's measures as the stored effective measures.
    corners = 0.1 * torch.randn(900, 3, 3, generator=_gen(1), dtype=D)
    corners = corners - corners.mean(1, keepdim=True)
    verts = (pts[:, None] + corners).reshape(-1, 3)
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    sph = sphere_icosahedral.load(radius=1.5, subdivisions=2)
    return {
        "in.soup_points": verts,
        "in.soup_cells": torch.arange(2700).reshape(900, 3),
        "in.soup_measures": w,
        "in.sphere_points": sph.points.float() + 40.0,
        "in.sphere_cells": sph.cells,
    }


def _draw_inputs_run(inputs, device, variant):
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY

    x = _to(inputs, device)
    fn = target("draw_inputs")
    soup_mesh = Mesh(
        points=x["in.soup_points"],
        cells=x["in.soup_cells"],
        cell_data={EFFECTIVE_MEASURE_KEY: x["in.soup_measures"]},
    )
    out = dict(zip(("out.soup_measure", "out.soup_centroids"), fn(soup_mesh)))
    ### float32 points and no stored measure: the geometric measures, in float64.
    sph = Mesh(points=x["in.sphere_points"], cells=x["in.sphere_cells"])
    out.update(zip(("out.sphere_measure", "out.sphere_centroids"), fn(sph)))
    return out


_case("G-SAMP", "draw_inputs", _draw_inputs_build, _draw_inputs_run)


# --------------------------------------------------------------------------- G-FOREST

_FOREST_FIELDS = (
    "frame",
    "node_offset",
    "cells",
    "range_start",
    "range_count",
    "left",
    "right",
    "leaf_ids",
    "position_leaf",
    "aabb_min",
    "aabb_max",
    "diameter_sq",
    "center",
    "normal",
    "vecs",
    "measure",
    "coherent",
)
#: The trees' own integer tables (their float boxes are the forest's ``aabb_*``).
_TREE_FIELDS = (
    "node_left_child",
    "node_right_child",
    "leaf_start",
    "leaf_count",
    "node_range_start",
    "node_range_count",
    "sorted_source_order",
    "max_depth",
    "root_ids",
)
_BUILDS = {"cpu": ("reference", "fast"), "cuda": ("fast", "reference")}


def _forest_outputs(f) -> dict:
    out = {
        f"out.{k}": getattr(f, k) for k in _FOREST_FIELDS if getattr(f, k) is not None
    }
    for i, level in enumerate(f.levels):
        out.update(
            {
                f"out.levels.{i}.{k}": t
                for k, t in zip(("nodes", "left", "right"), level)
            }
        )
    for t, tree in enumerate(f.trees):
        out[f"out.tree_cells.{t}"] = f.tree_cells[t]
        out.update({f"out.trees.{t}.{k}": getattr(tree, k) for k in _TREE_FIELDS})
    return out


def _forest_build(d, dtype, kind):
    def build():
        pts, nrm, w, _ = _cloud(1500, d=d, seed=21, dtype=dtype)
        vecs = torch.randn(1500, 2, d, generator=_gen(22), dtype=D).to(dtype)
        if kind == "dup":
            pts = pts.clone()
            pts[:700] = pts[0]  # equal Morton codes: the midpoint splits
            return {"in.points": pts, "in.measure": w, "in.normals": nrm}
        if kind == "tiny":
            return {
                "in.points": pts[:9],
                "in.measure": w[:9],
                "in.normals": nrm[:9],
                "in.vecs": vecs[:9],
            }
        return {"in.points": pts, "in.measure": w, "in.normals": nrm, "in.vecs": vecs}

    return build


def _forest_run(inputs, device, variant):
    x = _to(inputs, device)
    f = target("BoundaryForest").build(
        x["in.points"],
        x["in.measure"],
        x["in.normals"],
        x.get("in.vecs"),
        theta=1.0,
        leaf_size=4,
        build=variant,
    )
    return _forest_outputs(f)


for _d in (3, 2):
    for _dt, _dn in ((D, "f64"), (F, "f32")):
        for _kind in ("cloud", "dup", "tiny"):
            _case(
                "G-FOREST",
                f"forest_{_kind}_{_d}d_{_dn}",
                _forest_build(_d, _dt, _kind),
                _forest_run,
                variants=_BUILDS,
                requires=("warp",),
            )


def _node_means_build(dtype):
    def build():
        pts, nrm, w, _ = _cloud(300, d=3, seed=9)
        x = torch.randn(300, 6, generator=_gen(10), dtype=D).to(dtype)
        return {"in.points": pts, "in.normals": nrm, "in.measure": w, "in.x": x}

    return build


def _node_means_run(inputs, device, variant):
    x = _to(inputs, device)
    f = target("BoundaryForest").build(
        x["in.points"],
        x["in.measure"],
        x["in.normals"],
        None,
        theta=0.5,
        leaf_size=8,
        build="reference",
    )
    rows = x["in.x"].clone().requires_grad_()
    fn = target("fused_node_means") if variant == "fused" else target("node_means")
    means = fn(rows, f, x["in.measure"])
    (means * _cotangent(means, 11)).sum().backward()
    return {"out.means": means.detach(), "grad.x": rows.grad}


_case(
    "G-FOREST",
    "node_means_f64",
    _node_means_build(D),
    _node_means_run,
    variants={"cpu": ("reference",), "cuda": ("reference",)},
    family=("out.means",),
)
_case(
    "G-FOREST",
    "node_means_f32",
    _node_means_build(F),
    _node_means_run,
    variants={"cpu": ("reference",), "cuda": ("fused", "reference")},
    family=("out.means",),
    requires=("triton",),
)


# --------------------------------------------------------------------------- G-WALK

_WALKS = {"cpu": ("reference", "fast"), "cuda": ("fast", "reference")}


def _walk_build(d, dtype):
    def build():
        pts, nrm, w, _ = _cloud(900, d=d, seed=3, dtype=dtype)
        q, _, _, _ = _cloud(150, d=d, seed=4, dtype=dtype, scale=(3.0,) * d)
        q[5] = float("nan")  # opened down to every cell
        q[6, 0] = float("inf")
        return {"in.points": pts, "in.normals": nrm, "in.measure": w, "in.queries": q}

    return build


def _walk_run(theta):
    def run(inputs, device, variant):
        x = _to(inputs, device)
        f = target("BoundaryForest").build(
            x["in.points"], x["in.measure"], x["in.normals"], None, theta=theta
        )
        qi, si, n_node = f.point_pairs(x["in.queries"], walk=variant)
        counts = f.frame_pairs(x["in.queries"] @ f.frame, walk=variant, count_only=True)
        return {
            "out.qi": qi,
            "out.si": si,
            "out.n_node": torch.tensor(n_node),
            "out.counts": counts,
        }

    return run


for _d in (3, 2):
    for _dt, _dn in ((D, "f64"), (F, "f32")):
        for _theta in (0.5, 1.0):
            _case(
                "G-WALK",
                f"walk_{_d}d_{_dn}_theta{_theta}",
                _walk_build(_d, _dt),
                _walk_run(_theta),
                variants=_WALKS,
                requires=("warp",),
            )


# --------------------------------------------------------------------------- G-SEG


def _plan_build():
    g = _gen(4)
    n_q, n_b = 60, 500
    counts = torch.randint(0, 200, (n_q,), generator=g)
    counts[::7] = 0
    counts[-3:] = 0  # the last points have no pairs
    qi = torch.repeat_interleave(torch.arange(n_q), counts)
    bj = torch.randint(
        0, n_b - 40, (qi.shape[0],), generator=g
    )  # the last sources have none
    bj[: qi.shape[0] // 3] = 17  # one source read by a third of the pairs
    return {"in.qi": qi, "in.bj": bj, "in.n": torch.tensor([n_q, n_b])}


def _plan_run(inputs, device, variant):
    x = _to(inputs, device)
    n_q, n_b = inputs["in.n"].tolist()
    out = {}
    for tag, cast in (("i64", lambda t: t), ("i32", lambda t: t.int())):
        plan = target("PairPlan").build(cast(x["in.qi"]), cast(x["in.bj"]), n_q, n_b)
        out.update(
            {
                f"out.{tag}.{k}": getattr(plan, k)
                for k in ("q_offsets", "perm", "b_offsets", "qi_by_b")
            }
        )
    return out


_case("G-SEG", "pair_plan", _plan_build, _plan_run)


def _seg_build(dtype, kernel_read=False):
    def build():
        g = _gen(3)
        n_q, n_b, h, kc, c, f = 37, 50, 8, 4, 8, 16
        counts = torch.randint(0, 500, (n_q,), generator=g)
        counts[3] = 0
        p = int(counts.sum())
        qi = torch.repeat_interleave(torch.arange(n_q), counts)
        bj = torch.randint(
            0, n_b - 5, (p,), generator=g
        )  # sources 45..49 have no pairs
        bj[: p // 2] = 0  # one source read by half of the pairs
        x = {"in.qi": qi, "in.bj": bj, "in.n": torch.tensor([n_q, n_b])}
        if kernel_read:
            x["in.basis"] = torch.randn(p, f, generator=g, dtype=D).to(dtype)
            x["in.w"] = (0.3 * torch.randn(h, f, generator=g, dtype=D)).to(dtype)
            x["in.logm"] = (-torch.rand(n_b, generator=g, dtype=D) * 8).to(
                F
            )  # per source
        else:
            x["in.ker"] = torch.randn(p, h, generator=g, dtype=D).to(dtype)
            x["in.logm"] = (-torch.rand(p, generator=g, dtype=D) * 8).to(
                D if dtype == D else F
            )
        x["in.q"] = torch.randn(n_q, h, kc, generator=g, dtype=D).to(dtype)
        x["in.k"] = torch.randn(n_b, h, kc, generator=g, dtype=D).to(dtype)
        x["in.v"] = torch.randn(n_b, h, c, generator=g, dtype=D).to(dtype)
        return x

    return build


def _seg_run(inputs, device, variant):
    x = _to(inputs, device)
    n_q, n_b = inputs["in.n"].tolist()
    plan = target("PairPlan").build(x["in.qi"], x["in.bj"], n_q, n_b)
    leaves = {k: x[f"in.{k}"].clone().requires_grad_() for k in ("ker", "q", "k", "v")}
    out = target("segment_softmax_read")(
        leaves["ker"], x["in.logm"], leaves["q"], leaves["k"], leaves["v"], plan, 0.5
    )
    (out * _cotangent(out, 5)).sum().backward()
    return {
        "out.read": out.detach(),
        **{f"grad.{k}": t.grad for k, t in leaves.items()},
    }


def _kernel_read_run(inputs, device, variant):
    """``fused_kernel_read`` on CUDA (variant "auto"); its definition,
    ``segment_softmax_read(basis @ w.T, logm[bj], ...)``, otherwise."""
    x = _to(inputs, device)
    n_q, n_b = inputs["in.n"].tolist()
    plan = target("PairPlan").build(x["in.qi"], x["in.bj"], n_q, n_b)
    lv = {
        k: x[f"in.{k}"].clone().requires_grad_() for k in ("basis", "w", "q", "k", "v")
    }
    if variant == "auto":
        out = target("fused_kernel_read")(
            lv["basis"], lv["w"], x["in.logm"], lv["q"], lv["k"], lv["v"], plan, 0.5
        )
    else:
        ker = lv["basis"] @ lv["w"].t()
        out = target("segment_softmax_read")(
            ker, x["in.logm"][plan.bj.long()], lv["q"], lv["k"], lv["v"], plan, 0.5
        )
    (out * _cotangent(out, 5)).sum().backward()
    return {"out.read": out.detach(), **{f"grad.{k}": t.grad for k, t in lv.items()}}


#: Measured run-to-run on CUDA: the Triton read's ``dk``/``dv`` accumulate by atomics,
#: and the eager read (float64 there) sums by ``index_add``, atomics too.
_SEG_FAMILY = ("grad.k", "grad.v")
_SEG_FAMILY_F64 = ("out.read", "grad.*")
#: bfloat16 ``dk``/``dv`` are not compared on CUDA: the CPU golden is the eager read's,
#: whose backward accumulates the gathered keys' and values' gradients in bfloat16
#: (29% off float64 at the largest ``dv``: it gathers before it casts; reported), the Triton read's in float32.
_BF16_CUDA = {"grad.k": None, "grad.v": None}
for _dt, _dn in ((D, "f64"), (F, "f32"), (torch.bfloat16, "bf16")):
    _case(
        "G-SEG",
        f"segment_softmax_read_{_dn}",
        _seg_build(_dt),
        _seg_run,
        family=_SEG_FAMILY_F64 if _dt == D else _SEG_FAMILY,
        cuda_tol=_BF16_CUDA if _dt == torch.bfloat16 else {},
        requires=("triton",),
    )
for _dt, _dn in ((F, "f32"), (torch.bfloat16, "bf16")):
    _case(
        "G-READ",
        f"kernel_read_{_dn}",
        _seg_build(_dt, kernel_read=True),
        _kernel_read_run,
        variants={"cpu": ("reference",), "cuda": ("auto", "reference")},
        family=_SEG_FAMILY,
        cuda_tol=_BF16_CUDA if _dt == torch.bfloat16 else {},
        requires=("triton",),
    )


# --------------------------------------------------------------------------- G-GEO
### Surfaces after the geometry test files' builders (copied, so that a change there
### cannot move these goldens): closed, open, tied, on-wall, thin and degenerate sets.


def _oriented(points, cells, outward):
    from physicsnemo.mesh import Mesh

    m = Mesh(points=points, cells=cells)
    cells = cells.clone()
    flip = (m.cell_normals * outward(m.cell_centroids)).sum(-1) < 0
    cells[flip, -2:] = cells[flip, -2:].flip(-1)
    return points, cells


def _polygon(verts):
    n = verts.shape[0]
    cells = torch.stack([torch.arange(n), torch.arange(1, n + 1) % n], -1)
    x, y = verts[:, 0], verts[:, 1]
    if (x * y.roll(-1) - x.roll(-1) * y).sum() > 0:
        cells = cells.flip(-1)  # outward normals
    return verts, cells


def _near_wall(dim, n, seed, band, radius=1.0):
    g = _gen(seed)
    dirs = _directions(g, n, dim)
    return dirs * (radius + band * (2 * torch.rand(n, 1, dtype=D, generator=g) - 1))


def _geo_sphere():
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    m = sphere_icosahedral.load(radius=1.0, subdivisions=2)
    p, c = _oriented(m.points.double(), m.cells, lambda x: x)
    q = torch.cat([_near_wall(3, 200, 1, band=0.5), p, p[c[:64, :2]].mean(1)])
    return p, c, q


def _geo_torus():
    from physicsnemo.mesh.primitives.surfaces import torus

    m = torus.load(n_major=16, n_minor=8)

    def outward(x):
        return x - _unit(x * torch.tensor([1.0, 1.0, 0.0], dtype=x.dtype))

    p, c = _oriented(m.points.double(), m.cells, outward)
    ### (generic queries only: the core circle's ties reach the planar quads' diagonals)
    return p, c, torch.randn(200, 3, dtype=D, generator=_gen(2))


def _cube(levels=0):
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.primitives.surfaces import cube_surface

    m = cube_surface.load()
    m = Mesh(points=m.points.double(), cells=m.cells)
    return m.subdivide(levels=levels, filter="linear") if levels else m


def _geo_box_ties(dim):
    """The unit square/cube (outward) in a generic pose and points on its medial
    surface: equidistant from two sides, three, and all (the centre)."""
    a = (torch.arange(1, 10, dtype=D) / 20) - 0.5  # -0.45, ..., -0.05
    z = torch.zeros_like(a)
    if dim == 2:
        p, c = _polygon(
            torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=D)
        )
        q = torch.cat(
            [
                torch.stack([s * a, t * a], -1)
                for s, t in ((1, 1), (-1, 1), (1, -1), (-1, -1))
            ]
        )
    else:
        m = _cube()
        p, c = m.points, m.cells
        ### (not the corner diagonals or the centre: their closest points lie on the
        ### faces' quad diagonals, a near-coplanar tie; see README.md, "Exclusions")
        q = torch.cat([torch.stack(v, -1) for v in ((a, a, z), (-a, z, a), (z, a, -a))])
    if dim == 2:
        q = torch.cat([q, q.new_zeros(1, dim)])
    R, t = _rotation(dim), torch.arange(1, dim + 1, dtype=D) / 3
    return _apply(R, p) + t, c, _apply(R, q) + t


def _geo_box_wall(dim):
    """The unit square/cube (outward), its sides split, and points exactly on it: the
    cells' centroids, and the vertices and edge midpoints on the box's own edges and
    corners (not those inside a side, where coplanar cells meet: see README.md,
    "Exclusions")."""
    if dim == 2:
        h = torch.tensor([-0.5, 0.0, 0.5, 0.5, 0.5, 0.0, -0.5, -0.5], dtype=D)
        p, c = _polygon(torch.stack([h, h.roll(-2)], -1))
        pairs = ((0, 1),)
    else:
        m = _cube(1)
        p, c = m.points, m.cells
        pairs = ((0, 1), (1, 2), (0, 2))
    q = torch.cat([p, *[p[c[:, [a, b]]].mean(1) for a, b in pairs]])
    q = torch.cat([p[c].mean(1), q[(q.abs() == 0.5).sum(-1) >= 2]])
    return p, c, q


def _geo_thin_cap():
    """The unit cube with a cap of width 1e-6 in its bottom face (the thin set),
    rotated and shifted, with queries on, just off and beside the cap, and in the box."""
    p = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 1e-6, 0],
        ],
        dtype=D,
    )
    c = torch.tensor(
        [[0, 1, 8], [0, 8, 3], [8, 2, 3], [8, 1, 2], [4, 5, 6], [4, 6, 7], [0, 4, 5], [0, 5, 1], [1, 5, 6],
         [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
    )  # fmt: skip
    p, c = _oriented(p, c, lambda x: x - 0.5)
    g = _gen(11)
    tri = p[c[0]]
    w = torch.rand(20, 3, dtype=D, generator=g)
    w = w / (w[:, :1] + w[:, 1:2] + w[:, 2:])
    on = w[:, :1] * tri[0] + w[:, 1:2] * tri[1] + w[:, 2:] * tri[2]
    nrm = torch.tensor(
        [0.0, 0.0, 1.0], dtype=D
    )  # the cap lies in z = 0 (before the rotation)
    side = torch.rand(20, 1, dtype=D, generator=g) * 2 - 1
    q = torch.cat(
        [
            on,
            on + 1e-6 * side * nrm,
            on + 1e-5 * side * nrm,
            torch.rand(40, 3, dtype=D, generator=g) * 1.4 - 0.2,
        ]
    )
    R, t = _rotation(3), torch.tensor([0.3, -0.2, 0.7], dtype=D)
    return _apply(R, p) + t, c, _apply(R, q) + t


def _geo_sliver_soup():
    """Random triangles, the first eight slivers 1e-13 off their line (the degenerate set)."""
    g = _gen(500)
    tri = torch.rand(48, 3, 3, dtype=D, generator=g)
    tri[:8, 1] = 0.5 * (tri[:8, 0] + tri[:8, 2]) + 1e-13 * torch.randn(
        8, 3, dtype=D, generator=g
    )
    mid = 0.5 * (tri[:8, 0] + tri[:8, 2])
    q = torch.cat(
        [
            torch.rand(80, 3, dtype=D, generator=g) * 1.4 - 0.2,
            (mid[:, None] + 0.01 * torch.randn(8, 5, 3, dtype=D, generator=g)).reshape(
                -1, 3
            ),
            tri[:8].reshape(-1, 3),
            mid,
        ]
    )
    return tri.reshape(-1, 3), torch.arange(144).reshape(-1, 3), q


def _geo_zero_area_box():
    """A closed box with zero-area cells (three coincident vertices, and two coincident)
    added at its vertices and along its edges."""
    m = _cube(1)
    p, c = m.points, m.cells
    extra = torch.cat([c[::6][:, [0, 0, 0]], c[1::6][:, [0, 0, 1]]])
    corners = p[
        (p.abs() == 0.5).all(-1)
    ]  # not the vertices inside a side (coplanar cells)
    q = torch.cat(
        [torch.rand(120, 3, dtype=D, generator=_gen(12)) * 1.6 - 0.8, corners]
    )
    return p, torch.cat([c, extra]), q


def _geo_open_cube():
    m = _cube(1)
    keep = m.points[m.cells].mean(1)[:, 2] < 0.49
    return m.points, m.cells[keep], torch.randn(200, 3, dtype=D, generator=_gen(13))


def _geo_circle():
    from physicsnemo.mesh.primitives.curves import circle_2d

    m = circle_2d.load(n_points=48)
    p, c = _oriented(m.points.double(), m.cells, lambda x: x)
    return p, c, torch.cat([_near_wall(2, 150, 3, band=0.5), p, p[c].mean(1)])


def _geo_naca():
    """A NACA 0012 section with a sharp trailing edge (24 stations at ``x = t^2``,
    clustered at the leading edge, so that ``sqrt(x) = t`` takes no square root)."""
    t = torch.arange(24, dtype=D) / 23
    x = t * t
    x2 = x * x
    yt = 0.6 * (
        0.2969 * t - 0.1260 * x - 0.3516 * x2 + 0.2843 * x2 * x - 0.1036 * x2 * x2
    )
    verts = torch.cat(
        [torch.stack([x, yt], -1), torch.stack([x, -yt], -1).flip(0)[1:-1]]
    )
    p, c = _polygon(verts)
    q = torch.rand(150, 2, dtype=D, generator=_gen(14)) * torch.tensor(
        [1.6, 0.6], dtype=D
    ) - torch.tensor([0.3, 0.3], dtype=D)
    return p, c, torch.cat([q, p])


_GEO_SETS = {
    "sphere": _geo_sphere,
    "torus": _geo_torus,
    "box_ties_3d": lambda: _geo_box_ties(3),
    "box_wall_3d": lambda: _geo_box_wall(3),
    "thin_cap": _geo_thin_cap,
    "sliver_soup": _geo_sliver_soup,
    "zero_area_box": _geo_zero_area_box,
    "open_cube": _geo_open_cube,
    "circle": _geo_circle,
    "naca": _geo_naca,
    "box_ties_2d": lambda: _geo_box_ties(2),
    "box_wall_2d": lambda: _geo_box_wall(2),
}
_GEO_SEARCH = ("cell", "tied", "tie_u", "sign", "on_wall", "weights")


def _geo_build(maker, dtype):
    def build():
        p, c, q = maker()
        return {"in.queries": q.to(dtype), "in.points": p.to(dtype), "in.cells": c}

    return build


#: Outputs not recorded on the thin cap: its cap and the cells around it are coplanar
#: (near-coplanar after the rotation), so its queries meet near-coplanar ties, where
#: the direction is a known defect (README.md, "Exclusions"); d, the closest cell,
#: the sign, the on-wall decision, the weights and the winding number stay.
_GEO_DROP = {
    "thin_cap": (
        "out.u",
        "grad.queries",
        "grad.points",
        "out.search.tied",
        "out.search.tie_u",
    ),
    ### The closest point's weights on slivers 1e-13 off their line are ill-posed
    ### (x86 and Grace differ by 3e-5 on the same inputs); d and the rest stay.
    "sliver_soup": ("out.search.weights",),
    ### At a tie the reported cell, its weights and the gradients (taken through that
    ### cell: a selection from the subdifferential) differ between x86, Grace and
    ### GB300 on the same inputs, and move under a one-ulp change of the vertices;
    ### d, u, the tie flag, its direction, the sign and the winding numbers stay.
    "box_ties_3d": (
        "out.search.cell",
        "out.search.weights",
        "grad.queries",
        "grad.points",
    ),
    "box_ties_2d": (
        "out.search.cell",
        "out.search.weights",
        "grad.queries",
        "grad.points",
    ),
}


def _geo_run(inputs, device, variant, drop=()):
    """``d``, ``u``, their gradients in the queries and the vertices (a fixed
    cotangent), the six outputs of the closest-feature search and the winding number,
    recorded by spies on the geometry module's search ops (as test_isla_geometry_fast)."""
    x = _to(inputs, device)
    geo = target("geometry_spies")
    seen = {}
    saved = {
        n: getattr(geo, n)
        for n in (
            "_closest_candidates",
            "_closest_candidates_fast",
            "_winding_sign",
            "_tree_winding_numbers",
        )
    }

    def search(*args, _op):
        seen["search"] = _op(*args)
        return seen["search"]

    def sign(query, simplex, parity, tree, w=None, oriented=None):
        if w is not None:
            seen["w"] = w / (4 * torch.pi)
        return saved["_winding_sign"](query, simplex, parity, tree, w, oriented)

    def tree_w(*args):
        seen["w"] = saved["_tree_winding_numbers"](*args)
        return seen["w"]

    geo._closest_candidates = lambda *a: search(*a, _op=saved["_closest_candidates"])
    geo._closest_candidates_fast = lambda *a: search(
        *a, _op=saved["_closest_candidates_fast"]
    )
    geo._winding_sign, geo._tree_winding_numbers = sign, tree_w
    try:
        q = x["in.queries"].clone().requires_grad_()
        p = x["in.points"].clone().requires_grad_()
        d, u = target("exact_boundary_geometry")(q, p, x["in.cells"], kernel=variant)
        ramp = (
            torch.arange(u.numel(), dtype=u.dtype, device=u.device)
            / max(u.numel() - 1, 1)
        ).reshape(u.shape)
        (d.sum() + (u * ramp).sum()).backward()
    finally:
        for n, f in saved.items():
            setattr(geo, n, f)
    out = {
        "out.d": d.detach(),
        "out.u": u.detach(),
        "grad.queries": q.grad,
        "grad.points": p.grad,
    }
    out.update({f"out.search.{k}": t for k, t in zip(_GEO_SEARCH, seen["search"])})
    if "w" in seen:  # the 3-D tree's winding numbers (2-D signs by its segment sums)
        out["out.w"] = seen["w"]
    ### On the wall d is 0 whatever the sign, and the winding number there is 1/2 up
    ### to rounding: both carry no information and differ across platforms, so they
    ### are recorded as 0 at the on-wall queries.
    wall = out["out.search.on_wall"]
    out["out.search.sign"] = torch.where(wall, 0, out["out.search.sign"])
    if "out.w" in out:
        out["out.w"] = torch.where(wall, 0, out["out.w"])
    ### Where the closest point lies on a cell's edge or vertex (a weight within 16 eps
    ### of the queries' dtype of 0), every cell that shares it is at the same distance:
    ### the reported cell and its weights are a selection among them, which the search's
    ### order of exact distance ties decides (GEO-TIEORDER's T1 changes it; README.md,
    ### "Exclusions"). They are recorded as -1 and 0 there; d and u stay.
    weights = out["out.search.weights"]
    edge = weights.min(-1).values <= 16 * torch.finfo(x["in.queries"].dtype).eps
    out["out.search.cell"] = torch.where(edge, -1, out["out.search.cell"])
    out["out.search.weights"] = torch.where(edge[:, None], 0.0, weights)
    return {k: v for k, v in out.items() if k not in drop}


#: Measured run-to-run on CUDA: the vertex gradient accumulates by float atomics.
_GEO_FAMILY = ("grad.points",)
#: The thin cap's closest-point weights: conditioned by 1/h = 1e6 (x86 against Grace,
#: with queries one ulp apart: 2.8e-11).
_GEO_TOL = {"thin_cap": {"out.search.weights": 1e-8}}
#: The 3-D winding numbers are not compared with the golden (the A/B harness records
#: them bitwise): their values follow the tree's float32 opening decisions, which
#: differ between x86, Grace and GB300 on the same inputs (by up to 4.4e-3 near the
#: sphere), within the far-field approximation; their sign, which is what reaches d,
#: is compared (``out.search.sign``).
_GEO_TOL = {**_GEO_TOL, "*": {"out.w": None}}
for _name, _maker in _GEO_SETS.items():
    _dtypes = (
        ((D, "f64"), (F, "f32"))
        if _name in ("sphere", "box_wall_3d", "circle", "thin_cap")
        else ((D, "f64"),)
    )
    for _dt, _dn in _dtypes:
        _case(
            "G-GEO",
            f"exact_geometry_{_name}_{_dn}",
            _geo_build(_maker, _dt),
            functools.partial(_geo_run, drop=_GEO_DROP.get(_name, ())),
            variants={"cpu": ("reference",), "cuda": ("fast", "reference")},
            family=_GEO_FAMILY,
            tol={**_GEO_TOL["*"], **_GEO_TOL.get(_name, {})},
            requires=("warp",),
        )


def _geo_parts_build():
    g = _gen(15)
    new = torch.rand(3000, generator=g) < 0.1
    new[0] = True
    p, c, _ = _geo_sphere()
    tri = torch.rand(400, 3, 3, dtype=D, generator=g)
    tri[:40, 1] = 0.5 * (tri[:40, 0] + tri[:40, 2]) + 1e-9 * torch.randn(
        40, 3, dtype=D, generator=g
    )  # thin: guarded
    tri[40:60, 2] = tri[40:60, 1]  # zero area: their segments
    q = torch.rand(400, 3, dtype=D, generator=g) * 1.4 - 0.2
    return {"in.new": new, "in.points": p, "in.cells": c, "in.q": q, "in.simplex": tri}


def _geo_parts_run(inputs, device, variant):
    """``run_start``, ``canonical_cells`` and ``closest_weights`` of geometry_fast:
    the Triton kernels on CUDA, the reference they stand for elsewhere."""
    x = _to(inputs, device)
    geo = target("closest_weights_constants")
    vert, parity = target("canonical_cells")(
        x["in.points"], x["in.cells"], target("canonical_cells_reference")
    )
    simplex = x["in.simplex"]
    flat, _ = target("zero_area")(
        simplex, torch.arange(simplex.shape[0] * 3, device=device).reshape(-1, 3)
    )
    w = target("closest_weights")(
        x["in.q"],
        simplex,
        flat,
        target("closest_weights_reference"),
        geo._ERICSON_K,
        geo._SWITCH_UNITS,
        geo._ORIENT_ERRBOUND,
    )
    return {
        "out.run_start": target("run_start")(x["in.new"]),
        "out.canonical_cells": vert,
        "out.parity": parity,
        "out.flat": flat,
        "out.closest_weights": w,
    }


_case(
    "G-GEO",
    "geometry_fast_parts",
    _geo_parts_build,
    _geo_parts_run,
    requires=("triton",),
)


# --------------------------------------------------------------------------- G-READ

_READS = {"cpu": ("reference",), "cuda": ("auto", "reference")}
_READ_RANGES = ((0, 40), (40, 95), (95, 120))


def _read_build(d, dtype, n_vecs):
    def build():
        pts, nrm, w, drv = _cloud(300, d=d, seed=0, dtype=D)
        g = _gen(1)
        near = pts[:110] + 0.15 * torch.randn(110, d, generator=g, dtype=D)
        far = 40.0 + torch.randn(10, d, generator=g, dtype=D)
        x = {
            "in.boundary_r": pts[None],
            "in.boundary_n": nrm[None],
            ### log measure fractions about log(1 / 300), drawn directly (no log)
            "in.log_frac": (-6.0 + w)[None],
            "in.query_r": torch.cat([near, far])[None],
            "in.query_n": _directions(g, 1, 120, d),
            "in.query_wall": 0.05 + torch.rand(1, 120, generator=g, dtype=D),
            "in.g_unit": drv[None, None],
            "in.h_b": torch.randn(300, 16, generator=g, dtype=D),
            "in.h_q": torch.randn(120, 16, generator=g, dtype=D),
        }
        if n_vecs:
            x["in.boundary_vecs"] = torch.randn(1, 300, n_vecs, d, generator=g, dtype=D)
        x = {k: v.to(dtype) for k, v in x.items()}
        basis = target("KernelBasis")(target("kernel_geo_width")(1, n_vecs))
        x.update({f"basis.{k}": v.to(dtype) for k, v in _params(basis, 2).items()})
        x.update(
            {
                f"read.{k}": v.to(dtype)
                for k, v in _params(target("TreeKernelRead")(16), 3).items()
            }
        )
        return x

    return build


def _read_run(n_vecs):
    def run(inputs, device, variant):
        """The tree read of one block on one case at theta 0.5: the walk's plan and
        lists, the pair invariants, the kernel basis, the read and every gradient, and
        the pairs of three ranges of points from a walk made without its lists."""
        x = _to(inputs, device)
        dtype = x["in.query_r"].dtype
        basis = _load(
            target("KernelBasis")(target("kernel_geo_width")(1, n_vecs)),
            x,
            device,
            dtype,
            "basis.param.",
        )
        read = _load(target("TreeKernelRead")(16), x, device, dtype, "read.param.")
        fast = variant == "auto"
        walk_kw = dict(
            query_r=x["in.query_r"],
            query_n=x["in.query_n"],
            query_wall=x["in.query_wall"],
            boundary_r=x["in.boundary_r"],
            boundary_n=x["in.boundary_n"],
            boundary_vecs=x.get("in.boundary_vecs"),
            log_frac=x["in.log_frac"],
            g_unit=x["in.g_unit"],
            theta=0.5,
            eps=1e-12,
            walk="auto" if fast else "reference",
            build="auto" if fast else "reference",
            kernel="fused" if fast else "reference",
        )
        walk = target("tree_read_walk")(**walk_kw)
        pairs = target("walk_range_pairs")(basis, walk)
        out = {
            f"out.plan.{k}": getattr(pairs.plan, k)
            for k in ("qi", "bj", "q_offsets", "perm", "b_offsets", "qi_by_b")
        }
        out["out.n_pairs_node"] = torch.tensor(pairs.n_pairs_node)
        out["out.log_m"], out["out.src_log_m"] = pairs.log_m, pairs.src_log_m
        qi, si, _ = walk.lists[0]
        geometry = walk.geometry[0]
        with torch.no_grad():
            if fast and dtype == F:
                inv = target("pair_invariants_fused")(qi, si, 0, 0, geometry)
            else:
                inv = target("tree_pair_invariants")(qi, si, **geometry)
        out["out.invariants"] = inv
        h_b = x["in.h_b"].clone().requires_grad_()
        h_q = x["in.h_q"].clone().requires_grad_()
        y = read(h_b, h_q, pairs)
        (y * _cotangent(y, 6)).sum().backward()
        out["out.basis"], out["out.read"] = pairs.basis.detach(), y.detach()
        out.update({"grad.h_b": h_b.grad, "grad.h_q": h_q.grad})
        out.update(_param_grads(read, "grad.read."))
        ### (the basis MLP's last bias adds a per-head constant to every pair's logit,
        ### which cancels in each point's softmax: its gradient is roundoff)
        out.update(_param_grads(basis, "grad.basis.", zero=("net.4.bias",)))
        ### The pairs of ranges of points (query_chunks' path): each range's lists and basis.
        lazy = target("tree_read_walk")(**walk_kw, walk_points=False)
        out["out.pair_counts"] = target("walk_pair_counts")(lazy)
        with torch.no_grad():
            for i, (a, b) in enumerate(_READ_RANGES):
                part = target("walk_range_pairs")(
                    basis, target("walk_points_range")(lazy, a, b)
                )
                out[f"out.range{i}.qi"], out[f"out.range{i}.bj"] = (
                    part.plan.qi,
                    part.plan.bj,
                )
                out[f"out.range{i}.basis"] = part.basis
        return out

    return run


#: Measured run-to-run on CUDA: the reference read's node means accumulate by float
#: atomics (index_add), and so does everything downstream of them.
_READ_FAMILY = ("out.read", "grad.*")
for _d, _dt, _dn, _nv in (
    (3, D, "f64", 1),
    (3, F, "f32", 1),
    (2, D, "f64", 0),
    (2, F, "f32", 0),
):
    _case(
        "G-READ",
        f"tree_read_{_d}d_{_dn}",
        _read_build(_d, _dt, _nv),
        _read_run(_nv),
        variants=_READS,
        family=_READ_FAMILY,
        requires=("warp", "triton"),
    )


# --------------------------------------------------------------------------- G-BLOCKS


def _block_inputs(g, n, hidden, dtype, k=1, n_vecs=1, n_tens=0):
    x = {
        "in.h": torch.randn(1, n, hidden, generator=g, dtype=D),
        "in.log_w": -3.0 * torch.rand(1, n, 1, generator=g, dtype=D),
        "in.r": torch.randn(1, n, 3, generator=g, dtype=D),
        "in.n_hat": _directions(g, 1, n, 3),
        "in.g_hat": _directions(g, 1, 1, k, 3).expand(1, n, k, 3).contiguous(),
    }
    if n_vecs:
        x["in.vecs"] = torch.randn(1, n, n_vecs, 3, generator=g, dtype=D)
    if n_tens:
        t = torch.randn(1, n, n_tens, 3, 3, generator=g, dtype=D)
        x["in.tens"] = t + t.transpose(-1, -2)
    return {k: v.to(dtype) for k, v in x.items()}


def _slice_block_build(dtype, n_vecs, n_tens):
    def build():
        g = _gen(30)
        x = _block_inputs(g, 200, 32, dtype, n_vecs=n_vecs, n_tens=n_tens)
        x.update(
            {
                f"q.{k}": v
                for k, v in _block_inputs(
                    g, 50, 32, dtype, n_vecs=n_vecs, n_tens=n_tens
                ).items()
            }
        )
        block = target("SliceBlock")(
            32, 8, n_global_vectors=1, n_block_vecs=n_vecs, n_block_tens=n_tens
        )
        x.update({k: v.to(dtype) for k, v in _params(block, 31, 0.2).items()})
        return x

    return build


def _slice_block_run(n_vecs, n_tens):
    def run(inputs, device, variant):
        """One slice block over 200 tokens (``forward(slices=True)``) and its read for
        50 passive tokens, with a fixed cotangent on both: outputs, the slices, and
        the gradients in the states, positions and parameters."""
        x = _to(inputs, device)
        dtype = x["in.h"].dtype
        geo = "fused" if variant == "auto" else "eager"
        block = _load(
            target("SliceBlock")(32, 8, geo, 1, n_vecs, n_tens), x, device, dtype
        )
        h, r, hq = (x[k].clone().requires_grad_() for k in ("in.h", "in.r", "q.in.h"))
        y, slices = block(
            h,
            x["in.log_w"],
            r,
            x["in.n_hat"],
            x["in.g_hat"],
            1e-12,
            x.get("in.vecs"),
            x.get("in.tens"),
            slices=True,
        )
        yq = block.read(
            hq,
            slices,
            x["q.in.r"],
            x["q.in.n_hat"],
            x["q.in.g_hat"],
            1e-12,
            x.get("q.in.vecs"),
            x.get("q.in.tens"),
        )
        ((y * _cotangent(y, 32)).sum() + (yq * _cotangent(yq, 33)).sum()).backward()
        out = {"out.h": y.detach(), "out.h_read": yq.detach()}
        out.update(
            {
                f"out.slices.{k}": t.detach()
                for k, t in zip(("protos", "z_pos", "m_s", "z"), slices)
            }
        )
        ### (geo_logit's bias is a constant over the slices, which cancels in both
        ### softmaxes: its gradient is roundoff)
        out.update({"grad.h": h.grad, "grad.r": r.grad, "grad.h_read": hq.grad})
        out.update(_param_grads(block, zero=("geo_logit.bias",)))
        return out

    return run


for _dt, _dn in ((D, "f64"), (F, "f32")):
    _case(
        "G-BLOCKS",
        f"slice_block_{_dn}",
        _slice_block_build(_dt, 1, 0),
        _slice_block_run(1, 0),
        variants={"cpu": ("reference",), "cuda": ("auto", "reference")},
        requires=("triton",),
    )
_case(
    "G-BLOCKS",
    "slice_block_tensors_f64",
    _slice_block_build(D, 1, 1),
    _slice_block_run(1, 1),
    variants={
        "cpu": ("reference",),
        "cuda": ("reference",),
    },  # the fused region takes no rank-2 inputs
)


def _routing_build(dtype):
    def build():
        g = _gen(34)
        x = {
            "in.x": torch.randn(1, 100, 32, generator=g, dtype=D).to(dtype),
            "in.log_w": (-3.0 * torch.rand(1, 100, 1, generator=g, dtype=D)).to(dtype),
        }
        x.update(
            {
                k: v.to(dtype)
                for k, v in _params(target("RoutingLogits")(32, 8), 35).items()
            }
        )
        return x

    return build


def _routing_run(inputs, device, variant):
    x = _to(inputs, device)
    mod = _load(target("RoutingLogits")(32, 8), x, device, x["in.x"].dtype)
    h = x["in.x"].clone().requires_grad_()
    protos = mod.prototypes(h, x["in.log_w"])
    logits = mod.logits(h, protos)
    a = target("softmax_over_points")(logits + x["in.log_w"])
    ((logits * _cotangent(logits, 36)).sum() + (a * _cotangent(a, 37)).sum()).backward()
    return {
        "out.protos": protos.detach(),
        "out.logits": logits.detach(),
        "out.a": a.detach(),
        "grad.x": h.grad,
        **_param_grads(mod),
    }


for _dt, _dn in ((D, "f64"), (F, "f32")):
    _case("G-BLOCKS", f"routing_logits_{_dn}", _routing_build(_dt), _routing_run)


def _region_build(dtype, k, n_vecs, n_tens=0):
    def build():
        g = _gen(38)
        x = _block_inputs(g, 300, 32, dtype, k=k, n_vecs=n_vecs, n_tens=n_tens)
        del x["in.h"], x["in.log_w"]
        x["in.logits_pre"] = torch.randn(1, 300, 16, generator=g, dtype=D).to(dtype)
        x["in.z_pos"] = torch.randn(1, 16, 3, generator=g, dtype=D).to(dtype)
        x["in.m_s"] = _directions(g, 1, 16, 3).to(dtype)
        lin = torch.nn.Linear(target("geo_width")(k, n_vecs, n_tens), 1)
        x.update({k_: v.to(dtype) for k_, v in _params(lin, 39).items()})
        return x

    return build


def _region_run(k, n_vecs, n_tens=0):
    def run(inputs, device, variant):
        """The geometry region (bias, point->slice mix, pooled invariants) and its
        gradients; the invariants themselves; the fused kernel on CUDA."""
        x = _to(inputs, device)
        dtype = x["in.r"].dtype
        lin = _load(
            torch.nn.Linear(target("geo_width")(k, n_vecs, n_tens), 1), x, device, dtype
        )
        names = ("in.logits_pre", "in.r", "in.n_hat", "in.z_pos", "in.m_s")
        lv = {n: x[n].clone().requires_grad_() for n in names}
        args = [lv["in.r"], lv["in.n_hat"], x["in.g_hat"], lv["in.z_pos"], lv["in.m_s"]]
        if variant == "auto":
            vecs = target("fused_vectors")(x["in.g_hat"], x.get("in.vecs"))
            res = target("fused_geo_region")(
                lin,
                lv["in.logits_pre"],
                lv["in.r"],
                lv["in.n_hat"],
                vecs,
                lv["in.z_pos"],
                lv["in.m_s"],
                1e-12,
                n_global=k,
            )
        else:
            res = target("geo_region")(
                lin,
                lv["in.logits_pre"],
                *args,
                1e-12,
                x.get("in.vecs"),
                x.get("in.tens"),
            )
        sum((t * _cotangent(t, 40 + i)).sum() for i, t in enumerate(res)).backward()
        out = {f"out.{n}": t.detach() for n, t in zip(("bias", "mix", "pooled"), res)}
        out.update({f"grad.{n[3:]}": t.grad for n, t in lv.items()})
        out.update(_param_grads(lin))
        with torch.no_grad():
            out["out.invariants"] = target("relational_invariants")(
                *args, 1e-12, x.get("in.vecs"), x.get("in.tens")
            )
        return out

    return run


for _dt, _dn in ((D, "f64"), (F, "f32")):
    _case(
        "G-BLOCKS",
        f"geo_region_k2_v1_{_dn}",
        _region_build(_dt, 2, 1),
        _region_run(2, 1),
        variants={"cpu": ("reference",), "cuda": ("auto", "reference")},
        requires=("triton",),
    )
_case(
    "G-BLOCKS",
    "geo_region_tensors_f64",
    _region_build(D, 1, 1, 1),
    _region_run(1, 1, 1),
    variants={
        "cpu": ("reference",),
        "cuda": ("reference",),
    },  # the fused region takes no rank-2 inputs
)


# --------------------------------------------------------------------------- G-HEADS


def _no_inputs():
    return {}


def _head_tables_run(inputs, device, variant):
    """The isotropic tensors of every rank, symmetry and parity in 2 and 3 dimensions,
    the Levi-Civita symbols, and the coefficient index of a set of output layouts."""
    from physicsnemo.mesh.fields import RankSpec

    out = {f"out.levi_civita_{d}": target("levi_civita")(d) for d in (2, 3)}
    for d in (2, 3):
        for rank in range(5):
            for sym in (False, True):
                for parity in ("even", "odd"):
                    iso = target("isotropic_tensors")(rank, sym, parity, d)
                    if iso:
                        out[
                            f"out.iso.{d}d.r{rank}.{'sym' if sym else 'gen'}.{parity}"
                        ] = torch.stack(iso)
    for d, rank, sym, parity, m, p in _LAYOUTS:
        lay = target("OutputLayout")(
            RankSpec(rank, symmetric=sym, parity=parity), m, p, d
        )
        key = f"out.layout.{d}d.r{rank}.{'sym' if sym else 'gen'}.{parity}.m{m}p{p}"
        out[key + ".index"] = lay.index
        out[key + ".sizes"] = torch.tensor([lay.n_vec, lay.n_free, lay.n_coeffs])
    return out


#: (dim, rank, symmetric, parity, true vectors, pseudovectors)
_LAYOUTS = (
    (3, 1, False, "even", 3, 3),
    (3, 1, False, "odd", 3, 3),
    (3, 2, True, "even", 3, 3),
    (3, 2, False, "odd", 2, 1),
    (3, 3, False, "odd", 2, 1),
    (3, 4, True, "even", 2, 1),
    (2, 1, False, "odd", 2, 2),
    (2, 2, False, "odd", 2, 2),
    (2, 2, True, "even", 2, 2),
)
_case("G-HEADS", "tables", _no_inputs, _head_tables_run)


def _assemble_build():
    g = _gen(50)
    x = {}
    for d in (2, 3):
        x[f"in.true_{d}"] = torch.randn(2, 40, 3, d, generator=g, dtype=D)
        x[f"in.pseudo_{d}"] = torch.randn(2, 40, 3, d, generator=g, dtype=D)
    x["in.r_hat"] = _directions(g, 40, 3)
    x["in.r_hat"][:4] = 0.0  # zero where r_hat is zero
    x["in.v"] = torch.randn(40, 3, generator=g, dtype=D)
    x["in.coeffs"] = torch.randn(2, 40, 64, generator=g, dtype=D)
    return x


def _assemble_run(inputs, device, variant):
    """Every layout of :data:`_LAYOUTS` assembled from random coefficients and basis
    vectors (with the gradients in both), the pseudovectors of 2 and 3 dimensions,
    and the polar complement."""
    from physicsnemo.mesh.fields import RankSpec

    x = _to(inputs, device)
    out = {}
    for d, rank, sym, parity, m, p in _LAYOUTS:
        lay = target("OutputLayout")(
            RankSpec(rank, symmetric=sym, parity=parity), m, p, d
        )
        c = x["in.coeffs"][..., : lay.n_coeffs].clone().requires_grad_()
        tv = x[f"in.true_{d}"][:, :, :m].clone().requires_grad_()
        pv = x[f"in.pseudo_{d}"][:, :, :p].clone().requires_grad_()
        y = target("assemble_output")(c, lay, tv, pv, lay.index.to(device))
        (y * _cotangent(y, 51)).sum().backward()
        key = f"{d}d.r{rank}.{'sym' if sym else 'gen'}.{parity}.m{m}p{p}"
        out[f"out.{key}"] = y.detach()
        out[f"grad.{key}.coeffs"], out[f"grad.{key}.true"] = c.grad, tv.grad
        if pv.grad is not None:
            out[f"grad.{key}.pseudo"] = pv.grad
    for d in (2, 3):
        prim = list(x[f"in.true_{d}"].unbind(2))
        out[f"out.pseudovectors_{d}"] = torch.stack(target("pseudovectors")(prim, d), 2)
    out["out.polar_complement"] = target("polar_complement")(x["in.r_hat"], x["in.v"])
    return out


_case("G-HEADS", "assemble", _assemble_build, _assemble_run)


# --------------------------------------------------------------------------- G-MODEL, G-PARTS
### The model end to end on small configurations (hidden 32, two layers, eight slices):
### every output, and every parameter's gradient under a fixed cotangent (G-MODEL); and
### the intermediate values of the same forward pass, recorded by spies on the model's
### parts, so that a model-level difference is located at once (G-PARTS). Every
### parameter is drawn at random (std 0.1; ISLA-DTT's boundary-read projections, zero
### at construction, from ``_meshes.live``), the inputs from correctly rounded
### operations or the mesh primitives (hashed), in evaluation mode (the draw's fixed
### start).

_MODEL_IO = dict(
    output_schema={"p": {"rank": 0}, "v": {"rank": 1}},
    boundary_schemas={"wall": {}},
    global_schema={"g": {"rank": 1}},
)
_MODEL_SMALL = dict(hidden=32, n_layers=2, n_slices=8)
#: A surface model with every kind of input and output: boundary fields of rank 0, 1
#: and 2 on two conditions, a rank-1 field that defines the velocity scale (so the
#: derived direction g joins the global one), a rank-0 global that defines a density,
#: a dimensionless global, outputs of rank 0, 1 and 2 (symmetric and general), odd
#: outputs of rank 1 and 2, units, an output constant, and output_stats.
_RICH = dict(
    output_schema={
        "p": {"rank": 0, "units": {"U": 2, "rho": 1}},
        "v": {"rank": 1, "units": {"U": 1}},
        "tau": {"rank": 2, "symmetric": True, "units": {"U": 2, "rho": 1}},
        "grad_v": {"rank": 2, "units": {"U": 1, "L": -1}},
        "omega": {"rank": 1, "parity": "odd", "units": {"U": 1, "L": -1}},
        "spin": {"rank": 2, "parity": "odd"},
        "nut": {"rank": 0, "units": {"U": 1, "L": 1}, "scale": 1e-3},
    },
    boundary_schemas={
        "wall": {
            "trace": {"rank": 0},
            "traction": {"rank": 1, "units": {"U": 1}, "defines": "U"},
            "stress": {"rank": 2, "symmetric": True, "units": {"U": 2, "rho": 1}},
        },
        "outlet": {"p_out": {"rank": 0, "units": {"U": 2, "rho": 1}}},
    },
    global_schema={
        "g": {"rank": 1},
        "rho": {"rank": 0, "defines": "rho"},
        "re": {"rank": 0},
    },
    output_stats={"p": {"mean": 0.3, "std": 2.0}, "omega": {"mean": 0.0, "std": 1.5}},
)


@dataclass(frozen=True)
class _ModelSpec:
    """One model case: the constructor's arguments, the boundary meshes (condition name
    -> maker of ``(points, cells)`` in float64), the surface task's prediction
    boundaries or the interior task's number of prediction points, and the dtype."""

    cfg: dict
    meshes: dict
    dtype: torch.dtype = D
    prediction_boundaries: tuple = ()
    n_queries: int = 0
    d: int = 3
    seed: int = 60


def _soup_cells(n, d, seed, offset=0.0):
    """``n`` random simplices (a triangle soup in 3-D, segments in 2-D) about an
    anisotropic cloud: each vertex its point plus a Gaussian offset of 0.15."""
    pts, _, _, _ = _cloud(n, d=d, seed=seed)
    corners = 0.15 * torch.randn(n, d, d, generator=_gen(seed + 1), dtype=D)
    verts = (pts[:, None] + offset + corners).reshape(-1, d)
    return verts, torch.arange(n * d).reshape(n, d)


def _sphere_cells(subdivisions, radius=1.0):
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    m = sphere_icosahedral.load(radius=radius, subdivisions=subdivisions)
    return _oriented(m.points.double(), m.cells, lambda x: x)


def _shell_cells(levels, half):
    """The cube surface ``[-half, half]^3`` (``half`` a power of two), outward."""
    m = _cube(levels)
    return m.points * (2 * half), m.cells


def _circle_cells(n=48):
    from physicsnemo.mesh.primitives.curves import circle_2d

    m = circle_2d.load(n_points=n)
    return _oriented(m.points.double(), m.cells, lambda x: x)


def _model_build(spec: _ModelSpec):
    def build():
        from test.experimental.models.isla._meshes import live

        g = _gen(spec.seed)
        x = {}
        model = target("ISLA")(**spec.cfg)
        schemas = model.boundary_schemas
        for name, maker in spec.meshes.items():
            p, c = maker()
            x[f"in.{name}.points"], x[f"in.{name}.cells"] = p, c
            for f, fs in schemas[name].items():
                t = torch.randn(c.shape[0], *[spec.d] * fs.rank, generator=g, dtype=D)
                x[f"in.{name}.{f}"] = t + t.transpose(-1, -2) if fs.symmetric else t
        for f, fs in model.global_schema.items():
            if fs.rank == 1:
                x[f"in.global.{f}"] = _directions(g, spec.d)
            else:
                x[f"in.global.{f}"] = 1.0 + torch.rand((), generator=g, dtype=D)
        if spec.n_queries:
            x["in.queries"] = _near_wall(
                spec.d, spec.n_queries, spec.seed + 1, 0.8, 2.0
            )
        params = _params(model, spec.seed + 2, std=0.1)
        if model.kernel_reads is not None:
            live(model, spec.seed + 3)
            for n, p in model.named_parameters():
                if n.startswith("kernel_reads.") and n.endswith(".out.weight"):
                    params[f"param.{n}"] = p.detach().double()
        ### Every parameter in one flat input (one hash in the golden).
        x["in.params"] = torch.cat([v.reshape(-1) for v in params.values()])
        return x

    return build


def _model_kernels(model, variant: str):
    """The reference of every piece (CPU, and the CUDA reference variant), or every
    piece's fast path where it can run (``auto``)."""
    ref = variant == "reference"
    model.geo_kernel = "eager" if ref else "auto"
    model.tree_read_kernel = "reference" if ref else "auto"
    model.geometry_kernel = "reference" if ref else "auto"


def _model_setup(spec: _ModelSpec, inputs, device, variant):
    """The loaded model and the arguments of its forward pass."""
    x = _to(inputs, device)
    dt = spec.dtype
    model = target("ISLA")(**spec.cfg).to(device, dt).eval()
    with torch.no_grad():
        flat, i = x["in.params"], 0
        for p in model.parameters():
            p.copy_(flat[i : i + p.numel()].view_as(p))
            i += p.numel()
    _model_kernels(model, variant)
    return model, model_inputs(spec, x, device, model)


def _nested(fields: dict) -> dict:
    """``{"a.b": t}`` as ``{"a": {"b": t}}`` (the schemas' dotted names as groups)."""
    out = {}
    for name, t in fields.items():
        *groups, leaf = name.split(".")
        node = out
        for g in groups:
            node = node.setdefault(g, {})
        node[leaf] = t
    return out


def model_inputs(spec: _ModelSpec, inputs: dict, device, model) -> dict:
    """The keyword arguments of the forward pass of ``model`` (an ISLA of the case's
    configuration) on a model case's inputs (``build()``), on ``device`` in the
    case's dtype."""
    from physicsnemo.mesh import Mesh

    x, dt, schemas = _to(inputs, device), spec.dtype, model
    meshes = {}
    for name in spec.meshes:
        data = {f: x[f"in.{name}.{f}"].to(dt) for f in schemas.boundary_schemas[name]}
        meshes[name] = Mesh(
            points=x[f"in.{name}.points"].to(dt),
            cells=x[f"in.{name}.cells"],
            cell_data=_nested(data),
        )
    kw = dict(
        boundary_meshes=meshes,
        global_data=_nested(
            {f: x[f"in.global.{f}"].to(dt) for f in schemas.global_schema}
        ),
    )
    if spec.n_queries:
        kw["prediction_points"] = x["in.queries"].to(dt)
    else:
        kw["prediction_boundaries"] = list(spec.prediction_boundaries)
    return kw


def _flat_outputs(mesh) -> dict:
    from test.experimental.models.isla._meshes import outputs

    return outputs(mesh)


def _cotangent_loss(fields: dict, seed: int) -> torch.Tensor:
    return sum(
        (t * _cotangent(t, seed + i)).sum()
        for i, (_, t) in enumerate(sorted(fields.items()))
    )


#: Parameters whose gradient is zero in exact arithmetic (constants that cancel in a
#: softmax: each slice block's geo_logit bias, a constant over the slices, and the
#: boundary read's basis MLP's last bias, a per-head constant on every pair's logit),
#: so that what is computed is roundoff alone.
_MODEL_ZERO_GRADS = tuple(f"blocks.{i}.geo_logit.bias" for i in range(2)) + (
    "kernel_basis.net.4.bias",
)


def _model_run(spec: _ModelSpec):
    def run(inputs, device, variant):
        """The forward pass's outputs, and every parameter's gradient of a fixed
        cotangent on them."""
        model, kw = _model_setup(spec, inputs, device, variant)
        fields = _flat_outputs(model(**kw))
        _cotangent_loss(fields, 70).backward()
        out = {f"out.{k}": t.detach() for k, t in fields.items()}
        out.update(_param_grads(model, zero=_MODEL_ZERO_GRADS))
        return out

    return run


_CHUNKS = ((0, 20), (20, 45), (45, 60))


def _chunks_run(spec: _ModelSpec):
    def run(inputs, device, variant):
        """``query_chunks`` over three ranges of the points: each range's outputs, the
        pair counts, and every parameter's gradient after the chunks' backward passes
        (a fixed cotangent per range) and ``chunks.backward()``."""
        model, kw = _model_setup(spec, inputs, device, variant)
        chunks = model.query_chunks(**kw)
        out = {"out.pair_counts": chunks.pair_counts}
        for i, (a, b) in enumerate(_CHUNKS):
            fields = _flat_outputs(chunks(a, b))
            _cotangent_loss(fields, 80 + 10 * i).backward()
            out.update({f"out.range{i}.{k}": t.detach() for k, t in fields.items()})
        chunks.backward()
        out.update(_param_grads(model, zero=_MODEL_ZERO_GRADS))
        return out

    return run


#: The functions of the model module whose results G-PARTS records, by the names the
#: model calls them by (wherever they are defined).
_PART_FUNCTIONS = (
    "cell_second_moments",
    "rms_pairwise_distance",
    "soft_log_length",
    "rms_moments",
    "scale_factor",
    "sample_boundary_cells",
    "_pseudovectors",
    "_assemble_output",
)


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, torch.Tensor):
        out[prefix] = value.detach().clone()
    elif isinstance(value, (tuple, list)):
        for j, v in enumerate(value):
            _flatten(f"{prefix}.{j}", v, out)


def _parts_run(spec: _ModelSpec):
    def run(inputs, device, variant):
        """The forward pass's intermediate values, call by call: each slice block's
        output and passive read, the boundary reads' updates, the head frame's routing
        (prototypes and logits), the prediction points' input layer, and the results of
        :data:`_PART_FUNCTIONS` (the length unit and the cells' second moments, the
        soft log length, the scales' moments and factors, the draw, the pseudovectors
        and each assembled output)."""
        model, kw = _model_setup(spec, inputs, device, variant)
        mod = target("model_module")
        seen, counts = {}, {}

        def record(name, value):
            i = counts[name] = counts.get(name, -1) + 1
            _flatten(f"out.{name}.{i}", value, seen)

        def spy(name, fn):
            def call(*a, **k):
                y = fn(*a, **k)
                record(name, y)
                return y

            return call

        saved = {n: getattr(mod, n) for n in _PART_FUNCTIONS}
        hooks = []
        for i, block in enumerate(model.blocks):
            hooks.append(
                block.register_forward_hook(lambda m, a, y, i=i: record(f"block{i}", y))
            )
            block.read = spy(f"block{i}.read", block.read)
        if model.frame_assign is not None:
            fa = model.frame_assign
            fa.prototypes = spy("frame.prototypes", fa.prototypes)
            fa.logits = spy("frame.logits", fa.logits)
        if model.qt is not None:
            hooks.append(
                model.qt.seed.register_forward_hook(
                    lambda m, a, y: record("qt.seed", y)
                )
            )
        for i, read in enumerate(model.kernel_reads or ()):
            hooks.append(
                read.register_forward_hook(lambda m, a, y, i=i: record(f"read{i}", y))
            )
        for n, f in saved.items():
            setattr(mod, n, spy(n, f))
        try:
            with torch.no_grad():
                model(**kw)
        finally:
            for n, f in saved.items():
                setattr(mod, n, f)
            for h in hooks:
                h.remove()
        return seen

    return run


_MODEL_SPECS = {
    "surface": lambda dt: _ModelSpec(
        dict(**_MODEL_SMALL, **_MODEL_IO),
        {"wall": lambda: _soup_cells(150, 3, 61)},
        dt,
        prediction_boundaries=("wall",),
    ),
    "rich_surface": lambda dt: _ModelSpec(
        dict(**_MODEL_SMALL, **_RICH),
        {
            "wall": lambda: _soup_cells(120, 3, 62),
            "outlet": lambda: _soup_cells(40, 3, 63, offset=6.0),
        },
        dt,
        prediction_boundaries=("wall",),
    ),
    "dtt": lambda dt: _ModelSpec(
        dict(**_MODEL_SMALL, **_MODEL_IO, interior=True),
        {"wall": lambda: _sphere_cells(2)},
        dt,
        n_queries=60,
    ),
    "dtt_2d": lambda dt: _ModelSpec(
        dict(**_MODEL_SMALL, **_MODEL_IO, interior=True, n_spatial_dims=2),
        {"wall": _circle_cells},
        dt,
        n_queries=60,
        d=2,
    ),
    ### About 2,000 cells over two conditions, 512 tokens drawn (at the fixed start).
    "dtt_draw": lambda dt: _ModelSpec(
        dict(
            **_MODEL_SMALL,
            output_schema=_MODEL_IO["output_schema"],
            global_schema=_MODEL_IO["global_schema"],
            boundary_schemas={"wall": {}, "farfield": {}},
            interior=True,
            n_boundary_tokens=512,
        ),
        {"wall": lambda: _sphere_cells(3), "farfield": lambda: _shell_cells(3, 4.0)},
        dt,
        n_queries=60,
    ),
}
_MODEL_DTYPES = {
    "surface": ((D, "f64"), (F, "f32")),
    "rich_surface": ((D, "f64"), (F, "f32")),
    "dtt": ((D, "f64"), (F, "f32")),
    "dtt_2d": ((D, "f64"),),
    "dtt_draw": ((F, "f32"),),
}
_MODEL_VARIANTS = {"cpu": ("reference",), "cuda": ("auto", "reference")}
#: Four elements of each parameter's gradient in the committed golden (beside its
#: summaries over every element), to keep the models' goldens within their budget.
_MODEL_SAMPLES = {"grad.param.*": 4}
#: The routings' gains and value projections (each slice block's ``assign`` and the
#: head frame's) reach the loss through softmaxes over the slices whose terms nearly
#: cancel: their gradients are 1e-5 to 1e-10 of the largest, and carry the rounding of
#: the terms. Measured between one, three and four threads (largest relative to their
#: own largest element): 1.2e-11 in float64, 1.1e-2 in float32. Bounds: 1e-9 and
#: 1e-1 (a sign or a term lost, not roundoff, in float32).
_ROUTING_GRADS = ("grad.param.*assign.gain", "grad.param.*assign.v_proj.*")
_MODEL_TOL = {
    D: dict.fromkeys(_ROUTING_GRADS, 1e-9),
    F: dict.fromkeys(_ROUTING_GRADS, 1e-1),
}
#: On CUDA the reference read's node means, and the exact geometry's vertex
#: gradient, accumulate by float atomics: ISLA-DTT's outputs and gradients are
#: judged against their run-to-run family there.
_DTT_FAMILY = ("out.*", "grad.*")

for _name, _make in _MODEL_SPECS.items():
    for _dt, _dn in _MODEL_DTYPES[_name]:
        _spec = _make(_dt)
        _dtt = _spec.n_queries > 0
        _kw = dict(
            variants=_MODEL_VARIANTS,
            samples=_MODEL_SAMPLES,
            tol=_MODEL_TOL[_dt],
            family=_DTT_FAMILY if _dtt else (),
            requires=("warp", "triton"),
        )
        _case(
            "G-MODEL",
            f"model_{_name}_{_dn}",
            _model_build(_spec),
            _model_run(_spec),
            **_kw,
        )
        if _name == "dtt":
            _case(
                "G-MODEL",
                f"query_chunks_{_dn}",
                _model_build(_spec),
                _chunks_run(_spec),
                **_kw,
            )
        _draw = ("out.sample_boundary_cells.*.0",) if _name == "dtt_draw" else ()
        _weights = ("out.sample_boundary_cells.*.1",) if _name == "dtt_draw" else ()
        _case(
            "G-PARTS",
            f"parts_{_name}_{_dn}",
            _model_build(_spec),
            _parts_run(_spec),
            kept=_draw,
            weights=_weights,
            **_kw,
        )


GROUPS = tuple(dict.fromkeys(c.group for c in CASES))


def select(groups=None, names=None) -> list[Case]:
    """The cases of the named groups (all by default), optionally only those whose
    name matches one of the fnmatch patterns ``names``."""
    out = [c for c in CASES if groups is None or c.group in groups]
    if names:
        out = [c for c in out if any(fnmatch.fnmatchcase(c.name, p) for p in names)]
    return out


def available(case: Case, device: str) -> str | None:
    """Why ``case`` cannot run on ``device`` here, or None."""
    if device == "cuda" and not torch.cuda.is_available():
        return "CUDA is not available"
    if device == "cuda" and "triton" in case.requires:
        from physicsnemo.core.version_check import is_package_available

        if not is_package_available("triton"):
            return "Triton is not installed"
    return None


def run_case(case: Case, device: str, variant: str, inputs: dict | None = None) -> dict:
    """The case's tensors on ``device`` under ``variant``: its inputs (``in.*``, the
    given ones or freshly built) and every output and gradient, detached, on the CPU."""
    inputs = case.build() if inputs is None else inputs
    out = case.run(inputs, device, variant)
    tensors = {**inputs, **out}
    return {k: v.detach().cpu() for k, v in tensors.items()}
