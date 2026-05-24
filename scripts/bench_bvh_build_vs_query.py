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

"""Throwaway BVH-build-vs-query throughput benchmark for the M1 milestone.

Measures the relative cost of (a) ``wp.Mesh`` construction, (b) BVH
distance queries at three query sizes, and (c) the end-to-end
``signed_distance_field`` call (which rebuilds the BVH every call) on the
fixed analytic mesh set (sphere / cube / torus). Optionally cross-checks
against FCPW build + 100k-query when the ``fcpw`` Python binding is
available.

Implements deliverable 5 of ``geopt-datagen-round1-plan.md`` §3.2 and
provides the data table for exit criterion G4 in §3.3:

    > BVH build + 10k-query throughput on a 50k-tri ShapeNet mesh ≥ 100×
    > the per-CPU-core throughput we measure for the same op single-thread
    > on the same mesh (the parent plan's §3.4 conservative target).
    > If build-only dominates, downgrade to **measured** speedup with a
    > note.

This script is intentionally throwaway — not part of the upstream
benchmark suite (``benchmarks/`` lives next to ``physicsnemo/``); it
emits a markdown table to stdout and saves it to
``reports/m1-bvh-build-vs-query.md``.

Usage
-----
::

    python scripts/bench_bvh_build_vs_query.py --device cuda
    python scripts/bench_bvh_build_vs_query.py --device cpu --n-trials 10

If CUDA is unavailable, the script falls back to CPU with a warning. The
absolute numbers on CPU are not the H100 target numbers in the plan; the
*ratios* (build vs query, query vs end-to-end) and *trends* are what
this script measures.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp

from physicsnemo.nn.functional import signed_distance_field
from test.experimental.pnm_pretraining.conftest import (
    TriangleMesh,
    make_cube,
    make_sphere,
    make_torus,
)

wp.config.quiet = True


# ---------------------------------------------------------------------------
# Timing harness.
# ---------------------------------------------------------------------------


def time_op(fn, n_warmup: int = 3, n_trials: int = 5) -> tuple[float, float]:
    """Time ``fn()`` and return ``(median_ms, iqr_ms)``.

    Runs ``fn`` ``n_warmup`` times to absorb first-call costs (kernel
    compile, allocator warmup, BVH refit caches, etc.), then ``n_trials``
    timed iterations using ``time.perf_counter``. CUDA syncs are issued
    before and after each timed run when CUDA is available, so the
    reported time covers the device-side work too.
    """
    cuda_available = torch.cuda.is_available()

    for _ in range(n_warmup):
        fn()
        if cuda_available:
            torch.cuda.synchronize()

    samples = []
    for _ in range(n_trials):
        if cuda_available:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda_available:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e3)

    arr = np.asarray(samples)
    median = float(np.median(arr))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    return median, iqr


def _fmt(median: float, iqr: float) -> str:
    """Format ``(median_ms, iqr_ms)`` as ``median ± iqr/2``."""
    return f"{median:.3f} ± {iqr / 2:.3f}"


# ---------------------------------------------------------------------------
# Warp kernel for the query-only timing path.
# ---------------------------------------------------------------------------


@wp.kernel
def _query_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    sdf: wp.array(dtype=wp.float32),
):
    """Minimal SDF kernel; mirrors ``physicsnemo.nn.functional.geometry.sdf``.

    Uses ``mesh_query_point_sign_winding_number`` and writes only the
    signed distance (no closest-point output) to keep the kernel cheap
    and isolate BVH-traversal cost.
    """
    tid = wp.tid()
    res = wp.mesh_query_point_sign_winding_number(mesh_id, points[tid], max_dist)
    mesh = wp.mesh_get(mesh_id)
    p0 = mesh.points[mesh.indices[3 * res.face + 0]]
    p1 = mesh.points[mesh.indices[3 * res.face + 1]]
    p2 = mesh.points[mesh.indices[3 * res.face + 2]]
    p_closest = res.u * p0 + res.v * p1 + (1.0 - res.u - res.v) * p2
    sdf[tid] = res.sign * wp.abs(wp.length(points[tid] - p_closest))


# ---------------------------------------------------------------------------
# Per-mesh benchmark.
# ---------------------------------------------------------------------------


def _build_wp_mesh(
    mesh: TriangleMesh, device: torch.device
) -> tuple[wp.Mesh, torch.Tensor, torch.Tensor]:
    """Build a ``wp.Mesh`` from a ``TriangleMesh`` on the requested device."""
    v_t = torch.as_tensor(
        mesh.vertices, dtype=torch.float32, device=device
    ).contiguous()
    i_t = (
        torch.as_tensor(mesh.indices, dtype=torch.int32, device=device)
        .reshape(-1)
        .contiguous()
    )
    wp_v = wp.from_torch(v_t, dtype=wp.vec3)
    wp_i = wp.from_torch(i_t, dtype=wp.int32)
    wp_mesh = wp.Mesh(points=wp_v, indices=wp_i, support_winding_number=True)
    return wp_mesh, v_t, i_t


def bench_mesh(
    mesh: TriangleMesh,
    device: torch.device,
    query_sizes: tuple[int, ...],
    n_warmup: int,
    n_trials: int,
    fcpw_available: bool,
) -> dict[str, str | int]:
    """Run all timing variants for one mesh; return a row dict for the table."""
    wp_device_str = "cuda" if device.type == "cuda" else "cpu"

    # Materialize torch tensors once; reuse for every timed iteration.
    v_t = torch.as_tensor(
        mesh.vertices, dtype=torch.float32, device=device
    ).contiguous()
    i_t = (
        torch.as_tensor(mesh.indices, dtype=torch.int32, device=device)
        .reshape(-1)
        .contiguous()
    )
    wp_v = wp.from_torch(v_t, dtype=wp.vec3)
    wp_i = wp.from_torch(i_t, dtype=wp.int32)

    # --- Build timing ---------------------------------------------------
    def _build():
        # Hold a reference so the BVH isn't GC'd before the next iteration.
        nonlocal _last_mesh
        _last_mesh = wp.Mesh(points=wp_v, indices=wp_i, support_winding_number=True)

    _last_mesh: wp.Mesh | None = None
    build_med, build_iqr = time_op(_build, n_warmup=n_warmup, n_trials=n_trials)

    # Persistent BVH for the query timings.
    bvh = wp.Mesh(points=wp_v, indices=wp_i, support_winding_number=True)

    query_results: dict[int, tuple[float, float]] = {}
    for n in query_sizes:
        rng = np.random.default_rng(seed=12345 + n)
        pts = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)
        pts_t = torch.as_tensor(pts, dtype=torch.float32, device=device)
        sdf_t = torch.zeros(n, dtype=torch.float32, device=device)
        wp_pts = wp.from_torch(pts_t, dtype=wp.vec3)
        wp_sdf = wp.from_torch(sdf_t, dtype=wp.float32)

        def _query():
            wp.launch(
                kernel=_query_kernel,
                dim=n,
                inputs=[bvh.id, wp_pts, np.float32(1e8), wp_sdf],
                device=wp_device_str,
            )

        med, iqr = time_op(_query, n_warmup=n_warmup, n_trials=n_trials)
        query_results[n] = (med, iqr)

    # --- End-to-end signed_distance_field timing at the largest query
    # size (matches G4 table semantics: build + query mixed). ---------
    n_e2e = max(query_sizes)
    rng = np.random.default_rng(seed=99999 + n_e2e)
    pts_t_e2e = torch.as_tensor(
        rng.uniform(-2.0, 2.0, size=(n_e2e, 3)).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )

    def _e2e():
        signed_distance_field(
            mesh_vertices=v_t,
            mesh_indices=i_t,
            input_points=pts_t_e2e,
            use_sign_winding_number=True,
        )

    e2e_med, e2e_iqr = time_op(_e2e, n_warmup=n_warmup, n_trials=n_trials)

    # --- Optional FCPW timing -----------------------------------------
    fcpw_build_str = "n/a"
    fcpw_query_str = "n/a"
    if fcpw_available:
        try:
            import fcpw  # noqa: F401

            v_np = mesh.vertices.astype(np.float32)
            i_np = mesh.indices.astype(np.int32)

            def _fcpw_build():
                scene = fcpw.scene_3D()
                scene.set_object_count(1)
                scene.set_object_vertices(v_np, 0)
                scene.set_object_triangles(i_np, 0)
                scene.build(fcpw.aggregate_type.bvh_surface_area, True)
                return scene

            fcpw_b_med, fcpw_b_iqr = time_op(
                _fcpw_build, n_warmup=n_warmup, n_trials=n_trials
            )
            fcpw_build_str = _fmt(fcpw_b_med, fcpw_b_iqr)

            scene = _fcpw_build()
            n_fcpw = max(query_sizes)
            pts_fcpw = (
                np.random.default_rng(seed=7)
                .uniform(-2.0, 2.0, size=(n_fcpw, 3))
                .astype(np.float32)
            )

            def _fcpw_query():
                interactions = fcpw.interaction_3D_list()
                scene.find_closest_points(pts_fcpw, interactions)

            fcpw_q_med, fcpw_q_iqr = time_op(
                _fcpw_query, n_warmup=n_warmup, n_trials=n_trials
            )
            fcpw_query_str = _fmt(fcpw_q_med, fcpw_q_iqr)
        except Exception as exc:  # pragma: no cover — best-effort
            fcpw_build_str = f"error: {exc}"
            fcpw_query_str = "n/a"

    row = {
        "mesh": mesh.name,
        "n_tris": int(mesh.indices.shape[0]),
        "build": _fmt(build_med, build_iqr),
        "q1k": _fmt(*query_results[1_000]) if 1_000 in query_results else "n/a",
        "q10k": _fmt(*query_results[10_000]) if 10_000 in query_results else "n/a",
        "q100k": _fmt(*query_results[100_000]) if 100_000 in query_results else "n/a",
        "e2e": _fmt(e2e_med, e2e_iqr),
        "fcpw_build": fcpw_build_str,
        "fcpw_query": fcpw_query_str,
    }
    return row


# ---------------------------------------------------------------------------
# Output formatting.
# ---------------------------------------------------------------------------


def render_table(rows: list[dict[str, str | int]], device: str, note: str) -> str:
    """Render the per-mesh timing rows as a markdown report."""
    lines = []
    lines.append("# M1 — BVH build vs query throughput")
    lines.append("")
    lines.append(
        "Round-1 plan: `geopt-datagen-round1-plan.md` §3.2 (deliverable 5), "
        "exit criterion G4."
    )
    lines.append(f"Device: `{device}`")
    if note:
        lines.append("")
        lines.append(f"> {note}")
    lines.append("")
    lines.append(
        "| mesh | n_tris | wp.Mesh build (ms) | warp query 1k (ms) | "
        "warp query 10k (ms) | warp query 100k (ms) | sdf end-to-end 100k (ms) | "
        "fcpw build (ms) | fcpw query 100k (ms) |"
    )
    lines.append(
        "|------|--------|--------------------|--------------------|"
        "----------------------|----------------------|--------------------------|"
        "------------------|----------------------|"
    )
    lines.extend(
        f"| {r['mesh']} | {r['n_tris']} | {r['build']} | {r['q1k']} | "
        f"{r['q10k']} | {r['q100k']} | {r['e2e']} | {r['fcpw_build']} | "
        f"{r['fcpw_query']} |"
        for r in rows
    )
    lines.append("")
    lines.append("All cells: `median ± IQR/2` over the configured trials.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint: parse args, run the per-mesh benchmark, emit a report.

    Resolves the requested device (falling back to CPU with a warning if
    CUDA was requested but is unavailable), detects FCPW availability,
    and runs ``bench_mesh`` on each of the three analytic meshes
    (sphere, cube, torus). The resulting markdown table is printed to
    stdout and saved to the path given by ``--out``
    (default ``reports/m1-bvh-build-vs-query.md``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run the benchmark on. Falls back to cpu with a "
        "warning if cuda is unavailable.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of timed iterations per measurement.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/m1-bvh-build-vs-query.md"),
        help="Path to write the markdown report.",
    )
    args = parser.parse_args()

    device_note = ""
    if args.device == "cuda" and not torch.cuda.is_available():
        device_note = (
            "CUDA was requested but is unavailable on this host; benchmark "
            "ran on CPU. Absolute numbers are NOT the H100 target numbers "
            "in the plan; the ratios (build vs query, query vs "
            "end-to-end) and trends are what this run measures."
        )
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Detect FCPW availability once.
    fcpw_available = False
    try:
        import fcpw  # noqa: F401

        fcpw_available = True
    except ImportError:
        device_note = (
            (device_note + " ") if device_note else ""
        ) + "FCPW unavailable (`fcpw` not installed); FCPW columns are `n/a`."

    wp.init()

    meshes = [make_sphere(), make_cube(), make_torus()]
    query_sizes = (1_000, 10_000, 100_000)

    rows = []
    for mesh in meshes:
        print(f"[bench] mesh={mesh.name} n_tris={mesh.indices.shape[0]} ...")
        row = bench_mesh(
            mesh=mesh,
            device=device,
            query_sizes=query_sizes,
            n_warmup=args.n_warmup,
            n_trials=args.n_trials,
            fcpw_available=fcpw_available,
        )
        rows.append(row)

    table = render_table(rows, device=str(device), note=device_note)
    print()
    print(table)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
