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

"""SDF parity tests against analytic ground truth, trimesh, and FCPW.

Validates ``physicsnemo.nn.functional.signed_distance_field`` (the existing
warp-backed ``SignedDistanceField`` ``FunctionSpec``) on the fixed analytic
mesh set defined in ``test/experimental/pnm_pretraining/conftest.py``
(sphere, cube, torus). Implements M1 deliverable 4 from
``geopt-datagen-round1-plan.md`` §3.2 and exit criteria G3 / G5 / I2 from
§3.3.

Test breakdown
--------------
- ``test_sdf_sphere_analytic_parity`` — exit criterion G3 on the sphere
  (closest point on the unit sphere is ``p / |p|``).
- ``test_sdf_cube_analytic_parity`` — analytic check on axis-aligned cube
  query points where the SDF and closest point are known in closed form.
- ``test_sdf_torus_trimesh_parity`` — G3 cross-check against
  ``trimesh.proximity.closest_point`` and ``signed_distance``.
- ``test_sdf_fcpw_parity`` — G5 cross-check against FCPW
  ``find_closest_points`` / ``contains`` (gated on FCPW availability).
- ``test_sdf_non_watertight_diagnostic`` — improvement I2 diagnostic on a
  cube with one triangle removed; documents that PhysicsNeMo's
  winding-number sign survives non-watertight inputs gracefully.

All Warp-backed tests are gated by ``@requires_module("warp")`` and run on
CPU by default (``signed_distance_field`` dispatches to whichever device
the input tensors live on; the conftest does not create CUDA fixtures).
Reproducibility: every test seeds NumPy via the ``rng`` fixture; PyTorch
and Warp seeds are inherited from the session-level autouse fixture in
``test/conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from physicsnemo.nn.functional import signed_distance_field
from test.conftest import requires_module
from test.experimental.pnm_pretraining.conftest import (
    TriangleMesh,
    make_cube,
    requires_fcpw,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rms(x: np.ndarray) -> float:
    """Root-mean-square of a numeric array (element-wise, then averaged)."""
    return float(np.sqrt(np.mean(np.square(x))))


def _run_sdf(
    mesh: TriangleMesh,
    query_points: np.ndarray,
    use_sign_winding_number: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``signed_distance_field`` on a mesh + numpy query points.

    Returns ``(sdf, hit_points)`` as numpy arrays for easy comparison with
    analytic / trimesh / FCPW references.
    """
    v, f = mesh.to_torch(device="cpu")
    q = torch.as_tensor(query_points, dtype=torch.float32, device="cpu")
    sdf, hit = signed_distance_field(
        mesh_vertices=v,
        mesh_indices=f,
        input_points=q,
        use_sign_winding_number=use_sign_winding_number,
    )
    return sdf.detach().cpu().numpy(), hit.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# (a) Sphere analytic parity — exit criterion G3.
# ---------------------------------------------------------------------------


@requires_module("warp")
def test_sdf_sphere_analytic_parity(
    sphere_mesh: TriangleMesh, rng: np.random.Generator
) -> None:
    """Closest point on a unit sphere is ``p / |p|`` (analytic).

    UV-tessellated spheres are not perfectly spherical: the conftest
    fixture is a 16-ring × 32-segment sphere whose surface deviates from
    the analytic unit sphere by up to ~5e-3 in the worst case
    (chord-error of an arc ≈ 0.5·(π/16)² ≈ 0.02). The tolerances here
    are sized for that fixed mesh; if conftest later resamples at higher
    resolution, tighten these. Sign agreement uses ``|p| > 1`` as the
    analytic inside/outside oracle.
    """
    n = 1000
    pts = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)

    sdf, hit = _run_sdf(sphere_mesh, pts, use_sign_winding_number=True)

    norms = np.linalg.norm(pts, axis=1)

    # (i) |sdf| ≈ ||p| - 1|, up to UV-sphere tessellation error.
    abs_sdf_err = np.abs(sdf) - np.abs(norms - 1.0)
    assert _rms(abs_sdf_err) < 1e-2, (
        f"sphere |sdf| RMS error {_rms(abs_sdf_err):.4e} ≥ 1e-2"
    )

    # (ii) sign agreement vs analytic outside test (|p| > 1).
    outside_analytic = norms > 1.0
    outside_sdf = sdf > 0.0
    sign_agreement = float(np.mean(outside_analytic == outside_sdf))
    assert sign_agreement > 0.99, f"sphere sign agreement {sign_agreement:.4f} ≤ 0.99"

    # (iii) closest-point RMS for query points strictly outside the sphere
    # (where the analytic projection p/|p| is well-defined). Bound is
    # set by the discrete mesh's tangential surface deviation (~2-3e-2
    # for a 16×32 UV sphere), not the BVH precision.
    outside = norms > 1.0
    expected_closest = pts[outside] / norms[outside, None]
    cp_err = hit[outside] - expected_closest
    assert _rms(cp_err) < 5e-2, f"sphere closest-point RMS {_rms(cp_err):.4e} ≥ 5e-2"


# ---------------------------------------------------------------------------
# (b) Cube analytic parity.
# ---------------------------------------------------------------------------


@requires_module("warp")
def test_sdf_cube_analytic_parity(
    cube_mesh: TriangleMesh, rng: np.random.Generator
) -> None:
    """Axis-aligned cube ``[-1, 1]^3``: closest point and SDF are analytic.

    Use exterior query points strictly outside the cube along axis-aligned
    or off-axis directions; for any such ``p``, the analytic closest point
    is ``clip(p, -1, 1)`` and the analytic SDF is ``||p - clip(p, -1, 1)||``.
    Tolerances tight (1e-5 RMS) because the cube mesh is exact.
    """
    n = 100
    # Sample in [1.5, 2.5]^3 quadrants to keep all queries strictly outside.
    pts = rng.uniform(1.5, 2.5, size=(n, 3)).astype(np.float32)
    # Randomly negate axes so queries spread across all 8 octants.
    signs = rng.choice([-1.0, 1.0], size=(n, 3)).astype(np.float32)
    pts = pts * signs

    sdf, hit = _run_sdf(cube_mesh, pts, use_sign_winding_number=True)

    expected_closest = np.clip(pts, -1.0, 1.0)
    expected_sdf = np.linalg.norm(pts - expected_closest, axis=1)

    sdf_err = sdf - expected_sdf
    cp_err = hit - expected_closest
    assert _rms(sdf_err) < 1e-5, f"cube SDF RMS {_rms(sdf_err):.4e} ≥ 1e-5"
    assert _rms(cp_err) < 1e-5, f"cube closest-point RMS {_rms(cp_err):.4e} ≥ 1e-5"


# ---------------------------------------------------------------------------
# (c) Trimesh cross-check on torus — exit criterion G3.
# ---------------------------------------------------------------------------


@requires_module("warp")
@requires_module("trimesh")
def test_sdf_torus_trimesh_parity(
    torus_mesh: TriangleMesh, rng: np.random.Generator
) -> None:
    """Cross-check ``signed_distance_field`` vs ``trimesh.proximity`` on torus.

    Closest-point RMS < 1e-4 over 1000 query points in ``[-3, 3]^3``;
    sign agreement vs ``trimesh.proximity.signed_distance`` > 99%.
    """
    import trimesh

    n = 1000
    pts = rng.uniform(-3.0, 3.0, size=(n, 3)).astype(np.float32)

    tm = trimesh.Trimesh(
        vertices=torus_mesh.vertices.astype(np.float64),
        faces=torus_mesh.indices.astype(np.int64),
        process=False,
    )
    tm_closest, tm_dist, _ = trimesh.proximity.closest_point(tm, pts)
    tm_signed = trimesh.proximity.signed_distance(tm, pts)

    sdf, hit = _run_sdf(torus_mesh, pts, use_sign_winding_number=True)

    cp_err = hit - tm_closest.astype(np.float32)
    assert _rms(cp_err) < 1e-4, (
        f"torus closest-point RMS vs trimesh {_rms(cp_err):.4e} ≥ 1e-4"
    )

    # trimesh.signed_distance uses inside-positive sign; we use
    # outside-positive. Compare on inside/outside as a boolean.
    sign_agreement = float(np.mean((sdf > 0.0) == (tm_signed < 0.0)))
    assert sign_agreement > 0.99, (
        f"torus sign agreement vs trimesh {sign_agreement:.4f} ≤ 0.99"
    )

    # Bonus: |sdf| RMS vs trimesh |signed_distance| (informational, not
    # gated; the absolute values are an extra sanity check beyond
    # closest-point).
    abs_err = np.abs(sdf) - np.abs(tm_signed.astype(np.float32))
    assert _rms(abs_err) < 1e-3, (
        f"torus |sdf| RMS vs trimesh |signed_distance| {_rms(abs_err):.4e} ≥ 1e-3"
    )


# ---------------------------------------------------------------------------
# (d) FCPW cross-check on all three analytic meshes — exit criterion G5
# (gated; informational when FCPW is unavailable).
# ---------------------------------------------------------------------------


@requires_module("warp")
@requires_fcpw()
def test_sdf_fcpw_parity(
    analytic_meshes: list[TriangleMesh], rng: np.random.Generator
) -> None:
    """Cross-check ``signed_distance_field`` vs FCPW ``find_closest_points``.

    Tighter tolerance than the trimesh check (closest-point RMS < 1e-5)
    because FCPW and PhysicsNeMo's BVH are both numerically exact on the
    analytic meshes. Sign agreement uses FCPW ``contains()`` as the
    boolean reference. The plan's G5 target is > 99.9%, but FCPW and
    winding-number sign can disagree on ray-grazing edges even on
    watertight inputs, so this assertion is loosened to > 99% per the
    Round-1 calibration described in the prompt.
    """
    import fcpw

    n = 500
    for mesh in analytic_meshes:
        pts = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)

        # Build an FCPW scene from the same vertices/indices.
        scene = fcpw.scene_3D()
        scene.set_object_count(1)
        scene.set_object_vertices(mesh.vertices.astype(np.float32), 0)
        scene.set_object_triangles(mesh.indices.astype(np.int32), 0)
        scene.build(fcpw.aggregate_type.bvh_surface_area, True)

        # FCPW closest-point query.
        interactions = fcpw.interaction_3D_list()
        scene.find_closest_points(pts, interactions)
        fcpw_closest = np.asarray([list(it.p) for it in interactions], dtype=np.float32)

        # FCPW inside/outside.
        fcpw_inside = np.asarray(scene.contains(pts), dtype=bool)

        sdf, hit = _run_sdf(mesh, pts, use_sign_winding_number=True)

        cp_err = hit - fcpw_closest
        assert _rms(cp_err) < 1e-5, (
            f"{mesh.name} closest-point RMS vs FCPW {_rms(cp_err):.4e} ≥ 1e-5"
        )

        sign_agreement = float(np.mean((sdf < 0.0) == fcpw_inside))
        assert sign_agreement > 0.99, (
            f"{mesh.name} sign agreement vs FCPW {sign_agreement:.4f} ≤ 0.99"
        )


# ---------------------------------------------------------------------------
# (e) Non-watertightness diagnostic — improvement I2.
# ---------------------------------------------------------------------------


@requires_module("warp")
def test_sdf_non_watertight_diagnostic(rng: np.random.Generator) -> None:
    """Diagnostic: winding-number sign on a deliberately broken cube.

    Removes one triangle from the cube to create a hole, then verifies
    that ``signed_distance_field(use_sign_winding_number=True)`` still
    agrees with the analytic inside-cube test on > 95% of query points.
    Documents PhysicsNeMo's better behavior on non-watertight inputs vs
    FCPW (improvement I2 in ``geopt-datagen-round1-plan.md`` §8). This is
    a *diagnostic* — the loose 95% threshold acknowledges that some
    queries will land in/near the hole and may flip sign, but the bulk
    of the volume should still classify correctly.
    """
    cube = make_cube()
    # Remove one triangle to create a hole.
    broken = TriangleMesh(
        name="cube-hole",
        vertices=cube.vertices.copy(),
        indices=cube.indices[1:].copy(),
        note="cube with one triangle removed",
    )

    n = 500
    pts = rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)

    sdf, _ = _run_sdf(broken, pts, use_sign_winding_number=True)

    inside_analytic = (
        (np.abs(pts[:, 0]) < 1.0)
        & (np.abs(pts[:, 1]) < 1.0)
        & (np.abs(pts[:, 2]) < 1.0)
    )
    inside_sdf = sdf < 0.0
    sign_agreement = float(np.mean(inside_analytic == inside_sdf))

    # Diagnostic threshold: > 95% (not a hard correctness gate). The
    # actual measured rate goes into ``reports/m1-kernel-parity.md``.
    assert sign_agreement > 0.95, (
        f"non-watertight cube sign agreement {sign_agreement:.4f} ≤ 0.95"
    )

    # Stash the measurement on the test node so the report can quote it.
    pytest._last_non_watertight_agreement = sign_agreement  # type: ignore[attr-defined]
