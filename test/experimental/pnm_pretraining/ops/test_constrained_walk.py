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

"""Parity tests for the GeoPT constrained-walk composite operator.

Implements Milestone 2's deliverable 2 from
``geopt-datagen-round1-plan.md`` §4. Each test maps to one of the M2
exit criteria G6–G10.

Geometry-direction convention (load-bearing)
--------------------------------------------
Per ``geopt-datagen-round1-plan.md`` §A *Convention 1*, the supervise
output is **surface-pointing**: ``supervise[i] = closest - position``.
This is the **opposite sign** of the GeoPT reference at
``GeoPT_PreTraining_Data.py:319``. Test (e) negates the GeoPT
reference output before comparing.

Test breakdown
--------------
- ``test_constrained_walk_single_step_sphere`` — G6 (analytic
  single-step parity on a sphere).
- ``test_constrained_walk_surface_pin`` — G8 (surface-pin invariant,
  bit-exact).
- ``test_constrained_walk_boundary_rule_sticking`` — G7 (the GeoPT
  0.99 sticking haircut on collision).
- ``test_generate_walks_diversity_api`` — improvement I10 (honest
  ``n_independent + n_jittered_per_base + perturb_sigma`` API).
- ``test_constrained_walk_geopt_parity_sphere`` — G9 on the analytic
  sphere (gated on FCPW + ``PNM_GEOPT_REF``).
- ``test_constrained_walk_geopt_parity_broken_cube`` — illustrative
  diagnostic on a non-watertight cube (gated; documents I2 at the
  composite level).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from physicsnemo.experimental.pnm_pretraining.ops import (
    constrained_walk,
    generate_walks,
)
from test.conftest import requires_module
from test.experimental.pnm_pretraining.conftest import (
    TriangleMesh,
    make_cube,
    requires_fcpw,
    requires_geopt_reference,
)


def _rms(x: np.ndarray) -> float:
    """Root-mean-square (element-wise, then averaged)."""
    return float(np.sqrt(np.mean(np.square(x))))


def _import_geopt_reference():
    """Import the GeoPT reference module from ``$PNM_GEOPT_REF``.

    Stubs out optional viz deps (``polyscope``) that the reference
    module imports at top-level but the constrained-walk code path does
    not actually use.
    """
    geopt_ref_path = os.environ["PNM_GEOPT_REF"]

    # Stub polyscope: GeoPT_PreTraining_Data.py imports it at module
    # top-level for ``visualize_walk_results``, which we never call.
    if "polyscope" not in sys.modules:
        import types

        sys.modules["polyscope"] = types.ModuleType("polyscope")

    sys.path.insert(0, os.path.join(geopt_ref_path, "data_generation"))
    try:
        import GeoPT_PreTraining_Data as geopt_ref  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    return geopt_ref


# ---------------------------------------------------------------------------
# (a) Single-step analytic sphere — G6.
# ---------------------------------------------------------------------------


@requires_module("warp")
def test_constrained_walk_single_step_sphere(
    sphere_mesh: TriangleMesh, rng: np.random.Generator
) -> None:
    """Single-step supervise vectors point inward on outside-sphere queries.

    Volume points are sampled in ``[-2, 2]^3`` and rejection-filtered to
    those strictly outside the unit sphere (``|p| > 1``). Surface points
    are random unit-norm vectors.

    Assertions (per ``geopt-datagen-round1-plan.md`` §4.3 G6 and §A):

    * ``|supervise[i]| ≈ |p_i| - 1`` for volume rows (distance to the
      analytic unit-sphere surface; tolerance ``5e-2`` accommodates the
      UV-sphere chord error of the 16×32 fixture).
    * ``|supervise[j]| ≈ 0`` for surface rows.
    * ``supervise[i] · p_i < 0`` for outside-sphere volume rows
      (surface-pointing means: for outside queries, the supervise vector
      points *inward* toward the unit sphere; cf. plan §A Convention 1).
    """
    # Rejection-sample 256 volume points strictly outside the unit sphere.
    candidates = rng.uniform(-2.0, 2.0, size=(2048, 3)).astype(np.float32)
    norms = np.linalg.norm(candidates, axis=-1)
    vol_np = candidates[norms > 1.05][:256]  # margin to avoid near-surface ties
    assert vol_np.shape[0] == 256, "rejection sampling gave fewer than 256 points"

    # 64 surface points: random direction normalized.
    surf_dirs = rng.normal(size=(64, 3)).astype(np.float32)
    surf_dirs /= np.linalg.norm(surf_dirs, axis=-1, keepdims=True)
    surf_np = surf_dirs

    v, f = sphere_mesh.to_torch(device="cpu")
    vol = torch.as_tensor(vol_np, dtype=torch.float32)
    surf = torch.as_tensor(surf_np, dtype=torch.float32)

    out = constrained_walk(v, f, vol, surf, n_steps=1, max_step=2.0, seed=0)
    supervise = out["supervise"][:, 0, :].numpy()  # (N+M, 3)

    # Volume rows: |supervise| ≈ |p| - 1.
    vol_supervise = supervise[: vol_np.shape[0]]
    expected_dist = norms[norms > 1.05][:256] - 1.0
    abs_err = np.abs(np.linalg.norm(vol_supervise, axis=-1) - expected_dist)
    assert _rms(abs_err) < 5e-2, (
        f"volume |supervise| RMS error {_rms(abs_err):.4e} ≥ 5e-2"
    )

    # Volume rows: supervise · p < 0 (inward-pointing for outside queries).
    dot = np.sum(vol_supervise * vol_np, axis=-1)
    inward_frac = float(np.mean(dot < 0.0))
    assert inward_frac > 0.99, (
        f"volume supervise inward-pointing fraction {inward_frac:.4f} ≤ 0.99 — "
        "surface-pointing convention violated"
    )

    # Surface rows: |supervise| ≈ 0.
    surf_supervise = supervise[vol_np.shape[0] :]
    surf_norms = np.linalg.norm(surf_supervise, axis=-1)
    assert _rms(surf_norms) < 5e-2, (
        f"surface |supervise| RMS {_rms(surf_norms):.4e} ≥ 5e-2"
    )


# ---------------------------------------------------------------------------
# (b) Surface-pin invariant — G8.
# ---------------------------------------------------------------------------


@requires_module("warp")
def test_constrained_walk_surface_pin(sphere_mesh: TriangleMesh) -> None:
    """Surface points stay bit-exact pinned across all steps.

    Volume points: a few interior probes (we don't care about their
    motion here). Surface points: the six axis intersections with the
    unit sphere (``(±1, 0, 0)``, ``(0, ±1, 0)``, ``(0, 0, ±1)``).

    Assertions (G8):

    * ``positions_final[surface_rows] == surf_pts`` bit-exact after 3
      steps.
    * ``step_lengths[surface_rows] == 0`` (forced by the orchestrator).
    """
    surf_np = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )
    vol_np = np.array(
        [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]], dtype=np.float32
    )

    v, f = sphere_mesh.to_torch(device="cpu")
    vol = torch.as_tensor(vol_np)
    surf = torch.as_tensor(surf_np)

    out = constrained_walk(v, f, vol, surf, n_steps=3, max_step=1.0, seed=42)

    final = out["positions_final"].numpy()
    surf_final = final[vol_np.shape[0] :]
    np.testing.assert_array_equal(surf_final, surf_np)

    steps = out["step_lengths"].numpy()
    surf_steps = steps[vol_np.shape[0] :]
    np.testing.assert_array_equal(
        surf_steps, np.zeros(surf_np.shape[0], dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# (c) Boundary rule (0.99 sticking) — G7.
# ---------------------------------------------------------------------------


@requires_module("warp")
def test_constrained_walk_boundary_rule_sticking(sphere_mesh: TriangleMesh) -> None:
    """A volume point aimed straight at the surface stops at 99% of the hit distance.

    Setup (deliberate collision):
    - 1 volume point at ``(2, 0, 0)`` (radius 2, outside the unit sphere).
    - Direction ``(-1, 0, 0)`` (toward the sphere).
    - Step length ``5.0`` (would overshoot the sphere by 4 units).

    Per the GeoPT 0.99 sticking haircut (``GeoPT_PreTraining_Data.py:333``)
    and the kernel's collision branch, the post-step position should be
    ``p + d * hit_distance * 0.99``. With ``hit_distance = 1.0`` (from
    ``(2,0,0)`` to ``(1,0,0)``), the post-step position is ``(2 - 0.99,
    0, 0) = (1.01, 0, 0)``, i.e. radius ≈ 1.01.

    Tolerance ``0.05`` accounts for the UV-sphere chord error (the ray
    actually hits a slightly-inside-of-1 facet).

    Sign sanity (per plan §A Convention 1): at the start, ``p = (2,0,0)``,
    ``closest = (1,0,0)``, so ``supervise = closest - p = (-1, 0, 0)``
    and ``supervise · p = -2 < 0`` (surface-pointing means *inward* for
    outside queries).
    """
    v, f = sphere_mesh.to_torch(device="cpu")
    vol = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    surf = torch.zeros(0, 3, dtype=torch.float32)
    directions = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32)
    step_lengths = torch.tensor([5.0], dtype=torch.float32)

    out = constrained_walk(
        v, f, vol, surf, n_steps=1, directions=directions, step_lengths=step_lengths
    )

    # Single-step walk: positions_final == start (we don't move on the
    # final step; mirrors GeoPT). To check the boundary rule we need a
    # 2-step walk.
    out2 = constrained_walk(
        v, f, vol, surf, n_steps=2, directions=directions, step_lengths=step_lengths
    )
    pos_final = out2["positions_final"].numpy()[0]
    r = float(np.linalg.norm(pos_final))
    assert abs(r - 1.01) < 0.05, (
        f"post-step radius {r:.4f} not within ±0.05 of 1.01 (0.99 haircut violated)"
    )

    # Sign sanity on the initial supervise.
    supervise0 = out["supervise"][0, 0].numpy()
    p0 = vol[0].numpy()
    assert float(np.dot(supervise0, p0)) < 0.0, (
        "supervise · p ≥ 0 for an outside query — surface-pointing convention violated"
    )


# ---------------------------------------------------------------------------
# (d) Walk diversity API — improvement I10.
# ---------------------------------------------------------------------------


@requires_module("warp")
def test_generate_walks_diversity_api(sphere_mesh: TriangleMesh) -> None:
    """``generate_walks`` interleaves base + jittered walks per base.

    Spec (improvement I10):

    * ``n_walks == n_independent * (1 + n_jittered_per_base)``.
    * ``is_independent`` interleaves: True at index 0, False for the
      next ``n_jittered_per_base``, True at index
      ``1 + n_jittered_per_base``, etc.
    * Jittered walks reuse the immediately-preceding base walk's step
      lengths *bit-exact*.
    * Jittered walks have unit-norm directions within ``4σ = 0.20`` of
      the base direction (per-component; loose because we renormalize
      after jitter).
    """
    v, f = sphere_mesh.to_torch(device="cpu")
    vol = torch.tensor(
        [[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]], dtype=torch.float32
    )
    surf = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

    n_indep = 3
    n_jit = 2
    sigma = 0.05
    out = generate_walks(
        v,
        f,
        vol,
        surf,
        n_independent=n_indep,
        n_jittered_per_base=n_jit,
        perturb_sigma=sigma,
        n_steps=2,
        seed=7,
    )

    n_walks = n_indep * (1 + n_jit)
    n_tot = vol.shape[0] + surf.shape[0]
    assert out["directions"].shape == (n_walks, n_tot, 3)
    assert out["step_lengths"].shape == (n_walks, n_tot)
    assert out["supervise"].shape == (n_walks, n_tot, 2, 3)
    assert out["is_independent"].shape == (n_walks,)

    # Interleave pattern.
    expected = np.array(
        [True, False, False, True, False, False, True, False, False], dtype=bool
    )
    np.testing.assert_array_equal(out["is_independent"].numpy(), expected)

    # Per-base verification.
    step_lengths = out["step_lengths"].numpy()
    directions = out["directions"].numpy()
    stride = 1 + n_jit
    for base_idx in range(n_indep):
        base = base_idx * stride
        # Step-lengths reused bit-exact across all jittered siblings.
        base_steps = step_lengths[base]
        for k in range(1, stride):
            np.testing.assert_array_equal(step_lengths[base + k], base_steps)

        # Directions within 4σ on each component (after renormalization;
        # this is loose because the renorm changes each component but
        # bounded by ‖jitter‖ <≈ √3 · σ at 1σ).
        base_dirs = directions[base]
        for k in range(1, stride):
            diff = directions[base + k] - base_dirs
            assert np.max(np.abs(diff)) < 4.0 * sigma + 1e-2, (
                f"jittered walk {base + k} direction differs from base {base} "
                f"by {np.max(np.abs(diff)):.4f} > 4σ + 1e-2"
            )

        # Sanity: jittered walks are unit-norm.
        for k in range(stride):
            d = directions[base + k]
            d_norm = np.linalg.norm(d, axis=-1)
            assert np.allclose(d_norm, 1.0, atol=1e-5), (
                f"walk {base + k} directions not unit-norm: max |‖d‖-1| = "
                f"{float(np.max(np.abs(d_norm - 1.0))):.4e}"
            )


# ---------------------------------------------------------------------------
# (e) GeoPT-reference parity on the analytic sphere — G9.
# ---------------------------------------------------------------------------


@requires_module("warp")
@requires_fcpw()
@requires_geopt_reference()
def test_constrained_walk_geopt_parity_sphere(sphere_mesh: TriangleMesh) -> None:
    """Per-element supervise parity vs the GeoPT CPU reference.

    Stand-in for the planned ShapeNet-corpus parity test (the corpus is
    not provisioned; M1 G2 yellow carry-forward). The analytic sphere
    exercises the same closest-point + ray-cast + 0.99 haircut code
    paths.

    Pipeline parity steps (per ``geopt-datagen-round1-plan.md`` §A and
    §0 F0.8):

    1. Build a ``geopt_ref.FCPWScene`` from the same vertices/indices.
    2. Sample identical initial directions (uniform-S²) and step lengths
       (``U[0, 2]``) from a ``np.random.default_rng(0)`` so both
       pipelines see the same inputs.
    3. Run our ``constrained_walk`` and the GeoPT reference's
       ``multi_step_constrained_walk_with_surface``.
    4. **Negate the GeoPT supervise** (plan I1 sign convention).
    5. **Round-trip both through fp16** (matches GeoPT's on-disk
       storage; F0.8).
    6. Assert per-element supervise RMS < 1e-2 and per-pipeline
       percentile (5/50/95) of ``|supervise|`` matches within ±10%.

    The 1e-2 tolerance is intentionally looser than the per-kernel
    closest-point tolerance because:

    * FCPW's ``contains()`` and PhysicsNeMo's winding-number sign
      disagree on ~6% of the sphere queries (M1 finding I2). On those
      points the sign of the *reported distance* differs, but the
      **closest point itself** still agrees to within 1.4e-4 — and the
      composite supervise is computed from the closest point, not the
      sign. So the composite parity is dominated by floating-point
      reconstruction noise, not by the sign disagreement.
    * fp16 round-trip introduces ~3e-4 quantization noise on
      magnitude-1 vectors.
    """
    geopt_ref = _import_geopt_reference()

    import trimesh as tm

    np.random.seed(0)  # GeoPT's reference uses np.random internally
    rng = np.random.default_rng(0)

    n_vol = 64
    n_surf = 16
    # Rejection-sample volume points outside the unit sphere.
    candidates = rng.uniform(-2.0, 2.0, size=(2048, 3)).astype(np.float32)
    norms = np.linalg.norm(candidates, axis=-1)
    vol_np = candidates[norms > 1.05][:n_vol]
    assert vol_np.shape[0] == n_vol

    surf_dirs = rng.normal(size=(n_surf, 3)).astype(np.float32)
    surf_dirs /= np.linalg.norm(surf_dirs, axis=-1, keepdims=True)
    surf_np = surf_dirs

    n_tot = n_vol + n_surf

    # Initial directions (uniform-S²) and step lengths matching the
    # GeoPT (φ, cos θ) parameterization, sampled from our rng so both
    # pipelines see identical inputs.
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(n_tot, 1))
    cos_t = rng.uniform(-1.0, 1.0, size=(n_tot, 1))
    sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 0.0, None))
    init_dirs = np.concatenate(
        [sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1
    ).astype(np.float32)
    init_steps = rng.uniform(0.0, 2.0, size=(n_tot,)).astype(np.float32)
    init_steps[n_vol:] = 0.0

    # GeoPT reference run.
    tm_mesh = tm.Trimesh(
        vertices=sphere_mesh.vertices.astype(np.float64),
        faces=sphere_mesh.indices.astype(np.int64),
        process=False,
    )
    scene = geopt_ref.FCPWScene(tm_mesh, build_vectorized=True)
    geopt_result = geopt_ref.multi_step_constrained_walk_with_surface(
        scene,
        vol_np,
        surf_np,
        steps=3,
        init_directions=init_dirs.copy(),
        init_step_lengths=init_steps.copy(),
    )
    geopt_supervise = geopt_result["supervise"].astype(np.float32)  # (N+M, 9)

    # Negate per plan §A I1 sign convention.
    geopt_supervise = -geopt_supervise

    # Our run.
    v, f = sphere_mesh.to_torch(device="cpu")
    vol = torch.as_tensor(vol_np)
    surf = torch.as_tensor(surf_np)
    dirs_t = torch.as_tensor(init_dirs)
    steps_t = torch.as_tensor(init_steps)

    our_result = constrained_walk(
        v, f, vol, surf, n_steps=3, directions=dirs_t, step_lengths=steps_t
    )
    our_supervise = our_result["supervise"].numpy().reshape(n_tot, 9)

    # fp16 round-trip both (GeoPT's on-disk format; F0.8).
    geopt_supervise = geopt_supervise.astype(np.float16).astype(np.float32)
    our_supervise = our_supervise.astype(np.float16).astype(np.float32)

    # Per-element RMS.
    diff = our_supervise - geopt_supervise
    rms = _rms(diff)
    assert rms < 1e-2, (
        f"per-element supervise RMS {rms:.4e} ≥ 1e-2 vs GeoPT reference (sphere)"
    )

    # Per-pipeline percentile of |supervise| matches within ±10%.
    our_norms = np.linalg.norm(our_supervise.reshape(n_tot, 3, 3), axis=-1).reshape(-1)
    geopt_norms = np.linalg.norm(geopt_supervise.reshape(n_tot, 3, 3), axis=-1).reshape(
        -1
    )
    for p in (5.0, 50.0, 95.0):
        a = float(np.percentile(our_norms, p))
        b = float(np.percentile(geopt_norms, p))
        # ±10% relative; guard against tiny denominators.
        denom = max(abs(b), 1e-6)
        assert abs(a - b) / denom < 0.10, (
            f"|supervise| {p}-th percentile mismatch: ours {a:.4e}, geopt {b:.4e}"
        )

    # Stash measurements for the M2 report.
    pytest._m2_sphere_parity = {  # type: ignore[attr-defined]
        "rms_supervise_fp16": rms,
        "our_p5": float(np.percentile(our_norms, 5)),
        "our_p50": float(np.percentile(our_norms, 50)),
        "our_p95": float(np.percentile(our_norms, 95)),
        "geopt_p5": float(np.percentile(geopt_norms, 5)),
        "geopt_p50": float(np.percentile(geopt_norms, 50)),
        "geopt_p95": float(np.percentile(geopt_norms, 95)),
        "n_vol": n_vol,
        "n_surf": n_surf,
    }


# ---------------------------------------------------------------------------
# (f) GeoPT-reference parity on a non-watertight broken cube — illustrative
#     diagnostic; documents I2 at the composite level.
# ---------------------------------------------------------------------------


@requires_module("warp")
@requires_fcpw()
@requires_geopt_reference()
def test_constrained_walk_geopt_parity_broken_cube() -> None:
    """Diagnostic: composite-walk divergence on a non-watertight cube.

    Identical setup to test (e) but with a cube with one triangle
    removed (the ``test_sdf_non_watertight_diagnostic`` fixture from
    M1). Documents that even when the *kernel-level* contains/winding
    sign disagreement is 13.8% on the watertight cube (M1 I2 finding),
    the *composite-walk* supervise tensors diverge by less in
    aggregate, because the supervise computation uses closest-point
    (which agrees within 1.4e-4) and not contains.

    No tight tolerance is asserted; the per-percentile divergence is
    measured and stashed on ``pytest`` for the M2 report.
    """
    geopt_ref = _import_geopt_reference()

    import trimesh as tm

    cube = make_cube()
    broken = TriangleMesh(
        name="cube-hole",
        vertices=cube.vertices.copy(),
        indices=cube.indices[1:].copy(),
        note="cube with one triangle removed",
    )

    np.random.seed(0)
    rng = np.random.default_rng(0)

    n_vol = 64
    n_surf = 16
    vol_np = rng.uniform(-2.0, 2.0, size=(n_vol, 3)).astype(np.float32)
    surf_np = rng.uniform(-1.0, 1.0, size=(n_surf, 3)).astype(np.float32)
    # Snap surface points to a cube face (axis-aligned) for realism.
    snap_axis = rng.integers(0, 3, size=n_surf)
    snap_sign = rng.choice([-1.0, 1.0], size=n_surf).astype(np.float32)
    for i in range(n_surf):
        surf_np[i, snap_axis[i]] = snap_sign[i]

    n_tot = n_vol + n_surf

    phi = rng.uniform(0.0, 2.0 * np.pi, size=(n_tot, 1))
    cos_t = rng.uniform(-1.0, 1.0, size=(n_tot, 1))
    sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 0.0, None))
    init_dirs = np.concatenate(
        [sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1
    ).astype(np.float32)
    init_steps = rng.uniform(0.0, 2.0, size=(n_tot,)).astype(np.float32)
    init_steps[n_vol:] = 0.0

    tm_mesh = tm.Trimesh(
        vertices=broken.vertices.astype(np.float64),
        faces=broken.indices.astype(np.int64),
        process=False,
    )
    scene = geopt_ref.FCPWScene(tm_mesh, build_vectorized=True)
    geopt_result = geopt_ref.multi_step_constrained_walk_with_surface(
        scene,
        vol_np,
        surf_np,
        steps=3,
        init_directions=init_dirs.copy(),
        init_step_lengths=init_steps.copy(),
    )
    geopt_supervise = -geopt_result["supervise"].astype(np.float32)

    v = torch.as_tensor(broken.vertices, dtype=torch.float32)
    f = torch.as_tensor(broken.indices, dtype=torch.int32)
    vol = torch.as_tensor(vol_np)
    surf = torch.as_tensor(surf_np)
    dirs_t = torch.as_tensor(init_dirs)
    steps_t = torch.as_tensor(init_steps)
    our_result = constrained_walk(
        v, f, vol, surf, n_steps=3, directions=dirs_t, step_lengths=steps_t
    )
    our_supervise = our_result["supervise"].numpy().reshape(n_tot, 9)

    geopt_supervise = geopt_supervise.astype(np.float16).astype(np.float32)
    our_supervise = our_supervise.astype(np.float16).astype(np.float32)

    rms = _rms(our_supervise - geopt_supervise)
    our_norms = np.linalg.norm(our_supervise.reshape(n_tot, 3, 3), axis=-1).reshape(-1)
    geopt_norms = np.linalg.norm(geopt_supervise.reshape(n_tot, 3, 3), axis=-1).reshape(
        -1
    )

    pytest._m2_broken_cube_parity = {  # type: ignore[attr-defined]
        "rms_supervise_fp16": rms,
        "our_p5": float(np.percentile(our_norms, 5)),
        "our_p50": float(np.percentile(our_norms, 50)),
        "our_p95": float(np.percentile(our_norms, 95)),
        "geopt_p5": float(np.percentile(geopt_norms, 5)),
        "geopt_p50": float(np.percentile(geopt_norms, 50)),
        "geopt_p95": float(np.percentile(geopt_norms, 95)),
        "n_vol": n_vol,
        "n_surf": n_surf,
    }

    # Loose ceiling: divergence should be O(1) units, not catastrophically
    # large. This is a sanity bound, not a correctness gate.
    assert rms < 1.0, (
        f"broken-cube composite RMS {rms:.4e} > 1.0 — composite walk "
        "diverged catastrophically; investigate before claiming I2 holds"
    )
