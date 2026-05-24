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

"""GeoPT mesh-alignment port (General variant).

Ports the General-variant ``transform_mesh`` from
``external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data_General.py``
lines 205-264 into PhysicsNeMo. The General variant performs an
**X-flip** (``vertices[:, 0] = -vertices[:, 0]``) and does **not** swap
axes — it is the path that generalizes across all 52 ShapeNet
categories and is the path we adopt here.

NOTE — divergence from the category-specific variant.
    ``GeoPT_PreTraining_Data.py:145-195`` defines a *different*
    ``transform_mesh`` that performs an **X↔Z axis swap** (not an
    X-flip) and returns a tuple whose names mis-describe their values
    (see plan §0 finding F0.6). We do **not** port that variant. If a
    future round-2 task needs category-specific axis conventions, add a
    second function (``align_mesh_geopt_category``); do not overload
    this one.

This module also corrects the misleading variable names called out by
plan §0 F0.6: GeoPT's General variant returns
``(z_min, x_avg, y_avg, scale)`` where ``z_min`` is actually the
post-flip Y-min and ``y_avg`` is the post-scale Z-mean. Our
:class:`AlignmentRecord` exposes named fields that describe what each
value really represents.

World-frame definition (plan §A Convention 4, verbatim):

    After ``align_mesh_geopt_general``:
    - **+X axis**: longitudinal (length direction); mesh extent
      normalized to ``target_length = 5.0`` units.
    - **+Y axis**: vertical (gravity-up); mesh sits on the ``Y = 0``
      plane (Y-min == 0 exactly post-alignment).
    - **+Z axis**: lateral; mesh centered on ``Z = 0``.
    - X-flip vs no-flip: the General-variant flips X
      (``new_V[:,0] = −V[:,0]``). This puts the conventional "front"
      of an asymmetric vehicle in ``+X``.

References
----------
- ``external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data_General.py:205-264``
  — the reference implementation we port.
- ``geopt-datagen-round1-plan.md`` §A (Convention 4), §5 (M3 spec),
  §0 F0.6 (variable-name correction), §8 row I4 (named record contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import trimesh

__all__ = ["AlignmentRecord", "align_mesh_geopt_general"]


# Oversize-safety hard limits (verbatim from GeoPT
# ``GeoPT_PreTraining_Data_General.py`` line 231). If, after the unmodified
# ``target_length / x_extent`` scale, *any* of the three axis extents exceeds
# its limit below, GeoPT halves the scale and re-applies it. This is an
# inherited hack with no documented justification in the GeoPT repo; we
# preserve the behavior for parity but record when it triggers so callers
# can audit it.
_OVERSIZE_LIMIT_X: float = 5.5
_OVERSIZE_LIMIT_Y: float = 3.0
_OVERSIZE_LIMIT_Z: float = 6.0


@dataclass(frozen=True)
class AlignmentRecord:
    """Named record of the affine transform applied by :func:`align_mesh_geopt_general`.

    Replaces GeoPT's misleading ``(z_min, x_avg, y_avg, scale)`` return
    tuple (plan §0 F0.6, §8 row I4) where ``z_min`` was actually the
    post-axis-swap Y-min and ``y_avg`` was the post-axis-swap Z-mean.
    The fields below describe what each value really is, not what the
    GeoPT identifier was called.

    Composition (apply order matches :meth:`apply`):

    1. ``axis_flipped`` — if ``True``, negate the X-coordinate.
    2. Y-floor shift — subtract ``y_min_post_flip`` from Y.
    3. Uniform ``scale`` — multiply all three axes.
    4. XZ-recenter — subtract ``x_mean_post_scale`` from X and
       ``z_mean_post_scale`` from Z. Y is **not** re-centered (it stays
       grounded at ``Y=0`` from step 2).

    Attributes
    ----------
    axis_flipped
        ``True`` if the X-axis was flipped. Always ``True`` for the
        General variant; we keep the field for forward-compatibility
        with a future no-flip variant.
    y_min_post_flip
        Y-minimum of the (possibly X-flipped) vertices in the original
        scale. Subtracted from Y to ground the mesh on ``Y=0``.
    scale
        Final uniform scale factor applied to all three axes.
        ``target_length / x_extent`` for the typical case; halved when
        the oversize-safety branch fires.
    x_mean_post_scale
        Mean of the X-coordinates after the scale step. Subtracted to
        center the mesh on ``X=0``.
    z_mean_post_scale
        Mean of the Z-coordinates after the scale step. Subtracted to
        center the mesh on ``Z=0``.
    oversize_safety_applied
        ``True`` if the GeoPT oversize-safety branch (line 231-232 of
        ``GeoPT_PreTraining_Data_General.py``) fired, halving the
        scale.
    """

    axis_flipped: bool
    y_min_post_flip: float
    scale: float
    x_mean_post_scale: float
    z_mean_post_scale: float
    oversize_safety_applied: bool

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply the recorded transform to an ``(N, 3)`` point array.

        Mirrors GeoPT's ``transform_pointcloud``
        (``GeoPT_PreTraining_Data_General.py:248-264``). Used by
        :func:`build_pretraining_sample` to keep pre-sampled surface
        points consistent with the post-alignment mesh.

        Parameters
        ----------
        points
            ``(N, 3)`` array of points in the original frame.

        Returns
        -------
        np.ndarray
            ``(N, 3)`` array in the aligned frame. Same dtype as input.
        """
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(
                f"apply expects an (N, 3) array; got shape {points.shape}."
            )
        out = points.astype(points.dtype, copy=True)
        if self.axis_flipped:
            out[:, 0] = -out[:, 0]
        out[:, 1] -= self.y_min_post_flip
        out *= self.scale
        out[:, 0] -= self.x_mean_post_scale
        out[:, 2] -= self.z_mean_post_scale
        return out

    def inverse(self, points: np.ndarray) -> np.ndarray:
        """Recover original-frame coordinates from aligned-frame points.

        Inverts :meth:`apply` step-by-step in reverse. Useful for
        round-2 evaluation (mapping fine-tuning predictions back into
        the physical-units frame the user supplied).

        Parameters
        ----------
        points
            ``(N, 3)`` array of points in the aligned frame.

        Returns
        -------
        np.ndarray
            ``(N, 3)`` array in the original frame. Same dtype as
            input.
        """
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(
                f"inverse expects an (N, 3) array; got shape {points.shape}."
            )
        out = points.astype(points.dtype, copy=True)
        # Undo XZ-recenter.
        out[:, 0] += self.x_mean_post_scale
        out[:, 2] += self.z_mean_post_scale
        # Undo uniform scale.
        out /= self.scale
        # Undo Y-floor shift.
        out[:, 1] += self.y_min_post_flip
        # Undo X-flip.
        if self.axis_flipped:
            out[:, 0] = -out[:, 0]
        return out


def align_mesh_geopt_general(
    mesh: "trimesh.Trimesh",
    target_length: float = 5.0,
    oversize_safety: bool = True,
) -> tuple["trimesh.Trimesh", AlignmentRecord]:
    """Port of GeoPT General-variant ``transform_mesh`` (X-flip).

    Aligns a mesh to PhysicsNeMo's GeoPT world frame (plan §A Convention
    4): X is longitudinal with extent ``target_length``, Y is vertical
    with the mesh sitting on ``Y=0``, Z is lateral and centered on
    ``Z=0``. The X-axis is flipped so the conventional "front" of an
    asymmetric vehicle ends up at ``+X``.

    Steps (matching GeoPT
    ``GeoPT_PreTraining_Data_General.py:205-247`` line for line, with
    corrected variable names):

    1. **X-flip.** ``vertices[:, 0] = -vertices[:, 0]``.
       Always done in the General variant (line 220).
    2. **Y-floor shift.** Subtract ``y_min`` from Y so the mesh sits on
       ``Y=0``. Computed *after* the flip on the flipped vertices.
    3. **Uniform scale.** ``scale = target_length / x_extent``; applied
       to all three axes.
    4. **Oversize safety** (if ``oversize_safety=True``): if any of
       ``x_extent * scale > 5.5``, ``y_extent * scale > 3.0``,
       ``z_extent * scale > 6.0`` after the unmodified scale, halve
       the scale and re-apply. Inherited GeoPT hack (line 231-232);
       limits are verbatim from the reference. Recorded in
       :attr:`AlignmentRecord.oversize_safety_applied`.
    5. **XZ-recenter.** Subtract per-axis mean from X and Z. Y is
       **not** re-centered.
    6. **Validate** invariants: ``x_extent ≈ target_length``,
       ``y_min ≈ 0``, ``x_mean ≈ 0``, ``z_mean ≈ 0`` within float32
       tolerance. The M3 round-trip test asserts the same invariants
       end-to-end.

    Unlike the GeoPT General variant (which mutates its input mesh in
    place), this function copies via ``mesh.copy()`` first; the input
    mesh is unchanged on return. This matches general PhysicsNeMo
    discipline.

    Parameters
    ----------
    mesh
        Input ``trimesh.Trimesh``. Not mutated.
    target_length
        Desired post-alignment X-extent. Default ``5.0`` (the GeoPT
        General-variant default).
    oversize_safety
        If ``True`` (default), apply the GeoPT ``*0.5`` scale haircut
        when post-scale extents exceed ``(5.5, 3.0, 6.0)`` on any axis.
        Set to ``False`` to disable for testing or for callers who want
        to enforce strict ``target_length`` adherence.

    Returns
    -------
    transformed_mesh : trimesh.Trimesh
        A new mesh with the alignment applied. Faces are unchanged;
        only vertices are rewritten.
    alignment_record : AlignmentRecord
        Named record of the transform; can be re-applied to
        pre-sampled surface points via :meth:`AlignmentRecord.apply`
        and inverted via :meth:`AlignmentRecord.inverse`.

    See Also
    --------
    AlignmentRecord : the named-fields record this function returns.
    """
    import trimesh as _trimesh  # noqa: F401  (validates dependency at call time)

    # Defensive copy: we do not mutate the input mesh, unlike the GeoPT
    # General variant which writes back into ``mesh.vertices``.
    out_mesh = mesh.copy()
    vertices = np.asarray(out_mesh.vertices, dtype=np.float64).copy()

    # Step 1 — X-flip. Always done in the General variant.
    axis_flipped = True
    vertices[:, 0] = -vertices[:, 0]

    # Step 2 — Y-floor shift. Compute post-flip y_min, then subtract.
    y_min_post_flip = float(vertices[:, 1].min())
    vertices[:, 1] -= y_min_post_flip

    # Step 3 — Uniform scale from target_length / x_extent.
    bound_min = vertices.min(axis=0)
    bound_max = vertices.max(axis=0)
    extents = bound_max - bound_min  # (x_extent, y_extent, z_extent)
    x_extent = float(extents[0])
    if x_extent <= 0.0:
        raise ValueError(
            "align_mesh_geopt_general: x-extent of the X-flipped mesh is "
            f"non-positive ({x_extent}). Cannot scale to target_length."
        )
    scale = target_length / x_extent

    # Step 4 — Oversize safety. Inherited GeoPT hack
    # (GeoPT_PreTraining_Data_General.py line 231-232). The limits 5.5 / 3.0 /
    # 6.0 are taken verbatim from the reference; no documented justification
    # in the GeoPT repo. We preserve the behavior and record it.
    oversize_safety_applied = False
    if oversize_safety:
        post_scale_x = extents[0] * scale
        post_scale_y = extents[1] * scale
        post_scale_z = extents[2] * scale
        if (
            post_scale_x > _OVERSIZE_LIMIT_X
            or post_scale_y > _OVERSIZE_LIMIT_Y
            or post_scale_z > _OVERSIZE_LIMIT_Z
        ):
            scale *= 0.5
            oversize_safety_applied = True

    vertices *= scale

    # Step 5 — XZ-recenter. Y is NOT re-centered (stays on Y=0 from step 2).
    x_mean_post_scale = float(vertices[:, 0].mean())
    z_mean_post_scale = float(vertices[:, 2].mean())
    vertices[:, 0] -= x_mean_post_scale
    vertices[:, 2] -= z_mean_post_scale

    # Step 6 — Validate post-conditions. float32 ulp at scale ~5 is ~5e-7;
    # the slack below covers float64-arithmetic-then-float32-storage
    # round-off and the recenter mean-summation error.
    final_min = vertices.min(axis=0)
    final_max = vertices.max(axis=0)
    final_x_extent = float(final_max[0] - final_min[0])
    final_y_min = float(final_min[1])
    final_x_mean = float(vertices[:, 0].mean())
    final_z_mean = float(vertices[:, 2].mean())

    # When oversize_safety_applied is True, x_extent will NOT match
    # target_length (it is half by construction). Skip that check on that
    # branch.
    tol = 1e-5
    violations: list[str] = []
    if not oversize_safety_applied and abs(final_x_extent - target_length) >= tol:
        violations.append(f"x_extent={final_x_extent}, expected {target_length}")
    if abs(final_y_min) >= tol:
        violations.append(f"y_min={final_y_min}, expected 0")
    if abs(final_x_mean) >= tol:
        violations.append(f"x_mean={final_x_mean}, expected 0")
    if abs(final_z_mean) >= tol:
        violations.append(f"z_mean={final_z_mean}, expected 0")
    if violations:
        raise RuntimeError(
            "align_mesh_geopt_general post-condition violated "
            f"(tol={tol}): " + "; ".join(violations) + "."
        )

    # Write transformed vertices back. Preserve the original vertex dtype
    # of the copied mesh (typically float64 in trimesh 3.x).
    out_mesh.vertices = vertices.astype(out_mesh.vertices.dtype, copy=False)

    record = AlignmentRecord(
        axis_flipped=axis_flipped,
        y_min_post_flip=y_min_post_flip,
        scale=scale,
        x_mean_post_scale=x_mean_post_scale,
        z_mean_post_scale=z_mean_post_scale,
        oversize_safety_applied=oversize_safety_applied,
    )
    return out_mesh, record
