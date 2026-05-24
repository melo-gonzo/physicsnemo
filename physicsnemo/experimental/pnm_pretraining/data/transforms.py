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

"""GeoPT mesh-alignment port (General variant) and pretraining transforms.

Two responsibilities live here:

* :func:`align_mesh_geopt_general` and :class:`AlignmentRecord` — the
  General-variant ``transform_mesh`` from
  ``external-repos/GeoPT/data_generation/GeoPT_PreTraining_Data_General.py``
  lines 205-264, ported into PhysicsNeMo.
* :class:`WalkSampler` — a per-sample dataset transform that slices
  one walk out of the ``(n_walks, ...)`` arrays the M3 builder
  emits in ``interior.global_data`` and writes the per-point
  ``(supervise, directions, step_lengths)`` slices into
  ``interior.point_data`` so the recipe's ``extract_targets`` (which
  only reads from ``interior.point_data``) can find them. See
  :class:`WalkSampler` for the rationale and ``geopt-datagen-round1-plan.md``
  §8 row I20 for the design decision.

Mesh alignment — General-variant ``transform_mesh``
---------------------------------------------------

The General variant performs an **X-flip**
(``vertices[:, 0] = -vertices[:, 0]``) and does **not** swap axes — it
is the path that generalizes across all 52 ShapeNet categories and is
the path we adopt here.

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
  §0 F0.6 (variable-name correction), §8 row I4 (named record
  contract), §8 row I20 (WalkSampler).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh

if TYPE_CHECKING:
    import trimesh

__all__ = ["AlignmentRecord", "WalkSampler", "align_mesh_geopt_general"]


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


# ---------------------------------------------------------------------------
# WalkSampler — per-sample dataset transform
# ---------------------------------------------------------------------------


class WalkSampler(MeshTransform):
    """Pick one walk per sample and lift it onto ``interior.point_data``.

    The M3 builder writes **every** independent + jittered walk to disk
    in ``interior.global_data`` (per finding I16 — those arrays have a
    leading ``n_walks`` dim and cannot live in ``point_data``, which
    must satisfy the TensorDict ``batch_size == [n_points]`` invariant
    required by ``Mesh.__post_init__``). At training time the recipe's
    ``extract_targets`` reads exclusively from ``interior.point_data``,
    and per-point loss/metric kernels assume per-point shapes. This
    transform resolves the gap by:

    1. drawing a random walk index ``i ∈ [0, n_walks)`` from a
       per-instance ``torch.Generator`` seeded by the constructor,
    2. slicing the three walk arrays at index ``i`` into per-point
       shapes,
    3. writing the slices into ``interior.point_data`` under the
       names ``supervise``, ``directions``, ``step_lengths``, and
    4. dropping the original ``(n_walks, …)`` arrays from
       ``interior.global_data`` so they don't inflate batch payloads.

    The on-disk ``.pdmsh`` keeps every walk for offline reproducibility
    (improvement I20 in ``geopt-datagen-round1-plan.md`` §8); the
    per-sample reshape is purely a runtime concern.

    Parameters
    ----------
    seed
        Per-instance RNG seed. ``None`` falls back to PyTorch's global
        seed (typically a poor choice for reproducible runs; the
        recipe's seed-on-rank-0-and-broadcast convention assigns one
        explicitly). Stored as a ``torch.Generator`` so determinism
        does not depend on the global state of any other code path.
    walk_field_name
        Source key in ``interior.global_data`` for the supervise array
        of shape ``(n_walks, n_points, n_steps, 3)``. Override only if
        the producer renamed the field.
    directions_field_name
        Source key for the directions array of shape
        ``(n_walks, n_points, 3)``.
    step_lengths_field_name
        Source key for the step-lengths array of shape
        ``(n_walks, n_points)``.

    Attributes
    ----------
    generator
        The ``torch.Generator`` used for walk-index sampling. Exposed
        for tests and for explicit re-seeding (``set_generator`` from
        the :class:`MeshTransform` base also works).

    Notes
    -----
    The transform raises ``KeyError`` when applied to a ``DomainMesh``
    that does not carry the three walk arrays in
    ``interior.global_data``. This is deliberate: feeding a non-
    pretraining sample (e.g. a DrivAerML ``.pmsh``) through this
    transform is a configuration bug, not a runtime corner case.
    The error message names the missing key and points at the M3
    builder.

    See also
    --------
    physicsnemo.experimental.pnm_pretraining.data.builder.build_pretraining_sample
        The producer of the on-disk schema this transform consumes.
    """

    _MISSING_WALK_HINT = (
        " expected `{key}` in interior.global_data — was this DomainMesh "
        "built by `physicsnemo.experimental.pnm_pretraining.data."
        "build_pretraining_sample`?"
    )

    def __init__(
        self,
        seed: int | None = None,
        walk_field_name: str = "walks_supervise",
        directions_field_name: str = "walks_directions",
        step_lengths_field_name: str = "walks_step_lengths",
    ) -> None:
        super().__init__()
        self.seed = seed
        self.walk_field_name = walk_field_name
        self.directions_field_name = directions_field_name
        self.step_lengths_field_name = step_lengths_field_name
        # Per-instance generator. Stored under ``_generator`` so the
        # base class's ``stochastic`` property reports True; also
        # exposed publicly as ``generator`` for direct test access
        # without relying on the underscore-prefixed name.
        self._generator: torch.Generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(int(seed))

    @property
    def generator(self) -> torch.Generator:
        """The per-instance ``torch.Generator`` driving walk-index sampling."""
        return self._generator

    def __call__(self, data: Union[Mesh, DomainMesh]) -> Union[Mesh, DomainMesh]:  # type: ignore[override]
        """Dispatch on the input type.

        On a ``DomainMesh`` we run the per-walk slicer (real work). On
        a plain ``Mesh`` we have no boundaries / global_data invariants
        to lean on and the transform doesn't make sense — raise. The
        recipe's pipeline runner always hands us a ``DomainMesh``
        (see ``physicsnemo/datapipes/mesh_dataset.py:_load``); this
        branch matters for tests and for accidental misuse.
        """
        if isinstance(data, DomainMesh):
            return self.apply_to_domain(data)
        raise TypeError(
            "WalkSampler operates on DomainMesh; got "
            f"{type(data).__name__}. The walk arrays live in the "
            "interior Mesh's global_data, which is only reachable "
            "through a DomainMesh wrapper."
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Slice one walk and lift its per-point arrays into ``point_data``.

        Parameters
        ----------
        domain
            A pretraining-sample ``DomainMesh`` carrying the three
            walk arrays in ``interior.global_data``.

        Returns
        -------
        DomainMesh
            A new ``DomainMesh`` with ``supervise``, ``directions``,
            ``step_lengths`` injected into ``interior.point_data`` and
            the source ``(n_walks, …)`` arrays removed from
            ``interior.global_data``. The ``boundaries`` and the
            domain-level ``global_data`` are passed through unchanged.

        Raises
        ------
        KeyError
            If any of the three expected walk arrays is missing from
            ``domain.interior.global_data``.
        """
        interior = domain.interior
        global_data = interior.global_data

        # Validate up front — the recipe's transform pipeline will swallow
        # KeyErrors with limited context, so we make the message self-
        # explanatory.
        for key in (
            self.walk_field_name,
            self.directions_field_name,
            self.step_lengths_field_name,
        ):
            if key not in global_data.keys():
                raise KeyError(
                    f"WalkSampler: missing key {key!r}. "
                    + self._MISSING_WALK_HINT.format(key=key)
                )

        walks_supervise = global_data[self.walk_field_name]
        walks_directions = global_data[self.directions_field_name]
        walks_step_lengths = global_data[self.step_lengths_field_name]

        # Schema sanity (cheap and audit-friendly): all three arrays must
        # share the leading n_walks dim. We do not assert n_points parity
        # against ``interior.n_points`` here — the M3 builder enforces it
        # at write time, and this transform survives any consumer that
        # respects the documented schema.
        n_walks = walks_supervise.shape[0]
        if (
            walks_directions.shape[0] != n_walks
            or walks_step_lengths.shape[0] != n_walks
        ):
            raise ValueError(
                "WalkSampler: leading n_walks dim mismatch — "
                f"{self.walk_field_name}: {walks_supervise.shape[0]}, "
                f"{self.directions_field_name}: {walks_directions.shape[0]}, "
                f"{self.step_lengths_field_name}: {walks_step_lengths.shape[0]}."
            )

        # Sample the walk index. ``torch.randint`` honors the generator
        # we hold; the result is a 0-d int tensor, which we materialize
        # to a Python int for the slice call site.
        walk_idx = int(
            torch.randint(
                low=0,
                high=n_walks,
                size=(),
                generator=self._generator,
                device="cpu",  # generator is a CPU generator; slice idx is host-side.
            ).item()
        )

        # Slice + reshape. Trailing ``.contiguous()`` calls smooth out
        # any stride surprises a downstream model might trip on (the
        # reshape on a non-contiguous slice would fail loudly anyway,
        # but the explicit contiguous() makes the producer-side
        # invariant unambiguous).
        n_points = walks_supervise.shape[1]
        n_steps = walks_supervise.shape[2]
        supervise = (
            walks_supervise[walk_idx]
            .contiguous()
            .reshape(n_points, n_steps * 3)
            .contiguous()
        )
        directions = walks_directions[walk_idx].contiguous()
        step_lengths = walks_step_lengths[walk_idx].unsqueeze(-1).contiguous()

        # Build the new interior. point_data acquires three new keys;
        # global_data drops the n_walks source arrays. We use
        # TensorDict.exclude on global_data (null-safe per
        # DropMeshFields' implementation note in
        # physicsnemo/datapipes/transforms/mesh/transforms.py).
        new_point_data = interior.point_data.clone()
        new_point_data["supervise"] = supervise
        new_point_data["directions"] = directions
        new_point_data["step_lengths"] = step_lengths

        new_global_data = global_data.exclude(
            self.walk_field_name,
            self.directions_field_name,
            self.step_lengths_field_name,
        )

        new_interior = Mesh(
            points=interior.points,
            cells=interior.cells,
            point_data=new_point_data,
            cell_data=interior.cell_data,
            global_data=new_global_data,
        )

        return DomainMesh(
            interior=new_interior,
            boundaries=domain.boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return (
            f"seed={self.seed!r}, walk_field_name={self.walk_field_name!r}, "
            f"directions_field_name={self.directions_field_name!r}, "
            f"step_lengths_field_name={self.step_lengths_field_name!r}"
        )
