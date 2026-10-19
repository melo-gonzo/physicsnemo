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

"""
Recipe-local datapipe components that the library does not provide.

- :class:`ComputeFreestreamDirection` writes the unit freestream direction
  as a new ``global_data`` leaf (``U_inf`` stays physical for inference-side
  re-dimensionalization); ISLA reads it as its global direction.
- Samplers with exact per-cell Horvitz-Thompson weights beyond the library's
  uniform ``SubsampleMesh``: :class:`PrefixPlusRandomSubsampleMesh`,
  :class:`PoissonBiasedSubsampleMesh`, :class:`StratifiedSubsampleMesh` and
  :class:`SdfBiasedSubsampleInteriorPoints`.
- Evaluation protocols: :class:`FixedRandomPose` (a deterministic per-case
  rigid pose) with :class:`CaseKeyDomainMeshReader`, and
  :class:`FixedDrawDomainMeshReader` (an epoch-independent subsample).
- Field and geometry adapters for datasets outside external aerodynamics:
  :class:`SetConstantCellField`, :class:`StackScalarFields`,
  :class:`SqueezeScalarFields`, :class:`SplitGlobalVector`,
  :class:`RestrictToPlane`, :class:`SplitLineCells`,
  :class:`DropDegenerateCells` and :class:`ComposeQuadratureMeasure`.
- Adapters for a multi-boundary case with per-face boundary conditions
  (AeroZoo): :class:`ComputeLogReynolds`, :class:`HarmonizeBoundaryFields`,
  :class:`BoundaryFieldToGlobal` and :class:`ConcatGlobalFields`; and
  :class:`SplitBoundaryByLabels`, one boundary per condition for a dataset that
  labels its conditions inside one mesh.

Recipe-local module registered into the global datapipe component registry
so the classes can be referenced via ``${dp:...}`` short names in Hydra
YAML configs.  Import this module before Hydra instantiation
(``src/datasets.py`` does this at import time, like :mod:`nondim` and
:mod:`sdf`).
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Literal

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.keys import as_nested_key, key_to_str
from physicsnemo.datapipes.readers.mesh import DomainMeshReader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh import (
    RandomRotateMesh,
    SubsampleMesh,
)
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.datapipes.transforms.mesh.transforms import (
    _compact_points,
)
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import (
    EFFECTIVE_MEASURE_KEY,
    cell_measures,
    scale_measures,
)
from physicsnemo.mesh.projections import project
from physicsnemo.mesh.spatial.bvh import _compute_morton_codes


@register()
class ComputeFreestreamDirection(MeshTransform):
    r"""Write the unit freestream direction into ``global_data``.

    Computes ``global_data[output_field] = U / |U|`` from the physical
    freestream vector ``global_data[velocity_field]`` and stores it as a NEW
    leaf; the physical vector itself is left untouched, so inference-side
    re-dimensionalization (which derives ``q_inf`` from ``U_inf``) and the
    force-coefficient integration keep reading the correct physical
    freestream.

    Why this exists: ``NonDimensionalizeByMetadata`` non-dimensionalizes
    fields and geometry but never ``global_data``, so models that declare a
    freestream as a global vector input would otherwise consume the RAW
    physical vector (~39 m/s for DrivAerML). A surrogate whose inputs are
    declared nondimensional must not see it: an architecture of this program
    that consumed the raw vector multiplied it straight through its
    score/value products (initial predictions O(1e9), float32 overflow on
    the all-boundary arm; measured on the DrivAerML tunnel probe, 2026-07).
    The unit direction plus the Cp/Cf-nondimensionalized targets is the
    AirFRANS-campaign convention (``freestream_direction``); for DrivAerML
    the discarded magnitude is a per-dataset constant, so no per-case
    information is lost.

    Place it before the augmentation insertion point so that
    ``RandomRotateMesh(transform_global_data=true)`` rotates the direction
    together with ``U_inf`` and the geometry (it rotates every ``(3,)``
    global leaf).

    Domain-aware: on a ``DomainMesh`` the direction is computed from and
    written to the *domain-level* ``global_data`` (the level the recipe
    contract and DomainMesh-native models read).
    """

    def __init__(
        self,
        velocity_field: str = "U_inf",
        output_field: str = "U_inf_dir",
    ) -> None:
        super().__init__()
        self._velocity_field = velocity_field
        self._output_field = output_field

    def _direction(self, global_data: TensorDict) -> torch.Tensor:
        if self._velocity_field not in global_data.keys():
            raise KeyError(
                f"ComputeFreestreamDirection: {self._velocity_field!r} not "
                f"found in global_data (available: "
                f"{sorted(global_data.keys())!r})."
            )
        velocity = global_data[self._velocity_field].float()
        norm = torch.linalg.vector_norm(velocity)
        if not torch.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                f"ComputeFreestreamDirection: |{self._velocity_field}| must "
                f"be finite and positive, got {norm.item()!r}."
            )
        return velocity / norm

    def __call__(self, mesh: Mesh) -> Mesh:
        new_gd = mesh.global_data.clone()
        new_gd[self._output_field] = self._direction(mesh.global_data)
        new_mesh = mesh.copy()  # ty: ignore[unresolved-attribute]
        new_mesh.global_data = new_gd
        return new_mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Derive the flow direction from domain-level velocity metadata."""
        new_gd = domain.global_data.clone()
        new_gd[self._output_field] = self._direction(domain.global_data)
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=new_gd,
        )

    def extra_repr(self) -> str:
        return (
            f"{self._output_field} = {self._velocity_field} / |{self._velocity_field}|"
        )


@register()
class PrefixPlusRandomSubsampleMesh(SubsampleMesh):
    r"""Deterministic-prefix + random-complement subsampling (probe P4).

    Keeps the first ``n_prefix`` cells in stable cell-index order (the same
    PHYSICAL cells for every draw of a given case) plus a seeded random
    sample of the remainder up to ``n_cells``. Two evaluations with
    different ``n_cells`` (or seeds) then share the prefix exactly, so
    prediction differences AT the prefix measure companion-set sensitivity
    -- the query-independence probe. HT measure bookkeeping is inherited
    for the random stage and set to 1 for the prefix (deterministic
    inclusion).
    """

    def __init__(self, n_prefix: int, n_cells: int, compact: bool = True):
        super().__init__(n_cells=n_cells, compact=compact)
        if n_prefix > n_cells:
            raise ValueError("n_prefix must be <= n_cells")
        self.n_prefix = int(n_prefix)

    def __call__(self, mesh: Mesh) -> Mesh:
        n = mesh.n_cells
        if n <= self.n_cells:
            return mesh
        device = mesh.cells.device
        prefix = torch.arange(self.n_prefix, device=device)
        n_rest = self.n_cells - self.n_prefix
        if n_rest > 0:
            pool = n - self.n_prefix
            generator = self._generator
            if generator is not None and generator.device != device:
                generator = None
            perm = torch.randperm(pool, device=device, generator=generator)
            rest = self.n_prefix + perm[:n_rest]
            indices = torch.cat([prefix, rest])
        else:
            indices = prefix
        mesh = mesh.slice_cells(indices)
        if self.compact:
            mesh = _compact_points(mesh)
        ### Prefix cells have inclusion probability 1; the random stage
        ### carries the usual inverse inclusion probability.
        w = torch.ones(len(indices), device=device)
        if n_rest > 0:
            w[self.n_prefix :] = (n - self.n_prefix) / n_rest
        scale_measures(mesh, w)
        return mesh


### ---------------------------------------------------------------------
### Contract-axis probe transforms (prereg: contract_axis_probes_2026-08-09)
### Job-local additions for the 2026-08-06-defect4-incumbent task dir.
### ---------------------------------------------------------------------


@register()
class PoissonBiasedSubsampleMesh(MeshTransform):
    r"""Biased cell subsampling with EXACT per-cell HT weights (probe P3 v3).

    Independent (Poisson) sampling: cell i is kept with probability
    pi_i = min(1, c * w_i) where w_i is the sampling weight and c is set so
    the expected kept count is ``n_cells_expected``. Inclusion
    probabilities are exact by construction (no with-replacement
    approximation), at the cost of a variable per-sample cell count.

    ``weight_mode`` selects the weight:

    - ``"front_back"`` (default): the 10:1 (``bias``:1) front/back split
      on centroid x, the P3 density-bias probe.
    - ``"area"``: w_i = cell area (``mesh.cell_areas``), so before clamping
      pi_i is proportional to area -- the inclusion law a uniform
      point-cloud generator over the *surface* would produce, as opposed
      to the uniform-over-cells law that reads the mesher's refinement
      pattern (consistency benchmark, 2026-09-10). ``bias`` is unused.
    - ``"uniform"``: w_i = 1, the same uniform-over-cells inclusion law as
      ``SubsampleMesh`` / the reader's draw but through this class's Poisson
      draw and HT measure fold (CHAR-PROTOCOLS control, 2026-10-03: separates
      the Poisson/HT path from the area law). ``bias`` is unused.
    """

    _WEIGHT_MODES = ("front_back", "area", "uniform")

    def __init__(
        self,
        n_cells_expected: int,
        bias: float = 10.0,
        compact: bool = True,
        weight_mode: Literal["front_back", "area", "uniform"] = "front_back",
    ) -> None:
        super().__init__()
        self.n_cells_expected = int(n_cells_expected)
        self.bias = float(bias)
        self.compact = compact
        if weight_mode not in self._WEIGHT_MODES:
            raise ValueError(
                f"weight_mode must be one of {self._WEIGHT_MODES}, got {weight_mode!r}"
            )
        self.weight_mode = weight_mode
        self._generator: torch.Generator | None = None

    def _weights(self, mesh: Mesh) -> torch.Tensor:
        if self.weight_mode == "area":
            ### Degenerate (zero / non-finite) areas get weight 0: never
            ### drawn, so no 1/pi is ever formed for them.
            areas = mesh.cell_areas
            return torch.where(
                torch.isfinite(areas), areas, torch.zeros_like(areas)
            ).clamp_min(0.0)
        if self.weight_mode == "uniform":
            return torch.ones(
                mesh.n_cells, device=mesh.points.device, dtype=mesh.points.dtype
            )
        x = mesh.cell_centroids[:, 0]
        return torch.where(
            x < x.median(), torch.full_like(x, self.bias), torch.ones_like(x)
        )

    def __call__(self, mesh: Mesh) -> Mesh:
        n = mesh.n_cells
        if n <= self.n_cells_expected:
            return mesh
        w = self._weights(mesh)
        c = self.n_cells_expected / w.sum()
        pi = (c * w).clamp(max=1.0)
        ### One renormalization pass restores the expected count lost to
        ### clamping (exactness of pi is what matters, not the count).
        deficit = self.n_cells_expected - pi.sum()
        if deficit > 0:
            free = pi < 1.0
            pi[free] = (pi[free] * (1 + deficit / pi[free].sum())).clamp(max=1.0)
        generator = self._generator
        if generator is not None and generator.device != pi.device:
            generator = None
        keep = torch.rand(n, device=pi.device, generator=generator) < pi
        indices = keep.nonzero(as_tuple=True)[0]
        kept_pi = pi[indices]
        mesh = mesh.slice_cells(indices)
        if self.compact:
            mesh = _compact_points(mesh)
        scale_measures(mesh, 1.0 / kept_pi)
        return mesh


@register()
class StratifiedSubsampleMesh(MeshTransform):
    r"""Uniform-over-cells subsampling with spatial stratification (QUAD-VAR, 2026-09-17).

    Same marginal inclusion law as :class:`SubsampleMesh` -- every cell is kept
    with probability ``n / N`` and carries the measure weight ``N / n`` -- but
    the ``n`` kept cells are a *systematic* sample along the Morton (Z-order)
    ordering of the cell centroids: one cell per consecutive run of ``N / n``
    cells, with a single random offset per draw. A systematic sample has the
    same expectation as an independent draw for every cell-weighted integral
    and a lower variance for integrands that are smooth over the surface, so
    it lowers the finite-sample noise of an aggregate at a fixed token budget
    without changing the sampling convention the model was trained under.
    The offset is drawn from the transform's generator when one is attached
    (the pipeline seeds it), so the draw is reproducible; no cell is ever
    drawn twice.
    """

    def __init__(self, n_cells: int, compact: bool = True) -> None:
        super().__init__()
        self.n_cells = int(n_cells)
        self.compact = compact
        self._generator: torch.Generator | None = None

    @staticmethod
    def _morton_order(centroids: torch.Tensor) -> torch.Tensor:
        ### Z-order (Morton) curve through the centroid bounding box, so
        ### consecutive positions are spatial neighbours; the BVH's code
        ### quantizes every axis at the full int64 budget (21 bits in 3-D).
        return _compute_morton_codes(centroids).argsort(stable=True)

    def __call__(self, mesh: Mesh) -> Mesh:
        n_before = mesh.n_cells
        if n_before <= self.n_cells:
            return mesh
        order = self._morton_order(mesh.cell_centroids)
        step = n_before / self.n_cells
        generator = self._generator
        if generator is not None and generator.device != order.device:
            generator = None
        offset = torch.rand((), device=order.device, generator=generator) * step
        ### floor(offset + k * step), k = 0..n-1: n distinct positions in [0, N)
        ### because step >= 1, each cell included with probability exactly n / N.
        positions = (
            torch.floor(
                offset
                + step
                * torch.arange(self.n_cells, device=order.device, dtype=torch.float64)
            )
            .long()
            .clamp(max=n_before - 1)
        )
        indices = order[positions].sort().values
        mesh = mesh.slice_cells(indices)
        if self.compact:
            mesh = _compact_points(mesh)
        scale_measures(mesh, n_before / self.n_cells)
        return mesh

    def extra_repr(self) -> str:
        return f"n_cells={self.n_cells}"


def _pose_generator(key: int, salt: int) -> torch.Generator:
    """The keyed CPU generator every per-case pose draw comes from."""
    return torch.Generator().manual_seed(
        (int(key) * 1_000_003 + int(salt)) % (2**63 - 1)
    )


def _pose_rotation_for_key(
    key: int,
    salt: int,
    n_spatial_dims: int = 3,
    axis: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Rotation matrix drawn from a generator seeded by (key, salt).

    Three-dimensional points with ``axis=None`` (the default) get a uniform SO(3)
    rotation, the same construction as ``RandomRotateMesh(mode="uniform")``: an
    isotropic Gaussian 4-vector normalized to a unit quaternion. This stream is
    unchanged from POSE-BENCH, so earlier posed lanes reproduce bit for bit.

    A planar problem rotates IN its plane only (CHAR-PROTOCOLS, 2026-10-02): for
    two-dimensional points, or three-dimensional points with ``axis`` set (the
    coordinate normal to the plane, e.g. ``axis=2`` for a problem in the ``xy``
    plane stored with ``z = 0``), the draw is one angle uniform on ``[0, 2 pi)``
    and the matrix is the SO(2) rotation about that normal. A planar problem
    rotated out of its plane is meaningless, and the stored-in-3-D and
    restricted-to-2-D variants of one dataset receive the same in-plane angle.

    Deterministic in ``key`` and ``salt``, independent of process, worker, epoch
    or shuffle order. Pass ``generator`` to continue drawing (the translation)
    from the same keyed stream.
    """
    g = _pose_generator(key, salt) if generator is None else generator
    if n_spatial_dims == 3 and axis is None:
        q = torch.randn(4, generator=g, dtype=torch.float64)
        q = q / q.norm()
        return RandomRotateMesh._quaternion_to_rotation_matrix(q)
    if n_spatial_dims not in (2, 3):
        raise ValueError(
            f"FixedRandomPose supports 2-D and 3-D points, got {n_spatial_dims}-D"
        )
    if n_spatial_dims == 2 and axis is not None:
        raise ValueError(
            "FixedRandomPose: `axis` names the plane normal of 3-D points; 2-D points have none"
        )
    angle = torch.rand((), generator=g, dtype=torch.float64) * (2 * torch.pi)
    c, s = torch.cos(angle), torch.sin(angle)
    if n_spatial_dims == 2:
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
    if not 0 <= int(axis) < 3:
        raise ValueError(f"FixedRandomPose: axis must be 0, 1 or 2, got {axis}")
    i, j = [k for k in range(3) if k != int(axis)]
    R = torch.eye(3, dtype=torch.float64)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _pose_translation_direction(
    generator: torch.Generator, n_spatial_dims: int, axis: int | None
) -> torch.Tensor:
    """Unit direction uniform on the sphere (in the plane when ``axis`` is set or the points are 2-D)."""
    u = torch.randn(n_spatial_dims, generator=generator, dtype=torch.float64)
    if axis is not None:
        u[int(axis)] = 0.0
    return u / u.norm()


def _spatial_field_mask(data: TensorDict, n_spatial_dims: int) -> dict:
    """Nested bool mask selecting the leaves ``Mesh.transform`` can rotate.

    A leaf qualifies when every trailing (non-batch) axis has the spatial
    dimension: rank-1 vectors ``(..., d)`` and rank-2 tensors ``(..., d, d)``.
    Recipe data also carries ``(N, 1)`` scalar columns and non-spatial
    parameter vectors (``naca[5]``, ``params[6]``, ``load_type[2]``) that the
    all-fields spec ``True`` refuses with an error; the mask leaves them alone.
    """
    nb = len(data.batch_size)
    mask: dict = {}
    for key, value in data.items(include_nested=True, leaves_only=True):
        shape = tuple(value.shape[nb:])
        ok = len(shape) >= 1 and all(s == n_spatial_dims for s in shape)
        path = key if isinstance(key, tuple) else (key,)
        node = mask
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = ok
    return mask


@register()
class FixedRandomPose(MeshTransform):
    """Pose a case by a rotation (and optionally a translation) that is a deterministic function of the case.

    POSE-BENCH (transfer program, 2026-09-09). Every training and validation
    case receives an independent random pose, but the SAME pose every time
    the case is loaded, so that (a) the validation set is a fixed posed
    benchmark identical for every architecture, and (b) training on posed
    data is a data property, not an augmentation stream. The pose is seeded by
    the reader's ``case_key`` global field (``MeshReaderWithGlobalData(
    store_case_key=True)`` for a boundary Mesh, :class:`CaseKeyDomainMeshReader`
    for a DomainMesh) and ``salt``. Positions and every spatial vector or
    rank-2 tensor field in point/cell/global data (wall shear, normals,
    ``U_inf``, a load direction) rotate together; scalars and non-spatial
    vectors (parameter lists, one-hot codes) are invariant. Place it BEFORE
    ``ComputeFreestreamDirection`` so the unit direction is computed from the
    rotated freestream, and before ``CenterMesh``. Deterministic: no
    generator, never reseeded.

    CHAR-PROTOCOLS (2026-10-02) additions, all inert at their defaults:

    * ``translate``: a translation of length ``translate`` times the RMS
      radius of the points about their centroid, in a direction uniform on
      the sphere, drawn from the same keyed stream after the rotation. The
      recipe's ``CenterMesh`` (present in the external-aero and DeepJEB
      pipelines) removes any translation, so there the posed error measures
      the rotation plus the centering step; the AirfRANS, Tensile2d and lug
      pipelines have no ``CenterMesh``, so there the translation reaches the
      model. It is included because the protocol says rotated AND translated.
    * ``axis``: for a planar problem stored in 3-D coordinates, the plane
      normal (``2`` for ``z = 0``); the rotation is then in the plane only and
      the translation has no component along the normal. 2-D points (after
      ``RestrictToPlane``) are always rotated in their plane. Place the
      transform before ``RestrictToPlane`` so the 3-D baseline variant and the
      2-D variant of a dataset receive the identical pose.
    * ``rotate``: ``False`` disables the rotation (identity matrix) while the
      rotation draw is still consumed from the keyed stream, so a
      translation-only variant carries the SAME translation vector as the
      posed variant of the dataset (translation-only diagnostic, 2026-10-03).
    * ``symmetric_tensor_fields``: triplets ``[xx, yy, xy]`` of SCALAR field
      names (point or cell data) that are the in-plane components of a
      symmetric rank-2 tensor stored component by component (Tensile2d's
      ``sig11, sig22, sig12``). They are not scalars under rotation; the
      transform applies ``sigma' = R sigma R^T`` to them. Only defined for
      in-plane rotations (``axis`` set or 2-D points).
    """

    def __init__(
        self,
        salt: int = 0,
        key_field: str = "case_key",
        translate: float = 0.0,
        axis: int | None = None,
        symmetric_tensor_fields: Sequence[Sequence[str]] = (),
        rotate: bool = True,
    ) -> None:
        super().__init__()
        self.salt = int(salt)
        self.key_field = key_field
        self.rotate = bool(rotate)
        self.translate = float(translate)
        if self.translate < 0:
            raise ValueError(f"translate must be >= 0, got {translate}")
        self.axis = None if axis is None else int(axis)
        self.symmetric_tensor_fields = [
            tuple(str(n) for n in t) for t in symmetric_tensor_fields
        ]
        for t in self.symmetric_tensor_fields:
            if len(t) != 3:
                raise ValueError(
                    f"symmetric_tensor_fields entries are [xx, yy, xy] name triplets, got {t!r}"
                )

    def _pose(
        self, global_data: TensorDict, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The case's rotation matrix and translation vector (``None`` when ``translate == 0``)."""
        if self.key_field not in global_data.keys():
            raise KeyError(
                f"FixedRandomPose needs global_data[{self.key_field!r}]; enable store_case_key on "
                "MeshReaderWithGlobalData or read the DomainMesh with CaseKeyDomainMeshReader"
            )
        d = points.shape[-1]
        key = int(global_data[self.key_field])
        # The draws are made on a CPU generator (device-independent stream); the results
        # move to the mesh's device, since the pipeline may run its transforms on GPU.
        g = _pose_generator(key, self.salt)
        R = _pose_rotation_for_key(key, self.salt, d, self.axis, generator=g)
        if not self.rotate:
            R = torch.eye(
                d, dtype=R.dtype
            )  # the draw above was still consumed: same translation as the posed variant
        R = R.to(device=points.device, dtype=points.dtype)
        if self.translate == 0.0:
            return R, None
        u = _pose_translation_direction(g, d, self.axis if d == 3 else None)
        p = points.to(torch.float64)
        rms = (p - p.mean(dim=0)).norm(dim=-1).square().mean().sqrt()
        t = (self.translate * rms.cpu()) * u
        return R, t.to(device=points.device, dtype=points.dtype)

    def _in_plane_cos_sin(self, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if R.shape[-1] == 2:
            return R[0, 0], R[1, 0]
        if self.axis is None:
            raise ValueError(
                "FixedRandomPose: symmetric_tensor_fields are in-plane components and need an in-plane "
                "rotation (set `axis` for 3-D points)"
            )
        i, j = [k for k in range(3) if k != self.axis]
        return R[i, i], R[j, i]

    def _rotate_tensor_components(self, mesh: Mesh, R: torch.Tensor) -> Mesh:
        if not self.symmetric_tensor_fields:
            return mesh
        c, s = self._in_plane_cos_sin(R)
        new = {}
        for association in ("point_data", "cell_data"):
            data = getattr(mesh, association)
            hit = [
                t
                for t in self.symmetric_tensor_fields
                if all(n in data.keys() for n in t)
            ]
            if not hit:
                continue
            data = data.clone()
            for xx, yy, xy in hit:
                sxx, syy, sxy = data[xx], data[yy], data[xy]
                data[xx] = c * c * sxx + s * s * syy - 2 * c * s * sxy
                data[yy] = s * s * sxx + c * c * syy + 2 * c * s * sxy
                data[xy] = c * s * (sxx - syy) + (c * c - s * s) * sxy
            new[association] = data
        return mesh.with_data(**new) if new else mesh

    def _transform_mesh(
        self, mesh: Mesh, R: torch.Tensor, t: torch.Tensor | None
    ) -> Mesh:
        d = mesh.n_spatial_dims
        out = mesh.transform(
            R,
            transform_point_data=_spatial_field_mask(mesh.point_data, d),
            transform_cell_data=_spatial_field_mask(mesh.cell_data, d),
            transform_global_data=_spatial_field_mask(mesh.global_data, d),
            assume_invertible=True,
        )
        out = self._rotate_tensor_components(out, R)
        return out if t is None else out.translate(t)

    def __call__(self, mesh: Mesh) -> Mesh:
        R, t = self._pose(mesh.global_data, mesh.points)
        return self._transform_mesh(mesh, R, t)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Apply the sample-specific fixed pose to every sub-mesh and the domain's global vectors."""
        from physicsnemo.mesh.transformations.geometric import (
            _normalize_transform_mask,
            _transform_tensordict,
        )

        gd = (
            domain.global_data
            if self.key_field in domain.global_data.keys()
            else domain.interior.global_data
        )
        R, t = self._pose(gd, domain.interior.points)
        d = domain.interior.n_spatial_dims
        global_data = domain.global_data.clone()
        _transform_tensordict(
            global_data,
            R,
            d,
            "global_data",
            mask=_normalize_transform_mask(_spatial_field_mask(global_data, d)),
        )
        return DomainMesh(
            interior=self._transform_mesh(domain.interior, R, t),
            boundaries={
                name: self._transform_mesh(b, R, t)
                for name, b in domain.boundaries.items()
            },
            global_data=global_data,
        )

    def extra_repr(self) -> str:
        return (
            f"salt={self.salt}, key_field={self.key_field!r}, translate={self.translate}, rotate={self.rotate}, "
            f"axis={self.axis}, symmetric_tensor_fields={self.symmetric_tensor_fields}"
        )


@register()
class CaseKeyDomainMeshReader(DomainMeshReader):
    """``DomainMeshReader`` that stores the stable per-case integer ``case_key`` in ``global_data``.

    The DomainMesh twin of ``MeshReaderWithGlobalData(store_case_key=True)``
    (CHAR-PROTOCOLS, 2026-10-02): the key is the CRC-32 of the case directory
    name (the first path component below the reader root, e.g. ``run_1``,
    ``sample_000123``), so the surface and volume readings of one case share
    a key, and :class:`FixedRandomPose` draws the same pose for a case on
    every load regardless of worker, epoch or shuffle order. 0-dim fields are
    invariant under ``DomainMesh.transform``. Everything else is the stock
    reader.
    """

    def _case_key(self, index: int) -> torch.Tensor:
        sample_path = self._paths[index]
        try:
            name = sample_path.relative_to(self._root).parts[0]
        except ValueError:
            name = sample_path.name
        return torch.tensor(zlib.crc32(name.encode()) & 0x7FFFFFFF, dtype=torch.int64)

    def _load_sample(self, index: int) -> DomainMesh:
        dm = super()._load_sample(index)
        if "case_key" in dm.global_data.keys():
            raise ValueError("global_data already has a 'case_key' field")
        gd = dm.global_data.clone()
        gd["case_key"] = self._case_key(index)
        return DomainMesh(
            interior=dm.interior, boundaries=dm.boundaries, global_data=gd
        )


@register()
class SetConstantCellField(MeshTransform):
    r"""Write a constant per-cell scalar field (a per-case CONDITION input).

    ``cell_data[field_name] = value * ones(n_cells, 1)``.  The value is a
    property of the dataset the case comes from, so it is set per dataset
    YAML (e.g. ``0.0`` in the DrivAerML pipeline, ``1.0`` in the SHIFT-SUV
    pipelines) and rides along into ``boundaries.<name>.cell_data`` like
    ``normals``.  Transfer-program campaign C (T1, complete conditioning):
    lets ISLA read it as a rank-0 boundary input
    (``boundary_schemas: {vehicle: {cond: {rank: 0}}}``, read from the boundary
    mesh's ``cell_data``) and
    GeoTransolver / Transolver as an extra per-point functional channel
    (append the field to ``forward_kwargs.local_embedding`` and raise
    ``functional_dim`` by one).  A scalar is an invariant, so every ISLA
    covariance contract is untouched.  Place before ``MeshToDomainMesh``.
    """

    def __init__(self, field_name: str = "cond", value: float = 0.0) -> None:
        super().__init__()
        self._field_name = str(field_name)
        self._value = float(value)

    def __call__(self, mesh: Mesh) -> Mesh:
        n = mesh.n_cells
        if n == 0:
            raise ValueError(
                "SetConstantCellField: the mesh has no cells; place the "
                "transform before MeshToDomainMesh on a cell-carrying surface."
            )
        new_cd = mesh.cell_data.clone()
        new_cd[self._field_name] = torch.full(
            (n, 1), self._value, dtype=mesh.points.dtype, device=mesh.points.device
        )
        return mesh.with_data(cell_data=new_cd)

    def extra_repr(self) -> str:
        return f"cell_data[{self._field_name!r}] = {self._value}"


@register()
class StackScalarFields(MeshTransform):
    r"""Stack named scalar fields into one vector field, zero-padded to ``n_components``.

    ``fields: [bc_disp_x, bc_disp_y]`` (each ``(N,)`` or ``(N, 1)``) becomes
    ``output_field`` of shape ``(N, 3)`` with a zero third component: how a planar
    dataset that stored a boundary vector component by component presents it as
    ONE rank-1 field to a model with tensor-valued inputs (a field of one of ISLA's
    ``boundary_schemas``). ``association`` is ``"cell_data"`` or ``"point_data"``.
    The source fields are kept.
    """

    def __init__(
        self,
        fields: Sequence[str],
        output_field: str,
        association: Literal["cell_data", "point_data"] = "cell_data",
        n_components: int = 3,
    ) -> None:
        super().__init__()
        if association not in ("cell_data", "point_data"):
            raise ValueError(
                f"association must be 'cell_data' or 'point_data', got {association!r}"
            )
        self._fields = [str(f) for f in fields]
        if not self._fields or len(self._fields) > int(n_components):
            raise ValueError(
                f"StackScalarFields: need 1 to {n_components} fields, got {self._fields!r}"
            )
        self._output_field = str(output_field)
        self._association = association
        self._n_components = int(n_components)

    def __call__(self, mesh: Mesh) -> Mesh:
        data = getattr(mesh, self._association)
        columns = []
        for name in self._fields:
            if name not in data:
                raise KeyError(
                    f"StackScalarFields: {self._association}[{name!r}] not found "
                    f"(available: {sorted(data.keys())!r})"
                )
            columns.append(data[name].reshape(-1))
        stacked = torch.stack(columns, dim=-1)
        if stacked.shape[-1] < self._n_components:
            pad = stacked.new_zeros(
                stacked.shape[0], self._n_components - stacked.shape[-1]
            )
            stacked = torch.cat([stacked, pad], dim=-1)
        new_data = data.clone()
        new_data[self._output_field] = stacked
        return mesh.with_data(**{self._association: new_data})

    def _has_fields(self, mesh: Mesh) -> bool:
        data = getattr(mesh, self._association)
        return all(name in data for name in self._fields)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Stack on every sub-mesh that carries the fields (a boundary, typically);
        leave the others untouched; raise if no sub-mesh has them."""
        hit = False
        interior = domain.interior
        if self._has_fields(interior):
            interior = self(interior)
            hit = True
        boundaries = {}
        for name, bnd in domain.boundaries.items():
            if self._has_fields(bnd):
                boundaries[name] = self(bnd)
                hit = True
            else:
                boundaries[name] = bnd
        if not hit:
            raise KeyError(
                f"StackScalarFields: no sub-mesh carries {self._association} fields {self._fields!r}"
            )
        return DomainMesh(
            interior=interior, boundaries=boundaries, global_data=domain.global_data
        )

    def extra_repr(self) -> str:
        return (
            f"{self._association}[{self._output_field!r}] = stack({self._fields}, "
            f"pad to {self._n_components})"
        )


@register()
class SdfBiasedSubsampleInteriorPoints(MeshTransform):
    r"""Poisson-subsample the *interior point cloud* with an inclusion
    probability that rises with the signed distance to the wall (far-wake
    oversampling arm, 2026-09-10).

    Motivation: on DrivAerML the interior query-token model's one remaining
    eddy-viscosity deficit against GeoTransolver-volume sits in the far wake
    beyond 0.4 body lengths, which holds about 2% of a uniform interior
    sample, and slice anchors follow the query mass. This transform draws
    the training queries from a larger uniform pool with per-point inclusion
    probability ``pi_i = min(1, c * w(sdf_i))``, where ``w`` is a step
    function of the signed distance over ``band_edges`` with values
    ``band_weights`` and ``c`` sets the expected kept count to
    ``n_points_expected`` (one renormalization pass restores the count lost
    to clamping, as in :class:`PoissonBiasedSubsampleMesh`). Inclusion
    probabilities are exact and stored per kept point as
    ``point_data[pi_field]`` for provenance, so an importance-weighted loss
    can be built from them; this transform does not itself reweight the loss
    (the recipe loss is a plain mean over queries), so the objective's
    spatial weighting shifts toward the far field by construction. Run it
    after the SDF transform. Bare ``Mesh`` inputs pass through unchanged.
    """

    def __init__(
        self,
        n_points_expected: int,
        band_edges: Sequence[float] = (0.05, 0.4),
        band_weights: Sequence[float] = (1.0, 3.0, 8.0),
        sdf_field: str = "sdf",
        pi_field: str = "inclusion_pi",
    ) -> None:
        super().__init__()
        if n_points_expected <= 0:
            raise ValueError("n_points_expected must be positive")
        edges = [float(e) for e in band_edges]
        weights = [float(w) for w in band_weights]
        if len(weights) != len(edges) + 1:
            raise ValueError("band_weights must have one more entry than band_edges")
        if any(e2 <= e1 for e1, e2 in zip(edges, edges[1:])) or any(
            w <= 0 for w in weights
        ):
            raise ValueError(
                "band_edges must increase and band_weights must be positive"
            )
        self.n_points_expected = int(n_points_expected)
        self.band_edges = tuple(edges)
        self.band_weights = tuple(weights)
        self.sdf_field = sdf_field
        self.pi_field = pi_field
        self._generator: torch.Generator | None = None

    def set_generator(self, generator: torch.Generator) -> None:
        """Set the random generator used for inclusion draws."""
        self._generator = generator

    def _weights(self, sdf: torch.Tensor) -> torch.Tensor:
        edges = torch.tensor(self.band_edges, dtype=sdf.dtype, device=sdf.device)
        band = torch.bucketize(sdf.abs(), edges)  # 0 .. len(edges)
        w = torch.tensor(self.band_weights, dtype=sdf.dtype, device=sdf.device)[band]
        return torch.where(torch.isfinite(sdf), w, torch.zeros_like(w))

    def _inclusion(self, sdf: torch.Tensor) -> torch.Tensor:
        w = self._weights(sdf)
        c = self.n_points_expected / w.sum()
        pi = (c * w).clamp(max=1.0)
        deficit = self.n_points_expected - pi.sum()
        if deficit > 0:
            free = pi < 1.0
            if bool(free.any()):
                pi[free] = (pi[free] * (1 + deficit / pi[free].sum())).clamp(max=1.0)
        return pi

    def __call__(
        self, mesh: Mesh
    ) -> Mesh:  # bare Mesh: identity (needs the DomainMesh interior)
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Subsample interior points and correct explicit quadrature measures."""
        interior = domain.interior
        n = interior.points.shape[0]
        if n <= self.n_points_expected:
            return domain
        if self.sdf_field not in interior.point_data.keys():
            raise KeyError(
                f"SdfBiasedSubsampleInteriorPoints: interior point_data lacks {self.sdf_field!r}; "
                "run the SDF transform first"
            )
        sdf = interior.point_data[self.sdf_field].reshape(n).to(torch.float32)
        pi = self._inclusion(sdf)
        generator = self._generator
        if generator is not None and generator.device != pi.device:
            generator = None
        keep = torch.rand(n, device=pi.device, generator=generator) < pi
        idx = keep.nonzero(as_tuple=True)[0]
        kept = interior.slice_points(idx)
        pd = kept.point_data.clone()
        pd[self.pi_field] = pi[idx].to(kept.points.dtype)[:, None]
        new_interior = Mesh(
            points=kept.points,
            cells=kept.cells,
            point_data=pd,
            global_data=interior.global_data,
        )
        if EFFECTIVE_MEASURE_KEY in new_interior.point_data:
            scale_measures(new_interior, 1.0 / pi[idx], association="points")
        return DomainMesh(
            interior=new_interior,
            boundaries=domain.boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return (
            f"n_points_expected={self.n_points_expected}, band_edges={self.band_edges}, "
            f"band_weights={self.band_weights}, sdf_field={self.sdf_field!r}"
        )


@register()
class ComposeQuadratureMeasure(MeshTransform):
    """Materialize the shared effective cell measure for model field lookup.

    Readers and samplers have already applied their corrections through the
    mesh measure API. This adapter only ensures the complete measure is stored
    under the shared key; it does not normalize it or apply another factor.
    Meshes without cells are unchanged, including point-cloud interiors.
    """

    def __call__(self, mesh: Mesh) -> Mesh:
        if mesh.n_cells == 0:
            return mesh
        from physicsnemo.mesh.calculus.measure import cell_measures, set_cell_measures

        ### A copy: without a stored measure cell_measures returns the cached
        ### cell areas, and a field aliasing the cache recompiles a compiled model
        ### whenever the aliasing differs between cases.
        set_cell_measures(mesh, cell_measures(mesh).to(mesh.points.dtype, copy=True))
        return mesh


@register()
class DropDegenerateCells(MeshTransform):
    r"""Drop cells whose measure is zero or non-finite (2026-10-02).

    The HiLift wing surfaces carry a few exactly degenerate triangles per case (53 to 137 of
    200,000 sampled cells, zero area to the bit, from repeated vertices): they have no
    quadrature weight and no normal, so the loss and metrics give them zero weight, and a
    model that reads the boundary as tokens would carry them as garbage. This transform reads
    every cell's measure through :func:`~physicsnemo.mesh.calculus.measure.cell_measures` (the
    stored effective measure when a sampler wrote one, else the geometric measure of the
    current coordinates in any manifold dimension: segment length, triangle area, tetrahedron
    volume) and slices away the cells whose measure is not finite and positive, together with
    their cell data; vertices are kept. Meshes without such cells pass through unchanged.
    Place it before the terminal conversion so it sees the coordinates the model will see.
    """

    def __call__(self, mesh: Mesh) -> Mesh:
        if mesh.n_cells == 0:
            return mesh
        m = cell_measures(mesh)
        keep = torch.isfinite(m) & (m > 0)
        if bool(keep.all()):
            return mesh
        return mesh.slice_cells(keep)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Drop the degenerate cells of every cell-carrying mesh of the domain."""
        return DomainMesh(
            interior=self(domain.interior)
            if domain.interior.n_cells
            else domain.interior,
            boundaries={name: self(b) for name, b in domain.boundaries.items()},
            global_data=domain.global_data,
        )


@register()
class SqueezeScalarFields(MeshTransform):
    r"""Drop the trailing singleton channel of named scalar fields, ``(N, 1)`` to ``(N,)`` (2026-10-02).

    Converted datasets store some per-point scalars as one-channel columns (AirfRANS keeps its
    signed distance as ``(N, 1)``); a model with a declared field grammar reads a rank-0 field as a
    plain ``(N,)`` tensor (ISLA's ``query_schema``, ``boundary_schemas``) and refuses the column. This
    transform squeezes exactly one trailing axis of length one from each named field and leaves
    everything else untouched; a field that is already ``(N,)`` passes through, and a field whose
    trailing axis is not of length one is refused (it is not a scalar).

    Args:
        fields: dotted field names, e.g. ``interior.point_data.sdf`` or
            ``boundaries.airfoil.cell_data.bc_value``. Missing names are skipped so one list can
            serve several dataset variants.
    """

    def __init__(self, fields: Sequence[str]) -> None:
        super().__init__()
        self.fields = [str(f) for f in fields]

    @staticmethod
    def _squeeze(name: str, val: torch.Tensor) -> torch.Tensor:
        if val.ndim == 1:
            return val
        if val.ndim == 2 and val.shape[-1] == 1:
            return val[..., 0]
        raise ValueError(
            f"SqueezeScalarFields: {name} has shape {tuple(val.shape)}; a scalar field is (N,) or (N, 1)."
        )

    def _apply(self, mesh: Mesh, specs: list[tuple[str, str, str]]) -> Mesh:
        pd, cd = mesh.point_data.clone(), mesh.cell_data.clone()
        for assoc, key, full in specs:
            data = pd if assoc == "point_data" else cd
            if key in data.keys():
                data[key] = self._squeeze(full, data[key])
        return mesh.with_data(point_data=pd, cell_data=cd)

    def _split(self) -> tuple[list, dict[str, list]]:
        interior: list = []
        boundaries: dict[str, list] = {}
        for full in self.fields:
            parts = full.split(".")
            if (
                len(parts) == 3
                and parts[0] == "interior"
                and parts[1] in ("point_data", "cell_data")
            ):
                interior.append((parts[1], parts[2], full))
            elif (
                len(parts) == 4
                and parts[0] == "boundaries"
                and parts[2] in ("point_data", "cell_data")
            ):
                boundaries.setdefault(parts[1], []).append((parts[2], parts[3], full))
            else:
                raise ValueError(
                    f"SqueezeScalarFields: field {full!r} must be interior.<point_data|cell_data>.<name> or "
                    "boundaries.<boundary>.<point_data|cell_data>.<name>"
                )
        return interior, boundaries

    def __call__(self, mesh: Mesh) -> Mesh:
        ### A bare Mesh is treated as an interior: only the interior.* names apply.
        interior, _ = self._split()
        return self._apply(mesh, interior)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Squeeze the named interior and per-boundary fields."""
        interior_specs, boundary_specs = self._split()
        interior = self._apply(domain.interior, interior_specs)
        boundaries = {
            name: (
                self._apply(bnd, boundary_specs[name])
                if name in boundary_specs
                else bnd
            )
            for name, bnd in domain.boundaries.items()
        }
        return DomainMesh(
            interior=interior, boundaries=boundaries, global_data=domain.global_data
        )

    def extra_repr(self) -> str:
        return f"fields={self.fields}"


@register()
class SplitGlobalVector(MeshTransform):
    r"""Standardize a per-case global vector and split it into NAMED rank-0 global fields (2026-10-02).

    A converted dataset stores its per-case parameters as one vector in ``global_data`` (the lug's
    ``params`` = [hole radius, length, thickness, pressure], Tensile2d's six input scalars, DeepJEB++'s
    ``load_type`` one-hot). A model with a declared field grammar reads several independent scalars as several
    rank-0 names, never as an unnamed channel axis (ISLA's ``global_schema``); the tensor-input baselines
    read the vector whole as their global embedding. This transform serves both from one set of statistics:
    component ``i`` of ``global_data[field]`` becomes the leaf ``names[i]`` holding ``(x_i - mean_i) / std_i``
    (a 0-d tensor, a rank-0 field), and, when ``output_field`` is given, the standardized vector is written
    whole under that name. ``mean`` / ``std`` default to 0 / 1 (a one-hot tag is split without scaling); the
    raw field stays. Writing over an existing leaf is refused, so the raw per-case scalars a dataset also
    stores (the lug's ``pressure`` in Pa) are never silently replaced by their z-scores. Works on a bare Mesh
    (its ``global_data``, the surface pipelines before ``MeshToDomainMesh``) and on a DomainMesh (the
    domain-level ``global_data`` the recipe's ``forward_kwargs`` read).

    Example (GINOT lug; statistics over the train split from the converter manifest)::

        - _target_: ${dp:SplitGlobalVector}
          field: params
          names: [hole_radius_std, length_std, thickness_std, pressure_std]
          mean: [0.0124862, 0.100103, 0.0225110, 75300663.3]
          std: [0.00144079, 0.0115957, 0.00423982, 14360825.0]
          output_field: params_std
    """

    def __init__(
        self,
        field: str,
        names: Sequence[str],
        mean: Sequence[float] | float | None = None,
        std: Sequence[float] | float | None = None,
        output_field: str | None = None,
    ) -> None:
        super().__init__()
        self.field = str(field)
        self.names = [str(n) for n in names]
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError(
                f"SplitGlobalVector: names must be non-empty and distinct, got {self.names!r}"
            )
        self.output_field = None if output_field is None else str(output_field)
        n = len(self.names)
        self._mean = torch.as_tensor(
            0.0 if mean is None else mean, dtype=torch.float32
        ).reshape(-1)
        self._std = torch.as_tensor(
            1.0 if std is None else std, dtype=torch.float32
        ).reshape(-1)
        for label, t in (("mean", self._mean), ("std", self._std)):
            if t.numel() not in (1, n):
                raise ValueError(
                    f"SplitGlobalVector: {label} has {t.numel()} entries for {n} names {self.names!r}"
                )
        if bool((self._std <= 0).any()):
            raise ValueError(f"SplitGlobalVector: std must be positive, got {std!r}")
        self._mean = self._mean.expand(n).clone()
        self._std = self._std.expand(n).clone()

    def _split(self, gd: TensorDict | None) -> TensorDict:
        if gd is None or self.field not in gd.keys():
            available = [] if gd is None else sorted(str(k) for k in gd.keys())
            raise KeyError(
                f"SplitGlobalVector needs global_data[{self.field!r}]; available: {available!r}"
            )
        x = gd[self.field]
        if x.numel() != len(self.names):
            raise ValueError(
                f"SplitGlobalVector: global_data[{self.field!r}] has {x.numel()} entries "
                f"for {len(self.names)} names {self.names!r}"
            )
        taken = [
            n
            for n in (*self.names, *([self.output_field] if self.output_field else []))
            if n in gd.keys()
        ]
        if taken:
            raise ValueError(
                f"SplitGlobalVector: global_data already has {taken!r}; choose names that do not overwrite a leaf"
            )
        z = (x.reshape(-1).to(torch.float32) - self._mean.to(x.device)) / self._std.to(
            x.device
        )
        z = z.to(x.dtype if x.is_floating_point() else torch.float32)
        new_gd = gd.clone()
        for i, name in enumerate(self.names):
            new_gd[name] = z[i].clone()
        if self.output_field is not None:
            new_gd[self.output_field] = z
        return new_gd

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.with_data(global_data=self._split(mesh.global_data))

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Split the vector in the domain-level ``global_data``."""
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=self._split(domain.global_data),
        )

    def extra_repr(self) -> str:
        return f"field={self.field!r}, names={self.names}, output_field={self.output_field!r}"


@register()
class RestrictToPlane(MeshTransform):
    r"""Restrict a planar problem stored in three-dimensional coordinates to its plane (ND-GENERIC, 2026-09-28).

    Two-dimensional datasets (the OLBC square, Tensile2d, AirfRANS) are stored as ``DomainMesh`` files with a
    third coordinate that is constant (usually zero) on every point. This transform drops that axis from the
    points of the interior and of every boundary and from the named vector fields (normals, SDF normals,
    prescribed vectors), so the model sees ``d = 2`` components and runs its native two-dimensional path
    (``ISLA(dim=2)``) instead of a three-dimensional model on a plane. It refuses a mesh whose dropped coordinate
    is not constant within ``atol`` (the problem is then not planar). Cell centroids follow the points; scalar
    fields and cell connectivity are untouched. Place first in the pipeline so every later transform (normalization,
    field stacking) already works in the plane.

    Args:
        axis: the coordinate index to drop (default 2, the ``z`` of a problem in the ``xy`` plane).
        vector_fields: dotted field names whose trailing component is a spatial vector, e.g.
            ``interior.point_data.sdf_normals`` or ``boundaries.outer.cell_data.normals``. Missing names are
            skipped so one list can serve several dataset variants.
        global_fields: names of per-case vectors in ``global_data`` (``global_data.U_inf_dir`` or just
            ``U_inf_dir``) whose last axis is spatial and loses the dropped coordinate too, so a model's
            rank-1 global inputs carry ``d = 2`` components. Missing names are skipped.
        atol: tolerance for the constancy check of the dropped coordinate; ``None`` disables the check.
    """

    def __init__(
        self,
        axis: int = 2,
        vector_fields: Sequence[str] = (),
        global_fields: Sequence[str] = (),
        atol: float | None = 1e-6,
    ) -> None:
        super().__init__()
        self.axis = int(axis)
        self.vector_fields = [str(f) for f in vector_fields]
        self.global_fields = [
            str(f).removeprefix("global_data.") for f in global_fields
        ]
        self.atol = None if atol is None else float(atol)

    def _project_global(self, global_data):
        """Drop the axis from the named per-case vectors; scalars and unnamed entries pass through."""
        if not self.global_fields or global_data is None:
            return global_data
        gd = global_data.clone()
        for name in self.global_fields:
            if name in gd.keys():
                val = gd[name]
                gd[name] = val[..., self._keep(val.shape[-1])]
        return gd

    def _keep(self, n: int) -> list[int]:
        if not 0 <= self.axis < n:
            raise ValueError(
                f"RestrictToPlane: axis {self.axis} is not a coordinate of {n}-dimensional points."
            )
        return [i for i in range(n) if i != self.axis]

    def _check_planar(self, points: torch.Tensor, what: str) -> None:
        if self.atol is None or points.shape[0] == 0:
            return
        z = points[:, self.axis]
        spread = float((z.max() - z.min()).item())
        if spread > self.atol:
            raise ValueError(
                f"RestrictToPlane: coordinate {self.axis} of {what} spans {spread:.3e} > atol {self.atol:.1e}; "
                "the problem is not planar, so the axis cannot be dropped."
            )

    def _project_mesh(self, mesh: Mesh, what: str, vec: dict[str, set[str]]) -> Mesh:
        self._check_planar(mesh.points, what)
        keep = self._keep(mesh.n_spatial_dims)
        ### The named vector fields lose the axis by hand: ``project``'s own field
        ### detection (``transform_*_data=True``) raises on every non-scalar field
        ### whose leading component count differs from the spatial dimension, and
        ### recipe data carries ``(N, 1)`` and ``(N, C)`` scalar-channel fields.
        pd, cd = mesh.point_data.clone(), mesh.cell_data.clone()
        for name in vec.get("point_data", ()):
            if name in pd.keys():
                pd[name] = pd[name][..., keep]
        for name in vec.get("cell_data", ()):
            if name in cd.keys():
                cd[name] = cd[name][..., keep]
        return project(mesh.with_data(point_data=pd, cell_data=cd), keep_dims=keep)

    def _split_fields(
        self,
    ) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
        interior: dict[str, set[str]] = {}
        boundaries: dict[str, dict[str, set[str]]] = {}
        for f in self.vector_fields:
            parts = f.split(".")
            if parts[0] == "interior" and len(parts) == 3:
                interior.setdefault(parts[1], set()).add(parts[2])
            elif parts[0] == "boundaries" and len(parts) == 4:
                boundaries.setdefault(parts[1], {}).setdefault(parts[2], set()).add(
                    parts[3]
                )
            else:
                raise ValueError(
                    f"RestrictToPlane: vector field {f!r} must be 'interior.<point_data|cell_data>.<name>' or "
                    "'boundaries.<boundary>.<point_data|cell_data>.<name>'."
                )
        return interior, boundaries

    def __call__(self, mesh: Mesh) -> Mesh:
        interior, _ = self._split_fields()
        return self._project_mesh(mesh, "the mesh points", interior)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Drop the constant axis from every mesh, the named vector fields and global vectors."""
        interior, boundaries = self._split_fields()
        ### Not ``domain.apply_to_meshes``: the per-boundary vector-field lists
        ### are keyed by boundary name, which that callback does not receive.
        return DomainMesh(
            interior=self._project_mesh(
                domain.interior, "the interior points", interior
            ),
            boundaries={
                name: self._project_mesh(
                    b, f"boundary {name!r}", boundaries.get(name, {})
                )
                for name, b in domain.boundaries.items()
            },
            global_data=self._project_global(domain.global_data),
        )

    def extra_repr(self) -> str:
        return (
            f"axis={self.axis}, vector_fields={self.vector_fields}, "
            f"global_fields={self.global_fields}, atol={self.atol}"
        )


@register()
class FixedDrawDomainMeshReader(DomainMeshReader):
    r"""``DomainMeshReader`` whose per-case subsample is the same at every epoch (LOCAL-AGG stencil diagnostic, 2026-09-28).

    The base reader folds the epoch into each case's subsampling seed, so a point-cloud interior task (AirfRANS: 100,000 of
    ~177,000 nodes per airfoil) presents a different query draw every epoch. A local aggregation among tokens then sees a
    different neighbour stencil for the same physical point at every visit. This reader ignores ``set_epoch`` so each case
    keeps the draw of epoch 0 for the whole run; the draw still depends on the base seed and the case index. Training on a
    fixed draw removes that augmentation, so the diagnostic pairs an aggregation arm with a no-aggregation control on the
    same fixed draw and compares the ratio with the resampled pair. Test-time draws are unaffected (the evaluator sets its
    own generator).
    """

    def set_epoch(self, epoch: int) -> None:
        """Ignore the epoch, so every epoch reads the same subsample."""
        self._epoch = 0


@register()
class SplitLineCells(MeshTransform):
    r"""Split every line cell of a planar boundary into collinear equal pieces (AF-GEO refinement protocol, 2026-10-06).

    A count-raising re-panelization that keeps the geometry and the boundary measure exactly: only the panel pattern
    changes, which is the discretization change the AirfRANS airfoil (about 1,000 segments, never subsampled at the
    campaign budget) admits without losing information. Cells whose centroid x lies ahead of the mid-point of the
    boundary's x extent are split into ``pieces_front`` pieces, the others into ``pieces_back`` (equal values give a
    uniform refinement; 3 / 1 a front-loaded one). New points are the interior division points of each cell; cell data
    is copied to every piece and an explicit ``_effective_measure`` is divided by the piece count (the pieces are
    equal), so the represented measure is unchanged; point data of new points is interpolated linearly. Meshes without
    line cells (the interior point cloud, surface meshes) are returned unchanged. Two-dimensional boundaries only:
    apply after ``RestrictToPlane``.
    """

    def __init__(self, pieces_front: int = 2, pieces_back: int = 2) -> None:
        super().__init__()
        if int(pieces_front) < 1 or int(pieces_back) < 1:
            raise ValueError("pieces_front and pieces_back must be positive integers")
        self.pieces_front = int(pieces_front)
        self.pieces_back = int(pieces_back)

    def __call__(self, mesh: Mesh) -> Mesh:
        if mesh.cells is None or mesh.n_cells == 0 or mesh.cells.shape[1] != 2:
            return mesh
        cells, pts = mesh.cells, mesh.points
        device = pts.device
        cx = mesh.cell_centroids[:, 0]
        xmid = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
        k = torch.where(
            cx < xmid,
            torch.full_like(cells[:, 0], self.pieces_front),
            torch.full_like(cells[:, 0], self.pieces_back),
        )
        new_pts = [pts]
        new_cells = []
        parent = []
        interp = []  # (a, b, fraction) rows for the new points, in creation order
        n_pts = pts.shape[0]
        for kk in sorted(set(k.tolist())):
            sel = (k == kk).nonzero(as_tuple=True)[0]
            a, b = cells[sel, 0], cells[sel, 1]
            if kk == 1:
                new_cells.append(torch.stack([a, b], 1))
                parent.append(sel)
                continue
            fr = torch.arange(1, kk, dtype=pts.dtype, device=device) / kk  # (kk-1,)
            interior = (
                pts[a][:, None, :] + (pts[b] - pts[a])[:, None, :] * fr[None, :, None]
            )  # (M, kk-1, d)
            m = sel.numel()
            idx = n_pts + torch.arange(m * (kk - 1), device=device).reshape(m, kk - 1)
            n_pts += m * (kk - 1)
            new_pts.append(interior.reshape(-1, pts.shape[1]))
            interp.append(
                (a.repeat_interleave(kk - 1), b.repeat_interleave(kk - 1), fr.repeat(m))
            )
            chain = torch.cat([a[:, None], idx, b[:, None]], 1)  # (M, kk+1)
            new_cells.append(
                torch.stack([chain[:, :-1], chain[:, 1:]], -1).reshape(-1, 2)
            )
            parent.append(sel.repeat_interleave(kk))
        points = torch.cat(new_pts)
        cells_out = torch.cat(new_cells)
        parent = torch.cat(parent)
        cell_data = {}
        for key, value in mesh.cell_data.items():
            piece = value[parent]
            if key == EFFECTIVE_MEASURE_KEY:
                piece = piece / k[parent].to(piece.dtype)
            cell_data[key] = piece
        point_data = {}
        if len(mesh.point_data.keys()) and interp:
            ia = torch.cat([t[0] for t in interp])
            ib = torch.cat([t[1] for t in interp])
            f = torch.cat([t[2] for t in interp])
            for key, value in mesh.point_data.items():
                shape = (-1,) + (1,) * (value.dim() - 1)
                fv = f.reshape(shape).to(value.dtype)
                point_data[key] = torch.cat(
                    [value, value[ia] * (1 - fv) + value[ib] * fv]
                )
        elif len(mesh.point_data.keys()):
            point_data = {key: value for key, value in mesh.point_data.items()}
        return Mesh(
            points=points,
            cells=cells_out,
            point_data=point_data or None,
            cell_data=cell_data or None,
            global_data=mesh.global_data,
        )


def _global_leaf(global_data: TensorDict | None, name: str, who: str) -> torch.Tensor:
    """The named (possibly nested, ``"a.b"``) leaf of ``global_data``, or a KeyError naming the available leaves."""
    key = as_nested_key(name)
    if global_data is None or key not in global_data:
        available = (
            []
            if global_data is None
            else sorted(
                key_to_str(k)
                for k in global_data.keys(include_nested=True, leaves_only=True)
            )
        )
        raise KeyError(f"{who} needs global_data[{name!r}]; available: {available!r}")
    return global_data[key]


def _refuse_existing_globals(global_data: TensorDict, names: Sequence[str], who: str):
    taken = [n for n in names if as_nested_key(n) in global_data]
    if taken:
        raise ValueError(
            f"{who}: global_data already has {taken!r}; choose names that do not overwrite a leaf"
        )


@register()
class ComputeLogReynolds(MeshTransform):
    r"""Write the case's Reynolds number, as ``log10`` and standardized, into ``global_data`` (AeroZoo, 2026-10-09).

    ``global_data[output_field] = log10(U L / nu)`` from the reference speed, the reference length and the
    kinematic viscosity, and ``global_data[standardized_field] = (log10 Re - mean) / std``, the O(1) rank-0
    global input a model declares (the raw value stays for provenance and re-dimensionalization). The three
    source leaves are read by name; a ``"."`` addresses a nested leaf (``physics.kinematic_viscosity``). A
    dataset whose cases span decades of Reynolds number carries the information this way, as a scalar
    invariant next to the unit freestream direction of :class:`ComputeFreestreamDirection`, so nothing
    dimensional reaches the model. Domain-aware like that transform: on a ``DomainMesh`` the leaves are read
    from and written to the domain-level ``global_data``.
    """

    def __init__(
        self,
        speed_field: str,
        length_field: str,
        viscosity_field: str,
        mean: float,
        std: float,
        output_field: str = "log_re",
        standardized_field: str = "log_re_std",
    ) -> None:
        super().__init__()
        self.speed_field = str(speed_field)
        self.length_field = str(length_field)
        self.viscosity_field = str(viscosity_field)
        self.mean = float(mean)
        self.std = float(std)
        if self.std <= 0:
            raise ValueError(f"ComputeLogReynolds: std must be positive, got {std!r}")
        self.output_field = str(output_field)
        self.standardized_field = str(standardized_field)

    def _write(self, global_data: TensorDict) -> TensorDict:
        who = "ComputeLogReynolds"
        speed = _global_leaf(global_data, self.speed_field, who).float()
        length = _global_leaf(global_data, self.length_field, who).float()
        nu = _global_leaf(global_data, self.viscosity_field, who).float()
        reynolds = speed * length / nu
        if reynolds.numel() != 1 or not torch.isfinite(reynolds).all() or reynolds <= 0:
            raise ValueError(
                f"{who}: U L / nu must be one finite positive number, got {reynolds!r}"
            )
        log_re = torch.log10(reynolds.reshape(()))
        new_gd = global_data.clone()
        new_gd[self.output_field] = log_re
        new_gd[self.standardized_field] = (log_re - self.mean) / self.std
        return new_gd

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.with_data(global_data=self._write(mesh.global_data))

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Derive the Reynolds number from the domain-level ``global_data``."""
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=self._write(domain.global_data),
        )

    def extra_repr(self) -> str:
        return (
            f"{self.output_field} = log10({self.speed_field} * {self.length_field} / "
            f"{self.viscosity_field}), {self.standardized_field} = ({self.output_field} - "
            f"{self.mean}) / {self.std}"
        )


@register()
class HarmonizeBoundaryFields(MeshTransform):
    r"""Give every boundary of a ``DomainMesh`` the same ``cell_data`` fields (AeroZoo, 2026-10-09).

    A case whose boundaries are grouped by condition type stores on each only the values that type
    prescribes (the inlet its velocity and turbulence, the outlet its pressure and reverse-flow values, the
    wall nothing but zero velocity, a slip wall nothing). A model with one declared boundary grammar reads
    the same fields from every boundary mesh it is handed, so this transform writes, on every boundary
    (ISLA, which declares each condition's own fields by name, drops it):

    * a one-hot label per listed boundary, ``cell_data[field] = 1`` on the boundary of that name and ``0``
      on every other, shape ``(F,)``; and
    * every listed prescribed field a boundary lacks, as zeros of the field's rank (``(F,)`` for rank 0,
      ``(F, d)`` for rank 1, ``(F, d, d)`` for rank 2). A boundary that carries the field keeps its values,
      and their shape must match the declared rank (a ``(F, 1)`` column is not a rank-0 field).

    A boundary without faces (a case without a slip wall) receives the same fields with zero rows. Every
    boundary named in ``one_hot`` must exist; ``fields`` names may be nested (``prescribed.dirichlet.velocity``).
    Bare ``Mesh`` inputs pass through unchanged (there is no boundary name to label).

    Args:
        one_hot: ``{boundary_name: label_field}``, e.g. ``{inlet: bc_inlet, outlet: bc_outlet}``.
        fields: ``{cell_data field: rank}`` of the prescribed fields to equalize across the boundaries.
    """

    def __init__(self, one_hot: dict[str, str], fields: dict[str, int]) -> None:
        super().__init__()
        self.one_hot = {str(b): str(f) for b, f in one_hot.items()}
        self.fields = {str(name): int(rank) for name, rank in fields.items()}
        if len(set(self.one_hot.values())) != len(self.one_hot):
            raise ValueError(
                f"HarmonizeBoundaryFields: label fields must be distinct, got {self.one_hot!r}"
            )
        for name, rank in self.fields.items():
            if rank < 0:
                raise ValueError(
                    f"HarmonizeBoundaryFields: rank of {name!r} must be >= 0, got {rank}"
                )

    def _harmonize(self, name: str, mesh: Mesh) -> Mesh:
        n, d = mesh.n_cells, mesh.n_spatial_dims
        dtype, device = mesh.points.dtype, mesh.points.device
        cd = mesh.cell_data.clone()
        for boundary, label in self.one_hot.items():
            cd[label] = torch.full(
                (n,), float(boundary == name), dtype=dtype, device=device
            )
        for field, rank in self.fields.items():
            key = as_nested_key(field)
            shape = (n,) + (d,) * rank
            if key in cd:
                if tuple(cd[key].shape) != shape:
                    raise ValueError(
                        f"HarmonizeBoundaryFields: boundaries[{name!r}].cell_data[{field!r}] has shape "
                        f"{tuple(cd[key].shape)}; rank {rank} on {n} cells is {shape}"
                    )
            else:
                cd[key] = torch.zeros(shape, dtype=dtype, device=device)
        return mesh.with_data(cell_data=cd)

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Label and zero-fill every boundary of the domain."""
        missing = [b for b in self.one_hot if b not in domain.boundary_names]
        if missing:
            raise KeyError(
                f"HarmonizeBoundaryFields: boundaries {missing!r} not found; available: "
                f"{domain.boundary_names!r}"
            )
        return DomainMesh(
            interior=domain.interior,
            boundaries={
                name: self._harmonize(name, b) for name, b in domain.boundaries.items()
            },
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return f"one_hot={self.one_hot}, fields={self.fields}"


@register()
class BoundaryFieldToGlobal(MeshTransform):
    r"""Reduce ``cell_data`` fields of one boundary to per-case global leaves (AeroZoo, 2026-10-09).

    ``global_data[name] = sum_i m_i f_i / sum_i m_i`` over the cells of ``boundaries[boundary]``, with ``m``
    the effective cell measure (:func:`~physicsnemo.mesh.calculus.measure.cell_measures`: the cell area
    times the reader's Horvitz-Thompson subsample weight), so the leaf is the area average of the field over
    the whole boundary, independent of how the boundary was meshed or subsampled; a value prescribed
    uniformly over the boundary (an inlet's turbulence quantities) comes back exactly. A scalar ``(F,)``
    field gives a 0-d leaf (a rank-0 global input), a vector ``(F, d)`` field a ``(d,)`` leaf. Writing over an
    existing leaf is refused, as in :class:`SplitGlobalVector`. A boundary without cells gives zeros.
    Bare ``Mesh`` inputs pass through unchanged.

    Args:
        boundary: the boundary whose fields are reduced.
        fields: ``{cell_data field: global name}``; a ``"."`` in the field addresses a nested leaf.
    """

    def __init__(self, boundary: str, fields: dict[str, str]) -> None:
        super().__init__()
        self.boundary = str(boundary)
        self.fields = {str(f): str(g) for f, g in fields.items()}
        if not self.fields or len(set(self.fields.values())) != len(self.fields):
            raise ValueError(
                f"BoundaryFieldToGlobal: global names must be non-empty and distinct, got {self.fields!r}"
            )

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Write the measure-weighted means into the domain-level ``global_data``."""
        if self.boundary not in domain.boundary_names:
            raise KeyError(
                f"BoundaryFieldToGlobal: boundary {self.boundary!r} not found; available: "
                f"{domain.boundary_names!r}"
            )
        gd = domain.global_data
        _refuse_existing_globals(
            gd, list(self.fields.values()), "BoundaryFieldToGlobal"
        )
        mesh = domain.boundaries[self.boundary]
        measures = cell_measures(mesh).float()
        weights = measures / measures.sum().clamp_min(torch.finfo(torch.float32).tiny)
        new_gd = gd.clone()
        for field, name in self.fields.items():
            key = as_nested_key(field)
            if key not in mesh.cell_data:
                raise KeyError(
                    f"BoundaryFieldToGlobal: boundaries[{self.boundary!r}].cell_data lacks {field!r}"
                )
            values = mesh.cell_data[key].float()
            if values.shape[0] != mesh.n_cells or values.ndim not in (1, 2):
                raise ValueError(
                    f"BoundaryFieldToGlobal: {field!r} has shape {tuple(values.shape)}; a cell field is "
                    f"(F,) or (F, d) with F = {mesh.n_cells}"
                )
            w = weights if values.ndim == 1 else weights[:, None]
            new_gd[name] = (w * values).sum(dim=0)
        return DomainMesh(
            interior=domain.interior, boundaries=domain.boundaries, global_data=new_gd
        )

    def extra_repr(self) -> str:
        return f"boundary={self.boundary!r}, fields={self.fields}"


@register()
class ConcatGlobalFields(MeshTransform):
    r"""Concatenate named ``global_data`` leaves into one vector leaf (AeroZoo, 2026-10-09).

    The reverse direction of :class:`SplitGlobalVector`: the dataset carries, and a model with a declared
    field grammar reads, named global leaves (ISLA's ``global_schema``), while a tensor-input baseline reads
    one vector as its global embedding (GeoTransolver's ``global_embedding``). Each listed leaf is flattened
    and the pieces are concatenated in the listed order into ``global_data[output_field]``; the sources stay,
    and writing over an existing leaf is refused. Works on a bare ``Mesh`` and on the domain-level
    ``global_data`` of a ``DomainMesh``.
    """

    def __init__(self, fields: Sequence[str], output_field: str) -> None:
        super().__init__()
        self.fields = [str(f) for f in fields]
        if not self.fields:
            raise ValueError("ConcatGlobalFields: fields must not be empty")
        self.output_field = str(output_field)

    def _concat(self, global_data: TensorDict) -> TensorDict:
        _refuse_existing_globals(global_data, [self.output_field], "ConcatGlobalFields")
        pieces = [
            _global_leaf(global_data, name, "ConcatGlobalFields").float().reshape(-1)
            for name in self.fields
        ]
        new_gd = global_data.clone()
        new_gd[self.output_field] = torch.cat(pieces)
        return new_gd

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh.with_data(global_data=self._concat(mesh.global_data))

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Concatenate in the domain-level ``global_data``."""
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=self._concat(domain.global_data),
        )

    def extra_repr(self) -> str:
        return f"{self.output_field} = cat({self.fields})"


@register()
class FlipBoundaryOrientation(MeshTransform):
    r"""Reverse the orientation of named boundaries: cell winding and stored normals.

    A dataset's boundary cells carry their orientation in their winding (a cell's
    normal follows its vertex order), and the convention differs between datasets:
    DrivAerML's vehicle and AeroZoo's four boundaries are wound with their normals out
    of the fluid, AirfRANS's airfoil with its normals into it. A model that reads the
    normals, and signs its distances by them (ISLA: positive on the side the normals
    point to), sees one convention when this transform reverses the boundaries wound
    the other way, so the stored files stay as they are. For every boundary in
    ``boundaries`` the last two vertices of every cell are swapped (each normal
    reversed, the cell and its measure unchanged) and every ``cell_data`` field in
    ``normal_fields`` is negated (a stored normal; names a boundary lacks are
    skipped); the mesh's cached geometry is dropped with the old winding. Every named
    boundary must exist; bare ``Mesh`` inputs pass through unchanged.

    Args:
        boundaries: the names of the boundaries to reverse.
        normal_fields: ``cell_data`` fields holding stored normals, negated with the
            winding (default none).
    """

    def __init__(self, boundaries: Sequence[str], normal_fields: Sequence[str] = ()) -> None:
        super().__init__()
        self.boundaries = [str(b) for b in boundaries]
        self.normal_fields = [str(f) for f in normal_fields]
        if not self.boundaries:
            raise ValueError("FlipBoundaryOrientation: boundaries must not be empty")

    def _flip(self, mesh: Mesh) -> Mesh:
        d = mesh.cells.shape[-1]
        cells = mesh.cells[:, [*range(d - 2), d - 1, d - 2]]
        cell_data = mesh.cell_data.clone()
        for field in self.normal_fields:
            key = as_nested_key(field)
            if key in cell_data:
                cell_data[key] = -cell_data[key]
        return Mesh(
            points=mesh.points,
            cells=cells,
            point_data=mesh.point_data,
            cell_data=cell_data,
            global_data=mesh.global_data,
        )

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Reverse the named boundaries of the domain."""
        missing = [b for b in self.boundaries if b not in domain.boundary_names]
        if missing:
            raise KeyError(
                f"FlipBoundaryOrientation: boundaries {missing!r} not found; available: "
                f"{domain.boundary_names!r}"
            )
        return DomainMesh(
            interior=domain.interior,
            boundaries={
                name: self._flip(b) if name in self.boundaries else b
                for name, b in domain.boundaries.items()
            },
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return f"boundaries={self.boundaries}, normal_fields={self.normal_fields}"


@register()
class SplitBoundaryByLabels(MeshTransform):
    r"""Split one boundary of a ``DomainMesh`` into one boundary per boundary condition (2026-10-19).

    A dataset that stores several boundary conditions in one mesh tells them apart by 0/1 label
    fields in its ``cell_data`` (the lug's ``bc_fixed``, ``bc_hole`` and ``bc_loaded`` on one
    surface). ISLA declares its boundary conditions by name and reads each from its own mesh
    (``boundary_schemas``), so this transform adds, next to ``boundary``, one boundary per entry
    of ``labels``: the cells whose label field is nonzero, the first entry winning where two
    labels mark the same cell (list the more specific condition first, ``loaded`` before
    ``hole``), and, under the name ``rest``, the cells no label marks. Each new boundary is the
    source mesh's points with those cells and their ``cell_data`` without the label fields, so a
    condition's fields are its own cells' values; a condition no cell carries is a boundary with
    no cells. The source boundary is kept as it is, for the models that read it.

    With ``reorder_interior`` (a surface task, whose interior is the point cloud of the source
    boundary's cell centroids carrying the targets, as ``MeshToDomainMesh`` with
    ``interior_points: cell_centroids`` writes it) the interior points are permuted into the
    order of ISLA's surface predictions: the new boundaries in sorted name order, each its cells
    in mesh order. The scored points and their targets are the same, in that order.

    Args:
        boundary: the boundary to split.
        labels: ``{new boundary name: label field}``, in priority order.
        rest: the name of the boundary of the unlabelled cells; ``None`` requires every cell
            to carry a label.
        reorder_interior: permute the interior to the order of the new boundaries' cells.
    """

    def __init__(
        self,
        boundary: str,
        labels: dict[str, str],
        rest: str | None = None,
        reorder_interior: bool = False,
    ) -> None:
        super().__init__()
        self.boundary = str(boundary)
        self.labels = {str(name): str(field) for name, field in labels.items()}
        self.rest = None if rest is None else str(rest)
        self.reorder_interior = bool(reorder_interior)
        names = [*self.labels, *([self.rest] if self.rest is not None else [])]
        if not self.labels or len(set(names)) != len(names):
            raise ValueError(
                f"SplitBoundaryByLabels: labels must name distinct boundaries (rest included), got {names!r}"
            )

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Add the split boundaries (and reorder the interior when asked)."""
        if self.boundary not in domain.boundary_names:
            raise KeyError(
                f"SplitBoundaryByLabels: boundary {self.boundary!r} not found; available: "
                f"{domain.boundary_names!r}"
            )
        names = [*self.labels, *([self.rest] if self.rest is not None else [])]
        taken = [n for n in names if n in domain.boundary_names]
        if taken:
            raise ValueError(
                f"SplitBoundaryByLabels: boundaries {taken!r} exist already"
            )
        mesh = domain.boundaries[self.boundary]
        keys = [as_nested_key(field) for field in self.labels.values()]
        assigned = torch.zeros(mesh.n_cells, dtype=torch.bool, device=mesh.cells.device)
        index = {}
        for name, key in zip(self.labels, keys):
            hit = (mesh.cell_data[key] != 0) & ~assigned
            index[name] = torch.nonzero(hit).flatten()
            assigned |= hit
        if self.rest is not None:
            index[self.rest] = torch.nonzero(~assigned).flatten()
        elif not bool(assigned.all()):
            raise ValueError(
                f"SplitBoundaryByLabels: {int((~assigned).sum())} cells of {self.boundary!r} carry "
                "no label; name their boundary with `rest`"
            )
        split = {}
        for name, idx in index.items():
            part = mesh.slice_cells(idx)
            split[name] = part.with_data(cell_data=part.cell_data.exclude(*keys))
        interior = domain.interior
        if self.reorder_interior:
            if interior.n_points != mesh.n_cells:
                raise ValueError(
                    f"SplitBoundaryByLabels: reorder_interior needs the interior to be the cell "
                    f"centroids of {self.boundary!r} ({mesh.n_cells} cells); it has "
                    f"{interior.n_points} points"
                )
            interior = interior.slice_points(
                torch.cat([index[n] for n in sorted(index)])
            )
        return DomainMesh(
            interior=interior,
            boundaries={**domain.boundaries, **split},
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return (
            f"boundary={self.boundary!r}, labels={self.labels}, rest={self.rest!r}, "
            f"reorder_interior={self.reorder_interior}"
        )
