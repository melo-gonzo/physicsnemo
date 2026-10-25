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
Reader extension that merges an external on-disk ``TensorDict`` into each
sample's ``global_data`` at load time.

This recipe's surface configs read a boundary :class:`Mesh` directly out
of a parent :class:`~physicsnemo.mesh.DomainMesh`'s on-disk tensordict
tree (e.g. ``.../*.pdmsh/_tensordict/boundaries/vehicle``).  The boundary
tensordict's own ``global_data`` carries little or nothing of the
case-level metadata -- the freestream conditions (``U_inf``,
``rho_inf``, ``p_inf``, ``T_inf``, ``L_ref``) live on the *domain-level*
``_tensordict/global_data`` instead.  :class:`MeshReaderWithGlobalData`
walks a configurable relative path from each sample directory, loads
the tensordict it points at, and merges its keys into the boundary
``Mesh``'s ``global_data`` so downstream transforms
(:class:`NonDimensionalizeByMetadata`, ``RandomRotateMesh`` with
``transform_global_data: true``, ...) can read them straight off the
loaded sample.

With ``domain_boundaries`` the reader also loads the other boundaries of
the PDE domain that a case file stores beside the surface (DrivAerML's
tunnel: inlet, no_slip, slip, outlet), so that a model can read the whole
boundary while it predicts on the surface; :class:`SurfaceMeshToDomainMesh`
is that sample's terminal step.

Recipe-local module registered into the global datapipe component
registry so the classes can be referenced via
``${dp:MeshReaderWithGlobalData}`` in Hydra YAML configs.  Import this
module before Hydra instantiation.
"""

from __future__ import annotations

import inspect
import logging
import zlib
from collections.abc import Sequence
from typing import Any

import torch
from dedup import ComputedDuplicates, DuplicateMask
from tensordict import TensorDict

from physicsnemo.datapipes._indexing import _cyclic_block_indices
from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.datapipes.keys import key_to_str, leaf_keys
from physicsnemo.datapipes.readers.mesh import MeshReader, _subsample_mesh_points
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh import MeshToDomainMesh
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import scale_measures

logger = logging.getLogger(__name__)


@register()
class MeshReaderWithGlobalData(MeshReader):
    r"""MeshReader that merges an external tensordict into each Mesh's ``global_data``.

    Identical to :class:`MeshReader` except for the
    ``merge_global_data_from`` parameter, which names a path
    *relative to each matched sample* pointing to a saved
    ``TensorDict`` directory whose keys should be merged into the
    loaded :class:`Mesh`'s ``global_data``.  If a key appears in
    both the boundary's ``global_data`` and the external one, a
    :class:`ValueError` is raised: ``global_data`` is case-level
    by construction, so overlap is ambiguous and treated as a
    data-layer bug rather than silently resolved.

    This is intended for boundary tensordicts whose own
    ``global_data`` is empty (or only carries dataset-local fields
    like ``TimeValue``) but whose parent
    :class:`~physicsnemo.mesh.DomainMesh` carries the case-level
    freestream conditions (``U_inf``, ``rho_inf``, ``p_inf``, ...).
    Without this hook the boundary-only surface pipeline cannot see
    those parent fields.

    Example YAML::

        reader:
          _target_: ${dp:MeshReaderWithGlobalData}
          path: ${train_datadir}
          pattern: "run_*/*.pdmsh/_tensordict/boundaries/vehicle"
          # Walk up to the parent DomainMesh's global_data tensordict
          merge_global_data_from: "../../global_data"

    Parameters
    ----------
    merge_global_data_from : str or None
        Path *relative to each matched sample directory* of an
        on-disk ``TensorDict`` directory (one containing
        ``meta.json`` plus ``*.memmap`` files).  When ``None``
        (default) this class behaves exactly like
        :class:`MeshReader`.
    domain_boundaries : sequence of str
        Names of the other boundaries of the PDE domain stored beside the
        matched surface (siblings of its directory, ``.../boundaries/<name>``),
        loaded whole, never subsampled.  With any name the sample is a
        :class:`~physicsnemo.mesh.DomainMesh` whose interior is the surface
        itself (subsampled and with its targets, as without them) until the
        terminal :class:`SurfaceMeshToDomainMesh`, whose boundaries are these
        meshes and whose ``global_data`` is the case's.  So every transform of
        the chain (rotation, translation, centering, scaling) acts on all of
        them through ``apply_to_domain`` while the surface's draw, centre and
        pose stay those of the bare surface.  Empty (default): a ``Mesh``.
    whole_surface_transforms : sequence of MeshTransform
        Transforms that place or add domain boundaries from the surface at
        full resolution, before its subsample (DrivAerML's
        ``PlaceGroundUnderBody``: the ground through the whole vehicle's lowest
        point, which a subsample can miss; HiLift's ``AddSymmetryPlane``, which
        checks every stored root vertex).  Each is applied, in the files' raw
        coordinates, to a :class:`~physicsnemo.mesh.DomainMesh` whose
        boundaries are the whole surface (under its directory's name, read from
        its memory map) and the domain boundaries; a transform whose
        ``apply_to_domain`` declares a ``case`` parameter also gets the case
        name (the first path component below the reader root).  It may move
        the domain boundaries and add boundaries under new names; one that
        replaces the surface or drops a boundary is refused.  The boundaries it
        adds join the sample's, so with them the sample is a
        :class:`~physicsnemo.mesh.DomainMesh` even without ``domain_boundaries``.
    drop_duplicate_cells : bool, default False
        Drop every extra copy of a cell (a cell whose sorted vertex triple
        repeats a lower-indexed cell's; see :mod:`dedup`) before the model
        sees the mesh. With ``subsample_n_cells`` the cyclic block is drawn
        over all stored cells exactly as :class:`MeshReader` draws it, the
        drawn copies are then dropped, and the kept cells keep the factor
        ``N / k``: each first copy still has inclusion probability ``k / N``,
        so measure-weighted sums estimate the deduplicated mesh without bias,
        on slightly fewer than ``k`` cells.
    duplicate_cell_masks : str or None
        Directory of cached masks (``<case>.npy`` and ``<case>.json``,
        ``tools/build_duplicate_masks.py``), checked against each case's
        connectivity fingerprint at every load; a mismatch is an error.
        ``None`` computes the copies at load (exact, about 4 minutes for a
        HiLift wing).
    **kwargs
        Forwarded to :class:`MeshReader`.
    """

    def __init__(
        self,
        *args: Any,
        merge_global_data_from: str | None = None,
        store_case_key: bool = False,
        domain_boundaries: Sequence[str] = (),
        whole_surface_transforms: Sequence[Any] = (),
        drop_duplicate_cells: bool = False,
        duplicate_cell_masks: str | None = None,
        **kwargs: Any,
    ) -> None:
        if duplicate_cell_masks is not None and not drop_duplicate_cells:
            raise ValueError(
                "duplicate_cell_masks is set but drop_duplicate_cells is false"
            )
        super().__init__(*args, **kwargs)
        self._merge_rel_path = merge_global_data_from
        self._domain_boundaries = [str(n) for n in domain_boundaries]
        if len(set(self._domain_boundaries)) != len(self._domain_boundaries):
            raise ValueError(
                f"domain_boundaries names a boundary twice: {self._domain_boundaries!r}"
            )
        self._whole_surface_transforms = list(whole_surface_transforms)
        self._drop_duplicates = bool(drop_duplicate_cells)
        self._duplicate_masks = duplicate_cell_masks
        self._duplicate_logged = 0
        ### POSE-BENCH (transfer program, 2026-09-09): a stable per-case integer,
        ### the CRC-32 of the case directory name (the first path component
        ### below the reader root, e.g. ``run_1``), stored as the 0-dim int64
        ### global field ``case_key``. Deterministic transforms such as
        ### ``FixedRandomPose`` seed from it, so every load of a case sees the
        ### same draw regardless of worker, epoch or shuffle order. 0-dim
        ### fields are invariant under ``Mesh.transform``.
        self._store_case_key = bool(store_case_key)

    def _case_name(self, index: int) -> str:
        sample_path = self._paths[index]
        try:
            return sample_path.relative_to(self._root).parts[0]
        except ValueError:
            return sample_path.name

    def _case_key(self, index: int) -> torch.Tensor:
        name = self._case_name(index)
        return torch.tensor(zlib.crc32(name.encode()) & 0x7FFFFFFF, dtype=torch.int64)

    def _load_sample(self, index: int) -> Mesh:
        mesh = super()._load_sample(index)
        if self._store_case_key:
            if "case_key" in mesh.global_data.keys():
                raise ValueError("global_data already has a 'case_key' field")
            gd = mesh.global_data.clone()
            gd["case_key"] = self._case_key(index)
            mesh = mesh.with_data(global_data=gd)
        if self._merge_rel_path is None:
            return mesh

        sample_path = self._paths[index]
        ### Resolve ``../...`` against the sample directory itself; the
        ### result must already exist on disk (we don't lazily create).
        ext_path = (sample_path / self._merge_rel_path).resolve()
        if not ext_path.exists():
            raise FileNotFoundError(
                f"merge_global_data_from path not found: {ext_path} "
                f"(resolved from sample {sample_path} + "
                f"{self._merge_rel_path!r})"
            )

        ext_td = TensorDict.load_memmap(ext_path, mode="r")  # copy-on-write
        merged = mesh.global_data.clone()

        ### Compare leaf key *paths* (nested included): two groups sharing a
        ### name but holding different leaves are not a collision (``update``
        ### merges them recursively), but a leaf on one side whose path is a
        ### prefix of, or equal to, a leaf path on the other would be
        ### overwritten by ``update``, so that is.
        def _paths(td: TensorDict) -> list[tuple[str, ...]]:
            return [(k,) if isinstance(k, str) else k for k in leaf_keys(td)]

        ext_paths, merged_paths = _paths(ext_td), _paths(merged)
        collisions = sorted(
            {
                key_to_str(a if len(a) <= len(b) else b)
                for a in ext_paths
                for b in merged_paths
                if a[: len(b)] == b or b[: len(a)] == a
            }
        )
        if collisions:
            raise ValueError(
                f"global_data key collision while merging {ext_path} "
                f"into sample {sample_path}: keys {collisions} are "
                f"present on both the boundary tensordict and the "
                f"external one. global_data is case-level by "
                f"definition, so an overlapping key is ambiguous and "
                f"indicates inconsistent metadata at the data layer."
            )
        merged.update(ext_td)

        return mesh.with_data(global_data=merged)

    def _duplicates(self, index: int, mesh: Mesh):
        sample_path = self._paths[index]
        if (sample_path / "zarr.json").exists():
            raise NotImplementedError(
                "drop_duplicate_cells reads memmap meshes only, not zarr stores"
            )
        if self._duplicate_masks is None:
            return ComputedDuplicates(mesh.cells)
        case = sample_path.relative_to(self._root).parts[0]
        return DuplicateMask(self._duplicate_masks, case, mesh.cells)

    def _read_surface(self, index: int) -> tuple[Mesh, dict[str, Any]]:
        """The surface sample of :class:`MeshReader`, without the extra copies of
        its cells under ``drop_duplicate_cells``."""
        if not self._drop_duplicates:
            return super().__getitem__(index)
        mesh = self._load_sample(index)
        duplicates = self._duplicates(index, mesh)
        generator = (
            None
            if self._seed_base is None
            else spawn_generator(self._seed_base, self._epoch, index)
        )
        n_stored = mesh.n_cells
        mesh = drop_duplicates_and_subsample(
            mesh, duplicates, self.subsample_n_cells, generator=generator
        )
        if self._duplicate_logged < 3:
            self._duplicate_logged += 1
            logger.info(
                f"drop_duplicate_cells: {self._paths[index]} fingerprint checked; {duplicates.n_extra} of "
                f"{n_stored} stored cells are extra copies; {mesh.n_cells} cells kept"
            )
        if self.subsample_n_points is not None:
            mesh = _subsample_mesh_points(
                mesh, self.subsample_n_points, generator=generator
            )
        if self.pin_memory:
            mesh = mesh.pin_memory()
        metadata = self._get_sample_metadata(index)
        if self.include_index_in_metadata:
            metadata["index"] = index
        return mesh, metadata

    @staticmethod
    def _load_whole(path) -> Mesh:
        """A boundary at full resolution (a copy-on-write memory map for a pmsh)."""
        if (path / "zarr.json").exists():
            from physicsnemo.mesh.io import from_zarr

            return from_zarr(path)
        return Mesh.load(path, mode="r")

    def _load_domain_boundary(self, index: int, name: str) -> Mesh:
        path = self._paths[index].parent / name
        if not path.exists():
            raise FileNotFoundError(
                f"domain boundary {name!r} not found beside {self._paths[index]} (looked for {path})"
            )
        return self._load_whole(path)

    def _place_from_whole_surface(
        self, index: int, boundaries: dict[str, Mesh]
    ) -> dict[str, Mesh]:
        """The domain boundaries after ``whole_surface_transforms``, which see the
        surface at full resolution beside them, with the boundaries they add."""
        path = self._paths[index]
        name = path.name
        if name in boundaries:
            raise ValueError(
                f"domain_boundaries names {name!r}, the surface's own name"
            )
        whole = self._load_whole(path)
        domain = DomainMesh(
            interior=Mesh(points=whole.points[:0]),
            boundaries={name: whole, **boundaries},
        )
        case = self._case_name(index)
        for t in self._whole_surface_transforms:
            before = set(domain.boundary_names)
            if "case" in inspect.signature(t.apply_to_domain).parameters:
                domain = t.apply_to_domain(domain, case=case)
            else:
                domain = t.apply_to_domain(domain)
            if (
                name not in domain.boundary_names
                or domain.boundaries[name] is not whole
            ):
                raise ValueError(
                    f"whole_surface_transforms: {t!r} changed the surface {name!r}; they may "
                    "move only the domain boundaries and add new ones"
                )
            if not before <= set(domain.boundary_names):
                raise ValueError(
                    f"whole_surface_transforms: {t!r} dropped the boundaries "
                    f"{sorted(before - set(domain.boundary_names))!r}"
                )
        return {n: domain.boundaries[n] for n in domain.boundary_names if n != name}

    def __getitem__(self, index: int) -> tuple[Mesh | DomainMesh, dict[str, Any]]:
        mesh, metadata = self._read_surface(index)
        if not self._domain_boundaries and not self._whole_surface_transforms:
            return mesh, metadata
        boundaries = {
            name: self._load_domain_boundary(index, name)
            for name in self._domain_boundaries
        }
        if self._whole_surface_transforms:
            boundaries = self._place_from_whole_surface(index, boundaries)
        if self.pin_memory:
            boundaries = {name: b.pin_memory() for name, b in boundaries.items()}
        domain = DomainMesh(
            interior=mesh.with_data(
                global_data=TensorDict({}, batch_size=[], device=mesh.points.device)
            ),
            boundaries=boundaries,
            global_data=mesh.global_data,
        )
        metadata["boundary_names"] = domain.boundary_names
        return domain, metadata


@register()
class SurfaceMeshToDomainMesh(MeshToDomainMesh):
    r""":class:`MeshToDomainMesh` for a surface read with the domain's other boundaries.

    On a bare :class:`Mesh` it is :class:`MeshToDomainMesh`. On the
    :class:`DomainMesh` of :class:`MeshReaderWithGlobalData` with
    ``domain_boundaries``, whose interior is the surface, it applies
    :class:`MeshToDomainMesh` to the surface (the interior becomes its cell
    centroids with the targets, the surface the boundary ``boundary_name``) and
    keeps the other boundaries and the domain-level ``global_data``. A domain
    boundary named ``boundary_name`` is refused.
    """

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:  # type: ignore[override]
        """The surface (``domain.interior``) to the centroid interior and the boundary
        ``boundary_name``, beside the domain's other boundaries."""
        if self._boundary_name in domain.boundary_names:
            raise ValueError(
                f"SurfaceMeshToDomainMesh: the domain already has a boundary {self._boundary_name!r}, "
                "the name the surface takes"
            )
        surface = self(domain.interior)
        return DomainMesh(
            interior=surface.interior,
            boundaries={
                self._boundary_name: surface.boundaries[self._boundary_name],
                **{name: domain.boundaries[name] for name in domain.boundary_names},
            },
            global_data=domain.global_data,
        )


def drop_duplicates_and_subsample(
    mesh: Mesh,
    duplicates,
    n_cells: int | None,
    generator: torch.Generator | None = None,
    start: int | None = None,
) -> Mesh:
    """``mesh`` without the extra copies that ``duplicates`` marks, drawn as
    :func:`~physicsnemo.datapipes.readers.mesh._subsample_mesh_cells` draws.

    The cyclic block of ``n_cells`` is drawn over all ``N`` stored cells (from
    ``generator``, or at ``start``); the drawn extra copies are dropped and the
    kept cells' measures are multiplied by ``N / n_cells``. With ``n_cells`` unset
    or at least ``N``, every kept cell is returned with its own measure.
    """
    n_total = mesh.n_cells
    if n_cells is None or n_total <= n_cells:
        idx = torch.nonzero(~duplicates.all().to(mesh.cells.device)).flatten()
        factor = None
    else:
        block = _cyclic_block_indices(
            n_total, n_cells, start=start, device=mesh.cells.device, generator=generator
        )
        idx = block[~duplicates[block]]
        factor = n_total / n_cells
    mesh = mesh.slice_cells(idx)
    referenced = torch.unique(mesh.cells)
    if referenced.numel() < mesh.n_points:
        mesh = mesh.slice_points(referenced)
    if factor is not None:
        scale_measures(mesh, factor)
    return mesh
