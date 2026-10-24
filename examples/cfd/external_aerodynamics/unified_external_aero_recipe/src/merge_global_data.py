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

Recipe-local module registered into the global datapipe component
registry so the class can be referenced via
``${dp:MeshReaderWithGlobalData}`` in Hydra YAML configs.  Import this
module before Hydra instantiation.
"""

from __future__ import annotations

import logging
import zlib
from typing import Any

import torch
from dedup import ComputedDuplicates, DuplicateMask
from tensordict import TensorDict

from physicsnemo.datapipes._indexing import _cyclic_block_indices
from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.datapipes.keys import key_to_str, leaf_keys
from physicsnemo.datapipes.readers.mesh import MeshReader, _subsample_mesh_points
from physicsnemo.datapipes.registry import register
from physicsnemo.mesh import Mesh
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

    def _case_key(self, index: int) -> torch.Tensor:
        sample_path = self._paths[index]
        try:
            name = sample_path.relative_to(self._root).parts[0]
        except ValueError:
            name = sample_path.name
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

    def __getitem__(self, index: int) -> tuple[Mesh, dict[str, Any]]:
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
