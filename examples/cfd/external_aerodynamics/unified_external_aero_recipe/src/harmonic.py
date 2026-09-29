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
"""Harmonic boundary measures on a point cloud (HARMONIC, 2026-09-27).

For a domain whose boundary carries condition *types* (Dirichlet, Neumann, Robin,
wall, far field, ...) this module computes, at every interior node, two families
of fields that depend only on the geometry and on the boundary data of the one
sample:

* the **harmonic measure** ``phi_k`` of each type ``k``: the solution of Laplace's
  equation on the interior with ``phi_k = 1`` on the boundary nodes of type ``k``
  and ``phi_k = 0`` on the boundary nodes of every other type. It is the
  probability that a Brownian path started at the node first reaches the
  boundary on a patch of type ``k``; the ``phi_k`` are smooth, non-negative and
  sum to one;
* the **harmonic extension** ``hx_k`` of the prescribed data restricted to type
  ``k``: the same operator with the prescribed value ``g`` on the type-``k`` nodes
  and ``0`` on the other boundary nodes. Summed over ``k`` it is the harmonic
  extension of all the data.

The construction is mesh-free: the interior is a point cloud and the Laplacian is
the symmetric k-nearest-neighbour graph Laplacian with self-tuning Gaussian
weights ``w_ij = exp(-|x_i - x_j|^2 / (sigma_i sigma_j))``, ``sigma_i`` the
distance from ``x_i`` to its ``k``-th neighbour. Distances only enter, so every
field is invariant under rotations and translations of the sample. It is also
PDE-agnostic: the Laplacian is used as a geometric operator, not as the governing
equation of the dataset. One sparse LU factorization per case serves every
right-hand side.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu

from physicsnemo.nn.functional.neighbors import knn


def _knn(points: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(distances, indices)`` of the ``k`` nearest ``points`` to each query, float64 on the CPU."""
    idx, dist = knn(torch.from_numpy(points), torch.from_numpy(queries), k)
    return dist.numpy(), idx.numpy()


def default_k(points: np.ndarray) -> int:
    """12 neighbours for planar point sets (z constant), 24 otherwise."""
    return 12 if points.shape[1] < 3 or np.ptp(points[:, 2]) < 1e-9 * max(1.0, np.ptp(points[:, :2])) else 24


def knn_laplacian(points: np.ndarray, k: int) -> sp.csr_matrix:
    """Symmetric k-NN graph Laplacian ``L = D - W`` with self-tuning Gaussian weights."""
    x = np.asarray(points, dtype=np.float64)
    n = x.shape[0]
    dist, idx = _knn(x, x, k + 1)  # column 0 is the point itself
    dist, idx = dist[:, 1:], idx[:, 1:]
    sigma = np.maximum(dist[:, -1], 1e-300)
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    w = np.exp(-(dist.reshape(-1) ** 2) / (sigma[rows] * sigma[cols]))
    W = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)  # symmetrize: an edge present in either direction is kept
    d = np.asarray(W.sum(axis=1)).reshape(-1)
    return (sp.diags(d) - W).tocsr()


def harmonic_boundary_fields(
    points: np.ndarray,
    node_types: np.ndarray,
    n_types: int,
    values: np.ndarray | None = None,
    k: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Harmonic measures and per-type harmonic extensions on a point cloud.

    Parameters
    ----------
    points : (N, d) array
        Interior nodes, boundary nodes included.
    node_types : (N,) int array
        ``-1`` for a free (interior) node, ``0 .. n_types - 1`` for a boundary node of that type.
    n_types : int
        Number of condition types.
    values : (N, C) array or None
        Prescribed data at the boundary nodes (ignored at free nodes). ``None`` skips the extensions.
    k : int or None
        Neighbours per node; ``None`` picks :func:`default_k`.

    Returns
    -------
    phi : (N, n_types) float64
        Harmonic measures; ``1`` on the type's own boundary nodes, ``0`` on the other boundary nodes.
    hx : (N, n_types, C) float64 or None
        Harmonic extension of ``values`` restricted to each type.
    """
    x = np.asarray(points, dtype=np.float64)
    t = np.asarray(node_types).reshape(-1).astype(np.int64)
    n = x.shape[0]
    if k is None:
        k = default_k(x)
    L = knn_laplacian(x, k)
    bnd = t >= 0
    free = ~bnd
    if not bnd.any():
        raise ValueError("harmonic_boundary_fields: the sample has no boundary node")
    # Dirichlet data for every right-hand side, as columns: n_types measures, then n_types * C extensions.
    cols = [np.where(t == j, 1.0, 0.0) for j in range(n_types)]
    C = 0
    if values is not None:
        g = np.asarray(values, dtype=np.float64).reshape(n, -1)
        C = g.shape[1]
        for j in range(n_types):
            m = (t == j).astype(np.float64)[:, None]
            cols.extend(list((g * m).T))
    ub = np.stack(cols, axis=1)  # (N, n_rhs), meaningful on boundary rows
    ub[free] = 0.0
    nf = int(free.sum())
    if nf > 0:
        Lff = L[free][:, free].tocsc()
        Lfb = L[free][:, bnd]
        # a free component with no path to the boundary would make Lff singular; a relative ridge keeps the
        # factorization defined (the component then relaxes to zero), without measurably moving connected nodes
        ridge = 1e-12 * float(Lff.diagonal().max())
        lu = splu((Lff + ridge * sp.identity(nf, format="csc")).tocsc())
        rhs = -(Lfb @ ub[bnd])
        uf = lu.solve(np.ascontiguousarray(rhs))
        u = ub.copy()
        u[free] = uf
    else:
        u = ub
    phi = u[:, :n_types]
    hx = u[:, n_types:].reshape(n, n_types, C) if values is not None else None
    return phi, hx


def node_types_from_tags(tags: list[np.ndarray]) -> np.ndarray:
    """Per-node type index from a list of 0/1 per-node tag arrays (``-1`` where no tag is set)."""
    t = np.full(tags[0].reshape(-1).shape[0], -1, dtype=np.int64)
    for j, a in enumerate(tags):
        t[np.asarray(a).reshape(-1) > 0.5] = j
    return t


def node_types_from_boundary(
    points: np.ndarray, centroids: np.ndarray, cell_types: np.ndarray, tol: float | None = None
) -> np.ndarray:
    """Type the interior nodes that lie on the boundary mesh by their nearest boundary cell.

    A node is a boundary node when its distance to the nearest boundary cell centroid is below ``tol``;
    ``tol`` defaults to 0.75 times the median centroid spacing, so mesh nodes shared with the boundary are
    caught and the first interior layer is not.
    """
    x = np.asarray(points, dtype=np.float64)
    y = np.asarray(centroids, dtype=np.float64)
    if tol is None:
        dd, _ = _knn(y, y, 2)
        tol = 0.75 * float(np.median(dd[:, 1]))
    d, j = _knn(y, x, 1)
    d, j = d[:, 0], j[:, 0]
    t = np.full(x.shape[0], -1, dtype=np.int64)
    on = d <= tol
    t[on] = np.asarray(cell_types).reshape(-1)[j[on]]
    return t
