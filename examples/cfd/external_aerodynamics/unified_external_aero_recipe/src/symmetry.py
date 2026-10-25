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

"""HiLift's symmetry plane: the wing's root outline from the dataset's source STL, and the per-case check.

HiLiftAeroML's PDE domain is bounded by the wing, the hemisphere far field and the symmetry
plane ``y = 0``: the far field's equatorial disk minus the wing's root section. The case files
store no plane, and the stored wings' roots are open (their on-plane edges do not close into a
loop on most geometries), so the hole's outline is taken from the closed root cap of the
dataset's own source STL files (``<case>/geo_<case>.stl.pmsh``, stored in a few case folders,
the same cap for every geometry and angle of attack; Peter's ruling of 2026-10-24): the cells
of the first STL in sorted order whose three vertices lie on the plane, whose boundary edges
form one simple closed loop (:func:`load_root_cap`). Each case gets the plane only if its own
stored root lies on that outline, to within the tolerance :func:`outline_tolerance` derives
(:func:`check_root`); a case that fails gets no plane. Nothing in the dataset is modified.

The exact check reads the whole stored wing (13-23 s per case), so a dataset-level record of its
verdicts (``tools/build_symmetry_check.py``), checked against each case's fingerprints, serves
the loads; it holds no geometry.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dedup import CHUNK, connectivity_fingerprint, points_fingerprint

#: The source STL files below the dataset root; the first in sorted order is read.
STL_GLOB = "geo_*/*.stl.pmsh"
#: Query-chord pairs per block of the point-to-outline distance (about 64 MB per float64 array).
_DISTANCE_PAIRS = 1 << 22
_TRIANGLE_EDGES = np.array([[0, 1], [1, 2], [2, 0]])


def read_memmap_mesh(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """The float32 points and int64 cells of a memmap ``Mesh`` directory, as read-only numpy memmaps."""
    td = Path(path) / "_tensordict"
    meta = json.loads((td / "meta.json").read_text())
    out = []
    for key, dtype in (("points", np.float32), ("cells", np.int64)):
        if meta[key]["dtype"] != f"torch.{np.dtype(dtype).name}":
            raise ValueError(
                f"{td}: {key} are {meta[key]['dtype']}, expected {np.dtype(dtype).name}"
            )
        out.append(
            np.memmap(
                td / f"{key}.memmap",
                dtype=dtype,
                mode="r",
                shape=tuple(meta[key]["shape"]),
            )
        )
    return out[0], out[1]


def cap_md5(points: np.ndarray, cap_cells: np.ndarray) -> str:
    """HL-DUP's key of a root cap: the md5 of its vertices' float32 bit patterns, sorted per column."""
    capv = np.unique(cap_cells)
    bits = np.asarray(points)[capv].astype(np.float32).view(np.uint32).reshape(-1, 3)
    return hashlib.md5(
        np.ascontiguousarray(np.sort(bits, axis=0)).tobytes()
    ).hexdigest()


def root_loop(cap_cells: np.ndarray) -> np.ndarray:
    """The cap's boundary edges (edges of exactly one cap cell) walked into one simple closed loop.

    Returns the loop's vertex indices in loop order, from the smallest index toward its smaller
    neighbour, without repeating the start. Refuses a vertex of degree other than 2 and more than
    one loop.
    """
    edges = np.sort(cap_cells[:, _TRIANGLE_EDGES].reshape(-1, 2), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    adjacency: dict[int, list[int]] = {}
    for a, b in edges[counts == 1].tolist():
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if not adjacency:
        raise ValueError("root cap: no boundary edge")
    bad = {v: len(n) for v, n in adjacency.items() if len(n) != 2}
    if bad:
        raise ValueError(
            f"root cap: {len(bad)} boundary vertices of degree other than 2 "
            f"(e.g. {dict(list(bad.items())[:5])})"
        )
    start = min(adjacency)
    loop, prev, current = [start], start, min(adjacency[start])
    while current != start:
        loop.append(current)
        a, b = adjacency[current]
        prev, current = current, (b if a == prev else a)
    if len(loop) != len(adjacency):
        raise ValueError(
            f"root cap: its {len(adjacency)} boundary vertices form more than one loop "
            f"(the loop through vertex {start} has {len(loop)})"
        )
    return np.asarray(loop, dtype=np.int64)


def outline_tolerance(outline: np.ndarray) -> dict:
    """The distance within which a case's root must lie from the cap's outline (``tau``), with its inputs.

    ``outline`` is the cap's boundary loop in the plane's coordinates ``(x, z)``, float32 as stored.
    """
    ### The derivation. Distances are measured in the plane's coordinates (x, z), as curator
    ### projects the silhouette. The outline loop is v_0 .. v_{n-1} (float64 from the stored
    ### float32), chord j runs from v_j to v_{j+1} with length L_j.
    ### - Storage precision: u = spacing(float32(max |coordinate| of the outline)), and
    ###   eps = sqrt(2) u bounds the change of a point-to-segment distance when the query point and
    ###   the segment's ends each move by up to half a float32 spacing per coordinate
    ###   (sqrt(2) u / 2 for each side).
    ### - The turning theta_i at v_i is the signed angle from chord i-1 to chord i; its float32
    ###   uncertainty is delta_i = eps / L_{i-1} + eps / L_i.
    ### - The chord sagitta bound: between two consecutive cap vertices the true section is taken
    ###   to be a convex arc whose end tangents lie between the adjacent chords, so it makes an angle
    ###   of at most beta = |theta| + delta with the chord at each end. It then lies inside the
    ###   triangle on the chord with those base angles, of height
    ###   h_j = L_j / (cot beta_start + cot beta_end) = L_j sin(beta_start) sin(beta_end) / sin(beta_start + beta_end)
    ###   (cot 0 = +inf: a straight end gives 0). The bound is finite iff the cot sum is > 0, that is
    ###   iff beta_start + beta_end < pi; a cap where it is not is refused. The same formula is used
    ###   at the chords whose end turnings have opposite signs (an inflection inside the chord;
    ###   10 chords on HiLift's cap); there the largest bound is attained at a same-sign chord, so
    ###   the tolerance does not depend on that choice.
    ### - The tolerance tau = max_j h_j + eps.
    ### On HiLift's cap (968 vertices; the lane's measurements, 2026-10-24): u = 2.44e-4 in at
    ### 2583.33 in, eps = 3.453e-4 in, max h = 0.1341 in at the chord from (2483.343, 725.646) to
    ### (2482.437, 725.224) (ends turning 11.8 and 20.5 degrees), tau = 0.1345 in. Against the data:
    ### every on-plane CFD vertex of LHC001_AoA_10, LHC026_AoA_18 and LHC105_AoA_22 lies within
    ### 0.0613 in of the outline (the 0.061 in Hausdorff distance HL-DUP found between the cap and
    ### curator's hole); HL-DUP's 0.011 in is a different quantity (the broken roots' CFD vertices
    ### against LHC001's CFD outline, two meshes about 10 times denser than the STL), at 8 % of tau.
    ### The census (record md5 bc2ed653, LHR jobs 374043-374045): all 690 -medium cases that store a
    ### wing pass. The worst on-plane vertex is 0.0613 in (0.46 tau) on every case; the worst open-edge
    ### midpoint is 0.0601 in on 689, and 0.1266 in (0.94 tau) on geo_LHC024_AoA_8, where two open
    ### edges of about 20 in near the slat end within 0.004 in of the outline: their own chord sagitta.
    v = np.asarray(outline, dtype=np.float32).astype(np.float64)
    n = len(v)
    d = np.roll(v, -1, axis=0) - v  # chord j: v_j -> v_{j+1}
    length = np.hypot(d[:, 0], d[:, 1])
    if n < 3 or not bool((length > 0).all()):
        raise ValueError(f"root cap: the outline needs >= 3 distinct vertices, got {n}")
    u = float(np.spacing(np.abs(np.asarray(outline, dtype=np.float32)).max()))
    eps = float(np.sqrt(2.0)) * u
    dp = np.roll(d, 1, axis=0)  # chord i-1, ending at v_i
    theta = np.arctan2(dp[:, 0] * d[:, 1] - dp[:, 1] * d[:, 0], (dp * d).sum(axis=1))
    delta = eps / np.roll(length, 1) + eps / length
    beta = np.abs(theta) + delta  # at v_i
    start, end = beta, np.roll(beta, -1)  # at chord j's ends v_j and v_{j+1}
    finite = start + end < np.pi
    if not bool(finite.all()):
        j = int(np.flatnonzero(~finite)[0])
        raise ValueError(
            f"root cap: no finite sagitta bound on chord {j} (end angles {np.degrees(start[j]):.3f} "
            f"and {np.degrees(end[j]):.3f} degrees)"
        )
    h = length * np.sin(start) * np.sin(end) / np.sin(start + end)
    j = int(np.argmax(h))
    return {
        "tau": float(h[j]) + eps,
        "u": u,
        "eps": eps,
        "max_h": float(h[j]),
        "max_h_chord": [v[j].tolist(), v[(j + 1) % n].tolist()],
        "max_h_turning_deg": [
            float(np.degrees(theta[j])),
            float(np.degrees(theta[(j + 1) % n])),
        ],
        "n_chords": n,
    }


@dataclass(frozen=True)
class RootCap:
    """The root cap of the dataset's first source STL (:func:`load_root_cap`)."""

    file: str  # relative to the dataset root
    md5: str  # HL-DUP's key (cap_md5)
    n_cells: int
    outline: np.ndarray  # (n, 2) float32 (x, z), in loop order, as stored
    y_eps: float
    tolerance: dict  # outline_tolerance(outline)

    @property
    def tau(self) -> float:
        """The tolerance of :func:`outline_tolerance`."""
        return self.tolerance["tau"]


def load_root_cap(
    dataset_root: Path | str, accepted_md5, y_eps: float = 1e-3
) -> RootCap:
    """The root cap of ``sorted(Path(dataset_root).glob(STL_GLOB))[0]``: its cells with all three
    vertices at ``|y| < y_eps``, refused unless its key :func:`cap_md5` is in ``accepted_md5``,
    and its outline (:func:`root_loop`, strict). Read once per process and arguments."""
    return _load_root_cap(
        str(Path(dataset_root)), tuple(str(m) for m in accepted_md5), float(y_eps)
    )


@functools.lru_cache(maxsize=None)
def _load_root_cap(root: str, accepted_md5: tuple[str, ...], y_eps: float) -> RootCap:
    stls = sorted(Path(root).glob(STL_GLOB))
    if not stls:
        raise FileNotFoundError(f"root cap: no {STL_GLOB} below {root!r}")
    points, cells = read_memmap_mesh(stls[0])
    points, cells = np.asarray(points), np.asarray(cells)
    on_plane = np.abs(points[:, 1]) < y_eps
    cap_cells = cells[on_plane[cells].all(axis=1)]
    if not len(cap_cells):
        raise ValueError(f"root cap: {stls[0]} has no cell on |y| < {y_eps}")
    md5 = cap_md5(points, cap_cells)
    if md5 not in accepted_md5:
        raise ValueError(
            f"root cap: {stls[0]} has cap key {md5}, not among the accepted {list(accepted_md5)}"
        )
    outline = np.ascontiguousarray(points[root_loop(cap_cells)][:, [0, 2]])
    return RootCap(
        file=str(stls[0].relative_to(root)),
        md5=md5,
        n_cells=len(cap_cells),
        outline=outline,
        y_eps=y_eps,
        tolerance=outline_tolerance(outline),
    )


def outline_distance(query_xz: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """The exact distance from each ``(x, z)`` query point to the closed outline polyline (float64)."""
    a = np.asarray(outline, dtype=np.float32).astype(np.float64)
    d = np.roll(a, -1, axis=0) - a
    dd = (d * d).sum(axis=1)
    q = np.asarray(query_xz, dtype=np.float64)
    out = np.empty(len(q))
    step = max(1, _DISTANCE_PAIRS // len(a))
    for s in range(0, len(q), step):
        w = q[s : s + step, None, :] - a[None]
        t = np.clip((w * d).sum(axis=-1) / dd, 0.0, 1.0)
        out[s : s + step] = np.sqrt(((w - t[..., None] * d) ** 2).sum(axis=-1)).min(
            axis=1
        )
    return out


def _numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def check_root(points, cells, cap: RootCap, y_eps: float = 1e-3) -> dict:
    """Whether a stored wing's root lies on the cap's outline, exactly, on the whole wing.

    ``points`` ``(N, 3)`` float32 and ``cells`` ``(M, 3)`` (numpy, memmap or CPU tensors) are read
    in chunks of ``CHUNK`` rows. The on-plane vertices are every point with ``|y| < y_eps``
    (orphans included); the root cells, the cells with at least two of them, with exact repeats
    dropped (so the stored duplicate copies do not hide open edges); the open on-plane edges, the
    edges of the unique root cells with both ends on the plane used by exactly one of them. The
    case passes iff every on-plane vertex and every open edge's midpoint lies within ``cap.tau``
    of the outline, in ``(x, z)``; a wing with no on-plane vertex fails. A maximum over an empty
    set is ``None``.
    """
    points, cells = _numpy(points), _numpy(cells)
    on_plane = np.empty(len(points), dtype=bool)
    for s in range(0, len(points), CHUNK):
        on_plane[s : s + CHUNK] = np.abs(np.asarray(points[s : s + CHUNK, 1])) < y_eps
    idx = np.flatnonzero(on_plane)
    roots = []
    for s in range(0, len(cells), CHUNK):
        c = np.asarray(cells[s : s + CHUNK])
        r = c[on_plane[c].sum(axis=1) >= 2]
        if len(r):
            roots.append(np.unique(np.sort(r, axis=1), axis=0))
    roots = (
        np.unique(np.concatenate(roots), axis=0)
        if roots
        else np.zeros((0, 3), dtype=np.int64)
    )
    edges = np.sort(roots[:, _TRIANGLE_EDGES].reshape(-1, 2), axis=1)
    edges, counts = np.unique(
        edges[on_plane[edges].all(axis=1)], axis=0, return_counts=True
    )
    open_edges = edges[counts == 1]
    result = {
        "passed": False,
        "max_vertex_distance": None,
        "max_edge_distance": None,
        "n_on_plane": len(idx),
        "n_root_cells": len(roots),
        "n_open_edges": len(open_edges),
        "worst_vertex": None,
    }
    if not len(idx):
        return result
    xyz = np.asarray(points[idx])
    xz = xyz[:, [0, 2]].astype(np.float64)
    dist = outline_distance(xz, cap.outline)
    worst = int(np.argmax(dist))
    result["max_vertex_distance"] = float(dist[worst])
    result["worst_vertex"] = xyz[worst].astype(np.float64).tolist()
    if len(open_edges):
        mid = 0.5 * (
            xz[np.searchsorted(idx, open_edges[:, 0])]
            + xz[np.searchsorted(idx, open_edges[:, 1])]
        )
        result["max_edge_distance"] = float(outline_distance(mid, cap.outline).max())
    result["passed"] = result["max_vertex_distance"] <= cap.tau and (
        result["max_edge_distance"] is None or result["max_edge_distance"] <= cap.tau
    )
    return result


#: The record's format (``tools/build_symmetry_check.py``).
RECORD_FORMAT = "hl-symmetry-check/1"


@functools.lru_cache(maxsize=None)
def load_check_record(path: str) -> dict:
    """The record of ``tools/build_symmetry_check.py`` at ``path``, read once per process."""
    record = json.loads(Path(path).read_text())
    if record.get("format") != RECORD_FORMAT:
        raise ValueError(
            f"{path}: format {record.get('format')!r}, expected {RECORD_FORMAT!r}"
        )
    return record


def recorded_verdict(
    record: dict, cap: RootCap, case: str, points, cells, y_eps: float = 1e-3
) -> dict:
    """``case``'s entry of the record, after checking that the record was made with this cap,
    tolerance and ``y_eps`` and that the case's connectivity and points fingerprints match it."""
    made = (record["cap"]["md5"], record["tolerance"]["tau"], record["cap"]["y_eps"])
    if made != (cap.md5, cap.tau, y_eps):
        raise ValueError(
            f"symmetry check record: made with cap md5, tau and y_eps {made}, not the loaded "
            f"{(cap.md5, cap.tau, y_eps)}"
        )
    if case not in record["cases"]:
        raise KeyError(f"symmetry check record: no verdict for case {case!r}")
    entry = record["cases"][case]
    for key, got in (
        ("connectivity_fingerprint", connectivity_fingerprint(cells)),
        ("points_fingerprint", points_fingerprint(points)),
    ):
        if entry[key] != got:
            raise ValueError(
                f"symmetry check record: case {case!r} was checked on another wing ({key} "
                f"{entry[key]!r}, this case {got!r}); rebuild the record"
            )
    return entry
