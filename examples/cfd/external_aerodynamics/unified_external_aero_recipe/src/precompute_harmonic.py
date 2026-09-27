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
"""Precompute harmonic boundary measures for a ``.pdmsh`` dataset into a new dataset directory (HARMONIC, 2026-09-27).

Copies every case ``<src>/<case>/<case>.pdmsh`` to ``<dst>/<case>/<case>.pdmsh`` with the fields of
:class:`domain_transforms.HarmonicBoundaryMeasures` added to ``interior.point_data`` (``hm_<type>`` and, with a value
field, ``hx_<type>``); the source dataset is never modified. Top-level ``*.json`` files (manifest, mesh info) are copied.

Two ways to find the boundary nodes and their types:

* ``--node-tag-fields tag_dirichlet tag_neumann tag_robin`` (OLBC): per-node 0/1 tags stored in ``interior.point_data``,
  one per ``--label-fields`` entry, in the same order; ``--node-value-field bc_value_node`` gives the prescribed value
  at those nodes for the extensions;
* ``--wall-from-sdf`` (AirfRANS): the boundary mesh has one type, the wall; interior nodes with ``|sdf| <= tol`` are the
  wall nodes, and the far field, which is not stored as a boundary mesh, is represented by the nodes whose ``sdf``
  exceeds the case's 99th percentile, tagged as a second type ``farfield`` so the wall's harmonic measure is the hitting
  probability of the wall against the far field. No prescribed scalar exists there, so no extension is written.

Examples::

    python precompute_harmonic.py --src .../olbc_poisson_pdmsh --dst .../olbc_poisson_hm_pdmsh --boundary outer \
        --label-fields bc_dirichlet bc_neumann bc_robin --node-tag-fields tag_dirichlet tag_neumann tag_robin \
        --value-field bc_value_g --node-value-field bc_value_node --workers 64
    python precompute_harmonic.py --src .../airfrans_pdmsh --dst .../airfrans_hm_pdmsh --boundary airfoil \
        --wall-from-sdf --pattern 'airFoil2D_SST_*' --workers 32
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harmonic import harmonic_boundary_fields, node_types_from_tags  # noqa: E402
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402

ARGS: argparse.Namespace | None = None


def _case_paths(src: Path, pattern: str) -> list[Path]:
    return sorted(p for p in src.glob(f"{pattern}/*.pdmsh") if p.is_dir())


def _process(case: Path) -> tuple[str, float, str]:
    a = ARGS
    assert a is not None
    t0 = time.time()
    name = case.parent.name
    dst = Path(a.dst) / name / case.name
    if dst.exists() and not a.overwrite:
        return name, 0.0, "exists"
    try:
        dm = DomainMesh.load(case)
        x = dm.interior.points.detach().cpu().numpy()
        pd = dm.interior.point_data
        values = None
        if a.wall_from_sdf:
            sdf = pd["sdf"].detach().cpu().numpy().reshape(-1)
            wall = np.abs(sdf) <= a.wall_sdf_tol * float(np.abs(sdf).max())
            far = sdf >= np.percentile(sdf, 99.0)
            t = np.full(x.shape[0], -1, dtype=np.int64)
            t[far] = 1
            t[wall] = 0
            types = [a.boundary, "farfield"]
        else:
            t = node_types_from_tags([pd[f].detach().cpu().numpy() for f in a.node_tag_fields])
            types = [lf.replace("bc_", "") for lf in a.label_fields]
            if a.node_value_field:
                values = pd[a.node_value_field].detach().cpu().numpy().reshape(x.shape[0], -1)
        phi, hx = harmonic_boundary_fields(x, t, len(types), values=values, k=a.k)
        new_pd = pd.clone()
        for j, short in enumerate(types):
            new_pd[f"hm_{short}"] = torch.from_numpy(phi[:, j : j + 1].astype(np.float32))
            if hx is not None:
                new_pd[f"hx_{short}"] = torch.from_numpy(hx[:, j].astype(np.float32))
        interior = Mesh(
            points=dm.interior.points,
            cells=dm.interior.cells,
            point_data=new_pd,
            cell_data=dm.interior.cell_data,
            global_data=dm.interior.global_data,
        )
        out = DomainMesh(interior=interior, boundaries=dm.boundaries, global_data=dm.global_data)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst, copy_existing=True)  # loaded tensors are memory-mapped to the source; copy them
        for extra in case.parent.glob("*.json"):
            shutil.copy2(extra, dst.parent / extra.name)
        return name, time.time() - t0, f"ok n={x.shape[0]} bnd={int((t >= 0).sum())} sum1={float(np.abs(phi.sum(1) - 1).max()):.1e}"
    except Exception as e:  # noqa: BLE001 - one bad case must not kill the batch
        return name, time.time() - t0, f"FAILED {type(e).__name__}: {e}"


def main() -> None:
    global ARGS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--boundary", required=True, help="boundary mesh name (outer, airfoil, ...)")
    p.add_argument("--label-fields", nargs="*", default=[], help="per-cell 0/1 type labels on the boundary (bc_*)")
    p.add_argument("--node-tag-fields", nargs="*", default=None, help="per-node 0/1 tags in interior.point_data, one per label")
    p.add_argument("--value-field", default=None, help="prescribed value per boundary cell (documentation only when --node-value-field is given)")
    p.add_argument("--node-value-field", default=None, help="prescribed value per interior node (boundary nodes)")
    p.add_argument("--wall-from-sdf", action="store_true", help="single wall type from |sdf| <= tol, far field = sdf above the 99th percentile")
    p.add_argument("--wall-sdf-tol", type=float, default=1e-6, help="relative |sdf| tolerance for wall nodes (fraction of max |sdf|)")
    p.add_argument("--pattern", default="*", help="case directory glob under --src")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    ARGS = p.parse_args()
    if not ARGS.wall_from_sdf and not ARGS.node_tag_fields:
        p.error("give --node-tag-fields (typed nodes) or --wall-from-sdf")
    src, dst = Path(ARGS.src), Path(ARGS.dst)
    cases = _case_paths(src, ARGS.pattern)
    if ARGS.limit:
        cases = cases[: ARGS.limit]
    dst.mkdir(parents=True, exist_ok=True)
    for extra in src.glob("*.json"):
        shutil.copy2(extra, dst / extra.name)
    print(f"{len(cases)} cases {src} -> {dst}; workers={ARGS.workers}", flush=True)
    t0 = time.time()
    n_ok = n_fail = 0
    times = []
    with Pool(ARGS.workers) as pool:
        for i, (name, dt, msg) in enumerate(pool.imap_unordered(_process, cases, chunksize=4)):
            if msg.startswith("FAILED"):
                n_fail += 1
            elif msg != "exists":
                n_ok += 1
                times.append(dt)
            print(f"[{i + 1}/{len(cases)}] {name} {dt:.2f}s {msg}", flush=True)
    med = float(np.median(times)) if times else 0.0
    print(f"DONE ok={n_ok} failed={n_fail} wall={time.time() - t0:.0f}s median_case={med:.2f}s completed_units={n_ok}", flush=True)
    json.dump(
        {"src": str(src), "dst": str(dst), "ok": n_ok, "failed": n_fail, "median_case_s": med, "args": vars(ARGS)},
        open(dst / "harmonic_precompute.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
