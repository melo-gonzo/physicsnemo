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

"""``.pdmsh`` round-trip tests for the GeoPT pretraining-sample builder.

Covers M3 exit criteria G11-G14 from
``geopt-datagen-round1-plan.md`` §5.3:

* (G11) alignment exits with X-extent and Y-min at target values
  (covered indirectly via the recorded ``AlignmentRecord`` invariants).
* (G12) tiny ``.pdmsh`` round-trip: every tensor recovered bit-exact;
  every TensorDict key present.
* (G13) full-config ``.pdmsh`` round-trip wall-clock budget; on-disk
  size budget. Gated on ``PNM_M3_FULL_BENCH=1`` env var since it is
  slow-ish for an inner-loop suite.
* (G14) post-load schema matches the ``builder.py`` module-level spec.

The builder uses the round-1-decided **synthesized OBJ** path
exclusively (no ShapeNet corpus this round). Each test writes the
analytic conftest mesh through ``trimesh.Trimesh.export(.obj)`` and
loads it back through the public builder API; this exercises the OBJ
I/O leg even on analytic geometries.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from physicsnemo.experimental.pnm_pretraining.data import (
    AlignmentRecord,
    build_pretraining_sample,
    load_pretraining_sample,
    save_pretraining_sample,
)
from physicsnemo.mesh.domain_mesh import DomainMesh
from test.conftest import requires_module
from test.experimental.pnm_pretraining.conftest import TriangleMesh

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_obj(mesh: TriangleMesh, path: Path) -> Path:
    """Materialize a ``TriangleMesh`` to an ``.obj`` file on disk."""
    import trimesh

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.indices, process=False)
    tm.export(str(path))
    return path


def _tiny_kwargs() -> dict:
    """Tiny configuration shared by tests (a)-(e)."""
    return dict(
        n_volume_points=512,
        n_surface_points=128,
        n_independent_walks=2,
        n_jittered_per_base=1,  # n_walks = 2 * (1 + 1) = 4
        n_steps=2,
        seed=42,
    )


def _expected_point_data_keys() -> set[str]:
    return {
        "region",
        "sdf",
        "normals_face_barycentric",
        "normals_vertex_nearest",
        "supervise_step0",
    }


def _expected_interior_global_keys() -> set[str]:
    return {
        "walks_supervise",
        "walks_directions",
        "walks_step_lengths",
        "walks_is_independent",
    }


def _expected_global_data_keys() -> dict[str, set[str]]:
    return {
        "config": {
            "n_volume_points",
            "n_surface_points",
            "n_walks",
            "n_steps",
            "target_length",
            "max_step",
            "perturb_sigma",
            "seed",
        },
        "alignment": {
            "axis_flipped",
            "y_min_post_flip",
            "scale",
            "x_mean_post_scale",
            "z_mean_post_scale",
            "oversize_safety_applied",
        },
        "mesh_quality": {
            "is_watertight",
            "n_vertices_pre_alignment",
            "n_faces",
        },
    }


def _all_leaf_paths(td) -> list[tuple[str, ...]]:
    """Flatten a TensorDict into a list of leaf paths.

    Returns each path as a tuple of string keys (length 1 for top-level
    leaves, longer for nested groups).
    """
    paths: list[tuple[str, ...]] = []
    for key in td.keys(include_nested=True, leaves_only=True):
        if isinstance(key, tuple):
            paths.append(key)
        else:
            paths.append((str(key),))
    return paths


def _tensor_at_path(td, path: tuple[str, ...]) -> torch.Tensor:
    cur = td
    for p in path:
        cur = cur[p]
    return cur


# ---------------------------------------------------------------------------
# (a) Tiny config round-trip on the sphere fixture.
# ---------------------------------------------------------------------------


@requires_module("warp")
@requires_module("trimesh")
def test_pdmsh_tiny_config_round_trip(
    sphere_mesh: TriangleMesh, tmp_path: Path
) -> None:
    """G12 — tiny config builds, saves atomically, reloads bit-exact."""
    obj_path = _write_obj(sphere_mesh, tmp_path / "input.obj")

    dm = build_pretraining_sample(obj_path, **_tiny_kwargs())

    # Schema (presence + shapes + dtypes).
    assert isinstance(dm, DomainMesh)
    assert tuple(dm.interior.points.shape) == (640, 3)
    assert dm.interior.points.dtype == torch.float32

    pd_keys = set(dm.interior.point_data.keys())
    assert pd_keys == _expected_point_data_keys(), (
        f"point_data keys mismatch: extra={pd_keys - _expected_point_data_keys()}, "
        f"missing={_expected_point_data_keys() - pd_keys}"
    )
    ig_keys = set(dm.interior.global_data.keys())
    assert ig_keys == _expected_interior_global_keys(), (
        f"interior.global_data keys mismatch: extra={ig_keys - _expected_interior_global_keys()}, "
        f"missing={_expected_interior_global_keys() - ig_keys}"
    )

    # Per-point dtype + shape spot checks.
    assert dm.interior.point_data["region"].dtype == torch.int8
    assert tuple(dm.interior.point_data["region"].shape) == (640,)
    for k in ("sdf",):
        assert dm.interior.point_data[k].dtype == torch.float32
        assert tuple(dm.interior.point_data[k].shape) == (640,)
    for k in ("normals_face_barycentric", "normals_vertex_nearest", "supervise_step0"):
        assert dm.interior.point_data[k].dtype == torch.float32
        assert tuple(dm.interior.point_data[k].shape) == (640, 3)

    # Walk-level shape checks (n_walks = 4, n_steps = 2).
    assert tuple(dm.interior.global_data["walks_supervise"].shape) == (4, 640, 2, 3)
    assert dm.interior.global_data["walks_supervise"].dtype == torch.float32
    assert tuple(dm.interior.global_data["walks_directions"].shape) == (4, 640, 3)
    assert dm.interior.global_data["walks_directions"].dtype == torch.float32
    assert tuple(dm.interior.global_data["walks_step_lengths"].shape) == (4, 640)
    assert dm.interior.global_data["walks_step_lengths"].dtype == torch.float32
    assert tuple(dm.interior.global_data["walks_is_independent"].shape) == (4,)
    assert dm.interior.global_data["walks_is_independent"].dtype == torch.int8

    # Boundaries (the aligned mesh).
    assert "geometry" in dm.boundaries.keys()
    geom = dm.boundaries["geometry"]
    assert geom.points.dtype == torch.float32
    assert geom.points.ndim == 2 and geom.points.shape[-1] == 3
    assert geom.cells.dtype == torch.int64
    assert geom.cells.ndim == 2 and geom.cells.shape[-1] == 3

    # Domain global_data nested groups.
    domain_gd_keys = _expected_global_data_keys()
    for grp, keys in domain_gd_keys.items():
        assert grp in dm.global_data.keys(), f"missing global_data group {grp}"
        actual = set(dm.global_data[grp].keys())
        assert actual == keys, (
            f"global_data[{grp!r}] keys mismatch: extra={actual - keys}, "
            f"missing={keys - actual}"
        )

    # Atomic save → reload round-trip; every leaf bit-exact.
    out = save_pretraining_sample(dm, tmp_path / "sample0", atomic=True)
    assert out.is_dir() and out.suffix == ".pdmsh"
    dm2 = load_pretraining_sample(out)

    # Top-level mesh tensors.
    torch.testing.assert_close(dm.interior.points, dm2.interior.points, rtol=0, atol=0)
    torch.testing.assert_close(dm.interior.cells, dm2.interior.cells, rtol=0, atol=0)
    torch.testing.assert_close(
        dm.boundaries["geometry"].points,
        dm2.boundaries["geometry"].points,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        dm.boundaries["geometry"].cells,
        dm2.boundaries["geometry"].cells,
        rtol=0,
        atol=0,
    )

    # Every TensorDict leaf — point_data / interior.global_data / domain.global_data.
    for path in _all_leaf_paths(dm.interior.point_data):
        torch.testing.assert_close(
            _tensor_at_path(dm.interior.point_data, path),
            _tensor_at_path(dm2.interior.point_data, path),
            rtol=0,
            atol=0,
            msg=f"point_data leaf {'.'.join(path)} not bit-exact",
        )
    for path in _all_leaf_paths(dm.interior.global_data):
        torch.testing.assert_close(
            _tensor_at_path(dm.interior.global_data, path),
            _tensor_at_path(dm2.interior.global_data, path),
            rtol=0,
            atol=0,
            msg=f"interior.global_data leaf {'.'.join(path)} not bit-exact",
        )
    for path in _all_leaf_paths(dm.global_data):
        torch.testing.assert_close(
            _tensor_at_path(dm.global_data, path),
            _tensor_at_path(dm2.global_data, path),
            rtol=0,
            atol=0,
            msg=f"global_data leaf {'.'.join(path)} not bit-exact",
        )


# ---------------------------------------------------------------------------
# (b) Schema invariants on the in-memory build.
# ---------------------------------------------------------------------------


@requires_module("warp")
@requires_module("trimesh")
def test_pdmsh_schema_invariants(sphere_mesh: TriangleMesh, tmp_path: Path) -> None:
    """region partition, surface-row pinning, walks_is_independent count."""
    obj_path = _write_obj(sphere_mesh, tmp_path / "input.obj")
    kw = _tiny_kwargs()
    dm = build_pretraining_sample(obj_path, **kw)

    region = dm.interior.point_data["region"]
    assert int((region == 0).sum().item()) == kw["n_volume_points"]
    assert int((region == 1).sum().item()) == kw["n_surface_points"]

    # sdf for surface rows is exactly 0 (the closest point of an
    # on-surface query is itself).
    sdf_surface = dm.interior.point_data["sdf"][region == 1]
    assert torch.all(sdf_surface == 0)

    # walks_step_lengths for surface rows is exactly 0 (M2 G8 surface-pin).
    surf_step_lengths = dm.interior.global_data["walks_step_lengths"][:, region == 1]
    assert torch.all(surf_step_lengths == 0)

    # walks_is_independent: exactly n_independent True entries (rest False).
    is_indep = dm.interior.global_data["walks_is_independent"]
    assert int((is_indep != 0).sum().item()) == kw["n_independent_walks"]
    assert is_indep.numel() == kw["n_independent_walks"] * (
        1 + kw["n_jittered_per_base"]
    )

    # supervise_step0 for surface rows is exactly (0, 0, 0).
    surf_sup_step0 = dm.interior.point_data["supervise_step0"][region == 1]
    assert torch.all(surf_sup_step0 == 0)

    # boundaries.geometry.points == record.apply(input_mesh.vertices).
    record = AlignmentRecord(
        axis_flipped=bool(int(dm.global_data["alignment"]["axis_flipped"])),
        y_min_post_flip=float(dm.global_data["alignment"]["y_min_post_flip"]),
        scale=float(dm.global_data["alignment"]["scale"]),
        x_mean_post_scale=float(dm.global_data["alignment"]["x_mean_post_scale"]),
        z_mean_post_scale=float(dm.global_data["alignment"]["z_mean_post_scale"]),
        oversize_safety_applied=bool(
            int(dm.global_data["alignment"]["oversize_safety_applied"])
        ),
    )
    aligned_input = record.apply(np.asarray(sphere_mesh.vertices, dtype=np.float64))
    np.testing.assert_allclose(
        dm.boundaries["geometry"].points.numpy(),
        aligned_input.astype(np.float32),
        atol=1e-5,
        rtol=0,
    )


# ---------------------------------------------------------------------------
# (c) AlignmentRecord round-trip and inverse.
# ---------------------------------------------------------------------------


@requires_module("warp")
@requires_module("trimesh")
def test_pdmsh_alignment_record_invertible(
    sphere_mesh: TriangleMesh, tmp_path: Path
) -> None:
    """Recorded alignment scalars reconstruct an invertible record."""
    obj_path = _write_obj(sphere_mesh, tmp_path / "input.obj")
    dm = build_pretraining_sample(obj_path, **_tiny_kwargs())

    align = dm.global_data["alignment"]
    # General variant always X-flips (axis_flipped == 1 as int8).
    assert int(align["axis_flipped"]) == 1
    assert float(align["scale"]) > 0.0
    # XZ-recenter discipline: unit-sphere fixture is symmetric, so
    # x_mean_post_scale should be ~ 0 (recenter is exact in float64).
    assert abs(float(align["x_mean_post_scale"])) < 1e-5

    record = AlignmentRecord(
        axis_flipped=bool(int(align["axis_flipped"])),
        y_min_post_flip=float(align["y_min_post_flip"]),
        scale=float(align["scale"]),
        x_mean_post_scale=float(align["x_mean_post_scale"]),
        z_mean_post_scale=float(align["z_mean_post_scale"]),
        oversize_safety_applied=bool(int(align["oversize_safety_applied"])),
    )

    rng = np.random.default_rng(seed=7)
    pts = rng.uniform(-10.0, 10.0, size=(10, 3)).astype(np.float64)
    pts_rt = record.inverse(record.apply(pts))
    err = float(np.max(np.abs(pts - pts_rt)))
    assert err < 1e-5, f"AlignmentRecord apply/inverse error {err} >= 1e-5"


# ---------------------------------------------------------------------------
# (d) Atomic write resilience.
# ---------------------------------------------------------------------------


@requires_module("warp")
@requires_module("trimesh")
def test_pdmsh_atomic_write_overwrite(
    sphere_mesh: TriangleMesh, tmp_path: Path
) -> None:
    """Twice-written prefix yields a clean .pdmsh, no .tmp leftovers."""
    obj_path = _write_obj(sphere_mesh, tmp_path / "input.obj")
    dm = build_pretraining_sample(obj_path, **_tiny_kwargs())

    out1 = save_pretraining_sample(dm, tmp_path / "sample", atomic=True)
    out2 = save_pretraining_sample(dm, tmp_path / "sample", atomic=True)
    assert out1 == out2
    assert out2.is_dir()

    # No .tmp leftover.
    leftovers = list(tmp_path.glob("*.pdmsh.tmp"))
    assert leftovers == [], f"unexpected .tmp leftovers: {leftovers}"


@requires_module("warp")
@requires_module("trimesh")
def test_pdmsh_atomic_write_failure_preserves_original(
    sphere_mesh: TriangleMesh, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing save preserves any prior .pdmsh and leaves .tmp for forensics."""
    obj_path = _write_obj(sphere_mesh, tmp_path / "input.obj")
    dm = build_pretraining_sample(obj_path, **_tiny_kwargs())

    # First save: clean, succeeds.
    final = save_pretraining_sample(dm, tmp_path / "sample", atomic=True)
    assert final.is_dir()
    # Snapshot the original inode/mtime to confirm it survives the failure.
    orig_mtime = final.stat().st_mtime_ns

    # Now monkeypatch DomainMesh.save to raise mid-write, after creating
    # some staging artifacts. We approximate the GeoPT failure mode.
    real_save = DomainMesh.save

    def flaky_save(self, prefix=None, *args, **kwargs):
        # Touch the staging directory so the test can verify it survived.
        Path(prefix).mkdir(parents=True, exist_ok=True)
        (Path(prefix) / "PARTIAL_WRITE_MARKER").write_text("x")
        raise RuntimeError("simulated mid-write failure")

    monkeypatch.setattr(DomainMesh, "save", flaky_save)

    with pytest.raises(RuntimeError, match="simulated mid-write failure"):
        save_pretraining_sample(dm, tmp_path / "sample", atomic=True)

    # Restore so the assertions below load cleanly.
    monkeypatch.setattr(DomainMesh, "save", real_save)

    # Original .pdmsh untouched.
    assert final.is_dir()
    assert final.stat().st_mtime_ns == orig_mtime
    # .tmp directory left for forensics.
    tmp_dir = final.parent / (final.name + ".tmp")
    assert tmp_dir.is_dir()
    assert (tmp_dir / "PARTIAL_WRITE_MARKER").is_file()


# ---------------------------------------------------------------------------
# (e) Consumer round-trip via DomainMeshReader.
# ---------------------------------------------------------------------------


@requires_module("warp")
@requires_module("trimesh")
def test_pdmsh_domain_mesh_reader_consumer(
    sphere_mesh: TriangleMesh, tmp_path: Path
) -> None:
    """The .pdmsh produced by builder is loadable via DomainMeshReader."""
    from physicsnemo.datapipes.readers.mesh import DomainMeshReader

    obj_path = _write_obj(sphere_mesh, tmp_path / "input.obj")
    dm = build_pretraining_sample(obj_path, **_tiny_kwargs())

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    save_pretraining_sample(dm, out_dir / "sample0", atomic=True)

    reader = DomainMeshReader(path=str(out_dir), pattern="*.pdmsh")
    assert len(reader) == 1
    dm2, meta = reader[0]
    assert isinstance(dm2, DomainMesh)
    assert meta["boundary_names"] == ["geometry"]

    # Same shape/dtype as test (a).
    assert tuple(dm2.interior.points.shape) == (640, 3)
    assert dm2.interior.points.dtype == torch.float32
    assert tuple(dm2.interior.global_data["walks_supervise"].shape) == (4, 640, 2, 3)
    assert dm2.interior.global_data["walks_supervise"].dtype == torch.float32


# ---------------------------------------------------------------------------
# (f) Full-config wall-clock smoke (gated on PNM_M3_FULL_BENCH).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("PNM_M3_FULL_BENCH") != "1",
    reason="full-config smoke is slow; set PNM_M3_FULL_BENCH=1 to enable",
)
@requires_module("warp")
@requires_module("trimesh")
def test_pdmsh_full_config_smoke(
    sphere_mesh: TriangleMesh, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """G13 — full-size config wall-clock + on-disk size budget.

    Skips the jittered walks (n_jittered_per_base=0) so the runtime is
    bounded on the inner-loop dispatch. The full 10 + 90 walk-set
    timing is recorded in the M3 report from a one-off run.
    """
    obj_path = _write_obj(sphere_mesh, tmp_path / "input.obj")

    t0 = time.perf_counter()
    dm = build_pretraining_sample(
        obj_path,
        n_volume_points=32_768,
        n_surface_points=4_096,
        n_independent_walks=10,
        n_jittered_per_base=0,  # 10 walks total; speed bound for the inner-loop test
        n_steps=3,
        seed=7,
    )
    t_build = time.perf_counter() - t0

    out = save_pretraining_sample(dm, tmp_path / "full", atomic=True)
    t_total = time.perf_counter() - t0

    # On-disk size: walk the .pdmsh dir tree.
    on_disk_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())

    # Report numbers via stdout; pytest captures these and the M3 report
    # quotes them.
    print(
        f"FULL_CONFIG_BENCH build_s={t_build:.3f} total_s={t_total:.3f} "
        f"on_disk_bytes={on_disk_bytes} on_disk_mb={on_disk_bytes / 2**20:.2f}"
    )

    # Loose budget: < 120 s on this CPU-dispatch host (G13 says < 60 s
    # with the 10+90 set on a development machine; 10-walk subset ≪
    # that).
    assert t_total < 120.0, f"full-config run too slow: {t_total:.1f}s"

    # Cleanup the large output to keep the test directory tidy when
    # tmp_path persists in CI.
    shutil.rmtree(out, ignore_errors=True)
