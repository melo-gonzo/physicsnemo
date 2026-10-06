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

"""FixedRandomPose for the CHAR-PROTOCOLS posed evaluation (2026-10-02).

Pins: (a) the SO(3) stream of POSE-BENCH is unchanged, (b) ``translate`` adds a
deterministic translation of the stated length that the rotation draw does not
see, (c) planar problems rotate in their plane only and the 3-D-stored and 2-D
variants get the same angle, (d) non-spatial fields (``(N, 1)`` columns,
parameter vectors, one-hot codes) pass through where ``True`` would raise,
(e) component-stored symmetric tensors rotate as tensors, (f) the DomainMesh
path poses every sub-mesh and the domain global vectors with one pose, and
(g) ``CaseKeyDomainMeshReader`` stores the same key the surface reader would.
"""

import zlib

import pytest
import torch
from domain_transforms import (
    CaseKeyDomainMeshReader,
    FixedRandomPose,
    _pose_rotation_for_key,
)
from physicsnemo.datapipes.transforms.mesh import RandomRotateMesh
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh


def _reference_so3(key: int, salt: int) -> torch.Tensor:
    """The POSE-BENCH construction, verbatim."""
    g = torch.Generator().manual_seed((int(key) * 1_000_003 + int(salt)) % (2**63 - 1))
    q = torch.randn(4, generator=g, dtype=torch.float64)
    q = q / q.norm()
    return RandomRotateMesh._quaternion_to_rotation_matrix(q)


def _mesh(key: int = 12345, planar: bool = False, n: int = 40) -> Mesh:
    torch.manual_seed(0)
    pts = torch.randn(n, 3, dtype=torch.float64)
    if planar:
        pts[:, 2] = 0.0
    cells = torch.stack([torch.arange(0, n - 2), torch.arange(1, n - 1), torch.arange(2, n)], dim=1)
    nc = n - 2
    return Mesh(
        points=pts,
        cells=cells,
        point_data=TensorDict(
            {
                "sdf": torch.randn(n, 1, dtype=torch.float64),  # (N, 1) scalar column
                "velocity": torch.randn(n, 3, dtype=torch.float64),
            },
            batch_size=[n],
        ),
        cell_data=TensorDict(
            {
                "wss": torch.randn(nc, 3, dtype=torch.float64),
                "pressure": torch.randn(nc, dtype=torch.float64),
            },
            batch_size=[nc],
        ),
        global_data=TensorDict(
            {
                "U_inf": torch.tensor([30.0, 0.0, 0.0], dtype=torch.float64),
                "p_inf": torch.tensor(0.5, dtype=torch.float64),
                "params": torch.arange(6, dtype=torch.float64),  # non-spatial parameter vector
                "load_type": torch.tensor([1.0, 0.0], dtype=torch.float64),  # one-hot
                "case_key": torch.tensor(key, dtype=torch.int64),
            },
            batch_size=[],
        ),
    )


def test_so3_stream_is_byte_identical_to_pose_bench():
    for key in (0, 7, 12345, 2**31 - 1, 987654321):
        for salt in (0, 1):
            assert torch.equal(_pose_rotation_for_key(key, salt), _reference_so3(key, salt))
    ### A non-zero translate must not touch the rotation stream either.
    m = _mesh()
    R_ref = _reference_so3(12345, 0)
    a = FixedRandomPose(salt=0, translate=0.0)(m)
    b = FixedRandomPose(salt=0, translate=0.5)(m)
    assert torch.equal(a.points, m.points @ R_ref.T)
    assert torch.allclose(b.global_data["U_inf"], m.global_data["U_inf"] @ R_ref.T)


def test_non_spatial_fields_pass_through():
    m = _mesh()
    a = FixedRandomPose(salt=0)(m)  # `True` masks would raise on sdf (N, 1), params (6,), load_type (2,)
    R = _reference_so3(12345, 0)
    assert torch.equal(a.point_data["sdf"], m.point_data["sdf"])
    assert torch.equal(a.global_data["params"], m.global_data["params"])
    assert torch.equal(a.global_data["load_type"], m.global_data["load_type"])
    assert torch.allclose(a.point_data["velocity"], m.point_data["velocity"] @ R.T)
    assert torch.allclose(a.cell_data["wss"], m.cell_data["wss"] @ R.T)
    assert torch.equal(a.cell_data["pressure"], m.cell_data["pressure"])


def test_translate_is_deterministic_with_the_stated_length():
    m = _mesh()
    t = FixedRandomPose(salt=0, translate=0.5)
    a, b = t(m), t(_mesh())
    R = _reference_so3(12345, 0)
    shift = a.points - m.points @ R.T
    assert torch.allclose(shift, shift[:1].expand_as(shift), atol=1e-12)  # a rigid translation
    rms = (m.points - m.points.mean(0)).norm(dim=-1).square().mean().sqrt()
    assert torch.isclose(shift[0].norm(), 0.5 * rms)
    assert torch.equal(a.points, b.points)
    ### Vectors rotate but do not translate.
    assert torch.allclose(a.cell_data["wss"], m.cell_data["wss"] @ R.T)
    assert torch.allclose(a.global_data["U_inf"], m.global_data["U_inf"] @ R.T)
    ### Different cases, different translations; same salt-key, same translation.
    c = FixedRandomPose(salt=0, translate=0.5)(_mesh(key=12346))
    assert not torch.allclose(c.points - _mesh(key=12346).points @ _reference_so3(12346, 0).T, shift)


def test_in_plane_rotation_for_planar_problems():
    R3 = _pose_rotation_for_key(42, 0, 3, axis=2)
    R2 = _pose_rotation_for_key(42, 0, 2)
    assert torch.allclose(R3[:2, :2], R2)  # the stored-in-3-D and restricted-to-2-D variants share the angle
    assert torch.allclose(R3[2], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    assert torch.allclose(R3 @ R3.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert torch.det(R3) > 0 and torch.det(R2) > 0
    assert not torch.allclose(R2, _pose_rotation_for_key(43, 0, 2))
    ### Other plane normals embed the same 2x2 block in the other coordinates.
    R0 = _pose_rotation_for_key(42, 0, 3, axis=0)
    assert torch.allclose(R0[1:, 1:], R2) and torch.allclose(R0[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    with pytest.raises(ValueError):
        _pose_rotation_for_key(42, 0, 2, axis=2)
    ### On a z = 0 mesh the pose keeps z = 0 exactly, translation included.
    m = _mesh(key=42, planar=True)
    a = FixedRandomPose(salt=0, translate=1.0, axis=2)(m)
    assert torch.all(a.points[:, 2] == 0.0)
    assert torch.allclose(a.points[:, :2], m.points[:, :2] @ R2.T + (a.points[0, :2] - m.points[0, :2] @ R2.T))


def test_symmetric_tensor_components_rotate_as_a_tensor():
    m = _mesh(key=42, planar=True)
    n = m.n_points
    torch.manual_seed(1)
    sig = torch.randn(n, 2, 2, dtype=torch.float64)
    sig = sig + sig.transpose(1, 2)
    pd = m.point_data.clone()
    pd["sig11"], pd["sig22"], pd["sig12"] = sig[:, 0, 0], sig[:, 1, 1], sig[:, 0, 1]
    m = m.with_data(point_data=pd)
    a = FixedRandomPose(salt=0, axis=2, symmetric_tensor_fields=[["sig11", "sig22", "sig12"]])(m)
    R2 = _pose_rotation_for_key(42, 0, 2)
    expected = torch.einsum("ij,njk,lk->nil", R2, sig, R2)
    assert torch.allclose(a.point_data["sig11"], expected[:, 0, 0])
    assert torch.allclose(a.point_data["sig22"], expected[:, 1, 1])
    assert torch.allclose(a.point_data["sig12"], expected[:, 0, 1])
    ### The invariants of the tensor are untouched.
    assert torch.allclose(a.point_data["sig11"] + a.point_data["sig22"], sig[:, 0, 0] + sig[:, 1, 1])
    with pytest.raises(ValueError):
        FixedRandomPose(salt=0, symmetric_tensor_fields=[["sig11", "sig22", "sig12"]])(_mesh(key=42) .with_data(point_data=pd))
    with pytest.raises(ValueError):
        FixedRandomPose(symmetric_tensor_fields=[["sig11", "sig22"]])


def _domain(key: int = 77) -> DomainMesh:
    torch.manual_seed(3)
    interior = Mesh(
        points=torch.randn(50, 3, dtype=torch.float64),
        point_data=TensorDict(
            {"velocity": torch.randn(50, 3, dtype=torch.float64), "sdf": torch.randn(50, 1, dtype=torch.float64)},
            batch_size=[50],
        ),
    )
    bnd = _mesh(key=key)
    bnd = bnd.with_data(global_data=TensorDict({}, batch_size=[]))
    return DomainMesh(
        interior=interior,
        boundaries={"vehicle": bnd},
        global_data=TensorDict(
            {
                "U_inf": torch.tensor([0.0, 20.0, 0.0], dtype=torch.float64),
                "naca": torch.arange(5, dtype=torch.float64),
                "case_key": torch.tensor(key, dtype=torch.int64),
            },
            batch_size=[],
        ),
    )


def test_domain_path_poses_every_submesh_and_domain_globals_together():
    dm = _domain()
    out = FixedRandomPose(salt=0, translate=0.25).apply_to_domain(dm)
    R = _reference_so3(77, 0)
    t = out.interior.points[0] - dm.interior.points[0] @ R.T
    p = dm.interior.points
    rms = (p - p.mean(0)).norm(dim=-1).square().mean().sqrt()
    assert torch.isclose(t.norm(), 0.25 * rms)
    assert torch.allclose(out.interior.points, dm.interior.points @ R.T + t)
    assert torch.allclose(out.boundaries["vehicle"].points, dm.boundaries["vehicle"].points @ R.T + t)
    assert torch.allclose(out.interior.point_data["velocity"], dm.interior.point_data["velocity"] @ R.T)
    assert torch.equal(out.interior.point_data["sdf"], dm.interior.point_data["sdf"])
    assert torch.allclose(out.boundaries["vehicle"].cell_data["wss"], dm.boundaries["vehicle"].cell_data["wss"] @ R.T)
    assert torch.allclose(out.global_data["U_inf"], dm.global_data["U_inf"] @ R.T)
    assert torch.equal(out.global_data["naca"], dm.global_data["naca"])
    assert torch.equal(dm.global_data["U_inf"], torch.tensor([0.0, 20.0, 0.0], dtype=torch.float64))  # input untouched


def test_case_key_domain_mesh_reader_matches_the_surface_reader_key(tmp_path):
    for name in ("run_1", "run_2"):
        dm = DomainMesh(
            interior=Mesh(points=torch.randn(6, 3)),
            boundaries={"vehicle": Mesh(points=torch.randn(6, 3), cells=torch.tensor([[0, 1, 2], [3, 4, 5]]))},
            global_data=TensorDict({"U_inf": torch.tensor([30.0, 0.0, 0.0])}, batch_size=[]),
        )
        (tmp_path / name).mkdir()
        dm.save(tmp_path / name / "domain.pdmsh")
    r = CaseKeyDomainMeshReader(tmp_path, pattern="run_*/*.pdmsh", subsample_n_points=4, drop_interior_cells=True)
    k0 = int(r._load_sample(0).global_data["case_key"])
    k1 = int(r._load_sample(1).global_data["case_key"])
    assert k0 == zlib.crc32(b"run_1") & 0x7FFFFFFF and k1 == zlib.crc32(b"run_2") & 0x7FFFFFFF
    dm0, _ = r[0]  # the key survives __getitem__'s subsampling and the pose reads it
    assert int(dm0.global_data["case_key"]) == k0
    posed = FixedRandomPose(salt=0).apply_to_domain(dm0)
    assert torch.allclose(posed.global_data["U_inf"], dm0.global_data["U_inf"] @ _reference_so3(k0, 0).to(torch.float32).T)


def test_rotate_false_is_translation_only_with_the_posed_variants_translation():
    """Translation-only diagnostic (2026-10-03): no rotation, and the translation vector equals the one
    the rotating variant applies (the rotation draw is still consumed from the keyed stream)."""
    m = _mesh(key=42, planar=True)
    posed = FixedRandomPose(salt=0, translate=1.0, axis=2)(m)
    tr = FixedRandomPose(salt=0, translate=1.0, axis=2, rotate=False)(m)
    R2 = _pose_rotation_for_key(42, 0, 2)
    t_posed = posed.points[0, :2] - m.points[0, :2] @ R2.T
    t_only = tr.points[0] - m.points[0]
    assert torch.allclose(tr.points, m.points + t_only)  # rigid translation, no rotation
    assert torch.allclose(t_only[:2], t_posed) and t_only[2] == 0.0
    assert torch.allclose(tr.cell_data["wss"], m.cell_data["wss"])  # vectors untouched
    assert torch.allclose(tr.global_data["U_inf"], m.global_data["U_inf"])
    assert "rotate=False" in tr.__class__.__name__ or "rotate=False" in FixedRandomPose(rotate=False).extra_repr()
