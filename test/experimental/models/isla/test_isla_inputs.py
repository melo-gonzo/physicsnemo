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

"""Contract tests for ISLA's rank-spec per-token inputs.

Rank-1 and rank-2 boundary fields (in the meshes' ``cell_data``) and
prediction-point fields (in the prediction Mesh's ``point_data``) enter only
through invariants formed with true vectors (the normal, the global directions, the
token-to-anchor direction), so every rank-0 output stays invariant and every rank
>= 1 output stays covariant when the geometry AND the fields are rotated, reflected
or translated together; the checks are exact (float64).
"""

import pytest
import torch

from physicsnemo.experimental.models import ISLA
from test.experimental.models.isla._meshes import (
    cloud,
    law,
    orthogonal,
    outputs,
    points_mesh,
    soup,
    transform,
)

D = torch.float64
OUT = {"p": {"rank": 0}, "v": {"rank": 1}, "sig": {"rank": 2, "symmetric": True}}
BIN = {"trac": {"rank": 1}, "T": {"rank": 2, "symmetric": True}, "A": {"rank": 2}, "tag": {"rank": 0}}
QIN = {"sdf": {"rank": 0}, "u": {"rank": 1}, "S": {"rank": 2, "symmetric": True}}
DRV = torch.tensor([0.3, 1.0, 0.2], dtype=D)


def _fields(n, seed, symmetric_names, vector_names, scalar_names):
    g = torch.Generator().manual_seed(seed)
    f = {}
    for k in vector_names:
        f[k] = torch.randn(n, 3, dtype=D, generator=g)
    for k in symmetric_names:
        t = torch.randn(n, 3, 3, dtype=D, generator=g)
        f[k] = t + t.transpose(-1, -2)
    for k in scalar_names:
        f[k] = torch.rand(n, dtype=D, generator=g)
    return f


def _boundary_fields(n, seed=1):
    f = _fields(n, seed, ["T"], ["trac"], ["tag"])
    f["A"] = torch.randn(n, 3, 3, dtype=D, generator=torch.Generator().manual_seed(seed + 10))
    return f


def _query_fields(n, seed=2):
    return _fields(n, seed, ["S"], ["u"], ["sdf"])


def _move(fields, specs, R):
    """Transform every field by its rank: rank 0 unchanged, rank 1 ``v R^T``, rank 2
    ``R T R^T``."""
    out = {}
    for k, v in fields.items():
        r = specs[k].rank
        out[k] = v if r == 0 else (v @ R.T if r == 1 else R @ v @ R.T)
    return out


def _model(**kw):
    torch.manual_seed(0)
    base = dict(output_schema=OUT, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=2, n_slices=8)
    base.update(kw)
    return ISLA(**base).double().eval()


def _assert_covariant(m, call, improper):
    """``call(R, shift)`` runs the model on the transformed problem."""
    base = call(torch.eye(3, dtype=D), torch.zeros(3, dtype=D))
    R = orthogonal(3, 4, improper)
    out = call(R, torch.tensor([2.0, -5.0, 1.0], dtype=D))
    for name, spec in m.output_schema.items():
        assert torch.allclose(out[name], law(base[name], R, spec.rank), atol=1e-11), (name, improper)
    return base


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_boundary_inputs_covariance(improper):
    m = _model(boundary_schema=BIN)
    pts, nrm, w, _ = cloud(200)
    bf = _boundary_fields(200)
    mesh = soup(pts, nrm, w, cell_data=bf)

    def call(R, shift):
        with torch.no_grad():
            return outputs(
                m(boundary_meshes={"wall": transform(mesh, R, shift, specs=m.boundary_schema)}, global_data={"g": DRV @ R.T})
            )

    base = _assert_covariant(m, call, improper)
    ### the fields are live: scaling the traction changes every output
    with torch.no_grad():
        other = outputs(
            m(boundary_meshes={"wall": soup(pts, nrm, w, cell_data={**bf, "trac": 2.0 * bf["trac"]})}, global_data={"g": DRV})
        )
    assert all(not torch.allclose(other[k], base[k], atol=1e-6) for k in base)
    ### a rotation of the fields ALONE (geometry fixed) is not a symmetry
    R = orthogonal(3, 6, False)
    with torch.no_grad():
        skew = outputs(
            m(boundary_meshes={"wall": soup(pts, nrm, w, cell_data=_move(bf, m.boundary_schema, R))}, global_data={"g": DRV})
        )
    assert not torch.allclose(skew["p"], base["p"], atol=1e-6)


@pytest.mark.parametrize("improper", [False, True], ids=["rotation", "reflection"])
def test_query_inputs_covariance(improper):
    """Rank-0/1/2 per-prediction-point fields in the prediction Mesh's point_data."""
    pts, nrm, w, _ = cloud(200)
    q_pts, _, _, _ = cloud(40, seed=3)
    m = _model(interior=True, query_schema=QIN, boundary_schema={"trac": {"rank": 1}})
    bf = {"trac": torch.randn(200, 3, dtype=D, generator=torch.Generator().manual_seed(5))}
    mesh = soup(pts, nrm, w, cell_data=bf)
    n_q = 40
    qf = _query_fields(n_q)

    def call(R, shift):
        with torch.no_grad():
            return outputs(
                m(
                    boundary_meshes={"wall": transform(mesh, R, shift, specs=m.boundary_schema)},
                    prediction_points=points_mesh(q_pts @ R.T + shift, _move(qf, m.query_schema, R)),
                    global_data={"g": DRV @ R.T},
                )
            )

    base = _assert_covariant(m, call, improper)
    assert base["p"].shape == (n_q,) and base["sig"].shape == (n_q, 3, 3)
    ### the query vector field is live
    with torch.no_grad():
        other = outputs(
            m(
                boundary_meshes={"wall": mesh},
                prediction_points=points_mesh(q_pts, {**qf, "u": 3.0 * qf["u"]}),
                global_data={"g": DRV},
            )
        )
    assert not torch.allclose(other["p"], base["p"], atol=1e-6)


def test_geometry_width_and_seed_width():
    """One rank-1 input and one symmetric rank-2 input over K = 1: the geometry region
    gains P' = 1 + (1 + K) contracted vectors and Q = 1 quadratic form; the seed gains
    the frame cosines, the norm, the frame contractions, the trace and the norm."""
    for K in (0, 1, 2):
        m = _model(
            global_schema={f"g{k}": {"rank": 1} for k in range(K)},
            boundary_schema={"v": {"rank": 1}, "T": {"rank": 2, "symmetric": True}},
        )
        F = 1 + K
        assert m.blocks[0].n_geo == 5 + K + (1 + F) + 1
        assert m.n_block_vecs == 1 + F and m.n_block_tens == 1
        assert m.bin.n_features == (F + 1) + (F * (F + 1) // 2 + 2)
        assert m.embed[0].in_features == K + m.bin.n_features
    ### a non-symmetric rank-2 input contracts both T a and T^T a and all F^2 pairs
    m = _model(boundary_schema={"A": {"rank": 2}})
    assert m.n_block_vecs == 2 * 2 and m.bin.n_features == 4 + 2
    ### query inputs widen the union stack for every token, boundary ones included
    m = _model(interior=True, query_schema={"u": {"rank": 1}}, boundary_schema={"v": {"rank": 1}})
    assert m.blocks[0].n_geo == 5 + 1 + 2


def test_field_validation_errors():
    """Declared fields are read from cell_data / point_data by name and checked for
    shape; extras there are ignored (like GLOBE)."""
    m = _model(boundary_schema={"trac": {"rank": 1}, "tag": {"rank": 0}})
    pts, nrm, w, _ = cloud(20)
    gd = {"g": torch.ones(3, dtype=D)}
    trac, tag = torch.randn(20, 3, dtype=D), torch.rand(20, dtype=D)
    with pytest.raises(ValueError, match="cell_data is missing the declared fields \\['tag'\\]"):
        m(boundary_meshes={"wall": soup(pts, nrm, w, cell_data={"trac": trac})}, global_data=gd)
    with pytest.raises(ValueError, match="cell_data is missing the declared fields \\['tag', 'trac'\\]"):
        m(boundary_meshes={"wall": soup(pts, nrm, w)}, global_data=gd)
    with pytest.raises(ValueError, match="\\['trac'\\]: expected shape \\(20, 3\\) \\(rank 1\\), got \\(20, 2\\)"):
        m(boundary_meshes={"wall": soup(pts, nrm, w, cell_data={"trac": trac[:, :2], "tag": tag})}, global_data=gd)
    ### extras in cell_data are ignored
    out = m(boundary_meshes={"wall": soup(pts, nrm, w, cell_data={"trac": trac, "tag": tag, "extra": tag})}, global_data=gd)
    assert out.point_data["p"].shape == (20,)
    ### every boundary mesh must carry every declared field
    with pytest.raises(ValueError, match="boundary_meshes\\['b'\\].cell_data is missing"):
        m(
            boundary_meshes={
                "a": soup(pts, nrm, w, cell_data={"trac": trac, "tag": tag}),
                "b": soup(pts + 10, nrm, w, cell_data={"trac": trac}),
            },
            global_data=gd,
        )
    ### query fields are required where the model declares them
    mq = _model(interior=True, query_schema={"sdf": {"rank": 0}})
    q = pts[:7] * 2
    with pytest.raises(ValueError, match="prediction_points.point_data is missing the declared fields \\['sdf'\\]"):
        mq(boundary_meshes={"wall": soup(pts, nrm, w)}, prediction_points=points_mesh(q), global_data=gd)
    with pytest.raises(ValueError, match="\\['sdf'\\]: expected shape \\(7,\\) \\(rank 0\\)"):
        mq(boundary_meshes={"wall": soup(pts, nrm, w)}, prediction_points=points_mesh(q, {"sdf": q}), global_data=gd)
    with pytest.raises(ValueError, match="prediction_points has 2 spatial dimensions"):
        mq(boundary_meshes={"wall": soup(pts, nrm, w)}, prediction_points=points_mesh(q[:, :2], {"sdf": q[:, 0]}), global_data=gd)


def test_fused_kernel_auto_falls_back_with_tensor_inputs():
    """The fused kernel does not take rank-2 inputs: auto falls back to eager (off CUDA
    it does anyway)."""
    base = dict(output_schema=OUT, global_schema={"g": {"rank": 1}}, hidden=32, n_layers=1, n_slices=8)
    m = ISLA(interior=True, query_schema={"S": {"rank": 2}}, **base)
    assert m.geo_kernel == "auto"
    assert m.resolve_geo_kernel(torch.zeros(1, 4, 3)) == "eager"


def test_rank0_query_inputs_are_scaled_by_the_gauge():
    """A rank-0 query field is a length: rescaling the geometry (points x s, measures
    x s^2) together with the field leaves the output unchanged, while rank-1 query
    fields are never scaled (they are physical vectors, not lengths) and a rescale
    of the geometry alone changes the output. The scaled pair takes two embedding
    channels per rank-0 field, after the built-in distance pair."""
    m = _model(interior=True, query_schema={"sdf": {"rank": 0}, "u": {"rank": 1}})
    assert m.qin.n_features == 2 + (2 + 1) and m.qt.embed[0].in_features == 2 + m.qin.n_features
    pts, nrm, w, _ = cloud(60)
    mesh = soup(pts, nrm, w)
    q, _, _, _ = cloud(12, seed=1)
    drv = torch.tensor([1.0, 0.3, 0.0], dtype=D)
    sdf = torch.randn(12, dtype=D) * 0.3
    u = torch.randn(12, 3, dtype=D)

    def call(k_geo, k_sdf):
        with torch.no_grad():
            return outputs(
                m(
                    boundary_meshes={"wall": transform(mesh, scale=k_geo)},
                    prediction_points=points_mesh(k_geo * q, {"sdf": k_sdf * sdf, "u": u}),
                    global_data={"g": drv},
                )
            )

    s = 3.7
    base, scaled, geo_only = call(1.0, 1.0), call(s, s), call(s, 1.0)
    assert all(torch.allclose(scaled[k], base[k], atol=1e-10) for k in base)
    assert not torch.allclose(geo_only["p"], base["p"], atol=1e-6)


def test_fields_keep_their_declaration_order_as_columns():
    """The field inputs fill the seed, and the head's coefficients fill the
    outputs, in declaration order, not in key order: with the embedding reading
    only its first seed column and the head writing only its first coefficient,
    the first declared input alone moves the first declared output."""
    m = ISLA(
        output_schema={"z": {"rank": 0}, "a": {"rank": 0}},
        boundary_schema={"y": {"rank": 0}, "b.c": {"rank": 0}},
        hidden=16,
        n_layers=1,
        n_slices=4,
    ).double().eval()
    with torch.no_grad():
        m.embed[0].weight[:, 1:] = 0
        m.head.weight[1:] = 0
        m.head.bias[1:] = 0
    pts, nrm, w, _ = cloud(30, seed=0)
    y, c = torch.rand(30, dtype=D), torch.rand(30, dtype=D)

    def run(y, c):
        with torch.no_grad():
            mesh = soup(pts, nrm, w, cell_data={"y": y, "b": {"c": c}})
            return outputs(m(boundary_meshes={"wall": mesh}))

    base = run(y, c)
    assert list(base) == ["z", "a"]
    assert torch.equal(base["a"], torch.zeros_like(base["a"])) and base["z"].abs().max() > 0
    assert torch.equal(run(y, c + 1)["z"], base["z"])
    assert not torch.allclose(run(y + 1, c)["z"], base["z"])
