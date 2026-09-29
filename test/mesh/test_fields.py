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

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import FieldLayout, RankSpec, ScalarVectorFields
from physicsnemo.mesh.fields import (
    flatten_rank_spec,
    flatten_rank_specs,
    is_leaf_spec,
    rank_counts,
    ranks_from_tensordict,
    validate_data_contains_ranks,
    validate_rank_spec,
)


def _mixed_fields(n: int = 4) -> TensorDict:
    return TensorDict(
        {
            # Deliberately not in lexicographic order.
            "wind": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
            "state": {"temperature": torch.arange(n, dtype=torch.float32) + 20},
            "displacement": -torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
            "pressure": torch.arange(n, dtype=torch.float32) + 100,
            "ignored": torch.ones(n),
        },
        batch_size=[n],
    )


def test_rank_spec_helpers():
    ranks = {"wind": 1, "state": {"temperature": 0}, "pressure": 0}
    assert flatten_rank_spec(ranks) == {
        "wind": 1,
        "state.temperature": 0,
        "pressure": 0,
    }
    assert rank_counts(ranks) == {0: 2, 1: 1}
    assert ranks_from_tensordict(_mixed_fields()) == {
        "wind": 1,
        "state": {"temperature": 0},
        "displacement": 1,
        "pressure": 0,
        "ignored": 0,
    }


def test_validate_rank_spec_and_data_reports_schema_errors():
    validate_rank_spec({"scalar": 0, "tensor": 2})
    with pytest.raises(ValueError, match="must be one of .* got 2"):
        validate_rank_spec({"tensor": 2}, allowed_ranks=(0, 1))
    with pytest.raises(TypeError, match="must be an integer"):
        validate_rank_spec({"bad": True})
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_rank_spec({"bad": -1})

    data = TensorDict({"pressure": torch.zeros(3, 2)}, batch_size=[3])
    with pytest.raises(ValueError) as error:
        validate_data_contains_ranks(
            data=data,
            declared_ranks={"pressure": 0, "velocity": 1},
            source_label="boundary data",
        )
    assert str(error.value) == (
        "boundary data does not contain its declared rank spec:\n"
        "  - missing leaf 'velocity' (declared rank 1)\n"
        "  - rank mismatch for 'pressure': declared 0, got 1"
    )


def test_field_layout_pack_is_deterministic_and_round_trips():
    ranks = {
        "wind": 1,
        "state": {"temperature": 0},
        "pressure": 0,
        "displacement": 1,
    }
    reordered_ranks = {
        "pressure": 0,
        "displacement": 1,
        "state": {"temperature": 0},
        "wind": 1,
    }
    data = _mixed_fields()
    layout = FieldLayout(ranks, spatial_dim=3)
    reordered_layout = FieldLayout(reordered_ranks, spatial_dim=3)

    assert layout.scalar_names == ("pressure", "state.temperature")
    assert layout.vector_names == ("displacement", "wind")
    assert layout.flat_rank_spec == {
        "displacement": 1,
        "pressure": 0,
        "state.temperature": 0,
        "wind": 1,
    }

    packed = layout.pack(data)
    reordered = reordered_layout.pack(data)
    torch.testing.assert_close(packed.scalars, reordered.scalars)
    torch.testing.assert_close(packed.vectors, reordered.vectors)
    torch.testing.assert_close(
        packed.scalars,
        torch.stack((data["pressure"], data["state", "temperature"]), dim=-1),
    )
    torch.testing.assert_close(
        packed.vectors,
        torch.stack((data["displacement"], data["wind"]), dim=-2),
    )

    unpacked = layout.unpack(packed)
    assert set(unpacked.keys()) == {"displacement", "pressure", "state", "wind"}
    for key in ("pressure", ("state", "temperature"), "displacement", "wind"):
        torch.testing.assert_close(unpacked[key], data[key])


def test_field_layout_preserves_polar_vector_transformation():
    data = _mixed_fields()
    layout = FieldLayout(
        {"pressure": 0, "wind": 1, "displacement": 1},
        spatial_dim=3,
    )
    packed = layout.pack(data)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_data = data.clone()
    rotated_data["wind"] = data["wind"] @ rotation.T
    rotated_data["displacement"] = data["displacement"] @ rotation.T
    rotated = layout.pack(rotated_data)

    torch.testing.assert_close(rotated.scalars, packed.scalars)
    torch.testing.assert_close(rotated.vectors, packed.vectors @ rotation.T)


@pytest.mark.parametrize(
    ("ranks", "expected_scalar_shape", "expected_vector_shape"),
    [
        ({"pressure": 0}, (4, 1), (4, 0, 3)),
        ({"wind": 1}, (4, 0), (4, 1, 3)),
    ],
)
def test_field_layout_supports_scalar_or_vector_only(
    ranks, expected_scalar_shape, expected_vector_shape
):
    layout = FieldLayout(ranks, spatial_dim=3)
    packed = layout.pack(_mixed_fields())
    assert packed.scalars.shape == expected_scalar_shape
    assert packed.vectors.shape == expected_vector_shape
    round_trip = layout.pack(layout.unpack(packed))
    torch.testing.assert_close(round_trip.scalars, packed.scalars)
    torch.testing.assert_close(round_trip.vectors, packed.vectors)


def test_field_layout_rejects_non_vector_rank_and_wrong_vector_dimension():
    with pytest.raises(ValueError, match="must be one of .* got 2"):
        FieldLayout({"stress": 2}, spatial_dim=3)

    layout = FieldLayout({"velocity": 1}, spatial_dim=3)
    wrong_dim = TensorDict({"velocity": torch.zeros(5, 2)}, batch_size=[5])
    with pytest.raises(ValueError, match=r"must have shape \(5, 3\)"):
        layout.pack(wrong_dim)


def test_field_layout_validates_packed_shapes_and_dtype():
    layout = FieldLayout({"pressure": 0, "velocity": 1}, spatial_dim=3)
    with pytest.raises(ValueError, match="vectors must have shape"):
        layout.unpack(
            ScalarVectorFields(
                scalars=torch.empty(5, 1),
                vectors=torch.empty(4, 1, 3),
            )
        )
    with pytest.raises(ValueError, match="must have the same dtype"):
        layout.unpack(
            ScalarVectorFields(
                scalars=torch.empty(5, 1, dtype=torch.float32),
                vectors=torch.empty(5, 1, 3, dtype=torch.float64),
            )
        )


def test_rank_spec_parse_accepts_int_rank_spec_and_mapping():
    assert RankSpec.parse(1) == RankSpec(rank=1)
    spec = RankSpec(rank=2, symmetric=True, parity="odd")
    assert RankSpec.parse(spec) is spec
    assert RankSpec.parse({"rank": 2, "symmetric": True, "parity": "odd"}) == spec
    assert RankSpec.parse({"rank": 0}) == RankSpec(rank=0)

    assert RankSpec(rank=0).shape(3) == ()
    assert RankSpec(rank=0).size(3) == 1
    assert RankSpec(rank=1).shape(2) == (2,)
    assert RankSpec(rank=2).shape(3) == (3, 3)
    assert RankSpec(rank=2).size(3) == 9


@pytest.mark.parametrize(
    ("spec", "error", "match"),
    [
        (True, TypeError, "got a bool"),
        (1.0, TypeError, "must be an integer rank"),
        (-1, ValueError, "must be non-negative"),
        ({"symmetric": True}, ValueError, "must contain a 'rank' key"),
        ({"rank": 1, "channels": 3}, ValueError, r"unknown keys \['channels'\]"),
        ({"rank": True}, TypeError, "rank must be an integer"),
        ({"rank": 1, "parity": "axial"}, ValueError, "parity must be 'even' or 'odd'"),
        ({"rank": 1, "symmetric": True}, ValueError, "only meaningful for rank >= 2"),
        ({"rank": 2, "symmetric": 1}, TypeError, "symmetric must be a bool"),
    ],
)
def test_rank_spec_parse_rejects_invalid_specs(spec, error, match):
    with pytest.raises(error, match=match):
        RankSpec.parse(spec, source_label="outputs['x']")


def test_is_leaf_spec_distinguishes_groups_from_leaf_mappings():
    assert is_leaf_spec(0)
    assert is_leaf_spec(RankSpec(rank=1))
    assert is_leaf_spec({"rank": 2})
    assert not is_leaf_spec({"pressure": 0})
    assert not is_leaf_spec({})


def test_flatten_rank_specs_treats_nested_and_dotted_names_alike():
    stress = RankSpec(rank=2, symmetric=True)
    nested = {
        "surface": {"pressure": 0, "shear": {"rank": 1}},
        "stress": stress,
        "vorticity": {"rank": 1, "parity": "odd"},
    }
    dotted = {
        "surface.pressure": 0,
        "surface.shear": 1,
        "stress": {"rank": 2, "symmetric": True},
        "vorticity": RankSpec(rank=1, parity="odd"),
    }
    expected = {
        "surface.pressure": RankSpec(rank=0),
        "surface.shear": RankSpec(rank=1),
        "stress": stress,
        "vorticity": RankSpec(rank=1, parity="odd"),
    }
    assert flatten_rank_specs(nested) == expected
    assert flatten_rank_specs(dotted) == expected
    assert list(flatten_rank_specs(nested)) == list(expected)
    assert flatten_rank_spec(dotted) == {
        "surface.pressure": 0,
        "surface.shear": 1,
        "stress": 2,
        "vorticity": 1,
    }
    assert rank_counts(nested) == {0: 1, 1: 2, 2: 1}

    assert flatten_rank_specs({"a/b": {"c": 0}}, sep="/") == {"a/b/c": RankSpec(rank=0)}
    assert flatten_rank_spec({"a.b": 0}, sep="/") == {"a.b": 0}


def test_flatten_rank_specs_rejects_group_leaf_conflicts_and_bad_names():
    with pytest.raises(ValueError, match="'surface' is a group of 'surface.pressure'"):
        flatten_rank_specs({"surface": 0, "surface.pressure": 0})
    with pytest.raises(ValueError, match="is a group of"):
        flatten_rank_specs(
            {"surface": {"pressure": {"mean": 0}}, "surface.pressure": 0}
        )
    with pytest.raises(ValueError, match="more than once"):
        flatten_rank_specs({"surface": {"pressure": 0}, "surface.pressure": 1})
    with pytest.raises(ValueError, match="empty path component"):
        flatten_rank_specs({"surface..pressure": 0})
    with pytest.raises(TypeError, match="must be strings"):
        flatten_rank_specs({1: 0})
    with pytest.raises(TypeError, match="must be a mapping"):
        flatten_rank_specs([("pressure", 0)])


def test_validate_rank_spec_accepts_widened_leaves():
    validate_rank_spec(
        {
            "pressure": 0,
            "velocity": RankSpec(rank=1),
            "stress": {"rank": 2, "symmetric": True},
            "surface": {"vorticity": {"rank": 1, "parity": "odd"}},
        }
    )
    validate_rank_spec({"velocity": {"rank": 1}}, allowed_ranks=(0, 1))
    with pytest.raises(ValueError, match="must be one of .* got 2"):
        validate_rank_spec({"stress": RankSpec(rank=2)}, allowed_ranks=(0, 1))
    with pytest.raises(ValueError, match=r"outputs\['stress'\] has unknown keys"):
        validate_rank_spec(
            {"stress": {"rank": 2, "channels": 2}}, source_label="outputs"
        )
    with pytest.raises(ValueError, match="parity must be 'even' or 'odd'"):
        validate_rank_spec({"surface": {"shear": {"rank": 1, "parity": "axial"}}})


def test_field_layout_accepts_rank_spec_leaves_and_dotted_names():
    data = _mixed_fields()
    reference = FieldLayout({"pressure": 0, "state": {"temperature": 0}, "wind": 1}, 3)
    layout = FieldLayout(
        {
            "pressure": RankSpec(rank=0),
            "state.temperature": {"rank": 0},
            "wind": {"rank": 1, "parity": "even"},
        },
        spatial_dim=3,
    )
    assert layout.flat_rank_spec == reference.flat_rank_spec
    packed = layout.pack(data)
    expected = reference.pack(data)
    torch.testing.assert_close(packed.scalars, expected.scalars)
    torch.testing.assert_close(packed.vectors, expected.vectors)
    unpacked = layout.unpack(packed)
    torch.testing.assert_close(
        unpacked["state", "temperature"], data["state", "temperature"]
    )


def test_field_layout_rejects_pseudovectors_and_symmetric_leaves():
    with pytest.raises(ValueError, match="has parity 'odd'; FieldLayout packs"):
        FieldLayout({"vorticity": {"rank": 1, "parity": "odd"}}, spatial_dim=3)
    with pytest.raises(ValueError, match="has parity 'odd'"):
        FieldLayout({"vorticity": RankSpec(rank=1, parity="odd")}, spatial_dim=3)
    with pytest.raises(ValueError, match="only meaningful for rank >= 2"):
        FieldLayout({"wind": {"rank": 1, "symmetric": True}}, spatial_dim=3)
    with pytest.raises(ValueError, match="must be one of .* got 2"):
        FieldLayout({"stress": {"rank": 2, "symmetric": True}}, spatial_dim=3)
    with pytest.raises(ValueError, match="is a group of"):
        FieldLayout({"surface": 0, "surface.pressure": 0}, spatial_dim=3)


def test_validate_data_contains_ranks_reports_schema_errors():
    data = TensorDict({"pressure": torch.zeros(3, 2)}, batch_size=[3])
    with pytest.raises(ValueError) as error:
        validate_data_contains_ranks(
            data=data,
            declared_ranks={"pressure": 0, "velocity": 1},
            source_label="boundary data",
        )
    assert str(error.value) == (
        "boundary data does not contain its declared rank spec:\n"
        "  - missing leaf 'velocity' (declared rank 1)\n"
        "  - rank mismatch for 'pressure': declared 0, got 1"
    )
