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

"""Tests for shared datapipe indexing helpers."""

import pytest
import torch

from physicsnemo.datapipes._indexing import (
    _cyclic_block_indices,
    _subsample_indices,
    _uniform_indices,
)


class TestCyclicBlockIndices:
    def test_no_wrap(self):
        indices = _cyclic_block_indices(10, 4, start=3)
        assert indices.tolist() == [3, 4, 5, 6]

    def test_wraps_past_end(self):
        indices = _cyclic_block_indices(10, 4, start=8)
        assert indices.tolist() == [8, 9, 0, 1]

    def test_full_range_keeps_natural_order(self):
        indices = _cyclic_block_indices(
            5,
            5,
            generator=torch.Generator().manual_seed(0),
        )
        assert indices.tolist() == [0, 1, 2, 3, 4]

    def test_random_start_is_deterministic(self):
        generator_a = torch.Generator().manual_seed(42)
        generator_b = torch.Generator().manual_seed(42)
        for _ in range(20):
            indices_a = _cyclic_block_indices(64, 8, generator=generator_a)
            indices_b = _cyclic_block_indices(64, 8, generator=generator_b)
            assert torch.equal(indices_a, indices_b)

    def test_random_start_avoids_scalar_readback(self, monkeypatch):
        def fail_item(_):
            raise AssertionError("Tensor.item() must not be called")

        with monkeypatch.context() as context:
            context.setattr(torch.Tensor, "item", fail_item)
            indices = _cyclic_block_indices(
                10,
                4,
                generator=torch.Generator().manual_seed(2),
            )

        assert indices.tolist() == [8, 9, 0, 1]

    def test_inclusion_probability_is_exactly_uniform(self):
        # Over all N starts, every element appears in exactly k blocks, so
        # each inclusion probability is exactly k/N.
        total, k = 11, 4
        counts = torch.zeros(total, dtype=torch.long)
        for start in range(total):
            counts[_cyclic_block_indices(total, k, start=start)] += 1
        assert (counts == k).all()

    @pytest.mark.parametrize(
        ("total", "k"),
        [(-1, 0), (5, -1), (5, 6)],
    )
    def test_invalid_sizes_raise(self, total, k):
        with pytest.raises(ValueError):
            _cyclic_block_indices(total, k)

    def test_start_and_generator_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _cyclic_block_indices(
                10,
                4,
                start=3,
                generator=torch.Generator().manual_seed(0),
            )


class TestUniformIndices:
    @pytest.mark.parametrize(("total", "k"), [(100, 10), (100, 60), (7, 7), (5, 0)])
    def test_sorted_distinct_in_range(self, total, k):
        idx = _uniform_indices(total, k, generator=torch.Generator().manual_seed(0))
        assert idx.shape == (k,)
        assert (idx[1:] > idx[:-1]).all()
        assert ((idx >= 0) & (idx < total)).all()

    def test_deterministic_with_generator(self):
        a = _uniform_indices(1000, 50, generator=torch.Generator().manual_seed(3))
        b = _uniform_indices(1000, 50, generator=torch.Generator().manual_seed(3))
        assert torch.equal(a, b)

    @pytest.mark.parametrize("k", [3, 9])
    def test_inclusion_probability_is_uniform(self, k):
        # k=3 exercises the rejection path (4k < total), k=9 the randperm path.
        total, trials = 12, 6000
        generator = torch.Generator().manual_seed(0)
        counts = torch.zeros(total)
        for _ in range(trials):
            counts[_uniform_indices(total, k, generator=generator)] += 1
        torch.testing.assert_close(
            counts / trials, torch.full((total,), k / total), atol=0.03, rtol=0
        )

    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError):
            _uniform_indices(5, 6)


class TestSubsampleIndices:
    def test_block_matches_cyclic(self):
        a = _subsample_indices(
            64, 8, "block", generator=torch.Generator().manual_seed(1)
        )
        b = _cyclic_block_indices(64, 8, generator=torch.Generator().manual_seed(1))
        assert torch.equal(a, b)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="subsample mode"):
            _subsample_indices(10, 2, "random")
