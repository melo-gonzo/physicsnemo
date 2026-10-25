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

"""pytest configuration of the ISLA tests.

ISLA's CPU CI runs ``-m "not slow"``; the slow tests run in the GPU tier on a GB300.
The CUDA tests carry the ``cuda`` marker besides their skip. ``test/conftest.py``
registers both markers for the whole repository; they are registered here too so that
the ISLA directory stays self-contained (run from its own root, or copied with the
model)."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running (for optional exclusion)"
    )
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA (skipped if unavailable)"
    )
