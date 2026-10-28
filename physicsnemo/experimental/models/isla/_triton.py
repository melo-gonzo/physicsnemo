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

"""Triton, ISLA's optional dependency for its CUDA kernels.

Triton ships with the CUDA wheels of torch and is absent on CPU-only installs. It is
loaded through importlib so that the package's external-import contract sees no hard
dependency; the modules with ``@triton.jit`` kernels define them only when
``HAS_TRITON`` holds and fall back to their torch code otherwise. A broken install
counts as absent.
"""

import importlib

from physicsnemo.core.version_check import is_package_available

triton = tl = libdevice = None
HAS_TRITON = is_package_available("triton")
if HAS_TRITON:
    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
        libdevice = importlib.import_module("triton.language.extra.libdevice")
    except Exception:  # pragma: no cover - a broken triton install is treated as absent
        HAS_TRITON = False
