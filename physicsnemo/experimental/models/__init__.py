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

"""Experimental models for PhysicsNeMo.

Models in this subpackage are under active development and may have breaking API
changes between releases. Subpackages (``geotransolver``, ``globe``, ``isla``, ...)
are imported by their own paths, e.g. ``physicsnemo.experimental.models.isla.ISLA``.
The ``ISLA`` re-export below only keeps the research records' older
``physicsnemo.experimental.models.ISLA`` target working (``examples/cfd/isla/``);
main has no ``__init__`` here, so this file goes when ISLA is upstreamed.
"""

from .isla import ISLA

__all__ = ["ISLA"]
