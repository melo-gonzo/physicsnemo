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

"""RENDER-090 (2026-10-22): the committed copy of a routing probe (slices_probe_090.py output) within the repository's
5,000 KB file limit. A probe that stored at most N_DRAW tokens is copied unchanged. A larger one (DrivAerML Surface: all
200,000 tokens) keeps exactly the tokens the book draws, the 60,000-token subset that 03-isla.qmd's load_slices takes
(numpy default_rng(0).choice over the stored tokens, sorted), with:
  * points stored for those tokens only and keep = 0..n-1, so points[keep] is the drawn tokens' positions as before;
    token_index holds their indices among the n_tokens tokens of the forward pass;
  * the per-token fields (dominant, pi_max, top_pi) on those tokens;
  * top_slices and top_pi cut to the TOP largest territories of each probed layer (the book draws three); the json beside
    the npz keeps all eight;
  * measure and top_a_density dropped (the book reads neither).
Every per-slice field (q, qw, anchors, anchor_normals) and the json's per-layer statistics are over all tokens, unchanged.
The full probe output stays in the lane's scratch and on LHR ($T/book_figs_2026-10-22/routing/).
usage: python thin_routing.py <probe.npz> <out.npz>
"""

import sys

import numpy as np

N_DRAW = 60000
TOP = 3

src, dst = sys.argv[1:3]
z = dict(np.load(src))
if len(z["keep"]) <= N_DRAW:
    np.savez_compressed(dst, **z)
else:
    sub = np.sort(
        np.random.default_rng(0).choice(len(z["keep"]), N_DRAW, replace=False)
    )
    idx = z["keep"][sub]
    out = {k: v for k, v in z.items() if k not in ("measure", "top_a_density")}
    out.update(
        points=z["points"][idx],
        keep=np.arange(len(idx)),
        token_index=idx,
        n_tokens=np.array(len(z["points"])),
        dominant=z["dominant"][:, sub],
        pi_max=z["pi_max"][:, sub],
        top_pi=z["top_pi"][:, sub, :TOP],
        top_slices=z["top_slices"][:, :TOP],
    )
    np.savez_compressed(dst, **out)
print(dst, {k: v.shape for k, v in np.load(dst).items()})
