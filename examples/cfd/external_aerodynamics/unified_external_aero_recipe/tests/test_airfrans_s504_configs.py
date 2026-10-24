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

"""AF-EXTRAP S-RE and LRE (2026-10-22): the 504-case in-distribution control's dataset files airfrans_uvolume_{2d,ud}_s504.yaml
and the fixed-Reynolds-scale file airfrans_uvolume_2d_reynolds_lre.yaml."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

_RECIPE_ROOT = Path(__file__).resolve().parents[1]

#: DRIVE-FIX R1's standardized-speed transform of the baselines' full-split file (airfrans_uvolume_ud.yaml).
SPEED_STD = {
    "_target_": "${dp:SplitGlobalVector}",
    "field": "U_inf_mag",
    "names": ["U_inf_mag_std"],
    "mean": 62.116,
    "std": 17.914,
}

#: AF-EXTRAP S-RE (2026-10-22): the 504-case in-distribution control's constants (fitted on full_train_s504, LHR job 326139;
#: ISLA's refit with the far field, AF-WHOLE 2026-10-23, LHR job 361288).
S504 = {
    "manifest": "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0/kernel/afxs/manifest_s504.json",
    "nut": {"mean": 7.975e-4, "std": 2.404e-3},
    "speed": {"mean": 62.260, "std": 18.004},
    "isla": {
        "reynolds": {"mean": 9.066, "std": 0.1347},
        "output_scale": {"nut": 1.066e-07},
    },
}


@pytest.mark.parametrize("src", ["airfrans_uvolume_2d", "airfrans_uvolume_ud"])
def test_s504_file_is_its_source_with_its_manifest_and_constants(src):
    """airfrans_uvolume_{2d,ud}_s504.yaml is its full-split file with the manifest that holds full_train_s504 and the
    nut stats, and ISLA's isla_constants or the baselines' speed constants, fitted on those 504 cases."""
    datasets = _RECIPE_ROOT / "datasets"
    a = OmegaConf.to_container(
        OmegaConf.load(datasets / f"{src}_s504.yaml"), resolve=False
    )
    b = OmegaConf.to_container(OmegaConf.load(datasets / f"{src}.yaml"), resolve=False)
    b["manifest"] = S504["manifest"]
    (norm,) = [
        t
        for t in b["pipeline"]["transforms"]
        if t["_target_"] == "${dp:NormalizeMeshFields}"
    ]
    norm["fields"]["nut"] = {"type": "scalar", **S504["nut"]}
    if "_ud" in src:
        b["pipeline"]["transforms"][-1] = {**SPEED_STD, **S504["speed"]}
    else:
        b["isla_constants"] = S504["isla"]
    assert a == b


def test_lre_file_is_the_reynolds_file_with_the_fixed_reynolds_scale():
    """AF-EXTRAP LRE (2026-10-22): airfrans_uvolume_2d_reynolds_lre.yaml is airfrans_uvolume_2d_reynolds.yaml with ISLA's
    reynolds constants {mean 6, std 1}, so z = log10(Re_ISLA / 1e6), and nothing else changed."""
    datasets = _RECIPE_ROOT / "datasets"
    a = OmegaConf.to_container(
        OmegaConf.load(datasets / "airfrans_uvolume_2d_reynolds_lre.yaml"),
        resolve=False,
    )
    b = OmegaConf.to_container(
        OmegaConf.load(datasets / "airfrans_uvolume_2d_reynolds.yaml"), resolve=False
    )
    assert b["isla_constants"]["reynolds"] == {"mean": 9.084, "std": 0.06181}
    b["isla_constants"]["reynolds"] = {"mean": 6.0, "std": 1.0}
    assert a == b
