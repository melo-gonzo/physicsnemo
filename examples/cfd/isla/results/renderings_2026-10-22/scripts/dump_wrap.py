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

"""RENDER-090 (2026-10-22): run the recipe's own ``src/infer.py`` (the scorer's entry point, unmodified) on one chosen case
and write its truth and prediction arrays as ``.npz`` instead of ``.pdmsh``. renderings_2026-10-18/scripts/dump_wrap.py with
the scorer's data stream replayed, so the case is drawn exactly as the scorer drew it.

Only two names inside the imported ``infer`` module are wrapped; the inference loop, the model, the checkpoint load, the
norm-stats restore, the metrics and the re-dimensionalization are the scorer's:

* ``infer.build_dataloaders``: the validation loader is kept, but its sampler is replaced by the case indices given in
  ``FIG_SAMPLES`` (comma-separated, the five-digit prefix of the scorer's sample ids). With ``FIG_REPLAY=1`` the dataset's
  ``__getitem__`` replays the scorer's sequence of reads on the global torch RNG before it returns a case: infer.py reads
  the shard's first sample once (``_spatial_dims_of_first_sample``) and then every sample in sampler order, so the first
  read here returns the full sampler's first sample, and each later read for case ``k`` first reads (and discards) every
  sample before ``k`` in the full order. A pipeline with a random draw (DrivAerML and SHIFT-SUV cell subsampling, the
  DrivAerML Volume and AirfRANS point draws) then draws the case's cells or points exactly as the scorer did, which the
  scorer-record check in the renderers confirms (the dump's metrics.jsonl row against the scorer's).
* ``infer.attach_and_save``: instead of saving a DomainMesh, saves ``<FIG_OUT>/<sample_id>.npz`` holding the interior
  (prediction) points, every ``pred_<field>`` and ``true_<field>`` in physical units (float32), the case's global_data,
  the interior's sdf and measures, and, under the posed protocol, the rotation ``pose_R`` and translation ``pose_t`` that
  ``FixedRandomPose`` applied to the case (its ``_pose`` is wrapped; the last call is the case's own).

usage (the scorer's command line, with +save_predictions=true and Hydra's --config-path <recipe>/conf --config-name infer):
  FIG_SAMPLES=00079 FIG_REPLAY=1 FIG_OUT=/path/out PYTHONPATH=<code>:<recipe>/src python dump_wrap.py model=... dataset=... \
      run_id=... infer_split=val +save_predictions=true sampling_resolution=200000 precision=float32 checkpoint_dir=... output_dir=...
"""

import os
import types

import numpy as np

import infer  # the recipe's src/infer.py (on PYTHONPATH)

SAMPLES = [int(s) for s in os.environ["FIG_SAMPLES"].split(",") if s.strip()]
REPLAY = os.environ.get("FIG_REPLAY", "0") == "1"
OUT = os.environ["FIG_OUT"]
os.makedirs(OUT, exist_ok=True)
POSES = []

try:
    import domain_transforms as _dt  # recipe src (posed protocol)

    _orig_pose = _dt.FixedRandomPose._pose

    def _pose(self, global_data, points):
        R, t = _orig_pose(self, global_data, points)
        POSES.append(
            (
                R.detach().double().cpu().numpy(),
                None if t is None else t.detach().double().cpu().numpy(),
            )
        )
        return R, t

    _dt.FixedRandomPose._pose = _pose
except Exception as e:  # noqa: BLE001
    print("FIG: no FixedRandomPose hook:", repr(e), flush=True)

_orig_build = infer.build_dataloaders


def _replaying(ds, full):
    """Give ``ds`` a ``__getitem__`` that replays the scorer's read sequence over ``full`` (see the module docstring)."""
    base = type(ds)
    state = {"first": True, "pos": 0}

    def getitem(self, idx):
        if state["first"]:
            state["first"] = False
            print(f"FIG: replay probe read of sample {full[0]}", flush=True)
            return base.__getitem__(self, full[0])
        skipped = 0
        while full[state["pos"]] != idx:
            base.__getitem__(self, full[state["pos"]])
            state["pos"] += 1
            skipped += 1
        state["pos"] += 1
        print(f"FIG: replayed {skipped} reads before sample {idx}", flush=True)
        return base.__getitem__(self, idx)

    ds.__class__ = type(f"Replay{base.__name__}", (base,), {"__getitem__": getitem})


def build_dataloaders(cfg):
    """The scorer's loaders with the validation sampler restricted to FIG_SAMPLES."""
    tl, vl, norm, info = _orig_build(cfg)
    full = list(vl.sampler)
    missing = [s for s in SAMPLES if s not in full]
    assert not missing, (
        f"FIG_SAMPLES {missing} not in the {cfg.infer_split} sampler ({len(full)} samples)"
    )
    order = sorted(SAMPLES, key=full.index)
    print(
        f"FIG: restricting {len(full)} samples to {order} (replay={REPLAY})", flush=True
    )
    if REPLAY:
        _replaying(vl.dataset, full)
    vl2 = types.SimpleNamespace(
        dataset=vl.dataset, sampler=order, collate_fn=vl.collate_fn
    )
    return tl, vl2, norm, info


def _np(x):
    return x.detach().float().cpu().numpy()


def attach_and_save(
    domain, pred_phys, true_phys, target_config, out_path, *, rescale_geometry
):
    """Write the case as <FIG_OUT>/<sample_id>.npz in place of the scorer's DomainMesh save."""
    sid = os.path.basename(str(out_path)).rsplit(".", 1)[0]
    arr = {"points": _np(domain.interior.points)}
    for prefix, td in (("pred_", pred_phys), ("true_", true_phys)):
        for key, val in td.items(include_nested=True, leaves_only=True):
            name = key if isinstance(key, str) else ".".join(key)
            arr[prefix + name] = _np(val)
    for key, val in domain.global_data.items(include_nested=True, leaves_only=True):
        name = key if isinstance(key, str) else ".".join(key)
        try:
            arr["global." + name] = _np(val)
        except Exception:  # noqa: BLE001
            pass
    pd = domain.interior.point_data
    for key in pd.keys(include_nested=True, leaves_only=True):
        name = key if isinstance(key, str) else ".".join(key)
        if name in ("sdf",) or name.endswith("measure"):
            arr["interior." + name] = _np(pd[key])
    if POSES:
        R, t = POSES[-1]
        arr["pose_R"] = R
        if t is not None:
            arr["pose_t"] = t
    np.savez_compressed(os.path.join(OUT, sid + ".npz"), **arr)
    print(
        f"FIG: wrote {sid}.npz keys={sorted(arr)} n={arr['points'].shape}", flush=True
    )


infer.build_dataloaders = build_dataloaders
infer.attach_and_save = attach_and_save

if __name__ == "__main__":
    infer.main()
