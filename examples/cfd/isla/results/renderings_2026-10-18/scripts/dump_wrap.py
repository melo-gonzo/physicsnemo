"""FIG lane (ISLA book overhaul, 2026-10-18): run the recipe's own ``src/infer.py`` (the scorer's entry point, unmodified)
on a few chosen validation samples and write their truth and prediction arrays as ``.npz`` instead of ``.pdmsh``.

Only two names inside the imported ``infer`` module are wrapped; the inference loop, the model, the checkpoint load,
the metrics and the re-dimensionalization are the scorer's:

* ``infer.build_dataloaders``: the validation loader is kept, but its sampler is replaced by the sampler indices given in
  ``FIG_SAMPLES`` (comma-separated, the five-digit prefix of the scorer's sample ids), in that order.
* ``infer.attach_and_save``: instead of saving a DomainMesh, saves ``<FIG_OUT>/<sample_id>.npz`` holding the interior
  (prediction) points, every ``pred_<field>`` and ``true_<field>`` in physical units (float32), the case's global_data
  scalars/vectors, and, when the posed protocol is active, the rotation ``pose_R`` and translation ``pose_t`` that
  ``FixedRandomPose`` applied to the case (captured by wrapping its ``_pose``).

The per-sample relative L2 errors are the scorer's own rows in ``<output_dir>/<run_id>/metrics.jsonl``.

usage (same command line as the scorer, e.g. kernel/a2r/score.sbatch, with +save_predictions=true, plus Hydra's
--config-path <recipe>/conf --config-name infer, since infer.main is called from this module):
  FIG_SAMPLES=00015 FIG_OUT=/path/out PYTHONPATH=<code>:<recipe>/src python dump_wrap.py model=... dataset=... \
      run_id=... infer_split=val +save_predictions=true sampling_resolution=200000 precision=float32 \
      checkpoint_dir=... output_dir=...
"""
import os
import types

import numpy as np
import torch

import infer  # the recipe's src/infer.py (on PYTHONPATH)

SAMPLES = [int(s) for s in os.environ["FIG_SAMPLES"].split(",") if s.strip()]
OUT = os.environ["FIG_OUT"]
os.makedirs(OUT, exist_ok=True)
POSES = []

try:
    import domain_transforms as _dt  # recipe src (posed protocol)

    _orig_pose = _dt.FixedRandomPose._pose

    def _pose(self, global_data, points):
        R, t = _orig_pose(self, global_data, points)
        POSES.append((R.detach().double().cpu().numpy(), None if t is None else t.detach().double().cpu().numpy()))
        return R, t

    _dt.FixedRandomPose._pose = _pose
except Exception as e:  # noqa: BLE001
    print("FIG: no FixedRandomPose hook:", repr(e), flush=True)

_orig_build = infer.build_dataloaders


def build_dataloaders(cfg):
    tl, vl, norm, info = _orig_build(cfg)
    full = list(vl.sampler)
    missing = [s for s in SAMPLES if s not in full]
    assert not missing, f"FIG_SAMPLES {missing} not in the {cfg.infer_split} sampler ({len(full)} samples)"
    print(f"FIG: restricting {len(full)} samples to {SAMPLES}", flush=True)
    vl2 = types.SimpleNamespace(dataset=vl.dataset, sampler=list(SAMPLES), collate_fn=vl.collate_fn)
    return tl, vl2, norm, info


def _np(x):
    return x.detach().float().cpu().numpy()


def attach_and_save(domain, pred_phys, true_phys, target_config, out_path, *, rescale_geometry):
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
    print(f"FIG: wrote {sid}.npz keys={sorted(arr)} n={arr['points'].shape}", flush=True)


infer.build_dataloaders = build_dataloaders
infer.attach_and_save = attach_and_save

if __name__ == "__main__":
    infer.main()
