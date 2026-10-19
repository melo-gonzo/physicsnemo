"""GLOBE-SCORE scored-set identity check (2026-10-19).

Builds the validation loader exactly as the recipe's src/infer.py does (Hydra config `infer`, the given overrides,
build_dataloaders, then dataset[idx] for idx in the sampler) and prints, per case, a SHA-1 of the scored interior
points and of each boundary's points and cells, plus the target fields' hash. Run it once per (code, recipe, model)
triple: the GLOBE arm and the reference arms it is compared with. Identical hashes mean identical scored sets and
identical boundary inputs. Nothing is trained or scored.

usage: PYTHONPATH=<code>:<recipe>/src python scoredset_check.py <recipe dir> <max cases or 0 = all> <hydra overrides...>
"""

import os

### Run as a plain single process: a Slurm batch step's SLURM_* variables (without SLURM_LAUNCH_NODE_IPADDR) would send
### DistributedManager down its Slurm path, and RANK / WORLD_SIZE without a visible GPU down a failing NCCL one.
for _k in list(os.environ):
    if _k.startswith(("SLURM_", "OMPI_", "PMI_", "PMIX_")) or _k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        del os.environ[_k]
import hashlib
import json
import sys
from pathlib import Path

import torch

RECIPE = Path(sys.argv[1]).resolve()
MAX = int(sys.argv[2])
OVERRIDES = sys.argv[3:]

from hydra import compose, initialize_config_dir  # noqa: E402

sys.path.insert(0, str(RECIPE / "src"))
from datasets import build_dataloaders  # noqa: E402

from physicsnemo import datapipes  # noqa: E402,F401  (registers ${dp:...})
from physicsnemo.distributed import DistributedManager  # noqa: E402


def h(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().cpu().contiguous().float().numpy().tobytes()).hexdigest()[:16]


def hi(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().cpu().contiguous().long().numpy().tobytes()).hexdigest()[:16]


DistributedManager.initialize()   # single process (see above)
with initialize_config_dir(version_base=None, config_dir=str(RECIPE / "conf")):
    cfg = compose(config_name="infer", overrides=OVERRIDES)
_, val_loader, _, info = build_dataloaders(cfg)
dataset, sampler = val_loader.dataset, val_loader.sampler
print(json.dumps({"phase": "config", "recipe": str(RECIPE), "overrides": OVERRIDES, "n_cases": len(sampler),
                  "targets": {k: str(v) for k, v in info["targets"].items()}}), flush=True)
rows = []
for i, idx in enumerate(sampler):
    if MAX and i >= MAX:
        break
    domain, meta = dataset[idx]
    row = {"i": i, "idx": int(idx), "n_points": int(domain.interior.points.shape[0]),
           "points": h(domain.interior.points),
           "targets": {k: h(domain.interior.point_data[k]) for k in sorted(info["targets"])}}
    ### Planar comparison across the 2-D (RestrictToPlane) and 3-D (z = 0) AirfRANS files: the in-plane components.
    p = domain.interior.points
    if p.shape[-1] == 2 or bool((p[:, 2] == 0).all()):
        row["points_xy"] = h(p[:, :2])
        row["targets_xy"] = {k: h(v[..., :2] if v.ndim == 2 else v) for k, v in
                             ((k, domain.interior.point_data[k]) for k in sorted(info["targets"]))}
    for name in sorted(domain.boundaries.keys()):
        b = domain.boundaries[name]
        row[f"bnd.{name}"] = f"{b.n_cells}:{h(b.points)}:{hi(b.cells)}"
    gd = domain.global_data
    row["global"] = {k: [round(float(x), 6) for x in gd[k].reshape(-1).tolist()] for k in ("U_inf_dir", "Re", "U_inf_mag") if k in gd.keys()}
    rows.append(row)
    print(json.dumps(row), flush=True)
digest = hashlib.sha1(json.dumps([{k: v for k, v in r.items() if k not in ("global", "points_xy", "targets_xy")} for r in rows]).encode()).hexdigest()[:16]
xy = [{"idx": r["idx"], "points_xy": r.get("points_xy"), "targets_xy": r.get("targets_xy")} for r in rows]
digest_xy = hashlib.sha1(json.dumps(xy).encode()).hexdigest()[:16] if all(r.get("points_xy") for r in rows) else None
print(json.dumps({"phase": "summary", "cases": len(rows), "digest": digest, "digest_xy": digest_xy}), flush=True)
