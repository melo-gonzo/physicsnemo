"""INTERNAL-GEOMETRY snapshot check (cpu partition): code_v3 + recipe_v3 import, the four sg model
yamls compose with their datasets and instantiate, one forward pass each on random inputs shaped by
the resolved forward_kwargs, and the recipe's fields adapter consumes the TensorDict output.

Runs with PYTHONPATH=code_v3:recipe_v3/src; the fused kernel needs CUDA, so the CPU pass forces the
eager region (the kernel is not what changed). Prints VERIFY-OK on success.
"""

import inspect
import sys

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

RECIPE = sys.argv[1]

from physicsnemo.experimental.models import ISLA  # noqa: E402
from physicsnemo.experimental.models.isla.model import (  # noqa: E402
    nearest_boundary_geometry,
)
from output_normalize import normalize_output_to_tensordict  # noqa: E402

sig = inspect.signature(ISLA.__init__).parameters
assert "interior_geometry" in sig and "interior_nn_chunk" in sig, list(sig)
print("class:", ISLA.__module__, "interior_geometry default", sig["interior_geometry"].default)

### nearest-boundary geometry against brute force
g = torch.Generator().manual_seed(0)
pts = torch.randn(1, 500, 3, generator=g, dtype=torch.float64)
nrm = torch.nn.functional.normalize(torch.randn(1, 500, 3, generator=g, dtype=torch.float64), dim=-1)
q = torch.randn(1, 70, 3, generator=g, dtype=torch.float64) * 2
d, u = nearest_boundary_geometry(q, pts, nrm, chunk=16)
dist = (q[:, :, None] - pts[:, None]).norm(dim=-1)
idx = dist.argmin(-1)
assert torch.allclose(d.abs(), dist.gather(-1, idx[..., None])[..., 0], atol=1e-13)
print("nearest_boundary_geometry: brute-force match")

LANES = [
    ("isla_volume_ref", "drivaer_ml_volume_reference", []),
    ("isla_volume_ref_provided", "drivaer_ml_volume_reference", []),
    ("isla_volume_airfrans_ref", "airfrans_volume", ["train_split=full_train", "val_split=full_test"]),
    ("isla_volume_airfrans_ref_provided", "airfrans_volume", ["train_split=full_train", "val_split=full_test"]),
]
n, nq = 300, 120
for model, dataset, extra in LANES:
    with initialize_config_dir(config_dir=f"{RECIPE}/conf", version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"model={model}", f"dataset={dataset}", "+out_dim=5", "++model.geo_kernel=eager", *extra],
        )
    fk_spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    targets = OmegaConf.to_container(OmegaConf.load(f"{RECIPE}/datasets/{dataset}.yaml").targets, resolve=True)
    m = instantiate(cfg.model).eval()
    assert set(m.output_specs) == set(targets), (list(m.output_specs), list(targets))
    provided = model.endswith("_provided")
    assert m.interior_geometry == ("provided" if provided else "nearest_boundary"), m.interior_geometry
    assert ("query_normals" in fk_spec) == provided and ("query_inputs" in fk_spec) == provided, fk_spec
    fk = dict(
        points=torch.randn(1, n, 3),
        normals=torch.nn.functional.normalize(torch.randn(1, n, 3), dim=-1),
        measure_weights=torch.rand(1, n) + 0.5,
        global_inputs={k: (torch.randn(1, 3) if m.global_specs[k].rank == 1 else torch.rand(1)) for k in fk_spec["global_inputs"]},
        query_points=torch.randn(1, nq, 3) * 2,
    )
    if provided:
        fk["query_normals"] = torch.nn.functional.normalize(torch.randn(1, nq, 3), dim=-1)
        fk["query_inputs"] = {"sdf": torch.rand(1, nq)}
    with torch.no_grad():
        out = m(**fk)
    td = normalize_output_to_tensordict(out, targets, "fields")
    assert td.batch_size == torch.Size([1, nq]) and set(td.keys()) == set(targets), (td.batch_size, list(td.keys()))
    assert all(torch.isfinite(v).all() for v in td.values())
    print(f"{model} x {dataset}: forward ok, outputs {list(td.keys())}, {sum(p.numel() for p in m.parameters())/1e6:.1f}M params, fk={sorted(fk_spec)}")
print("VERIFY-OK")
