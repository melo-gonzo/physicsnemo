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

"""End-to-end smoke test for the PR 2 pretraining pipeline.

Exercises the model side of the GeoPT pretraining pipeline against a
4-sample synthetic ``.pdmsh`` corpus built from the conftest sphere:

1. Build 4 synthetic samples via
   :func:`physicsnemo.experimental.pnm_pretraining.data.build_pretraining_sample`.
2. Hydra-compose ``conf/train.yaml`` with
   ``model=transolver_pretrain dataset=geopt_pretrain
   dataset_paths.geopt_pretrain=<corpus>`` and shrink the model.
3. Instantiate the wrapper via ``hydra.utils.instantiate``.
4. Walk the dataset's reader + ``WalkSampler`` transform manually to
   produce one full per-walk sample, then resolve the model template's
   ``forward_kwargs:`` block against that sample to build a real
   ``embedding`` / ``fx`` batch.
5. Run forward, verify the output shape matches ``(B, N, n_steps * 3)``.
6. Run backward + one optimizer step (no NaNs).
7. Save a ``.mdlus`` checkpoint.
8. Load that checkpoint into a freshly-constructed plain
   :class:`Transolver` via :func:`load_pretrained_backbone` with
   ``key_remap={"transolver.": ""}`` and
   ``exclude_layers=["trajectory_head"]`` — the PR 2.5 round-trip.

**Simplification (per ``pr2-recipe-extension-plan.md`` "Test plan" and
the prompt's documented fallback).** This test stops short of running
the recipe's full training loop (loss / metric / checkpoint manager).
Two reasons:

1. The recipe's :func:`train.main` initializes a global
   ``DistributedManager``, mutates ``sys.path``, and writes to a
   well-known output directory; running it under pytest is brittle and
   slow on CPU.
2. As of this commit, the dataset YAML (subagent E) declares
   ``targets: supervise: vector`` (3 channels); the model emits
   ``n_steps * 3`` channels. The recipe's
   :func:`output_normalize.split_concat_by_target` will reject the
   shape mismatch until E (or a follow-up PR) introduces a richer
   field-type tag (e.g. ``"trajectory"``) or reshapes both sides to
   ``(B, N, n_steps, 3)``. This test stops at the model boundary so
   the model-side smoke isn't blocked on that coordination.

The PR 2.5 round-trip part is **also** unit-tested standalone in
``test/experimental/pnm_pretraining/models/test_transolver_pretrain.py::test_backbone_round_trip_via_load_pretrained_backbone``.

Gated by ``PNM_PR2_E2E_SMOKE=1`` because building 4 ``.pdmsh`` samples
takes ~10-20 s on CPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from test.conftest import requires_module

_RECIPE_ROOT = (
    Path(__file__).resolve().parent.parent  # examples/.../unified_external_aero_recipe
)
_SRC = _RECIPE_ROOT / "src"

# Side-effect imports are deferred into the test body because the
# top-level skip needs to succeed even on CPU-only / warp-less hosts.
# We only mutate sys.path inside the test (mirrors conftest.py:46-47).


def _has_subagent_e_artifacts() -> tuple[bool, str]:
    """Check whether subagent E's deliverables are present in this worktree.

    Returns
    -------
    tuple
        ``(ok, reason)``: ``ok`` is True iff every required file exists.
    """
    required = {
        "datasets/geopt_pretrain.yaml": _RECIPE_ROOT
        / "datasets"
        / "geopt_pretrain.yaml",
    }
    missing: list[str] = []
    for desc, path in required.items():
        if not path.exists():
            missing.append(desc)
    try:
        from physicsnemo.experimental.pnm_pretraining.data import (
            WalkSampler,  # noqa: F401
        )
    except ImportError:
        missing.append("WalkSampler")
    if missing:
        return False, f"subagent E artifacts missing: {missing}"
    return True, ""


@pytest.fixture(scope="module")
def _smoke_gate() -> None:
    """Aggregate skip conditions for the e2e smoke."""
    if os.environ.get("PNM_PR2_E2E_SMOKE", "0") != "1":
        pytest.skip("PNM_PR2_E2E_SMOKE != 1 (set to enable the e2e smoke)")
    ok, reason = _has_subagent_e_artifacts()
    if not ok:
        pytest.skip(reason)


@requires_module("warp")
@requires_module("trimesh")
def test_pretraining_e2e_smoke(_smoke_gate: None, tmp_path: Path) -> None:
    """End-to-end model-side smoke for the PR 2 pretraining pipeline."""
    # Lazy imports so module-level collection works on hosts where the
    # recipe / heavy deps aren't importable. Runs only after the skip
    # gates above have cleared.
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

    import hydra
    import nondim  # noqa: F401, recipe-local
    import sdf  # noqa: F401, recipe-local
    import torch
    from forward_kwargs import resolve_forward_kwargs
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # Side-effect imports register the ${dp:...} resolver and the
    # recipe-local transforms (mirrors recipe conftest.py:52-54).
    import physicsnemo.datapipes  # noqa: F401
    from physicsnemo.experimental.pnm_pretraining.data import (
        WalkSampler,
        build_pretraining_sample,
        save_pretraining_sample,
    )
    from physicsnemo.experimental.pnm_pretraining.models import (
        TransolverPretrainBackbone,
    )
    from physicsnemo.models.transolver import Transolver
    from physicsnemo.utils.checkpoint import load_pretrained_backbone

    # ----- 1. Build a 4-sample synthetic .pdmsh corpus from a sphere. -----
    # Reuse the conftest sphere generator. Tiny config so the test
    # finishes in a few seconds on CPU.
    from test.experimental.pnm_pretraining.conftest import make_sphere

    sphere = make_sphere(n_rings=8, n_segments=16)
    import trimesh

    obj_path = tmp_path / "sphere.obj"
    trimesh.Trimesh(
        vertices=sphere.vertices, faces=sphere.indices, process=False
    ).export(str(obj_path))

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    tiny_kwargs = dict(
        n_volume_points=64,
        n_surface_points=16,
        n_independent_walks=2,
        n_jittered_per_base=1,  # n_walks = 2 * (1 + 1) = 4
        n_steps=2,
    )
    for sample_idx in range(4):
        sample = build_pretraining_sample(
            obj_path, seed=100 + sample_idx, **tiny_kwargs
        )
        save_pretraining_sample(sample, corpus_dir / f"sample_{sample_idx}.pdmsh")
    pdmsh_paths = sorted(corpus_dir.glob("*.pdmsh"))
    assert len(pdmsh_paths) == 4, f"expected 4 samples, got {len(pdmsh_paths)}"

    # ----- 2. Hydra-compose the recipe cfg with our overrides. -----
    # Compose `train.yaml` to pick up the model template + recipe-side
    # cfg keys. The dataset YAML lives in `datasets/` (not as a Hydra
    # config group) and is loaded separately via the recipe's
    # `load_dataset_config` helper, mirroring what `build_dataloaders`
    # does at runtime.
    overrides = [
        "model=transolver_pretrain",
        "dataset=geopt_pretrain",
        "+out_dim=6",  # n_steps=2 * 3 = 6 (production sets via build_dataloaders)
        "model.n_steps=2",
        "model.n_layers=2",
        "model.n_hidden=32",
        "model.n_head=4",
        "model.slice_num=16",
        "model.head_hidden_dim=16",
        "model.head_n_layers=1",
    ]
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"), version_base=None
    ):
        cfg = compose(config_name="train", overrides=overrides)

    assert cfg.input_type == "tensors"
    assert cfg.output_type == "tensors"
    assert cfg.dataset == "geopt_pretrain"

    # Load the dataset YAML separately and override the corpus path so
    # the reader looks at our synthetic corpus, not `???`.
    from datasets import load_dataset_config

    dataset_cfg = load_dataset_config(_RECIPE_ROOT / "datasets" / f"{cfg.dataset}.yaml")
    dataset_cfg.dataset_paths.geopt_pretrain = str(corpus_dir)
    # Provide `training.seed` (referenced by WalkSampler's `seed:` Hydra
    # interpolation) — the train.yaml provides it but the dataset YAML
    # is loaded standalone here.
    dataset_cfg = OmegaConf.merge(
        dataset_cfg, OmegaConf.create({"training": {"seed": 0}})
    )

    # ----- 3. Instantiate the wrapper via Hydra. -----
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")
    assert isinstance(model, TransolverPretrainBackbone)
    model.train()

    # ----- 4. Manually walk reader + WalkSampler to produce one sample. -----
    reader = hydra.utils.instantiate(dataset_cfg.pipeline.reader, _convert_="partial")
    walk_sampler = WalkSampler(seed=0)

    domain_mesh, _meta = reader[0]
    domain_mesh = walk_sampler(domain_mesh)
    # WalkSampler must have populated the three keys the model YAML
    # references in `forward_kwargs:`.
    pd = domain_mesh.interior.point_data
    assert "directions" in pd.keys()
    assert "step_lengths" in pd.keys()
    assert "supervise" in pd.keys()

    # ----- 5. Resolve forward_kwargs against the sample, run forward. -----
    # The recipe's collate adds a leading batch dim before calling
    # forward_kwargs; the test simulates that by manually unsqueezing
    # the resolved tensors. Resolve first against the unbatched sample,
    # then add the batch dim.
    forward_spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    resolved = resolve_forward_kwargs(forward_spec, domain_mesh)
    embedding = resolved["embedding"].unsqueeze(0)
    fx = resolved["fx"].unsqueeze(0)

    n_points = embedding.shape[1]
    out = model(embedding=embedding, fx=fx)
    assert out.shape == (1, n_points, tiny_kwargs["n_steps"] * 3)

    # ----- 6. Backward + one optimizer step. -----
    target = domain_mesh.interior.point_data["supervise"].unsqueeze(0).to(out.dtype)
    assert target.shape == out.shape, (
        f"target shape {tuple(target.shape)} != model output {tuple(out.shape)}"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = ((out - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss), f"loss not finite: {float(loss)}"

    # ----- 7. Save .mdlus checkpoint. -----
    runs_dir = tmp_path / "runs" / "smoke" / "checkpoints"
    runs_dir.mkdir(parents=True)
    ckpt_path = runs_dir / "TransolverPretrainBackbone.0.0.last.mdlus"
    model.save(str(ckpt_path))
    assert ckpt_path.exists(), f"checkpoint was not written to {ckpt_path}"

    # ----- 8. Round-trip: load into a plain Transolver. -----
    fine_tune_model = Transolver(
        functional_dim=4,
        out_dim=4,  # arbitrary fine-tune head width
        embedding_dim=7,
        n_layers=2,
        n_hidden=32,
        n_head=4,
        slice_num=16,
        use_te=False,
    )
    report = load_pretrained_backbone(
        fine_tune_model,
        str(ckpt_path),
        key_remap={"transolver.": ""},
        exclude_layers=["trajectory_head"],
        strict=False,
        verbose=False,
    )
    # The trajectory head must be excluded; the backbone preprocess
    # must have loaded.
    assert any("trajectory_head" in k for k in report["skipped_excluded"])
    assert "preprocess.layers.0.weight" in report["loaded"]
    # Bit-exact match on a sentinel backbone parameter.
    wrapper_w = model.transolver.preprocess.layers[0].weight
    plain_w = fine_tune_model.preprocess.layers[0].weight
    assert torch.equal(wrapper_w, plain_w), (
        "backbone preprocess weight did not transfer bit-exact"
    )
