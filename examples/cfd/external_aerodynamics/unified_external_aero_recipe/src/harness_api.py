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

"""The recipe's interface to a scoring harness, such as the AeroZoo benchmark's.

A harness imports this module from the submission's own recipe tree. The
module offers two things:

- :class:`RecipePredictor`, which predicts one case at a time from a
  checkpoint directory this recipe wrote. A submission names it as
  ``checkpoint.predictor: "harness_api:RecipePredictor"``.
- The recipe functions a harness that reads the dataset files itself calls,
  under fixed names (``__all__`` lists them).

``API_VERSION`` changes when a name here changes what it does, so a harness
can refuse a version it does not know.

RecipePredictor
---------------

The harness builds it as ``RecipePredictor(checkpoint_dir, checkpoint, device,
options)``:

- ``checkpoint_dir``: the run's ``checkpoints/`` directory, as
  :class:`physicsnemo.utils.CheckpointManager` lays it out.
- ``checkpoint``: ``"best"`` (the epoch ``best/best.json`` ranks first, from
  ``top_model/``) or ``"final"`` (the newest epoch in ``checkpoint_dir``).
- ``device``: for example ``"cuda:0"`` or ``"cpu"``.
- ``options``: ``model_config`` and ``dataset_config`` (file names under
  ``conf/model/`` and ``datasets/``), ``precision``, ``fields`` (each
  canonical field name the harness scores mapped to the dataset's target
  name), ``inference`` and ``overrides`` (more Hydra overrides, a list).

``options["inference"]`` sets how a case's interior points pass through the
model:

- ``{"queries": "independent", "chunk_points": n}``: chunks of ``n`` points,
  each predicted on its own.
- ``{"queries": "support", "chunk_points": n, "support_points": m,
  "support_seed": s}``: each chunk with the same ``m`` random points (seeded
  by ``s``) beside it.
- ``{"queries": "streamed", "chunk_points": n}``: one forward pass over every
  point, computed ``n`` points at a time
  (:mod:`physicsnemo.models.geotransolver.streaming`, GeoTransolver and
  Transolver).

``predict(case)`` takes a case with ``case_id``, ``domain`` (a
``DomainMesh``: the interior points, the boundary meshes with their prescribed
values, and the physics leaves of ``global_data``) and ``seed``. It returns
each field of ``options["fields"]`` under its canonical name, in SI units, as
an ``(n_points, k)`` float32 array in the order of ``case.domain.interior``.
It runs three hooks:

- ``preprocess(case) -> sample``: the dataset config's transforms applied to
  ``case.domain``, with every random draw seeded by ``case.seed``.
- ``forward(sample) -> outputs``: the model's pointwise outputs, in the
  dataset's normalized units.
- ``postprocess(outputs, sample, case) -> fields``: the outputs returned to SI
  units and renamed to the canonical names.

Custom Models and Processing
----------------------------

A model or pipeline that is not in this recipe needs no change here when its
configs name its classes. Put the model class's import path in the model
config's ``model._target_`` and any new transform in the dataset config's
``pipeline.transforms`` (a ``_target_`` per step), in your own tree. The
recipe trains it, and :class:`RecipePredictor` scores it the same way.

For other pre- or post-processing, subclass :class:`RecipePredictor` in your
tree, override the hooks you need, and name your class in the manifest, for
example ``checkpoint.predictor: "my_predictor:MyPredictor"``. Code that does
not use this recipe at all implements the harness's factory and ``predict``
directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from collate import build_collate_fn
from datasets import (
    build_dataloaders,
    build_transforms,
    find_normalizer,
    load_dataset_config,
    model_out_dim,
    resolve_dataset_config,
)
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from infer import _norm_stats_match as norm_stats_match
from infer import _to_pointwise as to_pointwise
from infer import build_redim_field_types, redimensionalize
from nondim import NonDimensionalizeByMetadata
from omegaconf import OmegaConf
from output_normalize import (
    check_model_output_stats,
    model_dataset_constants,
    model_output_stats,
    normalize_output_to_tensordict,
    require_output_type,
)
from utils import get_autocast_context, recursive_to_device, resolve_dict

import physicsnemo
from physicsnemo.datapipes import MeshDataset
from physicsnemo.mesh import DomainMesh, Mesh

API_VERSION = 1

__all__ = [
    "API_VERSION",
    "NonDimensionalizeByMetadata",
    "RecipePredictor",
    "build_dataloaders",
    "build_redim_field_types",
    "get_autocast_context",
    "load_dataset_config",
    "model_dataset_constants",
    "model_output_stats",
    "norm_stats_match",
    "normalize_output_to_tensordict",
    "recursive_to_device",
    "redimensionalize",
    "require_output_type",
    "sample_paths",
    "set_generator",
    "to_pointwise",
]

_RECIPE_ROOT = Path(__file__).resolve().parent.parent
QUERIES = ("independent", "support", "streamed")
#: Relative RMS residual above which the processed interior points are not the given ones in order.
ALIGN_TOL = 1e-4


def set_generator(dataset: Any, generator: torch.Generator) -> None:
    """Seed every random step of ``dataset`` (its reader and transforms) from ``generator``."""
    dataset.set_generator(generator)


def sample_paths(dataset: Any) -> list[Path]:
    """The sample files of a dataset ``build_dataloaders`` built, in index order.

    A case's id is the name of its file's directory, ``Path(p).parent.name``.
    """
    paths = getattr(dataset.reader, "_paths", None)
    if paths is None:
        raise RuntimeError(
            f"the dataset's reader {type(dataset.reader).__name__} lists no sample files"
        )
    return [Path(p) for p in paths]


def checkpoint_file(
    checkpoint_dir: Path, model_name: str, checkpoint: str, suffix: str = ".mdlus"
) -> Path:
    """The weights file ``checkpoint`` selects in a CheckpointManager directory.

    ``"best"`` is ``top_model/<Model>.0.<epoch><suffix>`` for the epoch
    ``best/best.json`` ranks first. ``"final"`` is the newest
    ``<Model>.0.<epoch><suffix>`` in ``checkpoint_dir``. A missing file
    raises.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint == "best":
        index = checkpoint_dir / "best" / "best.json"
        if not index.is_file():
            raise FileNotFoundError(
                f"{index} does not exist: the run kept no best checkpoints by validation loss"
            )
        ranked = json.loads(index.read_text())
        if not ranked:
            raise ValueError(f"{index} ranks no epoch")
        epoch = int(ranked[0][1])
        path = checkpoint_dir / "top_model" / f"{model_name}.0.{epoch}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} does not exist, though {index} ranks epoch {epoch} first"
            )
        return path
    if checkpoint == "final":
        found = sorted(
            checkpoint_dir.glob(f"{model_name}.0.*{suffix}"),
            key=lambda p: int(p.name.split(".")[-2]),
        )
        if not found:
            raise FileNotFoundError(
                f"no {model_name}.0.<epoch>{suffix} in {checkpoint_dir}"
            )
        return found[-1]
    raise ValueError(f'checkpoint must be "best" or "final", got {checkpoint!r}')


def _check_order(processed: torch.Tensor, given: torch.Tensor) -> None:
    """Raise unless the processed interior points are the given ones, in order, up to an affine map.

    The dataset's transforms can scale or move coordinates, which the map
    absorbs. A subsample or a reordering cannot be absorbed and raises.
    """
    if processed.shape != given.shape:
        raise ValueError(
            f"the dataset's transforms turned {given.shape[0]} interior points into "
            f"{processed.shape[0]}; every given point needs a prediction"
        )
    x = processed.double().cpu()
    y = given.double().cpu()
    x, y = x - x.mean(0), y - y.mean(0)
    coef = torch.linalg.lstsq(x.T @ x, x.T @ y).solution
    spread = float((y**2).sum())
    resid = float(((x @ coef - y) ** 2).sum()) / spread if spread > 0 else 0.0
    if not resid**0.5 < ALIGN_TOL:
        raise ValueError(
            "the dataset's transforms reorder the interior points (relative residual "
            f"{resid**0.5:.3g} of an affine fit); predictions must follow the given order"
        )


class _GivenDomain:
    """A reader of one sample: the domain :meth:`RecipePredictor.preprocess` hands it."""

    def __init__(self) -> None:
        self.domain: DomainMesh | None = None

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[DomainMesh, dict[str, Any]]:
        if self.domain is None or index != 0:
            raise IndexError("the predictor's reader holds one domain at a time")
        return self.domain, {}


class RecipePredictor:
    """Predict a case's fields in SI units from a checkpoint this recipe trained.

    The module docstring describes the arguments, the case and the hooks.

    Attributes
    ----------
    cfg : DictConfig
        The composed inference config.
    model : torch.nn.Module
        The model, with the selected checkpoint's weights, in eval mode.
    checkpoint_file : Path
        The weights file loaded.
    num_params : int
        The model's parameter count.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        checkpoint: str,
        device: str,
        options: dict[str, Any],
    ) -> None:
        missing = [
            k
            for k in (
                "model_config",
                "dataset_config",
                "precision",
                "fields",
                "inference",
            )
            if k not in options
        ]
        if missing:
            raise KeyError(f"RecipePredictor options lack {missing}")
        self.device = torch.device(device)
        checkpoint_dir = Path(checkpoint_dir)

        GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=str(_RECIPE_ROOT / "conf"), version_base=None
        ):
            cfg = compose(
                config_name="infer",
                overrides=[
                    f"model={options['model_config']}",
                    f"dataset={options['dataset_config']}",
                    f"precision={options['precision']}",
                    "compile=false",
                    "sampling_resolution=1000000000",
                    *options.get("overrides", []),
                ],
            )
        self.cfg = cfg

        ds_yaml = resolve_dataset_config(cfg, cfg.dataset)
        self.targets = OmegaConf.to_container(ds_yaml.targets, resolve=True)
        OmegaConf.update(cfg, "out_dim", model_out_dim(self.targets), force_add=True)
        self._reader = _GivenDomain()
        self._dataset = MeshDataset(
            self._reader,
            transforms=build_transforms(ds_yaml) or None,
            device=self.device,
        )
        self.normalizer = find_normalizer([self._dataset])
        self.collate = build_collate_fn(
            cfg.input_type, resolve_dict(cfg, "forward_kwargs"), {}
        )
        self.output_type = require_output_type(cfg)

        ### As train.py builds it: a model that leaves its output scales or dataset constants to the dataset
        ### (ISLA) gets them injected.
        ds_cfg = load_dataset_config(
            _RECIPE_ROOT / "datasets" / f"{cfg.dataset}.yaml", cfg.dataset_paths
        )
        model = hydra.utils.instantiate(
            cfg.model,
            _convert_="partial",
            **model_output_stats(cfg.model, self.normalizer),
            **model_dataset_constants(cfg.model, ds_cfg),
        ).to(self.device)
        check_model_output_stats(model, self.normalizer)
        is_module = isinstance(model, physicsnemo.Module)
        self.checkpoint_file = checkpoint_file(
            checkpoint_dir,
            type(model).__name__,
            checkpoint,
            ".mdlus" if is_module else ".pt",
        )
        if is_module:
            model.load(str(self.checkpoint_file), map_location=self.device, strict=True)
        else:
            model.load_state_dict(
                torch.load(
                    self.checkpoint_file, map_location=self.device, weights_only=True
                )
            )
        self.model = model.eval()
        self.num_params = int(sum(p.numel() for p in model.parameters()))

        if self.normalizer is not None:
            stats_file = checkpoint_dir / "norm_stats.pt"
            if not stats_file.is_file():
                raise FileNotFoundError(
                    f"{stats_file} does not exist: the dataset normalizes its fields, so the run "
                    "must have saved its normalization stats"
                )
            if not norm_stats_match(
                self.normalizer.stats, torch.load(stats_file, weights_only=True)
            ):
                raise ValueError(
                    f"the run's normalization stats ({stats_file}) differ from the ones dataset "
                    f"config {cfg.dataset!r} gives; predict with the dataset config the model "
                    "was trained with"
                )

        self.field_types = build_redim_field_types(ds_cfg)
        self.nondim = (
            NonDimensionalizeByMetadata(fields=self.field_types)
            if self.field_types
            else None
        )
        self.fields = dict(options["fields"])
        absent = sorted(v for v in self.fields.values() if v not in self.targets)
        if absent:
            raise ValueError(
                f"fields name {absent}, which are not targets of dataset config "
                f"{cfg.dataset!r} ({sorted(self.targets)})"
            )
        self.inference = self._check_inference(dict(options["inference"]))
        if self.inference["queries"] == "streamed":
            from physicsnemo.models.geotransolver.streaming import streamed_forward

            self._stream = streamed_forward

    @staticmethod
    def _check_inference(inference: dict[str, Any]) -> dict[str, Any]:
        if inference.get("queries") not in QUERIES:
            raise ValueError(
                f"inference.queries must be one of {QUERIES}, got {inference.get('queries')!r}"
            )
        if not int(inference.get("chunk_points", 0)) >= 1:
            raise ValueError(
                f"inference.chunk_points must be a positive integer: {inference}"
            )
        support = "support_points" in inference and "support_seed" in inference
        if (inference["queries"] == "support") != support:
            raise ValueError(
                "inference.support_points and support_seed go with queries: support, "
                f"and only with it: {inference}"
            )
        return inference

    def predict(self, case: Any) -> dict[str, np.ndarray]:
        """The case's fields in SI units: ``postprocess(forward(preprocess(case)), ...)``."""
        sample = self.preprocess(case)
        return self.postprocess(self.forward(sample), sample, case)

    def preprocess(self, case: Any) -> tuple[DomainMesh, dict[str, Any]]:
        """The dataset config's transforms on ``case.domain``, every random draw seeded by ``case.seed``.

        The draws are the ones the recipe's dataset makes on a sample read
        from its file, given the same generator.
        """
        self._reader.domain = case.domain
        try:
            self._dataset.set_generator(torch.Generator().manual_seed(int(case.seed)))
            sample = self._dataset[0]
        finally:
            self._reader.domain = None
        interior = sample[0].interior
        if interior.n_cells:
            raise ValueError(
                "the predictor needs a point-cloud interior; the dataset's transforms kept "
                f"{interior.n_cells} interior cells"
            )
        _check_order(interior.points, case.domain.interior.points)
        return sample

    def forward(self, sample: tuple[DomainMesh, dict[str, Any]]) -> Any:
        """The model's pointwise outputs for every interior point, in the dataset's normalized units."""
        domain, meta = sample
        inference, step = self.inference, int(self.inference["chunk_points"])
        with torch.no_grad(), get_autocast_context(self.cfg.precision):
            if inference["queries"] == "streamed":
                batch = recursive_to_device(self.collate([(domain, meta)]), self.device)
                return self._pointwise(
                    self._stream(self.model, step, **batch["forward_kwargs"])
                )
            n = domain.interior.n_points
            support = None
            if inference["queries"] == "support":
                g = torch.Generator().manual_seed(int(inference["support_seed"]))
                support = (
                    torch.randperm(n, generator=g)[
                        : min(int(inference["support_points"]), n)
                    ]
                    .sort()
                    .values
                )
            parts = []
            for start in range(0, n, step):
                chunk = torch.arange(start, min(start + step, n))
                rows = chunk if support is None else torch.cat([support, chunk])
                rows = rows.to(domain.interior.points.device)
                interior = Mesh(
                    points=domain.interior.points[rows],
                    point_data=domain.interior.point_data[rows],
                )
                part = DomainMesh(
                    interior=interior,
                    boundaries=domain.boundaries,
                    global_data=domain.global_data,
                )
                batch = recursive_to_device(self.collate([(part, meta)]), self.device)
                out = self._pointwise(self.model(**batch["forward_kwargs"]))
                parts.append(out[len(rows) - len(chunk) :])
        return torch.cat(parts)

    def _pointwise(self, out: Any) -> Any:
        td = normalize_output_to_tensordict(out, self.targets, self.output_type)
        return to_pointwise(td.float(), self.output_type)

    def postprocess(
        self, outputs: Any, sample: tuple[DomainMesh, dict[str, Any]], case: Any
    ) -> dict[str, np.ndarray]:
        """The outputs in SI units under their canonical names, ``(n_points, k)`` float32 each."""
        phys = redimensionalize(
            outputs,
            normalizer=self.normalizer,
            nondim=self.nondim,
            field_types=self.field_types,
            global_data=sample[0].global_data,
        )
        n = case.domain.interior.n_points
        return {
            name: phys[target].reshape(n, -1).float().cpu().numpy()
            for name, target in self.fields.items()
        }
