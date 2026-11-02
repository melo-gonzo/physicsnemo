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

"""src/harness_api.py, the recipe's interface to a scoring harness: every name of API_VERSION 1 is the recipe's own,
and RecipePredictor, on a run this recipe saved, predicts the synthetic AeroZoo case from its label-free domain
exactly as the recipe's reader path does from the case file, in SI units, for the best or the final checkpoint."""

from types import SimpleNamespace

import datasets
import harness_api
import hydra
import infer
import nondim
import numpy as np
import output_normalize
import pytest
import torch
import utils
from collate import build_collate_fn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from test_aerozoo_configs import _RECIPE_ROOT, make_aerozoo_case

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.utils import CheckpointManager

#: Each recipe name of the harness contract and the recipe object it must be.
CONTRACT = {
    "build_dataloaders": datasets.build_dataloaders,
    "load_dataset_config": datasets.load_dataset_config,
    "recursive_to_device": utils.recursive_to_device,
    "get_autocast_context": utils.get_autocast_context,
    "require_output_type": output_normalize.require_output_type,
    "normalize_output_to_tensordict": output_normalize.normalize_output_to_tensordict,
    "model_output_stats": output_normalize.model_output_stats,
    "model_dataset_constants": output_normalize.model_dataset_constants,
    "build_redim_field_types": infer.build_redim_field_types,
    "redimensionalize": infer.redimensionalize,
    "to_pointwise": infer._to_pointwise,
    "norm_stats_match": infer._norm_stats_match,
    "NonDimensionalizeByMetadata": nondim.NonDimensionalizeByMetadata,
}
#: Canonical field -> the AeroZoo datasets' target name.
FIELDS = {
    "velocity": "velocity",
    "pressure": "pressure",
    "turbulent_viscosity": "nut",
    "tke": "tke",
    "omega": "omega",
    "intermittency": "intermittency",
    "retheta": "retheta",
}
WIDTHS = {"velocity": 3}
#: A per-case-scales tensor model and ISLA (a mesh model given its output scales and constants), shrunk for a CPU
#: test without changing their interface.
RUNS = {
    "transolver": (
        "transolver_volume_aerozoo_casescale",
        "aerozoo_uvolume_casescale",
        [
            "model.n_layers=2",
            "model.n_hidden=32",
            "model.slice_num=16",
            "model.n_head=2",
        ],
    ),
    "isla": (
        "isla_volume_aerozoo",
        "aerozoo_uvolume",
        [
            "model.hidden=16",
            "model.n_layers=1",
            "model.n_slices=4",
            "model.compile_blocks=false",
            "model.n_boundary_tokens=64",
        ],
    ),
}
SEED = 1234


def test_every_contract_name_is_the_recipes_own():
    """API_VERSION 1 lists exactly the names a harness uses, each the recipe's own object, not a copy."""
    assert harness_api.API_VERSION == 1
    assert set(harness_api.__all__) == {
        *CONTRACT,
        "API_VERSION",
        "RecipePredictor",
        "set_generator",
        "sample_paths",
    }
    for name, obj in CONTRACT.items():
        assert getattr(harness_api, name) is obj, name
    assert all(callable(getattr(harness_api, n)) for n in harness_api.__all__[1:])


def _case_file(root) -> DomainMesh:
    stored = make_aerozoo_case(True, seed=1)
    path = root / "shard-000" / "case-0001" / "domain.pdmsh"
    path.parent.mkdir(parents=True)
    stored.save(path)
    return stored


def _label_free(d: DomainMesh) -> DomainMesh:
    """What the harness hands a predictor: the interior points, each boundary with its prescribed values only, and
    the physics leaves of global_data without the closure id."""
    physics = d.global_data["physics"].exclude("closure_model_id")
    return DomainMesh(
        interior=Mesh(points=d.interior.points.clone()),
        boundaries={
            n: Mesh(
                points=d.boundaries[n].points,
                cells=d.boundaries[n].cells,
                cell_data=d.boundaries[n].cell_data.select("prescribed", strict=False),
            )
            for n in d.boundary_names
        },
        global_data=TensorDict({"physics": physics}),
    )


def test_sample_paths_and_set_generator_reach_the_reader(tmp_path):
    """A dataset built from a config with dataset_paths over the file's roots lists the case file, and seeding it
    twice the same way gives the same sample."""
    _case_file(tmp_path)
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"), version_base=None
    ):
        cfg = compose(
            config_name="infer",
            overrides=[
                "model=transolver_volume_aerozoo",
                "dataset=aerozoo_uvolume",
                f"++dataset_paths.aerozoo={tmp_path}",
                "sampling_resolution=64",
            ],
        )
    ds = datasets.build_dataset(
        datasets.resolve_dataset_config(cfg, cfg.dataset), device=None
    )
    assert harness_api.sample_paths(ds) == [
        tmp_path / "shard-000" / "case-0001" / "domain.pdmsh"
    ]
    draws = []
    for _ in range(2):
        harness_api.set_generator(ds, torch.Generator().manual_seed(SEED))
        draws.append(ds[0][0].interior.points)
    assert draws[0].shape == (64, 3) and torch.equal(*draws)


def _trained_run(tmp_path, run: str):
    """A run's checkpoints/ as the recipe writes it: random weights at epochs 1 (the best, validation loss 0.2) and 2
    (the final, 0.4), and norm_stats.pt. Returns the directory and the predictor options."""
    model_config, dataset_config, knobs = RUNS[run]
    root = tmp_path / "data"
    _case_file(root)
    overrides = [*knobs, f"++dataset_paths.aerozoo={root}"]
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"), version_base=None
    ):
        cfg = compose(
            config_name="infer",
            overrides=[
                f"model={model_config}",
                f"dataset={dataset_config}",
                *overrides,
            ],
        )
    ds_yaml = datasets.resolve_dataset_config(cfg, cfg.dataset)
    targets = OmegaConf.to_container(ds_yaml.targets, resolve=True)
    OmegaConf.update(cfg, "out_dim", datasets.model_out_dim(targets), force_add=True)
    normalizer = datasets.find_normalizer(
        [SimpleNamespace(transforms=datasets.build_transforms(ds_yaml))]
    )
    ds_cfg = datasets.load_dataset_config(
        _RECIPE_ROOT / "datasets" / f"{dataset_config}.yaml", cfg.dataset_paths
    )
    ckpt = tmp_path / "run" / "checkpoints"
    manager = CheckpointManager(ckpt, keep_best=2)
    for epoch, loss in ((1, 0.2), (2, 0.4)):
        torch.manual_seed(epoch)
        model = hydra.utils.instantiate(
            cfg.model,
            _convert_="partial",
            **output_normalize.model_output_stats(cfg.model, normalizer),
            **output_normalize.model_dataset_constants(cfg.model, ds_cfg),
        )
        manager.save(epoch, metric=loss, models=model)
    torch.save(normalizer.stats, ckpt / "norm_stats.pt")
    options = {
        "model_config": model_config,
        "dataset_config": dataset_config,
        "precision": "float32",
        "fields": FIELDS,
        "inference": {"queries": "independent", "chunk_points": 10**6},
        "overrides": overrides,
    }
    return ckpt, options


def _reader_path(pred: harness_api.RecipePredictor) -> dict[str, np.ndarray]:
    """The recipe's own path on the case file: the dataset's reader and transforms, collate, the predictor's model,
    and the redimensionalization infer.py applies."""
    cfg = pred.cfg
    ds = datasets.build_dataset(
        datasets.resolve_dataset_config(cfg, cfg.dataset), device=None
    )
    ds.set_generator(torch.Generator().manual_seed(SEED))
    domain, meta = ds[0]
    collate = build_collate_fn(
        cfg.input_type, utils.resolve_dict(cfg, "forward_kwargs"), pred.targets
    )
    with torch.no_grad():
        out = pred.model(**collate([(domain, meta)])["forward_kwargs"])
    td = infer._to_pointwise(
        output_normalize.normalize_output_to_tensordict(
            out, pred.targets, pred.output_type
        ).float(),
        pred.output_type,
    )
    ds_cfg = datasets.load_dataset_config(
        _RECIPE_ROOT / "datasets" / f"{cfg.dataset}.yaml", cfg.dataset_paths
    )
    types = infer.build_redim_field_types(ds_cfg)
    phys = infer.redimensionalize(
        td,
        normalizer=datasets.find_normalizer([ds]),
        nondim=nondim.NonDimensionalizeByMetadata(fields=types) if types else None,
        field_types=types,
        global_data=domain.global_data,
    )
    n = domain.interior.n_points
    return {k: phys[v].reshape(n, -1).numpy() for k, v in FIELDS.items()}


@pytest.mark.parametrize("run", sorted(RUNS))
def test_the_predictor_predicts_the_label_free_case_as_the_reader_path_does(
    tmp_path, run
):
    """From the label-free domain, every field comes back in SI units as an (n_points, k) float32 array in the
    domain's point order, equal to the recipe's reader path on the case file and the same on a second call. ISLA
    predicts through query_chunks (its boundary side once per case), equal to its forward pass up to summation
    order."""
    ckpt, options = _trained_run(tmp_path, run)
    pred = harness_api.RecipePredictor(ckpt, "best", "cpu", options)
    assert pred.checkpoint_file.parent.name == "top_model"
    stored = make_aerozoo_case(True, seed=1)
    case = SimpleNamespace(case_id="case-0001", domain=_label_free(stored), seed=SEED)
    out = pred.predict(case)
    n = stored.interior.n_points
    assert set(out) == set(FIELDS)
    for name, value in out.items():
        assert value.dtype == np.float32 and value.shape == (n, WIDTHS.get(name, 1))
        assert np.isfinite(value).all(), name
    reference = _reader_path(pred)
    assert pred._query_chunks == (run == "isla")
    tolerance = (
        {"rtol": 1e-5, "atol": 1e-5} if run == "isla" else {"rtol": 0, "atol": 0}
    )
    for name in FIELDS:
        np.testing.assert_allclose(
            out[name], reference[name], **tolerance, err_msg=name
        )
    again = pred.predict(case)
    assert all(np.array_equal(out[k], again[k]) for k in FIELDS)
    ### The case's labels are not needed, and the full case gives the same prediction.
    full = pred.predict(SimpleNamespace(case_id="case-0001", domain=stored, seed=SEED))
    assert all(np.array_equal(out[k], full[k]) for k in FIELDS)


def test_best_and_final_load_their_epochs(tmp_path):
    """ "best" loads top_model/'s epoch best/best.json ranks first (1) and "final" the newest epoch (2); without the
    ranking "best" raises, and a run without its normalization stats raises."""
    ckpt, options = _trained_run(tmp_path, "transolver")
    best = harness_api.RecipePredictor(ckpt, "best", "cpu", options)
    final = harness_api.RecipePredictor(ckpt, "final", "cpu", options)
    assert best.checkpoint_file == ckpt / "top_model" / "Transolver.0.1.mdlus"
    assert final.checkpoint_file == ckpt / "Transolver.0.2.mdlus"
    case = SimpleNamespace(
        case_id="case-0001", domain=_label_free(make_aerozoo_case(True, seed=1)), seed=0
    )
    assert not np.array_equal(
        best.predict(case)["pressure"], final.predict(case)["pressure"]
    )
    with pytest.raises(ValueError, match="best"):
        harness_api.RecipePredictor(ckpt, "last", "cpu", options)
    (ckpt / "norm_stats.pt").rename(ckpt / "norm_stats.moved")
    with pytest.raises(FileNotFoundError, match="norm_stats.pt"):
        harness_api.RecipePredictor(ckpt, "final", "cpu", options)
    (ckpt / "best" / "best.json").unlink()
    with pytest.raises(FileNotFoundError, match="best.json"):
        harness_api.RecipePredictor(ckpt, "best", "cpu", options)


def test_the_inference_modes_and_their_checks(tmp_path):
    """Streamed Transolver equals one pass over every point; independent chunks and chunks beside a fixed support
    predict every point; malformed options raise."""
    ckpt, options = _trained_run(tmp_path, "transolver")
    case = SimpleNamespace(
        case_id="case-0001", domain=_label_free(make_aerozoo_case(True, seed=1)), seed=0
    )

    def predict(**inference):
        return harness_api.RecipePredictor(
            ckpt, "final", "cpu", {**options, "inference": inference}
        ).predict(case)

    one_pass = predict(queries="independent", chunk_points=10**6)
    streamed = predict(queries="streamed", chunk_points=128)
    for name in FIELDS:
        np.testing.assert_allclose(
            streamed[name], one_pass[name], rtol=1e-4, atol=1e-6, err_msg=name
        )
    n = case.domain.interior.n_points
    for inference in (
        {"queries": "independent", "chunk_points": 128},
        {
            "queries": "support",
            "chunk_points": 128,
            "support_points": 64,
            "support_seed": 3,
        },
    ):
        out = predict(**inference)
        assert out["velocity"].shape == (n, 3) and np.isfinite(out["velocity"]).all()
    for bad in (
        {"queries": "all", "chunk_points": 128},
        {"queries": "independent", "chunk_points": 0},
        {"queries": "support", "chunk_points": 128},
        {
            "queries": "independent",
            "chunk_points": 128,
            "support_points": 64,
            "support_seed": 3,
        },
    ):
        with pytest.raises(ValueError, match="inference"):
            predict(**bad)
    with pytest.raises(ValueError, match="not targets"):
        harness_api.RecipePredictor(
            ckpt, "final", "cpu", {**options, "fields": {"pressure": "p"}}
        )
    with pytest.raises(KeyError, match="fields"):
        harness_api.RecipePredictor(
            ckpt, "final", "cpu", {k: v for k, v in options.items() if k != "fields"}
        )


def test_a_subclass_replaces_a_hook(tmp_path):
    """A collaborator's subclass that overrides postprocess keeps the recipe's preprocessing and forward pass."""
    ckpt, options = _trained_run(tmp_path, "transolver")

    class WithSpeed(harness_api.RecipePredictor):
        """The recipe's predictor plus one more field, the speed."""

        def postprocess(self, outputs, sample, case):
            """The recipe's fields and the speed, |velocity|."""
            fields = super().postprocess(outputs, sample, case)
            fields["speed"] = np.linalg.norm(fields["velocity"], axis=1, keepdims=True)
            return fields

    case = SimpleNamespace(
        case_id="case-0001", domain=_label_free(make_aerozoo_case(True, seed=1)), seed=0
    )
    base = harness_api.RecipePredictor(ckpt, "final", "cpu", options).predict(case)
    out = WithSpeed(ckpt, "final", "cpu", options).predict(case)
    assert all(np.array_equal(out[k], base[k]) for k in FIELDS)
    np.testing.assert_allclose(
        out["speed"][:, 0], np.linalg.norm(base["velocity"], axis=1), rtol=1e-6
    )
