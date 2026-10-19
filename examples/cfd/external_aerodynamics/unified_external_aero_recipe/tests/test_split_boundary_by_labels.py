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

"""``SplitBoundaryByLabels`` and the model configs' ``dataset_append_transforms``
(ISLA's boundary conditions by name, 2026-10-19).

A dataset that labels its boundary conditions inside one mesh (the lug, Tensile2d,
DeepJEB++) is split into one boundary per condition for ISLA only: the split
boundaries hold the same cells, the labels are not carried, the other models' view of
the dataset is unchanged, and on a surface task the scored points are permuted into
the order of ISLA's predictions.
"""

import sys
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RECIPE_ROOT / "src"))

from collate import build_collate_fn  # noqa: E402
from datasets import _apply_dataset_transform_appends  # noqa: E402
from domain_transforms import SplitBoundaryByLabels  # noqa: E402
from output_normalize import normalize_output_to_tensordict  # noqa: E402

from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402
from physicsnemo.mesh.calculus.measure import cell_measures  # noqa: E402

LUG_LABELS = {"loaded": "bc_loaded", "hole": "bc_hole", "fixed": "bc_fixed"}


def _surface(n: int = 40, seed: int = 0) -> Mesh:
    """A closed triangulated surface (an octahedron subdivided) with the lug's labels:
    ``bc_hole`` on the cells with z > 0.3, ``bc_loaded`` on half of those (x > 0),
    ``bc_fixed`` on the cells with z < -0.5, and ``bc_pressure`` on the loaded cells,
    zero elsewhere."""
    v = torch.tensor(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=torch.float64,
    )
    f = torch.tensor(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ]
    )
    mesh = Mesh(points=v, cells=f).subdivide(levels=2, filter="linear")
    c = mesh.cell_centroids
    hole = (c[:, 2] > 0.3).double()
    loaded = hole * (c[:, 0] > 0).double()
    g = torch.Generator().manual_seed(seed)
    pressure = loaded * (1 + torch.rand(mesh.n_cells, generator=g, dtype=torch.float64))
    return mesh.with_data(
        cell_data={
            "bc_hole": hole,
            "bc_loaded": loaded,
            "bc_fixed": (c[:, 2] < -0.5).double(),
            "bc_pressure": pressure,
        }
    )


def _domain(surface: Mesh, interior: Mesh | None = None) -> DomainMesh:
    if interior is None:
        interior = Mesh(points=0.2 * torch.randn(30, 3, dtype=torch.float64))
    return DomainMesh(
        interior=interior,
        boundaries={"surface": surface},
        global_data={"length_std": torch.tensor(0.1, dtype=torch.float64)},
    )


def test_the_split_holds_the_same_cells_without_the_labels():
    surface = _surface()
    out = SplitBoundaryByLabels("surface", LUG_LABELS, rest="free").apply_to_domain(
        _domain(surface)
    )
    assert set(out.boundary_names) == {"surface", "loaded", "hole", "fixed", "free"}
    assert out.boundaries["surface"] is surface
    cd = surface.cell_data
    expected = {
        "loaded": cd["bc_loaded"] > 0,
        "hole": (cd["bc_hole"] > 0) & ~(cd["bc_loaded"] > 0),
        "fixed": cd["bc_fixed"] > 0,
    }
    expected["free"] = ~(expected["loaded"] | expected["hole"] | expected["fixed"])
    total = 0
    for name, mask in expected.items():
        part = out.boundaries[name]
        assert part.n_cells == int(mask.sum()) > 0, name
        torch.testing.assert_close(part.cell_centroids, surface.cell_centroids[mask])
        torch.testing.assert_close(cell_measures(part), cell_measures(surface)[mask])
        torch.testing.assert_close(
            part.cell_data["bc_pressure"], cd["bc_pressure"][mask]
        )
        assert not set(LUG_LABELS.values()) & set(part.cell_data.keys()), name
        total += part.n_cells
    assert total == surface.n_cells


def test_a_label_no_cell_carries_is_a_boundary_without_cells_and_unlabelled_cells_need_rest():
    surface = _surface()
    surface.cell_data["bc_none"] = torch.zeros(surface.n_cells, dtype=torch.float64)
    out = SplitBoundaryByLabels(
        "surface", {**LUG_LABELS, "none": "bc_none"}, rest="free"
    ).apply_to_domain(_domain(surface))
    assert out.boundaries["none"].n_cells == 0
    with pytest.raises(ValueError, match="carry no label"):
        SplitBoundaryByLabels("surface", LUG_LABELS).apply_to_domain(_domain(surface))
    with pytest.raises(KeyError, match="'outer' not found"):
        SplitBoundaryByLabels("outer", LUG_LABELS, rest="free").apply_to_domain(
            _domain(surface)
        )
    with pytest.raises(ValueError, match="distinct"):
        SplitBoundaryByLabels("surface", LUG_LABELS, rest="hole")


def test_reorder_interior_puts_the_scored_points_in_the_order_of_the_predictions():
    """A surface task: the interior is the source boundary's cell centroids with the
    targets; after the split it is the new boundaries' cells in sorted name order, each
    point with its own target."""
    surface = _surface()
    target = torch.arange(surface.n_cells, dtype=torch.float64)
    interior = Mesh(points=surface.cell_centroids, point_data={"von_mises": target})
    split = SplitBoundaryByLabels(
        "surface", LUG_LABELS, rest="free", reorder_interior=True
    )
    out = split.apply_to_domain(_domain(surface, interior))
    names = sorted(["loaded", "hole", "fixed", "free"])
    torch.testing.assert_close(
        out.interior.points,
        torch.cat([out.boundaries[n].cell_centroids for n in names]),
    )
    torch.testing.assert_close(
        surface.cell_centroids[out.interior.point_data["von_mises"].long()],
        out.interior.points,
    )
    with pytest.raises(ValueError, match="cell centroids"):
        split.apply_to_domain(_domain(surface))


def _compose(model: str, dataset: str = "drivaer_ml_surface", *extra):
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"), version_base=None
    ):
        return compose(
            config_name="train",
            overrides=[f"model={model}", f"dataset={dataset}", "+out_dim=1", *extra],
        )


@pytest.mark.parametrize(
    "model", ["isla_lug", "isla_tensile2d", "isla_surface_deepjebpp"]
)
def test_the_split_is_appended_for_isla_only(model):
    """The ISLA config appends its split to the dataset's chain; a config without
    ``dataset_append_transforms`` leaves the chain as it is."""
    ds = OmegaConf.create({"pipeline": {"transforms": [{"_target_": "a.First"}]}})
    isla = _apply_dataset_transform_appends(ds, _compose(model))
    chain = [str(t["_target_"]) for t in isla.pipeline.transforms]
    assert len(chain) == 2 and chain[0] == "a.First"
    assert chain[1].endswith("SplitBoundaryByLabels")
    for baseline in (
        "transolver_lug",
        "transolver_surface_deepjebpp",
        "transolver_tensile2d",
    ):
        assert _apply_dataset_transform_appends(ds, _compose(baseline)) is ds


@pytest.mark.parametrize("deepjebpp", [False, True], ids=["lug", "deepjebpp"])
def test_the_split_configs_compose_and_run(deepjebpp):
    """The lug's and DeepJEB++'s ISLA configs: the appended split builds the named
    boundaries the forward_kwargs name, each condition reads its own fields, and the
    output reaches the targets point for point (DeepJEB++, a surface task, through the
    reordered interior)."""
    model = "isla_surface_deepjebpp" if deepjebpp else "isla_lug"
    cfg = _compose(
        model,
        "drivaer_ml_surface",
        "model.hidden=16",
        "model.n_layers=1",
        "model.n_slices=4",
    )
    surface = _surface()
    if deepjebpp:
        g = torch.Generator().manual_seed(1)
        load = cfg.model.boundary_schemas.loaded
        assert list(load) == ["bc_load"]
        surface.cell_data["bc_load"] = surface.cell_data["bc_loaded"][
            :, None
        ] * torch.randn(surface.n_cells, 3, generator=g, dtype=torch.float64)
        target = {"von_mises": "scalar"}
        interior = Mesh(
            points=surface.cell_centroids,
            point_data={
                "von_mises": torch.arange(surface.n_cells, dtype=torch.float64)
            },
        )
        gd = {
            "load_dir": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            "is_force": torch.tensor(1.0, dtype=torch.float64),
            "is_torque": torch.tensor(0.0, dtype=torch.float64),
        }
    else:
        target = {"von_mises": "scalar"}
        interior = Mesh(
            points=0.2 * torch.randn(30, 3, dtype=torch.float64),
            point_data={
                "von_mises": torch.randn(30, dtype=torch.float64),
                "sdf": torch.rand(30, dtype=torch.float64),
                "sdf_normals": torch.randn(30, 3, dtype=torch.float64),
            },
        )
        gd = {
            k: torch.tensor(0.1, dtype=torch.float64) for k in cfg.model.global_schema
        }
    domain = DomainMesh(
        interior=interior, boundaries={"surface": surface}, global_data=gd
    )
    for t in cfg.dataset_append_transforms:
        domain = hydra.utils.instantiate(t).apply_to_domain(domain)
    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    assert set(spec["boundary_meshes"]) == set(cfg.model.boundary_schemas)
    batch = build_collate_fn("mesh", spec, target)([(domain, {})])
    model_ = hydra.utils.instantiate(cfg.model).double()
    with torch.no_grad():
        out = model_(**batch["forward_kwargs"])
    torch.testing.assert_close(out.points, domain.interior.points)
    pred = normalize_output_to_tensordict(out, target, "mesh")
    assert pred["von_mises"].shape == batch["targets"]["von_mises"].shape
    assert torch.isfinite(pred["von_mises"]).all()
