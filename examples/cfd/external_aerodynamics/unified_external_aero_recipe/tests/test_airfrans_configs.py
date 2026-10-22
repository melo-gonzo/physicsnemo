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

"""AF-UNIFORM (2026-10-14): the AirfRANS dataset files of the reference lanes (airfrans_volume, airfrans_volume_2d
and their CHAR-PROTOCOLS _area / _posed scoring twins) and their uniform-interior twins airfrans_uvolume*.yaml.

AirfRANS stores the interior points in solver order, so the source files' reader cap (subsample_n_points, a
cyclic contiguous block) keeps one spatial window per draw. The twins read each case whole and draw the interior
with SubsampleMesh first in the chain. The tests pin that the twin is its source but for the draw, and run both
through the real reader and transforms on a synthetic AirfRANS-shaped case whose stored order is spatial. The
baselines' airfrans_uvolume_ud*.yaml (DRIVE-FIX, 2026-10-18) add the standardized freestream speed."""

import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_RECIPE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RECIPE_ROOT / "src"))

from collate import build_collate_fn  # noqa: E402
from datasets import build_dataset, load_dataset_config  # noqa: E402

from physicsnemo.datapipes.protocols import DatasetBase  # noqa: E402
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402

SOURCES = (
    "airfrans_volume",
    "airfrans_volume_2d",
    "airfrans_volume_area",
    "airfrans_volume_posed",
    "airfrans_volume_2d_area",
    "airfrans_volume_2d_posed",
)
TWINS = {src.replace("airfrans_volume", "airfrans_uvolume", 1): src for src in SOURCES}
N_INTERIOR = 400  # stored interior points of the synthetic case
N_AIRFOIL = 24  # airfoil segments
K = 100  # sampling_resolution in these tests (the campaign's 100,000 of 156k to 210k, scaled down)


def make_airfrans_case(seed: int = 0) -> DomainMesh:
    """An AirfRANS-shaped case as kernel/airfrans_convert.py writes it: an interior point cloud stored in a
    spatial (solver-like) order, sorted by x, and a closed airfoil polyline. The stored row index rides in `sdf`,
    which no transform of these pipelines changes, so a drawn point can be traced to its stored row."""
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(N_INTERIOR, 2, generator=g) * torch.tensor(
        [4.0, 2.0]
    ) - torch.tensor([1.5, 1.0])
    xy = xy[xy[:, 0].argsort()]
    points = torch.cat([xy, torch.zeros(N_INTERIOR, 1)], dim=1)
    velocity = torch.cat(
        [
            1.0 + 0.1 * torch.randn(N_INTERIOR, 2, generator=g),
            torch.zeros(N_INTERIOR, 1),
        ],
        dim=1,
    )
    interior = Mesh(
        points=points,
        point_data={
            "velocity": velocity * 30.0,
            "pressure": torch.randn(N_INTERIOR, generator=g) * 100.0,
            "nut": torch.rand(N_INTERIOR, generator=g) * 1e-3,
            "sdf": torch.arange(N_INTERIOR, dtype=torch.float32)[:, None],
            "sdf_normals": torch.nn.functional.normalize(
                torch.randn(N_INTERIOR, 3, generator=g), dim=1
            ),
        },
    )
    t = torch.arange(N_AIRFOIL) * (2 * torch.pi / N_AIRFOIL)
    foil = torch.stack(
        [0.5 + 0.5 * torch.cos(t), 0.06 * torch.sin(t), torch.zeros(N_AIRFOIL)], dim=1
    )
    cells = torch.stack(
        [torch.arange(N_AIRFOIL), (torch.arange(N_AIRFOIL) + 1) % N_AIRFOIL], dim=1
    )
    seg = foil[cells[:, 1]] - foil[cells[:, 0]]
    normals = torch.nn.functional.normalize(
        torch.stack([seg[:, 1], -seg[:, 0], torch.zeros(N_AIRFOIL)], dim=1), dim=1
    )
    airfoil = Mesh(
        points=foil,
        cells=cells,
        cell_data={"normals": normals, "quadrature_measure": seg.norm(dim=1)},
    )
    u = torch.tensor([29.0, 3.0, 0.0])
    return DomainMesh(
        interior=interior,
        boundaries={"airfoil": airfoil},
        global_data={
            "U_inf": u,
            "U_inf_dir": u / u.norm(),
            "U_inf_mag": u.norm(),
            "aoa_deg": torch.tensor(5.9),
            "Re": torch.tensor(2.0e6),
            "naca": torch.tensor([2.0, 4.0, 1.2, 0.0, 0.0]),
            "U_inf_freestream_mean": u,
            "rho_inf": torch.tensor(1.0),
            "p_inf": torch.tensor(0.0),
        },
    )


@pytest.fixture(scope="module")
def case_root(tmp_path_factory) -> Path:
    """A dataset root holding one synthetic AirfRANS case, shared by the module's tests."""
    root = tmp_path_factory.mktemp("airfrans")
    name = "airFoil2D_SST_29.2_5.9_2.0_4.0_12.0"
    (root / name).mkdir()
    make_airfrans_case().save(root / name / f"{name}.pdmsh")
    return root


def _dataset(name: str, root: Path, k: int):
    cfg = OmegaConf.merge(
        load_dataset_config(_RECIPE_ROOT / "datasets" / f"{name}.yaml"),
        {
            "train_datadir": str(root),
            "manifest": str(root / "manifest.json"),
            "sampling_resolution": k,
        },
    )
    ds = build_dataset(cfg, device=None, num_workers=1)
    ds.set_generator(torch.Generator().manual_seed(42))
    return ds


def _rows(mesh: DomainMesh) -> torch.Tensor:
    """The stored row index of every interior point (carried by `sdf`)."""
    return mesh.interior.point_data["sdf"].reshape(-1).round().long()


def _is_cyclic_block(rows: torch.Tensor, n: int) -> bool:
    s = set(rows.tolist())
    return any(s == {(start + i) % n for i in range(len(s))} for start in range(n))


@pytest.mark.parametrize("twin", sorted(TWINS))
def test_uniform_twin_is_its_source_but_the_interior_draw(twin):
    """airfrans_uvolume*.yaml is its airfrans_volume*.yaml with the reader's two caps removed and a
    SubsampleMesh(n_points=${sampling_resolution}) first in the chain, after the license and an AF-UNIFORM header."""
    src = TWINS[twin]
    datasets = _RECIPE_ROOT / "datasets"
    lines = (datasets / f"{twin}.yaml").read_text().splitlines()
    assert lines[16].startswith(
        f"# AF-UNIFORM (2026-10-14): {src}.yaml with a uniform interior draw."
    )
    a = OmegaConf.to_container(OmegaConf.load(datasets / f"{twin}.yaml"), resolve=False)
    b = OmegaConf.to_container(OmegaConf.load(datasets / f"{src}.yaml"), resolve=False)
    assert b["pipeline"]["reader"].pop("subsample_n_points") == "${sampling_resolution}"
    assert b["pipeline"]["reader"].pop("subsample_n_cells") == "${sampling_resolution}"
    first = a["pipeline"]["transforms"].pop(0)
    assert first == {
        "_target_": "${dp:SubsampleMesh}",
        "n_points": "${sampling_resolution}",
    }
    assert a == b


@pytest.mark.parametrize("twin", sorted(TWINS))
def test_uniform_twin_draws_a_uniform_interior_subset(twin, case_root):
    """Through the real reader and chain: the source's draw is one cyclic contiguous block of the stored rows (a
    spatial window here), the twin's a uniform random subset of K rows that changes with the epoch, with the
    airfoil whole; every drawn point carries the source pipeline's values for its stored row."""
    src = TWINS[twin]
    old, new = _dataset(src, case_root, K), _dataset(twin, case_root, K)
    whole = _dataset(
        src, case_root, 10**6
    )  # the source pipeline with no draw: every stored row
    try:
        ref = whole[0][0]
        ref_rows = _rows(ref)
        assert torch.equal(ref_rows.sort().values, torch.arange(N_INTERIOR))
        draws = set()
        for epoch in range(8):
            for ds in (old, new):
                ds.set_epoch(epoch)
            o, d = old[0][0], new[0][0]
            assert _is_cyclic_block(_rows(o), N_INTERIOR)
            rows = _rows(d)
            assert rows.numel() == rows.unique().numel() == K
            assert not _is_cyclic_block(rows, N_INTERIOR)
            assert (rows.sort().values.diff() > 1).sum() > 10  # many runs, not a window
            draws.add(frozenset(rows.tolist()))
            assert d.boundaries["airfoil"].n_cells == N_AIRFOIL
            ### The values of each drawn row are the whole-case pipeline's (the posed files' translation is one
            ### RMS radius of the points present, so their coordinates are compared only up to that shift).
            pos = torch.empty(N_INTERIOR, dtype=torch.long)
            pos[ref_rows] = torch.arange(N_INTERIOR)
            sel = pos[rows]
            for name in ("velocity", "pressure", "nut", "sdf_normals"):
                torch.testing.assert_close(
                    d.interior.point_data[name], ref.interior.point_data[name][sel]
                )
            shift = d.interior.points - ref.interior.points[sel]
            torch.testing.assert_close(shift, shift[:1].expand_as(shift))
            if "posed" not in twin:
                torch.testing.assert_close(shift, torch.zeros_like(shift))
        assert len(draws) == 8
    finally:
        for ds in (old, new, whole):
            DatasetBase.close(ds)


UD_TWINS = (
    "airfrans_uvolume_ud",
    "airfrans_uvolume_ud_area",
    "airfrans_uvolume_ud_posed",
)
#: DRIVE-FIX R1 (2026-10-18): the standardized freestream speed the AirfRANS baselines read, from the
#: full_train (800 cases) mean and population std of global_data.U_inf_mag.
SPEED_STD = {
    "_target_": "${dp:SplitGlobalVector}",
    "field": "U_inf_mag",
    "names": ["U_inf_mag_std"],
    "mean": 62.116,
    "std": 17.914,
}


@pytest.mark.parametrize("ud", UD_TWINS)
def test_ud_twin_is_its_uniform_twin_plus_the_standardized_speed(ud):
    """airfrans_uvolume_ud*.yaml (the AirfRANS baselines' datasets) is its airfrans_uvolume*.yaml with one
    last transform, the SplitGlobalVector that writes U_inf_mag_std with fixed constants."""
    datasets = _RECIPE_ROOT / "datasets"
    a = OmegaConf.to_container(OmegaConf.load(datasets / f"{ud}.yaml"), resolve=False)
    b = OmegaConf.to_container(
        OmegaConf.load(datasets / f"{ud.replace('_ud', '', 1)}.yaml"), resolve=False
    )
    assert a["pipeline"]["transforms"].pop() == SPEED_STD
    assert a == b


@pytest.mark.parametrize(
    "model", ["geotransolver_volume_airfrans", "transolver_volume_airfrans"]
)
@pytest.mark.parametrize("ud", UD_TWINS)
def test_ud_twin_feeds_the_baselines_the_direction_and_the_standardized_speed(
    ud, model, case_root
):
    """Through the real reader and chain and the recipe collate: the AirfRANS baseline configs read the unit
    freestream direction and (|U_inf| - 62.116) / 17.914, never the raw U_inf vector."""
    ds = _dataset(ud, case_root, K)
    try:
        sample = ds[0][0]
    finally:
        DatasetBase.close(ds)
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"), version_base=None
    ):
        cfg = compose(
            config_name="train", overrides=[f"model={model}", f"dataset={ud}"]
        )
    spec = OmegaConf.to_container(cfg.forward_kwargs, resolve=True)
    key = "global_embedding" if "global_embedding" in spec else "fx"
    targets = {"pressure": "scalar", "nut": "scalar", "velocity": "vector"}
    collate = build_collate_fn(
        input_type="tensors", forward_kwargs_spec=spec, target_config=targets
    )
    x = collate([(sample, {})])["forward_kwargs"][key]
    u = sample.global_data["U_inf"]
    expected = torch.cat([u / u.norm(), ((u.norm() - 62.116) / 17.914).reshape(1)])
    rows = x.reshape(-1, x.shape[-1])
    torch.testing.assert_close(rows, expected.expand_as(rows))


#: AF-EXTRAP (2026-10-21): the constants of the extrapolation splits' dataset files, each fitted on its split's train
#: side only (examples/cfd/isla/results/aerozoo_dataset/isla_constants_fit.py airfrans_fit, LHR job 309151).
SPLIT_FITS = {
    "reynolds": {
        "nut": {"mean": 7.967e-4, "std": 2.383e-3},
        "speed": {"mean": 62.587, "std": 8.762},
        "isla": {
            "reynolds": {"mean": 6.223, "std": 0.06224},
            "output_scale": {"nut": 7.460e-05},
        },
    },
    "aoa": {
        "nut": {"mean": 7.122e-4, "std": 2.131e-3},
        "speed": {"mean": 62.363, "std": 17.776},
        "isla": {
            "reynolds": {"mean": 6.207, "std": 0.1327},
            "output_scale": {"nut": 7.540e-05},
        },
    },
}


@pytest.mark.parametrize("task", sorted(SPLIT_FITS))
@pytest.mark.parametrize(
    "src",
    [
        "airfrans_uvolume_2d",
        "airfrans_uvolume_2d_posed",
        "airfrans_uvolume_ud",
        "airfrans_uvolume_ud_posed",
    ],
)
def test_extrapolation_split_file_is_its_source_with_train_side_constants(src, task):
    """airfrans_uvolume_{2d,ud}_<task>{,_posed}.yaml is its full-split file with the nut stats, and ISLA's
    isla_constants or the baselines' standardized-speed constants, replaced by their fit on <task>_train."""
    datasets = _RECIPE_ROOT / "datasets"
    stem, posed = (
        (src[: -len("_posed")], "_posed") if src.endswith("_posed") else (src, "")
    )
    a = OmegaConf.to_container(
        OmegaConf.load(datasets / f"{stem}_{task}{posed}.yaml"), resolve=False
    )
    b = OmegaConf.to_container(OmegaConf.load(datasets / f"{src}.yaml"), resolve=False)
    fit = SPLIT_FITS[task]
    (norm,) = [
        t
        for t in b["pipeline"]["transforms"]
        if t["_target_"] == "${dp:NormalizeMeshFields}"
    ]
    assert norm["fields"]["nut"] == {"type": "scalar", "mean": 7.9e-4, "std": 2.4e-3}
    norm["fields"]["nut"] = {"type": "scalar", **fit["nut"]}
    if "_ud" in src:
        assert b["pipeline"]["transforms"][-1] == SPEED_STD
        b["pipeline"]["transforms"][-1] = {**SPEED_STD, **fit["speed"]}
    else:
        assert b["isla_constants"] == {
            "reynolds": {"mean": 6.205, "std": 0.1347},
            "output_scale": {"nut": 7.511e-05},
        }
        b["isla_constants"] = fit["isla"]
    assert a == b
