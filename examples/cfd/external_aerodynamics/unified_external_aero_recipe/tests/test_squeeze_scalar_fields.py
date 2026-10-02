"""SqueezeScalarFields (2026-10-02): one-channel scalar columns become plain (N,) fields on the named sub-meshes;
already-flat fields pass; non-scalar fields are refused; unnamed fields are untouched."""

import pytest
import torch
from tensordict import TensorDict

from domain_transforms import SqueezeScalarFields
from physicsnemo.mesh import DomainMesh, Mesh


def _domain(n_int: int = 50, m: int = 8) -> DomainMesh:
    pts = torch.rand(m, 3)
    cells = torch.stack([torch.arange(m), (torch.arange(m) + 1) % m], 1)
    cd = TensorDict({"normals": torch.randn(m, 3), "bc_value": torch.randn(m, 1), "flat": torch.randn(m)}, batch_size=[m])
    bnd = Mesh(points=pts, cells=cells, cell_data=cd)
    pd = TensorDict({"sdf": torch.randn(n_int, 1), "sdf_normals": torch.randn(n_int, 3), "u": torch.randn(n_int, 1)}, batch_size=[n_int])
    interior = Mesh(points=torch.rand(n_int, 3), cells=torch.zeros(0, 1, dtype=torch.long), point_data=pd)
    return DomainMesh(interior=interior, boundaries={"airfoil": bnd})


def test_named_columns_are_squeezed_and_the_rest_untouched():
    d = _domain()
    out = SqueezeScalarFields(["interior.point_data.sdf", "boundaries.airfoil.cell_data.bc_value",
                               "boundaries.airfoil.cell_data.flat", "interior.point_data.missing"]).apply_to_domain(d)
    assert out.interior.point_data["sdf"].shape == (50,)
    torch.testing.assert_close(out.interior.point_data["sdf"], d.interior.point_data["sdf"][:, 0])
    assert out.interior.point_data["u"].shape == (50, 1)  # not named: untouched
    assert out.interior.point_data["sdf_normals"].shape == (50, 3)
    assert out.boundaries["airfoil"].cell_data["bc_value"].shape == (8,)
    assert out.boundaries["airfoil"].cell_data["flat"].shape == (8,)  # already flat: passes
    assert out.boundaries["airfoil"].cell_data["normals"].shape == (8, 3)


def test_non_scalar_field_is_refused_and_paths_are_validated():
    d = _domain()
    with pytest.raises(ValueError, match="a scalar field is"):
        SqueezeScalarFields(["interior.point_data.sdf_normals"]).apply_to_domain(d)
    with pytest.raises(ValueError, match="must be interior"):
        SqueezeScalarFields(["sdf"]).apply_to_domain(d)
