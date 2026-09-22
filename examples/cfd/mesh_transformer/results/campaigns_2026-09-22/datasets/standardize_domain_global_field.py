

@register()
class StandardizeDomainGlobalField(MeshTransform):
    r"""Write ``(global_data[field] - mean) / std`` to a NEW domain-level ``global_data`` key.

    ``NormalizeMeshFields`` standardizes point/cell fields only, and the base
    ``apply_to_domain`` broadcast would touch sub-mesh ``global_data`` rather
    than the domain-level dict the recipe's ``forward_kwargs`` read. This
    transform standardizes one global vector of per-case scalars (PDE
    parameters, loads, design variables) with per-component ``mean`` /
    ``std`` measured over the training split, leaving the raw field in place.

    Example (PLAID Tensile2d; stats from the converter manifest)::

        - _target_: ${dp:StandardizeDomainGlobalField}
          field: params
          output_field: params_std
          mean: [-45.0, 14.94, 450.8, 1502.6, 1496.5, 75434.8]
          std: [2.90, 2.87, 86.1, 286.1, 289.2, 14613.8]
    """

    def __init__(
        self,
        field: str,
        mean: Sequence[float] | float,
        std: Sequence[float] | float,
        output_field: str | None = None,
    ) -> None:
        super().__init__()
        self.field = field
        self.output_field = output_field or f"{field}_std"
        self._mean = torch.as_tensor(mean, dtype=torch.float32)
        self._std = torch.as_tensor(std, dtype=torch.float32)
        if bool((self._std <= 0).any()):
            raise ValueError(f"std must be positive, got {std!r}")

    def _standardize(self, gd: TensorDict, like: torch.Tensor) -> TensorDict:
        if self.field not in gd.keys():
            raise KeyError(
                f"StandardizeDomainGlobalField needs global_data[{self.field!r}]; "
                f"available: {sorted(gd.keys())!r}"
            )
        x = gd[self.field].to(dtype=torch.float32)
        mean = self._mean.to(x.device)
        std = self._std.to(x.device)
        if mean.numel() not in (1, x.numel()) or std.numel() not in (1, x.numel()):
            raise ValueError(
                f"global_data[{self.field!r}] has {x.numel()} entries but mean/std have "
                f"{mean.numel()}/{std.numel()}"
            )
        if mean.numel() > 1:
            mean = mean.reshape(x.shape)
        if std.numel() > 1:
            std = std.reshape(x.shape)
        new_gd = gd.clone()
        new_gd[self.output_field] = ((x - mean) / std).to(like.dtype)
        return new_gd

    def __call__(self, mesh: Mesh) -> Mesh:
        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=self._standardize(mesh.global_data, mesh.points),
        )

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=self._standardize(domain.global_data, domain.interior.points),
        )

    def extra_repr(self) -> str:
        return f"field={self.field!r}, output_field={self.output_field!r}"
