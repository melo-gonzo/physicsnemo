@register()
class BoundaryValueExtension(MeshTransform):
    r"""Extend prescribed boundary values into the interior as per-node query scalars (LIT-EXT, 2026-09-22).

    For each condition type ``c`` (a 0/1 per-cell label on the boundary mesh) the extension at an interior node
    ``x`` is the measure-weighted inverse-square-distance average of the prescribed value ``g`` over the cells of
    that type,

        ext_c(x) = sum_i w_i m_ci g_i / (|x - y_i|^2 + eps^2)  /  sum_i w_i m_ci / (|x - y_i|^2 + eps^2),

    with ``y_i`` the cell centroids, ``w_i`` the quadrature measure and ``m_ci`` the label; nodes of a case with
    no cell of type ``c`` get zero. A second field per type, ``dist_c(x) = min_i |x - y_i|`` over the labeled
    cells (a large constant when the type is absent), tells the query how far the nearest cell of that type is.
    The construction is parameter-free, translation- and rotation-invariant (distances only), and mesh-consistent
    (the measure weights make it a quadrature of an integral). Motivated by the learned function extensions of the
    OLBC paper (Mousavi, Mishra, De Lorenzis 2026) and BENO's boundary branch: it lets a boundary value reach the
    readout at a query directly, instead of only through the slice attention. Fields written to
    ``interior.point_data``: ``<prefix><type>`` and ``<prefix>dist_<type>``.
    """

    def __init__(
        self,
        boundary: str,
        value_field: str,
        label_fields: Sequence[str],
        measure_field: str = "quadrature_measure",
        prefix: str = "ext_",
        eps: float = 0.02,
        absent_distance: float = 2.0,
    ) -> None:
        super().__init__()
        self.boundary = boundary
        self.value_field = value_field
        self.label_fields = list(label_fields)
        self.measure_field = measure_field
        self.prefix = prefix
        self.eps = float(eps)
        self.absent_distance = float(absent_distance)

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh  # a bare Mesh has no interior/boundary split; nothing to extend

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        bnd = domain.boundaries[self.boundary]
        x = domain.interior.points  # (N, 3)
        y = bnd.cell_centroids.to(x.dtype)  # (M, 3)
        g = bnd.cell_data[self.value_field].to(x.dtype)
        g = g.reshape(y.shape[0], -1)  # (M, C): C = 1 for a scalar datum, 3 for a vector datum (rank-1 extension)
        w = bnd.cell_data[self.measure_field].reshape(-1).to(x.dtype)
        d2 = torch.cdist(x, y).pow(2)  # (N, M)
        kern = 1.0 / (d2 + self.eps**2)
        new_pd = domain.interior.point_data.clone()
        for lf in self.label_fields:
            m = bnd.cell_data[lf].reshape(-1).to(x.dtype)
            wm = w * m
            den = kern @ wm  # (N,)
            num = kern @ (wm[:, None] * g)  # (N, C)
            ext = torch.where(den[:, None] > 0, num / den.clamp_min(1e-30)[:, None], torch.zeros_like(num))
            if bool(m.sum() > 0):
                dist = d2[:, m > 0].min(dim=1).values.sqrt()
            else:
                dist = torch.full_like(den, self.absent_distance)
            short = lf.replace("bc_", "")
            new_pd[f"{self.prefix}{short}"] = ext  # (N, C)
            new_pd[f"{self.prefix}dist_{short}"] = dist.reshape(-1, 1)
        interior = Mesh(
            points=domain.interior.points,
            cells=domain.interior.cells,
            point_data=new_pd,
            cell_data=domain.interior.cell_data,
            global_data=domain.interior.global_data,
        )
        return DomainMesh(interior=interior, boundaries=domain.boundaries, global_data=domain.global_data)

    def extra_repr(self) -> str:
        return f"boundary={self.boundary!r}, value={self.value_field!r}, labels={self.label_fields}"
