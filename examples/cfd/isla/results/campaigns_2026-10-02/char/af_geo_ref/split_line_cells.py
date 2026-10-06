@register()
class SplitLineCells(MeshTransform):
    r"""Split every line cell of a planar boundary into collinear equal pieces (AF-GEO refinement protocol, 2026-10-06).

    A count-raising re-panelization that keeps the geometry and the boundary measure exactly: only the panel pattern
    changes, which is the discretization change the AirfRANS airfoil (about 1,000 segments, never subsampled at the
    campaign budget) admits without losing information. Cells whose centroid x lies ahead of the mid-point of the
    boundary's x extent are split into ``pieces_front`` pieces, the others into ``pieces_back`` (equal values give a
    uniform refinement; 3 / 1 a front-loaded one). New points are the interior division points of each cell; cell data
    is copied to every piece and an explicit ``_effective_measure`` is divided by the piece count (the pieces are
    equal), so the represented measure is unchanged; point data of new points is interpolated linearly. Meshes without
    line cells (the interior point cloud, surface meshes) are returned unchanged. Two-dimensional boundaries only:
    apply after ``RestrictToPlane``.
    """

    def __init__(self, pieces_front: int = 2, pieces_back: int = 2) -> None:
        super().__init__()
        if int(pieces_front) < 1 or int(pieces_back) < 1:
            raise ValueError("pieces_front and pieces_back must be positive integers")
        self.pieces_front = int(pieces_front)
        self.pieces_back = int(pieces_back)

    def __call__(self, mesh: Mesh) -> Mesh:
        if mesh.cells is None or mesh.n_cells == 0 or mesh.cells.shape[1] != 2:
            return mesh
        cells, pts = mesh.cells, mesh.points
        device = pts.device
        cx = mesh.cell_centroids[:, 0]
        xmid = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
        k = torch.where(cx < xmid, torch.full_like(cells[:, 0], self.pieces_front), torch.full_like(cells[:, 0], self.pieces_back))
        new_pts = [pts]
        new_cells = []
        parent = []
        interp = []  # (a, b, fraction) rows for the new points, in creation order
        n_pts = pts.shape[0]
        for kk in sorted(set(k.tolist())):
            sel = (k == kk).nonzero(as_tuple=True)[0]
            a, b = cells[sel, 0], cells[sel, 1]
            if kk == 1:
                new_cells.append(torch.stack([a, b], 1))
                parent.append(sel)
                continue
            fr = torch.arange(1, kk, dtype=pts.dtype, device=device) / kk  # (kk-1,)
            interior = pts[a][:, None, :] + (pts[b] - pts[a])[:, None, :] * fr[None, :, None]  # (M, kk-1, d)
            m = sel.numel()
            idx = n_pts + torch.arange(m * (kk - 1), device=device).reshape(m, kk - 1)
            n_pts += m * (kk - 1)
            new_pts.append(interior.reshape(-1, pts.shape[1]))
            interp.append((a.repeat_interleave(kk - 1), b.repeat_interleave(kk - 1), fr.repeat(m)))
            chain = torch.cat([a[:, None], idx, b[:, None]], 1)  # (M, kk+1)
            new_cells.append(torch.stack([chain[:, :-1], chain[:, 1:]], -1).reshape(-1, 2))
            parent.append(sel.repeat_interleave(kk))
        points = torch.cat(new_pts)
        cells_out = torch.cat(new_cells)
        parent = torch.cat(parent)
        cell_data = {}
        for key, value in mesh.cell_data.items():
            piece = value[parent]
            if key == EFFECTIVE_MEASURE_KEY:
                piece = piece / k[parent].to(piece.dtype)
            cell_data[key] = piece
        point_data = {}
        if len(mesh.point_data.keys()) and interp:
            ia = torch.cat([t[0] for t in interp])
            ib = torch.cat([t[1] for t in interp])
            f = torch.cat([t[2] for t in interp])
            for key, value in mesh.point_data.items():
                shape = (-1,) + (1,) * (value.dim() - 1)
                fv = f.reshape(shape).to(value.dtype)
                point_data[key] = torch.cat([value, value[ia] * (1 - fv) + value[ib] * fv])
        elif len(mesh.point_data.keys()):
            point_data = {key: value for key, value in mesh.point_data.items()}
        return Mesh(points=points, cells=cells_out, point_data=point_data or None, cell_data=cell_data or None, global_data=mesh.global_data)
