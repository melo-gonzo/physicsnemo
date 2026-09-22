"""Load-back check of converted OLBC DomainMesh trees: shapes, sdf sign, per-loop closure (sum of outward normal x length
per loop ~ 0), label coverage (every boundary cell exactly one BC type), nodal-tag consistency, manifest split sizes.

usage: python olbc_verify.py <out_root> <case> [<case> ...]     (cases are dir names, e.g. train_000123)
"""
import json, sys
from pathlib import Path
import torch
from physicsnemo.mesh import DomainMesh

root, cases = Path(sys.argv[1]), sys.argv[2:]
ok = True
for c in cases:
    dm = DomainMesh.load(root / c / f"{c}.pdmsh")
    I = dm.interior; B = dm.boundaries["outer"]
    print(c, "interior points", tuple(I.points.shape), "boundary points", tuple(B.points.shape), "cells", tuple(B.cells.shape))
    for k, v in I.point_data.items():
        print(f"   point_data.{k}: {tuple(v.shape)} {v.dtype} min={v.min().item():.4g} max={v.max().item():.4g} mean={v.float().mean().item():.4g}")
    for k, v in B.cell_data.items():
        s = f"sum={v.sum().item():.4g}" if v.ndim == 1 else f"norm-mean={v.norm(dim=-1).mean().item():.4g}"
        print(f"   cell_data.{k}: {tuple(v.shape)} {v.dtype} {s}")
    sdf = I.point_data["sdf"]
    print(f"   sdf: max={sdf.max().item():.3g} (must be 0: nodes on/inside the solid) min={sdf.min().item():.4g} n(sdf==0)={(sdf == 0).sum().item()}")
    ok &= sdf.max().item() <= 0.0
    n, L, lid = B.cell_data["normals"], B.cell_data["quadrature_measure"], B.cell_data["loop_id"]
    print(f"   |normals| in [{n.norm(dim=-1).min().item():.4f}, {n.norm(dim=-1).max().item():.4f}], normals.z max |{n[:, 2].abs().max().item():.1e}|")
    # geometric consistency: the cell normal is perpendicular to the segment
    a, b = B.points[B.cells[:, 0]], B.points[B.cells[:, 1]]
    t = b - a; seglen = t.norm(dim=-1); perp = ((t / seglen[:, None]) * n).sum(-1).abs().max().item()
    print(f"   segment length vs quadrature_measure max |diff| {(seglen - L).abs().max().item():.2e}; max |t . n| {perp:.2e}")
    ok &= perp < 1e-4
    for li in lid.unique().tolist():
        m = lid == li; cl = (n[m] * L[m, None]).sum(0).norm().item(); per = L[m].sum().item()
        # outward check: normals at the exterior loop point away from the origin (square/circle centred at 0); into the hole on holes
        cen = 0.5 * (a[m] + b[m]); mid_c = cen.mean(0); radial = ((cen - mid_c) * n[m]).sum(-1).mean().item()
        print(f"   loop {int(li)}: cells {int(m.sum())} perimeter {per:.4f} closure |sum n L| {cl:.2e} (rel {cl / per:.1e}); mean (c - centre).n = {radial:+.3f} ({'outward' if li == 0 else 'into the hole (negative expected)'})")
        ok &= cl / per < 1e-6
    groups = [["bc_dirichlet_x", "bc_neumann_x"], ["bc_dirichlet_y", "bc_neumann_y"]] if "bc_neumann_x" in B.cell_data else [["bc_dirichlet", "bc_neumann", "bc_robin"]]
    for labels in groups:
        cols = torch.stack([B.cell_data[k] for k in labels], 1); tot = cols.sum(1)
        print(f"   label coverage {labels}: exactly one type {(tot == 1).sum().item()}/{len(tot)}; none {(tot == 0).sum().item()}; >1 {(tot > 1).sum().item()}; per type {cols.sum(0).tolist()}")
        ok &= bool((tot == 1).all())
        # nodal tags on the boundary mesh: a cell of type X has at least one end tagged X
        for k in labels:
            tag = B.point_data["tag_" + k[3:]]
            ends = tag[B.cells].max(1).values
            bad = int(((B.cell_data[k] > 0) & (ends == 0)).sum()); ok &= bad == 0
            print(f"      {k}: cells whose no end carries tag_{k[3:]}: {bad}")
    print("  global_data:", {k: (round(v.item(), 6) if v.numel() == 1 else tuple(v.shape)) for k, v in dm.global_data.items()})
m = json.load(open(root / "manifest.json"))
print("manifest splits:", {k: len(v) for k, v in m.items() if isinstance(v, list)})
print("nested:", set(m["train_n512"]) <= set(m["train_n2048"]) <= set(m["train"]), "disjoint train/test:", not set(m["train"]) & set(m["test"]))
print("_stats.train:", json.dumps(m["_stats"]["train"], indent=1)[:3000])
print("VERIFY-DONE", "OK" if ok else "FAILED")
