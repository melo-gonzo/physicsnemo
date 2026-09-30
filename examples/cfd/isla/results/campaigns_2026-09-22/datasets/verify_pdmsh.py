"""Load-back check of converted DomainMesh trees (CPU job): shapes, dtypes, BC label sums, manifest split sizes.

usage: python verify_pdmsh.py <out_root> <boundary_name> <case> [<case> ...]
"""
import json, sys
from pathlib import Path
import torch
from physicsnemo.mesh import DomainMesh

root, bname, cases = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
for c in cases:
    p = root / c / f"{Path(c).name}.pdmsh"
    dm = DomainMesh.load(p)
    I = dm.interior
    print(c, "interior points", tuple(I.points.shape), I.points.dtype, "cells", None if I.cells is None else tuple(I.cells.shape))
    for k, v in I.point_data.items():
        print(f"   point_data.{k}: {tuple(v.shape)} {v.dtype} min={v.min().item():.4g} max={v.max().item():.4g}")
    B = dm.boundaries[bname]
    print(f"  boundary '{bname}': points {tuple(B.points.shape)} cells {tuple(B.cells.shape)} centroids {tuple(B.cell_centroids.shape)}")
    for k, v in B.cell_data.items():
        s = f"sum={v.sum().item():.4g}" if v.ndim == 1 else f"norm-mean={v.norm(dim=-1).mean().item():.4g}"
        print(f"   cell_data.{k}: {tuple(v.shape)} {v.dtype} {s}")
    for k, v in B.point_data.items():
        print(f"   point_data.{k}: {tuple(v.shape)} sum={v.sum().item():.4g}")
    # normals: unit length and, for 3-D triangles, aligned with the geometric cell normals
    n = B.cell_data["normals"]; print(f"   |normals| in [{n.norm(dim=-1).min().item():.4f}, {n.norm(dim=-1).max().item():.4f}]")
    if B.cells.shape[1] == 3:
        a, b, cc = (B.points[B.cells[:, i]] for i in range(3))
        g = torch.cross(b - a, cc - a, dim=-1); g = g / g.norm(dim=-1, keepdim=True)
        print(f"   normals . geometric normals: min={(g * n).sum(-1).min().item():.4f} (should be +1: winding matches the outward normal)")
    print("  global_data:", {k: (round(v.item(), 6) if v.numel() == 1 else tuple(v.shape)) for k, v in dm.global_data.items()})
m = json.load(open(root / "manifest.json"))
print("manifest splits:", {k: len(v) for k, v in m.items() if isinstance(v, list)})
print("VERIFY-DONE")
