"""Load-back check of converted DeepJEB++ DomainMesh trees: outward closed surface (sum n A ~ 0, positive signed volume,
normals parallel to the face area vectors), label coverage (fixed/loaded cells present and disjoint, u = 0 at fixed cells),
the traction field only on loaded cells with the right resultant (force) or moment (torque), target ranges, global_data,
manifest split sizes / nesting / pair disjointness.

usage: python deepjebpp_verify.py <out_root> <case> [<case> ...]     (cases are dir names, e.g. 012-015-IS02_ver)
"""
import json, sys
from pathlib import Path
import torch
from physicsnemo.mesh import DomainMesh

root, cases = Path(sys.argv[1]), sys.argv[2:]
ok = True
for c in cases:
    dm = DomainMesh.load(root / c / f"{c}.pdmsh")
    B = dm.boundaries["surface"]; I = dm.interior
    print(c, "surface points", tuple(B.points.shape), "cells", tuple(B.cells.shape), "interior points", tuple(I.points.shape))
    for k, v in B.cell_data.items():
        s = f"min={v.min().item():.4g} max={v.max().item():.4g} mean={v.float().mean().item():.4g}" if v.ndim == 1 else f"|.|max={v.norm(dim=-1).max().item():.4g}"
        print(f"   cell_data.{k}: {tuple(v.shape)} {v.dtype} {s}")
    n, A = B.cell_data["normals"], B.cell_data["quadrature_measure"]
    P, F = B.points.double(), B.cells
    av = 0.5 * torch.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]], dim=-1)
    area = av.norm(dim=-1)
    vol = (P[F[:, 0]] * torch.cross(P[F[:, 1]], P[F[:, 2]], dim=-1)).sum(-1).sum().item() / 6
    par = (av / area[:, None] * n.double()).sum(-1)
    clos = (n * A[:, None]).sum(0).norm().item() / A.sum().item()
    print(f"   area vs quadrature_measure max|diff| {(area - A.double()).abs().max().item():.2e}; normal.(face area vector) min {par.min().item():.6f}; "
          f"closure |sum n A|/sum A {clos:.2e}; signed volume {vol:.4g} mm^3 (global vol_mm3 {dm.global_data['vol_mm3'].item():.4g}); extent {(P.max(0).values - P.min(0).values).tolist()}")
    ok &= par.min().item() > 0.999 and clos < 1e-6 and vol > 0
    fx, ld = B.cell_data["bc_fixed"], B.cell_data["bc_loaded"]
    print(f"   fixed cells {int(fx.sum())}, loaded cells {int(ld.sum())}, both {int((fx * ld).sum())}; |u| at fixed cells max {B.cell_data['displacement'][fx > 0].norm(dim=-1).max().item():.3g} mm "
          f"(cell mean of vertices, so > 0 at the clamp edge is expected); interior tag_fixed nodes {int(I.point_data['tag_fixed'].sum())}: |u| max {I.point_data['displacement'][I.point_data['tag_fixed'] > 0].norm(dim=-1).max().item():.3g}")
    ok &= fx.sum() > 0 and ld.sum() > 0
    if (fx * ld).sum() > 0:
        print("   WARNING: cells touching both a bolt vertex and a lug vertex (reported, not fatal)")
    ok &= I.point_data["displacement"][I.point_data["tag_fixed"] > 0].norm(dim=-1).max().item() == 0.0
    t = B.cell_data["bc_load"].double(); Ad = A.double()
    off = t[ld == 0].norm(dim=-1).max().item()
    gd = dm.global_data; d = gd["load_dir"].double(); typ = gd["load_type"]; mag = gd["load_magnitude"].item()
    cen = P[F].mean(1)
    if typ[0] > 0:
        resid = ((t * Ad[:, None]).sum(0) - mag * d).norm().item() / mag; what = "force resultant"
    else:
        m = ld > 0; c0 = (cen[m] * Ad[m, None]).sum(0) / Ad[m].sum()
        mom = (torch.cross(cen[m] - c0, t[m], dim=-1) * Ad[m, None]).sum(0)
        resid = abs((mom @ d).item() - mag) / mag; what = f"axial moment about the lug centroid (transverse |M_perp|/M = {(mom - (mom @ d) * d).norm().item() / mag:.3g})"
    print(f"   load: type {typ.tolist()} dir {d.tolist()} magnitude {mag:g}; traction off loaded cells max {off:.2e}; {what} rel. residual {resid:.2e}; "
          f"|t| on loaded cells [{t[ld > 0].norm(dim=-1).min().item():.3g}, {t[ld > 0].norm(dim=-1).max().item():.3g}] MPa; bc_load_n/mag consistent "
          f"{((t * n.double()).sum(-1) - B.cell_data['bc_load_n'].double()).abs().max().item():.1e}/{(t.norm(dim=-1) - B.cell_data['bc_load_mag'].double()).abs().max().item():.1e}")
    ok &= off == 0.0 and resid < 1e-6
    vm = B.cell_data["von_mises"]; u = B.cell_data["displacement"]
    print(f"   targets: von_mises [{vm.min().item():.3g}, {vm.max().item():.3g}] MPa (label maxvm {gd['label_maxvm'].item():.3g}, p95 {gd['label_p95vm'].item():.3g}); "
          f"|displacement| max {u.norm(dim=-1).max().item():.3g} mm (label maxu {gd['label_maxu'].item():.3g})")
    ok &= vm.min().item() >= 0
    print("   global_data:", {k: (round(v.item(), 6) if v.numel() == 1 else v.tolist()) for k, v in gd.items()})
m = json.load(open(root / "manifest.json"))
print("manifest splits:", {k: len(v) for k, v in m.items() if isinstance(v, list)})
tb, te = set(m["train_brackets"]), set(m["test_brackets"])
pair = lambda b: "-".join(b.split("-")[:2])
print("nested:", set(m["train_n256"]) <= set(m["train_n1024"]) <= set(m["train_n4096"]) <= set(m["train"]),
      "disjoint train/test files:", not set(m["train"]) & set(m["test"]),
      "disjoint brackets:", not tb & te, "disjoint seed pairs:", not {pair(b) for b in tb} & {pair(b) for b in te},
      "n test brackets", len(te), "n train brackets", len(tb))
print("_stats.train:", json.dumps(m["_stats"]["train"], indent=1)[:4000])
print("VERIFY-DONE", "OK" if ok else "FAILED")
