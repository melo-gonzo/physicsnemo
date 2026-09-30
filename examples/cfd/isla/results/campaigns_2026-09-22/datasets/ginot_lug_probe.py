"""Geometry probe for GINOT PLASTIC_LUG (CPU job): surface faces of the hex8 mesh, planar faces, hole cylinder, stress by region."""
import pickle, sys, numpy as np
root = sys.argv[1]
d = pickle.load(open(f"{root}/LUG_node_S_PC.pkl", "rb"))
cells = pickle.load(open(f"{root}/LUG_cells.pkl", "rb"))
params = np.load(f"{root}/input_params.npy")
HEX_FACES = ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])
for i in [0, 1, 2, 1500, 2999]:
    v = d["vertices"][i].astype(np.float64); s = d["nodal_stress"][i]; pc = d["points_cloud"][i]; ce = np.asarray(cells[i])[:, 1:]
    r, L, t, P = params[i]
    f = np.concatenate([ce[:, q] for q in HEX_FACES])
    key = np.sort(f, 1); _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bf = f[cnt[inv.ravel()] == 1]; bn = np.unique(bf)
    print(f"=== sample {i}: r={r:.5f} L={L:.5f} t={t:.5f} P={P:.4g}  nodes={len(v)} hex={len(ce)} bfaces={len(bf)} bnodes={len(bn)} pc={len(pc)}")
    print("  bbox", v.min(0), v.max(0))
    x, y, z = v[bn].T; eps = 1e-6
    planes = {"x=xmin": np.abs(x - x.min()) < eps, "x=xmax": np.abs(x - x.max()) < eps, "y=ymin": np.abs(y - y.min()) < eps,
              "y=ymax": np.abs(y - y.max()) < eps, "z=0": np.abs(z - z.min()) < eps, "z=t": np.abs(z - z.max()) < eps}
    for k, m in planes.items():
        print(f"  {k}: {m.sum()} nodes, stress mean {s[bn][m].mean():.2f}")
    other = ~np.any(np.stack(list(planes.values())), 0)
    ox, oy = x[other], y[other]
    # fit a circle to the curved nodes (algebraic fit)
    A = np.stack([ox, oy, np.ones_like(ox)], 1); b = ox ** 2 + oy ** 2
    sol = np.linalg.lstsq(A, b, rcond=None)[0]; xc, yc = sol[0] / 2, sol[1] / 2; rad = np.sqrt(sol[2] + xc ** 2 + yc ** 2)
    rr = np.sqrt((ox - xc) ** 2 + (oy - yc) ** 2)
    print(f"  curved nodes {other.sum()}: circle fit center=({xc:.5f},{yc:.5f}) R={rad:.5f}; radial spread {rr.min():.5f}..{rr.max():.5f}")
    # classify curved nodes by radius clusters
    hist, edges = np.histogram(rr, bins=12); print("  radial hist", list(zip(np.round(edges[:-1], 4), hist)))
    hole = other.copy(); hole[other] = np.abs(rr - r) < 5e-4
    outer = other & ~hole
    print(f"  hole-radius nodes {hole.sum()} (stress mean {s[bn][hole].mean():.1f}); other curved {outer.sum()} (stress mean {s[bn][outer].mean():.1f}), their radii {np.unique(np.round(rr[np.abs(rr - r) >= 5e-4], 4))[:8]}")
    hy = y[hole]; hx = x[hole]
    for name, m in {"hole y<yc": hy < yc - 1e-6, "hole y>yc": hy > yc + 1e-6, "hole x<xc": hx < xc - 1e-6, "hole x>xc": hx > xc + 1e-6}.items():
        print(f"    {name}: {m.sum()} nodes stress mean {s[bn][hole][m].mean():.1f} max {s[bn][hole][m].max():.1f}")
    # angle-resolved stress around the hole
    ang = np.degrees(np.arctan2(hy - yc, hx - xc)); bins = np.arange(-180, 181, 30)
    idx = np.digitize(ang, bins) - 1
    print("    stress by angle bin (deg from +x, ccw):", [(int(bins[k]), round(float(s[bn][hole][idx == k].mean()), 1)) for k in range(12) if (idx == k).any()])
    # faces: face-node membership in the point cloud
    from scipy.spatial import cKDTree
    dd, ii = cKDTree(v).query(pc); print("  pc == boundary node set:", set(ii.tolist()) == set(bn.tolist()))
print("PROBE-DONE")
