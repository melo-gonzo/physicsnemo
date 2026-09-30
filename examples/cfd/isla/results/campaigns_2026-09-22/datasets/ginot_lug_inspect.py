"""Inspect the raw GINOT PLASTIC_LUG layout (run as a CPU batch job: LUG_cells.pkl is 1.7 GB)."""
import pickle, sys, numpy as np
root = sys.argv[1]
d = pickle.load(open(f"{root}/LUG_node_S_PC.pkl", "rb"))
c = pickle.load(open(f"{root}/LUG_cells.pkl", "rb"))
print("cells type", type(c), getattr(c, "keys", lambda: None)())
def desc(o, depth=0, n=3):
    p = "  " * depth
    if isinstance(o, dict):
        print(p + f"dict keys={list(o.keys())[:10]}")
        for k in list(o.keys())[:n]:
            print(p + f" [{k!r}]"); desc(o[k], depth + 1)
    elif isinstance(o, (list, tuple)):
        print(p + f"{type(o).__name__} len={len(o)} types={[type(x).__name__ for x in o[:4]]}")
        for x in o[:n]: desc(x, depth + 1)
    elif hasattr(o, "shape"):
        a = np.asarray(o); print(p + f"array {a.shape} {a.dtype} min={a.min():.4g} max={a.max():.4g} first={a.ravel()[:8]}")
    else:
        print(p + repr(o)[:120])
desc(c)
params = np.load(f"{root}/input_params.npy")
V, S, PC = d["vertices"], d["nodal_stress"], d["points_cloud"]
cells = c if not isinstance(c, dict) else c[list(c.keys())[0]]
for i in [0, 1, 2]:
    v, s, pc = V[i], S[i], PC[i]; ce = np.asarray(cells[i])
    print(f"--- sample {i}: params={params[i]} nodes={len(v)} pc={len(pc)} cells={ce.shape} {ce.dtype} min={ce.min()} max={ce.max()}")
    print("  bbox", v.min(0), v.max(0))
    # is the point cloud a subset of the vertices?
    from scipy.spatial import cKDTree
    dd, ii = cKDTree(v).query(pc); print("  pc subset of vertices: maxdist", dd.max(), "unique idx", len(np.unique(ii)))
    # surface faces of a tet/hex mesh: faces used by exactly one cell
    if ce.shape[1] == 4:
        f = np.concatenate([ce[:, [0, 1, 2]], ce[:, [0, 1, 3]], ce[:, [0, 2, 3]], ce[:, [1, 2, 3]]])
    elif ce.shape[1] == 10:
        f = np.concatenate([ce[:, [0, 1, 2]], ce[:, [0, 1, 3]], ce[:, [0, 2, 3]], ce[:, [1, 2, 3]]])
    elif ce.shape[1] == 8:
        f = np.concatenate([ce[:, q] for q in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])])
    else:
        f = None
    if f is not None:
        key = np.sort(f, 1); _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        bf = f[cnt[inv.ravel()] == 1]; bn = np.unique(bf)
        print("  boundary faces", len(bf), "boundary nodes", len(bn), "pc nodes in boundary set:", np.isin(ii, bn).mean())
        # stress near the pressure: which surface has |x| plane, hole, etc.
        print("  stress on boundary nodes: min/median/max", s[bn].min(), np.median(s[bn]), s[bn].max(), "; interior nodes", np.median(np.delete(s, bn)))
    print("  stress stats", s.min(), s.mean(), s.max())
print("node counts: min/mean/max", min(map(len, V)), np.mean([len(v) for v in V]), max(map(len, V)))
print("pc counts: min/mean/max", min(map(len, PC)), np.mean([len(v) for v in PC]), max(map(len, PC)))
print("param ranges", params.min(0), params.max(0))
print("INSPECT-DONE")
