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

"""SUPPORT readout: an interacting-trained ISLA made query-independent at inference by an independent support.

Every held-out case is read with an interior sample larger than the training request (a dataset variant whose
reader draws ``2 x request`` points, or every point); the sample is split into a support ``A`` (the first
``request`` points, the training law and count) and evaluation points ``B`` (the rest). Per case, float32:

- ``interacting``: the trained model's own protocol at ``B``: ``B`` requested as interacting tokens, padded with the
  leading points of ``A`` to the training request count when ``B`` is smaller (AirfRANS);
- ``support``: ``B`` as passive readers behind the support ``A`` (``support_points``);
- ``no_support``: ``B`` as passive readers with nothing in the interior writing (the routing mass removed);
- query independence of ``support``: 2,000 seeded sentinels of ``B`` alone behind ``A`` against the full ``B``;
- support law: ``B`` behind a *different* support ``A'`` (the leading points of ``B`` swapped with ``A``'s tail),
  the sensitivity of a prediction to which support sample was drawn.

usage: python support_probe.py --model M --dataset D --run-id R --request N --extra "..." --task T --out O.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

FIELDS = ("pressure", "velocity", "nut")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def exact_knn(points, queries, k):
    """Exact nearest neighbours by a chunked float64 cdist: the float64 checks use it instead of the library search
    (cuML on CUDA computes its distances in single precision and can return out-of-range indices on posed 2-D input)."""
    ii, vv = [], []
    for qc in queries.double().split(8192):
        v, i = torch.cdist(qc, points.double()).topk(k, largest=False)
        ii.append(i)
        vv.append(v)
    return torch.cat(ii), torch.cat(vv)


def main():
    p = argparse.ArgumentParser()
    for k in ("model", "dataset", "run-id", "task", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--request", type=int, required=True, help="the training request count")
    p.add_argument("--cells", type=int, required=True, help="boundary cells (sampling_resolution)")
    p.add_argument("--extra", default="")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--runs-dir", default=None, help="checkpoint root (default <task>/runs)")
    p.add_argument("--n-detail", type=int, default=2, help="cases with the float64 and posed checks")
    p.add_argument("--n-uniform", type=int, default=20000, help="size of the volume-uniform-like evaluation set (0: off)")
    a = p.parse_args()
    for k, v in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(27000 + os.getpid() % 20000)).items():
        os.environ.setdefault(k, v)
    import hydra
    from datasets import build_dataloaders, find_normalizer
    from hydra import compose, initialize_config_dir
    from metrics import MetricCalculator, resolve_metrics
    from output_normalize import normalize_output_to_tensordict, require_output_type
    from utils import get_autocast_context, recursive_to_device

    import physicsnemo.experimental.models.isla.geometry as geom
    from physicsnemo import datapipes  # noqa: F401
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.calculus.measure import cell_measures
    from physicsnemo.utils import load_checkpoint

    t0 = time.time()
    runs = a.runs_dir or f"{a.task}/runs"
    ov = [f"model={a.model}", *[o for o in a.extra.split() if "training." not in o], f"dataset={a.dataset}", f"run_id={a.run_id}",
          "infer_split=val", f"sampling_resolution={a.cells}", "precision=float32", f"checkpoint_dir={runs}", "++model.compile_blocks=false"]
    log(f"run {a.run_id}: {ov}")
    with initialize_config_dir(config_dir=str(Path.cwd() / "conf"), version_base=None):
        cfg = compose(config_name="infer", overrides=ov)
    DistributedManager.initialize()
    dev = DistributedManager().device
    _, vl, normalizer, info = build_dataloaders(cfg)
    tc, ot = info["targets"], require_output_type(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dev)
    ck = str(Path(runs) / a.run_id / "checkpoints")
    ep = load_checkpoint(path=ck, models=model, device=dev)
    if ep <= 0:
        raise FileNotFoundError(f"no checkpoint restored from {ck}")
    sp = Path(ck) / "norm_stats.pt"
    if normalizer is not None and sp.exists():
        st = torch.load(sp, weights_only=True)
        for live in {id(n): n for n in (normalizer, find_normalizer([vl.dataset])) if n is not None}.values():
            live.stats = st
    model.eval()
    if model.query_independent:
        raise ValueError("the support readout needs an interacting-trained model")
    logw = float(model.qt.logw)
    log(f"loaded epoch {ep}, learned query routing mass log-weight {logw:.4f}")
    calc = MetricCalculator(target_config=tc, metrics=resolve_metrics(cfg))

    def predict(fk, pts, support=None, dtype=torch.float32):
        with torch.no_grad(), get_autocast_context(cfg.precision):
            out = model(**{**fk, "prediction_points": pts, "support_points": support})
        return normalize_output_to_tensordict(out, tc, ot).to(dtype)

    library_knn = geom.knn
    vq = [n for n, sp in model.query_schema.items() if sp.rank == 1]
    vg = [n for n, sp in model.global_schema.items() if sp.rank == 1]

    def move(fk, sets, R, t):
        """The case under x -> R x + t: fresh boundary meshes (geometry recomputed), point sets and their true vectors
        moved, rank-1 global directions rotated; R = I, t = 0 is the identically rebuilt reference."""
        dt = sets[0].points.dtype
        R, t = R.to(dev, dt), t.to(dev, dt)
        bm = {k: Mesh(points=m.points.to(dt) @ R.T + t, cells=m.cells, cell_data=m.cell_data.to(dt)) for k, m in fk["boundary_meshes"].items()}
        out = []
        for q in sets:
            pd = q.point_data.to(dt).clone()
            for nme in vq:
                pd[nme] = pd[nme] @ R.T
            out.append(Mesh(points=q.points @ R.T + t, point_data=pd))
        gd = fk["global_data"].to(dt).clone()
        for nme in vg:
            gd[nme] = gd[nme] @ R.T
        return {**fk, "boundary_meshes": bm, "global_data": gd}, out

    def scores(pred, tgt):
        m = calc(pred, tgt, None)
        return {f: float(m[f"{f}_l2"]) for f in FIELDS}

    def maxdiff(x, y, tgt):
        out = {}
        for f in FIELDS:
            dd = (x[f] - y[f]).reshape(x[f].shape[0], -1).norm(dim=-1)
            rms = tgt[f].reshape(tgt[f].shape[0], -1).norm(dim=-1).square().mean().sqrt()
            out[f] = float(dd.max() / rms.clamp_min(1e-30))
        return out

    cases = []
    idxs = list(vl.sampler)
    if a.max_cases > 0:
        idxs = idxs[: a.max_cases]
    for ci, idx in enumerate(idxs):
        tcase = time.time()
        b = recursive_to_device(vl.collate_fn([vl.dataset[idx]]), dev)
        fk, tgt = b["forward_kwargs"], b["targets"].float()
        pp = fk["prediction_points"]
        n = pp.n_points
        if n <= a.request + 2000:
            raise ValueError(f"case {ci}: the interior sample has {n} points; the readout needs more than request + 2,000")
        ### A and B are seeded random halves of the case's interior sample, whatever the stored point order.
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(4000 + ci)).to(dev)
        A = perm[: a.request]
        B = perm[a.request : min(n, 2 * a.request)]
        nb = B.numel()
        ppA, ppB, tB = pp.slice_points(A), pp.slice_points(B), tgt[B]
        ### interacting at B on the training request count (padded with A's leading points when B is smaller)
        pad = a.request - nb
        req = torch.cat([B, A[:pad]]) if pad > 0 else B
        inter = predict(fk, pp.slice_points(req))[:nb]
        sup = predict(fk, ppB, support=ppA)
        model.query_independent = True  # the routing mass removed: nothing in the interior writes
        try:
            nosup = predict(fk, ppB)
        finally:
            model.query_independent = False
        gen = torch.Generator().manual_seed(3000 + ci)
        s = torch.randperm(nb, generator=gen)[:2000].to(dev)
        alone = predict(fk, ppB.slice_points(s), support=ppA)
        ### a different support sample of the same law and count: A's tail swapped for B's head
        k = min(nb // 2, a.request // 2)
        A2 = torch.cat([A[: a.request - k], B[:k]])
        Brest = B[k:]
        sup2 = predict(fk, pp.slice_points(Brest), support=pp.slice_points(A2))
        sdf = pp.point_data["sdf"].reshape(-1).abs() if "sdf" in pp.point_data.keys() else pp.points.norm(dim=-1)
        qs = torch.tensor([0.1, 0.5, 0.9], device=dev, dtype=sdf.dtype)
        ### support count: B behind the leading quarter and half of A (the mass is split over the support)
        count = {str(a.request // m): scores(predict(fk, ppB, support=pp.slice_points(A[: a.request // m])), tB) for m in (4, 2)}
        uni = {}
        if a.n_uniform > 0:
            ### "Evaluate anywhere": a volume-uniform-like subset U of B near the body (|sdf| < 0.5 L, L the boundary's
            ### RMS pairwise distance), drawn with probability proportional to r_16^d (the distance to B's 16th nearest
            ### neighbour, a local-volume proxy) without replacement. U behind the support A (passive), U requested alone
            ### (interacting, the request law shifted), and U inside an on-protocol interacting request (padded with A).
            dd = pp.points.shape[-1]
            bpts = torch.cat([m.cell_centroids for m in fk["boundary_meshes"].values()])
            meas = torch.cat([cell_measures(m) for m in fk["boundary_meshes"].values()])
            L = float(geom.rms_pairwise_distance(bpts[None], meas[None]).reshape(()))
            xb = pp.points[B]
            rk = torch.cat([torch.cdist(c, xb).topk(17, largest=False).values[:, -1] for c in xb.split(8192)])
            sdfb = pp.point_data["sdf"].reshape(-1)[B].abs() if "sdf" in pp.point_data.keys() else torch.zeros_like(rk)
            wgt = rk.double() ** dd * (sdfb < 0.5 * L)
            m_u = min(a.n_uniform, int((wgt > 0).sum()) // 2)
            U = B[torch.multinomial(wgt / wgt.sum(), m_u, replacement=False, generator=torch.Generator(device=dev).manual_seed(6000 + ci))]
            ppU, tU = pp.slice_points(U), tgt[U]
            u_sup = predict(fk, ppU, support=ppA)
            u_alone = predict(fk, ppU)
            u_in = predict(fk, pp.slice_points(torch.cat([U, A[: a.request - m_u]])))[:m_u]
            uni = {"n": m_u, "L": L, "support": scores(u_sup, tU), "interacting_alone": scores(u_alone, tU),
                   "interacting_in_request": scores(u_in, tU),
                   "sdf_quantiles_B": [float(x) for x in sdfb.quantile(sdfb.new_tensor([0.1, 0.5, 0.9]))],
                   "sdf_quantiles_U": [float(x) for x in pp.point_data["sdf"].reshape(-1)[U].abs().quantile(sdfb.new_tensor([0.1, 0.5, 0.9]))]
                   if "sdf" in pp.point_data.keys() else None}
        detail = {}
        if ci < a.n_detail:
            ### covariance with the support moved along (float32 and float64), and float64 independence
            gen = torch.Generator().manual_seed(5000 + ci)
            q, _ = torch.linalg.qr(torch.randn(pp.points.shape[-1], pp.points.shape[-1], generator=gen, dtype=torch.float64))
            if torch.det(q) < 0:
                q[:, 0] = -q[:, 0]
            t = torch.randn(pp.points.shape[-1], generator=gen, dtype=torch.float64) * float(ppA.points.std(0).norm())
            eye, zero = torch.eye(q.shape[0], dtype=torch.float64), torch.zeros(q.shape[0], dtype=torch.float64)
            geom.knn = exact_knn  # the posed checks compare like with like under an exact search
            for tag, dt in (("fp32", torch.float32), ("fp64", torch.float64)):
                if dt == torch.float64:
                    model.double()
                A_, B_ = ppA.to(dt), ppB.to(dt)
                fk0, (A0, B0) = move(fk, [A_, B_], eye, zero)
                fk1, (A1, B1) = move(fk, [A_, B_], q, t)
                y0 = predict(fk0, B0, support=A0, dtype=dt)
                y1 = predict(fk1, B1, support=A1, dtype=dt)
                y1["velocity"] = y1["velocity"] @ q.to(dev, dt)
                detail[f"posed_rel_{tag}"] = {f: float((y1[f] - y0[f]).norm() / y0[f].norm()) for f in FIELDS}
                if dt == torch.float64:
                    al = predict(fk0, B0.slice_points(s), support=A0, dtype=dt)
                    detail["sentinel_alone_maxdiff_fp64"] = maxdiff(al, y0[s], tB[s].double())
                    model.float()
            geom.knn = library_knn
            log("detail " + json.dumps(detail))
        rec = {"case": ci, "sample_index": int(idx), "n_interior": n, "n_support": a.request, "n_eval": nb, **detail,
               **({"uniform": uni} if uni else {}), "support_count": count,
               "law_check_quantiles": {"A": [float(x) for x in sdf[A].quantile(qs)], "B": [float(x) for x in sdf[B].quantile(qs)]},
               "interacting": scores(inter, tB), "support": scores(sup, tB), "no_support": scores(nosup, tB),
               "sentinel_alone_maxdiff": maxdiff(alone, sup[s], tB[s]),
               "support_swap": {"n_eval": int(Brest.numel()), "support": scores(sup[k:], tgt[Brest]), "swapped": scores(sup2, tgt[Brest]),
                                "rel_change": {f: float((sup2[f] - sup[k:][f]).norm() / sup[k:][f].norm()) for f in FIELDS}},
               "seconds": round(time.time() - tcase, 2)}
        cases.append(rec)
        r = {f: rec["support"][f] / rec["interacting"][f] for f in FIELDS}
        log(f"case {ci + 1}/{len(idxs)} n={n} eval={nb} interacting " + " ".join(f"{f}={rec['interacting'][f]:.4f}" for f in FIELDS)
            + " | support/interacting " + " ".join(f"{f}={v:.3f}" for f, v in r.items())
            + " | no_support/interacting " + " ".join(f"{f}={rec['no_support'][f] / rec['interacting'][f]:.3f}" for f in FIELDS)
            + f" | sentinel maxdiff p={rec['sentinel_alone_maxdiff']['pressure']:.1e} | swap change p={rec['support_swap']['rel_change']['pressure']:.1e}"
            + (f" | uniform n={uni['n']} support/in-request p={uni['support']['pressure'] / uni['interacting_in_request']['pressure']:.3f}"
               f" u={uni['support']['velocity'] / uni['interacting_in_request']['velocity']:.3f}, alone/in-request p={uni['interacting_alone']['pressure'] / uni['interacting_in_request']['pressure']:.3f}"
               f" u={uni['interacting_alone']['velocity'] / uni['interacting_in_request']['velocity']:.3f}" if uni else "")
            + f" ({rec['seconds']} s)")
    mean = lambda key: {f: sum(c[key][f] for c in cases) / len(cases) for f in FIELDS}  # noqa: E731
    summ = {k: mean(k) for k in ("interacting", "support", "no_support")}
    summ["support_over_interacting"] = {f: summ["support"][f] / summ["interacting"][f] for f in FIELDS}
    summ["no_support_over_interacting"] = {f: summ["no_support"][f] / summ["interacting"][f] for f in FIELDS}
    summ["sentinel_alone_maxdiff_max"] = {f: max(c["sentinel_alone_maxdiff"][f] for c in cases) for f in FIELDS}
    summ["support_swap_rel_change_mean"] = {f: sum(c["support_swap"]["rel_change"][f] for c in cases) / len(cases) for f in FIELDS}
    summ["support_swap_score_ratio"] = {f: sum(c["support_swap"]["swapped"][f] for c in cases) / sum(c["support_swap"]["support"][f] for c in cases) for f in FIELDS}
    summ["support_count"] = {k: {f: sum(c["support_count"][k][f] for c in cases) / len(cases) for f in FIELDS} for k in cases[0]["support_count"]}
    un = [c for c in cases if "uniform" in c]
    if un:
        summ["uniform"] = {k: {f: sum(c["uniform"][k][f] for c in un) / len(un) for f in FIELDS} for k in ("support", "interacting_alone", "interacting_in_request")}
    det = [c for c in cases if "posed_rel_fp64" in c]
    if det:
        for k in ("posed_rel_fp32", "posed_rel_fp64", "sentinel_alone_maxdiff_fp64"):
            summ[f"{k}_max"] = {f: max(c[k][f] for c in det) for f in FIELDS}
    out = {"run_id": a.run_id, "model": a.model, "dataset": a.dataset, "epoch": ep, "query_logw": logw, "request": a.request,
           "n_cases": len(cases), "summary": summ, "cases": cases, "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".tmp").write_text(json.dumps(out, indent=1))
    os.replace(a.out + ".tmp", a.out)
    log(f"wrote {a.out}: support/interacting {json.dumps({f: round(v, 4) for f, v in summ['support_over_interacting'].items()})}, "
        f"no_support/interacting {json.dumps({f: round(v, 4) for f, v in summ['no_support_over_interacting'].items()})}")


if __name__ == "__main__":
    main()
