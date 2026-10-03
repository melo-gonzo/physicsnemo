"""Diagnose the float64 posed-test residual on one held-out case: rotation only, translation only, both; per-point
residual distribution; nearest-boundary index flips; and the same with an exact float64 cdist nearest-neighbour search
monkeypatched into the model's geometry module.  usage: python posed_diag.py --model M --dataset D --run-id R --cells N
--extra "..." --task T"""

import argparse
import os
import time

import torch


def main():
    p = argparse.ArgumentParser()
    for k in ("model", "dataset", "run-id", "task"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--cells", type=int, required=True)
    p.add_argument("--extra", default="")
    a = p.parse_args()
    for k, v in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(25000 + os.getpid() % 20000)).items():
        os.environ.setdefault(k, v)
    from pathlib import Path

    import hydra
    from datasets import build_dataloaders, find_normalizer
    from hydra import compose, initialize_config_dir
    from output_normalize import normalize_output_to_tensordict, require_output_type
    from utils import recursive_to_device

    import physicsnemo.experimental.models.isla.geometry as geom
    from physicsnemo import datapipes  # noqa: F401
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.mesh import Mesh
    from physicsnemo.utils import load_checkpoint

    ov = [f"model={a.model}", *[o for o in a.extra.split() if "training." not in o], f"dataset={a.dataset}", f"run_id={a.run_id}",
          "infer_split=val", f"sampling_resolution={a.cells}", "precision=float32", f"checkpoint_dir={a.task}/runs",
          "++model.compile_blocks=false"]
    with initialize_config_dir(config_dir=str(Path.cwd() / "conf"), version_base=None):
        cfg = compose(config_name="infer", overrides=ov)
    DistributedManager.initialize()
    dev = DistributedManager().device
    _, vl, normalizer, info = build_dataloaders(cfg)
    tc, ot = info["targets"], require_output_type(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(dev)
    ck = str(Path(a.task) / "runs" / a.run_id / "checkpoints")
    ep = load_checkpoint(path=ck, models=model, device=dev)
    sp = Path(ck) / "norm_stats.pt"
    if normalizer is not None and sp.exists():
        st = torch.load(sp, weights_only=True)
        for live in {id(n): n for n in (normalizer, find_normalizer([vl.dataset])) if n is not None}.values():
            live.stats = st
    model.eval().double()
    print(f"epoch {ep} query_independent={getattr(model, 'query_independent', None)} dim={model.dim}", flush=True)
    idx = list(vl.sampler)[0]
    b = recursive_to_device(vl.collate_fn([vl.dataset[idx]]), dev)
    fk, pp = b["forward_kwargs"], b["forward_kwargs"]["prediction_points"]
    vq = [n for n, s in model.query_schema.items() if s.rank == 1]
    vg = [n for n, s in model.global_schema.items() if s.rank == 1]
    d = pp.points.shape[-1]

    def build(R, t):
        R, t = R.to(dev, torch.float64), t.to(dev, torch.float64)
        bm = {k: Mesh(points=m.points.double() @ R.T + t, cells=m.cells, cell_data=m.cell_data.to(torch.float64)) for k, m in fk["boundary_meshes"].items()}
        pd = pp.point_data.to(torch.float64).clone()
        for n in vq:
            pd[n] = pd[n] @ R.T
        q = Mesh(points=pp.points.double() @ R.T + t, point_data=pd)
        gd = fk["global_data"].to(torch.float64).clone()
        for n in vg:
            gd[n] = gd[n] @ R.T
        return {**fk, "boundary_meshes": bm, "global_data": gd, "prediction_points": q}

    def run(kw):
        with torch.no_grad():
            return normalize_output_to_tensordict(model(**kw), tc, ot)

    def nn_index(kw):
        bp = torch.cat([m.cell_centroids for m in kw["boundary_meshes"].values()])
        return geom.knn(bp.contiguous(), kw["prediction_points"].points.contiguous(), 1)[0][:, 0]

    gen = torch.Generator().manual_seed(7)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    L = 0.5 * float(pp.points.double().std(0).norm())
    t = torch.randn(d, generator=gen, dtype=torch.float64) * 3.0 * L
    eye, zero = torch.eye(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64)
    cases = {"rotation": (q, zero), "translation": (eye, t), "both": (q, t)}
    library = geom.knn
    for label, knn_impl in (("exact float64 cdist", "cdist"), ("library knn (cuML on CUDA)", None)):
        if knn_impl is None:
            geom.knn = library
        if knn_impl == "cdist":
            def exact(points, queries, k):
                ii, vv = [], []
                for qc in queries.double().split(8192):
                    v, i = torch.cdist(qc, points.double()).topk(k, largest=False)
                    ii.append(i)
                    vv.append(v)
                return torch.cat(ii), torch.cat(vv)
            geom.knn = exact
        ref_kw = build(eye, zero)
        ref = run(ref_kw)
        i_ref = nn_index(ref_kw)
        for name, (R, tt) in cases.items():
            kw = build(R, tt)
            out = run(kw)
            back = out.clone()
            back["velocity"] = out["velocity"] @ R.to(dev, torch.float64)
            i_p = nn_index(kw)
            flips = int((i_p != i_ref).sum())
            line = [f"[{label}] {name}: nn flips {flips}/{i_ref.numel()}"]
            for f in ("pressure", "velocity", "nut"):
                r = (back[f] - ref[f]).reshape(ref[f].shape[0], -1).norm(dim=-1)
                rms = ref[f].reshape(ref[f].shape[0], -1).norm(dim=-1).square().mean().sqrt()
                rel = float((back[f] - ref[f]).norm() / ref[f].norm())
                big = int((r > 1e-6 * rms).sum())
                on_flip = float(r[i_p != i_ref].square().sum() / r.square().sum().clamp_min(1e-300)) if flips else 0.0
                line.append(f"{f}: rel {rel:.2e}, pts>1e-6 {big}, err share on flipped pts {on_flip:.3f}")
            print("; ".join(line), flush=True)
            if knn_impl == "cdist":
                ### Where does the residual live: the built-in geometry of the points that move.
                def built_in(kw2):
                    bp = torch.cat([m.cell_centroids for m in kw2["boundary_meshes"].values()])[None]
                    bn = torch.cat([m.cell_normals for m in kw2["boundary_meshes"].values()])[None]
                    return geom.nearest_boundary_geometry(kw2["prediction_points"].points[None], bp, bn)
                d0, u0 = built_in(ref_kw)
                d1, u1 = built_in(kw)
                u1b = u1[0] @ R.to(dev, torch.float64)
                r = (back["pressure"] - ref["pressure"]).abs()
                rms = ref["pressure"].square().mean().sqrt()
                mov = (r > 1e-6 * rms).nonzero(as_tuple=True)[0]
                sflip = (torch.sign(d0[0]) != torch.sign(d1[0]))
                print(f"    moving pts {mov.numel()}: built-in d sign flips overall {int(sflip.sum())}, among moving {int(sflip[mov].sum())}; "
                      f"max |u - R^T u'| overall {float((u0[0] - u1b).norm(dim=-1).max()):.2e}, max |d - d'| {float((d0[0] - d1[0]).abs().max()):.2e}", flush=True)
                if mov.numel():
                    sdf = kw["prediction_points"].point_data["sdf"].reshape(-1)[mov]
                    print(f"    moving pts: |stored sdf| quantiles {[round(float(x), 6) for x in sdf.abs().quantile(torch.tensor([0., .5, 1.], device=dev, dtype=sdf.dtype))]}, "
                          f"|built-in d| quantiles {[float('%.3g' % x) for x in d0[0][mov].abs().quantile(torch.tensor([0., .5, 1.], device=dev, dtype=d0.dtype))]}; "
                          f"top-5 |dp|/rms {[float('%.3g' % x) for x in (r[mov] / rms).topk(min(5, mov.numel())).values]}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f} s", flush=True)
