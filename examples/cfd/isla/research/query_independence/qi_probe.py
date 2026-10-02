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

"""QI readouts 2 to 5 (and a wall-distance anatomy) on one trained interior checkpoint.

Runs inside the unified external-aero recipe (``PYTHONPATH=<code>:<recipe>/src``, cwd the recipe) on one GPU,
float32, the held-out split read exactly as ``infer.py`` reads it (same Hydra composition, loaders, checkpoint and
persisted normalization). For every held-out case it predicts at the native request (the reader's interior sample)
and then:

- readout 2 (first ``--n-detail`` cases): 2,000 seeded-random sentinels requested alone and with 8,000, 48,000 and
  all remaining companions; sentinel relative L2 per request and max |prediction - native| / RMS(target);
- readout 3 (every case): the rear region (coordinate along the freestream direction in the rear quarter of the
  boundary's extent or beyond) requested alone against the same points inside the native request;
- readout 4 (first ``--n-detail`` cases): one random proper rotation and shift of the whole case, posed prediction
  mapped back against the native prediction;
- readout 5 (first ``--n-detail`` cases): the native request in 20 chunks against one call;
- anatomy (every case): per-field squared error and squared target summed in bands of |stored sdf| / L, L the
  boundary's measure-weighted RMS pairwise distance (the model's own length unit).

usage: python qi_probe.py --model M --dataset D --run-id R --cells N --extra "<hydra overrides>" --task T --out O.json
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

STAMP = lambda: time.strftime("%Y-%m-%dT%H:%M:%S")  # noqa: E731
FIELDS = ("pressure", "velocity", "nut")
BANDS = (0.0, 0.01, 0.1, 1.0, math.inf)  # |sdf| / L


def log(msg):
    print(f"[{STAMP()}] {msg}", flush=True)


def parse():
    p = argparse.ArgumentParser()
    for k in ("model", "dataset", "run-id", "task", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--cells", type=int, required=True)
    p.add_argument("--extra", default="")
    p.add_argument("--n-detail", type=int, default=4)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--fp64", type=int, default=1, help="float64 structural checks on the first case")
    return p.parse_args()


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
    a = parse()
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(24000 + os.getpid() % 20000))
    import hydra
    from datasets import build_dataloaders, find_normalizer
    from hydra import compose, initialize_config_dir
    from metrics import MetricCalculator, resolve_metrics
    from output_normalize import normalize_output_to_tensordict, require_output_type
    from utils import get_autocast_context, recursive_to_device

    import physicsnemo.experimental.models.isla.geometry as geom
    from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...}
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.experimental.models.isla.geometry import rms_pairwise_distance
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.calculus.measure import cell_measures
    from physicsnemo.utils import load_checkpoint

    t0 = time.time()
    recipe = Path.cwd()
    overrides = [f"model={a.model}", *a.extra.split(), f"dataset={a.dataset}", f"run_id={a.run_id}",
                 "infer_split=val", f"sampling_resolution={a.cells}", "precision=float32",
                 f"checkpoint_dir={a.task}/runs", "+save_predictions=false", f"output_dir={a.task}/eval/tmp",
                 "++model.compile_blocks=false"]  # eager: the probe's requests have many token counts
    overrides = [o for o in overrides if not o.startswith("+training.") and not o.startswith("training.")]
    log(f"run {a.run_id}: composing infer config with {overrides}")
    with initialize_config_dir(config_dir=str(recipe / "conf"), version_base=None):
        cfg = compose(config_name="infer", overrides=overrides)
    DistributedManager.initialize()
    device = DistributedManager().device
    _, val_loader, normalizer, info = build_dataloaders(cfg)
    target_config = info["targets"]
    output_type = require_output_type(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    ckpt = str(Path(a.task) / "runs" / a.run_id / "checkpoints")
    epoch = load_checkpoint(path=ckpt, models=model, device=device)
    if epoch <= 0:
        raise FileNotFoundError(f"no checkpoint restored from {ckpt}")
    stats_path = Path(ckpt) / "norm_stats.pt"
    if normalizer is not None and stats_path.exists():
        saved = torch.load(stats_path, weights_only=True)
        for live in {id(n): n for n in (normalizer, find_normalizer([val_loader.dataset])) if n is not None}.values():
            live.stats = saved
    model.eval()
    qi = bool(getattr(model, "query_independent", False))
    n_params = sum(p.numel() for p in model.parameters())
    log(f"loaded {type(model).__name__} epoch {epoch} ({n_params:,} params, query_independent={qi}) on {device}; "
        f"float32 matmul precision {torch.get_float32_matmul_precision()!r}, tf32 matmul {torch.backends.cuda.matmul.allow_tf32}, "
        f"tf32 cudnn {torch.backends.cudnn.allow_tf32}")
    calc = MetricCalculator(target_config=target_config, metrics=resolve_metrics(cfg))
    dataset, sampler, collate = val_loader.dataset, val_loader.sampler, val_loader.collate_fn
    autocast = lambda: get_autocast_context(cfg.precision)  # noqa: E731

    def predict(fk, pp, dtype=torch.float32):
        with torch.no_grad(), autocast():
            out = model(**{**fk, "prediction_points": pp})
        return normalize_output_to_tensordict(out, target_config, output_type).to(dtype)

    def scores(pred, tgt, tm):
        m = calc(pred, tgt, tm)
        return {f: float(m[f"{f}_l2"]) for f in FIELDS}

    def maxdiff(p, q, tgt):
        """Per field: max over points of |p - q| (vector norm) / RMS |target|."""
        out = {}
        for f in FIELDS:
            d = (p[f] - q[f]).reshape(p[f].shape[0], -1).norm(dim=-1)
            rms = tgt[f].reshape(tgt[f].shape[0], -1).norm(dim=-1).square().mean().sqrt()
            out[f] = float(d.max() / rms.clamp_min(1e-30))
        return out

    def rotation(dim, gen):
        q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=gen, dtype=torch.float64))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q

    library_knn = geom.knn
    vec_queries = [n for n, s in getattr(model, "query_schema", {}).items() if s.rank == 1]
    vec_globals = [n for n, s in model.global_schema.items() if s.rank == 1]

    def posed(fk, pp, R, t):
        """The whole case under x -> R x + t (true vectors rotate, scalars and measures ride along)."""
        Rf, tf = R.to(device, pp.points.dtype), t.to(device, pp.points.dtype)
        bm = {k: Mesh(points=m.points @ Rf.T + tf, cells=m.cells, cell_data=m.cell_data) for k, m in fk["boundary_meshes"].items()}
        pdata = pp.point_data.clone()
        for n in vec_queries:
            pdata[n] = pdata[n] @ Rf.T
        ppp = Mesh(points=pp.points @ Rf.T + tf, point_data=pdata)
        gd = fk["global_data"].clone()
        for n in vec_globals:
            gd[n] = gd[n] @ Rf.T
        return {**fk, "boundary_meshes": bm, "global_data": gd}, ppp

    cases, n_total = [], len(sampler)
    n_cases = n_total if a.max_cases <= 0 else min(n_total, a.max_cases)
    log(f"{n_cases} held-out cases; detail readouts on the first {a.n_detail}")
    for ci, idx in enumerate(list(sampler)[:n_cases]):
        tc = time.time()
        sample = dataset[idx]
        batch = recursive_to_device(collate([sample]), device)
        fk, tgt, tm = batch["forward_kwargs"], batch["targets"].float(), batch.get("target_measure")
        pp = fk["prediction_points"]
        n = pp.n_points
        native = predict(fk, pp)
        rec = {"case": ci, "sample_index": int(idx), "n_points": n, "native": scores(native, tgt, tm)}
        ### readout 3: the rear region requested alone
        bpts = torch.cat([m.cell_centroids for m in fk["boundary_meshes"].values()])
        g = fk["global_data"][vec_globals[0]].reshape(-1).to(bpts.dtype)
        g = g / g.norm()
        sb, sq = bpts @ g, pp.points @ g
        rear = (sq >= sb.max() - 0.25 * (sb.max() - sb.min())).nonzero(as_tuple=True)[0]
        if rear.numel():
            alone = predict(fk, pp.slice_points(rear))
            rec["rear"] = {"n": int(rear.numel()), "inside_native": scores(native[rear], tgt[rear], None if tm is None else tm[rear]),
                           "alone": scores(alone, tgt[rear], None if tm is None else tm[rear]),
                           "maxdiff": maxdiff(alone, native[rear], tgt[rear])}
        ### anatomy: error by wall-distance band
        meas = torch.cat([cell_measures(m) for m in fk["boundary_meshes"].values()])
        L = float(rms_pairwise_distance(bpts[None], meas[None]).reshape(()))
        sdf = pp.point_data["sdf"].reshape(-1).abs() / L
        band = {}
        for lo, hi in zip(BANDS[:-1], BANDS[1:]):
            sel = ((sdf >= lo) & (sdf < hi)).nonzero(as_tuple=True)[0]
            band[f"{lo}-{hi}"] = {"n": int(sel.numel()), **{f: [float((native[f][sel] - tgt[f][sel]).square().sum()), float(tgt[f][sel].square().sum())] for f in FIELDS}}
        rec["bands"], rec["L"] = band, L
        if ci < a.n_detail:
            ### readout 2: sentinels with growing companion sets
            gen = torch.Generator().manual_seed(1000 + ci)
            perm = torch.randperm(n, generator=gen).to(device)
            s = perm[:2000]
            sent = {"native": {"scores": scores(native[s], tgt[s], None if tm is None else tm[s]), "maxdiff": {f: 0.0 for f in FIELDS}}}
            for k in (0, 8000, 48000):
                req = torch.cat([s, perm[2000:2000 + k]])
                pr = predict(fk, pp.slice_points(req))[:2000]
                sent[str(2000 + k)] = {"scores": scores(pr, tgt[s], None if tm is None else tm[s]), "maxdiff": maxdiff(pr, native[s], tgt[s])}
            rec["sentinels"] = sent
            ### readout 5: chunked native request
            chunks = [predict(fk, pp.slice_points(c)) for c in torch.arange(n, device=device).tensor_split(20)]
            cat = torch.cat(chunks)
            rec["chunked"] = {"scores": scores(cat, tgt, tm), "maxdiff": maxdiff(cat, native, tgt)}
            ### readout 4: rotation and shift
            gen = torch.Generator().manual_seed(2000 + ci)
            R = rotation(pp.points.shape[-1], gen)
            t = torch.randn(pp.points.shape[-1], generator=gen, dtype=torch.float64) * 3.0 * L
            ### The posed and the reference case are built the same way (fresh meshes, geometry recomputed from the
            ### vertices); the loaded meshes may carry cached geometry from the converter, whose effect on the native
            ### prediction is reported separately as rebuild_vs_native.
            eye, zero = torch.eye(pp.points.shape[-1], dtype=torch.float64), torch.zeros(pp.points.shape[-1], dtype=torch.float64)
            geom.knn = exact_knn  # the posed readout compares like with like under an exact search
            fkr, ppr = posed(fk, pp, eye, zero)
            ref = predict(fkr, ppr)
            fkp, ppp = posed(fk, pp, R, t)
            pz = predict(fkp, ppp)
            Rf = R.to(device, torch.float32)
            back = pz.clone()
            back["velocity"] = pz["velocity"] @ Rf
            rec["posed"] = {f: float((back[f] - ref[f]).norm() / ref[f].norm()) for f in FIELDS}
            geom.knn = library_knn
            rec["rebuild_vs_native"] = {f: float((ref[f] - native[f]).norm() / native[f].norm()) for f in FIELDS}
            rec["cached_boundary_geometry"] = {k: sorted(".".join(x) if isinstance(x, tuple) else x for x in m._cache.keys(include_nested=True, leaves_only=True))
                                               for k, m in fk["boundary_meshes"].items()}
            if ci == 0 and a.fp64:
                ### The same readouts in float64 (structural: exact up to float64 roundoff), and the float32 floor
                ### (native float32 against native float64). The model is cast in place and cast back (exact).
                model.double()
                geom.knn = exact_knn
                fk64 = {**fk, "boundary_meshes": {k: m.to(torch.float64) for k, m in fk["boundary_meshes"].items()},
                        "global_data": fk["global_data"].to(torch.float64)}
                pp64, tgt64, d64 = pp.to(torch.float64), tgt.double(), torch.float64
                nat64 = predict(fk64, pp64, d64)
                sent64 = predict(fk64, pp64.slice_points(s), d64)
                cat64 = torch.cat([predict(fk64, pp64.slice_points(c), d64) for c in torch.arange(n, device=device).tensor_split(5)])
                fkr64, ppr64 = posed(fk64, pp64, eye, zero)
                ref64 = predict(fkr64, ppr64, d64)
                fkp64, ppp64 = posed(fk64, pp64, R, t)
                pz64 = predict(fkp64, ppp64, d64)
                back64 = pz64.clone()
                back64["velocity"] = pz64["velocity"] @ R.to(device, d64)
                rec["fp64"] = {"fp32_floor_maxdiff": maxdiff(native.double(), nat64, tgt64),
                               "fp32_floor_rel": {f: float((native[f].double() - nat64[f]).norm() / nat64[f].norm()) for f in FIELDS},
                               "sentinel_alone_maxdiff": maxdiff(sent64, nat64[s], tgt64[s]),
                               "chunked5_maxdiff": maxdiff(cat64, nat64, tgt64),
                               "posed_rel": {f: float((back64[f] - ref64[f]).norm() / ref64[f].norm()) for f in FIELDS},
                               "rebuild_vs_native_rel": {f: float((ref64[f] - nat64[f]).norm() / nat64[f].norm()) for f in FIELDS}}
                model.float()
                geom.knn = library_knn
                log("fp64 " + json.dumps(rec["fp64"]))
        rec["seconds"] = round(time.time() - tc, 2)
        cases.append(rec)
        log(f"case {ci + 1}/{n_cases} n={n} native " + " ".join(f"{f}={rec['native'][f]:.4f}" for f in FIELDS)
            + (f" | rear n={rec['rear']['n']} alone/inside p={rec['rear']['alone']['pressure'] / rec['rear']['inside_native']['pressure']:.3f}"
               f" u={rec['rear']['alone']['velocity'] / rec['rear']['inside_native']['velocity']:.3f}" if "rear" in rec else "")
            + (f" | sentinel maxdiff@2000 p={rec['sentinels']['2000']['maxdiff']['pressure']:.2e} posed p={rec['posed']['pressure']:.2e}" if "sentinels" in rec else "")
            + f" ({rec['seconds']} s)")
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    summary = {"native": {f: mean([c["native"][f] for c in cases]) for f in FIELDS},
               "rear_alone": {f: mean([c["rear"]["alone"][f] for c in cases if "rear" in c]) for f in FIELDS},
               "rear_inside_native": {f: mean([c["rear"]["inside_native"][f] for c in cases if "rear" in c]) for f in FIELDS},
               "rear_maxdiff_max": {f: max(c["rear"]["maxdiff"][f] for c in cases if "rear" in c) for f in FIELDS}}
    det = [c for c in cases if "sentinels" in c]
    if det:
        summary["sentinels"] = {k: {f: mean([c["sentinels"][k]["scores"][f] for c in det]) for f in FIELDS} for k in det[0]["sentinels"]}
        summary["sentinel_maxdiff_max"] = {k: {f: max(c["sentinels"][k]["maxdiff"][f] for c in det) for f in FIELDS} for k in det[0]["sentinels"]}
        summary["chunked_maxdiff_max"] = {f: max(c["chunked"]["maxdiff"][f] for c in det) for f in FIELDS}
        summary["posed_max"] = {f: max(c["posed"][f] for c in det) for f in FIELDS}
        if "fp64" in det[0]:
            summary["fp64_case0"] = det[0]["fp64"]
    pooled = {}
    for b in cases[0]["bands"]:
        pooled[b] = {"n": sum(c["bands"][b]["n"] for c in cases),
                     **{f: math.sqrt(sum(c["bands"][b][f][0] for c in cases) / max(sum(c["bands"][b][f][1] for c in cases), 1e-30)) for f in FIELDS}}
    summary["bands_pooled_rel_l2"] = pooled
    out = {"run_id": a.run_id, "model": a.model, "dataset": a.dataset, "extra": a.extra, "epoch": epoch, "query_independent": qi,
           "n_parameters": n_params, "cells": a.cells, "n_cases": len(cases), "summary": summary, "cases": cases,
           "seconds": round(time.time() - t0, 1), "written": STAMP()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, a.out)
    log(f"wrote {a.out} ({len(cases)} cases, {out['seconds']} s); native " + json.dumps(summary["native"]))


if __name__ == "__main__":
    main()
