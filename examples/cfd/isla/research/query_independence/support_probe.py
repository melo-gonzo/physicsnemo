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


def main():
    p = argparse.ArgumentParser()
    for k in ("model", "dataset", "run-id", "task", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--request", type=int, required=True, help="the training request count")
    p.add_argument("--cells", type=int, required=True, help="boundary cells (sampling_resolution)")
    p.add_argument("--extra", default="")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--runs-dir", default=None, help="checkpoint root (default <task>/runs)")
    a = p.parse_args()
    for k, v in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(27000 + os.getpid() % 20000)).items():
        os.environ.setdefault(k, v)
    import hydra
    from datasets import build_dataloaders, find_normalizer
    from hydra import compose, initialize_config_dir
    from metrics import MetricCalculator, resolve_metrics
    from output_normalize import normalize_output_to_tensordict, require_output_type
    from utils import get_autocast_context, recursive_to_device

    from physicsnemo import datapipes  # noqa: F401
    from physicsnemo.distributed import DistributedManager
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

    def predict(fk, pts, support=None):
        with torch.no_grad(), get_autocast_context(cfg.precision):
            out = model(**{**fk, "prediction_points": pts, "support_points": support})
        return normalize_output_to_tensordict(out, tc, ot).float()

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
        rec = {"case": ci, "sample_index": int(idx), "n_interior": n, "n_support": a.request, "n_eval": nb,
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
            + f" | sentinel maxdiff p={rec['sentinel_alone_maxdiff']['pressure']:.1e} | swap change p={rec['support_swap']['rel_change']['pressure']:.1e} ({rec['seconds']} s)")
    mean = lambda key: {f: sum(c[key][f] for c in cases) / len(cases) for f in FIELDS}  # noqa: E731
    summ = {k: mean(k) for k in ("interacting", "support", "no_support")}
    summ["support_over_interacting"] = {f: summ["support"][f] / summ["interacting"][f] for f in FIELDS}
    summ["no_support_over_interacting"] = {f: summ["no_support"][f] / summ["interacting"][f] for f in FIELDS}
    summ["sentinel_alone_maxdiff_max"] = {f: max(c["sentinel_alone_maxdiff"][f] for c in cases) for f in FIELDS}
    summ["support_swap_rel_change_mean"] = {f: sum(c["support_swap"]["rel_change"][f] for c in cases) / len(cases) for f in FIELDS}
    summ["support_swap_score_ratio"] = {f: sum(c["support_swap"]["swapped"][f] for c in cases) / sum(c["support_swap"]["support"][f] for c in cases) for f in FIELDS}
    out = {"run_id": a.run_id, "model": a.model, "dataset": a.dataset, "epoch": ep, "query_logw": logw, "request": a.request,
           "n_cases": len(cases), "summary": summ, "cases": cases, "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".tmp").write_text(json.dumps(out, indent=1))
    os.replace(a.out + ".tmp", a.out)
    log(f"wrote {a.out}: support/interacting {json.dumps({f: round(v, 4) for f, v in summ['support_over_interacting'].items()})}, "
        f"no_support/interacting {json.dumps({f: round(v, 4) for f, v in summ['no_support_over_interacting'].items()})}")


if __name__ == "__main__":
    main()
