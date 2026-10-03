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

"""Full-field readout: every interior point of each held-out case predicted four ways and scored together.

For a dataset whose stored point order is spatially coherent (AirfRANS), the protocol's request (one contiguous
cyclic block) is a spatial chunk of the domain, so the native score covers part of the field under the request law the
model was trained on. This readout scores the *whole* field, float32, per case:

- ``tiles``: the interacting model on contiguous cyclic blocks of the training count at four offsets; every point is
  predicted in the block whose centre is nearest to it in the stored order (the training law, request-dependent);
- ``random``: the interacting model on random requests of the training count (a seeded partition of the points, the
  last request padded with random others);
- ``support``: the interacting model with a seeded random support of the training count, every point read passively
  (query-independent);
- ``passive``: the query-independent (``query_independent=True``) checkpoint of the same seed, every point.

usage: python full_field_probe.py --run-int R1 --run-pas R2 --model M --dataset ALL --cells N --request N --extra ...
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
    for k in ("run-int", "run-pas", "model", "dataset", "task", "out"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--cells", type=int, required=True)
    p.add_argument("--request", type=int, required=True)
    p.add_argument("--extra", default="")
    p.add_argument("--max-cases", type=int, default=0)
    a = p.parse_args()
    for k, v in dict(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", MASTER_ADDR="127.0.0.1", MASTER_PORT=str(30000 + os.getpid() % 20000)).items():
        os.environ.setdefault(k, v)
    import hydra
    from datasets import build_dataloaders, find_normalizer
    from hydra import compose, initialize_config_dir
    from metrics import MetricCalculator, resolve_metrics
    from output_normalize import normalize_output_to_tensordict, require_output_type
    from tensordict import TensorDict
    from utils import get_autocast_context, recursive_to_device

    from physicsnemo import datapipes  # noqa: F401
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint

    runs = f"{a.task}/runs"

    def compose_cfg(run, qi):
        ov = [f"model={a.model}", *[o for o in a.extra.split() if "training." not in o], f"dataset={a.dataset}", f"run_id={run}",
              "infer_split=val", f"sampling_resolution={a.cells}", "precision=float32", f"checkpoint_dir={runs}",
              "++model.compile_blocks=false"] + (["++model.query_independent=true"] if qi else [])
        with initialize_config_dir(config_dir=str(Path.cwd() / "conf"), version_base=None):
            return compose(config_name="infer", overrides=ov)

    DistributedManager.initialize()
    dev = DistributedManager().device
    cfg = compose_cfg(a.run_int, False)
    _, vl, normalizer, info = build_dataloaders(cfg)
    tc, ot = info["targets"], require_output_type(cfg)
    models = {}
    for tag, run, qi in (("int", a.run_int, False), ("pas", a.run_pas, True)):
        c = compose_cfg(run, qi)
        m = hydra.utils.instantiate(c.model, _convert_="partial").to(dev)
        ck = str(Path(runs) / run / "checkpoints")
        ep = load_checkpoint(path=ck, models=m, device=dev)
        models[tag] = m.eval()
        log(f"{tag}: {run} epoch {ep} query_independent={m.query_independent}")
    ### one dataset config, so one set of normalization statistics; the interacting checkpoint's persisted copy is used
    st = torch.load(Path(runs) / a.run_int / "checkpoints" / "norm_stats.pt", weights_only=True)
    for live in {id(n): n for n in (normalizer, find_normalizer([vl.dataset])) if n is not None}.values():
        live.stats = st
    calc = MetricCalculator(target_config=tc, metrics=resolve_metrics(cfg))

    def predict(m, fk, pts, support=None):
        with torch.no_grad(), get_autocast_context(cfg.precision):
            out = m(**{**fk, "prediction_points": pts, "support_points": support})
        return normalize_output_to_tensordict(out, tc, ot).float()

    def scores(pred, tgt):
        mm = calc(pred, tgt, None)
        return {f: float(mm[f"{f}_l2"]) for f in FIELDS}

    def assemble(n, parts):
        """Scatter (indices, prediction) parts into one TensorDict over n points."""
        idx0, pr0 = parts[0]
        out = TensorDict({k: pr0[k].new_zeros((n, *pr0[k].shape[1:])) for k in pr0.keys()}, batch_size=[n], device=dev)
        for idx, pr in parts:
            for k in pr.keys():
                out[k][idx] = pr[k]
        return out

    R = a.request
    cases = []
    idxs = list(vl.sampler)
    if a.max_cases > 0:
        idxs = idxs[: a.max_cases]
    for ci, idx in enumerate(idxs):
        t0 = time.time()
        b = recursive_to_device(vl.collate_fn([vl.dataset[idx]]), dev)
        fk, tgt = b["forward_kwargs"], b["targets"].float()
        pp = fk["prediction_points"]
        n = pp.n_points
        ar = torch.arange(n, device=dev)
        ### tiles: four cyclic blocks of R; each point predicted in the block whose centre is nearest in the stored order
        offs = [int(round(k * n / 4)) for k in range(4)]
        centre = torch.tensor([(o + R / 2) % n for o in offs], device=dev, dtype=torch.float64)
        dist = (ar[:, None].double() - centre[None]).abs()
        dist = torch.minimum(dist, n - dist)
        owner = dist.argmin(dim=1)
        parts = []
        for k, o in enumerate(offs):
            blk = (o + torch.arange(R, device=dev)) % n
            pr = predict(models["int"], fk, pp.slice_points(blk))
            mine = owner[blk] == k
            parts.append((blk[mine], pr[mine]))
        tiles = assemble(n, parts)
        ### random requests of R (seeded partition; the last one padded with random other points)
        gen = torch.Generator().manual_seed(8000 + ci)
        perm = torch.randperm(n, generator=gen).to(dev)
        parts = []
        for start in range(0, n, R):
            req = perm[start : start + R]
            k = req.numel()
            if k < R:
                req = torch.cat([req, perm[torch.randperm(start, generator=gen)[: R - k].to(dev)]])
            parts.append((req[:k], predict(models["int"], fk, pp.slice_points(req))[:k]))
        rand = assemble(n, parts)
        ### support: a seeded random support of R; every point read passively (in chunks of R)
        sup_idx = torch.randperm(n, generator=gen)[:R].to(dev)
        support = pp.slice_points(sup_idx)
        sup = assemble(n, [(c, predict(models["int"], fk, pp.slice_points(c), support=support)) for c in ar.split(R)])
        ### the query-independent checkpoint, every point
        pas = assemble(n, [(c, predict(models["pas"], fk, pp.slice_points(c))) for c in ar.split(R)])
        rec = {"case": ci, "n": n, "tiles": scores(tiles, tgt), "random": scores(rand, tgt), "support": scores(sup, tgt),
               "passive": scores(pas, tgt), "seconds": round(time.time() - t0, 2)}
        cases.append(rec)
        log(f"case {ci + 1}/{len(idxs)} n={n} | tiles " + " ".join(f"{f}={rec['tiles'][f]:.4f}" for f in FIELDS)
            + " | random/tiles " + " ".join(f"{f}={rec['random'][f] / rec['tiles'][f]:.3f}" for f in FIELDS)
            + " | support/tiles " + " ".join(f"{f}={rec['support'][f] / rec['tiles'][f]:.3f}" for f in FIELDS)
            + " | passive/tiles " + " ".join(f"{f}={rec['passive'][f] / rec['tiles'][f]:.3f}" for f in FIELDS) + f" ({rec['seconds']} s)")
    summ = {k: {f: sum(c[k][f] for c in cases) / len(cases) for f in FIELDS} for k in ("tiles", "random", "support", "passive")}
    summ["over_tiles"] = {k: {f: summ[k][f] / summ["tiles"][f] for f in FIELDS} for k in ("random", "support", "passive")}
    out = {"run_int": a.run_int, "run_pas": a.run_pas, "dataset": a.dataset, "request": R, "n_cases": len(cases), "summary": summ, "cases": cases}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".tmp").write_text(json.dumps(out, indent=1))
    os.replace(a.out + ".tmp", a.out)
    log(f"wrote {a.out}: " + json.dumps(summ["over_tiles"]))


if __name__ == "__main__":
    main()
