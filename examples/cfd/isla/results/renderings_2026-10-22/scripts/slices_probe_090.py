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

"""ROUTING-090 (RENDER-090 lane, 2026-10-22): the trained routing of an ISLA 0.9.0 checkpoint on one validation case, in the
schema of renderings_2026-10-18/scripts/slices_probe_a2.py, ported to 0.9.0's forward (boundary conditions by name, the
full-resolution boundary drawn into tokens inside the model, exact query geometry).

Instead of composing the model itself, the probe runs the scorer's own ``src/infer.py`` through dump_wrap.py (the case
restricted and the scorer's reads replayed, the npz of truth and predictions written), so the config, the norm-stats
restore, the model build (output stats, dataset constants), the checkpoint and the forward are the scorer's, and the run
writes the case's own metrics.jsonl row, which the renderers compare with the scorer's record. The only difference from the
scorer's command is model.compile_blocks=false (and compile=false): the blocks run as plain modules so their hooks fire.
Read-only wrappers:
  * blocks._softmax_over_points records its input (logits + log_w) and output (a_is, normalized over tokens). Each
    _SliceBlock calls it twice; the SECOND call is the refined routing (the first builds only the anchors).
  * model.rms_pairwise_distance records the length unit L (the gauge of the full-resolution boundary, computed once per
    forward), so token positions are r * L in the model's input coordinates (the dataset's, after its CenterMesh and
    non-dimensionalization where it has them).
  * ISLA._forward_tokens records the number of boundary tokens (for an interior task, the cells drawn inside the model).
Forward pre-hooks on every block record (log_w, r, n_hat) and delimit the block's softmax calls.

Per layer, as slices_probe_a2.py: the soft assignment pi_is = softmax over slices of the refined logits (the recorded input
minus log_w), the dominant slice, its weight, slice masses q and qw, anchors z_s = sum_i a_is r_i (times L) and anchor
normals m_s; for three layers the 8 slices with the largest argmax territories with their pi and a_is / w_i on the stored
tokens. Arrays: points, n_boundary, measure, keep, dominant, pi_max, q, qw, anchors, anchor_normals, top_layers, top_slices,
top_pi, top_a_density, gauge, sample_id; json: per-layer n_eff, H_slices, H_points, anchor_spread, pi_max_mean.
usage (env, plus the scorer's infer.py command line as dump_wrap.py takes it):
  FIG_SAMPLES=00079 FIG_REPLAY=1 FIG_OUT=<dir> PROBE_TAG=<run>_s00079 PROBE_KEEP=200000 PYTHONPATH=<code>:<recipe>/src \
      python slices_probe_090.py --config-path <recipe>/conf --config-name infer model=... <extra> model.compile_blocks=false \
      compile=false dataset=... run_id=... infer_split=val +save_predictions=true sampling_resolution=N precision=float32 \
      checkpoint_dir=... output_dir=<dir>
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dump_wrap  # noqa: E402,F401  (patches infer: case restriction, replay, npz output)
import infer  # noqa: E402
import physicsnemo.experimental.models.isla.blocks as isla_blocks  # noqa: E402
import physicsnemo.experimental.models.isla.model as isla_model  # noqa: E402

OUT = Path(os.environ["FIG_OUT"])
TAG = os.environ["PROBE_TAG"]
KEEP = int(os.environ.get("PROBE_KEEP", "60000"))
TOP = int(os.environ.get("PROBE_TOP", "8"))
LAYERS = [int(s) for s in os.environ.get("PROBE_LAYERS", "0,5,11").split(",")]

REC, BLK, GAUGE, NB, STATE = [], [], [], [], {}
_orig_softmax = isla_blocks._softmax_over_points
_orig_rms = isla_model.rms_pairwise_distance
_orig_tokens = isla_model.ISLA._forward_tokens


def _rec_softmax(x):
    a = _orig_softmax(x)
    REC.append((x.detach(), a.detach()))
    return a


def _rec_rms(points, *a, **k):
    g = _orig_rms(points, *a, **k)
    GAUGE.append((g.detach(), int(points.shape[1])))
    return g


def _rec_tokens(self, boundary, query, *a, **k):
    NB.append(
        (
            int(boundary["points"].shape[1]),
            0 if query is None else int(query["points"].shape[1]),
        )
    )
    return _orig_tokens(self, boundary, query, *a, **k)


isla_blocks._softmax_over_points = _rec_softmax
isla_model.rms_pairwise_distance = _rec_rms
isla_model.ISLA._forward_tokens = _rec_tokens


def _pre(mod, a):
    BLK.append(
        {
            "start": len(REC),
            "log_w": a[1].detach(),
            "r": a[2].detach(),
            "n_hat": a[3].detach(),
        }
    )


def _post(mod, a, o):
    BLK[-1]["end"] = len(REC)


_orig_compile = infer.compile_model


def compile_model(model, cfg):
    """The scorer's compile_model, then the probe's hooks on the plain blocks."""
    model = _orig_compile(model, cfg)
    assert not model.compile_blocks, "the probe needs model.compile_blocks=false"
    for blk in model.blocks:
        blk.register_forward_pre_hook(_pre)
        blk.register_forward_hook(_post)
    STATE["model"] = model
    STATE["cfg"] = cfg
    return model


infer.compile_model = compile_model


def entropy(p, dim):
    """Shannon entropy of p along dim."""
    return -(p * (p + 1e-30).log()).sum(dim)


def reduce():
    """Per-layer routing arrays and statistics from the recorded calls; write the npz and json."""
    model, cfg = STATE["model"], STATE["cfg"]
    assert len(NB) == 1, NB  # one case, one forward
    n_boundary, n_query = NB[0]
    gauge = float(GAUGE[0][0].reshape(-1)[0])
    L = len(BLK)
    assert L == len(model.blocks), (L, len(model.blocks))
    assert all(b["end"] - b["start"] == 2 for b in BLK), [
        (b["start"], b["end"]) for b in BLK
    ]
    log_w = BLK[0]["log_w"][0, :, 0].float()
    w = log_w.exp()
    r0 = BLK[0]["r"][0].float()
    pts = r0 * gauge
    n_tok = pts.shape[0]
    assert n_tok == n_boundary + n_query, (n_tok, n_boundary, n_query)
    keep = torch.arange(n_tok, device=pts.device)
    if n_tok > KEEP:
        g = torch.Generator(device="cpu").manual_seed(0)
        keep = torch.randperm(n_tok, generator=g)[:KEEP].sort().values.to(pts.device)
    active = torch.isfinite(
        log_w
    )  # passive (query-independent) tokens carry log_w = -inf
    dominant, pi_max, q, qw, anchors, anchor_n, stats = [], [], [], [], [], [], []
    top_slices, top_pi, top_ad = [], [], []
    for li, b in enumerate(BLK):
        x, a = REC[b["end"] - 1]
        x = x[0].float()
        a = a[0].float()
        lw = torch.where(active, log_w, torch.zeros_like(log_w))
        pi = torch.softmax(x - lw[:, None], dim=-1)
        qq = pi[active].mean(0)
        qwq = (w[active, None] * pi[active]).sum(0) / w[active].sum()
        r = b["r"][0].float()
        z = a.T @ r
        m = a.T @ b["n_hat"][0].float()
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        rms_r = (r[active] - r[active].mean(0)).norm(dim=-1).pow(2).mean().sqrt()
        stats.append(
            dict(
                n_eff=torch.exp(entropy(qq, -1)).item(),
                n_eff_w=torch.exp(entropy(qwq, -1)).item(),
                H_slices=(entropy(pi[active], -1) / math.log(pi.shape[1]))
                .mean()
                .item(),
                H_points=(entropy(a[active], 0) / math.log(int(active.sum())))
                .mean()
                .item(),
                anchor_spread=(
                    (z - z.mean(0)).norm(dim=-1).pow(2).mean().sqrt() / rms_r
                ).item(),
                pi_max_mean=pi[active].max(-1).values.mean().item(),
            )
        )
        dominant.append(pi[keep].argmax(-1).to(torch.int16).cpu().numpy())
        pi_max.append(pi[keep].max(-1).values.half().cpu().numpy())
        q.append(qq.cpu().numpy())
        qw.append(qwq.cpu().numpy())
        anchors.append((z * gauge).cpu().numpy())
        anchor_n.append(m.cpu().numpy())
        if li in LAYERS:
            counts = torch.bincount(pi[active].argmax(-1), minlength=pi.shape[1])
            ts = torch.argsort(counts, descending=True)[:TOP]
            top_slices.append(ts.cpu().numpy())
            top_pi.append(pi[keep][:, ts].half().cpu().numpy())
            top_ad.append(
                (a[keep][:, ts] / w[keep, None].clamp_min(1e-30)).half().cpu().numpy()
            )
    sids = [p.stem for p in OUT.glob("*.npz") if not p.stem.startswith(TAG)]
    assert len(sids) == 1, sids
    arrays = dict(
        points=pts.cpu().numpy().astype(np.float32),
        n_boundary=np.array(n_boundary),
        measure=w.cpu().numpy().astype(np.float32),
        keep=keep.cpu().numpy().astype(np.int64),
        dominant=np.stack(dominant),
        pi_max=np.stack(pi_max),
        q=np.stack(q),
        qw=np.stack(qw),
        anchors=np.stack(anchors),
        anchor_normals=np.stack(anchor_n),
        top_layers=np.array(LAYERS),
        top_slices=np.stack(top_slices),
        top_pi=np.stack(top_pi),
        top_a_density=np.stack(top_ad),
        gauge=np.array(gauge),
        sample_id=np.array(sids[0]),
    )
    np.savez_compressed(OUT / f"{TAG}.npz", **arrays)
    run_dir = Path(cfg.output_dir) / str(cfg.run_id)
    rows = [json.loads(ln) for ln in open(run_dir / "metrics.jsonl")]
    conf = [r for r in rows if r.get("phase") == "config"][0]
    step = [r for r in rows if r.get("phase") == "infer_step"]
    meta = dict(
        run=str(cfg.run_id),
        epoch=int(conf["epoch"]),
        checkpoint=conf["checkpoint"],
        model_yaml=str(cfg.model._target_),
        dataset_yaml=str(cfg.dataset),
        overrides=sys.argv[1:],
        sampling_resolution=int(cfg.sampling_resolution),
        precision=str(cfg.precision),
        sample_id=sids[0],
        probe_metrics=step[-1] if step else None,
        geo_kernel=str(getattr(model.blocks[0], "geo_kernel", "")),
        query_independent=bool(model.query_independent),
        n_tokens=int(n_tok),
        n_boundary=int(n_boundary),
        n_query=int(n_query),
        n_active=int(active.sum()),
        n_stored=int(keep.numel()),
        n_layers=int(L),
        n_slices=int(REC[0][1].shape[-1]),
        gauge=gauge,
        layers=stats,
        top_layers=LAYERS,
        top_slices=[t.tolist() for t in top_slices],
    )
    (OUT / f"{TAG}.json").write_text(json.dumps(meta, indent=2))
    print(
        TAG,
        "epoch",
        meta["epoch"],
        "tokens",
        n_tok,
        "boundary",
        n_boundary,
        "gauge",
        gauge,
        "sample",
        sids[0],
        flush=True,
    )
    for i, st in enumerate(stats):
        print(
            f"  layer {i:2d}: " + "  ".join(f"{kk} {v:.3f}" for kk, v in st.items()),
            flush=True,
        )
    print("PROBE-DONE", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        infer.main()
    reduce()
