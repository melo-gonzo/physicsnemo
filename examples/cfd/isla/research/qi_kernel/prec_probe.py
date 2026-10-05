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

"""Which part of a trained kernel-read model depends on its bf16 training arithmetic (no training).

The bf16-trained kernel-read lanes score much worse in float32 (the protocol) than in bf16. This loads one trained
checkpoint and predicts the first ``--cases`` held-out cases with the whole model in float32 or under bf16 autocast,
and with one part switched to the other arithmetic: the kernel basis MLP, the per-block kernel reads, or both. Each
variant's relative L2 per field (case mean) shows which part carries the dependence. It also prints, per kernel-read
block, the RMS of the read's update against the RMS of the point state it is added to, in float32 and in bf16, and
the relative difference of the two updates.

Run from the recipe directory with ``PYTHONPATH=<code>:<recipe>/src`` (like ``qi_probe.py``).
"""

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import torch

FIELDS = ("pressure", "velocity", "nut")


def log(msg):
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    """Load the checkpoint, run the variants, print and write the table."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--cells", type=int, required=True)
    p.add_argument("--extra", default="")
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--cases", type=int, default=6)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--groups",
        type=int,
        default=0,
        help="also switch the slice blocks (before / with the reads), the output head and the token embeddings",
    )
    a = p.parse_args()
    for k, v in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
    ):
        os.environ.setdefault(k, v)
    os.environ.setdefault("MASTER_PORT", str(24000 + os.getpid() % 20000))
    import hydra
    from datasets import build_dataloaders, find_normalizer
    from hydra import compose, initialize_config_dir
    from metrics import MetricCalculator, resolve_metrics
    from output_normalize import normalize_output_to_tensordict, require_output_type
    from utils import recursive_to_device

    from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...}
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint

    overrides = [
        f"model={a.model}",
        *a.extra.split(),
        f"dataset={a.dataset}",
        f"run_id={a.run_id}",
        "infer_split=val",
        f"sampling_resolution={a.cells}",
        "precision=float32",
        f"checkpoint_dir={a.ckpt_root}",
        "+save_predictions=false",
        "output_dir=/tmp/prec_probe",
        "++model.compile_blocks=false",
    ]
    overrides = [o for o in overrides if "training." not in o]
    with initialize_config_dir(config_dir=str(Path.cwd() / "conf"), version_base=None):
        cfg = compose(config_name="infer", overrides=overrides)
    DistributedManager.initialize()
    device = DistributedManager().device
    _, val_loader, normalizer, info = build_dataloaders(cfg)
    target_config = info["targets"]
    output_type = require_output_type(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    ckpt = str(Path(a.ckpt_root) / a.run_id / "checkpoints")
    epoch = load_checkpoint(path=ckpt, models=model, device=device)
    stats_path = Path(ckpt) / "norm_stats.pt"
    if normalizer is not None and stats_path.exists():
        saved = torch.load(stats_path, weights_only=True)
        for live in {
            id(n): n
            for n in (normalizer, find_normalizer([val_loader.dataset]))
            if n is not None
        }.values():
            live.stats = saved
    model.eval()
    log(f"loaded {a.run_id} epoch {epoch}")
    calc = MetricCalculator(target_config=target_config, metrics=resolve_metrics(cfg))

    bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    fp32 = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731

    def in_ctx(fwd, ctx, cast_in, cast_out):
        """``fwd`` run under ``ctx`` with float tensor inputs cast to ``cast_in`` and outputs to ``cast_out``."""

        def wrapped(*args, **kw):
            dt_out = next(
                (x.dtype for x in args if torch.is_tensor(x) and x.is_floating_point()),
                None,
            )
            args = [
                x.to(cast_in) if torch.is_tensor(x) and x.is_floating_point() else x
                for x in args
            ]
            with ctx():
                y = fwd(*args, **kw)
            want = cast_out or dt_out
            return y.to(want) if torch.is_tensor(y) and want is not None else y

        return wrapped

    basis_fwd = model.kernel_basis.forward
    read_fwds = [r.forward for r in model.kernel_reads]
    deltas = {}

    def set_variant(basis, reads):
        """basis / reads: None (follow the outer arithmetic), 'fp32' or 'bf16'."""
        model.kernel_basis.forward = basis_fwd
        if basis == "fp32":
            model.kernel_basis.forward = in_ctx(
                basis_fwd, fp32, torch.float32, torch.float32
            )
        elif basis == "bf16":
            model.kernel_basis.forward = in_ctx(basis_fwd, bf16, torch.float32, None)
        for i, (r, f) in enumerate(zip(model.kernel_reads, read_fwds)):
            g = f
            if reads == "fp32":

                def g(h_b, h_q, pairs, f=f):
                    plan, basis_t, log_m = pairs
                    with fp32():
                        y = f(h_b.float(), h_q.float(), (plan, basis_t.float(), log_m))
                    return y.to(h_q.dtype)

            elif reads == "bf16":

                def g(h_b, h_q, pairs, f=f):
                    with bf16():
                        y = f(h_b, h_q, pairs)
                    return y.to(h_q.dtype)

            def record(h_b, h_q, pairs, g=g, i=i):
                y = g(h_b, h_q, pairs)
                deltas.setdefault(i, []).append(
                    (
                        y.detach().float(),
                        float(y.float().square().mean().sqrt()),
                        float(h_q.float().square().mean().sqrt()),
                    )
                )
                return y

            r.forward = record

    first_read = len(model.blocks) - len(model.kernel_reads)
    groups = {
        "early blocks": list(model.blocks[:first_read]),
        "late blocks": list(model.blocks[first_read:]),
        "head": [model.norm_out, model.head],
        "embed": [model.embed, model.qt],
    }
    group_fwds = {id(m): m.forward for ms in groups.values() for m in ms}

    def set_groups(spec):
        """spec {group: 'fp32' | 'bf16'}: run that group's modules in that arithmetic."""
        for ms in groups.values():
            for m in ms:
                m.forward = group_fwds[id(m)]
        for g, mode in (spec or {}).items():
            for m in groups[g]:
                f = group_fwds[id(m)]
                m.forward = (
                    in_ctx(f, fp32, torch.float32, None)
                    if mode == "fp32"
                    else in_ctx(f, bf16, torch.float32, torch.float32)
                )

    variants = [
        ("fp32", "fp32", None, None),
        ("bf16", "bf16", None, None),
        ("bf16, basis fp32", "bf16", "fp32", None),
        ("bf16, reads fp32", "bf16", None, "fp32"),
        ("bf16, basis+reads fp32", "bf16", "fp32", "fp32"),
        ("fp32, basis bf16", "fp32", "bf16", None),
        ("fp32, reads bf16", "fp32", None, "bf16"),
        ("fp32, basis+reads bf16", "fp32", "bf16", "bf16"),
    ]
    variants = [(*v, None) for v in variants]
    if a.groups:
        for g in groups:
            variants.append((f"bf16, {g} fp32", "bf16", None, None, {g: "fp32"}))
            variants.append((f"fp32, {g} bf16", "fp32", None, None, {g: "bf16"}))
    dataset, sampler, collate = (
        val_loader.dataset,
        val_loader.sampler,
        val_loader.collate_fn,
    )
    res = {v[0]: {f: [] for f in FIELDS} for v in variants}
    upd = {}
    for ci, idx in enumerate(list(sampler)[: a.cases]):
        batch = recursive_to_device(collate([dataset[idx]]), device)
        fk, tgt, tm = (
            batch["forward_kwargs"],
            batch["targets"].float(),
            batch.get("target_measure"),
        )
        for name, outer, basis, reads, spec in variants:
            deltas.clear()
            set_variant(basis, reads)
            set_groups(spec)
            with (
                torch.no_grad(),
                bf16() if outer == "bf16" else contextlib.nullcontext(),
            ):
                out = model(**fk)
            pred = normalize_output_to_tensordict(out, target_config, output_type).to(
                torch.float32
            )
            m = calc(pred, tgt, tm)
            for f in FIELDS:
                res[name][f].append(float(m[f"{f}_l2"]))
            if name in ("fp32", "bf16"):
                upd[name] = {i: v[0] for i, v in deltas.items()}
                for i, v in deltas.items():
                    upd.setdefault(f"rms_{name}", {}).setdefault(i, []).append(v[0][1:])
            if name == "bf16":
                for i in upd["fp32"]:
                    d32, d16 = upd["fp32"][i][0], upd["bf16"][i][0]
                    rel = float((d16 - d32).norm() / d32.norm().clamp_min(1e-30))
                    upd.setdefault("rel", {}).setdefault(i, []).append(rel)
        log(
            f"case {ci}: "
            + "; ".join(f"{n} U={res[n]['velocity'][-1]:.4f}" for n, *_ in variants)
        )
    table = {n: {f: sum(v) / len(v) for f, v in r.items()} for n, r in res.items()}
    blocks = {
        i: {
            "rms_update_fp32": sum(x[0] for x in upd["rms_fp32"][i])
            / len(upd["rms_fp32"][i]),
            "rms_state_fp32": sum(x[1] for x in upd["rms_fp32"][i])
            / len(upd["rms_fp32"][i]),
            "rms_update_bf16": sum(x[0] for x in upd["rms_bf16"][i])
            / len(upd["rms_bf16"][i]),
            "rel_diff_update": sum(upd["rel"][i]) / len(upd["rel"][i]),
        }
        for i in upd["rel"]
    }
    print(f"{a.run_id} epoch {epoch}, {a.cases} cases, case-mean relative L2")
    for n, r in table.items():
        print(f"  {n:24s} " + "  ".join(f"{f} {r[f]:.4f}" for f in FIELDS))
    for i, b in blocks.items():
        print(
            f"  read {i}: rms update fp32 {b['rms_update_fp32']:.4g} bf16 {b['rms_update_bf16']:.4g}, "
            f"rms state {b['rms_state_fp32']:.4g}, |d_bf16 - d_fp32| / |d_fp32| {b['rel_diff_update']:.3g}"
        )
    Path(a.out).write_text(
        json.dumps(
            {"run": a.run_id, "epoch": epoch, "variants": table, "reads": blocks},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
