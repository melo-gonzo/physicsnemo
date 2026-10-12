"""Checkpoint drift probe for the AeroZoo training transition (2026-10-11).

Every ISLA lane on AeroZoo trains normally to epoch 43-50 and then its training loss rises
three- to four-fold within a few epochs and stays there; GeoTransolver lanes never do. The
validation loss follows the training loss, so this is an optimization event, not
overfitting. This probe asks WHICH weights move at the transition, from the kept
checkpoints alone (odd epochs, no training):

  * per-tensor L2 norm and max-abs of every parameter at every requested epoch;
  * the relative change ||w(e2) - w(e1)|| / ||w(e1)|| of every tensor between consecutive
    checkpoints, ranked, so the pair that spans the transition can be compared with the
    pairs before it;
  * the scalar parameters of the ISLA blocks (gain, logw) as trajectories;
  * from the optimizer checkpoint, the Adam effective relative step
    lr * ||m / (sqrt(v) + eps)|| / ||w|| per tensor (m, v = exp_avg, exp_avg_sq), which
    says which tensors the optimizer was moving hardest at each checkpoint.

usage: python ckpt_drift.py --run az_full_isla_seed43 --epochs 29 31 ... --out drift.json
Runs on CPU. The model file <Model>.0.<E>.mdlus is a tar with model.pt inside (physicsnemo
Module.save); the optimizer file checkpoint.0.<E>.pt holds the AdamW state. Parameter names
for the optimizer state come from the state-dict order after dropping buffers (tensors whose
exp_avg shape has no match are reported by index).
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import tarfile
import zipfile

import torch

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"


def load_model_state(path: str) -> dict[str, torch.Tensor]:
    """State dict out of a .mdlus (tar or zip with model.pt)."""
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("model.pt"))
            data = tf.extractfile(member).read()
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith("model.pt"))
            data = zf.read(name)
    else:
        raise RuntimeError(f"{path}: neither tar nor zip")
    sd = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return {k: v for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}


def load_optimizer_state(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        return None, None
    opt = ck.get("optimizer")
    if opt is None:
        for v in ck.values():
            if isinstance(v, dict) and "param_groups" in v and "state" in v:
                opt = v
                break
    if opt is None:
        return None, None
    lr = opt["param_groups"][0].get("lr")
    eps = opt["param_groups"][0].get("eps", 1e-8)
    return opt, (lr, eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    ckdir = f"{T}/runs/{a.run}/checkpoints"

    norms: dict[int, dict[str, float]] = {}
    maxabs: dict[int, dict[str, float]] = {}
    scalars: dict[int, dict[str, float]] = {}
    adam_rel_step: dict[int, dict[str, float]] = {}
    adam_v_rms: dict[int, dict[str, float]] = {}
    prev: dict[str, torch.Tensor] | None = None
    prev_epoch = None
    rel_change: dict[str, dict[str, float]] = {}
    for e in a.epochs:
        mdlus = glob.glob(f"{ckdir}/*.0.{e}.mdlus")
        if not mdlus:
            print(f"epoch {e}: no .mdlus; skipped")
            continue
        sd = load_model_state(mdlus[0])
        norms[e] = {k: float(v.norm()) for k, v in sd.items()}
        maxabs[e] = {k: float(v.abs().max()) for k, v in sd.items()}
        scalars[e] = {k: float(v.flatten()[0]) for k, v in sd.items() if v.numel() == 1}
        if prev is not None:
            pair = f"{prev_epoch}->{e}"
            rel_change[pair] = {
                k: float((sd[k] - prev[k]).norm() / (prev[k].norm() + 1e-12))
                for k in sd if k in prev and sd[k].shape == prev[k].shape
            }
        # optimizer
        optpath = f"{ckdir}/checkpoint.0.{e}.pt"
        if os.path.exists(optpath):
            opt, meta = load_optimizer_state(optpath)
            if opt is not None:
                lr, eps = meta
                names = list(sd.keys())
                shapes = [tuple(sd[k].shape) for k in names]
                used = set()
                rel = {}
                vrms = {}
                for idx, st in sorted(opt["state"].items(), key=lambda kv: int(kv[0])):
                    m, v = st.get("exp_avg"), st.get("exp_avg_sq")
                    if m is None or v is None:
                        continue
                    shp = tuple(m.shape)
                    # the i-th parameter in optimizer order is the i-th state-dict tensor of that shape not yet used
                    name = None
                    for j, (n, s) in enumerate(zip(names, shapes)):
                        if s == shp and j not in used:
                            name = n
                            used.add(j)
                            break
                    key = name if name is not None else f"param#{idx}:{shp}"
                    w = sd[name] if name is not None else None
                    upd = m / (v.sqrt() + eps)
                    rel[key] = float(lr * upd.norm() / (w.norm() + 1e-12)) if w is not None else float("nan")
                    vrms[key] = float(v.mean().sqrt())
                adam_rel_step[e] = rel
                adam_v_rms[e] = vrms
                print(f"epoch {e}: lr={lr} params_in_opt={len(opt['state'])} tensors={len(sd)}")
        prev, prev_epoch = sd, e
        print(f"epoch {e}: {len(sd)} tensors; total norm {math.sqrt(sum(x*x for x in norms[e].values())):.3f}; "
              f"scalars {json.dumps({k.split('.')[-2] + '.' + k.split('.')[-1] if '.' in k else k: round(v, 4) for k, v in list(scalars[e].items())[:6]})}")

    out = {"run": a.run, "epochs": sorted(norms), "norms": norms, "maxabs": maxabs, "scalars": scalars,
           "rel_change": rel_change, "adam_rel_step": adam_rel_step, "adam_v_rms": adam_v_rms}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"))

    # Readable summary: top tensors by relative change per pair, and the total relative change per pair.
    print("\n=== relative weight change per consecutive pair (total over all tensors, and the top movers)")
    for pair, rc in rel_change.items():
        e1, e2 = (int(x) for x in pair.split("->"))
        tot = math.sqrt(sum(((norms[e2][k] - norms[e1][k]) ** 2) for k in rc)) / (math.sqrt(sum(norms[e1][k] ** 2 for k in rc)) + 1e-12)
        movers = sorted(rc.items(), key=lambda kv: -kv[1])[: a.top]
        print(f"{pair}: norm-vector change {tot:.4f}; median tensor rel-change {sorted(rc.values())[len(rc)//2]:.4f}")
        for k, v in movers[:8]:
            print(f"    {v:.4f}  {k}  |w|: {norms[e1][k]:.3f} -> {norms[e2][k]:.3f}  max|w|: {maxabs[e1][k]:.3f} -> {maxabs[e2][k]:.3f}")
    print("\n=== scalar parameters")
    for k in sorted(next(iter(scalars.values())).keys()) if scalars else []:
        print(f"{k}: " + " ".join(f"e{e}={scalars[e][k]:.4f}" for e in sorted(scalars)))
    print("\n=== Adam effective relative step (top tensors at each checkpoint)")
    for e in sorted(adam_rel_step):
        rs = adam_rel_step[e]
        vals = sorted((v for v in rs.values() if not math.isnan(v)))
        top = sorted(rs.items(), key=lambda kv: -(kv[1] if not math.isnan(kv[1]) else -1))[:6]
        print(f"e{e}: median {vals[len(vals)//2]:.2e}, max {vals[-1]:.2e}; top: " + "; ".join(f"{k}={v:.2e}" for k, v in top))
    print("\n=== largest-norm tensors at the first and last epoch")
    for e in (min(norms), max(norms)):
        top = sorted(norms[e].items(), key=lambda kv: -kv[1])[:8]
        print(f"e{e}: " + "; ".join(f"{k}={v:.2f}(max {maxabs[e][k]:.2f})" for k, v in top))


if __name__ == "__main__":
    main()
