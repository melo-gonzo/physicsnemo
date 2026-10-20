"""H-RESID probe, Instrument 1 (code_v26 ISLA, AeroZoo; book notebook #sec-nb-aerozoo-hresid-prereg): does the raw
read of h in the slice block's broadcast make the residual stream multiplicative and collapse the routing input?

Adapted 2026-10-15 from kernel/aerozoo/hlogit/hlogit_probe.py (same model build, same K cached validation cases and
draws, same bf16-autocast forward, compile_blocks off). Hooks only read (outputs checked bitwise on the first forward).
Per block l and case, from the block's own forward (blocks.py: `logits = assign(norm_assign(h), log_w)` ...
`h = h + broadcast(cat([h, back, geo_pool], -1)); return h + mlp(norm_mlp(h))`):

 (a) mean per-token L2 norm of h entering the block, after the broadcast add (mid) and leaving it; amplification =
     mean over tokens of |h_out| / |h_in| (also of |h_mid|/|h_in| and |h_out|/|h_mid|); query / boundary splits.
     Stream: block-0 input (after the embedding and the query tokens' qt), the boundary embedding output, the
     norm_out input (query tokens only in the interior mode) and the block-11 output over all tokens.
 (b) per checkpoint: spectral norms of W_h = broadcast.weight[:, :H], of I + W_h, of W_b = [:, H:2H], of
     W_g = [:, 2H:]. Column order checked at run time: the broadcast input's first H columns equal h_in bitwise, and
     F.linear of the three column blocks plus bias reproduces the broadcast in float32.
 (c) for u = norm_assign(h) (the hooked LayerNorm output, what the routing reads) and for raw h_in, over all tokens:
     top-direction energy share sigma_1^2 / ||X||_F^2, uncentered and centered over tokens (float64 Gram, explicit
     centering), the top-2 shares too, and the mean |cos| between token states over a fixed random subset of 4096
     tokens per case (off-diagonal pairs; torch.Generator seed 1234 + case index, same subset for every checkpoint).
 (d) bf16 rounding noise at the broadcast input, all products in float32: msg = W_b back + W_g geo_pool per token;
     "lit": e = bf16(h_in) - h_in (zero when the stream is already bf16; the dtype is recorded);
     "obs": e = the rounding of the add that produced h_in (block l-1's `h_mid + mlp`: stored bf16 minus the float32
     sum of its bf16 operands); block 0 has none. Reported: mean over tokens of |W_h e| / |msg|, the ratio of the
     token means, and mean |e|/|h|. Also |W_h h| / |msg| (the raw read against the messages) and the fraction of each
     residual add's update lost to rounding: mean |stored - (h + update)| / |update| for both adds.
 (e) routing effective slices per block from the 2nd _softmax_over_points input (logits + bias + log_w; log_w is
     constant over slices): exp H over slices (mean over tokens) and n_eff of the slice mass (exp H of mean_N
     softmax_S), as H-LOGIT's route_slices "bf" statistics.
 (f) the terms of the update h_out - h_in = W_h h + W_b back + W_g geo_pool + bias + mlp(norm_mlp(h_mid)), with
     back = point_mix (z + slice_mlp(z)) split into W_b point_mix z (z = einsum(a, h), pooled from the RAW h) and
     W_b point_mix slice_mlp(z); point_mix captured from the (eager or fused) geometry region's output. Per term the
     mean token norm and the mean of |term| / |h_in|; also lin_h = W_h h + W_b point_mix z (both linear in h).

usage (from recipe_v9, PYTHONPATH=code_v26:recipe_v9/src, RANK/... set):
  python hresid_probe.py --plan "run=1,5,9;run2=1,latest" --k 4 --out-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn.functional as F

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
# STREAM-AF (2026-10-19): recipe snapshot and lane manifest from the environment (each AirfRANS lane's own recipe;
# kernel/airfrans_u/lanes.tsv); defaults are the original AeroZoo values.
RECIPE = os.environ.get("PROBE_RECIPE", f"{T}/recipe_v9")
MANIFEST = os.environ.get("PROBE_MANIFEST", f"{T}/kernel/aerozoo/lanes.tsv")
RUNS_ROOT = os.environ.get("PROBE_RUNS", f"{T}/runs")  # STREAM-AF: checkpoint root (copied lanes live outside runs/)
NCOS = 4096


def support(z: torch.Tensor, dim: int) -> torch.Tensor:
    logp = torch.log_softmax(z.float(), dim=dim)
    return torch.exp(-(logp.exp() * logp).sum(dim))


def slice_mass_neff(z: torch.Tensor) -> float:
    q = torch.softmax(z.float(), dim=-1).mean(0)
    return float(torch.exp(-(q * (q + 1e-30).log()).sum()))


def energy_shares(X: torch.Tensor) -> dict:
    """X (N, H): top-1 / top-2 energy share of the uncentered and the token-centered matrix (float64)."""
    X = X.double()
    out = {}
    for tag, Y in (("unc", X), ("cen", X - X.mean(0, keepdim=True))):
        ev = torch.linalg.eigvalsh(Y.T @ Y)  # ascending
        tr = float(ev.sum())
        out[f"top1_{tag}"] = float(ev[-1]) / tr
        out[f"top2_{tag}"] = float(ev[-1] + ev[-2]) / tr
    return out


def mean_abs_cos(X: torch.Tensor, idx: torch.Tensor) -> float:
    Y = F.normalize(X[idx].double(), dim=-1)
    C = (Y @ Y.T).abs()
    n = C.shape[0]
    return float((C.sum() - C.diagonal().sum()) / (n * (n - 1)))


def tok_norm(x: torch.Tensor) -> torch.Tensor:
    return x.float().norm(dim=-1)


def block_stats(c: dict, H: int, cos_idx: torch.Tensor, nq: int, prev_round: torch.Tensor | None) -> dict:
    """c: captured tensors of one block (batch dim dropped). Returns scalar statistics."""
    h_in, h_mid, h_out = c["h_in"], c["h_mid"], c["h_out"]
    s = {"dtype_h_in": str(h_in.dtype), "dtype_h_mid": str(h_mid.dtype), "dtype_h_out": str(h_out.dtype),
         "dtype_bcast_in": str(c["x_b"].dtype), "dtype_u": str(c["u"].dtype)}
    n_in, n_mid, n_out = tok_norm(h_in), tok_norm(h_mid), tok_norm(h_out)
    for tag, n in (("in", n_in), ("mid", n_mid), ("out", n_out)):
        s[f"norm_{tag}"] = float(n.mean())
        s[f"norm_{tag}_query"] = float(n[-nq:].mean())
        s[f"norm_{tag}_boundary"] = float(n[:-nq].mean())
    s["norm_in_max"] = float(n_in.max())
    s["amp"] = float((n_out / n_in).mean())
    s["amp_bcast"] = float((n_mid / n_in).mean())
    s["amp_mlp"] = float((n_out / n_mid).mean())
    s["amp_query"] = float((n_out / n_in)[-nq:].mean())
    s["amp_boundary"] = float((n_out / n_in)[:-nq].mean())
    s["amp_of_means"] = s["norm_out"] / s["norm_in"]
    ### (c)
    for tag, X in (("u", c["u"]), ("h", h_in)):
        for k, v in energy_shares(X).items():
            s[f"{tag}_{k}"] = v
        s[f"{tag}_abscos"] = mean_abs_cos(X, cos_idx)
    ### (b)-check and (d)
    x_b = c["x_b"]
    s["check_cat_h_bitwise"] = bool(torch.equal(x_b[:, :H], h_in.to(x_b.dtype)))
    W = c["W"]  # float32 (H, 2H + H/2)
    Wh, Wb, Wg, b = W[:, :H], W[:, H:2 * H], W[:, 2 * H:], c["bias"]
    xf = x_b.float()
    msg = F.linear(xf[:, H:2 * H], Wb) + F.linear(xf[:, 2 * H:], Wg)
    hread = F.linear(xf[:, :H], Wh)
    full = F.linear(xf, W, b)
    s["check_split_maxrel"] = float(((hread + msg + b) - full).norm(dim=-1).max() / full.norm(dim=-1).max())
    s["check_bcast_out_rel"] = float((c["b_out"].float() - full).norm() / full.norm())  # bf16 matmul vs fp32
    msg_n = msg.norm(dim=-1).clamp_min(1e-30)
    s["msg_norm"] = float(msg_n.mean())
    s["hread_norm"] = float(hread.norm(dim=-1).mean())
    s["hread_over_msg"] = float((hread.norm(dim=-1) / msg_n).mean())
    e_lit = h_in.to(torch.bfloat16).float() - h_in.float()
    nz = tok_norm(h_in).clamp_min(1e-30)
    for tag, e in (("lit", e_lit), ("obs", prev_round)):
        if e is None:
            s[f"noise_{tag}_over_msg"] = s[f"noise_{tag}_over_msg_of_means"] = s[f"round_{tag}_rel_h"] = None
            continue
        we = F.linear(e, Wh).norm(dim=-1)
        s[f"noise_{tag}_over_msg"] = float((we / msg_n).mean())
        s[f"noise_{tag}_over_msg_of_means"] = float(we.mean() / msg_n.mean())
        s[f"noise_{tag}_norm"] = float(we.mean())
        s[f"round_{tag}_rel_h"] = float((e.norm(dim=-1) / nz).mean())
    ### update lost to rounding in the two residual adds
    upd1 = c["b_out"].float()
    s["add1_lost_frac"] = float(((h_mid.float() - (h_in.float() + upd1)).norm(dim=-1)
                                 / upd1.norm(dim=-1).clamp_min(1e-30)).mean())
    upd2 = c["mlp_out"].float()
    r2 = h_out.float() - (h_mid.float() + upd2)
    s["add2_lost_frac"] = float((r2.norm(dim=-1) / upd2.norm(dim=-1).clamp_min(1e-30)).mean())
    s["upd_bcast_over_h"] = float((upd1.norm(dim=-1) / nz).mean())
    s["upd_mlp_over_h"] = float((upd2.norm(dim=-1) / tok_norm(h_mid).clamp_min(1e-30)).mean())
    ### (f) the separate terms of the block's update h_out - h_in, float32, per token, against |h_in|
    Pm = c["point_mix"].float()  # (N, S)
    zs, zm = c["z"].float(), c["zmlp"].float()  # (S, H): slice states pooled from raw h, and slice_mlp(z)
    pz, pm = Pm @ zs, Pm @ zm
    s["check_back_rel"] = float((pz + pm - xf[:, H:2 * H]).norm() / xf[:, H:2 * H].norm().clamp_min(1e-30))
    t_bz, t_bm = F.linear(pz, Wb), F.linear(pm, Wb)
    t_g = F.linear(xf[:, 2 * H:], Wg)
    terms = {"Wh_h": hread, "Wb_Pz": t_bz, "Wb_Pmlpz": t_bm, "Wb_back": t_bz + t_bm, "Wg_geo": t_g,
             "bias": b.expand_as(hread), "mlp": c["mlp_out"].float(), "lin_h": hread + t_bz,
             "bcast_total": c["b_out"].float(), "update_total": h_out.float() - h_in.float()}
    for name, t in terms.items():
        tn = t.norm(dim=-1)
        s[f"term_{name}_norm"] = float(tn.mean())
        s[f"term_{name}_over_h"] = float((tn / nz).mean())
    s["pz_over_h"] = float((pz.norm(dim=-1) / nz).mean())  # |P z| / |h|: pooled raw h read back
    s["z_norm_mean"] = float(zs.norm(dim=-1).mean())
    del Pm, zs, zm, pz, pm, t_bz, t_bm, t_g, terms
    ### (e)
    x2 = c["x2"]
    sup = support(x2, -1)
    s["route_support_mean"] = float(sup.mean())
    s["route_neff_slice_mass"] = slice_mass_neff(x2)
    return s, r2


def install_hooks(model, blocks, H, device, cos_idx, isla_blocks):
    """Read-only hooks on every block; returns (ctx, handles, orig_softmax_over_points)."""
    ctx = {"on": False, "cur": None, "cap": None, "calls": 0, "stream": {}}

    def cap(key, t):
        ctx["cap"][ctx["cur"]][key] = t.detach()[0]

    def blk_pre(mod, inputs):
        ctx["cur"] = blocks.index(mod)
        ctx["calls"] = 0
        if ctx["on"]:
            cap("h_in", inputs[0])

    def blk_post(mod, inputs, out):
        if not ctx["on"]:
            return
        cap("h_out", out)
        i, k = ctx["cur"], ctx["k"]
        c = ctx["cap"][i]
        missing = {"h_in", "h_mid", "h_out", "u", "x_b", "b_out", "mlp_out", "x2", "point_mix", "z", "zmlp"} - set(c)
        assert not missing, (i, missing)
        N = c["h_in"].shape[0]
        if k not in cos_idx:
            g = torch.Generator(device="cpu").manual_seed(1234 + k)
            cos_idx[k] = torch.randperm(N, generator=g)[: min(NCOS, N)].to(device)
        c["W"] = mod.broadcast.weight.detach().float()
        c["bias"] = mod.broadcast.bias.detach().float()
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            st, ctx["prev"] = block_stats(c, H, cos_idx[k], ctx["nq"], ctx["prev"])
        st["n_tokens"] = N
        ctx["blocks_out"][i] = st
        ctx["cap"][i] = None
        ctx["cur"] = None

    def norm_assign_post(mod, inputs, out):
        if ctx["on"]:
            cap("u", out)

    def bcast_pre(mod, inputs):
        if ctx["on"]:
            cap("x_b", inputs[0])

    def bcast_post(mod, inputs, out):
        if ctx["on"]:
            cap("b_out", out)

    def norm_mlp_pre(mod, inputs):
        if ctx["on"]:
            cap("h_mid", inputs[0])

    def mlp_post(mod, inputs, out):
        if ctx["on"]:
            cap("mlp_out", out)

    def embed_post(mod, inputs, out):
        if ctx["on"] and "embed_boundary" not in ctx["stream"]:  # first embed call = boundary tokens
            ctx["stream"]["embed_boundary"] = float(tok_norm(out.detach()[0]).mean())
            ctx["stream"]["embed_dtype"] = str(out.dtype)

    def norm_out_pre(mod, inputs):
        if ctx["on"]:
            ctx["stream"]["norm_out_in"] = float(tok_norm(inputs[0].detach()[0]).mean())
            ctx["stream"]["norm_out_in_tokens"] = int(inputs[0].shape[1])

    def slice_mlp_post(mod, inputs, out):
        if ctx["on"]:
            cap("z", inputs[0])
            cap("zmlp", out)

    orig_geo, orig_fused = isla_blocks._geo_region, isla_blocks._fused_geo_region

    def geo_region(*a, **kw):
        out = orig_geo(*a, **kw)
        if ctx["on"] and ctx["cur"] is not None:
            cap("point_mix", out[1])
        return out

    def fused_geo_region():
        fn = orig_fused()

        def call(*a, **kw):
            out = fn(*a, **kw)
            if ctx["on"] and ctx["cur"] is not None:
                cap("point_mix", out[1])
            return out
        return call

    isla_blocks._geo_region, isla_blocks._fused_geo_region = geo_region, fused_geo_region
    orig_sp = isla_blocks._softmax_over_points

    def sp(x):
        if ctx["on"] and ctx["cur"] is not None:
            ctx["calls"] += 1
            if ctx["calls"] == 2:
                cap("x2", x)
        return orig_sp(x)

    handles = []
    for b in blocks:
        handles += [b.register_forward_pre_hook(blk_pre), b.register_forward_hook(blk_post),
                    b.norm_assign.register_forward_hook(norm_assign_post),
                    b.broadcast.register_forward_pre_hook(bcast_pre), b.broadcast.register_forward_hook(bcast_post),
                    b.norm_mlp.register_forward_pre_hook(norm_mlp_pre), b.mlp.register_forward_hook(mlp_post),
                    b.slice_mlp.register_forward_hook(slice_mlp_post)]
    handles += [model.embed.register_forward_hook(embed_post), model.norm_out.register_forward_pre_hook(norm_out_pre)]
    isla_blocks._softmax_over_points = sp
    return ctx, handles, orig_sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help='"run=e1,e2,...;run2=...", "latest" = newest ISLA.0.<N>.mdlus')
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cells", type=int, default=200000)
    ap.add_argument("--dataset", default="aerozoo_uvolume")
    ap.add_argument("--precision", default="bfloat16")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import hydra
    from hydra import compose, initialize_config_dir

    from datasets import build_dataloaders
    from infer import _sample_id
    from output_normalize import normalize_output_to_tensordict, require_output_type
    from utils import get_autocast_context, recursive_to_device, set_seed
    from physicsnemo import datapipes  # noqa: F401
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils import load_checkpoint
    import physicsnemo.experimental.models.isla.blocks as isla_blocks

    plan = []
    for part in args.plan.split(";"):
        run, eps = part.split("=")
        plan.append((run.strip(), [e.strip() for e in eps.split(",") if e.strip()]))
    rows = {r["run_id"]: r for r in csv.DictReader(open(MANIFEST), delimiter="\t")}
    models_extras = {(rows[r]["model"], rows[r]["extra"]) for r, _ in plan}
    assert len(models_extras) == 1, models_extras
    row = rows[plan[0][0]]
    extra = [t for t in (row["extra"] if row["extra"] != "-" else "").split()
             if not t.startswith("+training.") and not t.startswith("dataloader.")
             and not t.startswith("model.compile_blocks=")]  # STREAM-AF: the probe runs eager (hooks), as for AeroZoo

    DistributedManager.initialize()
    device = DistributedManager().device
    overrides = [
        f"model={row['model']}", f"dataset={args.dataset}", f"run_id={plan[0][0]}", "infer_split=val",
        "+save_predictions=false", f"sampling_resolution={args.cells}", f"precision={args.precision}",
        f"checkpoint_dir={RUNS_ROOT}", f"output_dir={args.out_dir}/_scratch", "model.compile_blocks=false", *extra,
    ]
    with initialize_config_dir(config_dir=f"{RECIPE}/conf", version_base=None):
        cfg = compose(config_name="infer", overrides=overrides)
    print("overrides:", overrides, flush=True)
    set_seed(cfg.training.get("seed", None), rank=0)
    _tl, val_loader, _norm, info = build_dataloaders(cfg)
    target_config = info["targets"]
    output_type = require_output_type(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    model.eval()
    blocks = list(model.blocks)
    nb = len(blocks)
    H = blocks[0].norm_assign.normalized_shape[0]
    assert blocks[0].broadcast.in_features == 2 * H + H // 2, blocks[0].broadcast

    ds, collate = val_loader.dataset, val_loader.collate_fn
    idxs = list(val_loader.sampler)[: args.k]
    batches, case_ids = [], []
    for idx in idxs:
        dom, meta = ds[idx]
        case_ids.append(_sample_id(meta, idx))
        batches.append(recursive_to_device(collate([(dom, meta)]), device))
    print(f"{len(list(val_loader.sampler))} {cfg.val_split} samples; using {idxs} {case_ids}", flush=True)

    cos_idx = {}
    ctx, handles, orig_sp = install_hooks(model, blocks, H, device, cos_idx, isla_blocks)

    def param_stats():
        out = {}
        spec = lambda A: float(torch.linalg.matrix_norm(A.double(), ord=2))  # noqa: E731
        I = torch.eye(H, device=device, dtype=torch.float64)
        for i, b in enumerate(blocks):
            W = b.broadcast.weight.detach().double()
            Wh = W[:, :H]
            out[i] = {"Wh_spec": spec(Wh), "I_plus_Wh_spec": spec(I + Wh), "Wb_spec": spec(W[:, H:2 * H]),
                      "Wg_spec": spec(W[:, 2 * H:]), "Wh_fro": float(Wh.norm()),
                      "I_plus_Wh_smin": float(torch.linalg.svdvals(I + Wh)[-1]),
                      "bias_norm": float(b.broadcast.bias.detach().double().norm())}
        return out

    def rel_l2(output, batch):
        try:
            pred = normalize_output_to_tensordict(output, target_config, output_type)
            tgt = batch["targets"]
            return {k: float((pred[k].float().reshape(-1) - tgt[k].float().reshape(-1)).norm()
                             / tgt[k].float().reshape(-1).norm().clamp_min(1e-30)) for k in tgt.keys()}
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)[:200]}

    def nq_of(batch):
        try:
            return int(batch["forward_kwargs"]["prediction_points"].points.shape[-2])
        except Exception:  # noqa: BLE001
            return args.cells

    os.makedirs(args.out_dir, exist_ok=True)
    checked = False
    for run, eps in plan:
        ckpt_path = os.path.join(RUNS_ROOT, run, "checkpoints")
        available = sorted(int(f.split(".")[2]) for f in os.listdir(ckpt_path) if f.startswith("ISLA.0.") and f.endswith(".mdlus"))
        epochs = []
        for e in eps:
            e = available[-1] if e == "latest" else int(e)
            if e in available and e not in epochs:
                epochs.append(e)
            elif e not in available:
                print(f"WARNING {run} epoch {e} not on disk", flush=True)
        print(f"{run}: available {available[:3]}...{available[-3:]}; probing {epochs}", flush=True)
        os.makedirs(os.path.join(args.out_dir, run), exist_ok=True)
        for epoch in epochs:
            out_path = os.path.join(args.out_dir, run, f"e{epoch}.json")
            if os.path.exists(out_path):
                print("exists, skip", out_path, flush=True)
                continue
            t0 = time.time()
            if os.path.exists(os.path.join(ckpt_path, f"checkpoint.0.{epoch}.pt")):
                loaded = load_checkpoint(path=ckpt_path, models=model, device=device, epoch=epoch)
                assert loaded == epoch, (loaded, epoch)
            else:
                model.load(os.path.join(ckpt_path, f"ISLA.0.{epoch}.mdlus"))
                model.to(device)
            model.eval()
            params = param_stats()
            per_case, metrics, n_tokens, identical = [], [], [], None
            for k, batch in enumerate(batches):
                nq = nq_of(batch)
                if not checked:
                    ctx["on"] = False
                    with torch.no_grad(), get_autocast_context(cfg.precision):
                        ref = model(**batch["forward_kwargs"])
                    ref_td = normalize_output_to_tensordict(ref, target_config, output_type)
                ctx["on"], ctx["cap"], ctx["stream"], ctx["cur"] = True, [dict() for _ in range(nb)], {}, None
                ctx["k"], ctx["nq"], ctx["prev"], ctx["blocks_out"] = k, nq, None, {}
                with torch.no_grad(), get_autocast_context(cfg.precision):
                    out = model(**batch["forward_kwargs"])
                ctx["on"], ctx["cur"], ctx["prev"] = False, None, None
                if not checked:
                    out_td = normalize_output_to_tensordict(out, target_config, output_type)
                    identical = {kk: float((out_td[kk].float() - ref_td[kk].float()).abs().max()) for kk in ref_td.keys()}
                    print("hooks on vs off, max |diff| per output:", identical, flush=True)
                    del ref, ref_td, out_td
                metrics.append(rel_l2(out, batch))
                del out
                with torch.no_grad():
                    assert sorted(ctx["blocks_out"]) == list(range(nb)), sorted(ctx["blocks_out"])
                    case = {"blocks": ctx["blocks_out"], "stream": dict(ctx["stream"])}
                    n_tokens.append(case["blocks"][0]["n_tokens"])
                    st = case["stream"]
                    st["block0_in"] = case["blocks"][0]["norm_in"]
                    st["block_last_out"] = case["blocks"][nb - 1]["norm_out"]
                    st["block_last_out_query"] = case["blocks"][nb - 1]["norm_out_query"]
                    st["growth"] = st["norm_out_in"] / st["block0_in"]
                    st["growth_all_tokens"] = st["block_last_out"] / st["block0_in"]
                    st["sum_log_amp"] = sum(math.log(case["blocks"][i]["amp"]) for i in range(nb))
                    st["sum_log_amp_of_means"] = sum(math.log(case["blocks"][i]["amp_of_means"]) for i in range(nb))
                    st["frac_log_growth_by_amp"] = st["sum_log_amp"] / math.log(st["growth"]) if st["growth"] != 1 else None
                    per_case.append(case)
                if not checked:
                    b0 = case["blocks"][0]
                    print("dtypes:", {k2: v for k2, v in b0.items() if k2.startswith("dtype")}, "embed", st.get("embed_dtype"),
                          "| cat[:H]==h_in", [case["blocks"][i]["check_cat_h_bitwise"] for i in range(nb)],
                          "| split maxrel", max(case["blocks"][i]["check_split_maxrel"] for i in range(nb)),
                          "| bcast bf16 vs fp32", max(case["blocks"][i]["check_bcast_out_rel"] for i in range(nb)),
                          "| P(z+mlp z) vs back", max(case["blocks"][i]["check_back_rel"] for i in range(nb)), flush=True)
                    checked = True
            ### mean over cases
            def avg(get):
                vals = [get(c) for c in per_case]
                if any(v is None for v in vals):
                    return None
                if isinstance(vals[0], (bool, str)):
                    return vals[0] if all(v == vals[0] for v in vals) else vals
                return float(sum(vals) / len(vals))
            agg_blocks = {i: {key: avg(lambda c, i=i, key=key: c["blocks"][i][key]) for key in per_case[0]["blocks"][i]}
                          for i in range(nb)}
            agg_stream = {key: avg(lambda c, key=key: c["stream"].get(key)) for key in per_case[0]["stream"]}
            result = {"run": run, "epoch": epoch, "lr": rows[run]["lr"], "train_dataset": rows[run]["dataset"],
                      "eval_dataset": args.dataset, "precision": cfg.precision, "code": row["code"], "hidden": H,
                      "sample_indices": [int(i) for i in idxs], "case_ids": case_ids, "n_tokens": n_tokens,
                      "hook_identity_maxdiff": identical, "rel_l2_normalized": metrics,
                      "blocks": agg_blocks, "stream": agg_stream, "params": params, "per_case": per_case,
                      "seconds": time.time() - t0}
            with open(out_path + ".tmp", "w") as f:
                json.dump(result, f, indent=1)
            os.replace(out_path + ".tmp", out_path)
            s = agg_stream
            print(f"{run} e{epoch} t={result['seconds']:.1f}s | growth {s['growth']:.3g} sumlogamp/loggrowth "
                  f"{s['frac_log_growth_by_amp']} | amp {[round(agg_blocks[i]['amp'], 3) for i in range(nb)]}"
                  f" | u top1 unc {[round(agg_blocks[i]['u_top1_unc'], 3) for i in range(nb)]}"
                  f" cen {[round(agg_blocks[i]['u_top1_cen'], 3) for i in range(nb)]}"
                  f" | I+Wh {[round(params[i]['I_plus_Wh_spec'], 2) for i in range(nb)]}"
                  f" | noise_obs/msg blk{nb - 1} {agg_blocks[nb - 1]['noise_obs_over_msg']}", flush=True)
    for h in handles:
        h.remove()
    isla_blocks._softmax_over_points = orig_sp
    print("HRESID-PROBE-DONE", flush=True)


if __name__ == "__main__":
    main()
