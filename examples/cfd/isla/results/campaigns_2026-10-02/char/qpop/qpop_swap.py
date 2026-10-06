"""QUERY-POP (2026-10-06): does the volume model's prediction at a point depend on the POPULATION of prediction points it
is scored with? Peter's question: a model trained on CFD-mesh queries (count-uniform over the mesh's points, wall-heavy)
and inferred on a volume-uniform point cloud (far-field heavy). The interior queries are interacting tokens whose total
routing mass is a learned fraction of the boundary measure split EQUALLY over them, so the slice states see the query
population; the built-in geometry (VOL-GEO, SOFT-GEO) is a separate channel and this probe holds it fixed by construction
(every scored point keeps its own native geometry: companions do not change a scored point's nearest token).

Scores one natively trained ISLA volume checkpoint on the 48 validation cars, float32, 200,000 native interior points,
and writes <out>/<condition>/metrics.jsonl (infer_step rows) plus summary.json. Every metric is computed on the SAME
scored set, the first half (100,000) of the native points, so the conditions differ only in the query population:

  native_h     the native 200,000 tokens; metric on the first 100,000 (the control)
  half         tokens = the first 100,000 only (a count change at the native law)
  box          tokens = the native 200,000 + 200,000 companions drawn uniformly in the interior's bounding box
               (far-field heavy; a few percent fall inside the body); metric on the first 100,000
  wall         tokens = the native 200,000 + 200,000 companions resampled from the native points with probability
               proportional to 1 / |sdf| (near-wall heavy); metric on the first 100,000
  full         the native 200,000 tokens, metric on all 200,000 (reproduces evals_char/<run>/native per case)

Also records exp(qt.logw), the queries' total routing mass as a fraction of the boundary measure, read from the checkpoint.
Mirrors recipe_v8/src/infer.py main() step for step. usage (from recipe_v8, PYTHONPATH=code_v20:recipe_v8/src, RANK/... set):
  python qpop_swap.py <run_id> <out_dir>
"""

import json
import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

T = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0")
R = T / "recipe_v8"
sys.path.insert(0, str(R / "src"))

from datasets import build_dataloaders, find_normalizer  # noqa: E402
from infer import (  # noqa: E402
    _norm_stats_match,
    _sample_id,
    _spatial_dims_of_first_sample,
    resolve_checkpoint_path,
)
from metrics import MetricCalculator, resolve_metrics  # noqa: E402
from output_normalize import normalize_output_to_tensordict, require_output_type  # noqa: E402
from utils import get_autocast_context, recursive_to_device, set_seed  # noqa: E402

from physicsnemo import datapipes  # noqa: E402,F401  (registers ${dp:...})
from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.mesh import Mesh  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402

FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
CONDITIONS = ("native_h", "half", "box", "wall", "full")
N_COMPANIONS = 200000


def compose_cfg(run_id: str, out_dir: Path):
    overrides = [
        "model=isla_volume",
        "model.hidden=512",
        "model.n_slices=512",
        "dataset=drivaer_ml_volume_reference",
        f"run_id={run_id}",
        "infer_split=val",
        "+save_predictions=false",
        "sampling_resolution=200000",
        "precision=float32",
        f"checkpoint_dir={T / 'runs'}",
        f"output_dir={out_dir}",
    ]
    with initialize_config_dir(config_dir=str(R / "conf"), version_base=None):
        return compose(config_name="infer", overrides=overrides)


def load_reference(run_id: str) -> dict[str, dict[str, float]]:
    path = T / "evals_char" / run_id / "native" / run_id / "metrics.jsonl"
    rows = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("phase") == "infer_step":
                rows[rec["sample_id"]] = rec["metrics"]
    return rows


def main(run_id: str, out_dir: Path) -> None:
    DistributedManager.initialize()
    dm = DistributedManager()
    device = dm.device
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    for c in CONDITIONS:
        (out_dir / c).mkdir(exist_ok=True)
        writers[c] = open(out_dir / c / "metrics.jsonl", "w")
    cfg = compose_cfg(run_id, out_dir / "full")
    set_seed(cfg.training.get("seed", None), rank=dm.rank)
    _, vl, norm, info = build_dataloaders(cfg)
    target_config = info["targets"]
    output_type = require_output_type(cfg)
    ckpt_path = resolve_checkpoint_path(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    epoch = load_checkpoint(path=ckpt_path, models=model, device=device)
    assert epoch > 0, ckpt_path
    query_mass = float(torch.exp(model.qt.logw.detach().float()).item())
    print(f"[{run_id}] checkpoint {ckpt_path} epoch {epoch}; query routing mass exp(logw) = {query_mass:.4f} of the boundary measure", flush=True)
    norm_stats_path = Path(ckpt_path) / "norm_stats.pt"
    if norm_stats_path.exists():
        saved = torch.load(norm_stats_path, weights_only=True)
        if norm is not None and not _norm_stats_match(norm.stats, saved):
            print(f"[{run_id}] norm_stats.pt differs from the yaml; using the persisted stats", flush=True)
            val_norm = find_normalizer([vl.dataset])
            for live in {id(n): n for n in (norm, val_norm) if n is not None}.values():
                live.stats = saved
    model.eval()
    ds, smp, collate = vl.dataset, vl.sampler, vl.collate_fn
    n_spatial = _spatial_dims_of_first_sample(ds, smp)
    calc = MetricCalculator(target_config=target_config, metrics=resolve_metrics(cfg), n_spatial_dims=n_spatial)
    ref = load_reference(run_id)
    gen = torch.Generator(device="cpu").manual_seed(2026)

    def forward(batch, prediction_points):
        kw = dict(batch["forward_kwargs"])
        kw["prediction_points"] = prediction_points
        with torch.no_grad(), get_autocast_context(cfg.precision):
            output = model(**kw)
        return normalize_output_to_tensordict(output, target_config, output_type)

    def take(td, n):
        """The first n points of a per-point TensorDict, whatever its batch_size."""
        if n is None:
            return td
        if len(td.batch_size) and td.batch_size[0] >= n:
            return td[:n]
        return TensorDict({k: v[:n] for k, v in td.items()}, batch_size=[])

    def score(pred_td, batch, n_score):
        pred = take(pred_td, n_score)
        tgt = take(batch["targets"], n_score)
        tm = batch.get("target_measure")
        if tm is not None and n_score is not None:
            tm = tm[..., :n_score]
        md = calc(pred.float(), tgt.float(), None if tm is None else tm.float())
        return {k: v.item() for k, v in md.cpu().items()}

    per_case = {c: [] for c in CONDITIONS}
    sanity = []
    idx_list = list(smp)
    print(f"[{run_id}] {len(idx_list)} validation cases", flush=True)
    for i, idx in enumerate(idx_list):
        dom, meta = ds[idx]
        sid = _sample_id(meta, idx)
        batch = recursive_to_device(collate([(dom, meta)]), device)
        pp = batch["forward_kwargs"]["prediction_points"]
        pts = pp.points
        q = pts.shape[0]
        h = q // 2
        pd = pp.point_data
        sdf = pd["sdf"].reshape(-1).float().abs().clamp_min(1e-6) if "sdf" in pd.keys() else None
        lo, hi = pts.min(0).values, pts.max(0).values
        box = lo + (hi - lo) * torch.rand(N_COMPANIONS, pts.shape[1], generator=gen).to(pts)
        if sdf is not None:
            wall_idx = torch.multinomial((1.0 / sdf).cpu(), N_COMPANIONS, replacement=True, generator=gen)
        else:
            wall_idx = torch.randint(0, q, (N_COMPANIONS,), generator=gen)
        wall = pts[wall_idx.to(pts.device)]
        pred_full = forward(batch, pp)
        res = {
            "full": score(pred_full, batch, None),
            "native_h": score(pred_full, batch, h),
            "half": score(forward(batch, Mesh(points=pts[:h])), batch, h),
            "box": score(forward(batch, Mesh(points=torch.cat([pts, box]))), batch, h),
            "wall": score(forward(batch, Mesh(points=torch.cat([pts, wall]))), batch, h),
        }
        for c, m in res.items():
            per_case[c].append(m)
            writers[c].write(json.dumps({"phase": "infer_step", "step": i, "sample_id": sid, "metrics": m}) + "\n")
            writers[c].flush()
        r = ref.get(sid)
        diffs = {k: abs(res["full"][k] - r[k]) for k in FIELDS} if r else None
        sanity.append({"sample_id": sid, "found": r is not None, "abs_diff": diffs})
        line = "  ".join(f"{c}: p={res[c]['pressure_l2']:.4f} v={res[c]['velocity_l2']:.4f} nut={res[c]['nut_l2']:.4f}" for c in CONDITIONS)
        print(f"[{run_id}] [{i + 1}/{len(idx_list)}] {sid} q={q}  {line}  " + (f"dfull={max(diffs.values()):.1e}" if diffs else "dfull=NA"), flush=True)

    means = {c: {k: sum(m[k] for m in per_case[c]) / len(per_case[c]) for k in per_case[c][0]} for c in CONDITIONS}
    for c in CONDITIONS:
        writers[c].write(json.dumps({"phase": "infer_summary", "space": "training", "num_samples": len(per_case[c]), "metrics": means[c]}) + "\n")
        writers[c].close()
    summary = {
        "run_id": run_id,
        "epoch": epoch,
        "query_routing_mass_fraction_of_boundary": query_mass,
        "num_samples": len(idx_list),
        "n_companions": N_COMPANIONS,
        "means": means,
        "ratio_to_native_h": {c: {f: means[c][f] / means["native_h"][f] for f in FIELDS} for c in ("half", "box", "wall")},
        "sanity_full": {
            "n_found": sum(s["found"] for s in sanity),
            "max_abs_diff": {f: max(s["abs_diff"][f] for s in sanity if s["abs_diff"]) for f in FIELDS} if any(s["abs_diff"] for s in sanity) else None,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
