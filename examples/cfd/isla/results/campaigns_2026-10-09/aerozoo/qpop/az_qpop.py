"""ISLA-DTT QUERY-POP on AeroZoo (recipe_v9, datasets/aerozoo_volume.yaml): does a prediction depend on the POPULATION
of prediction points it is scored with? The DrivAerML probe (dtt_qpop.py, E4) carried to AeroZoo's interior task and to
three arms: ISLA and GeoTransolver at their reference configurations, and ISLA with the tree read (query_independent,
kernel_read=mean, kernel_read_theta=1.0), which is exactly query independent by construction (a point's walk sees the
boundary tree and the point alone), so its ratios below must be 1.000 to float32 roundoff.

Scores one lane's native checkpoint (the latest, or --epoch N) on the validation side of the `full` split (386 cases),
float32, the recipe's 200,000 native interior points (the reader's seeded draw, 50,000 faces per boundary role), and
writes <out>/<condition>/metrics.jsonl (a config row, one infer_step row per case, an infer_summary row) plus
summary.json. Every metric is computed on the SAME scored set, the first half (100,000) of the native points (the
interior is the row-shuffled copy, so the first half is a random half), so the conditions differ only in the query
population:

  native_h     the native 200,000 tokens; metric on the first 100,000 (the control)
  half         tokens = the first 100,000 only
  box          tokens = the native 200,000 + 200,000 companions drawn uniformly in the interior's bounding box
  wall         tokens = the native 200,000 + 200,000 companions resampled from the native points with probability
               proportional to 1 / |sdf_wall| (near the no_slip body; sdf_wall is the pipeline's signed distance to it)
  full         the native 200,000 tokens, metric on all 200,000 (what infer.py writes per case at the same checkpoint)

A swapped condition is a new case, not a new prediction_points argument: the interior point cloud is replaced (the
first 100,000 native points, or the native 200,000 followed by the companions), the pipeline's own two
ComputeSDFFromBoundary transforms recompute sdf_wall / sdf_wall_normals / sdf_slip on every point, and the recipe's
collate builds the forward kwargs from it. ISLA reads only the points (no query_schema); GeoTransolver reads the swapped
set as geometry, local_embedding (points + the three SDF fields) and local_positions, as its model yaml wires them. On
the native points the recomputed SDF fields must equal the pipeline's to float32 roundoff (asserted on the first case,
recorded for every case). A case without slip walls has sdf_slip = the diagonal of the interior's bounding box, which
depends on the point set; the probe holds it at the native sample's value so the native points' inputs are identical
in every condition. Companion point data are zero (targets, never scored) except the quadrature measure
`_target_quadrature_measure`, which gets the mean native cell volume: neither model reads it and the metric is the
unweighted mean over the shared native points, so the choice affects nothing scored.

usage (from recipe_v9, PYTHONPATH=<code>:recipe_v9/src, RANK/... set):
  python az_qpop.py <run_id> <out_dir> <isla|gt|dtt> [--n-cases N] [--epoch N] [hydra override ...]
"""

import json
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir

D = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-10-06-isla-dtt")
T = Path("/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0")
R = T / "recipe_v9"
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
from sdf import ComputeSDFFromBoundary  # noqa: E402
from utils import get_autocast_context, recursive_to_device, set_seed  # noqa: E402

from physicsnemo import datapipes  # noqa: E402,F401  (registers ${dp:...})
from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.mesh import DomainMesh, Mesh  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402

FIELDS = ("pressure_l2", "velocity_l2", "nut_l2", "tke_l2", "omega_l2")
CONDITIONS = ("native_h", "half", "box", "wall", "full")
N_COMPANIONS = 200000
MEASURE = "_target_quadrature_measure"
### arm -> (model yaml, model overrides): the lanes' own (kernel/aerozoo/lanes.tsv, extra column minus the splits).
ARMS = {
    "isla": ("isla_volume_aerozoo", []),
    "gt": ("geotransolver_volume_aerozoo", []),
    "dtt": (
        "isla_volume_aerozoo",
        [
            "++model.query_independent=true",
            "++model.kernel_read=mean",
            "++model.kernel_read_radius=null",
            "++model.kernel_read_theta=1.0",
        ],
    ),
}


def compose_cfg(run_id: str, arm: str, out_dir: Path, extra: list[str]):
    model, arm_overrides = ARMS[arm]
    overrides = [
        f"model={model}",
        *arm_overrides,
        *extra,
        "dataset=aerozoo_volume",
        f"run_id={run_id}",
        "train_split=full.train",
        "infer_split=full.val",
        "+save_predictions=false",
        "sampling_resolution=200000",
        "precision=float32",
        f"checkpoint_dir={T / 'runs'}",
        f"output_dir={out_dir}",
    ]
    with initialize_config_dir(config_dir=str(R / "conf"), version_base=None):
        return compose(config_name="infer", overrides=overrides)


def load_reference(run_id: str) -> tuple[int | None, dict[str, dict[str, float]]]:
    """The fleet scorer's float32 validation-side infer.py rows (kernel/p200k_fleet), keyed by sample id, and their epoch."""
    path = T / "p200k_fp32" / run_id / run_id / "metrics.jsonl"
    epoch, rows = None, {}
    if not path.exists():
        return epoch, rows
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("phase") == "config":
                epoch = rec.get("epoch")
            elif rec.get("phase") == "infer_step":
                rows[rec["sample_id"]] = rec["metrics"]
    return epoch, rows


def main(run_id: str, out_dir: Path, arm: str, argv: list[str]) -> None:
    n_cases, pin_epoch, extra = None, None, []
    it = iter(argv)
    for a in it:
        if a == "--n-cases":
            n_cases = int(next(it))
        elif a == "--epoch":
            pin_epoch = int(next(it))
        else:
            extra.append(a)
    DistributedManager.initialize()
    dm = DistributedManager()
    device = dm.device
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    for c in CONDITIONS:
        (out_dir / c).mkdir(exist_ok=True)
        writers[c] = open(out_dir / c / "metrics.jsonl", "w")
    cfg = compose_cfg(run_id, arm, out_dir / "full", extra)
    set_seed(cfg.training.get("seed", None), rank=dm.rank)
    _, vl, norm, info = build_dataloaders(cfg)
    target_config = info["targets"]
    output_type = require_output_type(cfg)
    ckpt_path = resolve_checkpoint_path(cfg)
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    epoch = load_checkpoint(path=ckpt_path, models=model, device=device, epoch=pin_epoch)
    assert epoch > 0, ckpt_path
    print(
        f"[{run_id}] arm {arm}: {model.__class__.__name__} checkpoint {ckpt_path} epoch {epoch}; "
        f"query_independent={getattr(model, 'query_independent', None)} "
        f"kernel_read_theta={getattr(model, 'kernel_read_theta', None)}",
        flush=True,
    )
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
    ### The pipeline's own SDF transforms (sdf_wall + sdf_wall_normals from no_slip, sdf_slip from slip), in chain order.
    sdf_transforms = [t for t in ds.transforms if isinstance(t, ComputeSDFFromBoundary)]
    sdf_fields = [f for t in sdf_transforms for f in (t.sdf_field, t.normals_field) if f is not None]
    assert len(sdf_transforms) == 2 and len(sdf_fields) == 3, sdf_transforms
    n_spatial = _spatial_dims_of_first_sample(ds, smp)
    calc = MetricCalculator(target_config=target_config, metrics=resolve_metrics(cfg), n_spatial_dims=n_spatial)
    ref_epoch, ref = load_reference(run_id)
    gen = torch.Generator(device="cpu").manual_seed(2026)
    config_row = {
        "phase": "config",
        "model": model.__class__.__name__,
        "dataset": cfg.dataset,
        "infer_split": cfg.infer_split,
        "checkpoint": ckpt_path,
        "epoch": epoch,
        "redimensionalize": False,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    for c in CONDITIONS:
        writers[c].write(json.dumps({**config_row, "qpop_condition": c}) + "\n")

    def forward(domain, meta):
        batch = recursive_to_device(collate([(domain, meta)]), device)
        with torch.no_grad(), get_autocast_context(cfg.precision):
            output = model(**batch["forward_kwargs"])
        return normalize_output_to_tensordict(output, target_config, output_type), batch

    def take(td, n):
        """The first n points of a per-point TensorDict: batch [N] (mesh outputs) or [1, N] (tensor outputs)."""
        if n is None:
            return td
        return td[:n] if td.batch_dims == 1 else td[:, :n]

    def score(pred_td, batch, n_score):
        pred = take(pred_td, n_score)
        tgt = take(batch["targets"], n_score)
        tm = batch.get("target_measure")
        if tm is not None and n_score is not None:
            tm = tm[..., :n_score]
        md = calc(pred.float(), tgt.float(), None if tm is None else tm.float())
        return {k: v.item() for k, v in md.cpu().items()}

    def deviation(a_td, b_td, n):
        """Relative L2 between two predictions on the same first n points, per field."""
        a, b = take(a_td, n), take(b_td, n)
        return {k: float((a[k].float() - b[k].float()).norm() / b[k].float().norm().clamp_min(1e-30)) for k in b.keys()}

    def swapped(dom, n_keep, companions):
        """The case with its interior replaced by the first n_keep native points followed by `companions` (or none),
        the SDF fields recomputed by the pipeline's transforms; returns it and the native rows' largest
        |recomputed - pipeline| / max(1, max |pipeline|) over the three fields."""
        interior = dom.interior
        pd = interior.point_data[:n_keep]
        pts = interior.points[:n_keep]
        if companions is not None:
            n = companions.shape[0]
            comp = pd.apply(lambda t: t.new_zeros((n, *t.shape[1:])), batch_size=[n])
            if MEASURE in pd.keys():
                m = interior.point_data[MEASURE]
                comp[MEASURE] = m.mean(0, keepdim=True).expand(n, *m.shape[1:]).clone()
            pd = torch.cat([pd, comp])
            pts = torch.cat([pts, companions])
        new = DomainMesh(
            interior=Mesh(points=pts, point_data=pd, global_data=interior.global_data),
            boundaries=dom.boundaries,
            global_data=dom.global_data,
        )
        for t in sdf_transforms:
            new = t.apply_to_domain(new)
        new_pd = new.interior.point_data.clone()
        for t in sdf_transforms:
            ### A role the case lacks: hold its constant (the native sample's bounding-box diagonal).
            if t.allow_empty_boundary and dom.boundaries[t.boundary_name].n_cells == 0:
                new_pd[t.sdf_field] = interior.point_data[t.sdf_field][:1].expand_as(new_pd[t.sdf_field]).clone()
        new = DomainMesh(interior=new.interior.with_data(point_data=new_pd), boundaries=dom.boundaries, global_data=dom.global_data)
        sdf_err = 0.0
        for f in sdf_fields:
            a, b = new_pd[f][:n_keep].float(), interior.point_data[f][:n_keep].float()
            sdf_err = max(sdf_err, float((a - b).abs().max() / b.abs().max().clamp_min(1.0)))
        return new, sdf_err

    per_case = {c: [] for c in CONDITIONS}
    devs = []
    sanity = []
    sdf_errs = []
    idx_list = list(smp)
    print(f"[{run_id}] {len(idx_list)} validation cases" + (f"; scoring the first {n_cases}" if n_cases else ""), flush=True)
    idx_list = idx_list[:n_cases] if n_cases else idx_list
    for i, idx in enumerate(idx_list):
        t0 = time.perf_counter()
        dom, meta = ds[idx]
        sid = _sample_id(meta, idx)
        pts = dom.interior.points
        pd = dom.interior.point_data
        q = pts.shape[0]
        h = q // 2
        if i == 0:
            print(
                f"[{run_id}] first case {sid}: interior point_data {sorted(str(k) for k in pd.keys(include_nested=True, leaves_only=True))}; "
                f"boundaries {{{', '.join(f'{b}: {dom.boundaries[b].n_cells}' for b in dom.boundary_names)}}} faces",
                flush=True,
            )
        sdf = pd["sdf_wall"].reshape(-1).float().abs().clamp_min(1e-6)
        lo, hi = pts.min(0).values, pts.max(0).values
        box = lo + (hi - lo) * torch.rand(N_COMPANIONS, pts.shape[1], generator=gen).to(pts)
        wall_idx = torch.multinomial((1.0 / sdf).cpu(), N_COMPANIONS, replacement=True, generator=gen)
        wall = pts[wall_idx.to(pts.device)]
        pred_full, batch = forward(dom, meta)
        cases = {"half": swapped(dom, h, None), "box": swapped(dom, q, box), "wall": swapped(dom, q, wall)}
        errs = {c: e for c, (_, e) in cases.items()}
        sdf_errs.append(errs)
        if i == 0:
            print(f"[{run_id}] recomputed SDF fields {sdf_fields} on the native points: max relative |diff| {errs}", flush=True)
            assert max(errs.values()) <= 1e-6, f"recomputed SDF differs from the pipeline's on the native points: {errs}"
        preds = {c: forward(d, meta)[0] for c, (d, _) in cases.items()}
        res = {"full": score(pred_full, batch, None), "native_h": score(pred_full, batch, h)}
        res.update({c: score(p, batch, h) for c, p in preds.items()})
        dev = {c: deviation(p, pred_full, h) for c, p in preds.items()}
        devs.append(dev)
        for c, m in res.items():
            per_case[c].append(m)
            writers[c].write(json.dumps({"phase": "infer_step", "step": i, "sample_id": sid, "metrics": m}) + "\n")
            writers[c].flush()
        r = ref.get(sid) if ref_epoch == epoch else None
        diffs = {k: abs(res["full"][k] - r[k]) for k in FIELDS} if r else None
        sanity.append({"sample_id": sid, "found": r is not None, "abs_diff": diffs})
        line = "  ".join(f"{c}: p={res[c]['pressure_l2']:.4f} v={res[c]['velocity_l2']:.4f} nut={res[c]['nut_l2']:.4f}" for c in CONDITIONS)
        dline = "  ".join(f"{c}: {max(dev[c].values()):.1e}" for c in dev)
        print(f"[{run_id}] [{i + 1}/{len(idx_list)}] {sid} q={q}  {line}  | prediction deviation {dline}  | {time.perf_counter() - t0:.1f}s", flush=True)

    means = {c: {k: sum(m[k] for m in per_case[c]) / len(per_case[c]) for k in per_case[c][0]} for c in CONDITIONS}
    for c in CONDITIONS:
        writers[c].write(json.dumps({"phase": "infer_summary", "space": "training", "num_samples": len(per_case[c]), "metrics": dict(sorted(means[c].items()))}) + "\n")
        writers[c].close()
    swaps = ("half", "box", "wall")
    summary = {
        "run_id": run_id,
        "arm": arm,
        "epoch": epoch,
        "checkpoint": ckpt_path,
        "model_overrides": ARMS[arm][1] + extra,
        "num_samples": len(idx_list),
        "n_companions": N_COMPANIONS,
        "means": means,
        "ratio_to_native_h": {c: {f: means[c][f] / means["native_h"][f] for f in FIELDS} for c in swaps},
        "prediction_deviation_max": {c: {k: max(d[c][k] for d in devs) for k in devs[0][c]} for c in swaps},
        "prediction_deviation_max_all": {c: max(max(d[c].values()) for d in devs) for c in swaps},
        "sdf_native_max_rel_diff": {c: max(e[c] for e in sdf_errs) for c in swaps},
        "sanity_full": {
            "reference": str(T / "p200k_fp32" / run_id / run_id / "metrics.jsonl"),
            "reference_epoch": ref_epoch,
            "n_found": sum(s["found"] for s in sanity),
            "max_abs_diff": {f: max(s["abs_diff"][f] for s in sanity if s["abs_diff"]) for f in FIELDS} if any(s["abs_diff"] for s in sanity) else None,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), sys.argv[3], list(sys.argv[4:]))
