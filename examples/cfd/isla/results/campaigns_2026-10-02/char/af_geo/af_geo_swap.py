"""AF-GEO (2026-10-06): does the built-in nearest-boundary-token geometry carry a boundary-sampling sensitivity on
AirfRANS, as it does on DrivAerML Volume (VOL-GEO, 2026-10-05)?

The campaign's area protocol is the identity on AirfRANS by construction (the airfoil's ~1,000 segments are fewer
than the cell budget, so the boundary is never subsampled), so this probe RESAMPLES the airfoil at evaluation: the
natively trained ISLA reference checkpoint (whole boundary) is scored on the 200 full_test cases, float32, 100,000
interior points, under five conditions, writing <out>/<condition>/metrics.jsonl (infer_step rows as infer.py writes
them) plus summary.json:

  native       the lane's own yaml (airfrans_volume_2d): the whole airfoil; reproduces evals_char/<run>/native per case
  subu         airfrans_volume_2d_sub400u: PoissonBiasedSubsampleMesh(400, weight_mode=uniform) on the airfoil, i.e. an
               expected 400 of its segments under the uniform-over-cells law (the mesher's refinement pattern), with
               exact Horvitz-Thompson measures; the same native interior points
  suba         airfrans_volume_2d_sub400a: the same count under the length-proportional law (uniform along the curve)
  refu / reff  airfrans_volume_2d_refu / _reff: the airfoil re-panelized by SplitLineCells (every segment split into
               collinear equal pieces: 2 / 2 uniform, 3 ahead of mid-chord / 1 behind), geometry and measure unchanged,
               only the panel pattern (count-raising: no information is lost; added after step one, 2026-10-06)
  <k>_natgeo   the variant's tokens, built-in (d, u) of the queries taken from the NATIVE (whole) boundary: the swap control
  AFGEO_VARIANTS (env, default "subu suba refu reff") selects the variants.

The two swap conditions attribute a resampling excess to the built-in channel (if they return to native) or to the
slice attention over the resampled tokens (if they do not). The reference arm's stored sdf / sdf_normals query inputs
are untouched by the boundary resampling (they are interior point data).

Mirrors recipe_v8/src/infer.py main() step for step (set_seed, build_dataloaders, instantiate, load_checkpoint,
norm_stats.pt, ONE pre-read of the first sample by _spatial_dims_of_first_sample, then the sampler loop). The three
datasets are iterated in lockstep (same seed -> same reader seed base -> the same per-index interior draw, asserted
per case). The geometry swap monkeypatches physicsnemo.experimental.models.isla.model.nearest_boundary_geometry for
one forward. Per case it also records the input shift between the whole and the resampled boundary on the native
queries: the fraction whose nearest token changes, the RMS log ratio of |d| and the direction cosine over the
near-wall queries (|sdf| < 0.01 chord; the chord is 1 m in AirfRANS).

usage (from recipe_v8, PYTHONPATH=code_v20:recipe_v8/src, RANK/WORLD_SIZE/LOCAL_RANK/MASTER_* set):
  python af_geo_swap.py <run_id> <out_dir>
"""

import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import hydra
import torch
from hydra import compose, initialize_config_dir

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

import physicsnemo.experimental.models.isla.model as isla_model  # noqa: E402
from physicsnemo import datapipes  # noqa: E402,F401  (registers ${dp:...})
from physicsnemo.distributed import DistributedManager  # noqa: E402
from physicsnemo.experimental.models.isla.geometry import _nearest_index, nearest_boundary_geometry  # noqa: E402
from physicsnemo.mesh import DomainMesh  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402

FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
import os  # noqa: E402

### Resampled variants: coarsening to an expected 400 segments under two laws (information lost), and the count-raising
### refinement under two panel patterns (SplitLineCells: uniform 2 / 2, front-loaded 3 / 1; geometry and measure unchanged).
ALL_VARIANTS = {
    "subu": "airfrans_volume_2d_sub400u",
    "suba": "airfrans_volume_2d_sub400a",
    "refu": "airfrans_volume_2d_refu",
    "reff": "airfrans_volume_2d_reff",
}
VARIANTS = {k: ALL_VARIANTS[k] for k in os.environ.get("AFGEO_VARIANTS", "subu suba refu reff").split()}
CONDITIONS = ("native",) + tuple(VARIANTS) + tuple(f"{k}_natgeo" for k in VARIANTS)
NEAR_WALL = 0.01  # |sdf| < NEAR_WALL * chord
CHORD = 1.0


def compose_cfg(dataset: str, run_id: str, out_dir: Path):
    """The protocol_eval.sbatch command line for cl_r_af_isla_seed4x, as infer.py would compose it."""
    overrides = [
        "model=isla_volume_airfrans",
        "train_split=full_train",
        "val_split=full_test",
        "+training.val_interval=10",
        "model.hidden=512",
        "model.n_slices=512",
        f"dataset={dataset}",
        f"run_id={run_id}",
        "infer_split=val",
        "+save_predictions=false",
        "sampling_resolution=100000",
        "precision=float32",
        f"checkpoint_dir={T / 'runs'}",
        f"output_dir={out_dir}",
    ]
    with initialize_config_dir(config_dir=str(R / "conf"), version_base=None):
        return compose(config_name="infer", overrides=overrides)


def boundary_geometry(boundary_meshes, eps):
    """points, unit normals exactly as ISLA.forward assembles them (sorted key order)."""
    names = sorted(boundary_meshes)
    meshes = [boundary_meshes[n] for n in names]
    points = torch.cat([m.cell_centroids for m in meshes])
    normals = torch.cat([m.cell_normals for m in meshes])
    n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(eps)
    return points[None], n_hat[None]


class _FixedGeometry:
    """Stands in for nearest_boundary_geometry for one forward: returns precomputed (d, u)."""

    def __init__(self, d, u):
        self.d, self.u = d, u
        self.calls = 0

    def __call__(self, query_points, points, normals, *, eps):
        assert query_points.shape[:2] == self.d.shape, (query_points.shape, self.d.shape)
        self.calls += 1
        return self.d.to(query_points.dtype), self.u.to(query_points.dtype)


def load_reference(run_id: str, protocol: str) -> dict[str, dict[str, float]]:
    path = T / "evals_char" / run_id / protocol / run_id / "metrics.jsonl"
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
    shift_f = open(out_dir / "shift.jsonl", "w")

    cfg_n = compose_cfg("airfrans_volume_2d", run_id, out_dir / "native")
    cfgs = {k: compose_cfg(v, run_id, out_dir / k) for k, v in VARIANTS.items()}
    ### infer.py order: set_seed, build_dataloaders, instantiate, load_checkpoint, norm_stats, pre-read, loop.
    set_seed(cfg_n.training.get("seed", None), rank=dm.rank)
    _, vl_n, norm_n, info = build_dataloaders(cfg_n)
    loaders = {k: build_dataloaders(c)[1:3] for k, c in cfgs.items()}  # k -> (val_loader, normalizer)
    target_config = info["targets"]
    output_type = require_output_type(cfg_n)

    ckpt_path = resolve_checkpoint_path(cfg_n)
    model = hydra.utils.instantiate(cfg_n.model, _convert_="partial").to(device)
    epoch = load_checkpoint(path=ckpt_path, models=model, device=device)
    assert epoch > 0, ckpt_path
    print(f"[{run_id}] checkpoint {ckpt_path} epoch {epoch}; model {model.__class__.__name__} eps={model.eps} dim={model.dim}", flush=True)

    norm_stats_path = Path(ckpt_path) / "norm_stats.pt"
    if norm_stats_path.exists():
        saved = torch.load(norm_stats_path, weights_only=True)
        for normalizer, vl in [(norm_n, vl_n)] + [(nz, vl) for vl, nz in loaders.values()]:
            if normalizer is not None and not _norm_stats_match(normalizer.stats, saved):
                print(f"[{run_id}] norm_stats.pt differs from the yaml; using the persisted stats", flush=True)
                val_norm = find_normalizer([vl.dataset])
                for live in {id(n): n for n in (normalizer, val_norm) if n is not None}.values():
                    live.stats = saved
    model.eval()

    ds_n, smp_n, collate_n = vl_n.dataset, vl_n.sampler, vl_n.collate_fn
    n_spatial = _spatial_dims_of_first_sample(ds_n, smp_n)
    for vl, _ in loaders.values():
        _spatial_dims_of_first_sample(vl.dataset, vl.sampler)
    calc = MetricCalculator(target_config=target_config, metrics=resolve_metrics(cfg_n), n_spatial_dims=n_spatial)
    ref = {"native": load_reference(run_id, "native")}

    def score(batch, geo):
        ctx = mock.patch.object(isla_model, "nearest_boundary_geometry", geo) if geo is not None else nullcontext()
        with torch.no_grad(), get_autocast_context(cfg_n.precision), ctx:
            output = model(**batch["forward_kwargs"])
        if geo is not None:
            assert geo.calls == 1, geo.calls
        pred_td = normalize_output_to_tensordict(output, target_config, output_type)
        tm = batch.get("target_measure")
        md = calc(pred_td.float(), batch["targets"].float(), None if tm is None else tm.float())
        return {k: v.item() for k, v in md.cpu().items()}

    def rms(x):
        return float(x.double().pow(2).mean().sqrt()) if x.numel() else float("nan")

    per_case = {c: [] for c in CONDITIONS}
    shifts: list[dict] = []
    sanity = []
    idx_list = list(smp_n)
    for k, (vl, _) in loaders.items():
        assert list(vl.sampler) == idx_list, k
    print(f"[{run_id}] {len(idx_list)} validation cases", flush=True)
    eps = model.eps
    for i, idx in enumerate(idx_list):
        dom_n, meta_n = ds_n[idx]
        sid = _sample_id(meta_n, idx)
        batch_n = recursive_to_device(collate_n([(dom_n, meta_n)]), device)
        q = batch_n["forward_kwargs"]["prediction_points"].points.float()[None]  # (1, Q, 2) native queries
        pts_n, nh_n = boundary_geometry(batch_n["forward_kwargs"]["boundary_meshes"], eps)
        with torch.no_grad():
            d_nat, u_nat = nearest_boundary_geometry(q, pts_n.float(), nh_n.float(), eps=eps)
            i_n = _nearest_index(q[0], pts_n[0].float())
            pd = batch_n["forward_kwargs"]["prediction_points"].point_data
            sdf = pd["sdf"].reshape(1, -1).float()
            near = sdf[0].abs() < NEAR_WALL * CHORD
        res = {"native": score(batch_n, None)}
        shift = {
            "sample_id": sid,
            "n_queries": int(q.shape[1]),
            "n_tokens_native": int(pts_n.shape[1]),
            "n_nearwall": int(near.sum()),
            "median_abs_dNat_nearwall": float(d_nat[0][near].abs().median()) if near.any() else float("nan"),
            "median_abs_sdf_nearwall": float(sdf[0][near].abs().median()) if near.any() else float("nan"),
            "rms_log_absdNat_over_abssdf_nearwall": rms(torch.log(d_nat[0][near].abs().clamp_min(eps) / sdf[0][near].abs().clamp_min(eps))),
            "sign_agree_dNat_sdf": float((torch.sign(d_nat[0]) == torch.sign(sdf[0])).float().mean()),
            "frac_dNat_positive": float((d_nat[0] > 0).float().mean()),
            "frac_sdf_positive": float((sdf[0] > 0).float().mean()),
        }
        for k, (vl, _) in loaders.items():
            dom_k, meta_k = vl.dataset[idx]
            assert _sample_id(meta_k, idx) == sid
            qk = dom_k.interior.points.to(device).float()
            same_interior = qk.shape == dom_n.interior.points.shape and bool(torch.equal(qk, dom_n.interior.points.to(device).float()))
            ### Swap batch: native interior (queries + targets + stored sdf), resampled boundary tokens.
            dom_m = DomainMesh(interior=dom_n.interior, boundaries=dom_k.boundaries, global_data=dom_n.global_data)
            batch_m = recursive_to_device(vl.collate_fn([(dom_m, meta_k)]), device)
            pts_k, nh_k = boundary_geometry(batch_m["forward_kwargs"]["boundary_meshes"], eps)
            with torch.no_grad():
                d_k, u_k = nearest_boundary_geometry(q, pts_k.float(), nh_k.float(), eps=eps)
                i_k = _nearest_index(q[0], pts_k[0].float())
                changed = (pts_n[0][i_n] - pts_k[0][i_k]).norm(dim=-1) > 1e-7 * CHORD
                meas = batch_m["forward_kwargs"]["boundary_meshes"]
                tot = float(sum(m.cell_data["_effective_measure"].sum() for m in meas.values() if "_effective_measure" in m.cell_data)) if meas else float("nan")
            shift.update(
                {
                    f"{k}_same_interior_points": same_interior,
                    f"{k}_n_tokens": int(pts_k.shape[1]),
                    f"{k}_total_measure": tot,
                    f"{k}_frac_nearest_token_changed": float(changed.float().mean()),
                    f"{k}_frac_nearest_token_changed_nearwall": float(changed[near].float().mean()) if near.any() else float("nan"),
                    f"{k}_rms_log_absd_over_absdNat_nearwall": rms(torch.log(d_k[0][near].abs().clamp_min(eps) / d_nat[0][near].abs().clamp_min(eps))),
                    f"{k}_rms_log_absd_over_abssdf_nearwall": rms(torch.log(d_k[0][near].abs().clamp_min(eps) / sdf[0][near].abs().clamp_min(eps))),
                    f"{k}_cos_u_uNat_nearwall": float((u_k[0][near] * u_nat[0][near]).sum(-1).mean()) if near.any() else float("nan"),
                    f"{k}_sign_agree_d_dNat": float((torch.sign(d_k[0]) == torch.sign(d_nat[0])).float().mean()),
                }
            )
            res[k] = score(batch_m, None)
            res[f"{k}_natgeo"] = score(batch_m, _FixedGeometry(d_nat, u_nat))
        shift_f.write(json.dumps(shift) + "\n")
        shift_f.flush()
        shifts.append(shift)
        for c, m in res.items():
            per_case[c].append(m)
            writers[c].write(json.dumps({"phase": "infer_step", "step": i, "sample_id": sid, "metrics": m}) + "\n")
            writers[c].flush()
        r = ref["native"].get(sid)
        diffs = {k: abs(res["native"][k] - r[k]) for k in FIELDS} if r else None
        sanity.append({"sample_id": sid, "found": r is not None, "abs_diff": diffs})
        line = "  ".join(f"{c}: p={res[c]['pressure_l2']:.4f} v={res[c]['velocity_l2']:.4f} nut={res[c]['nut_l2']:.4f}" for c in CONDITIONS)
        print(
            f"[{run_id}] [{i + 1}/{len(idx_list)}] {sid} tokens={shift['n_tokens_native']}/" + "/".join(str(shift[f"{k}_n_tokens"]) for k in VARIANTS)
            + " chg=" + "/".join(f"{shift[f'{k}_frac_nearest_token_changed']:.3f}" for k in VARIANTS) + f"  {line}  "
            + (f"dnative={max(diffs.values()):.1e}" if diffs else "dnative=NA"),
            flush=True,
        )

    means = {c: {k: sum(m[k] for m in per_case[c]) / len(per_case[c]) for k in per_case[c][0]} for c in CONDITIONS}
    for c in CONDITIONS:
        writers[c].write(json.dumps({"phase": "infer_summary", "space": "training", "num_samples": len(per_case[c]), "metrics": means[c]}) + "\n")
        writers[c].close()
    shift_f.close()
    excess = {k: {f: math.log(means[k][f] / means["native"][f]) for f in FIELDS} for k in VARIANTS}
    removed = {
        k: {f: (math.log(means[k][f]) - math.log(means[f"{k}_natgeo"][f])) / excess[k][f] if excess[k][f] != 0 else float("nan") for f in FIELDS}
        for k in VARIANTS
    }

    def shift_means(rows):
        keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return {k: sum(r[k] for r in rows) / len(rows) for k in keys}

    summary = {
        "run_id": run_id,
        "epoch": epoch,
        "num_samples": len(idx_list),
        "same_interior_points_all_cases": all(all(s[f"{k}_same_interior_points"] for k in VARIANTS) for s in shifts),
        "shift_means_all": shift_means(shifts),
        "means": means,
        "ratio_to_native": {c: {f: means[c][f] / means["native"][f] for f in FIELDS} for c in CONDITIONS},
        "log_excess_over_native": excess,
        "fraction_of_log_excess_removed_by_native_geometry": removed,
        "sanity_native": {
            "n_found": sum(s["found"] for s in sanity),
            "max_abs_diff": {f: max(s["abs_diff"][f] for s in sanity if s["abs_diff"]) for f in FIELDS} if any(s["abs_diff"] for s in sanity) else None,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
