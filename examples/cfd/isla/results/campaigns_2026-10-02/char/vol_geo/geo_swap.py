"""VOL-GEO (2026-10-05): how much of the DrivAerML Volume area-draw excess is the built-in query geometry?

Scores one natively trained ISLA volume checkpoint on the 48 validation cars, float32, 200,000 cells / points, under
six conditions and writes <out>/<condition>/metrics.jsonl (infer_step rows as infer.py writes them) plus summary.json:

  native        the lane's own yaml (drivaer_ml_volume_reference): reproduces evals_char/<run>/native per case
  area          the _area yaml: reproduces evals_char/<run>/area per case
  area_natq     area boundary tokens, NATIVE interior points (the query set of every swap below), built-in geometry
                from the area tokens -- the swap control: isolates the interior-draw difference from the geometry
  area_natgeo   area tokens, native queries, nearest_boundary_geometry replaced by (d, u) from the NATIVE tokens
  area_sdfgeo   area tokens, native queries, (d, u) = the stored full-mesh sdf / sdf_normals of the queries, in the
                model's convention (d = -sdf, u = sign(sdf) grad sdf; see the comment at the computation)
  native_sdfgeo native tokens and queries, the same stored (d, u) (the built-in geometry's own gap to the exact field)
Pass 1 (pass1_rawsign/) fed d = +sdf, the opposite sign to the checkpoint's convention; its sdfgeo rows are invalid.

Mirrors recipe_v8/src/infer.py main() step for step (set_seed, build_dataloaders, instantiate, load_checkpoint,
norm_stats.pt, ONE pre-read of the first sample by _spatial_dims_of_first_sample, then the sampler loop), because
the area pipeline's PoissonBiasedSubsampleMesh draws from a sequential per-transform generator: the read order is
the draw. The native and area datasets are iterated in lockstep (same seed -> same reader seed base -> the same
per-index interior point draw, asserted per case; falls back to the native interior with a flag if they differ).
The geometry swap monkeypatches physicsnemo.experimental.models.isla.model.nearest_boundary_geometry for one
forward. Also records per case the fraction of queries whose nearest boundary centroid changes between the two
draws and the RMS of (d_area - d_native) / L_ref over the near-wall queries (|sdf| < 0.01 L_ref).

usage (from recipe_v8, PYTHONPATH=code_v20:recipe_v8/src, RANK/WORLD_SIZE/LOCAL_RANK/MASTER_* set):
  python geo_swap.py <run_id> <out_dir>
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
from omegaconf import OmegaConf

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
from physicsnemo.experimental.models.isla.geometry import nearest_boundary_geometry  # noqa: E402
from physicsnemo.mesh import DomainMesh  # noqa: E402
from physicsnemo.utils import load_checkpoint  # noqa: E402

FIELDS = ("pressure_l2", "velocity_l2", "nut_l2")
CONDITIONS = ("native", "area", "area_natq", "area_natgeo", "area_sdfgeo", "native_sdfgeo")
NEAR_WALL = 0.01  # |sdf| < NEAR_WALL * L_ref


def compose_cfg(dataset: str, run_id: str, out_dir: Path):
    """The protocol_eval.sbatch command line for cl_r_vol_isla_seed4x, as infer.py would compose it."""
    overrides = [
        "model=isla_volume",
        "model.hidden=512",
        "model.n_slices=512",
        f"dataset={dataset}",
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

    cfg_n = compose_cfg("drivaer_ml_volume_reference", run_id, out_dir / "native")
    cfg_a = compose_cfg("drivaer_ml_volume_reference_area", run_id, out_dir / "area")
    ### infer.py order: set_seed, build_dataloaders, instantiate, load_checkpoint, norm_stats, pre-read, loop.
    set_seed(cfg_n.training.get("seed", None), rank=dm.rank)
    _, vl_n, norm_n, info = build_dataloaders(cfg_n)
    _, vl_a, norm_a, _ = build_dataloaders(cfg_a)
    target_config = info["targets"]
    output_type = require_output_type(cfg_n)

    ckpt_path = resolve_checkpoint_path(cfg_n)
    model = hydra.utils.instantiate(cfg_n.model, _convert_="partial").to(device)
    epoch = load_checkpoint(path=ckpt_path, models=model, device=device)
    assert epoch > 0, ckpt_path
    print(f"[{run_id}] checkpoint {ckpt_path} epoch {epoch}; model {model.__class__.__name__} eps={model.eps}", flush=True)

    norm_stats_path = Path(ckpt_path) / "norm_stats.pt"
    if norm_stats_path.exists():
        saved = torch.load(norm_stats_path, weights_only=True)
        for normalizer, vl in ((norm_n, vl_n), (norm_a, vl_a)):
            if normalizer is not None and not _norm_stats_match(normalizer.stats, saved):
                print(f"[{run_id}] norm_stats.pt differs from the yaml; using the persisted stats", flush=True)
                val_norm = find_normalizer([vl.dataset])
                for live in {id(n): n for n in (normalizer, val_norm) if n is not None}.values():
                    live.stats = saved
    model.eval()

    ds_n, smp_n, collate_n = vl_n.dataset, vl_n.sampler, vl_n.collate_fn
    ds_a, smp_a, collate_a = vl_a.dataset, vl_a.sampler, vl_a.collate_fn
    ### infer.py reads the first sample once before its loop; the area pipeline's Poisson draw is sequential.
    n_spatial = _spatial_dims_of_first_sample(ds_n, smp_n)
    _spatial_dims_of_first_sample(ds_a, smp_a)
    calc = MetricCalculator(target_config=target_config, metrics=resolve_metrics(cfg_n), n_spatial_dims=n_spatial)

    ref = {"native": load_reference(run_id, "native"), "area": load_reference(run_id, "area")}

    def score(batch, geo):
        ctx = (
            mock.patch.object(isla_model, "nearest_boundary_geometry", geo)
            if geo is not None
            else nullcontext()
        )
        with torch.no_grad(), get_autocast_context(cfg_n.precision), ctx:
            output = model(**batch["forward_kwargs"])
        if geo is not None:
            assert geo.calls == 1, geo.calls
        pred_td = normalize_output_to_tensordict(output, target_config, output_type)
        tm = batch.get("target_measure")
        md = calc(pred_td.float(), batch["targets"].float(), None if tm is None else tm.float())
        return {k: v.item() for k, v in md.cpu().items()}

    per_case = {c: [] for c in CONDITIONS}
    shifts: list[dict] = []
    sanity = {"native": [], "area": []}
    idx_n_list, idx_a_list = list(smp_n), list(smp_a)
    assert idx_n_list == idx_a_list, (idx_n_list[:5], idx_a_list[:5])
    print(f"[{run_id}] {len(idx_n_list)} validation cases", flush=True)
    eps = model.eps
    same_interior_all = True
    for i, idx in enumerate(idx_n_list):
        dom_n, meta_n = ds_n[idx]
        dom_a, meta_a = ds_a[idx]
        sid = _sample_id(meta_n, idx)
        assert sid == _sample_id(meta_a, idx), (sid, _sample_id(meta_a, idx))

        ### Interior query identity between the two draws.
        qn = dom_n.interior.points.to(device).float()
        qa = dom_a.interior.points.to(device).float()
        same_interior = qn.shape == qa.shape and bool(torch.equal(qn, qa))
        same_interior_all &= same_interior

        batch_n = recursive_to_device(collate_n([(dom_n, meta_n)]), device)
        batch_a = recursive_to_device(collate_a([(dom_a, meta_a)]), device)
        ### Swap batch: native interior (queries + targets + stored sdf), area boundary tokens.
        dom_m = DomainMesh(interior=dom_n.interior, boundaries=dom_a.boundaries, global_data=dom_n.global_data)
        batch_m = recursive_to_device(collate_a([(dom_m, meta_a)]), device)

        q = batch_m["forward_kwargs"]["prediction_points"].points.float()[None]  # (1, Q, 3) native queries
        pts_n, nh_n = boundary_geometry(batch_n["forward_kwargs"]["boundary_meshes"], eps)
        pts_a, nh_a = boundary_geometry(batch_m["forward_kwargs"]["boundary_meshes"], eps)
        with torch.no_grad():
            d_nat, u_nat = nearest_boundary_geometry(q, pts_n.float(), nh_n.float(), eps=eps)
            d_area, u_area = nearest_boundary_geometry(q, pts_a.float(), nh_a.float(), eps=eps)
            pd = batch_m["forward_kwargs"]["prediction_points"].point_data
            sdf = pd["sdf"].reshape(1, -1).float()
            sdfn = pd["sdf_normals"].reshape(1, -1, 3).float()
            grad_sdf = sdfn / sdfn.norm(dim=-1, keepdim=True).clamp_min(eps)
            ### Convention of the model's built-in channel on this dataset (pass 1, 2026-10-05): d_nat < 0 for 94.7% of
            ### the queries while the stored sdf > 0 for 98.4% (the vehicle cell normals as the model reads them point
            ### into the body), so the checkpoint was trained with d < 0 outside: the stored field enters as d = -sdf.
            ### u is the direction from the nearest boundary point to the query, = sign(sdf) * grad(sdf) for an SDF.
            sdf_sign = torch.where(sdf >= 0, 1.0, -1.0).to(sdf)
            d_sdf = -sdf
            u_sdf = sdf_sign[..., None] * grad_sdf
            L = float(dom_n.global_data["L_ref"].reshape(-1)[0])

            ### Input shift between the draws, measured on the native queries.
            from physicsnemo.experimental.models.isla.geometry import _nearest_index

            i_n = _nearest_index(q[0], pts_n[0].float())
            i_a = _nearest_index(q[0], pts_a[0].float())
            y_n, y_a = pts_n[0][i_n], pts_a[0][i_a]
            changed = (y_n - y_a).norm(dim=-1) > 1e-7 * L
            near = sdf[0].abs() < NEAR_WALL * L
            near_nat = d_nat[0].abs() < NEAR_WALL * L

            def rms(x):
                return float(x.double().pow(2).mean().sqrt()) if x.numel() else float("nan")

            shift = {
                "sample_id": sid,
                "L_ref": L,
                "n_queries": int(q.shape[1]),
                "same_interior_points": same_interior,
                "n_tokens_native": int(pts_n.shape[1]),
                "n_tokens_area": int(pts_a.shape[1]),
                "frac_nearest_token_changed": float(changed.float().mean()),
                "frac_nearest_token_changed_nearwall": float(changed[near].float().mean()) if near.any() else float("nan"),
                "n_nearwall_sdf": int(near.sum()),
                "n_nearwall_dnat": int(near_nat.sum()),
                "rms_dArea_minus_dNat_over_L_nearwall_sdf": rms((d_area[0][near] - d_nat[0][near]) / L),
                "rms_dArea_minus_dNat_over_L_nearwall_dnat": rms((d_area[0][near_nat] - d_nat[0][near_nat]) / L),
                "rms_dNat_minus_sdf_over_L_nearwall_sdf": rms((d_nat[0][near] - sdf[0][near]) / L),
                "rms_dArea_minus_sdf_over_L_nearwall_sdf": rms((d_area[0][near] - sdf[0][near]) / L),
                "rms_dArea_minus_dNat_over_L_all": rms((d_area[0] - d_nat[0]) / L),
                ### The model reads sign(s) log|s|: the shift in the channel's own units, over the near-wall queries.
                "rms_log_absdArea_over_absdNat_nearwall_sdf": rms(torch.log(d_area[0][near].abs().clamp_min(eps) / d_nat[0][near].abs().clamp_min(eps))),
                "rms_log_absdNat_over_abssdf_nearwall_sdf": rms(torch.log(d_nat[0][near].abs().clamp_min(eps) / sdf[0][near].abs().clamp_min(eps))),
                "rms_log_absdArea_over_abssdf_nearwall_sdf": rms(torch.log(d_area[0][near].abs().clamp_min(eps) / sdf[0][near].abs().clamp_min(eps))),
                "median_abs_dNat_over_L_nearwall_sdf": float(d_nat[0][near].abs().median() / L) if near.any() else float("nan"),
                "median_abs_dArea_over_L_nearwall_sdf": float(d_area[0][near].abs().median() / L) if near.any() else float("nan"),
                "median_abs_sdf_over_L_nearwall_sdf": float(sdf[0][near].abs().median() / L) if near.any() else float("nan"),
                "p10_abs_sdf_over_L_nearwall_sdf": float(torch.quantile(sdf[0][near].abs().float(), 0.10) / L) if near.any() else float("nan"),
                "p10_abs_dNat_over_L_nearwall_sdf": float(torch.quantile(d_nat[0][near].abs().float(), 0.10) / L) if near.any() else float("nan"),
                "mean_abs_dNat_over_L_nearwall_sdf": float((d_nat[0][near].abs() / L).mean()) if near.any() else float("nan"),
                "mean_abs_dArea_over_L_nearwall_sdf": float((d_area[0][near].abs() / L).mean()) if near.any() else float("nan"),
                "mean_abs_sdf_over_L_nearwall_sdf": float((sdf[0][near].abs() / L).mean()) if near.any() else float("nan"),
                "sign_agree_dNat_sdf": float((torch.sign(d_nat[0]) == torch.sign(sdf[0])).float().mean()),
                "sign_agree_dNat_minus_sdf": float((torch.sign(d_nat[0]) == torch.sign(d_sdf[0])).float().mean()),
                "frac_sdf_positive": float((sdf[0] > 0).float().mean()),
                "frac_dNat_positive": float((d_nat[0] > 0).float().mean()),
                "cos_uNat_gradSdf_nearwall_sdf": float((u_nat[0][near] * grad_sdf[0][near]).sum(-1).mean()) if near.any() else float("nan"),
                "cos_uNat_uSdf_nearwall_sdf": float((u_nat[0][near] * u_sdf[0][near]).sum(-1).mean()) if near.any() else float("nan"),
                "cos_uArea_uNat_nearwall_sdf": float((u_area[0][near] * u_nat[0][near]).sum(-1).mean()) if near.any() else float("nan"),
            }
        shift_f.write(json.dumps(shift) + "\n")
        shift_f.flush()

        res = {
            "native": score(batch_n, None),
            "area": score(batch_a, None),
            "area_natq": score(batch_m, None),
            "area_natgeo": score(batch_m, _FixedGeometry(d_nat, u_nat)),
            "area_sdfgeo": score(batch_m, _FixedGeometry(d_sdf, u_sdf)),
            "native_sdfgeo": score(batch_n, _FixedGeometry(d_sdf, u_sdf)),
        }
        shifts.append(shift)
        for c, m in res.items():
            per_case[c].append(m)
            writers[c].write(json.dumps({"phase": "infer_step", "step": i, "sample_id": sid, "metrics": m}) + "\n")
            writers[c].flush()
        for proto in ("native", "area"):
            r = ref[proto].get(sid)
            diffs = {k: res[proto][k] - r[k] for k in FIELDS} if r else None
            sanity[proto].append({"sample_id": sid, "found": r is not None, "abs_diff": {k: abs(v) for k, v in diffs.items()} if diffs else None})
        line = "  ".join(f"{c}: p={res[c]['pressure_l2']:.4f} v={res[c]['velocity_l2']:.4f} nut={res[c]['nut_l2']:.4f}" for c in CONDITIONS)
        san = "  ".join(
            f"d{proto}={max(sanity[proto][-1]['abs_diff'].values()):.1e}" if sanity[proto][-1]["abs_diff"] else f"d{proto}=NA"
            for proto in ("native", "area")
        )
        print(
            f"[{run_id}] [{i + 1}/{len(idx_n_list)}] {sid} sameQ={same_interior} chg={shift['frac_nearest_token_changed']:.3f} "
            f"rmsdd={shift['rms_dArea_minus_dNat_over_L_nearwall_sdf']:.2e}  {line}  {san}",
            flush=True,
        )

    means = {c: {k: sum(m[k] for m in per_case[c]) / len(per_case[c]) for k in per_case[c][0]} for c in CONDITIONS}
    for c in CONDITIONS:
        writers[c].write(json.dumps({"phase": "infer_summary", "space": "training", "num_samples": len(per_case[c]), "metrics": means[c]}) + "\n")
        writers[c].close()
    shift_f.close()
    san_sum = {
        proto: {
            "n_found": sum(s["found"] for s in sanity[proto]),
            "max_abs_diff": {k: max(s["abs_diff"][k] for s in sanity[proto] if s["abs_diff"]) for k in FIELDS},
        }
        for proto in ("native", "area")
    }
    excess = {k: math.log(means["area"][k] / means["native"][k]) for k in FIELDS}
    removed = {
        c: {k: (math.log(means["area"][k]) - math.log(means[c][k])) / excess[k] for k in FIELDS}
        for c in ("area_natq", "area_natgeo", "area_sdfgeo")
    }
    def shift_means(rows):
        keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return {k: sum(r[k] for r in rows) / len(rows) for k in keys}

    summary = {
        "run_id": run_id,
        "epoch": epoch,
        "num_samples": len(idx_n_list),
        "same_interior_points_all_cases": same_interior_all,
        "shift_means_first4": shift_means(shifts[:4]),
        "shift_means_all": shift_means(shifts),
        "means": means,
        "ratio_to_native": {c: {k: means[c][k] / means["native"][k] for k in FIELDS} for c in CONDITIONS},
        "log_excess_area_over_native": excess,
        "fraction_of_log_excess_removed_vs_area": removed,
        "sanity": san_sum,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
