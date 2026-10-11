#!/usr/bin/env python
"""Pin the loss formula on the identified cases with the recipe's own pipeline (AEROZOO, 2026-10-11).

Builds the dataset from the split's yaml through ``datasets.build_dataset`` (the reader, NonDimensionalizeByMetadata,
NormalizeMeshFields and the rest of the chain; ``sampling_resolution`` unset so the whole interior comes through;
the Warp-backed ``ComputeSDFFromBoundary`` transforms are dropped because this runs on a CPU node and they do not
touch the targets), then for each case:

  1. per-field mean |z| and max |z| of the standardized targets, to compare with outliers_scan.py's numpy chain;
  2. the recipe ``LossCalculator`` (huber, delta 1, channel-normalized) with a zero prediction on the whole case;
  3. the same on a 200,000-row contiguous block that contains the case's worst row, divided by the 4 ranks whose
     mean the training log prints, to compare with the logged spike.

usage: python outliers_verify.py --split nonbox --cases case-... [case-...] --out verify_nonbox.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

TASK = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
)
RECIPE = TASK / "recipe_v9"
sys.path.insert(0, str(RECIPE / "src"))

from datasets import build_dataset, load_dataset_config  # noqa: E402
from loss import LossCalculator  # noqa: E402
from physicsnemo.datapipes.protocols import DatasetBase  # noqa: E402

YAMLS = {
    "full": "aerozoo_volume",
    "nonbox": "aerozoo_volume_nonbox",
    "blobs": "aerozoo_volume_blobs",
    "reynolds": "aerozoo_volume_reynolds",
}
WORLD_SIZE = 4
BLOCK = 200_000


def dataset_for(split: str):
    ds = load_dataset_config(RECIPE / "datasets" / f"{YAMLS[split]}.yaml")
    transforms = [
        t
        for t in OmegaConf.to_container(ds.pipeline.transforms, resolve=False)
        if "ComputeSDFFromBoundary" not in t["_target_"]
    ]
    cfg = OmegaConf.merge(ds, {"sampling_resolution": None, "pipeline": {"transforms": transforms}})
    targets = OmegaConf.to_container(cfg.targets, resolve=True)
    return build_dataset(cfg, device=None, num_workers=1), targets


def zero_loss(calc: LossCalculator, fields: dict[str, torch.Tensor], rows: slice | None = None) -> dict:
    target = TensorDict({k: (v if rows is None else v[rows]) for k, v in fields.items()})
    pred = TensorDict({k: torch.zeros_like(v) for k, v in target.items()})
    total, td = calc(pred, target)
    return {"total": float(total), **{k: float(v) for k, v in td.items() if k != "loss/total"}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=sorted(YAMLS))
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    t0 = time.perf_counter()
    dataset, targets = dataset_for(args.split)
    calc = LossCalculator(target_config=targets, loss_type="huber")
    paths = dataset.reader._paths
    index_of = {p.parent.name: i for i, p in enumerate(paths)}
    print(f"{args.split}: {len(paths)} reader paths, loss {calc}", file=sys.stderr, flush=True)
    out = {"split": args.split, "yaml": YAMLS[args.split], "sdf_transforms_dropped": True, "cases": {}}
    try:
        for case in args.cases:
            t1 = time.perf_counter()
            idx = index_of[case]
            domain, _meta = dataset[idx]
            pd = domain.interior.point_data
            fields = {name: torch.as_tensor(pd[name]).double() for name in targets}
            n = next(iter(fields.values())).shape[0]
            stats = {}
            worst_row, worst_abs = 0, -1.0
            for name, z in fields.items():
                a = z.abs()
                per_row = a if a.ndim == 1 else a.max(dim=1).values
                row = int(per_row.argmax())
                stats[name] = {"mean_abs_z": float(a.mean()), "max_abs_z": float(a.max()), "argmax_row": row}
                if float(per_row[row]) > worst_abs:
                    worst_abs, worst_row = float(per_row[row]), row
            start = max(0, min(worst_row - BLOCK // 2, n - BLOCK))
            block = zero_loss(calc, fields, slice(start, start + BLOCK))
            out["cases"][case] = {
                "reader_index": idx,
                "n_points": n,
                "fields": stats,
                "worst_row": worst_row,
                "worst_abs_z": worst_abs,
                "zero_pred_loss_whole_case": zero_loss(calc, fields),
                "zero_pred_loss_block_with_worst_row": block,
                "block_rows": [start, start + BLOCK],
                "logged_equivalent_block_over_4_ranks": block["total"] / WORLD_SIZE,
                "seconds": round(time.perf_counter() - t1, 1),
            }
            print(
                f"{case}: {n:,} points, worst |z| {worst_abs:.4g} at row {worst_row}; zero-pred whole case "
                f"{out['cases'][case]['zero_pred_loss_whole_case']['total']:.4g}, block with the worst row "
                f"{block['total']:.5g} -> /4 ranks {block['total'] / WORLD_SIZE:.5g}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        DatasetBase.close(dataset)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
