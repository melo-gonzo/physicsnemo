#!/usr/bin/env python
"""Which training case produced each large per-step loss in the AeroZoo lanes (AEROZOO, 2026-10-11).

The recipe's ManifestSampler is deterministic: ``g.manual_seed(seed + epoch)`` (the 0-based epoch that
``train_loader.set_epoch`` receives), ``torch.randperm`` over the sorted train indices, ``drop_last``
truncation to a multiple of the world size (4 ranks), then ``indices[rank::4]``. The logged step counter is
1-based (``[i + 1/num_steps]``) and the logged loss is the all-rank mean of that step, so the case behind a
spike sat on ANY of the four ranks: step s of epoch e holds permutation positions 4 (s - 1) + r, r = 0..3.
This script re-derives the four candidates per spike and keeps, per lane and loss class, the candidate common
to every spike of that class (one case per class if the reproduction is right).

usage: python outliers_repro.py --out outliers_repro.json [--threshold 3.0] [--lanes az_*_isla_seed42 ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

TASK = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
)
ROOT = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets/aerozoo_v1.2_shuf_pdmsh"
)
PATTERN = "shard-*/case-*/domain.pdmsh"  # the dataset yaml's reader pattern
WORLD_SIZE = 4
STEP_RE = re.compile(r"Epoch (\d+) \[(\d+)/(\d+)\] Loss: ([0-9.]+) Step")


def reader_paths(root: Path) -> list[Path]:
    """DomainMeshReader's discovery: sorted Path objects of glob.glob(root / pattern, recursive=True)."""
    return sorted(Path(p) for p in glob.glob(str(root / PATTERN), recursive=True))


def resolve_indices(paths: list[Path], root: Path, entries: list[str]) -> list[int]:
    """datasets.resolve_manifest_indices: reader indices whose path has a parent dir named in the manifest."""
    entry_set = set(entries)
    indices = []
    for idx, full_path in enumerate(paths):
        rel = full_path.relative_to(root)
        for part in rel.parts[:-1]:
            if part in entry_set:
                indices.append(idx)
                break
        else:
            if rel.parent.name in entry_set:
                indices.append(idx)
    return sorted(indices)


def epoch_order(train_indices: list[int], seed: int, epoch: int) -> list[int]:
    """ManifestSampler.__iter__ (shuffle=True, drop_last=True) before the per-rank shard."""
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    perm = torch.randperm(len(train_indices), generator=g).tolist()
    indices = [train_indices[i] for i in perm]
    n_keep = (len(indices) // WORLD_SIZE) * WORLD_SIZE
    return indices[:n_keep]


def loss_class(loss: float) -> str:
    """A coarse label so spikes of one case group together across epochs (the model changes the digits)."""
    if loss >= 100:
        return f"~{round(loss / 10) * 10:.0f}"
    return f"~{round(loss):.0f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=3.0, help="a logged step loss at or above this is a spike")
    parser.add_argument(
        "--lanes",
        nargs="*",
        default=None,
        help="run directory names under runs/ (default: every az_<split>_<arm>_seed<NN>)",
    )
    args = parser.parse_args(argv)

    paths = reader_paths(ROOT)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    case_of = {i: p.relative_to(ROOT).parts[1] for i, p in enumerate(paths)}
    print(f"reader: {len(paths)} DomainMesh paths under {ROOT}", file=sys.stderr)

    lanes = args.lanes or sorted(
        p.name for p in (TASK / "runs").glob("az_*_seed*") if (p / "train.log").exists()
    )
    split_indices: dict[str, list[int]] = {}
    out = {"root": str(ROOT), "world_size": WORLD_SIZE, "threshold": args.threshold, "lanes": {}}
    class_votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    class_hits: dict[tuple[str, str], int] = Counter()

    for lane in lanes:
        run = TASK / "runs" / lane
        cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
        seed = int(cfg["training"]["seed"])
        train_split = cfg["train_split"]
        if train_split not in split_indices:
            entries = sorted(manifest[train_split])  # load_manifest returns the sorted entries
            split_indices[train_split] = resolve_indices(paths, ROOT, entries)
            print(f"{train_split}: {len(entries)} manifest cases -> {len(split_indices[train_split])} reader indices", file=sys.stderr)
        train_indices = split_indices[train_split]
        n_steps_expected = len(train_indices) // WORLD_SIZE

        hits = []
        seen_epochs: dict[int, list[int]] = {}
        for line in (run / "train.log").read_text().splitlines():
            m = STEP_RE.search(line)
            if m is None:
                continue
            epoch, step, n_steps, loss = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
            if n_steps != n_steps_expected:
                continue  # the validation lines carry their own step count
            if loss < args.threshold:
                continue
            if epoch not in seen_epochs:
                seen_epochs[epoch] = epoch_order(train_indices, seed, epoch)
            order = seen_epochs[epoch]
            candidates = {r: case_of[order[WORLD_SIZE * (step - 1) + r]] for r in range(WORLD_SIZE)}
            hit = {"epoch": epoch, "step": step, "n_steps": n_steps, "loss": loss, "class": loss_class(loss), "candidates": candidates}
            hits.append(hit)
            key = (train_split, hit["class"])
            class_hits[key] += 1
            for c in set(candidates.values()):
                class_votes[key][c] += 1
        out["lanes"][lane] = {
            "seed": seed,
            "train_split": train_split,
            "n_train": len(train_indices),
            "steps_per_epoch": n_steps_expected,
            "hits": hits,
        }
        print(f"{lane}: seed {seed}, {train_split}, {len(hits)} spikes >= {args.threshold}", file=sys.stderr)

    # The consensus per (split, class): the case present in every spike of that class, pooled over lanes. A
    # class with few spikes can have several such cases (4 candidates per spike); the big classes are decided
    # first and a small class takes a case already identified when it is among the unanimous candidates.
    consensus = {}
    identified: set[str] = set()
    for key in sorted(class_votes, key=lambda k: -class_hits[k]):
        votes, n = class_votes[key], class_hits[key]
        unanimous = [c for c, k in votes.most_common() if k == n]
        if len(unanimous) == 1:
            case = unanimous[0]
        elif len(unanimous) > 1:
            known = [c for c in unanimous if c in identified]
            case = known[0] if len(known) == 1 else None
        else:
            case = None
        if case is not None:
            identified.add(case)
        consensus[f"{key[0]} {key[1]}"] = {
            "train_split": key[0],
            "loss_class": key[1],
            "n_spikes": n,
            "case": case,
            "unanimous_candidates": unanimous,
            "runners_up": [[c, k] for c, k in votes.most_common(5) if k < n][:3],
        }
    consensus = dict(sorted(consensus.items()))
    out["consensus"] = consensus
    for hit_lane in out["lanes"].values():
        for hit in hit_lane["hits"]:
            c = consensus[f"{hit_lane['train_split']} {hit['class']}"]["case"]
            hit["case"] = c
            hit["rank"] = next((r for r, cc in hit["candidates"].items() if cc == c), None)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    for k, v in consensus.items():
        print(f"{k}: {v['n_spikes']} spikes, case {v['case']} (unanimous {v['unanimous_candidates']}; runners-up {v['runners_up'][:2]})")


if __name__ == "__main__":
    main()
