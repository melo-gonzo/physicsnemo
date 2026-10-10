#!/usr/bin/env python
"""Write the recipe's AeroZoo v1.2 manifest.json from the release's splits and index.

Usage::

    python build_manifest.py <release_root> <out.json>

Reads ``<release_root>/splits/splits.json`` (a flat dict of ``<split>_<part>`` lists of case ids, 22
splits x train/val/test) and ``<release_root>/index.json`` (``"samples"``: the cases that exist on disk;
545 planned cases failed or timed out and are absent). Writes the recipe's Style B manifest
(``src/datasets.py``: ``{split_key: [case_dir_name, ...]}``; case directory names equal the case ids) with
one key per part, ``<split>.<part>``, restricted to the released cases, plus ``_meta`` (release version,
the sha256 of index.json as recorded in release-v1.2.json and as computed, planned and effective counts
per part). Dotted keys are legal Hydra override values (``train_split=geometry_holdout.train``).

Pass the RELEASE root (splits.json and index.json live there); the recipe reads the row-shuffled copy,
whose case directories have the same names, so the manifest serves both. Write the output next to the
data the recipe reads (``dataset_paths.aerozoo_manifest``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PARTS = ("train", "val", "test")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def released_case_ids(index: dict) -> set[str]:
    """The case ids of ``index.json["samples"]`` (a dict keyed by id, or a list of ids or of records)."""
    samples = index["samples"]
    if isinstance(samples, dict):
        return set(map(str, samples))
    ids: set[str] = set()
    for sample in samples:
        if isinstance(sample, str):
            ids.add(sample)
            continue
        for key in ("case_id", "id", "case", "name"):
            if key in sample:
                ids.add(str(sample[key]))
                break
        else:
            raise ValueError(f"index.json sample without a case id: {sample!r}")
    return ids


def split_key(name: str) -> tuple[str, str, str]:
    """``<split>_<part>`` -> (``<split>.<part>``, split, part); the split name may contain underscores."""
    split, _, part = name.rpartition("_")
    if not split or part not in PARTS:
        raise ValueError(f"splits.json key {name!r} is not <split>_<train|val|test>")
    return f"{split}.{part}", split, part


def _find_sha256(node, path: str = "") -> dict[str, str]:
    """Every 64-hex string in a JSON tree, keyed by its dotted path."""
    found: dict[str, str] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            found.update(_find_sha256(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found.update(_find_sha256(v, f"{path}[{i}]"))
    elif isinstance(node, str) and _HEX64.match(node):
        found[path] = node
    return found


def release_meta(root: Path) -> dict:
    """Release version and the index sha256, recorded (release-v1.2.json) and computed."""
    computed = hashlib.sha256((root / "index.json").read_bytes()).hexdigest()
    meta = {"index_sha256_computed": computed}
    release_files = sorted(root.glob("release-v*.json"))
    if not release_files:
        print(
            "warning: no release-v*.json under the root; recording the computed sha256 only",
            file=sys.stderr,
        )
        return meta
    release_path = release_files[-1]
    release = json.loads(release_path.read_text())
    meta["release_file"] = release_path.name
    for key in ("version", "release", "release_version"):
        if key in release:
            meta["release_version"] = str(release[key])
            break
    digests = _find_sha256(release)
    index_digests = {
        p: d for p, d in digests.items() if "index" in p.lower()
    } or digests
    if index_digests:
        path, recorded = next(iter(index_digests.items()))
        meta["index_sha256"] = recorded
        meta["index_sha256_source"] = f"{release_path.name}:{path}"
        if recorded != computed:
            raise SystemExit(
                f"index.json sha256 {computed} does not match {recorded} recorded at {path} in {release_path.name}"
            )
    else:
        print(
            f"warning: {release_path.name} records no sha256; recording the computed one only",
            file=sys.stderr,
        )
    return meta


def build(root: Path) -> dict:
    splits = json.loads((root / "splits" / "splits.json").read_text())
    released = released_case_ids(json.loads((root / "index.json").read_text()))
    on_disk = {p.parent.name for p in root.glob("shard-*/*/domain.pdmsh")}
    if on_disk and on_disk != released:
        print(
            f"warning: index.json lists {len(released)} cases, the tree holds {len(on_disk)} "
            f"({len(released - on_disk)} indexed but absent, {len(on_disk - released)} present but unindexed)",
            file=sys.stderr,
        )
    manifest: dict = {}
    counts: dict[str, dict] = {}
    for name, ids in sorted(splits.items()):
        key, split, part = split_key(name)
        kept = sorted(str(i) for i in ids if str(i) in released)
        manifest[key] = kept
        counts.setdefault(split, {})[part] = {
            "planned": len(ids),
            "effective": len(kept),
        }
    if len(manifest) != len(splits):
        raise ValueError("two splits.json keys map to the same <split>.<part> key")
    manifest["_meta"] = {
        **release_meta(root),
        "released_cases": len(released),
        "splits": counts,
        "key_format": "<split>.<part>, entries are case directory names (= case ids) restricted to index.json",
    }
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "root",
        type=Path,
        help="the AeroZoo release root (holds splits/splits.json and index.json)",
    )
    parser.add_argument("out", type=Path, help="the manifest.json to write")
    args = parser.parse_args(argv)
    manifest = build(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    meta = manifest["_meta"]
    print(
        f"{args.out}: {len(manifest) - 1} split parts over {meta['released_cases']} released cases"
    )
    for split, parts in meta["splits"].items():
        print(
            f"  {split:<32s} "
            + "  ".join(
                f"{p}: {c['effective']}/{c['planned']}" for p, c in parts.items()
            )
        )


if __name__ == "__main__":
    main()
