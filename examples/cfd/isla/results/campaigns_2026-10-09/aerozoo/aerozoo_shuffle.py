"""Write a row-permuted copy of the interior of AeroZoo v1.2 samples (AEROZOO benchmark, 2026-10-09).

Why: the release stores interior points in OpenFOAM cell order, which is spatially coherent (on
case-0000000000000000000-a48a547f8d4c a contiguous block of 200,000 of 5.87 M rows covers 8 to 42 percent of
the domain's bounding box, a uniform random draw 98 percent). The recipe's DomainMeshReader draws the interior
subsample as one contiguous block of rows, so without this step a training sample would be a spatial chunk.
After a uniformly random permutation of the rows, any contiguous block is an exchangeable uniform sample and
the cheap block read stays cheap.

What is written, per case, under <dst_root>/shard-NNN/<case_id>/:
  domain.pdmsh/meta.json, domain.pdmsh/_tensordict/meta.json           copied
  domain.pdmsh/_tensordict/boundaries, .../global_data                 symlinks to the release (read-only data)
  domain.pdmsh/_tensordict/interior/                                   every meta.json copied; every memmap under
                                                                        it rewritten with the same permutation
  the sidecar files (case-manifest.json, result.yaml, ...)             symlinks
  SHUFFLE.json                                                          seed, permutation hash, source path, row count
The permutation seed is derived from the case id, so the copy is reproducible without storing the permutation.

usage: python aerozoo_shuffle.py <src_root> <dst_root> <case_id> [<case_id> ...]
       python aerozoo_shuffle.py <src_root> <dst_root> --index <index.json> --start I --stop J   (cases I..J-1 of the
       sorted index; one JSON line per finished case is appended to <dst_root>/shuffle_log/<start>-<stop>.jsonl, and a
       case whose SHUFFLE.json exists is skipped, so the job is resumable)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

DTYPES = {"torch.float32": np.float32, "torch.float64": np.float64, "torch.int64": np.int64, "torch.int32": np.int32,
          "torch.uint8": np.uint8, "torch.bool": np.bool_}


def _leaf_tensors(meta_dir: Path) -> list[tuple[Path, tuple[int, ...], np.dtype]]:
    """Every memmap declared by the meta.json files under `meta_dir` (recursively): (path, shape, dtype)."""
    out = []
    for meta_path in meta_dir.rglob("meta.json"):
        meta = json.load(open(meta_path))
        for name, spec in meta.items():
            if isinstance(spec, dict) and "shape" in spec and "dtype" in spec:
                shape = tuple(int(s) for s in spec["shape"])
                out.append((meta_path.parent / f"{name}.memmap", shape, np.dtype(DTYPES[spec["dtype"]])))
    return out


def shuffle_case(src_root: Path, dst_root: Path, case_id: str, shard: str) -> dict:
    src = src_root / shard / case_id
    dst = dst_root / shard / case_id
    if (dst / "SHUFFLE.json").exists():
        return {"case_id": case_id, "status": "exists"}
    t0 = time.time()
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "domain.pdmsh" / "_tensordict").mkdir(parents=True)
    shutil.copy2(src / "domain.pdmsh" / "meta.json", tmp / "domain.pdmsh" / "meta.json")
    shutil.copy2(src / "domain.pdmsh" / "_tensordict" / "meta.json", tmp / "domain.pdmsh" / "_tensordict" / "meta.json")
    for sub in ("boundaries", "global_data"):
        os.symlink(src / "domain.pdmsh" / "_tensordict" / sub, tmp / "domain.pdmsh" / "_tensordict" / sub)
    for side in src.iterdir():
        if side.name != "domain.pdmsh":
            os.symlink(side, tmp / side.name)
    ### the interior: copy the meta.json skeleton, then rewrite every declared memmap with one permutation of the rows
    src_int = src / "domain.pdmsh" / "_tensordict" / "interior"
    dst_int = tmp / "domain.pdmsh" / "_tensordict" / "interior"
    for meta_path in src_int.rglob("meta.json"):
        rel = meta_path.relative_to(src_int)
        (dst_int / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(meta_path, dst_int / rel)
    leaves = _leaf_tensors(src_int)
    n_rows = None
    for path, shape, _ in leaves:
        if len(shape) >= 1 and shape[0] > 0 and path.exists():
            n_rows = shape[0] if n_rows is None else n_rows
            if shape[0] != n_rows:
                raise ValueError(f"{case_id}: interior leaf {path.name} has {shape[0]} rows, expected {n_rows}")
    if n_rows is None:
        raise ValueError(f"{case_id}: no interior rows")
    seed = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:8], "little") % (2**63)
    perm = np.random.default_rng(seed).permutation(n_rows)
    written = []
    for path, shape, dtype in leaves:
        if not path.exists():
            continue  # zero-size tensors have no memmap (schema v5)
        rel = path.relative_to(src_int)
        arr = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        out = np.ascontiguousarray(np.asarray(arr)[perm]) if shape[0] == n_rows else np.asarray(arr)
        out.tofile(dst_int / rel)
        written.append(str(rel))
    record = {"case_id": case_id, "shard": shard, "source": str(src), "n_rows": int(n_rows), "seed": seed,
              "permutation_sha256_head": hashlib.sha256(perm[:4096].tobytes()).hexdigest()[:16],
              "leaves": written, "seconds": round(time.time() - t0, 1), "status": "written"}
    json.dump(record, open(tmp / "SHUFFLE.json", "w"), indent=1)
    os.rename(tmp, dst)
    return record


def main(argv: list[str]) -> None:
    src_root, dst_root = Path(argv[0]), Path(argv[1])
    index = json.load(open(src_root / "index.json"))["samples"]
    if argv[2] == "--index":
        start, stop = int(argv[argv.index("--start") + 1]), int(argv[argv.index("--stop") + 1])
        case_ids = sorted(index)[start:stop]
        log_dir = dst_root / "shuffle_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = open(log_dir / f"{start}-{stop}.jsonl", "a")
    else:
        case_ids, log = argv[2:], None
    for case_id in case_ids:
        shard = f"shard-{int(index[case_id]['shard']):03d}"
        rec = shuffle_case(src_root, dst_root, case_id, shard)
        print(json.dumps(rec), flush=True)
        if log is not None:
            log.write(json.dumps(rec) + "\n")
            log.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
