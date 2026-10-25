# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Record the golden cases, or regenerate the committed CI golden.

A/B record (one process per version; custom ops register process-wide)::

    python -m test.experimental.models.isla.goldens.record --device cpu --out DIR [--groups G-GEO ...]

writes ``DIR/record.json`` (the platform fingerprint, which file and md5 of every
ISLA module ran, the registered ``physicsnemo::isla_*`` ops, and per case, variant
and run the int-view SHA-256, dtype and shape of every input, output and gradient)
and ``DIR/tensors.pt`` (the tensors of the first run, and of any later run that
differs from it). CPU records run on one thread. On CUDA every case runs
``--repeats`` times in-process (5 by default), so that compare.py can judge the
outputs known to vary run to run against their family.

Regenerate the committed data (CPU, one thread)::

    python -m test.experimental.models.isla.goldens.record --regenerate FILE ... | all

writes the named files of ``data/`` (``all``: every one): ``goldens_v1.pt`` and its
manifest ``goldens_v1.json``, and the goldens and reference checkpoints of
``test_isla_release.py`` (its ``WRITERS``). Nothing else writes them. A bare
``--regenerate`` is refused, so that no one rewrites the older goldens by accident.
"""

import argparse
import datetime
import hashlib
import json
import lzma
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from test.experimental.models.isla.goldens import cases as C

DATA = Path(__file__).resolve().parents[1] / "data"
GOLDEN = DATA / "goldens_v1.pt"
MANIFEST = DATA / "goldens_v1.json"
#: Elements of each float tensor the committed golden stores (evenly spaced over the
#: flattened tensor), beside its summaries over every element.
SAMPLES = 16
_INT_VIEW = {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}


def tensor_hash(t: torch.Tensor) -> str:
    """SHA-256 of the tensor's bytes viewed as integers of its width (bitwise: -0.0
    and 0.0 differ, every NaN payload counts), in C order."""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
    elif t.is_floating_point() or t.is_complex():
        t = t.view(_INT_VIEW[t.element_size()])
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "cpu part")):
                return line.split(":", 1)[1].strip()
    except Exception:  # pragma: no cover - informational only
        return platform.processor()


def fingerprint(device: str) -> dict:
    """The platform a record ran on: library versions, CPU, BLAS/LAPACK, threads, GPU."""
    info = dict(
        re.findall(r"\b(BLAS_INFO|LAPACK_INFO)=([^,\s]*)", torch.__config__.show())
    )
    fp = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "blas": info.get("BLAS_INFO", "?"),
        "lapack": info.get("LAPACK_INFO", "?"),
        "threads": torch.get_num_threads(),
        "device": device,
    }
    for lib in ("warp", "triton", "numpy"):
        try:
            fp[lib] = __import__(lib).__version__
        except Exception:
            fp[lib] = None
    if device == "cuda":
        fp["gpu"] = torch.cuda.get_device_name()
        fp["cuda"] = torch.version.cuda
        fp["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    return fp


def module_versions() -> dict:
    """File and md5 of every imported ISLA module and every module :data:`cases.TARGETS`
    names, and the ``physicsnemo::isla_*`` custom ops registered in this process."""
    names = {C.TARGETS[k].partition(":")[0] for k in C.TARGETS}
    names |= {n for n in sys.modules if n.startswith(C.ISLA)}
    mods = {}
    for n in sorted(names):
        f = getattr(sys.modules.get(n), "__file__", None)
        if f:
            mods[n] = {
                "file": f,
                "md5": hashlib.md5(
                    Path(f).read_bytes(), usedforsecurity=False
                ).hexdigest(),
            }
    ops = sorted(
        n
        for n in torch._C._dispatch_get_all_op_names()
        if n.startswith("physicsnemo::isla_")
    )
    return {"modules": mods, "ops": ops}


def _key(case: C.Case, device: str, variant: str) -> str:
    return f"{case.id}@{device}:{variant}"


def record(cases, device: str, repeats: int = 1, log=print) -> dict:
    """Run ``cases`` on ``device`` (each of its variants, ``repeats`` times on the same
    inputs) and return the record: ``{"fingerprint", "versions", "cases": {key: {...}},
    "tensors": {key: {name: [tensor per distinct run]}}}``."""
    rec = {
        "fingerprint": fingerprint(device),
        "cases": {},
        "tensors": {},
        "skipped": {},
        "errors": {},
    }
    for case in cases:
        why = C.available(case, device)
        variants = case.variants.get(device, ())
        if why or not variants:
            rec["skipped"][case.id] = why or f"no {device} variant"
            continue
        inputs = case.build()
        for variant in variants:
            key = _key(case, device, variant)
            ### On CUDA a case without a declared family needs two runs to show that it
            ### repeats bitwise; a family gets every repeat.
            n_runs = repeats if device == "cpu" or case.family else min(repeats, 2)
            try:
                runs = [
                    C.run_case(case, device, variant, inputs) for _ in range(n_runs)
                ]
            except Exception:
                rec["errors"][key] = traceback.format_exc()
                log(f"ERROR {key}:\n{rec['errors'][key]}")
                continue
            meta, store = {}, {}
            for name, t0 in runs[0].items():
                hashes = [tensor_hash(r[name]) for r in runs]
                meta[name] = {
                    "dtype": str(t0.dtype).removeprefix("torch."),
                    "shape": list(t0.shape),
                    "sha256": hashes,
                }
                store[name] = [t0] + [
                    r[name] for r, h in zip(runs[1:], hashes[1:]) if h != hashes[0]
                ]
            rec["cases"][key] = {
                "group": case.group,
                "variant": variant,
                "device": device,
                "repeats": n_runs,
                "kept": [n for n in meta if case.matches(case.kept, n)],
                "weights": [n for n in meta if case.matches(case.weights, n)],
                "family": [n for n in meta if case.matches(case.family, n)],
                "family_rtol": {
                    n: next(
                        v for q, v in case.family_rtol.items() if case.matches((q,), n)
                    )
                    for n in meta
                    if case.matches(case.family_rtol, n)
                },
                "tensors": meta,
            }
            rec["tensors"][key] = store
            log(f"recorded {key}: {len(meta)} tensors")
    rec["versions"] = module_versions()
    return rec


def save(rec: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "record.json").write_text(
        json.dumps({k: v for k, v in rec.items() if k != "tensors"}, indent=1)
    )
    torch.save(rec["tensors"], out / "tensors.pt")


def load(path: Path) -> dict:
    rec = json.loads((path / "record.json").read_text())
    rec["tensors"] = torch.load(path / "tensors.pt", weights_only=True)
    return rec


def print_versions(rec: dict, tag: str = "", log=print) -> None:
    """Which version of each module ran, and the ops registered (as audit_cuts.py prints)."""
    fp = rec["fingerprint"]
    log(f"{tag}platform: " + ", ".join(f"{k}={v}" for k, v in fp.items()))
    for n, m in rec["versions"]["modules"].items():
        log(f"{tag}module {n}: {m['file']} md5 {m['md5']}")
    log(f"{tag}ops registered: {rec['versions']['ops']}")


# --------------------------------------------------------------------------- the committed golden


def _weights(n: int) -> torch.Tensor:
    """Fixed weights in [-1/2, 1/2) per flat index (exact in float64 on every platform)."""
    return torch.frac(torch.arange(n, dtype=torch.float64) * 0.6180339887498949) - 0.5


def sample_index(n: int, k: int = SAMPLES) -> torch.Tensor:
    """``min(n, k)`` evenly spaced indices of a flattened tensor of ``n`` elements."""
    return torch.unique(
        torch.linspace(0, max(n - 1, 0), min(n, k), dtype=torch.float64).round().long()
    )


def is_output(name: str) -> bool:
    return name.startswith(("out.", "grad."))


def summarize(
    t: torch.Tensor, exact: bool, samples: int = SAMPLES, chunk: int | None = None
) -> dict:
    """What the golden keeps of a tensor: dtype and shape; the SHA-256 of integer and
    boolean tensors, and of every tensor with ``exact`` (the inputs); for floats the
    largest finite magnitude ``scale``, the sum of magnitudes ``l1`` and a fixed
    weighted sum ``dot`` of every finite element (in float64), the count of non-finite
    elements, and ``samples`` elements (:data:`SAMPLES` unless the case says fewer);
    with ``chunk``, that length and the sums ``chunk_sums`` of the finite elements
    over each run of ``chunk`` elements of the flattened tensor."""
    t = t.detach().cpu().contiguous()
    entry = {"dtype": str(t.dtype).removeprefix("torch."), "shape": list(t.shape)}
    if exact or not t.is_floating_point():
        entry["sha256"] = tensor_hash(t)
    if not t.is_floating_point():
        x = t.reshape(-1).double()
        entry.update(sum=float(x.sum()), dot=float((x * _weights(x.numel())).sum()))
        return entry
    x = t.reshape(-1).double()
    finite = torch.isfinite(x)
    xz = torch.where(finite, x, 0.0)
    entry.update(
        scale=float(xz.abs().max()) if x.numel() else 0.0,
        l1=float(xz.abs().sum()),
        dot=float((xz * _weights(x.numel())).sum()),
        nonfinite=int((~finite).sum()),
        samples=x[sample_index(x.numel(), samples)].tolist(),
    )
    if chunk:
        entry.update(chunk=chunk, chunk_sums=[float(c.sum()) for c in xz.split(chunk)])
    return entry


def regenerate(log=print) -> dict:
    """Record every case's first CPU variant on one thread and write the committed
    golden and its manifest."""
    torch.set_num_threads(1)
    golden = {}
    for case in C.CASES:
        variant = case.variants["cpu"][0]
        tensors = C.run_case(case, "cpu", variant)
        golden[case.id] = {
            name: summarize(
                t, not is_output(name), case.n_samples(name), case.chunk_length(name)
            )
            for name, t in tensors.items()
        }
        log(f"golden {case.id} ({variant}): {len(tensors)} tensors")
    try:
        git = shutil.which("git")
        commit = subprocess.run(  # noqa: S603 - git and its arguments are constants
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, cwd=DATA
        ).stdout.strip()
        ### The code a golden depends on: the model's directory, every module TARGETS
        ### names (the library's, after LIB-MOVE), and this harness with its cases.
        paths = {
            str(Path(sys.modules[C.ISLA].__file__).parent),
            str(Path(__file__).parent),
        }
        paths |= {v["file"] for v in module_versions()["modules"].values()}
        status = subprocess.run(  # noqa: S603 - git and its arguments are constants
            [git, "status", "--porcelain", "--", *sorted(paths)],
            capture_output=True,
            text=True,
            cwd=DATA,
        )
        dirty = bool(status.stdout.strip())
    except Exception:  # pragma: no cover
        commit, dirty = "?", None
    manifest = {
        "format": 1,
        "commit": commit,
        "code_modified": dirty,
        "harness": {
            f.name: hashlib.md5(f.read_bytes(), usedforsecurity=False).hexdigest()
            for f in sorted(Path(__file__).parent.glob("*.py"))
        },
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": fingerprint("cpu"),
        "versions": module_versions()["modules"],
        "samples": SAMPLES,
        "cases": len(golden),
        "tensors": sum(len(v) for v in golden.values()),
    }
    ### The case entries as xz-compressed JSON (exact float64 reprs; a third of the size
    ### of a pickle of the same entries), in a weights-only loadable file.
    payload = lzma.compress(json.dumps(golden).encode(), preset=9)
    torch.save({"format": 1, "manifest": manifest, "cases_json_xz": payload}, GOLDEN)
    MANIFEST.write_text(
        json.dumps(
            {
                **manifest,
                "versions": {k: v["md5"] for k, v in manifest["versions"].items()},
            },
            indent=1,
        )
        + "\n"
    )
    log(f"wrote {GOLDEN} ({GOLDEN.stat().st_size} bytes) and {MANIFEST}")
    return golden


def load_golden(path: Path = GOLDEN) -> dict:
    """The committed golden: ``{"manifest": ..., "cases": {case id: {tensor: entry}}}``."""
    blob = torch.load(path, weights_only=True)
    return {
        "manifest": blob["manifest"],
        "cases": json.loads(lzma.decompress(blob["cases_json_xz"])),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--groups",
        nargs="*",
        choices=C.GROUPS,
        help=f"case groups (default all): {' '.join(C.GROUPS)}",
    )
    p.add_argument("--cases", nargs="*", help="fnmatch patterns of case names")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, help="record directory")
    p.add_argument(
        "--repeats",
        type=int,
        help="runs per case and variant (default 1 on CPU, 5 on CUDA)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU threads (default 1: CPU records are bitwise per thread count)",
    )
    p.add_argument(
        "--regenerate",
        nargs="*",
        metavar="FILE",
        help="write the named files of data/, or `all` for every one (goldens_v1.pt "
        "with its manifest, and the files of test_isla_release.py); a bare "
        "--regenerate is refused",
    )
    args = p.parse_args(argv)
    if args.regenerate is not None:
        from test.experimental.models.isla import test_isla_release as release

        files = [GOLDEN.name, *release.WRITERS]
        if not args.regenerate:
            p.error(f"--regenerate needs the files to write ({files}) or `all`")
        names = set(args.regenerate)
        if "all" in names:
            names = (names - {"all"}) | set(files)
        unknown = names - set(files)
        if unknown:
            p.error(f"unknown data files {sorted(unknown)}; the files are {files}")
        torch.set_num_threads(1)
        if GOLDEN.name in names:
            regenerate()
        release.regenerate(names)
        return 0
    if args.out is None:
        p.error("--out is required (or --regenerate)")
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    repeats = args.repeats or (1 if args.device == "cpu" else 5)
    cases = C.select(args.groups, args.cases)
    unmatched = [q for q in args.cases or () if not C.select(args.groups, [q])]
    if unmatched or not cases:
        p.error(
            f"no case matches {unmatched or args.cases} in the groups {args.groups or C.GROUPS}"
        )
    rec = record(cases, args.device, repeats)
    save(rec, args.out)
    print_versions(rec)
    for cid, why in rec["skipped"].items():
        print(f"skipped {cid}: {why}")
    for key in rec["errors"]:
        print(f"ERROR {key} (traceback in record.json)")
    print(f"wrote {args.out}: {len(rec['cases'])} case variants, pid {os.getpid()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
