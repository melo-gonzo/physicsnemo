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

"""Committed characterization goldens of ISLA's modules (the regression guard).

Every case of ``goldens/cases.py`` (G-FRAME, G-SAMP, G-FOREST, G-WALK, G-SEG, G-GEO,
G-READ, G-BLOCKS, G-HEADS) is run here and compared with ``data/goldens_v1.pt``,
which was recorded on one CPU (its manifest ``data/goldens_v1.json`` names it) and is
written only by ``python -m test.experimental.models.isla.goldens.record
--regenerate``; a missing golden fails. The reading, per tensor:

- inputs (``in.*`` and the parameters): bitwise (their SHA-256), so a change of a
  fixture or of a library primitive feeding one is reported as such, not as a
  change of the module under test;
- integer and boolean outputs (forest tables, walk lists, ``PairPlan``, kept cells,
  search flags): exact (SHA-256);
- float outputs: every stored element and the summaries over every element (largest
  magnitude, sum of magnitudes, a fixed weighted sum) within ``rtol * max|golden|``
  per element, the summaries within the bound that per-element bound implies
  (``n * rtol * max|golden|``, plus 1e-14 of the sum of magnitudes for float64
  summation order). ``rtol`` is 1e-12 in float64, which holds across BLAS builds and
  CPU architectures; float32 and bfloat16 bounds are stated at :data:`CPU_RTOL`.

The CUDA variants (the fast and fused kernels, and the references on CUDA) are
compared with the same CPU golden at :data:`CUDA_RTOL`, integers exact where the code
claims the fast path is the reference's bitwise.
"""

import fnmatch
import math

import pytest
import torch

from test.experimental.models.isla.goldens import cases as C
from test.experimental.models.isla.goldens import compare as K
from test.experimental.models.isla.goldens import record as R

#: Relative tolerance (of ``max|golden|``) of the CPU variants against the golden, by
#: dtype. float64: 1e-12, the plan's bound across BLAS builds, CPU architectures and
#: thread counts (measured: at most 1.3e-13, a slice block's prototype-gain gradient,
#: a scalar reduction over every token and slice, between one and four threads).
#: float32: the same bound scaled by the ratio of the unit roundoffs, 2**29 (the
#: computations are the same, so their conditioning is; measured at most 2.2e-5, the
#: same gradient). bfloat16 (the segment-softmax read's bf16 gradients, computed in
#: float32 and rounded once): two bf16 ulps of the largest element, 2**-6.
CPU_RTOL = {"float64": 1e-12, "float32": 1e-12 * 2**29, "bfloat16": 2**-6}
#: The same for the CUDA variants (TF32 off).
CUDA_RTOL = dict(CPU_RTOL)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA paths need a GPU"
)


@pytest.fixture(scope="module")
def golden():
    if not R.GOLDEN.exists():
        pytest.fail(
            f"{R.GOLDEN} is missing: regenerate it with `python -m {R.__name__} --regenerate`"
        )
    return R.load_golden()


def _float_failures(name, entry, t, rtol) -> list[str]:
    x = t.reshape(-1).double()
    finite = torch.isfinite(x)
    xz = torch.where(finite, x, 0.0)
    n, scale = x.numel(), entry["scale"]
    tol = rtol * scale
    bad = []
    if int((~finite).sum()) != entry["nonfinite"]:
        bad.append(
            f"{name}: {int((~finite).sum())} non-finite elements, golden {entry['nonfinite']}"
        )
    ref = torch.tensor(entry["samples"], dtype=torch.float64)
    got = x[R.sample_index(n)]
    same_nan = ref.isnan() == got.isnan()
    err = torch.where(
        ref.isfinite() & got.isfinite(),
        (got - ref).abs(),
        torch.where(got == ref, 0.0, math.inf),
    )
    worst = float(err.max()) if err.numel() else 0.0
    if not bool(same_nan.all()) or worst > tol:
        bad.append(
            f"{name}: samples off by {worst:.3e} > {tol:.3e} ({rtol:g} of max|golden| {scale:.3e})"
        )
    slack = n * tol + 1e-14 * entry["l1"]
    stats = {
        "max": (float(xz.abs().max()) if n else 0.0, scale, tol),
        "l1": (float(xz.abs().sum()), entry["l1"], slack),
        "dot": (float((xz * R._weights(n)).sum()), entry["dot"], slack),
    }
    for k, (v, r, b) in stats.items():
        if abs(v - r) > b:
            bad.append(
                f"{name}: {k} {v!r} against golden {r!r}, off by {abs(v - r):.3e} > {b:.3e}"
            )
    return bad


def check(
    golden_case: dict,
    tensors: dict,
    rtol: dict,
    exact_inputs: bool = True,
    overrides=None,
) -> tuple[list, dict]:
    """The failures of ``tensors`` against one case of the golden, and the largest
    relative error (of ``max|golden|``) seen per float dtype."""
    bad, worst = [], {}
    if golden_case.keys() != tensors.keys():
        missing, extra = (
            sorted(golden_case.keys() - tensors.keys()),
            sorted(tensors.keys() - golden_case.keys()),
        )
        bad.append(f"tensors differ from the golden's: missing {missing}, new {extra}")
    for name in sorted(golden_case.keys() & tensors.keys()):
        entry, t = golden_case[name], tensors[name]
        if [str(t.dtype).removeprefix("torch."), list(t.shape)] != [
            entry["dtype"],
            entry["shape"],
        ]:
            bad.append(
                f"{name}: {t.dtype}{list(t.shape)} against golden {entry['dtype']}{entry['shape']}"
            )
            continue
        if not R.is_output(name):
            if exact_inputs and R.tensor_hash(t) != entry["sha256"]:
                bad.append(
                    f"{name}: the input differs from the golden's (fixture drift, not the module under test)"
                )
            continue
        if "sha256" in entry:
            if R.tensor_hash(t) != entry["sha256"]:
                x = t.reshape(-1).double()
                bad.append(
                    f"{name}: integers differ from the golden's (sum {float(x.sum())!r} against {entry['sum']!r}, "
                    f"weighted sum {float((x * R._weights(x.numel())).sum())!r} against {entry['dot']!r})"
                )
            continue
        tol = rtol[entry["dtype"]]
        for pattern, value in (overrides or {}).items():
            if fnmatch.fnmatchcase(name, pattern):
                tol = value
        if tol is None:  # not compared here (the case says why)
            continue
        bad += _float_failures(name, entry, t, tol)
        if entry["scale"] > 0 and t.numel():
            ref = torch.tensor(entry["samples"], dtype=torch.float64)
            got = t.reshape(-1).double()[R.sample_index(t.numel())]
            ok = ref.isfinite() & got.isfinite()
            diff = (got - ref).abs()[ok]
            rel = (float(diff.max()) if diff.numel() else 0.0) / entry["scale"]
            worst[entry["dtype"]] = max(worst.get(entry["dtype"], 0.0), rel)
    return bad, worst


def _params(device):
    out = []
    for case in C.CASES:
        for variant in case.variants.get(device, ()):
            marks = [cuda_only] if device == "cuda" else []
            out.append(
                pytest.param(case, variant, id=f"{case.id}:{variant}", marks=marks)
            )
    return out


def test_the_golden_holds_every_case_and_its_manifest():
    """The committed golden has exactly the cases of cases.py (none missing, none
    stale) and a manifest naming its commit, platform and module versions."""
    g = R.load_golden()
    assert set(g["cases"]) == {c.id for c in C.CASES}
    m = g["manifest"]
    assert {"commit", "date", "platform", "versions"} <= m.keys()
    assert m["platform"]["device"] == "cpu" and m["platform"]["threads"] == 1


@pytest.mark.parametrize("case, variant", _params("cpu"))
def test_cpu_matches_the_golden(case, variant, golden):
    """Each CPU variant of the case (the reference, and the fast build and walk, which
    claim the reference's results bitwise) against the committed golden: inputs and
    integers exact, floats within :data:`CPU_RTOL` of ``max|golden|``."""
    bad, worst = check(
        golden["cases"][case.id],
        C.run_case(case, "cpu", variant),
        CPU_RTOL,
        overrides=case.tol,
    )
    print("largest relative error of the samples per dtype:", worst)  # shown with -rP
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("case, variant", _params("cuda"))
def test_cuda_matches_the_cpu_golden(case, variant, golden):
    """Each CUDA variant (kernels auto: the fast build, walk and geometry search, the
    Triton and fused kernels; and the references on CUDA) against the committed CPU
    golden: inputs and integers exact, floats within :data:`CUDA_RTOL` of
    ``max|golden|`` (or the case's own stated bound, ``Case.cuda_tol``); the draw's
    kept cells exact and its weights at the float64 bound."""
    why = C.available(case, "cuda")
    if why:
        pytest.skip(why)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        tensors = C.run_case(case, "cuda", variant)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    bad, worst = check(
        golden["cases"][case.id],
        tensors,
        CUDA_RTOL,
        overrides={**case.tol, **case.cuda_tol},
    )
    print("largest relative error of the samples per dtype:", worst)  # shown with -rP
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------- the A/B harness


def _quick_cases():
    return C.select(["G-HEADS", "G-SAMP", "G-SEG"])


def test_the_ab_harness_passes_a_rerun_and_catches_a_one_ulp_change(tmp_path):
    """Recorded twice in one process on CPU (one thread), the harness reads PASS; the
    records survive a save and load; one output moved by one ulp reads FAIL, a missing
    case FAIL, a flipped kept cell on CUDA INCONCLUSIVE (on CPU FAIL), weights moved
    within 1e-15 relative on CUDA PASS, and a declared family within its spread PASS."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        a = R.record(_quick_cases(), "cpu", log=lambda *_: None)
        b = R.record(_quick_cases(), "cpu", log=lambda *_: None)
    finally:
        torch.set_num_threads(threads)
    quiet = lambda *_: None  # noqa: E731
    assert K.compare(a, b, log=quiet) == K.PASS
    R.save(a, tmp_path / "a")
    a = R.load(tmp_path / "a")
    assert K.compare(a, b, log=quiet) == K.PASS

    def edited(rec, key, name, fn, device=None):
        import copy

        rec = copy.deepcopy(rec)
        t = fn(rec["tensors"][key][name][0])
        rec["tensors"][key][name][0] = t
        rec["cases"][key]["tensors"][name]["sha256"][0] = R.tensor_hash(t)
        if device:
            for c in rec["cases"].values():
                c["device"] = device
        return rec

    key = "G-SEG/segment_softmax_read_f64@cpu:reference"
    ulp = lambda t: torch.nextafter(t, torch.full_like(t, math.inf))  # noqa: E731
    assert K.compare(a, edited(b, key, "out.read", ulp), log=quiet) == K.FAIL
    missing = {**b, "cases": {k: v for k, v in b["cases"].items() if k != key}}
    assert K.compare(a, missing, log=quiet) == K.FAIL
    draw = "G-SAMP/sample_boundary_cells_3d@cpu:reference"
    kept = "out.kept_300_0.37"
    flip = lambda t: torch.cat([t[:-1], t[-1:] + 1])  # noqa: E731
    assert K.compare(a, edited(b, draw, kept, flip), log=quiet) == K.FAIL
    a_cuda = edited(a, draw, kept, lambda t: t, device="cuda")
    assert (
        K.compare(a_cuda, edited(b, draw, kept, flip, device="cuda"), log=quiet)
        == K.INCONCLUSIVE
    )
    weights = "out.weights_300_0.37"
    assert (
        K.compare(a_cuda, edited(b, draw, weights, ulp, device="cuda"), log=quiet)
        == K.PASS
    )
    assert K.compare(a, edited(b, draw, weights, ulp), log=quiet) == K.FAIL
    ### A family (here grad.k, declared for the Triton read's atomics) passes within the
    ### larger repeat spread of either side, and fails beyond it.
    grad_k = "grad.k"
    fam = edited(a, key, grad_k, lambda t: t, device="cuda")
    shifted = edited(b, key, grad_k, ulp, device="cuda")
    assert (
        K.compare(fam, shifted, log=quiet) == K.FAIL
    )  # each side repeats bitwise: bitwise demanded
    spread = edited(fam, key, grad_k, lambda t: t)
    spread["tensors"][key][grad_k].append(ulp(ulp(spread["tensors"][key][grad_k][0])))
    spread["cases"][key]["tensors"][grad_k]["sha256"].append(
        R.tensor_hash(spread["tensors"][key][grad_k][1])
    )
    assert K.compare(spread, shifted, log=quiet) == K.PASS
