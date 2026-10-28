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
G-READ, G-BLOCKS, G-HEADS, G-MODEL, G-PARTS) is run here and compared with
``data/goldens_v1.pt``, which was recorded on one CPU (its manifest
``data/goldens_v1.json`` names it) and is
written only by ``python -m test.experimental.models.isla.goldens.record
--regenerate goldens_v1.pt``; a missing golden fails. The reading, per tensor:

- inputs (``in.*`` and the parameters): bitwise (their SHA-256), so a change of a
  fixture or of a library primitive feeding one is reported as such, not as a
  change of the module under test;
- integer and boolean outputs (forest tables, walk lists, ``PairPlan``, kept cells,
  search flags): exact (SHA-256);
- float outputs: every stored element and the summaries over every element (largest
  magnitude, sum of magnitudes, a fixed weighted sum) within ``rtol * max|golden|``
  per element, the summaries within ``sqrt(n) * rtol * max|golden|`` (the rounding
  of n elements adding up as a random walk; so one element of a large tensor cannot
  move by much unseen), plus 1e-14 of the sum of magnitudes for float64 summation
  order; where a case keeps sums over chunks of a large tensor (``Case.chunks``),
  each within ``sqrt(m) * rtol * max|golden|`` for a chunk of m elements, and the
  same 1e-14 term. ``rtol`` is 1e-12 in float64, which holds across BLAS builds and
  CPU architectures; float32 and bfloat16 bounds are stated at :data:`CPU_RTOL`; a
  case may state its own bound for a named output, with its reason
  (``Case.tol``, ``Case.cuda_tol``; ``None``: not compared, see goldens/README.md,
  "Exclusions").

The inputs are built so that they are the same bits on every platform (correctly
rounded operations only; see ``cases.py``). Measured across x86 (the golden's
platform), the LHR CPU partition, and Grace with GB300 (jobs 377997, 378376), the
largest relative errors were 4.6e-14 in float64 (6.9e-12 on the thin cap's weights,
bound 1e-8 there), 2.3e-5 in float32 and 5.7e-3 in bfloat16.

The CUDA variants (the fast and fused kernels, and the references on CUDA) are
compared with the same CPU golden at :data:`CUDA_RTOL`, integers exact where the code
claims the fast path is the reference's bitwise.
"""

import fnmatch
import math

import pytest
import torch

from test.experimental.models.isla._marks import requires_cuda
from test.experimental.models.isla.goldens import cases as C
from test.experimental.models.isla.goldens import compare as K
from test.experimental.models.isla.goldens import record as R

#: Relative tolerance (of ``max|golden|``) of the CPU variants against the golden, by
#: dtype. float64: 1e-12, the plan's bound across BLAS builds, CPU architectures and
#: thread counts (measured: at most 1.3e-13, a slice block's prototype-gain gradient,
#: a scalar reduction over every token and slice, between one and four threads).
#: float32: the same bound scaled by the ratio of the unit roundoffs, 2**29 (the
#: computations are the same, so their conditioning is; measured at most 2.3e-5,
#: between thread counts and across x86, Grace and GB300). bfloat16 (the segment-softmax read's bf16 gradients, computed in
#: float32 and rounded once): two bf16 ulps of the largest element, 2**-6.
CPU_RTOL = {"float64": 1e-12, "float32": 1e-12 * 2**29, "bfloat16": 2**-6}
#: The same for the CUDA variants (TF32 off).
CUDA_RTOL = dict(CPU_RTOL)


@pytest.fixture(scope="module")
def golden():
    if not R.GOLDEN.exists():
        pytest.fail(
            f"{R.GOLDEN} is missing: regenerate it with "
            f"`python -m {R.__name__} --regenerate {R.GOLDEN.name}`"
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
    got = x[R.sample_index(n, len(ref))]
    same_nan = ref.isnan() == got.isnan()
    err = torch.where(
        ref.isfinite() & got.isfinite(),
        (got - ref).abs(),
        torch.where((got == ref) | (got.isnan() & ref.isnan()), 0.0, math.inf),
    )
    worst = float(err.max()) if err.numel() else 0.0
    if not bool(same_nan.all()) or worst > tol:
        bad.append(
            f"{name}: samples off by {worst:.3e} > {tol:.3e} ({rtol:g} of max|golden| {scale:.3e})"
        )
    ### A sum of n elements, each within tol: rounding errors add up as a random walk,
    ### sqrt(n) tol (a bound of n tol would let one element of a large tensor move by
    ### max|golden| unseen), and 1e-14 of the sum of magnitudes for float64 summation
    ### order.
    slack = math.sqrt(n) * tol + 1e-14 * entry["l1"]
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
    if "chunk" in entry:
        chunks, ref_chunks = xz.split(entry["chunk"]), entry["chunk_sums"]
        for j, (c, r) in enumerate(zip(chunks, ref_chunks, strict=True)):
            b = math.sqrt(c.numel()) * tol + 1e-14 * entry["l1"]
            if abs(float(c.sum()) - r) > b:
                bad.append(
                    f"{name}: chunk {j} sum {float(c.sum())!r} against golden {r!r}, "
                    f"off by {abs(float(c.sum()) - r):.3e} > {b:.3e}"
                )
    return bad


def check(
    golden_case: dict,
    tensors: dict,
    rtol: dict,
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
            if R.tensor_hash(t) != entry["sha256"]:
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
            got = t.reshape(-1).double()[R.sample_index(t.numel(), len(ref))]
            ok = ref.isfinite() & got.isfinite()
            diff = (got - ref).abs()[ok]
            rel = (float(diff.max()) if diff.numel() else 0.0) / entry["scale"]
            worst[entry["dtype"]] = max(worst.get(entry["dtype"], 0.0), rel)
    return bad, worst


def _params(device):
    out = []
    for case in C.CASES:
        for variant in case.variants.get(device, ()):
            marks = [pytest.mark.cuda, requires_cuda] if device == "cuda" else []
            out.append(
                pytest.param(case, variant, id=f"{case.id}:{variant}", marks=marks)
            )
    return out


def test_the_golden_holds_every_case_and_its_manifest():
    """The committed golden has exactly the cases of cases.py (none missing, none
    stale) and a manifest naming its commit, platform and module versions."""
    g = R.load_golden()
    assert set(g["cases"]) == {c.id for c in C.CASES}
    for case in C.CASES:  # the chunk sums a case asks for are there
        for name, entry in g["cases"][case.id].items():
            assert entry.get("chunk") == case.chunk_length(name), (case.id, name)
    m = g["manifest"]
    assert {"commit", "date", "platform", "versions"} <= m.keys()
    assert m["platform"]["device"] == "cpu" and m["platform"]["threads"] == 1


@pytest.mark.parametrize("case, variant", _params("cpu"))
def test_cpu_matches_the_golden(case, variant, golden):
    """Each CPU variant of the case (the reference, and the fast build and walk, which
    claim the reference's results bitwise) against the committed golden: inputs and
    integers exact, floats within :data:`CPU_RTOL` of ``max|golden|``. The G-SAMP draws
    and G-MODEL/G-PARTS' dtt_draw pin today's draw (DRAW-EXACT); G-GEO's tie sets keep
    off the tie selections (GEO-TIEORDER). Regenerate: ``python -m
    test.experimental.models.isla.goldens.record --regenerate goldens_v1.pt``."""
    bad, worst = check(
        golden["cases"][case.id],
        C.run_case(case, "cpu", variant),
        CPU_RTOL,
        overrides=case.tol,
    )
    print("largest relative error of the samples per dtype:", worst)  # shown with -rP
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("case, variant", _params("cuda"))
def test_cuda_matches_the_cpu_golden(case, variant, golden, monkeypatch):
    """Each CUDA variant (kernels auto: the fast build, walk and geometry search, the
    Triton and fused kernels; and the references on CUDA) against the committed CPU
    golden: inputs and integers exact, floats within :data:`CUDA_RTOL` of
    ``max|golden|`` (or the case's own stated bound, ``Case.cuda_tol``); the draw's
    kept cells exact and its weights at the float64 bound (DRAW-EXACT: today's draw).
    Regenerate: ``python -m test.experimental.models.isla.goldens.record --regenerate
    goldens_v1.pt``."""
    why = C.available(case, "cuda")
    if why:
        pytest.skip(why)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    tensors = C.run_case(case, "cuda", variant)
    bad, worst = check(
        golden["cases"][case.id],
        tensors,
        CUDA_RTOL,
        overrides={**case.tol, **case.cuda_tol},
    )
    print("largest relative error of the samples per dtype:", worst)  # shown with -rP
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("spec", C.draw_specs(), ids=lambda s: s[0])
def test_the_draw_goldens_keep_off_the_cell_boundaries(spec):
    """No recorded draw has a hit within :data:`cases.DRAW_MARGIN` of a boundary
    between two cells on the line: the kept cells are the same with the start moved
    by that much either way (every hit moves with it). A draw whose golden held a hit
    there would move with the rounding of the line's sums, not with the sampling
    (DRAW-EXACT: its exact sums move them by that rounding)."""
    _, measures, centroids, budget, start = spec
    draw = C.target("sample_boundary_cells")
    kept = [
        draw(measures, centroids, budget, torch.tensor(start + s, dtype=torch.float64))[
            0
        ]
        for s in (0.0, -C.DRAW_MARGIN, C.DRAW_MARGIN)
    ]
    assert torch.equal(kept[0], kept[1]) and torch.equal(kept[0], kept[2])


# --------------------------------------------------------------------------- the A/B harness


def _quick_cases():
    return C.select(["G-HEADS", "G-SAMP", "G-SEG"])


def test_the_ab_harness_passes_a_rerun_and_catches_a_one_ulp_change(tmp_path):
    """Recorded twice in one process on CPU (one thread), the harness reads PASS; the
    records survive a save and load; one output moved by one ulp reads FAIL, a missing
    case FAIL, a flipped kept cell on CUDA INCONCLUSIVE (on CPU FAIL), weights moved
    within 1e-15 relative on CUDA PASS, and a declared family within its spread PASS.
    DRAW-EXACT: the kept and weights rules read today's CUDA draw, which is not
    run-to-run bitwise."""
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


def test_the_ab_harness_reads_every_repeat_and_refuses_an_empty_comparison(tmp_path):
    """On CUDA a kept cell that flips between the repeats of one side reads
    INCONCLUSIVE, weights that wobble within 1e-15 between repeats PASS (beyond it
    FAIL), a family whose head repeats noisily beyond its ceiling FAILs even though
    that noise widens its own spread, and a comparison of no case at all FAILs.
    DRAW-EXACT: the kept and weights rules read today's CUDA draw."""
    import copy

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        a = R.record(C.select(["G-SAMP", "G-SEG"]), "cpu", log=lambda *_: None)
    finally:
        torch.set_num_threads(threads)
    for c in a["cases"].values():
        c["device"] = "cuda"
    quiet = lambda *_: None  # noqa: E731

    def repeated(rec, key, name, t):
        """``rec`` with one more repeat of ``name``, the tensor ``t``."""
        rec = copy.deepcopy(rec)
        rec["tensors"][key][name].append(t)
        rec["cases"][key]["tensors"][name]["sha256"].append(R.tensor_hash(t))
        return rec

    ulp = lambda t: torch.nextafter(t, torch.full_like(t, math.inf))  # noqa: E731
    draw = "G-SAMP/sample_boundary_cells_3d@cpu:reference"
    kept, weights = "out.kept_300_0.37", "out.weights_300_0.37"
    k0, w0 = a["tensors"][draw][kept][0], a["tensors"][draw][weights][0]
    flip = torch.cat([k0[:-1], k0[-1:] + 1])
    assert K.compare(a, repeated(a, draw, kept, flip), log=quiet) == K.INCONCLUSIVE
    assert K.compare(a, repeated(a, draw, weights, ulp(w0)), log=quiet) == K.PASS
    assert (
        K.compare(a, repeated(a, draw, weights, w0 * (1 + 1e-12)), log=quiet) == K.FAIL
    )
    seg = "G-SEG/segment_softmax_read_f32@cpu:reference"
    grad_k = a["tensors"][seg]["grad.k"][0]
    noise = 1e-3 * float(grad_k.abs().max()) * torch.ones_like(grad_k)
    noisy = repeated(a, seg, "grad.k", grad_k + noise)
    noisy["tensors"][seg]["grad.k"][0] = grad_k - noise
    noisy["cases"][seg]["tensors"]["grad.k"]["sha256"][0] = R.tensor_hash(
        grad_k - noise
    )
    assert K.compare(a, noisy, log=quiet) == K.FAIL
    assert K.compare(a, a, log=quiet, only=["G-GEO/nothing*"]) == K.FAIL
    empty = {**a, "cases": {}, "tensors": {}}
    assert K.compare(empty, empty, log=quiet) == K.FAIL
    with pytest.raises(SystemExit):
        R.main(["--groups", "G-Geo", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        R.main(["--cases", "no_such_case*", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):  # name the files, or `all`
        R.main(["--regenerate"])
