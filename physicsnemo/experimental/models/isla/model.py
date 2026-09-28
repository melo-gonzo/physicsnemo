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

r"""ISLA: Invariant Slice Attention.

An O(d)/SE(d)-equivariant soft-slice transformer (any spatial dimension d >= 2;
the program's datasets are two- and three-dimensional) for steady boundary-value PDE
surrogates. A boundary sample (positions, unit normals, quadrature measures)
goes in, together with the problem's data as *named fields of declared
tensor rank*: global inputs (a direction, a diffusivity, a Reynolds number),
per-cell boundary inputs (a prescribed traction, a Dirichlet trace, a
condition tag) and, in the interior modes, optional per-query inputs (a
material label, a source term). Fields of declared rank come out: scalars,
vectors, rank-2 tensors, up to rank 4, each exactly covariant. The name states the
design principle: the attention operates only on invariants of the per-point
vector set :math:`\{r_i, n_i, \hat g_1, \dots, \hat g_K, v_{i1}, \dots\}`
(scaled position, unit normal, unit global vectors, the token's own tensor
inputs), and the frame is re-attached only at the heads, so exact rotation,
reflection and translation covariance is paid once at the network's edges
instead of in every layer.

**Rank specs.** Inputs and outputs are declared as mappings from a field
name to a rank spec: an ``int`` rank (0 scalar, 1 vector, 2 rank-2 tensor,
...) or a mapping ``{"rank": r, "symmetric": bool, "parity": "even" |
"odd", "channels": int}``. ``symmetric`` (rank :math:`\ge 2`) means fully
symmetric under index permutation, ``parity="odd"`` (outputs only) a
pseudotensor that flips sign under a reflection (a vorticity), and
``channels`` (rank 0 only) packs several independent scalars under one name
(a standardized parameter vector). The same grammar serves the constructor's
``outputs``, ``boundary_inputs``, ``query_inputs`` and ``global_inputs``.

**Names may nest.** A field name may contain ``"."`` to denote nesting, the
project's TensorDict convention (:mod:`physicsnemo.datapipes.keys`): every
value of the four mappings is a leaf rank spec, so a mapping value is never a
group and ``{"surface.pressure": 0, "surface.wall_shear": 1}`` declares two
leaves of one group. ``forward`` returns a :class:`~tensordict.TensorDict` of
batch size ``(B, N_out)`` with the dotted names unflattened
(``out["surface", "pressure"]``, ``out["surface"]["pressure"]``; a name
without a dot is a top-level leaf as before), and accepts each named input
mapping flat with dotted names, nested, or as a TensorDict.

**Inputs of any rank enter through invariants.** A rank-1 boundary input
:math:`v_i` at a token contributes the seed scalars :math:`v_i \cdot \hat
n_i`, :math:`v_i \cdot \hat g_k`, :math:`|v_i|` and its dot products with
the token's other vector inputs, and, inside every slice layer where the
token-to-anchor direction :math:`\hat\rho_{is}` exists, the contraction
:math:`v_i \cdot \hat\rho_{is}` next to the existing :math:`\hat n_i \cdot
\hat\rho_{is}` and :math:`\hat g_k \cdot \hat\rho_{is}` (one shared helper
contracts every per-token vector with the anchor direction; the normal is
just the first of them). A rank-2 input :math:`T_i` contributes the seed
scalars :math:`a^{\mathsf T} T_i b` over the frame vectors :math:`a, b \in
\{\hat n_i, \hat g_k\}`, :math:`\operatorname{tr} T_i` and
:math:`\|T_i\|_F`, and in the layers :math:`\hat\rho^{\mathsf T} T_i
\hat\rho`, :math:`\hat\rho \cdot T_i \hat n_i` and :math:`\hat\rho \cdot
T_i \hat g_k` (the last two are contractions of the derived true vectors
:math:`T_i \hat n_i`, :math:`T_i \hat g_k` with the anchor direction, so they
ride on the same helper). Only true vectors appear on the input side, no
cross products, so every input invariant is invariant under improper
rotations as well; pseudotensor *inputs* are therefore refused. Rank-0
inputs are the scalars themselves. Inputs of rank above 2 are refused (the
contraction set grows combinatorially and no dataset of the program has
one).

**Outputs of any rank leave through an equivariant basis.** One linear head
on the normalized final token state emits invariant coefficients; a rank-
:math:`r` output is the contraction of those coefficients with tensor
products of :math:`r` vectors from the token's basis: the *true* vectors
:math:`t_1, \dots, t_m` (the global directions, the normal, the radial
direction from the token's soft anchor and the :math:`e_\theta`
complements; :math:`m = 2K + 3` in every dimension) for even parity, and
products with exactly one *pseudovector* factor for odd parity, the
pseudovectors being the Levi-Civita construction from the primary true vectors
:math:`\hat g_k, \hat n_i, \hat u_i`: their pairwise cross products in three
dimensions (:math:`p = \binom{K+2}{2}`), their perpendiculars in two
(:math:`p = K + 2`); odd parity is refused in four or more dimensions, where a
pseudovector needs :math:`d - 1` vectors. Plus the isotropic tensors of that
rank and parity (the identity at rank 2, the three products of Kronecker
deltas at rank 4, the Levi-Civita symbol, of rank :math:`d`, for odd parity).
A symmetric spec uses the symmetrized basis with one
coefficient per unordered multi-index. Coefficient counts are in the class
docstring; ranks above 4 are refused.

**The frame.** ISLA has no frame origin: positions enter only as
point-to-anchor differences, and the one length scale is the
measure-weighted root-mean-square pairwise distance of the sample
(``scale_mode="rms_distance"``, the default: a normalized two-point
statistic that is invariant to the weights' overall scale and converges
under refinement of any boundary with a finite second moment, fractal ones
included) or, to reproduce lanes trained before 2026-09-18 only, the square
root of the total quadrature measure (``scale_mode="total_measure"``; the
two agree within 1.3% of surface-pressure error on DrivAerML; TODO: remove
once those lanes are retired). No centroid, no per-dataset
reference length and no external frame appears anywhere in the forward
pass, so the frame carries no information about where the mesher placed
its cells.

**Interior modes.** ``query_tokens=True`` lets interior query points join
the encoder as interacting tokens; ``query_independent=True`` decodes queries
passively through read-only blocks, so a prediction at one point does not
depend on which other points are queried; ``support_tokens=True`` (with
``query_independent``) adds a problem-derived interior support set as
interacting tokens while the requested queries stay passive. Per-query
inputs (``query_inputs``) are carried by the query and support tokens;
per-cell boundary inputs (``boundary_inputs``) by the boundary cells, with
zeros on every other token.

**Interior geometry.** An interior token needs a direction (its "normal" in
every invariant) and a distance to the wall, and both are functions of the
boundary sample and the query position, so by default the model computes
them itself (``interior_geometry="nearest_boundary"``): for each query the
nearest boundary sample :math:`y_{i^*}` gives the direction :math:`u_q =
(x_q - y_{i^*}) / |x_q - y_{i^*}|` and the distance :math:`d_q = \pm |x_q -
y_{i^*}|`, signed by the nearest sample's normal (positive on the side the
normal points to: outside, for outward normals, the convention of the stored
signed-distance fields of the program's datasets). The search is a chunked
nearest-neighbour pass over the queries (:func:`nearest_boundary_geometry`:
an exact brute-force pass, a fused Triton kernel on CUDA and a float64
matmul expansion elsewhere; the :math:`N_q \times N` distance matrix is never
materialized; ``interior_nn_chunk`` sizes the fallback's blocks), and
:math:`d_q / L` with its signed logarithm enters the token's embedding as a
built-in rank-0 channel ahead of the declared ``query_inputs``. The caller
passes ``query_points`` (and ``support_points``) only. Because the boundary
is a sample (200,000 of up to 17 million cells on a DrivAerML car), the
internal distance overestimates the true wall distance near the wall by up
to about the sample spacing, and the direction deviates from the true SDF
gradient by the same order; far from the wall both agree with the full-mesh
field (the price is measured, not assumed: INTERNAL-GEOMETRY, 2026-09-23).
``interior_geometry="provided"`` is the form of every interior lane trained
before that date: the caller supplies ``query_normals`` (``support_normals``;
e.g. the stored SDF gradient) and any distance as a declared ``query_inputs``
field, and no built-in channel exists. Both settings compute the geometry
from pairwise differences, so every covariance contract holds in either.

**Global inputs.** Rank-1 global inputs are the :math:`K` global directions:
each is normalized to a unit direction inside the model (a direction is what
the invariants consume; a physically meaningful magnitude belongs among the
rank-0 global inputs), enters the seed through :math:`\hat n_i \cdot \hat
g_k`, the layers through :math:`\hat g_k \cdot \hat\rho_{is}`, and the head
basis as :math:`\hat g_k` with the complements of :math:`(\hat u_i, \hat
g_k)`. Rank-0 global inputs are appended to every token's seed. Global
inputs of rank 2 are refused (broadcast the tensor to the boundary cells and
declare it as a boundary input instead).

**Kernels.** The per-layer geometry region has two exact implementations
of the same arithmetic: ``geo_kernel="eager"`` (PyTorch; ``geo_checkpoint``
rebuilds its intermediates in backward instead of storing them) and
``geo_kernel="fused"`` (one Triton kernel per direction that never
materializes the (B, N, S, .) tensor; CUDA, float32/float64 geometry, one
to three per-point vectors counting the global directions and the token's
rank-1 inputs together, at most 1024 slices, no rank-2 inputs). The default
``geo_kernel=None`` means *auto*: at the first forward
pass the model picks the fused kernel when the inputs are on CUDA, Triton
is importable and the constraints hold, and the eager region otherwise,
logging the choice once; at 200,000 cells the fused kernel runs the
training step in about a third of the eager time and memory (cost sweep,
2026-09-18). The per-token rank-2 contractions (rho^T T rho) are the standing
kernel follow-up; until then a model with rank-2 boundary or query inputs
runs the eager region.

Every constructor argument and every ``forward`` input is keyword-only, so a
call reads as the recipe's ``forward_kwargs`` mapping does and models can be
swapped without positional bookkeeping.
"""

import itertools
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn.functional.equivariant_ops import vector_project

logger = logging.getLogger(__name__)


### ---------------------------------------------------------------------------
### Rank specs
### ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankSpec:
    """A declared tensor field: its rank, symmetry, parity and (rank 0) channel count.

    ``rank`` is the number of spatial indices (0 scalar, 1 vector, 2 rank-2
    tensor, ...). ``symmetric`` means fully symmetric under any permutation of
    the indices and is meaningful for ``rank >= 2`` only (ignored below).
    ``parity`` is ``"even"`` for a true tensor and ``"odd"`` for a pseudotensor
    (one that flips sign under an improper rotation). ``channels`` packs several
    independent scalars under one rank-0 name; it is 1 for every other rank.
    """

    rank: int
    symmetric: bool = False
    parity: str = "even"
    channels: int = 1

    def size(self, dim: int = 3) -> int:
        """Number of components per token in ``dim`` spatial dimensions: ``channels``
        at rank 0, ``dim**rank`` above."""
        return self.channels if self.rank == 0 else int(dim) ** self.rank

    def shape(self, dim: int = 3) -> tuple[int, ...]:
        """Trailing component shape: ``()`` at rank 0, ``(dim,) * rank`` above."""
        return () if self.rank == 0 else (int(dim),) * self.rank


def _parse_rank_spec(kind: str, name: str, spec) -> RankSpec:
    """Validate one ``{name: spec}`` entry of an input or output mapping.

    ``spec`` is an ``int`` rank or a mapping with the keys ``rank`` (required),
    ``symmetric`` (default ``False``), ``parity`` (default ``"even"``) and
    ``channels`` (default 1, rank 0 only). ``kind`` names the mapping in error
    messages (``"outputs"``, ``"boundary_inputs"``, ...).
    """
    where = f"{kind}[{name!r}]"
    if isinstance(spec, bool):
        raise TypeError(f"{where}: a rank spec is an int or a mapping, got a bool")
    if isinstance(spec, int):
        spec = {"rank": spec}
    elif not isinstance(spec, Mapping):
        raise TypeError(
            f"{where}: a rank spec is an int rank or a mapping "
            f"{{rank, symmetric, parity, channels}}, got {type(spec).__name__}"
        )
    unknown = set(spec) - {"rank", "symmetric", "parity", "channels"}
    if unknown:
        raise ValueError(f"{where}: unknown rank-spec keys {sorted(unknown)!r}")
    if "rank" not in spec or isinstance(spec["rank"], bool):
        raise ValueError(f"{where}: a rank spec mapping needs an int 'rank'")
    rank = int(spec["rank"])
    if rank < 0:
        raise ValueError(f"{where}: rank must be non-negative, got {rank}")
    parity = spec.get("parity", "even")
    if parity not in ("even", "odd"):
        raise ValueError(f"{where}: parity must be 'even' or 'odd', got {parity!r}")
    channels = int(spec.get("channels", 1))
    if channels < 1:
        raise ValueError(f"{where}: channels must be >= 1, got {channels}")
    if channels != 1 and rank != 0:
        raise ValueError(f"{where}: 'channels' applies to rank-0 fields only")
    return RankSpec(
        rank=rank,
        symmetric=bool(spec.get("symmetric", False)) and rank >= 2,
        parity=parity,
        channels=channels,
    )


def _keys():
    """The project's nested-key helpers (``physicsnemo.datapipes.keys``), imported on
    first use: the datapipes package is heavy and the model must import without it."""
    from physicsnemo.datapipes import keys

    return keys


def _parse_rank_specs(kind: str, specs, *, max_rank: int, allow_odd: bool):
    """Parse a whole ``{name: spec}`` mapping (``None`` is the empty mapping).

    A name may contain ``"."`` to denote nesting (``"surface.pressure"`` is the
    leaf ``("surface", "pressure")`` of the returned or consumed TensorDict);
    every value is a leaf spec, so a mapping value is never a nested group. A
    name that is a group of another (``"surface"`` next to
    ``"surface.pressure"``) is refused.
    """
    if specs is None:
        return {}
    if not isinstance(specs, Mapping):
        raise TypeError(f"{kind} must be a mapping of field name to rank spec")
    out: dict[str, RankSpec] = {}
    paths: dict[str, tuple[str, ...]] = {}
    for name, spec in specs.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{kind}: field names must be non-empty strings")
        try:
            path = _keys().as_nested_key(name)
        except ValueError as e:
            raise ValueError(f"{kind}[{name!r}]: {e}") from None
        paths[name] = (path,) if isinstance(path, str) else path
        rs = _parse_rank_spec(kind, name, spec)
        if rs.rank > max_rank:
            raise ValueError(
                f"{kind}[{name!r}]: rank {rs.rank} is not supported (maximum {max_rank}"
                + (
                    "; the equivariant basis grows as m**rank)"
                    if kind == "outputs"
                    else "; see the class docstring for how higher ranks would enter)"
                )
            )
        if rs.parity == "odd" and not allow_odd:
            raise ValueError(
                f"{kind}[{name!r}]: pseudotensor (parity='odd') inputs are not supported; "
                "input invariants are formed with true vectors only so that they are "
                "invariant under reflections"
            )
        out[name] = rs
    for a, pa in paths.items():
        for b, pb in paths.items():
            if len(pb) > len(pa) and pb[: len(pa)] == pa:
                raise ValueError(
                    f"{kind}: {a!r} is a group of {b!r}; a name cannot be both a field "
                    "and a group of fields"
                )
    return out


def _flatten_fields(kind: str, fields):
    """Normalize a forward-time ``{name: tensor}`` mapping to flat dotted names.

    Accepts a flat mapping with dotted names (``{"surface.pressure": t}``), a
    nested mapping (``{"surface": {"pressure": t}}``) or a (nested)
    :class:`~tensordict.TensorDict`; returns ``{"surface.pressure": t}``.
    ``None`` passes through.
    """
    if fields is None:
        return None
    if isinstance(fields, TensorDict):
        k = _keys()
        return {k.key_to_str(key): fields.get(key) for key in k.leaf_keys(fields)}
    if not isinstance(fields, Mapping):
        raise TypeError(f"{kind} must be a mapping of field name to tensor")
    sep = _keys().KEY_SEPARATOR
    flat: dict[str, torch.Tensor] = {}

    def _walk(prefix: str, mapping: Mapping) -> None:
        for name, value in mapping.items():
            if not isinstance(name, str):
                raise TypeError(f"{kind}: field names must be strings, got {name!r}")
            full = f"{prefix}{sep}{name}" if prefix else name
            if isinstance(value, Mapping):
                _walk(full, value)
            else:
                flat[full] = value

    _walk("", fields)
    return flat


def _fields_to_tensordict(fields: dict[str, torch.Tensor], batch_size) -> TensorDict:
    """Pack flat dotted names into a nested TensorDict (``"a.b"`` -> ``td["a", "b"]``)."""
    k = _keys()
    out = TensorDict({}, batch_size=batch_size)
    for name, value in fields.items():
        out.set(k.as_nested_key(name), value)
    return out


### ---------------------------------------------------------------------------
### Input invariants
### ---------------------------------------------------------------------------


def _pair_indices(f: int, symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Index pairs ``(a, b)`` over ``f`` frame vectors: ``a <= b`` for a symmetric
    tensor (``a^T T b == b^T T a``), all ordered pairs otherwise."""
    if symmetric:
        return torch.triu_indices(f, f)
    grid = torch.cartesian_prod(torch.arange(f), torch.arange(f))
    return grid[:, 0], grid[:, 1]


class _InputSet:
    """The per-token tensor inputs of one token class (boundary cells, or interior
    queries and support points) and the invariants they contribute.

    With ``F = 1 + K`` frame vectors ``{n_hat, g_1..g_K}`` per token, ``P`` rank-1
    inputs and ``Q`` rank-2 inputs (``Q_s`` of them symmetric):

    * seed features (into the token's embedding): the rank-0 channels
      (``x 2`` when ``scaled_scalars``: ``[s, sign(s) log(|s| + eps)]``), then
      per rank-1 input its ``F`` frame cosines and its norm and the
      ``P(P-1)/2`` pairwise dot products of the rank-1 inputs, then per rank-2
      input the ``F(F+1)/2`` (symmetric) or ``F**2`` frame contractions
      ``a^T T b``, its trace and its Frobenius norm;
    * block vectors (contracted with the token-to-anchor direction in every
      slice layer): the ``P`` rank-1 inputs and, per rank-2 input, the ``F``
      derived true vectors ``T a`` (plus ``T^T a`` when not symmetric);
    * block tensors (``rho^T T rho`` in every slice layer): the ``Q`` rank-2 inputs.
    """

    def __init__(self, specs: dict[str, RankSpec], n_frame: int, scaled_scalars: bool, dim: int = 3):
        self.specs = specs
        self.dim = int(dim)
        self.n_frame = f = int(n_frame)
        self.scaled_scalars = bool(scaled_scalars)
        self.n_scalar = sum(s.channels for s in specs.values() if s.rank == 0)
        self.rank1 = [name for name, s in specs.items() if s.rank == 1]
        self.rank2 = [(name, s.symmetric) for name, s in specs.items() if s.rank == 2]
        p, q = len(self.rank1), len(self.rank2)
        n_pairs = sum(f * (f + 1) // 2 if sym else f * f for _, sym in self.rank2)
        self.n_features = (
            (2 if self.scaled_scalars else 1) * self.n_scalar
            + p * f
            + p
            + p * (p - 1) // 2
            + n_pairs
            + 2 * q
        )
        self.n_block_vecs = p + sum(f if sym else 2 * f for _, sym in self.rank2)
        self.n_block_tens = q

    def __bool__(self) -> bool:
        return bool(self.specs)

    def validate(self, kind: str, fields, b: int, n: int) -> None:
        """Presence and element-count checks for the forward-time mapping."""
        if not self.specs:
            if fields:
                raise ValueError(
                    f"this model declares no {kind}; got fields {sorted(fields)!r}"
                )
            return
        if fields is None:
            raise ValueError(
                f"{kind} declared {sorted(self.specs)!r}; pass {kind}={{name: tensor}}"
            )
        if not isinstance(fields, Mapping):
            raise TypeError(f"{kind} must be a mapping of field name to tensor")
        missing = [k for k in self.specs if k not in fields]
        extra = [k for k in fields if k not in self.specs]
        if missing or extra:
            raise ValueError(
                f"{kind} fields do not match the declared set: missing {missing!r}, "
                f"unexpected {extra!r}"
            )
        for name, spec in self.specs.items():
            t = fields[name]
            size = spec.size(self.dim)
            if not isinstance(t, torch.Tensor) or t.numel() != b * n * size:
                raise ValueError(
                    f"{kind}[{name!r}]: expected a tensor of shape "
                    f"{(b, n, *spec.shape(self.dim)) if spec.rank else (b, n) + ((spec.channels,) if spec.channels > 1 else ())} "
                    f"({b * n * size} elements), got "
                    f"{tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__}"
                )

    def features(self, fields, b: int, n: int, frame, *, gauge=None, eps=0.0):
        """Invariants of one token class.

        ``frame`` is ``(B, N, F, d)``: the token's true frame vectors
        ``[n_hat, g_1..g_K]`` in ``d`` spatial dimensions. Returns ``(seed (B, N,
        n_features), vecs (B, N, n_block_vecs, d), tens (B, N, n_block_tens, d, d))``; ``gauge``
        (``(B, 1, 1)``) divides the rank-0 channels when ``scaled_scalars``.
        """
        ### The invariants and the derived vectors are geometry: computed in the
        ### input precision even under autocast (the embedding that consumes the
        ### seed is autocast on its own).
        with torch.autocast(frame.device.type, enabled=False):
            return self._features(fields, b, n, frame, gauge, eps)

    def _features(self, fields, b, n, frame, gauge, eps):
        dtype = frame.dtype
        d = frame.shape[-1]
        seed, vecs, tens = [], [], []
        for name, spec in self.specs.items():
            if spec.rank == 0:
                s = fields[name].reshape(b, n, spec.channels).to(dtype)
                if self.scaled_scalars:
                    s = s / gauge
                    s = torch.cat([s, torch.sign(s) * torch.log(s.abs() + eps)], -1)
                seed.append(s)
        if self.rank1:
            v = torch.stack(
                [fields[name].reshape(b, n, d).to(dtype) for name in self.rank1], dim=2
            )  # (B, N, P, d)
            seed.append(_contract_vectors(v, frame).flatten(2))  # (B, N, P*F)
            seed.append(v.norm(dim=-1))  # (B, N, P)
            p = v.shape[2]
            if p > 1:
                iu, ju = torch.triu_indices(p, p, offset=1, device=v.device)
                seed.append((v[:, :, iu] * v[:, :, ju]).sum(-1))  # (B, N, P(P-1)/2)
            vecs.append(v)
        for name, symmetric in self.rank2:
            t = fields[name].reshape(b, n, d, d).to(dtype)
            ### a^T T b over the frame pairs, as (T b) . a with T b a derived vector.
            tb = torch.einsum("bnij,bnfj->bnfi", t, frame)  # (B, N, F, d): T a
            ia, ib = _pair_indices(self.n_frame, symmetric)
            seed.append(
                (frame[:, :, ia.to(t.device)] * tb[:, :, ib.to(t.device)]).sum(-1)
            )
            seed.append(torch.diagonal(t, dim1=-2, dim2=-1).sum(-1, keepdim=True))
            seed.append(t.flatten(2).norm(dim=-1, keepdim=True))
            vecs.append(tb)
            if not symmetric:
                vecs.append(torch.einsum("bnji,bnfj->bnfi", t, frame))  # T^T a
            tens.append(t)
        seed_t = torch.cat(seed, -1) if seed else frame.new_zeros(b, n, 0)
        vecs_t = torch.cat(vecs, 2) if vecs else frame.new_zeros(b, n, 0, d)
        tens_t = torch.stack(tens, 2) if tens else frame.new_zeros(b, n, 0, d, d)
        return seed_t, vecs_t, tens_t


def _contract_vectors(x, y):
    """Dot products of per-token vector stacks ``x (B, N, P, d)`` and ``y (B, N, F, d)``
    -> ``(B, N, P, F)``. Broadcast multiply-and-sum, not einsum: einsum is autocast
    to bf16 and the geometry must stay in the input precision."""
    return (x[:, :, :, None, :] * y[:, :, None, :, :]).sum(-1)


### ---------------------------------------------------------------------------
### Backbone
### ---------------------------------------------------------------------------


def _softmax_over_points(
    x: Float[torch.Tensor, "batch tokens slices"], fast: bool = True
):
    """Softmax over the point dimension (dim 1) of a (B, N, S) tensor.

    ISLA-PERF (2026-09-09): PyTorch serves a softmax over a *middle* dimension
    with its "spatial" kernel, which parallelizes over the B*S columns only and
    walks the N points serially. On a GB300 that kernel was 85-95% of ISLA's
    training step (203 of 238 ms at 10k tokens, 3.3 of 3.5 s at 80k) and the
    source of the superlinear token scaling. Reducing along a contiguous last
    dimension instead uses the row-parallel kernel. Same arithmetic; the
    floating-point summation order differs (roundoff). ``fast=False`` keeps
    the original kernel for bitwise reproduction of checkpoints trained before
    2026-09-10 (bf16 outputs differ 1-1.5% between kernels; float32 model outputs
    agree to 1e-6). On raw softmax weights the two kernels are not equally
    precise: against a float64 reference the middle-dimension kernel carries a
    ~1e-4 absolute float32 floor at every N from 40k to 100k, the row-parallel
    kernel ~1e-7 (measured on a GB300, 2026-09-10; no defect at any N, including
    across 2^16 points). Which kernel a model runs is decided by this flag at
    instantiation, not by the checkpoint: a checkpoint written before the flag
    existed takes the current default when loaded.
    """
    if not fast:
        return torch.softmax(x, dim=1)
    return torch.softmax(x.transpose(1, 2), dim=-1).transpose(1, 2)


def _relational_invariants(
    r: Float[torch.Tensor, "batch tokens dim"],
    n_hat: Float[torch.Tensor, "batch tokens dim"],
    g_hat: Float[torch.Tensor, "batch tokens vectors dim"],
    z_pos: Float[torch.Tensor, "batch slices dim"],
    m_s: Float[torch.Tensor, "batch slices dim"],
    eps: float,
    vecs: Float[torch.Tensor, "batch tokens extra dim"] | None = None,
    tens: Float[torch.Tensor, "batch tokens tensors dim dim"] | None = None,
) -> Float[torch.Tensor, "batch tokens slices geo"]:
    """The point-anchor invariants: distance and its log, the unit relative
    vector dotted with each global direction (K terms), with the point normal
    and with the anchor normal, the point normal dotted with the anchor normal,
    then the unit relative vector dotted with each of the token's extra vectors
    (its rank-1 inputs and the vectors derived from its rank-2 inputs) and the
    quadratic form of each rank-2 input in the unit relative vector. Width
    ``5 + K + P' + Q``; for K = 1 and no inputs this is the original 6-wide set
    in its original order. Every term depends on the anchors' positions
    relative to the point only, so no frame origin is referred to. Shared by
    the encoder slice blocks and the passive decoder blocks."""
    z = z_pos[:, None, :, :]
    m = m_s[:, None, :, :]
    rel = r[:, :, None, :] - z  # (B, N, S, 3)
    dist = rel.norm(dim=-1, keepdim=True).clamp_min(eps)
    rel_hat = rel / dist
    n_exp = n_hat[:, :, None, :]
    feats = [
        dist,
        torch.log(dist),
        ### Broadcast multiply-and-sum, not einsum: einsum is autocast to bf16 and
        ### the geometry must stay in the input precision (for K = 1 this is the
        ### former (rel_hat * d).sum(-1) arithmetic exactly).
        (rel_hat[..., None, :] * g_hat[:, :, None, :, :]).sum(-1),  # (B, N, S, K)
        (rel_hat * n_exp).sum(-1, keepdim=True),
        (rel_hat * m).sum(-1, keepdim=True),
        (n_exp * m).sum(-1, keepdim=True),
    ]
    if vecs is not None and vecs.shape[2]:
        ### The token's vector inputs, contracted with the anchor direction exactly
        ### as the global directions are above.
        feats.append((rel_hat[..., None, :] * vecs[:, :, None, :, :]).sum(-1))
    if tens is not None and tens.shape[2]:
        ### rho^T T rho per rank-2 input. einsum (autocast disabled, so the geometry
        ### stays in the input precision) rather than a broadcast product, which
        ### would materialize a (B, N, S, Q, 3, 3) intermediate.
        with torch.autocast(rel_hat.device.type, enabled=False):
            t_rho = torch.einsum("bnqij,bnsj->bnsqi", tens, rel_hat)  # (B, N, S, Q, 3)
            feats.append(torch.einsum("bnsqi,bnsi->bnsq", t_rho, rel_hat))
    return torch.cat(feats, dim=-1)


def _geo_bias_head(n_geo: int, geo_bias: str) -> nn.Module:
    """The map from a pair's invariants to its routing-bias scalar. ``"linear"`` (the
    class default) is one Linear(n_geo, 1); ``"mlp"`` (GEO-BIAS study, 2026-09-25) is
    Linear(n_geo, 16) -> GELU -> Linear(16, 1), a nonlinear function of the same
    invariants. Both run on the eager region and inside the fused kernel (the kernel
    evaluates the hidden units one tile at a time, 2026-09-26)."""
    if geo_bias == "linear":
        return nn.Linear(n_geo, 1)
    if geo_bias == "mlp":
        return nn.Sequential(nn.Linear(n_geo, 16), nn.GELU(), nn.Linear(16, 1))
    raise ValueError(f"geo_bias must be 'linear' or 'mlp', got {geo_bias!r}")


def _geo_region(
    lin: nn.Module,
    logits_pre,
    r,
    n_hat,
    g_hat,
    z_pos,
    m_s,
    eps: float,
    vecs=None,
    tens=None,
):
    """One recompute region per layer: the per-slice routing bias from the
    invariants and the invariants pooled over slices by the resulting
    point->slice mix. Returns (bias (B,N,S), mix (B,N,S), pooled (B,N,geo)); the
    (B,N,S,geo) invariants and their (B,N,S,3) intermediates never leave the
    region, so under checkpointing they are rebuilt in backward, not stored."""
    geo = _relational_invariants(r, n_hat, g_hat, z_pos, m_s, eps, vecs, tens)
    bias = lin(geo).squeeze(-1)
    mix = torch.softmax(logits_pre + bias, dim=-1)  # normalized over slices
    return bias, mix, torch.einsum("bns,bnsg->bng", mix, geo)


def _fused_vectors(g_hat, vecs):
    """The fused kernel's per-point vector set: the K global directions followed by
    the token's rank-1 vectors (if any), (B, N, K + P', 3)."""
    if vecs is None or vecs.shape[2] == 0:
        return g_hat
    return torch.cat([g_hat, vecs], dim=2)


def _fused_geo_region():
    """The Triton-fused dense geometry region (geo_kernel="fused"); imported lazily so
    that the module loads without triton and the eager path never touches it."""
    from .geo_kernel import fused_geo_region

    return fused_geo_region


def _geo_width(n_global_vectors: int, n_vecs: int = 0, n_tens: int = 0) -> int:
    """Width of _relational_invariants: dist, log dist, K rel.g_k, rel.n, rel.m,
    n.m (5 + K), then P' rel.v and Q rho^T T rho; K = 1 alone gives the original 6."""
    return 5 + int(n_global_vectors) + int(n_vecs) + int(n_tens)


class _RoutingLogits(nn.Module):
    """Point-to-slice routing logits, ``(B, N, hidden) -> (B, N, S)``.

    ``"linear"``: an affine map of the normalized token state (the original
    form). ``"cosine"``: ``temperature`` times the cosine between the token
    state and a learned prototype per slice, so every logit lies in
    ``[-temperature, temperature]`` and no block can route every point to
    one slice however its weights grow (ROUTING-SAT measured that state as the
    mean-field plateau's mechanism). The parameter is ``weight`` in both modes
    so the two share a state-dict layout; ``bias`` exists only in linear mode.
    """

    def __init__(
        self,
        hidden: int,
        n_slices: int,
        mode: str = "cosine",
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        if mode not in ("linear", "cosine"):
            raise ValueError(f"routing_norm must be 'linear' or 'cosine', got {mode!r}")
        self.mode = mode
        self.temperature = float(temperature)
        lin = nn.Linear(hidden, n_slices, bias=(mode == "linear"))
        self.weight = lin.weight
        self.bias = lin.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            return F.linear(x, self.weight, self.bias)
        return self.temperature * F.linear(
            F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1)
        )


def _local_region(
    h: torch.Tensor,
    log_w: torch.Tensor,
    r: torch.Tensor,
    n_hat: torch.Tensor,
    g_hat: torch.Tensor,
    rho: float,
    eps: float,
    chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local relational aggregation (LOCAL-AGG, 2026-09-27): for every token, the
    measure-weighted Gaussian-kernel average over all tokens of (i) the token states
    and (ii) the pair invariants, with kernel ``softmax_j(-|r_i - r_j|^2 / rho^2 +
    log w_j)`` over the source tokens. The pair invariants are the anchor
    invariants of :func:`_relational_invariants` written for a token pair: the
    distance in units of ``rho``, its log, the unit separation dotted with each
    global direction (K terms), with the receiving token's normal and with the
    source token's normal, and the two normals' dot product (width 5 + K). The
    kernel is exactly invariant to a uniform rescale of the measure weights and
    every term depends on relative positions only, so the block keeps the
    class's rotation and translation contracts. Row-chunked and recomputed in
    backward (the (chunk, N) kernel and (chunk, N, 3) separations are never
    stored). Cost is O(N^2) per layer: written for the fixed-mesh sets of 10 to
    50 thousand tokens; a neighbour-restricted kernel is the follow-up for
    200,000-token surfaces. Returns (pooled states (B, N, H), pooled invariants
    (B, N, 5 + K))."""
    b, n, _ = r.shape
    k_dirs = g_hat.shape[2]

    def _chunk(r_c, n_c, g_c, r_all, n_all, h_all, logw_all):
        rel = r_c[:, :, None, :] - r_all[:, None, :, :]  # (B, c, N, 3), receiver minus source
        d2 = rel.square().sum(-1)
        dist = d2.clamp_min(eps * eps).sqrt()
        rel_hat = rel / dist[..., None]
        logits = -d2 / (rho * rho) + logw_all[:, None, :, 0]
        kern = torch.softmax(logits, dim=-1)  # (B, c, N)
        feats = [
            dist / rho,
            torch.log(dist / rho),
            (rel_hat[..., None, :] * g_c[:, :, None, :, :]).sum(-1),  # (B, c, N, K)
            (rel_hat * n_c[:, :, None, :]).sum(-1, keepdim=True),
            (rel_hat * n_all[:, None, :, :]).sum(-1, keepdim=True),
            (n_c[:, :, None, :] * n_all[:, None, :, :]).sum(-1, keepdim=True),
        ]
        feats = [f if f.dim() == 4 else f[..., None] for f in feats]
        geo = torch.cat(feats, dim=-1)  # (B, c, N, 5 + K)
        pooled_h = torch.einsum("bcn,bnh->bch", kern.to(h_all.dtype), h_all)
        pooled_geo = torch.einsum("bcn,bcng->bcg", kern, geo)
        return pooled_h, pooled_geo

    outs_h, outs_g = [], []
    for i0 in range(0, n, chunk):
        sl = slice(i0, i0 + chunk)
        args = (r[:, sl], n_hat[:, sl], g_hat[:, sl], r, n_hat, h, log_w)
        if torch.is_grad_enabled():
            ph, pg = checkpoint(_chunk, *args, use_reentrant=False)
        else:
            ph, pg = _chunk(*args)
        outs_h.append(ph)
        outs_g.append(pg)
    return torch.cat(outs_h, dim=1), torch.cat(outs_g, dim=1)


def _knn_indices(r: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    """Indices of the ``k`` nearest tokens (the token itself included) for every
    token, (B, N, k), by row-chunked exact distances; positions only, so the
    result is shared by every slice block of the forward pass."""
    b, n, _ = r.shape
    k = min(int(k), n)
    out = []
    with torch.no_grad():
        for i0 in range(0, n, chunk):
            d2 = torch.cdist(r[:, i0 : i0 + chunk], r).square()
            out.append(torch.topk(d2, k, dim=-1, largest=False).indices)
    return torch.cat(out, dim=1)


def _lpr_weights(
    rel_over_rho: torch.Tensor, kern: torch.Tensor, n_hat: torch.Tensor, g_hat: torch.Tensor, gradient: bool, reg: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Local-polynomial-reproduction stencil weights (LOCAL-LPR, 2026-09-28). For a
    token with neighbour separations ``rel_over_rho = (r_i - r_j) / rho`` (B, N, k, d)
    and positive kernel weights ``kern`` (B, N, k), the weighted least-squares fit
    ``h_j ~ c + G . (r_j - r_i)`` gives value weights ``a0`` (B, N, k) that reproduce
    every affine field exactly at the token (they sum to one and have zero first
    moment), and gradient weights ``rho * G`` (B, N, d, k). Unlike a normalized
    Gaussian kernel, whose first moment is O(h) on an asymmetric stencil, the fit
    depends on the neighbour positions only through what a consistent estimator
    needs, so the pooled value is the same continuum quantity for any draw of the
    neighbours (the meshfree GMLS / moment-matching idea, cf. Mirzaei 2012 and the
    MEEC construction of Shaffer et al. 2026). With ``gradient`` the gradient is
    projected on the receiving normal and on each global direction, giving
    (B, N, 1 + K, k) invariant finite-difference weights (units of the pooled
    state per ``rho``); the projections rotate with the field, so the contracts
    hold. Solved in float32 as (d + 1) x (d + 1) normal equations with a relative
    ridge ``reg``. Returns ``(a0, grad_weights or None)``."""
    b, n, k, d = rel_over_rho.shape
    x = -rel_over_rho.float()  # (r_j - r_i) / rho
    P = torch.cat([torch.ones(b, n, k, 1, device=x.device, dtype=x.dtype), x], dim=-1)  # (B, N, k, d+1)
    W = kern.float()[..., None]  # (B, N, k, 1)
    PtW = (P * W).transpose(-1, -2)  # (B, N, d+1, k)
    A = PtW @ P  # (B, N, d+1, d+1)
    tr = A.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)[..., None] / (d + 1)
    A = A + reg * tr * torch.eye(d + 1, device=x.device, dtype=x.dtype)
    M = torch.linalg.solve(A, PtW)  # (B, N, d+1, k): rows = value weights, rho * gradient weights
    a0 = M[..., 0, :]
    if not gradient:
        return a0, None
    Gw = M[..., 1:, :]  # (B, N, d, k)
    proj = [torch.einsum("bnd,bndk->bnk", n_hat.float(), Gw)[:, :, None, :]]
    if g_hat.shape[2] > 0:
        proj.append(torch.einsum("bnvd,bndk->bnvk", g_hat.float(), Gw))
    return a0, torch.cat(proj, dim=2)  # (B, N, 1 + K, k)


def _local_region_knn(
    h: torch.Tensor,
    log_w: torch.Tensor,
    r: torch.Tensor,
    n_hat: torch.Tensor,
    g_hat: torch.Tensor,
    idx: torch.Tensor,
    rho: float,
    eps: float,
    weights: str = "gaussian",
    gradient: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`_local_region` restricted to each token's ``k`` nearest tokens
    (``idx`` from :func:`_knn_indices`): the same kernel and pair invariants,
    with the softmax over the neighbourhood instead of over all tokens. Equal to
    the dense form when ``k`` is the token count; O(N k) otherwise, which is what
    makes the channel affordable at 200,000 tokens. Recomputed in backward.
    ``weights="lpr"`` replaces the normalized Gaussian weights by the
    local-polynomial-reproduction weights of :func:`_lpr_weights` (consistent
    value estimate, draw-independent to first order), and ``gradient`` appends the
    normal- and direction-projected finite-difference channels to the pooled
    states (width ``H * (2 + K)`` instead of ``H``)."""
    def _run(h, log_w, r, n_hat, g_hat, idx):
        b, n, k = idx.shape
        gather = lambda t: torch.gather(t, 1, idx.reshape(b, n * k, 1).expand(-1, -1, t.shape[-1])).reshape(b, n, k, t.shape[-1])
        r_j, n_j, h_j, logw_j = gather(r), gather(n_hat), gather(h), gather(log_w)
        rel = r[:, :, None, :] - r_j  # (B, N, k, 3)
        d2 = rel.square().sum(-1)
        dist = d2.clamp_min(eps * eps).sqrt()
        rel_hat = rel / dist[..., None]
        kern = torch.softmax(-d2 / (rho * rho) + logw_j[..., 0], dim=-1)  # (B, N, k)
        grad_w = None
        if weights == "lpr":
            kern, grad_w = _lpr_weights(rel / rho, kern, n_hat, g_hat, gradient)
        feats = [
            (dist / rho)[..., None],
            torch.log(dist / rho)[..., None],
            (rel_hat[..., None, :] * g_hat[:, :, None, :, :]).sum(-1),  # (B, N, k, K)
            (rel_hat * n_hat[:, :, None, :]).sum(-1, keepdim=True),
            (rel_hat * n_j).sum(-1, keepdim=True),
            (n_hat[:, :, None, :] * n_j).sum(-1, keepdim=True),
        ]
        geo = torch.cat(feats, dim=-1)
        pooled_h = (kern[..., None].to(h_j.dtype) * h_j).sum(2)
        pooled_geo = (kern[..., None].to(geo.dtype) * geo).sum(2)
        if grad_w is not None:
            fd = torch.einsum("bnvk,bnkh->bnvh", grad_w.to(h_j.dtype), h_j)  # (B, N, 1 + K, H)
            pooled_h = torch.cat([pooled_h, fd.flatten(2)], dim=-1)
        return pooled_h, pooled_geo

    if torch.is_grad_enabled():
        return checkpoint(_run, h, log_w, r, n_hat, g_hat, idx, use_reentrant=False)
    return _run(h, log_w, r, n_hat, g_hat, idx)


class _SliceBlock(nn.Module):
    """One pre-LN layer of measure-weighted soft-slice attention + MLP."""

    def __init__(
        self,
        hidden: int,
        n_slices: int,
        mlp_ratio: int = 4,
        use_relational_geo: bool = True,
        geo_checkpoint: bool = False,
        fast_point_softmax: bool = True,
        geo_kernel: str = "eager",
        n_global_vectors: int = 1,
        n_block_vecs: int = 0,
        n_block_tens: int = 0,
        routing_norm: str = "cosine",
        routing_temperature: float = 10.0,
        geo_bias: str = "linear",
        local_rho: float | None = None,
        local_weights: str = "gaussian",
        local_gradient: bool = False,
    ) -> None:
        super().__init__()
        self.use_relational_geo = use_relational_geo
        self.fast_point_softmax = bool(fast_point_softmax)
        ### LOCAL-AGG (2026-09-27): optional local relational aggregation among the
        ### tokens before the slice routing (_local_region); None disables it and the
        ### block is the released arithmetic. LOCAL-LPR (2026-09-28): local_weights
        ### "lpr" = polynomial-reproduction stencil weights, local_gradient = the
        ### projected finite-difference channels (k-nearest path only).
        self.local_rho = None if local_rho is None else float(local_rho)
        self.local_weights = str(local_weights)
        self.local_gradient = bool(local_gradient)
        if self.local_rho is not None:
            self.local_norm = nn.LayerNorm(hidden)
            self.local_value = nn.Linear(hidden, hidden)
            self.local_geo = nn.Linear(5 + n_global_vectors, hidden // 2)
            pooled_width = hidden * (2 + n_global_vectors) if self.local_gradient else hidden
            self.local_out = nn.Linear(pooled_width + hidden // 2, hidden)
        ### KERNEL STUDY (2026-09-13): "fused" evaluates the dense geometry region
        ### (invariants -> routing bias -> point->slice mix -> pooled invariants) in
        ### one Triton kernel per direction that never materializes a (B,N,S,.)
        ### tensor (see geo_kernel.py); same arithmetic as _geo_region, recompute
        ### built in, so geo_checkpoint is moot on that path.
        self.geo_kernel = geo_kernel
        self.n_geo = _geo_width(n_global_vectors, n_block_vecs, n_block_tens)
        ### Activation recompute (2026-09-07 memory attribution): the per-slice
        ### geometry tensors -- rel (B,N,S,3), rel_hat, dist and two bf16 copies
        ### of the (B,N,S,8) invariants -- are 76% of ISLA's saved activations
        ### at 10k tokens. With geo_checkpoint the invariants are rebuilt from
        ### (r, n_hat, g_hat, z_pos, m_s) inside backward instead of stored;
        ### the forward is bitwise unchanged (same ops, same order).
        self.geo_checkpoint = bool(geo_checkpoint)
        self.norm_assign = nn.LayerNorm(hidden)
        self.assign = _RoutingLogits(
            hidden, n_slices, routing_norm, routing_temperature
        )
        ### Relational geometry (v2): per-slice equivariant anchors and
        ### point-anchor invariants -- many local, data-adaptive reference
        ### points instead of any global frame (smooth by construction).
        ### Built only when used: parameters that never receive gradients
        ### crash DDP (A35b nogeo, 2026-09-05).
        self.geo_width = hidden // 2
        if use_relational_geo:
            self.geo_logit = _geo_bias_head(self.n_geo, geo_bias)
            self.geo_feat = nn.Linear(self.n_geo, self.geo_width)
        self.slice_mlp = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )
        self.broadcast = nn.Linear(2 * hidden + hidden // 2, hidden)
        self.norm_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )

    def forward(
        self,
        h: Float[torch.Tensor, "batch tokens hidden"],
        log_w: Float[torch.Tensor, "batch tokens 1"],
        r: Float[torch.Tensor, "batch tokens dim"],
        n_hat: Float[torch.Tensor, "batch tokens dim"],
        g_hat: Float[torch.Tensor, "batch tokens vectors dim"],
        eps: float,
        vecs=None,
        tens=None,
        nbr_idx=None,
    ) -> Float[torch.Tensor, "batch tokens hidden"]:
        if self.local_rho is not None:
            hv = self.local_value(self.local_norm(h))
            if nbr_idx is not None:
                pooled_h, pooled_geo = _local_region_knn(
                    hv, log_w, r, n_hat, g_hat, nbr_idx, self.local_rho, eps, self.local_weights, self.local_gradient
                )
            else:
                if self.local_weights != "gaussian" or self.local_gradient:
                    raise ValueError("local_weights='lpr' and local_gradient need the k-nearest path (local_k > 0)")
                pooled_h, pooled_geo = _local_region(hv, log_w, r, n_hat, g_hat, self.local_rho, eps)
            h = h + self.local_out(torch.cat([pooled_h, self.local_geo(pooled_geo.to(h.dtype))], dim=-1))
        ### Soft assignment of points to slices; measure weights enter as a
        ### log-space bias so slice states are quadrature-weighted means.
        logits = self.assign(self.norm_assign(h))  # (B, N, S)
        a = _softmax_over_points(
            logits + log_w, self.fast_point_softmax
        )  # normalized over points
        ### Equivariant anchors: weighted mean position AND mean normal
        ### direction per slice (v3b) -- anchors gain orientation.
        z_pos = torch.einsum("bns,bnc->bsc", a, r)  # (B, S, d)
        m_s = torch.einsum("bns,bnc->bsc", a, n_hat)
        m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(eps)
        ### Geometry refines the routing and the readback. A35b ablation:
        ### use_relational_geo=False removes the anchor-relational invariants
        ### from routing and readback (Transolver-style feature-only slicing).
        if self.use_relational_geo:
            ### Pool the invariants over slices FIRST, then project: exactly
            ### equal to projecting then pooling (the projection is affine and
            ### point_mix sums to one over slices), but the saved activation is
            ### (B, N, geo) instead of (B, N, S, hidden/2) -- ~0.5 GB per layer at
            ### 10k tokens, 256 slices, hidden 192 (A35b memory derivation).
            if self.geo_kernel == "fused":
                ### The fused kernel takes the global directions and the token's rank-1
                ### vectors as one per-point vector set (at most four in all, no rank-2
                ### inputs; enforced at construction / resolution).
                bias, point_mix, pooled = _fused_geo_region()(
                    self.geo_logit,
                    logits,
                    r,
                    n_hat,
                    _fused_vectors(g_hat, vecs),
                    z_pos,
                    m_s,
                    eps,
                    True,
                    n_global=g_hat.shape[2],
                )
            elif self.geo_checkpoint:
                bias, point_mix, pooled = checkpoint(
                    _geo_region,
                    self.geo_logit,
                    logits,
                    r,
                    n_hat,
                    g_hat,
                    z_pos,
                    m_s,
                    eps,
                    vecs,
                    tens,
                    use_reentrant=False,
                )
            else:
                bias, point_mix, pooled = _geo_region(
                    self.geo_logit,
                    logits,
                    r,
                    n_hat,
                    g_hat,
                    z_pos,
                    m_s,
                    eps,
                    vecs,
                    tens,
                )
            logits = logits + bias
        else:
            point_mix = torch.softmax(logits, dim=-1)  # normalized over slices
        a = _softmax_over_points(logits + log_w, self.fast_point_softmax)
        z = torch.einsum("bns,bnh->bsh", a, h)  # slice states
        z = z + self.slice_mlp(z)
        back = torch.einsum("bns,bsh->bnh", point_mix, z)
        if self.use_relational_geo:
            geo_pool = self.geo_feat(pooled)
        else:
            geo_pool = h.new_zeros(h.shape[0], h.shape[1], self.geo_width)
        h = h + self.broadcast(torch.cat([h, back, geo_pool], dim=-1))
        return h + self.mlp(self.norm_mlp(h))


class _ReadBlock(nn.Module):
    """Passive decoder layer (v5a): queries read encoder slices/anchors,
    never write. Removing the write-back is what makes predictions at one
    query independent of every other query (given a fixed source sample)."""

    # v5a2: the full v3b relational-feature set (thin decoder pipes collapse
    # training -- measured twice now, v3a and v5a-v1)

    def __init__(
        self,
        hidden: int,
        n_slices: int,
        mlp_ratio: int = 4,
        geo_checkpoint: bool = False,
        geo_kernel: str = "eager",
        n_global_vectors: int = 1,
        n_block_vecs: int = 0,
        n_block_tens: int = 0,
        routing_norm: str = "cosine",
        routing_temperature: float = 10.0,
        geo_bias: str = "linear",
    ) -> None:
        super().__init__()
        self.geo_checkpoint = bool(geo_checkpoint)
        self.geo_kernel = geo_kernel  # see _SliceBlock
        self.n_geo = n_geo = _geo_width(n_global_vectors, n_block_vecs, n_block_tens)
        self.norm = nn.LayerNorm(hidden)
        self.assign = _RoutingLogits(
            hidden, n_slices, routing_norm, routing_temperature
        )
        self.geo_logit = _geo_bias_head(n_geo, geo_bias)
        self.geo_feat = nn.Linear(n_geo, hidden // 2)
        self.broadcast = nn.Linear(2 * hidden + hidden // 2, hidden)
        self.local_read = nn.Linear(2 * hidden, hidden)
        self.norm_mlp = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden, hidden),
        )

    def forward(
        self,
        q_h,
        q_r,
        q_n,
        q_g,
        z_states,
        z_pos,
        m_s,
        eps,
        src_r=None,
        src_h=None,
        src_w=None,
        local_rho=None,
        q_vecs=None,
        q_tens=None,
    ):
        logits_pre = self.assign(self.norm(q_h))
        if self.geo_kernel == "fused":
            _, mix, pooled = _fused_geo_region()(
                self.geo_logit,
                logits_pre,
                q_r,
                q_n,
                _fused_vectors(q_g, q_vecs),
                z_pos,
                m_s,
                eps,
                True,
                n_global=q_g.shape[2],
            )
        elif self.geo_checkpoint:
            _, mix, pooled = checkpoint(
                _geo_region,
                self.geo_logit,
                logits_pre,
                q_r,
                q_n,
                q_g,
                z_pos,
                m_s,
                eps,
                q_vecs,
                q_tens,
                use_reentrant=False,
            )
        else:
            _, mix, pooled = _geo_region(
                self.geo_logit,
                logits_pre,
                q_r,
                q_n,
                q_g,
                z_pos,
                m_s,
                eps,
                q_vecs,
                q_tens,
            )
        back = torch.einsum("bqs,bsh->bqh", mix, z_states)
        geo_pool = self.geo_feat(pooled)  # pool-then-project (exact)
        q_h = q_h + self.broadcast(torch.cat([q_h, back, geo_pool], dim=-1))
        if src_h is not None:
            ### v5a3: local token readout -- the per-point detail 256 slice
            ### states cannot carry. Measure-weighted Gaussian kernel over
            ### SOURCE positions attending to encoder states; queries still
            ### never write, so query-independence is preserved exactly.
            local = _kernel_readout(q_r, src_r, src_h, src_w, local_rho, eps)
            q_h = q_h + self.local_read(torch.cat([q_h, local], dim=-1))
        return q_h + self.mlp(self.norm_mlp(q_h))


def _kernel_readout(q_r, src_r, src_h, src_logw, rho, eps):
    """Measure-weighted Gaussian-kernel average of source states at query
    positions, row-chunked; a pure function of the source (passive).

    Evaluated as a softmax over sources with logits ``-d^2 / rho^2 + log w``:
    exactly invariant to rescaling the source measure, well defined for a
    query arbitrarily far from every source (it attends to the nearest ones
    instead of underflowing), and equal to the plain weighted average where
    the kernel mass is not tiny. ``src_logw`` is the log measure weight.
    """
    b, nq, _ = q_r.shape
    outs = []
    chunk = 4096

    def _chunk(q_chunk, src_r_, src_h_, src_logw_):
        d2 = torch.cdist(q_chunk, src_r_).square()
        logits = -d2 / (rho * rho) + src_logw_[:, None, :]
        k = torch.softmax(logits, dim=-1)
        return torch.einsum("bcn,bnh->bch", k, src_h_)

    ### Memory (2026-09-21): autograd would otherwise keep the (chunk, N) kernel of every chunk for the backward
    ### pass, about 3 GB per chunk at 200,000 sources and several hundred GB per step at 200,000 queries. The
    ### kernel is a cheap function of the positions, so under grad it is rebuilt in backward instead of stored;
    ### the forward is unchanged.
    for i0 in range(0, nq, chunk):
        q_chunk = q_r[:, i0 : i0 + chunk]
        if torch.is_grad_enabled():
            outs.append(
                checkpoint(_chunk, q_chunk, src_r, src_h, src_logw, use_reentrant=False)
            )
        else:
            outs.append(_chunk(q_chunk, src_r, src_h, src_logw))
    return torch.cat(outs, dim=1)


def rms_pairwise_distance(
    points: torch.Tensor, measure_weights: torch.Tensor | None, eps: float = 1e-8
) -> torch.Tensor:
    """Measure-weighted RMS pairwise distance of a sampled boundary, shape (B, 1, 1).

    ``L^2 = sum_ij w~_i w~_j |x_i - x_j|^2`` with ``w~ = w / sum(w)`` (uniform
    weights when ``measure_weights`` is ``None``), evaluated in O(N) through the
    variance identity ``L^2 = 2 sum_i w~_i |x_i - c|^2`` with ``c`` the weighted
    mean. A normalized two-point statistic: it converges under refinement of any
    boundary whose normalized measure converges (including fractal boundaries
    whose total measure diverges), is invariant to a uniform rescale of the
    weights, and is covariant under a geometric rescale. The mean ``c`` is an
    intermediate of this scalar only; no centre is exposed.
    """
    b, n = points.shape[0], points.shape[1]
    if measure_weights is None:
        w = points.new_ones(b, n, 1)
    else:
        w = measure_weights.reshape(b, n, 1).to(points.dtype)
    w_n = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
    c = (w_n * points).sum(dim=1, keepdim=True)
    var = (w_n * (points - c).square().sum(-1, keepdim=True)).sum(dim=1, keepdim=True)
    return (2.0 * var).sqrt().clamp_min(eps)


#: Memory budget, in bytes, of one chunk of the nearest-boundary distance block
#: when ``interior_nn_chunk`` is not given (2 GiB of float64: about 1,300 queries
#: per chunk against 200,000 boundary points; the search is memory-bound, so
#: fewer, larger chunks are faster).
NN_CHUNK_BYTES = 2 * 2**30


def _nn_kernel_available() -> bool:
    """Whether the Triton nearest-neighbour kernel can run (imported lazily so
    the module loads without triton)."""
    try:
        from .nn_kernel import HAS_TRITON
    except Exception:  # pragma: no cover
        return False
    return bool(HAS_TRITON)


def nearest_boundary_geometry(
    query_points: Float[torch.Tensor, "batch queries dim"],
    points: Float[torch.Tensor, "batch tokens dim"],
    normals: Float[torch.Tensor, "batch tokens dim"],
    *,
    chunk: int | None = None,
    eps: float = 1e-12,
    projection: str = "centroid",
) -> tuple[
    Float[torch.Tensor, "batch queries"], Float[torch.Tensor, "batch queries dim"]
]:
    r"""Signed distance and direction from the nearest boundary sample to each query.

    For every query :math:`x_q` the nearest boundary point :math:`y_{i^*}` is
    found by an exact brute-force search: on CUDA with Triton a fused kernel
    (:mod:`nn_kernel`) streams the samples past ``BLOCK_Q`` queries at a time
    and keeps the running minimum of the exact float32 ``|x_q - y_i|^2``, so no
    ``N_q x N`` block is ever written; elsewhere a chunked float64 matmul
    expansion ``|y|^2 - 2 x.y`` on centred, scaled coordinates (whose distance
    resolution, ``sqrt(eps64) R ~ 1e-8 R``, is below the float32 resolution of
    the coordinates) selects the same sample; then

    .. math::

        u_q = \frac{x_q - y_{i^*}}{\lvert x_q - y_{i^*} \rvert}, \qquad
        d_q = \operatorname{sign}(u_q \cdot \hat n_{i^*})\,\lvert x_q - y_{i^*} \rvert ,

    the direction from the boundary to the query and the distance signed by
    the nearest sample's normal: **positive on the side the normal points to**
    (outside, for outward normals), which is the convention of the stored
    signed-distance fields of the program's datasets. A query that coincides
    with its nearest sample takes that sample's normal as :math:`u_q` and
    :math:`d_q = 0`. Both quantities are functions of pairwise differences,
    so :math:`d_q` is invariant and :math:`u_q` covariant under every rigid
    motion and reflection, and :math:`d_q` scales with the geometry.

    The search runs without gradient (an argmin); :math:`d_q, u_q` are then
    rebuilt from the gathered sample, so they are differentiable in the
    positions. ``normals`` are renormalized. ``chunk`` is the number of
    queries per distance block of the float64 fallback; ``None`` derives it
    from :data:`NN_CHUNK_BYTES`.

    Because the boundary is a *sample* (200,000 of up to 17 million cells on
    a DrivAerML car), :math:`\lvert d_q \rvert` overestimates the true wall
    distance by up to about the sample spacing near the wall and :math:`u_q`
    deviates from the true SDF gradient by the same order; far from the wall
    both agree with the full-mesh field.

    ``projection="plane"`` (INTERNAL-GEOMETRY-PROJ, 2026-09-25) measures the
    distance to the nearest sample's *tangent plane* instead of to the sample
    itself: :math:`d_q = (x_q - y_{i^*}) \cdot \hat n_{i^*}` and
    :math:`u_q = \operatorname{sign}(d_q)\, \hat n_{i^*}`, the direction from
    the foot point on that plane. For a query whose nearest surface point lies
    inside the nearest cell this is the true wall distance to first order in the
    cell size (exact on a flat wall); at edges and corners it can understate it.
    The INTERNAL-GEOMETRY verdict (@sec-nb-internal-geometry-verdict) located
    the internal geometry's velocity price in the centroid measurement, which
    this option removes.
    """
    if projection not in ("centroid", "plane"):
        raise ValueError(f"projection must be 'centroid' or 'plane', got {projection!r}")
    b, nq, _ = query_points.shape
    n = points.shape[1]
    with torch.autocast(query_points.device.type, enabled=False), torch.no_grad():
        if query_points.is_cuda and points.shape[-1] == 3 and _nn_kernel_available():
            ### The Triton search is written for three components; other dimensions
            ### take the exact float64 expansion below.
            from .nn_kernel import nearest_index

            idx = nearest_index(query_points, points)  # (B, Q)
        else:
            ### Float64 expanded form |y|^2 - 2 x.y (|x|^2 is constant per query) on
            ### coordinates centred and scaled by the boundary sample. The expansion
            ### cancels catastrophically: its squared-distance error is ~eps R^2, a
            ### distance resolution of sqrt(eps) R, which at fp32 would be 2e-4 R
            ### (coarser than the near-wall sample spacing) and at fp64 is 1e-8 R
            ### (below the fp32 resolution of the coordinates themselves). The exact
            ### element-wise cdist path is ~100x slower at 200,000 x 200,000
            ### (46 s per training step, INTERNAL-GEOMETRY lanes, 2026-09-23).
            if chunk is None:
                chunk = max(1, NN_CHUNK_BYTES // (b * n * 8))  # the block is float64
            c = points.mean(dim=1, keepdim=True)
            scale = (points - c).norm(dim=-1).amax(dim=1).clamp_min(eps)[:, None, None]
            y_s = ((points - c) / scale).double()  # (B, N, d)
            y_sq = y_s.square().sum(-1)  # (B, N)
            y_t = y_s.transpose(1, 2)  # (B, d, N)
            idx = []
            for i0 in range(0, nq, chunk):
                q_s = ((query_points[:, i0 : i0 + chunk] - c) / scale).double()
                approx = torch.baddbmm(
                    y_sq[:, None, :].expand(b, q_s.shape[1], n), q_s, y_t, beta=1.0, alpha=-2.0
                )  # (B, q, N), one fused pass
                idx.append(approx.argmin(dim=-1))
            idx = torch.cat(idx, dim=1)  # (B, Q)
    gather = idx[..., None].expand(b, nq, points.shape[-1])
    y = torch.gather(points, 1, gather)
    n_star = torch.gather(normals, 1, gather)
    n_star = n_star / n_star.norm(dim=-1, keepdim=True).clamp_min(eps)
    diff = query_points - y
    if projection == "plane":
        d_plane = (diff * n_star).sum(-1, keepdim=True)
        sign = torch.where(d_plane >= 0, 1.0, -1.0).to(d_plane)
        return d_plane.squeeze(-1), sign * n_star
    dist = diff.norm(dim=-1, keepdim=True)
    on_wall = dist <= eps
    u = torch.where(on_wall, n_star, diff / dist.clamp_min(eps))
    ### The tangent case (u . n = 0) counts as the positive side, so d never
    ### vanishes off the wall.
    sign = torch.where((u * n_star).sum(-1, keepdim=True) >= 0, 1.0, -1.0).to(dist)
    d = torch.where(on_wall, torch.zeros_like(dist), sign * dist)
    return d.squeeze(-1), u


### ---------------------------------------------------------------------------
### Output heads: the rank-r equivariant expansion
### ---------------------------------------------------------------------------


def _levi_civita(dim: int) -> torch.Tensor:
    """The Levi-Civita symbol of ``dim`` dimensions, shape ``(dim,)*dim``: the sign
    of the permutation at each permuted multi-index, zero elsewhere. An isotropic
    pseudotensor of rank ``dim`` in every dimension (``J`` at ``dim = 2``, the
    familiar rank-3 symbol at ``dim = 3``)."""
    eps = torch.zeros((dim,) * dim, dtype=torch.float64)
    for perm in itertools.permutations(range(dim)):
        inversions = sum(1 for i in range(dim) for j in range(i + 1, dim) if perm[i] > perm[j])
        eps[perm] = -1.0 if inversions % 2 else 1.0
    return eps


def _isotropic_tensors(rank: int, symmetric: bool, parity: str, dim: int = 3) -> list[torch.Tensor]:
    """The constant (isotropic) tensors of a rank and parity in ``dim`` spatial
    dimensions, each of shape ``(dim,)*rank``.

    Even parity: the identity at rank 2; at rank 4 the three products of
    Kronecker deltas ``d_ij d_kl, d_ik d_jl, d_il d_jk`` (their sum alone for a
    symmetric spec, the only fully symmetric isotropic rank-4 tensor). Odd
    parity: the Levi-Civita symbol, which has rank ``dim`` (rank 2 in two
    dimensions, rank 3 in three; a pseudotensor, antisymmetric, so not for a
    symmetric spec). No isotropic tensor exists at ranks 0, 1, at even rank 3,
    or at odd ranks other than ``dim``.
    """
    eye = torch.eye(dim, dtype=torch.float64)
    if parity == "even":
        if rank == 2:
            return [eye]
        if rank == 4:
            terms = [
                torch.einsum("ij,kl->ijkl", eye, eye),
                torch.einsum("ik,jl->ijkl", eye, eye),
                torch.einsum("il,jk->ijkl", eye, eye),
            ]
            return [sum(terms)] if symmetric else terms
        return []
    if rank == dim and not symmetric:
        return [_levi_civita(dim)]
    return []


class _OutputLayout:
    """Coefficient bookkeeping of one output field.

    With ``m`` true and ``p`` pseudo basis vectors per token, a rank-``r``
    output of even parity is expanded in tensor products of ``r`` true vectors
    (``m**r`` ordered products, or one per unordered multi-index for a
    symmetric spec), one of odd parity in products with exactly one
    pseudovector factor (``r p m**(r-1)`` ordered; ``p`` times the unordered
    multisets of ``r-1`` true vectors for a symmetric spec), plus the isotropic
    tensors of that rank and parity. ``index`` maps every ordered multi-index
    over the relevant vector set (the true vectors alone for even parity, true
    then pseudo for odd) to its free coefficient, or to the trailing zero slot.
    """

    def __init__(self, spec: RankSpec, n_true: int, n_pseudo: int, dim: int = 3) -> None:
        self.spec = spec
        self.dim = int(dim)
        r = spec.rank
        self.n_true, self.n_pseudo = n_true, n_pseudo
        self.n_vec = n_true if spec.parity == "even" else n_true + n_pseudo
        if r == 0:
            self.n_free, self.index, self.iso = 1, None, []
            self.n_coeffs = 1
            return
        free: dict[tuple[int, ...], int] = {}
        index = []
        for j in itertools.product(range(self.n_vec), repeat=r):
            n_odd = sum(1 for a in j if a >= n_true)
            valid = n_odd == (0 if spec.parity == "even" else 1)
            if not valid:
                index.append(-1)
                continue
            key = tuple(sorted(j)) if spec.symmetric else j
            index.append(free.setdefault(key, len(free)))
        self.n_free = len(free)
        ### -1 (no coefficient) points at the appended zero slot n_free.
        self.index = torch.tensor(
            [i if i >= 0 else self.n_free for i in index], dtype=torch.long
        )
        self.iso = _isotropic_tensors(r, spec.symmetric, spec.parity, self.dim)
        self.n_coeffs = self.n_free + len(self.iso)


def _assemble_output(coeffs, layout: _OutputLayout, true_vecs, pseudo_vecs):
    """Contract one output's coefficients ``(B, N, n_coeffs)`` with the token basis.

    ``true_vecs`` is ``(B, N, m, d)``, ``pseudo_vecs`` ``(B, N, p, d)``. Returns
    ``(B, N)`` at rank 0 and ``(B, N, d, ..., d)`` with ``rank`` trailing axes
    otherwise. The full coefficient tensor over ordered multi-indices is
    gathered from the free coefficients and contracted one index at a time,
    so the basis itself is never materialized (peak ``(B, N, n_vec**(r-1), d)``).
    """
    b, n = coeffs.shape[:2]
    r = layout.spec.rank
    if r == 0:
        return coeffs[..., 0]
    u = (
        true_vecs
        if layout.spec.parity == "even"
        else torch.cat([true_vecs, pseudo_vecs], dim=2)
    )
    c_free = coeffs[..., : layout.n_free]
    c_full = torch.cat([c_free, c_free.new_zeros(b, n, 1)], dim=-1)[
        ..., layout.index.to(coeffs.device)
    ]
    x = c_full  # (B, N, n_vec**r), indices (j1, ..., jr) row-major
    for _ in range(r):
        x = x.reshape(b, n, layout.n_vec, -1)  # split j1 from the rest
        x = torch.einsum("bnjR,bnja->bnRa", x, u)  # (B, N, rest, d)
        x = x.reshape(b, n, -1)  # (rest, a1) flattened: j2 leads again
    out = x.reshape(b, n, *layout.spec.shape(layout.dim))
    if layout.iso:
        iso = torch.stack(layout.iso).to(device=out.device, dtype=out.dtype)
        c_iso = coeffs[..., layout.n_free :].to(out.dtype)
        out = out + torch.einsum("bnk,kR->bnR", c_iso, iso.flatten(1)).reshape(
            out.shape
        )
    return out


def _polar_complement(r_hat: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The component of ``v`` orthogonal to the unit vector ``r_hat``, with the
    sign convention of the polar basis vector ``e_theta`` of
    :func:`physicsnemo.nn.functional.equivariant_ops.spherical_basis` (``-v``
    projected off ``r_hat``; zero where ``r_hat`` is zero), in any dimension.
    Left unnormalized, so it vanishes smoothly where ``v`` is parallel to
    ``r_hat``. In three dimensions this is exactly that function's ``e_theta``."""
    e = vector_project(-v, r_hat)
    zero = torch.all(r_hat == 0.0, dim=-1, keepdim=True)
    return torch.where(zero, torch.zeros_like(e), e)


def _pseudovectors(primaries: list[torch.Tensor], dim: int) -> list[torch.Tensor]:
    """The pseudovector basis from a list of true vectors ``(..., dim)`` by the
    Levi-Civita construction: in three dimensions the pairwise cross products
    (in list order, ``i < j``), in two the perpendiculars ``J v = (-v_y, v_x)`` of
    each vector. A pseudovector in ``dim`` dimensions is the contraction of the
    Levi-Civita symbol with ``dim - 1`` vectors; other dimensions return no
    pseudovectors (odd-parity outputs are refused there at construction)."""
    if dim == 3:
        return [
            torch.cross(primaries[i], primaries[j], dim=-1)
            for i in range(len(primaries))
            for j in range(i + 1, len(primaries))
        ]
    if dim == 2:
        return [torch.stack([-v[..., 1], v[..., 0]], dim=-1) for v in primaries]
    return []


### ---------------------------------------------------------------------------
### The model
### ---------------------------------------------------------------------------


class ISLA(Module):
    r"""ISLA (Invariant Slice Attention): invariant backbone, equivariant edges.

    A soft-slice transformer for steady boundary-value problems whose attention
    operates only on invariants of the per-point vector set and whose outputs of
    any rank are re-assembled in an equivariant basis at the heads, so exact
    rotation, reflection and translation covariance holds by construction (see
    the module docstring for the contracts and the reference configuration).
    Every argument is keyword-only.

    **Rank specs.** ``outputs``, ``boundary_inputs``, ``query_inputs`` and
    ``global_inputs`` map a field name to a rank spec: an ``int`` rank, or a
    mapping ``{"rank": r, "symmetric": False, "parity": "even", "channels":
    1}``. ``symmetric`` means fully symmetric under index permutation (rank
    :math:`\ge 2`); ``parity="odd"`` declares a pseudotensor output (outputs
    only); ``channels`` packs several independent scalars under one rank-0 name.
    A name may contain ``"."`` to denote nesting (``"surface.pressure"``); every
    value is a leaf spec, so a mapping value is never a group.

    Args:
        outputs: Output fields, ``{name: rank spec}``. ``forward`` returns a
            :class:`~tensordict.TensorDict` of batch size ``(B, N)`` with these
            names as (nested) leaves; a rank-0 field has shape ``(B, N)`` (or
            ``(B, N, channels)`` when ``channels > 1``), a rank-``r`` field
            ``(B, N, d, ..., d)`` with ``r`` trailing axes (``d`` the spatial
            dimension, ``dim``). Ranks up to 4. Each
            field's coefficients come from the shared linear head; with ``m =
            2K + 3`` true basis vectors and ``p`` pseudovectors (``(K+2)(K+1)/2``
            in three dimensions, ``K + 2`` in two) the counts per field are ``1`` at rank 0; ``m`` (even) or ``p``
            (odd) at rank 1; ``m^2 + 1`` (even, the identity included) or
            ``m(m+1)/2 + 1`` (even symmetric), ``2pm`` or ``pm`` (odd) at rank
            2; ``m^3`` or ``C(m+2, 3)`` (even), ``3pm^2 + 1`` (odd, the
            Levi-Civita symbol included) or ``p\,m(m+1)/2`` (odd symmetric) at
            rank 3; ``m^4 + 3`` or ``C(m+3, 4) + 1`` (even), ``4pm^3`` or
            ``p\,C(m+2, 3)`` (odd) at rank 4. For ``K = 1`` (``m = 5, p = 3``):
            a vector costs 5, a symmetric rank-2 tensor 16, a pseudovector 3.
        boundary_inputs: Per-cell boundary fields, ``{name: rank spec}`` with
            ranks 0 to 2 and even parity (a prescribed traction as ``1``, a
            Dirichlet trace or a condition tag as ``0``). Their invariants
            widen the boundary tokens' seed; tokens that are not boundary cells
            carry zeros there. Rank-1 and rank-2 inputs also contract with the
            token-to-anchor direction in every slice layer (module docstring).
        query_inputs: Optional per-query fields for the interior modes, ``{name:
            rank spec}`` with ranks 0 to 2 (a material label, a source term; the
            wall distance and direction are built in under the default
            ``interior_geometry``). Carried by
            the query tokens, the passive queries and the support tokens
            through their own embedding; the same invariants as for boundary
            inputs. Rank-0 entries are scaled by ``query_scalar_scale``.
        global_inputs: Per-case fields, ``{name: rank spec}`` with rank 0
            (appended to every token's seed; ``channels`` for a parameter
            vector) or rank 1 (a global direction, normalized to unit length
            inside the model). The ``K`` rank-1 entries, in declaration order,
            are the global directions of the geometry region and the head basis.
        hidden: Token width of the slice blocks. The defaults ``hidden=384``,
            ``n_layers=12`` and ``n_slices=256`` are the book's reference
            configuration (REF verdict, 2026-09-23), together with the other
            defaults below: cosine routing, the RMS-pairwise-distance length
            unit, the automatic geometry kernel and ``source_total`` query
            mass. The reference training rate is 5e-4.
        n_layers: Number of slice-attention blocks in the encoder.
        n_slices: Number of soft slices (data-adaptive anchors) per block.
        mlp_ratio: Expansion ratio of the per-token and per-slice MLPs.
        use_measure_weights: Route with the quadrature measure as a log-space
            bias so slice states are measure-weighted (quadrature) means. ``False``
            is the "weights-off" ablation, which reads the sampling density.
        fast_point_softmax: Use the row-parallel softmax kernel over points
            (``True``, the default since 2026-09-10) or the reference
            middle-dimension kernel (roundoff-level difference).
        use_relational_geo: Feed the point-anchor relational invariants to the
            routing bias and the read-back (``False`` is the feature-only
            slicing ablation).
        query_independent: Decode queries passively through read blocks, so a
            prediction at one point does not depend on which other points are
            queried (given the boundary sample).
        n_decoder_layers: Number of passive read blocks when
            ``query_independent`` is set.
        local_readout_rho: Radius (in frame units) of the measure-weighted
            Gaussian local readout of the passive decoder.
        query_tokens: Admit ``query_points`` as interacting tokens (the interior
            reference configuration).
        geo_checkpoint: Rebuild the per-slice geometric invariants in the
            backward pass instead of storing them (bitwise-identical forward and
            gradients, lower peak memory).
        query_scalar_scale: ``"length"`` divides rank-0 query inputs by the
            frame's length unit and enters ``[s, sign(s) log(|s| + eps)]``;
            ``"none"`` enters them raw. Rank-1 and rank-2 query inputs are never
            scaled.
        query_mass: Routing weight of interior/support tokens (inert without
            them): ``"source_total"`` (default) lets the tokens share a learned
            fraction of the total boundary measure, invariant to
            re-representing the same measure; ``"geometric_mean"`` gives each
            token the mean boundary log-weight plus a learned offset, which
            depends on the boundary refinement.
        support_tokens: With ``query_independent``, admit a per-case
            computational support (``support_points`` / ``support_normals`` /
            ``support_inputs``) as interacting tokens next to the boundary.
        scale_mode: The one length unit of the (origin-free) frame.
            ``"rms_distance"`` (the default): divide positions by the
            measure-weighted RMS pairwise distance of the sample,
            ``sqrt(sum_ij w_i w_j |x_i - x_j|^2)`` with the weights normalized
            to one, a *normalized* two-point statistic that converges under
            refinement of a boundary whose total measure does not (a fractal
            boundary) and that names no center; it is also invariant to a
            uniform rescale of the weights alone. ``"total_measure"`` (the
            default until 2026-09-18): divide by the square root of the total
            quadrature measure, an integral of the geometry, equivalent to
            the default within 1.3% of surface-pressure error on DrivAerML
            (LEN-RMS); needs ``measure_weights`` and, unlike the default,
            inherits any error in their absolute scale. TODO: remove this
            option; it exists only to reproduce lanes trained before
            2026-09-18 and has no use case of its own (ruling 2026-09-20).
        geo_kernel: ``None`` (default) selects automatically at the first
            forward pass: ``"fused"`` (exact Triton kernel for the per-layer
            geometry region) when the inputs are on CUDA, Triton is available,
            there are one to four per-point vectors (global directions plus rank-1
            boundary or query inputs), at most 1024 slices and no rank-2 inputs,
            else ``"eager"`` (the
            PyTorch reference region). Pass ``"eager"`` or ``"fused"`` to force
            one; ``"fused"`` raises when its constraints cannot be met.
        geo_bias: ``"linear"`` (default): the per-pair routing bias is one linear map of the
            geometric invariants. ``"mlp"``: a two-layer map (16 hidden units, GELU) of the
            same invariants, on the eager region and in the fused kernel alike. A study
            option (GEO-BIAS, 2026-09-25): does a nonlinear bias make extra global
            directions unnecessary? Exploration scale said yes for the vector field
            (0.96 of the two-axis velocity gain at K = 1) and no for pressure (0.22).
        local_rho: ``None`` (default): no local aggregation. A positive radius, in
            frame units (the ``rms_distance`` length unit), adds to every slice block
            a measure-weighted Gaussian-kernel aggregation over all tokens of the
            token states and of the token-pair invariants (distance, its log, the
            unit separation against each global direction and both normals, the
            normals' dot product), before the slice routing. Relative geometry
            only, so the contracts hold (LOCAL-AGG study, 2026-09-27).
        local_k: With ``local_rho``, the number of nearest tokens each token's kernel
            runs over (default 32; the neighbour indices are computed once per forward
            from the positions). ``0`` is the dense form over all tokens, O(N^2) per
            layer and equal to ``local_k = N``; the default is O(N k).
        local_weights: ``"gaussian"`` (default): the aggregation's weights are the
            normalized measure-weighted Gaussian kernel over the neighbours, a
            smoother whose first moment is O(h) on an asymmetric stencil, so the
            pooled value depends on which neighbours a point cloud's draw supplied.
            ``"lpr"``: local-polynomial-reproduction weights (the weighted
            least-squares fit of an affine field over the neighbours), which
            reproduce affine fields exactly and make the pooled value a consistent
            estimate of the same continuum quantity for any neighbour draw
            (LOCAL-LPR study, 2026-09-28; the meshfree GMLS construction). Needs
            ``local_k > 0``.
        local_gradient: With ``local_weights="lpr"``, also pool the fitted gradient
            of the token states projected on the receiving token's normal and on
            each global direction (``1 + K`` invariant finite-difference channels
            per hidden unit, in units of the state per ``local_rho``), the
            edge-frame directional derivative of a meshfree flux law. Default off.
        routing_norm: ``"cosine"`` (default): each block's point-to-slice
            routing logits are ``routing_temperature`` times the cosine between
            the normalized token state and a learned prototype per slice, which
            bounds every logit in ``[-T, T]`` so no block can route every point
            to a single slice (the routing-collapse state behind the mean-field
            plateau that linear routing reaches above a learning-rate ceiling
            falling like 1/width and 1/depth). ``"linear"``: an affine map of
            the normalized token state, the form of the lanes trained before
            2026-09-22; accuracy is neutral between the two at the rule rates
            (ROUTE-NORM, DrivAerML and HiLiftAeroML). The frame and head
            contracts are unchanged either way.
        routing_temperature: The bound ``T`` of the cosine routing logits.
        interior_geometry: Where an interior token's direction and wall distance
            come from. ``"nearest_boundary"`` (default): computed inside the
            model from the boundary sample by an exact chunked nearest-neighbour
            search (:func:`nearest_boundary_geometry`; direction from the nearest
            sample to the query, distance signed by that sample's normal,
            positive on the side the normal points to), and ``d / L`` with its
            signed log enters the query embedding as a built-in channel; the
            caller passes ``query_points`` / ``support_points`` only and
            ``query_normals`` / ``support_normals`` are refused.
            ``"nearest_plane"`` (INTERNAL-GEOMETRY-PROJ, 2026-09-25): the same
            search, but the distance to the nearest sample's tangent plane and
            the direction from its foot point; exact on a flat wall, understated
            at edges. Measured worse than ``"provided"`` on both interior datasets
            (2026-09-26: velocity 1.06x / 1.16x, pressure 1.12x / 1.05x the stored
            field on AirfRANS / DrivAerML Volume) and kept as a documented option
            only. Where the dataset carries a full-mesh signed-distance field and
            gradient, pass them with ``"provided"``: the internal geometries cost
            14 to 27% on interior velocity against it. The default's
            distance is that to the *sampled* boundary and overestimates the wall distance
            near the wall by up to about the sample spacing (module docstring).
            ``"provided"``: the caller supplies ``query_normals`` (and
            ``support_normals``; e.g. the stored SDF gradient) and any distance
            as a declared rank-0 ``query_inputs`` field; no built-in channel.
            The form of every interior lane trained before 2026-09-23.
        interior_nn_chunk: Queries per distance block of the nearest-neighbour
            search under ``"nearest_boundary"``; ``None`` derives it from a
            512 MiB budget (about 640 queries against 200,000 float32 boundary
            points, i.e. some 300 blocks for 200,000 queries). Cost only, never
            the result.
        dim: Spatial dimension of the points (default 3). Every geometric
            quantity is built for ``dim`` dimensions and the head's basis
            follows it (pseudovectors by the Levi-Civita construction: cross
            products in three dimensions, perpendiculars in two; odd-parity
            outputs are refused for ``dim >= 4``). The fused Triton geometry
            kernel and the Triton nearest-neighbour search are written for
            three dimensions; other dimensions run the eager region and the
            exact float64 search. A two-dimensional problem should be passed
            with ``dim=2`` rather than embedded with a zero third coordinate.
        eps: Numerical floor for norms and logarithms.

    Forward inputs (all keyword-only): ``points`` and ``normals`` of shape
    ``(B, N, d)`` with ``d = dim`` (3 by default); ``measure_weights`` ``(B, N)`` (required by
    ``scale_mode="rms_distance"`` and ``"total_measure"``; Horvitz-Thompson
    corrected so they sum to the boundary measure); ``global_inputs`` ``{name:
    tensor}`` with a rank-0 field of shape ``(B,)`` or ``(B, channels)`` and a
    rank-1 field ``(B, d)`` (a single case's values, shape ``(d,)`` or
    ``(channels,)``, is shared across the batch); ``boundary_inputs`` ``{name:
    tensor}`` with shapes ``(B, N)`` / ``(B, N, channels)``, ``(B, N, d)``,
    ``(B, N, d, d)`` by rank; ``query_points`` ``(B, Q, d)`` and
    ``query_inputs`` (shapes as for boundary inputs over ``Q``) for the
    interior modes, plus ``query_normals`` ``(B, Q, d)`` under
    ``interior_geometry="provided"`` only; ``support_points`` ``(B, M, d)``
    (``support_normals`` likewise) and ``support_inputs`` (the ``query_inputs``
    fields over ``M``) with ``support_tokens``. Each named mapping may be flat
    with dotted names, nested, or a TensorDict. Returns a
    :class:`~tensordict.TensorDict` of batch size ``(B, N_out)`` whose leaves
    are the ``outputs`` names unflattened on ``"."``, with ``N_out`` the
    boundary token count, or the query count in the interior modes.

    Output contract: a field declared with rank ``r`` and even parity
    transforms as a true rank-``r`` tensor under every orthogonal map of the
    inputs (``T -> R^{(x)r} T``; rotations and reflections alike) and is
    invariant under translations; an odd field picks up the extra sign
    ``det R``. Components in any frame are only ever a view of such a field.
    """

    class MetaData(ModelMetaData):
        jit: bool = False
        amp: bool = True

    #: Largest output rank; the ordered basis grows as ``m**rank``.
    MAX_OUTPUT_RANK = 4
    #: Largest per-token input rank; see the module docstring.
    MAX_INPUT_RANK = 2

    def __init__(
        self,
        *,
        outputs: Mapping[str, int | Mapping],
        boundary_inputs: Mapping[str, int | Mapping] | None = None,
        query_inputs: Mapping[str, int | Mapping] | None = None,
        global_inputs: Mapping[str, int | Mapping] | None = None,
        hidden: int = 384,
        n_layers: int = 12,
        n_slices: int = 256,
        mlp_ratio: int = 4,
        use_measure_weights: bool = True,
        fast_point_softmax: bool = True,
        use_relational_geo: bool = True,
        query_independent: bool = False,
        n_decoder_layers: int = 4,
        local_readout_rho: float = 0.02,
        query_tokens: bool = False,
        geo_checkpoint: bool = False,
        query_scalar_scale: str = "length",
        query_mass: str = "source_total",
        support_tokens: bool = False,
        scale_mode: str = "rms_distance",
        geo_kernel: str | None = None,
        routing_norm: str = "cosine",
        routing_temperature: float = 10.0,
        interior_geometry: str = "nearest_boundary",
        interior_nn_chunk: int | None = None,
        geo_bias: str = "linear",
        local_rho: float | None = None,
        local_k: int = 32,
        local_weights: str = "gaussian",
        local_gradient: bool = False,
        dim: int = 3,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(meta=self.MetaData())
        ### ND-GENERIC (2026-09-27): the spatial dimension of the points. Every
        ### geometric quantity is built for d dimensions; the head's basis and the
        ### coefficient count depend on d (pseudovectors exist by the Levi-Civita
        ### construction: perpendiculars in two dimensions, cross products in three),
        ### which is why d is a constructor argument and forward checks the points.
        if int(dim) < 2:
            raise ValueError(f"dim must be at least 2, got {dim!r}")
        self.dim = int(dim)
        ### LOCAL-AGG (2026-09-27): radius, in frame units (the rms_distance length unit), of
        ### the local relational aggregation among tokens in every slice block; None = off.
        ### local_k > 0 restricts each token's kernel to its k nearest tokens (O(N k));
        ### local_k = 0 is the dense O(N^2) form.
        if local_rho is not None and not local_rho > 0:
            raise ValueError(f"local_rho must be positive or None, got {local_rho!r}")
        if int(local_k) < 0:
            raise ValueError(f"local_k must be 0 (dense) or positive, got {local_k!r}")
        self.local_rho = None if local_rho is None else float(local_rho)
        self.local_k = int(local_k)
        ### LOCAL-LPR (2026-09-28): stencil weights and the finite-difference channels of the
        ### local aggregation; both need the k-nearest path.
        if local_weights not in ("gaussian", "lpr"):
            raise ValueError(f"local_weights must be 'gaussian' or 'lpr', got {local_weights!r}")
        if (local_weights == "lpr" or bool(local_gradient)) and not (self.local_rho is not None and self.local_k > 0):
            raise ValueError("local_weights='lpr' and local_gradient require local_rho and local_k > 0")
        self.local_weights = str(local_weights)
        self.local_gradient = bool(local_gradient)
        ### GEO-BIAS (2026-09-25): the pair invariants -> routing-bias map; "mlp" is the
        ### eager-only nonlinear variant of the direction-count study (#sec-nb-geo-bias-prereg).
        if geo_bias not in ("linear", "mlp"):
            raise ValueError(f"geo_bias must be 'linear' or 'mlp', got {geo_bias!r}")
        self.geo_bias = geo_bias
        if interior_geometry not in ("nearest_boundary", "nearest_plane", "provided"):
            raise ValueError(
                "interior_geometry must be 'nearest_boundary', 'nearest_plane' or 'provided', "
                f"got {interior_geometry!r}"
            )
        self.interior_geometry = interior_geometry
        self.interior_nn_chunk = (
            None if interior_nn_chunk is None else int(interior_nn_chunk)
        )
        if routing_norm not in ("linear", "cosine"):
            raise ValueError(
                f"routing_norm must be 'linear' or 'cosine', got {routing_norm!r}"
            )
        self.routing_norm = routing_norm
        self.routing_temperature = float(routing_temperature)
        ### RANK SPECS (2026-09-22): every input and output is a named field of
        ### declared rank; the grammar is shared (RankSpec).
        self.output_specs = _parse_rank_specs(
            "outputs", outputs, max_rank=self.MAX_OUTPUT_RANK, allow_odd=True
        )
        if not self.output_specs:
            raise ValueError("outputs must declare at least one field")
        self.boundary_specs = _parse_rank_specs(
            "boundary_inputs",
            boundary_inputs,
            max_rank=self.MAX_INPUT_RANK,
            allow_odd=False,
        )
        self.query_specs = _parse_rank_specs(
            "query_inputs", query_inputs, max_rank=self.MAX_INPUT_RANK, allow_odd=False
        )
        self.global_specs = _parse_rank_specs(
            "global_inputs", global_inputs, max_rank=1, allow_odd=False
        )
        ### Checkpoints store the constructor arguments as JSON: keep the spec
        ### mappings there as plain dicts whatever container the caller passed
        ### (a Hydra DictConfig, for instance).
        for key, specs in (
            ("outputs", self.output_specs),
            ("boundary_inputs", self.boundary_specs),
            ("query_inputs", self.query_specs),
            ("global_inputs", self.global_specs),
        ):
            self._args["__args__"][key] = (
                {name: vars(spec) for name, spec in specs.items()} if specs else None
            )
        ### GLOBAL INPUTS (2026-09-14, ruling: the architecture targets steady
        ### boundary-value problems in general, so the problem's global data
        ### are zero or more global DIRECTIONS (each enters as a unit direction:
        ### one seed term n.g_k per token, one direction cosine rel.g_k per
        ### point-anchor pair, and the head basis gains g_k with the spherical
        ### complements of (u, g_k)) and zero or more global SCALARS (PDE
        ### parameters; appended to every token's seed features).
        self.global_vector_names = [
            n for n, s in self.global_specs.items() if s.rank == 1
        ]
        self.global_scalar_names = [
            n for n, s in self.global_specs.items() if s.rank == 0
        ]
        self.n_global_vectors = K = len(self.global_vector_names)
        self.n_global_scalars = sum(
            self.global_specs[n].channels for n in self.global_scalar_names
        )
        ### KERNEL STUDY (2026-09-13): geo_kernel selects the implementation of the
        ### per-layer geometry region, not its arithmetic. "eager" is the PyTorch
        ### region (with geo_checkpoint deciding whether its (B,N,S,.) intermediates
        ### are stored or rebuilt in backward); "fused" is the Triton kernel of
        ### geo_kernel.py, which reads the pre-logits once and writes the bias and
        ### mix once per direction, with the recompute built in (CUDA only).
        ### AUTO (2026-09-19, ruling): geo_kernel=None resolves at the first forward
        ### pass to "fused" wherever the fused kernel can run (CUDA, Triton, one
        ### global direction, <= 1024 slices, no per-token vector inputs) and to
        ### "eager" otherwise; the blocks read the resolved name. An explicit
        ### "eager" or "fused" is honored.
        if geo_kernel is None:
            geo_kernel = "auto"
        if geo_kernel not in ("auto", "eager", "fused"):
            raise ValueError(
                f"geo_kernel must be None (auto), 'eager' or 'fused', got {geo_kernel!r}"
            )
        self.geo_kernel = geo_kernel
        self.n_slices = int(n_slices)
        ### The frame (RELFRAME, 2026-09-10; the only frame since 2026-09-20):
        ### r = points / L with no centering. Every position-dependent quantity
        ### is a point-to-anchor difference (seeds reduce to n.g_k; the measure
        ### enters the routing as before), and the head's radial basis vector
        ### is the direction from the point's soft slice anchor (a
        ### measure-weighted mean, the same construction as the relational
        ### anchors), so the head stays translation covariant and exact
        ### translation invariance holds without any centering. L is a
        ### measure-weighted integral of the geometry: LEN-RMS (2026-09-18)
        ### made the RMS pairwise distance the default after validation
        ### against the total-measure unit at 435 DrivAerML cars (0.0526 vs
        ### 0.0533 under the stratified draw, seed ranges overlapping); it is
        ### normalized, center-free and finite on fractal boundaries where the
        ### total measure diverges.
        if scale_mode not in ("total_measure", "rms_distance"):
            raise ValueError(
                f"scale_mode must be 'total_measure' or 'rms_distance', got {scale_mode!r}"
            )
        self.scale_mode = scale_mode
        ### Density-factorial knob (prereg 3f4e4af7 follow-up): with False,
        ### the assignment softmax ignores quadrature weights entirely,
        ### isolating the measure-bias pathway of density sensitivity.
        self.use_measure_weights = use_measure_weights
        ### ISLA-PERF (2026-09-09): point softmaxes reduce along a contiguous last
        ### dimension (see _softmax_over_points); False reproduces the original
        ### middle-dimension kernel bitwise (roundoff-level difference otherwise).
        self.fast_point_softmax = bool(fast_point_softmax)
        self.eps = eps
        ### PER-TOKEN INPUTS (2026-09-22): the boundary cells' fields and the
        ### interior queries' fields, each set with its own invariant layout
        ### (_InputSet). Every token carries the UNION of the two block-level
        ### vector/tensor stacks (zeros where a class has no such field), so one
        ### geometry region serves boundary, support and query tokens alike.
        if query_scalar_scale not in ("length", "none"):
            raise ValueError(f"unknown query_scalar_scale {query_scalar_scale!r}")
        self.query_scalar_scale = query_scalar_scale
        n_frame = 1 + K  # {n_hat, g_1..g_K}
        self.bin = _InputSet(self.boundary_specs, n_frame, scaled_scalars=False, dim=self.dim)
        self.qin = _InputSet(
            self.query_specs, n_frame, scaled_scalars=(query_scalar_scale == "length"), dim=self.dim
        )
        self.n_block_vecs = self.bin.n_block_vecs + self.qin.n_block_vecs
        self.n_block_tens = self.bin.n_block_tens + self.qin.n_block_tens
        if geo_kernel == "fused":
            if self.dim != 3:
                raise ValueError(
                    f"geo_kernel='fused' is written for three spatial dimensions; got dim={self.dim} "
                    "(use geo_kernel='eager' or None, which resolves to the eager region)"
                )
            if self.n_block_tens:
                raise ValueError(
                    "geo_kernel='fused' does not support rank-2 boundary or query inputs "
                    "(kernel follow-up); use geo_kernel='eager' or None"
                )
            if not 1 <= K + self.n_block_vecs <= 4 or K < 1:
                raise ValueError(
                    f"geo_kernel='fused' is written for one to four per-point vectors "
                    f"(global directions plus rank-1 token inputs); got {K} global "
                    f"direction(s) and {self.n_block_vecs} token vector(s)"
                )
        ### Seed width: the K invariants n.g_k (the only origin-free scalars of
        ### {n, g_k}; positions enter through the anchors), the boundary inputs'
        ### invariants and the global scalars.
        n_seed = K + self.bin.n_features + self.n_global_scalars
        ### With no global inputs and no other seed channel the seed is empty;
        ### every token then starts from one learned embedding and all
        ### separation comes from the slice blocks' relational anchors.
        self.constant_seed = n_seed == 0
        if self.constant_seed:
            n_seed = 1
        self.embed = nn.Sequential(
            nn.Linear(n_seed, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList(
            _SliceBlock(
                hidden,
                n_slices,
                mlp_ratio,
                use_relational_geo=use_relational_geo,
                geo_checkpoint=geo_checkpoint,
                fast_point_softmax=self.fast_point_softmax,
                geo_kernel=geo_kernel,
                n_global_vectors=K,
                n_block_vecs=self.n_block_vecs,
                n_block_tens=self.n_block_tens,
                routing_norm=routing_norm,
                routing_temperature=routing_temperature,
                geo_bias=geo_bias,
                local_rho=self.local_rho,
                local_weights=self.local_weights,
                local_gradient=self.local_gradient,
            )
            for _ in range(n_layers)
        )
        ### Head frame: per-point soft slice anchor for the radial basis vector.
        ### Built only when an output of rank >= 1 reads the token basis: a scalar-only
        ### model never touches the radial direction, and a parameter that receives no
        ### gradient stops DDP at the second step (found on the OLBC Poisson lanes,
        ### 2026-09-27).
        self.needs_frame = any(spec.rank >= 1 for spec in self.output_specs.values())
        self.frame_assign = (
            _RoutingLogits(hidden, n_slices, routing_norm, routing_temperature)
            if self.needs_frame
            else None
        )
        ### v5a (query_independent): encode/decode split. Queries decode
        ### passively from final encoder slices and anchors: query-independent
        ### by construction given the source sample.
        self.query_independent = query_independent
        self.local_readout_rho = float(local_readout_rho)
        if query_independent:
            self.final_assign = nn.Sequential(
                nn.LayerNorm(hidden),
                _RoutingLogits(hidden, n_slices, routing_norm, routing_temperature),
            )
            self.read_blocks = nn.ModuleList(
                _ReadBlock(
                    hidden,
                    n_slices,
                    mlp_ratio,
                    geo_checkpoint=geo_checkpoint,
                    geo_kernel=geo_kernel,
                    n_global_vectors=K,
                    n_block_vecs=self.n_block_vecs,
                    n_block_tens=self.n_block_tens,
                    routing_norm=routing_norm,
                    routing_temperature=routing_temperature,
                    geo_bias=geo_bias,
                )
                for _ in range(n_decoder_layers)
            )
        self.norm_out = nn.LayerNorm(hidden)
        ### HEADS (2026-09-22): one linear head emits every output's invariant
        ### coefficients; each output is contracted with tensor products of the
        ### token's basis vectors (_OutputLayout / _assemble_output). True
        ### vectors: g_1..g_K, n, rhat, e_th(rhat, n), e_th(rhat, g_k), m = 2K+3 in
        ### every dimension. Pseudovectors (ND-GENERIC, 2026-09-27) by the
        ### Levi-Civita construction from the primaries {g_k, n, rhat}: their
        ### perpendiculars in two dimensions (p = K + 2), their pairwise cross
        ### products in three (p = C(K+2, 2)); a pseudovector in d dimensions needs
        ### d - 1 vectors, so odd-parity outputs are refused for d >= 4 until a
        ### dataset needs them. For K = 1 in 3-D and outputs {scalar: 0, vector: 1}
        ### the head has width 1 + 5.
        self.n_true_basis = 3 + 2 * K
        if self.dim == 3:
            self.n_pseudo_basis = (K + 2) * (K + 1) // 2
        elif self.dim == 2:
            self.n_pseudo_basis = K + 2
        else:
            self.n_pseudo_basis = 0
            odd = [n for n, sp in self.output_specs.items() if sp.parity == "odd"]
            if odd:
                raise NotImplementedError(
                    f"odd-parity outputs {odd!r} in dim={self.dim}: a pseudovector needs "
                    f"{self.dim - 1} vectors (the Levi-Civita construction); only two and "
                    "three dimensions are implemented"
                )
        self.output_layouts = {
            name: _OutputLayout(spec, self.n_true_basis, self.n_pseudo_basis, self.dim)
            for name, spec in self.output_specs.items()
        }
        self.n_head_coeffs = sum(
            self.output_layouts[name].n_coeffs * spec.channels
            for name, spec in self.output_specs.items()
        )
        self.head = nn.Linear(hidden, self.n_head_coeffs)
        ### QUERY TOKENS (boundary->interior exploration, 2026-09-07): interior
        ### query points join the encoder as INTERACTING tokens, carrying the
        ### same {q, n_q, g_k} invariant seeds as the surface tokens (the
        ### query normal must be supplied, e.g. the SDF gradient), a learned
        ### measure weight and a learned token-type offset. This is
        ### GeoTransolver's interior mechanism (queries as tokens) on ISLA's
        ### equivariant routing: exactly SE(d)-covariant, NOT query-independent.
        ### Outputs are returned for the query tokens only.
        self.query_tokens = bool(query_tokens)
        if self.query_tokens:
            if query_independent:
                raise ValueError(
                    "query_tokens is an interacting mode; set query_independent=False"
                )
            self.qt_logw = nn.Parameter(torch.zeros(1))
            self.qt_type = nn.Parameter(torch.zeros(hidden))
        ### Interior/support token measure weight. "source_total": the tokens'
        ### total weight is a learned fraction exp(logw) of the TOTAL source
        ### measure, split equally over them: invariant to any re-representation
        ### of the same discrete measure (splitting a source cell in two changes
        ### nothing) and scaling like an area, so the similarity-gauge contract
        ### holds. "geometric_mean": each token carries logw + the MEAN source
        ### log-weight, which halves when every source cell is split in two.
        if query_mass not in ("geometric_mean", "source_total"):
            raise ValueError(f"unknown query_mass {query_mass!r}")
        self.query_mass = query_mass
        ### SUPPORT TOKENS (transfer program D1, 2026-09-09; audit agenda
        ### sec-direction-support): a problem-derived SUPPORT set (e.g. a fixed
        ### per-case sample of interior points with their signed distance) joins
        ### the encoder as interacting tokens exactly like query tokens do, while
        ### the requested output points are decoded PASSIVELY through the read
        ### blocks (query_independent=True). The prediction at a query therefore
        ### cannot depend on which other queries are requested, only on the
        ### case (surface + support), which is the deployment contract the
        ### interacting query-token configuration lacks. The support tokens'
        ### total routing weight follows `query_mass` over the support set
        ### ("source_total" recommended: invariant to refinement of the source
        ### measure).
        self.support_tokens = bool(support_tokens)
        if self.support_tokens:
            if not query_independent:
                raise ValueError(
                    "support_tokens requires query_independent=True (passive read blocks decode the queries)"
                )
            if self.query_tokens:
                raise ValueError("support_tokens excludes query_tokens")
            self.sp_logw = nn.Parameter(torch.zeros(1))
            self.sp_type = nn.Parameter(torch.zeros(hidden))
        ### Per-query inputs (e.g. the signed distance to the wall, which
        ### GeoTransolver's volume configuration receives at every interior
        ### point) enter the query, passive-query and support tokens through
        ### their own embedding of the fields' invariants; every covariance
        ### contract is untouched.
        if self.qin and not (self.query_tokens or query_independent):
            raise ValueError(
                "query_inputs require query_tokens=True or query_independent=True"
            )
        ### INTERIOR GEOMETRY (2026-09-23): under "nearest_boundary" the model
        ### derives each interior token's direction (its "normal") and signed
        ### distance from the boundary sample itself (nearest_boundary_geometry);
        ### the distance enters the token's embedding as a built-in rank-0
        ### feature [d/L, sign(d) log(|d|/L + eps)] ahead of the declared
        ### query_inputs. Under "provided" the caller passes query_normals (and
        ### any distance as a declared query input), the form of every interior
        ### lane trained before this date.
        self.n_query_builtin = 0 if interior_geometry == "provided" else 2
        self.n_query_features = self.n_query_builtin + self.qin.n_features
        if self.n_query_features and (self.query_tokens or query_independent):
            width = self.n_query_features
            if self.query_tokens:
                self.qt_embed = nn.Sequential(
                    nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )
            if query_independent:
                self.rq_embed = nn.Sequential(
                    nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )
            if self.support_tokens:
                self.sp_embed = nn.Sequential(
                    nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden)
                )

    @staticmethod
    def _dots(x, g_hat):
        """x (B, N, d) against the K unit global directions g_hat (B, N, K, d) -> (B, N, K).
        Multiply-and-sum rather than einsum so autocast leaves the geometry in fp32."""
        return (x[:, :, None, :] * g_hat).sum(-1)

    def _seed_invariants(self, n_hat, g_hat):
        """The per-token seed invariants: the K terms n.g_k (an empty tensor when
        K = 0). Positions have no origin, so they carry no seed scalar; they
        enter through the point-to-anchor invariants of the slice blocks."""
        return self._dots(n_hat, g_hat)

    def _with_global_scalars(self, inv, g_scalars, n_tokens: int):
        """Append the global scalar inputs (B, S) to every token's seed features;
        substitute the constant seed when the feature set is empty."""
        b = inv.shape[0]
        if g_scalars is not None and g_scalars.shape[-1]:
            inv = torch.cat(
                [inv, g_scalars[:, None, :].expand(b, n_tokens, -1).to(inv.dtype)],
                dim=-1,
            )
        if self.constant_seed:
            inv = inv.new_ones(b, n_tokens, 1)
        return inv

    def _token_seed(self, n_hat, g_hat, b_feat, g_scalars, n_tokens: int):
        """A token class's seed: [n.g_k, boundary-input invariants (zeros for tokens
        that are not boundary cells), global scalars]."""
        inv = self._seed_invariants(n_hat, g_hat)
        if self.bin.n_features:
            if b_feat is None:
                b_feat = inv.new_zeros(inv.shape[0], n_tokens, self.bin.n_features)
            inv = torch.cat([inv, b_feat.to(inv.dtype)], dim=-1)
        return self._with_global_scalars(inv, g_scalars, n_tokens)

    def _block_extras(self, b_vecs, b_tens, q_vecs, q_tens, b: int, n: int, like):
        """The union block-level stacks of one token class: [boundary-derived vectors,
        query-derived vectors] (B, N, P', d) and [boundary tensors, query tensors]
        (B, N, Q, d, d); zeros where the class has no such field."""
        if not (self.n_block_vecs or self.n_block_tens):
            return None, None
        d = self.dim
        if b_vecs is None:
            b_vecs = like.new_zeros(b, n, self.bin.n_block_vecs, d)
            b_tens = like.new_zeros(b, n, self.bin.n_block_tens, d, d)
        if q_vecs is None:
            q_vecs = like.new_zeros(b, n, self.qin.n_block_vecs, d)
            q_tens = like.new_zeros(b, n, self.qin.n_block_tens, d, d)
        return torch.cat([b_vecs, q_vecs], 2), torch.cat([b_tens, q_tens], 2)

    def resolve_geo_kernel(self, points: torch.Tensor) -> str:
        """The geometry-kernel implementation this forward pass will use.

        ``"eager"`` or ``"fused"`` when the constructor fixed it; under the
        default (auto) the fused kernel when ``points`` is on CUDA, Triton is
        importable, the model has one to four per-point vectors (global directions
        plus rank-1 token inputs), at most 1024 slices and no rank-2 inputs, else the
        eager region. The
        decision is cached per device and logged once.
        """
        if self.geo_kernel != "auto":
            return self.geo_kernel
        key = (points.device.type, points.device.index)
        cached = getattr(self, "_geo_kernel_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from .geo_kernel import HAS_TRITON, MAX_DIRECTIONS, MAX_SLICES
        except Exception:  # pragma: no cover - a broken triton install falls back
            HAS_TRITON, MAX_SLICES, MAX_DIRECTIONS = False, 0, 0
        has_token_inputs = bool(self.n_block_vecs or self.n_block_tens)
        ok = (
            points.is_cuda
            and HAS_TRITON
            and self.dim == 3  # the fused kernel is written for three components
            and self.n_global_vectors >= 1
            and self.n_global_vectors + self.n_block_vecs <= MAX_DIRECTIONS
            and self.n_slices <= MAX_SLICES
            and not self.n_block_tens
        )
        chosen = "fused" if ok else "eager"
        self._geo_kernel_cache = (key, chosen)
        logger.info(
            "ISLA geo_kernel=auto resolved to %r on %s (cuda=%s, triton=%s, global_vectors=%d, slices=%d, token_vector_inputs=%s)",
            chosen,
            points.device,
            points.is_cuda,
            HAS_TRITON,
            self.n_global_vectors,
            self.n_slices,
            has_token_inputs,
        )
        if points.is_cuda and HAS_TRITON and has_token_inputs and chosen == "eager":
            logger.warning(
                "ISLA: rank-1 or rank-2 boundary/query inputs are present, so the fused "
                "geometry kernel is not used (kernel follow-up); running the eager region."
            )
        return chosen

    def _apply_geo_kernel(self, points: torch.Tensor) -> None:
        """Write the resolved kernel name into the blocks, once per resolution."""
        kernel = self.resolve_geo_kernel(points)
        if getattr(self, "_geo_kernel_applied", None) == kernel:
            return
        for block in self.blocks:
            block.geo_kernel = kernel
        if self.query_independent:
            for rb in self.read_blocks:
                rb.geo_kernel = kernel
        self._geo_kernel_applied = kernel

    def _validate_inputs(
        self,
        points,
        normals,
        measure_weights,
        boundary_inputs,
        query_points,
        query_normals,
        query_inputs,
        support_points,
        support_normals,
        support_inputs,
    ) -> None:
        """Shape and presence checks for :meth:`forward` (MOD-005 format)."""
        d = self.dim
        if points.ndim != 3 or points.shape[-1] != d:
            raise ValueError(
                f"Expected points of shape (B, N, {d}) (this model has dim={d}) but got tensor of shape {tuple(points.shape)}"
            )
        b, n, _ = points.shape
        if normals.shape != points.shape:
            raise ValueError(
                f"Expected normals of shape {(b, n, d)} but got tensor of shape {tuple(normals.shape)}"
            )
        if measure_weights is not None and measure_weights.numel() != b * n:
            raise ValueError(
                f"Expected measure_weights of shape {(b, n)} but got tensor of shape {tuple(measure_weights.shape)}"
            )
        self.bin.validate("boundary_inputs", boundary_inputs, b, n)
        internal = self.interior_geometry != "provided"
        for name, pts, nrm, fields in (
            ("query", query_points, query_normals, query_inputs),
            ("support", support_points, support_normals, support_inputs),
        ):
            if pts is not None and (
                pts.ndim != 3 or pts.shape[0] != b or pts.shape[-1] != d
            ):
                raise ValueError(
                    f"Expected {name}_points of shape ({b}, M, {d}) but got tensor of shape {tuple(pts.shape)}"
                )
            if nrm is not None and internal:
                raise ValueError(
                    f"{name}_normals given, but interior_geometry='nearest_boundary' derives "
                    f"the {name} directions from the boundary sample; pass "
                    "interior_geometry='provided' to supply them"
                )
            if nrm is not None and (pts is None or nrm.shape != pts.shape):
                raise ValueError(
                    f"Expected {name}_normals of shape {None if pts is None else tuple(pts.shape)} "
                    f"but got tensor of shape {tuple(nrm.shape)}"
                )
            if pts is not None and nrm is None and not internal:
                ### Passive queries that ARE the boundary points reuse the boundary
                ### normals (checked in forward); every other query set needs its own.
                if not (
                    name == "query"
                    and self.query_independent
                    and pts.shape[1] == n
                    and not self.query_tokens
                ):
                    raise ValueError(
                        f"interior_geometry='provided' needs {name}_normals (e.g. the SDF "
                        f"gradient at each {name} point); or use the default "
                        "interior_geometry='nearest_boundary'"
                    )
            ### The per-query fields are validated against the point set they
            ### belong to; passive queries that ARE the boundary points (no
            ### query_points) carry them over the boundary token count.
            if pts is not None:
                self.qin.validate(f"{name}_inputs", fields, b, pts.shape[1])
            elif fields is not None and name == "support":
                raise ValueError("support_inputs given without support_points")
            elif name == "query" and self.query_independent and (fields or self.qin):
                self.qin.validate("query_inputs", fields, b, n)
        if self.support_tokens and support_points is None:
            raise ValueError(
                "this model was built with support_tokens=True and needs support_points (and support_normals)"
            )
        if self.query_tokens and query_points is None:
            raise ValueError(
                "this model was built with query_tokens=True and needs query_points (and query_normals)"
            )

    def _global_fields(self, global_inputs, b: int, points):
        """Assemble the unit global directions (B, K, d) and the global scalars
        (B, S) from the named per-case fields. A single case's values (one row) are
        shared across the batch."""
        fields = _flatten_fields("global_inputs", global_inputs) or {}
        missing = [k for k in self.global_specs if k not in fields]
        extra = [k for k in fields if k not in self.global_specs]
        if missing or extra:
            raise ValueError(
                f"global_inputs fields do not match the declared set: missing {missing!r}, "
                f"unexpected {extra!r}"
            )

        def _rows(name, t, width):
            if not isinstance(t, torch.Tensor) or t.numel() % width:
                raise ValueError(
                    f"global_inputs[{name!r}]: expected {width} values per case, got "
                    f"{tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__}"
                )
            t = t.reshape(-1, width).to(points.dtype)
            if t.shape[0] == 1 and b != 1:
                t = t.expand(b, width)
            elif t.shape[0] != b:
                raise ValueError(
                    f"global_inputs[{name!r}] gives {t.shape[0]} rows for a batch of {b}"
                )
            return t

        K = self.n_global_vectors

        if K:
            gv = torch.stack(
                [_rows(k, fields[k], self.dim) for k in self.global_vector_names], 1
            )
            ### Each direction is normalized; the magnitude, if it means anything,
            ### is a rank-0 global input.
            g_unit = gv / gv.norm(dim=-1, keepdim=True).clamp_min(self.eps)  # (B, K, d)
        else:
            g_unit = points.new_zeros(b, 0, self.dim)
        if self.n_global_scalars:
            g_scalars = torch.cat(
                [
                    _rows(k, fields[k], self.global_specs[k].channels)
                    for k in self.global_scalar_names
                ],
                dim=-1,
            )
        else:
            g_scalars = None
        return g_unit, g_scalars

    def forward(
        self,
        *,
        points: Float[torch.Tensor, "batch tokens dim"],
        normals: Float[torch.Tensor, "batch tokens dim"],
        measure_weights: Float[torch.Tensor, "batch tokens"] | None = None,
        global_inputs: Mapping[str, torch.Tensor] | None = None,
        boundary_inputs: Mapping[str, torch.Tensor] | None = None,
        query_points: Float[torch.Tensor, "batch queries dim"] | None = None,
        query_normals: Float[torch.Tensor, "batch queries dim"] | None = None,
        query_inputs: Mapping[str, torch.Tensor] | None = None,
        support_points: Float[torch.Tensor, "batch support dim"] | None = None,
        support_normals: Float[torch.Tensor, "batch support dim"] | None = None,
        support_inputs: Mapping[str, torch.Tensor] | None = None,
    ) -> TensorDict:
        ### Named fields arrive flat with dotted names, nested, or as TensorDicts;
        ### the model works on flat dotted names throughout.
        boundary_inputs = _flatten_fields("boundary_inputs", boundary_inputs)
        query_inputs = _flatten_fields("query_inputs", query_inputs)
        support_inputs = _flatten_fields("support_inputs", support_inputs)
        ### One un-batched case: every per-case input gains the batch axis.
        if points.ndim == 2:
            points, normals = points[None], normals[None]
            if measure_weights is not None and measure_weights.ndim == 1:
                measure_weights = measure_weights[None]
            if query_points is not None and query_points.ndim == 2:
                query_points = query_points[None]
            if query_normals is not None and query_normals.ndim == 2:
                query_normals = query_normals[None]
            if support_points is not None and support_points.ndim == 2:
                support_points = support_points[None]
            if support_normals is not None and support_normals.ndim == 2:
                support_normals = support_normals[None]
            ### Named per-token fields are reshaped by element count downstream.
        b, n, _ = points.shape
        ### Input validation (skipped under torch.compile): fail at the API
        ### boundary with the offending shape, not inside the network.
        if not torch.compiler.is_compiling():
            self._validate_inputs(
                points,
                normals,
                measure_weights,
                boundary_inputs,
                query_points,
                query_normals,
                query_inputs,
                support_points,
                support_normals,
                support_inputs,
            )
        self._apply_geo_kernel(points)
        K = self.n_global_vectors
        g_unit, g_scalars = self._global_fields(global_inputs, b, points)
        g_hat = g_unit[:, None].expand(b, n, K, self.dim)

        ### Frame: no origin; the one length unit is a measure-weighted integral
        ### of the geometry (the RMS pairwise distance or the square root of the
        ### total measure).
        if self.scale_mode == "total_measure":
            if measure_weights is None:
                raise ValueError("scale_mode='total_measure' needs measure_weights")
            gauge = (
                measure_weights.reshape(b, n, 1)
                .to(points.dtype)
                .sum(dim=1, keepdim=True)
                .clamp_min(self.eps)
                .sqrt()
            )
        else:
            ### LEN-RMS (2026-09-18): the measure-weighted RMS pairwise distance,
            ### computed through the variance identity sum_ij w_i w_j |x_i-x_j|^2
            ### = 2 sum_i w_i |x_i - c|^2 with c the weighted mean (used only
            ### inside this scalar; the frame keeps no centre). Normalized
            ### weights make it converge under refinement where the total
            ### measure diverges, and make it invariant to a weight rescale.
            gauge = rms_pairwise_distance(points, measure_weights, self.eps)
        r = points / gauge
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        if measure_weights is None or not self.use_measure_weights:
            ### No measures given (or the weights-off ablation): every token
            ### carries the same measure, i.e. the count-weighted quadrature.
            log_w = points.new_zeros(b, n, 1)
        else:
            measure_weights = measure_weights.reshape(b, n).to(points.dtype)
            log_w = torch.log(measure_weights.clamp_min(self.eps))[..., None]

        ### Boundary inputs: invariants against the token's frame {n, g_k}.
        if self.bin:
            frame = torch.cat([n_hat[:, :, None], g_hat], dim=2)  # (B, N, 1+K, d)
            b_feat, b_vecs, b_tens = self.bin.features(boundary_inputs, b, n, frame)
        else:
            b_feat, b_vecs, b_tens = None, None, None
        h = self.embed(self._token_seed(n_hat, g_hat, b_feat, g_scalars, n))
        vecs, tens = self._block_extras(b_vecs, b_tens, None, None, b, n, points)

        internal = self.interior_geometry != "provided"

        def _query_class(q_pts, q_nrm, fields, kind: str):
            """Frame quantities, seed state and block stacks of a query-side token
            class (query tokens, support tokens or passive queries). Under
            interior_geometry='nearest_boundary' the class's directions and signed
            distances come from the boundary sample (q_nrm is ignored)."""
            bq, nq, _ = q_pts.shape
            q_r = q_pts / gauge
            builtin = None
            if internal:
                with torch.autocast(points.device.type, enabled=False):
                    d, q_nhat = nearest_boundary_geometry(
                        q_pts,
                        points[:, :n_boundary],
                        n_hat[:, :n_boundary],
                        chunk=self.interior_nn_chunk,
                        eps=self.eps,
                        projection="plane" if self.interior_geometry == "nearest_plane" else "centroid",
                    )
                    s = (d / gauge.squeeze(-1))[..., None]  # (B, Q, 1), in frame units
                    builtin = torch.cat(
                        [s, torch.sign(s) * torch.log(s.abs() + self.eps)], -1
                    )
            else:
                q_nhat = q_nrm / q_nrm.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            q_g = g_unit[:, None].expand(bq, nq, K, self.dim)
            if self.qin:
                if fields is None:
                    raise ValueError(f"query_inputs declared: {kind} needs its fields")
                q_frame = torch.cat([q_nhat[:, :, None], q_g], dim=2)
                q_feat, q_vecs, q_tens = self.qin.features(
                    fields, bq, nq, q_frame, gauge=gauge, eps=self.eps
                )
            else:
                if fields:
                    raise ValueError(
                        f"{kind} given but the model declares no query_inputs"
                    )
                q_feat, q_vecs, q_tens = None, None, None
            if builtin is not None:
                q_feat = builtin if q_feat is None else torch.cat([builtin, q_feat], -1)
            return q_r, q_nhat, q_g, q_feat, q_vecs, q_tens

        ### n_boundary counts the SURFACE tokens only: the surface measure
        ### statistics that the support and query tokens' routing weights refer
        ### to must not see the tokens appended below.
        n_boundary = n
        if self.support_tokens and support_points is not None:
            ### Support tokens (see __init__): interacting interior tokens that
            ### are a function of the case, not of the requested queries.
            if support_normals is None and not internal:
                raise ValueError(
                    "interior_geometry='provided' needs support_normals (e.g. the SDF gradient at each support point)"
                )
            bs_, ns_, _ = support_points.shape
            s_r, s_nhat, s_g, s_feat, s_vecs, s_tens = _query_class(
                support_points, support_normals, support_inputs, "support_inputs"
            )
            h_s = self.embed(self._token_seed(s_nhat, s_g, None, g_scalars, ns_))
            h_s = h_s + self.sp_type.to(h.dtype)
            if self.n_query_features:
                h_s = h_s + self.sp_embed(s_feat).to(h_s.dtype)
            if self.query_mass == "source_total":
                s_logw = (
                    self.sp_logw.to(log_w.dtype)
                    + torch.logsumexp(log_w[:, :n_boundary], dim=1, keepdim=True)
                    - math.log(ns_)
                )
            else:
                s_logw = self.sp_logw.to(log_w.dtype) + log_w[:, :n_boundary].mean(
                    dim=1, keepdim=True
                )
            s_vecs, s_tens = self._block_extras(
                None, None, s_vecs, s_tens, bs_, ns_, points
            )
            h = torch.cat([h, h_s], dim=1)
            r = torch.cat([r, s_r], dim=1)
            n_hat = torch.cat([n_hat, s_nhat], dim=1)
            g_hat = torch.cat([g_hat, s_g], dim=1)
            log_w = torch.cat([log_w, s_logw.expand(b, ns_, 1)], dim=1)
            if vecs is not None:
                vecs = torch.cat([vecs, s_vecs], dim=1)
                tens = torch.cat([tens, s_tens], dim=1)
            n = n + ns_
        qt_active = self.query_tokens and query_points is not None
        if qt_active:
            ### Interior queries as interacting tokens (see __init__).
            if query_normals is None and not internal:
                raise ValueError(
                    "interior_geometry='provided' needs query_normals (e.g. the SDF gradient at each query)"
                )
            bq, nq, _ = query_points.shape
            q_r, q_nhat, q_g, q_feat, q_vecs, q_tens = _query_class(
                query_points, query_normals, query_inputs, "query_inputs"
            )
            h_q = self.embed(self._token_seed(q_nhat, q_g, None, g_scalars, nq))
            h_q = h_q + self.qt_type.to(h.dtype)
            if self.n_query_features:
                h_q = h_q + self.qt_embed(q_feat).to(h_q.dtype)
            ### The query tokens' routing weight is learned RELATIVE to the
            ### surface measure (mean surface log-weight), so that rescaling
            ### the geometry (areas x k^2) shifts every token's log-weight by
            ### the same 2 log k and the similarity-gauge scale contract holds
            ### exactly; an absolute learned weight broke it (2026-09-07).
            ### "source_total" (see __init__) ties the queries' total weight to
            ### the total surface measure instead of the per-token mean.
            if self.query_mass == "source_total":
                q_logw = (
                    self.qt_logw.to(log_w.dtype)
                    + torch.logsumexp(log_w[:, :n_boundary], dim=1, keepdim=True)
                    - math.log(nq)
                )
            else:
                q_logw = self.qt_logw.to(log_w.dtype) + log_w[:, :n_boundary].mean(
                    dim=1, keepdim=True
                )
            q_vecs, q_tens = self._block_extras(
                None, None, q_vecs, q_tens, bq, nq, points
            )
            h = torch.cat([h, h_q], dim=1)
            r = torch.cat([r, q_r], dim=1)
            n_hat = torch.cat([n_hat, q_nhat], dim=1)
            g_hat = torch.cat([g_hat, q_g], dim=1)
            log_w = torch.cat([log_w, q_logw.expand(b, nq, 1)], dim=1)
            if vecs is not None:
                vecs = torch.cat([vecs, q_vecs], dim=1)
                tens = torch.cat([tens, q_tens], dim=1)
            n = n + nq

        nbr_idx = None
        if self.local_rho is not None and self.local_k > 0:
            nbr_idx = _knn_indices(r, self.local_k)  # positions only; shared by every block
        for block in self.blocks:
            h = block(h, log_w, r, n_hat, g_hat, self.eps, vecs, tens, nbr_idx=nbr_idx)

        ### The head frame builds its slice anchors from the SOURCE tokens
        ### (src_r, src_logw, and h); the branches below overwrite n_hat, g_out
        ### and n with the query-side values, so the source tensors are
        ### captured here.
        src_r, src_logw = r, log_w
        r_out = r  # positions of the tokens the head reads (radial basis)
        g_out = g_hat  # the global directions seen by the tokens the head reads
        if qt_active:
            ### Heads read the query tokens only (surface tokens were context).
            h_out = h[:, n_boundary:]
            n_hat, g_out, b, n = q_nhat, q_g, bq, nq
            r_out = q_r
        elif self.query_independent:
            ### Final encoder slice states and anchors (read-only for queries).
            logits = self.final_assign(h)
            a = _softmax_over_points(logits + log_w, self.fast_point_softmax)
            z_states = torch.einsum("bns,bnh->bsh", a, h)
            z_pos = torch.einsum("bns,bnc->bsc", a, r)
            m_s = torch.einsum("bns,bnc->bsc", a, n_hat)
            m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            if query_points is None:
                q_pts, q_nrm = points, normals
            else:
                q_pts = query_points
                if query_normals is not None or internal:
                    q_nrm = query_normals
                elif query_points.shape[1] != normals.shape[1]:
                    raise ValueError(
                        "query_points are not the boundary points: pass query_normals (e.g. the SDF "
                        "gradient at each query) or use interior_geometry='nearest_boundary'"
                    )
                else:
                    q_nrm = normals
            bq, nq, _ = q_pts.shape
            q_r, q_nhat, q_g, q_feat, q_vecs, q_tens = _query_class(
                q_pts, q_nrm, query_inputs, "query_inputs"
            )
            ### The remaining seed channels must match the encoder's seed
            ### layout. Boundary inputs are per-boundary-cell data: the
            ### queries carry them only when the queries ARE the boundary
            ### points, and zeros otherwise.
            own_boundary = query_points is None
            q_h = self.embed(
                self._token_seed(
                    q_nhat, q_g, b_feat if own_boundary else None, g_scalars, nq
                )
            )
            if self.n_query_features:
                q_h = q_h + self.rq_embed(q_feat).to(q_h.dtype)
            q_vecs, q_tens = self._block_extras(
                b_vecs if own_boundary else None,
                b_tens if own_boundary else None,
                q_vecs,
                q_tens,
                bq,
                nq,
                points,
            )
            ### The local readout is a softmax over sources with logits
            ### -d^2/rho^2 + log w: exactly invariant to a rescale of the source
            ### measure and well defined however far a query sits from the sources.
            src_w = src_logw.squeeze(-1)
            for rb in self.read_blocks:
                q_h = rb(
                    q_h,
                    q_r,
                    q_nhat,
                    q_g,
                    z_states,
                    z_pos,
                    m_s,
                    self.eps,
                    src_r=r,
                    src_h=h,
                    src_w=src_w,
                    local_rho=self.local_readout_rho,
                    q_vecs=q_vecs,
                    q_tens=q_tens,
                )
            h_out, n_hat, g_out, b, n = q_h, q_nhat, q_g, bq, nq
            r_out = q_r
        else:
            h_out = h
        coeffs = self.head(self.norm_out(h_out))
        if not self.needs_frame:
            ### Scalar-only outputs read the coefficients directly; no token basis.
            out: dict[str, torch.Tensor] = {}
            start = 0
            for name, spec in self.output_specs.items():
                out[name] = coeffs[..., start] if spec.channels == 1 else coeffs[..., start : start + spec.channels]
                start += spec.channels
            return _fields_to_tensordict(out, [b, n])
        ### Head frame: the radial basis vector is the direction from the token's
        ### soft slice anchor (measure-weighted mean of source positions), not
        ### from a centroid; translation covariant.
        a_f = _softmax_over_points(
            self.frame_assign(h) + src_logw, self.fast_point_softmax
        )
        z_f = torch.einsum("bns,bnc->bsc", a_f, src_r)
        b_f = torch.softmax(self.frame_assign(h_out), dim=-1)
        u = r_out - torch.einsum("bns,bsc->bnc", b_f, z_f).to(r_out.dtype)
        r_hat = u / u.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        ### The token basis. True vectors: the global directions, the normal, the
        ### radial direction and the polar complements of (rhat, n) and each
        ### (rhat, g_k) (the vector's component orthogonal to rhat, left
        ### unnormalized so it vanishes smoothly where the pair is parallel; any
        ### dimension). Pseudovectors: the Levi-Civita construction from the
        ### primaries {g_k, n, rhat}: in three dimensions their pairwise cross
        ### products (for K = 1: g x n, g x rhat, n x rhat), in two their
        ### perpendiculars; both flip sign under a reflection while their
        ### coefficients do not.
        gs = [g_out[:, :, k] for k in range(K)]
        true_basis = gs + [n_hat, r_hat, _polar_complement(r_hat, n_hat)]
        for g in gs:
            true_basis.append(_polar_complement(r_hat, g))
        primaries = gs + [n_hat, r_hat]
        pseudo_basis = _pseudovectors(primaries, self.dim)
        true_vecs = torch.stack(true_basis, dim=-2)  # (B, N, m, d)
        pseudo_vecs = (
            torch.stack(pseudo_basis, dim=-2)
            if pseudo_basis
            else true_vecs.new_zeros(b, n, 0, self.dim)
        )

        out: dict[str, torch.Tensor] = {}
        start = 0
        for name, spec in self.output_specs.items():
            layout = self.output_layouts[name]
            if spec.rank == 0 and spec.channels > 1:
                out[name] = coeffs[..., start : start + spec.channels]
                start += spec.channels
                continue
            c = coeffs[..., start : start + layout.n_coeffs]
            out[name] = _assemble_output(c, layout, true_vecs, pseudo_vecs)
            start += layout.n_coeffs
        ### A TensorDict over (batch, tokens the head read); a dotted output name
        ### becomes a nested leaf ("surface.pressure" -> out["surface", "pressure"]).
        return _fields_to_tensordict(out, [b, n])
