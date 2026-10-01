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
surrogates on the mesh-native API of the mesh models (GLOBE's): one case at a
time, the boundary goes in as a mapping of named
:class:`~physicsnemo.mesh.Mesh` objects (each a ``Mesh[d-1, d]``; its cells
are the tokens, with their centroids, normals and effective measures), the
problem's data as *named fields of declared tensor rank*: global inputs in a
TensorDict (a direction, a diffusivity, a Reynolds number), per-cell boundary
inputs in the meshes' ``cell_data`` (a prescribed traction, a Dirichlet
trace, a condition tag) and, for an interior task, per-prediction-point
inputs in the prediction Mesh's ``point_data`` (a material label, a source
term). A point-cloud ``Mesh`` comes out whose ``point_data`` holds fields of
declared rank: scalars, vectors, rank-2 tensors, up to rank 4, each exactly
covariant. The name states the design principle: the attention operates only
on invariants of the per-point vector set :math:`\{r_i, n_i, \hat g_1, \dots,
\hat g_K, v_{i1}, \dots\}` (scaled position, unit normal, unit global vectors,
the token's own tensor inputs), and the frame is re-attached only at the
heads, so exact rotation, reflection and translation covariance is paid once
at the network's edges instead of in every layer.

**Rank specs.** Inputs and outputs are declared in the field-schema grammar
shared by every mesh model (:mod:`physicsnemo.mesh.fields`): a mapping from a
field name to a field declaration, a :class:`~physicsnemo.mesh.fields.RankSpec`
or a mapping of values ``{"rank": r, "symmetric": bool, "parity": "even" |
"odd"}`` (``r`` = 0 scalar, 1 vector, 2 rank-2 tensor, ...); a mapping of
mappings is a nested group, and a flat name may contain ``"."`` for the
same nesting (``{"surface": {"pressure": {"rank": 0}}}`` and
``{"surface.pressure": {"rank": 0}}`` declare the same field). Each parsed
declaration is a :class:`~physicsnemo.mesh.fields.FieldSchema`. ``symmetric`` (rank
:math:`\ge 2`) means fully symmetric under index permutation and
``parity="odd"`` (outputs only) a pseudotensor that flips sign under a
reflection (a vorticity). One name declares one field; several independent
scalars are several rank-0 names. The same grammar serves the constructor's
``output_schema``, ``boundary_schema``, ``query_schema`` and ``global_schema``, and
the same dotted names address the fields in the meshes' ``cell_data`` and
``point_data`` and in ``global_data`` (``td["surface", "pressure"]``).

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
measure-weighted root-mean-square pairwise distance of the sample, a
normalized two-point statistic that is invariant to the weights' overall
scale and converges under refinement of any boundary with a finite second
moment, fractal ones included. No centroid, no per-dataset reference length
and no external frame appears anywhere in the forward pass, so the frame
carries no information about where the mesher placed its cells.

**Routing.** Each block routes tokens to slices by bounded cosine logits:
:data:`ROUTING_TEMPERATURE` times the cosine between the normalized token
state and a learned prototype per slice, so every logit lies in
:math:`[-T, T]` and no block can route every point to one slice however its
weights grow. The quadrature measure enters the routing as a log-space bias,
so slice states are measure-weighted (quadrature) means; with no measures the
weights are uniform.

**Surface and interior tasks.** ``interior=False`` (a surface task) predicts
at the boundary cells themselves. ``interior=True`` (an interior task)
predicts at the ``prediction_points`` given to ``forward`` (a point-cloud
``Mesh``, or a bare ``(N, d)`` tensor when ``query_schema`` declares no
fields): they join the encoder as interacting tokens next to the boundary
cells and the outputs are returned for them only. Per-prediction-point inputs
(declared in ``query_schema``, read from the prediction Mesh's ``point_data``)
are carried by those tokens; per-cell boundary inputs (declared in
``boundary_schema``) by the boundary cells, with zeros on the
prediction-point tokens. The prediction-point
tokens' total routing weight is a learned fraction of the total boundary
measure, split equally over them, so it is invariant to re-representing the
same measure (splitting a boundary cell in two changes nothing) and scales
like an area. The two tasks own different parameter sets (the interior task
adds the prediction-point embedding, type offset and routing mass), which is
why the choice is a constructor argument: a parameter that takes no part in
a forward pass stops DDP.

**Interior geometry.** A prediction-point token needs a direction (its
"normal" in every invariant) and a distance to the wall. Both are computed
inside the model from the boundary tokens, never taken as inputs: for each
prediction point the nearest boundary token :math:`y_{i^*}` gives
:math:`d_q = \pm |x_q - y_{i^*}|`, signed by that token's normal (positive
on the side the normal points to: outside, for outward normals, the
convention of the stored signed-distance fields of the program's datasets),
and the direction :math:`u_q = (x_q - y_{i^*}) / |x_q - y_{i^*}|`.
:math:`d_q / L` with its signed logarithm enters the token's embedding as a
built-in rank-0 channel ahead of the fields declared in ``query_schema``. A
dataset's stored signed-distance field or its gradient enter, if wanted, as
ordinary ``query_schema`` fields (rank 0 / rank 1) like any other PDE data. The
nearest token comes from the library's nearest-neighbour search
(:func:`physicsnemo.nn.functional.neighbors.knn.knn`, ``k=1``: cuML on
CUDA, a scipy KD-tree on CPU, torch otherwise, in any dimension), and
:func:`nearest_boundary_geometry` builds the distance and direction from the
gathered sample. Because the boundary is a
sample (200,000 of up to 17 million cells on a DrivAerML car), the internal
distance overestimates the true wall distance near the wall by up to about
the sample spacing, and the internal direction deviates from the true SDF
gradient by the same order; far from the wall both agree with the full-mesh
field. The geometry is built from pairwise differences, so every covariance
contract holds.

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
to four per-point vectors counting the global directions and the token's
rank-1 inputs together, at most 1024 slices, no rank-2 inputs). The default
``geo_kernel=None`` means *auto*: at the first forward
pass the model picks the fused kernel when the inputs are on CUDA, Triton
is importable and the constraints hold, and the eager region otherwise,
logging the choice once; at 200,000 cells the fused kernel runs the
training step in about a third of the eager time and memory. The per-token
rank-2 contractions (rho^T T rho) are the standing kernel follow-up; until
then a model with rank-2 boundary or query inputs runs the eager region.

Every constructor argument and every ``forward`` input is keyword-only, so a
call reads as the recipe's ``forward_kwargs`` mapping does and models can be
swapped without positional bookkeeping. ``forward(boundary_meshes=...,
prediction_points=..., global_data=...)`` returns a point-cloud ``Mesh``; see
:meth:`ISLA.forward`.
"""

import itertools
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.calculus.measure import cell_measures
from physicsnemo.mesh.fields import FieldSchema, FieldSchemaLike, RankSpec
from physicsnemo.nn.functional.equivariant_ops import vector_project
from physicsnemo.nn.functional.neighbors.knn.knn import knn

logger = logging.getLogger(__name__)

#: Bound of the cosine routing logits: each block's point-to-slice logit is this
#: times the cosine between the token state and a slice prototype.
GeoKernel = Literal["eager", "fused"]
"""Geometry-region backend: the PyTorch reference region or the exact fused Triton kernel."""
SlicePrototypes = Literal["static", "centered"]
"""Slice-routing prototypes: one learned prototype per slice, or the seeds plus a centered, scale-bounded sample-conditioned term."""

ROUTING_TEMPERATURE = 10.0


### ---------------------------------------------------------------------------
### Rank specs and named fields
### ---------------------------------------------------------------------------


def _declare(
    kind: str, specs: FieldSchemaLike | None, *, max_rank: int, allow_odd: bool
) -> FieldSchema:
    """Parse and check one of the constructor's field declarations (``None`` is
    the empty schema).

    The grammar is :meth:`~physicsnemo.mesh.fields.FieldSchema.parse`: a field
    is a :class:`~physicsnemo.mesh.fields.RankSpec` or a mapping of values such
    as ``{"rank": 0}``; a mapping of mappings is a nested group, and a flat name
    may contain ``"."`` for the same nesting. ISLA's own limits are applied to
    the parsed fields, raising :class:`NotImplementedError` (as GLOBE does):
    ``max_rank`` per declaration, and pseudotensor inputs refused.
    """
    out = FieldSchema({}) if specs is None else FieldSchema.parse(specs, label=kind)
    for name, spec in out.items():
        if spec.rank > max_rank:
            if kind == "output_schema":
                why = "the equivariant basis grows as m**rank"
            elif kind == "global_schema":
                why = (
                    "broadcast the tensor to the boundary cells and declare it as a "
                    "boundary input"
                )
            else:
                why = "see the module docstring for how higher ranks would enter"
            raise NotImplementedError(
                f"{kind}[{name!r}]: rank {spec.rank} is not supported "
                f"(maximum {max_rank}; {why})"
            )
        if spec.parity == "odd" and not allow_odd:
            raise NotImplementedError(
                f"{kind}[{name!r}]: pseudotensor (parity='odd') inputs are not supported; "
                "input invariants are formed with true vectors only so that they are "
                "invariant under reflections"
            )
    return out


def _leaf(data: TensorDict, name: str) -> torch.Tensor | None:
    """The leaf tensor at a dotted name of a TensorDict, ``None`` when absent
    (including when a prefix of the name is a tensor rather than a group)."""
    try:
        value = data.get(FieldSchema.key(name), None)
    except ValueError:
        return None
    return value if isinstance(value, torch.Tensor) else None


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


@dataclass(frozen=True, eq=False)
class _InputSet:
    """The per-token tensor inputs of one token class (boundary cells, or interior
    queries) and the invariants they contribute.

    With ``F = 1 + K`` frame vectors ``{n_hat, g_1..g_K}`` per token, ``P`` rank-1
    inputs and ``Q`` rank-2 inputs (``Q_s`` of them symmetric):

    * seed features (into the token's embedding): the rank-0 fields
      (``x 2`` when ``scaled_scalars``: ``[s, sign(s) log(|s| + eps)]``), then
      per rank-1 input its ``F`` frame cosines and its norm and the
      ``P(P-1)/2`` pairwise dot products of the rank-1 inputs, then per rank-2
      input the ``F(F+1)/2`` (symmetric) or ``F**2`` frame contractions
      ``a^T T b``, its trace and its Frobenius norm;
    * block vectors (contracted with the token-to-anchor direction in every
      slice layer): the ``P`` rank-1 inputs and, per rank-2 input, the ``F``
      derived true vectors ``T a`` (plus ``T^T a`` when not symmetric);
    * block tensors (``rho^T T rho`` in every slice layer): the ``Q`` rank-2 inputs.

    ``scaled_scalars`` marks the query set, whose rank-0 fields are lengths in
    the frame's unit; boundary scalars enter raw.
    """

    specs: Mapping[str, RankSpec]
    n_frame: int
    scaled_scalars: bool
    dim: int = 3
    n_scalar: int = field(init=False)
    rank1: list[str] = field(init=False)
    rank2: list[tuple[str, bool]] = field(init=False)
    n_features: int = field(init=False)
    n_block_vecs: int = field(init=False)
    n_block_tens: int = field(init=False)

    def __post_init__(self) -> None:
        f = int(self.n_frame)
        rank1 = [name for name, s in self.specs.items() if s.rank == 1]
        rank2 = [(name, s.symmetric) for name, s in self.specs.items() if s.rank == 2]
        n_scalar = sum(1 for s in self.specs.values() if s.rank == 0)
        p, q = len(rank1), len(rank2)
        n_pairs = sum(f * (f + 1) // 2 if sym else f * f for _, sym in rank2)
        set_ = object.__setattr__
        set_(self, "n_frame", f)
        set_(self, "dim", int(self.dim))
        set_(self, "scaled_scalars", bool(self.scaled_scalars))
        set_(self, "n_scalar", n_scalar)
        set_(self, "rank1", rank1)
        set_(self, "rank2", rank2)
        set_(
            self,
            "n_features",
            (2 if self.scaled_scalars else 1) * n_scalar
            + p * f
            + p
            + p * (p - 1) // 2
            + n_pairs
            + 2 * q,
        )
        set_(self, "n_block_vecs", p + sum(f if sym else 2 * f for _, sym in rank2))
        set_(self, "n_block_tens", q)

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
            size = spec.numel(self.dim)
            if not isinstance(t, torch.Tensor) or t.numel() != b * n * size:
                raise ValueError(
                    f"{kind}[{name!r}]: expected a tensor of shape "
                    f"{(b, n, *spec.shape(self.dim))} ({b * n * size} elements), got "
                    f"{tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__}"
                )

    def features(self, fields, b: int, n: int, frame, *, gauge=None, eps=0.0):
        """Invariants of one token class.

        ``frame`` is ``(B, N, F, d)``: the token's true frame vectors
        ``[n_hat, g_1..g_K]`` in ``d`` spatial dimensions. Returns ``(seed (B, N,
        n_features), vecs (B, N, n_block_vecs, d), tens (B, N, n_block_tens, d, d))``; ``gauge``
        (``(B, 1, 1)``) divides the rank-0 fields when ``scaled_scalars``.
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
                s = fields[name].reshape(b, n, 1).to(dtype)
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


def _softmax_over_points(x: Float[torch.Tensor, "batch tokens slices"]):
    """Softmax over the point dimension (dim 1) of a (B, N, S) tensor.

    Reduces along a contiguous last dimension after a transpose, which uses
    PyTorch's row-parallel softmax kernel: a softmax over a *middle* dimension
    is served by the "spatial" kernel, which parallelizes over the B*S columns
    only and walks the N points serially (85-95% of the training step at 10k
    to 80k tokens on a GB300, and the source of the superlinear token
    scaling). Same arithmetic as ``torch.softmax(x, dim=1)`` up to summation
    order; against a float64 reference the row-parallel kernel carries a
    ~1e-7 float32 floor where the middle-dimension kernel carries ~1e-4.
    """
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
    relative to the point only, so no frame origin is referred to."""
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

    :data:`ROUTING_TEMPERATURE` times the cosine between the token state and a
    learned prototype per slice (the rows of ``weight``), so every logit lies in
    ``[-T, T]`` and no block can route every point to one slice however its
    weights grow.

    ``slice_prototypes="static"``: the prototypes are the rows of ``weight``,
    the same for every sample. ``"centered"``: the prototypes are synthesized
    per sample. Each static row acts as a seed that attends over the token
    states with a measure-weighted softmax (``log_w`` enters as a log-space
    bias, so the summary is a quadrature of the tokens and invariant to a
    uniform rescale of the weights); the attended value term is centered
    across the slices and scale-bounded before it is added to the seeds:
    ``P = seeds + Delta_c * b / (rms(Delta_c) + b)`` with ``b = g rms(seeds)``,
    ``Delta_c = Delta - mean_S(Delta)``, ``rms`` over all entries (per sample for
    ``Delta_c``) and ``g`` one learned scalar (init 0.1). The seed-independent
    component of the value term is common to every prototype, carries no routing
    information under cosine logits and, when it dominates, collapses the
    prototypes onto one direction; centering removes it exactly. The bound is a
    smooth saturation: a small centered term (near-uniform attention, roundoff)
    passes through unchanged and a large one is held to an rms of at most
    ``g rms(seeds)``, so the sample-dependent part never exceeds a fraction
    ``g`` of the seeds' spread and is never amplified to it. Token states are
    invariant features, so the prototypes are invariant and the contracts hold.
    Adds ``2 hidden^2 + 1`` parameters per block (the key and value maps and
    the gain).
    """

    def __init__(
        self, hidden: int, n_slices: int, slice_prototypes: SlicePrototypes = "static"
    ) -> None:
        super().__init__()
        if slice_prototypes not in ("static", "centered"):
            raise ValueError(
                f"slice_prototypes must be 'static' or 'centered', got {slice_prototypes!r}"
            )
        self.centered = slice_prototypes == "centered"
        self.weight = nn.Linear(hidden, n_slices, bias=False).weight
        if self.centered:
            self.k_proj = nn.Linear(hidden, hidden, bias=False)
            self.v_proj = nn.Linear(hidden, hidden, bias=False)
            self.scale = 1.0 / math.sqrt(hidden)
            self.gain = nn.Parameter(torch.tensor(0.1))

    def prototypes(self, x: torch.Tensor, log_w: torch.Tensor | None) -> torch.Tensor:
        """Per-sample prototypes (B, S, hidden) from the seeds and the token states."""
        seeds = self.weight.to(x.dtype)  # (S, H)
        att = torch.einsum("sh,bnh->bsn", seeds, self.k_proj(x)) * self.scale  # (B, S, N)
        if log_w is not None:
            att = att + log_w[..., 0][:, None, :].to(att.dtype)
        att = torch.softmax(att if att.dtype in (torch.float32, torch.float64) else att.float(), dim=-1).to(x.dtype)
        delta = torch.einsum("bsn,bnh->bsh", att, self.v_proj(x))  # (B, S, H)
        delta = delta - delta.mean(dim=1, keepdim=True)
        bound = self.gain.to(x.dtype) * seeds.square().mean().sqrt()  # g rms(seeds), scalar
        rms_delta = delta.square().mean(dim=(1, 2), keepdim=True).sqrt()  # (B, 1, 1)
        return seeds[None] + delta * (bound / (rms_delta + bound))

    def forward(self, x: torch.Tensor, log_w: torch.Tensor | None = None) -> torch.Tensor:
        if self.centered:
            protos = self.prototypes(x, log_w)
            return ROUTING_TEMPERATURE * torch.einsum(
                "bnh,bsh->bns", F.normalize(x, dim=-1), F.normalize(protos, dim=-1)
            )
        return ROUTING_TEMPERATURE * F.linear(
            F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1)
        )


class _QueryTokens(nn.Module):
    """The query tokens' own parameters (the interior mode): a learned token-type
    offset, the log of the learned fraction of the total boundary measure that is
    their routing weight, and the embedding of their per-query features (the
    built-in signed distance to the sampled boundary as ``[d/L, sign(d) log(|d|/L +
    eps)]`` ahead of the fields declared in ``query_schema``)."""

    def __init__(self, hidden: int, n_features: int) -> None:
        super().__init__()
        self.logw = nn.Parameter(torch.zeros(1))
        self.type_offset = nn.Parameter(torch.zeros(hidden))
        self.embed = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )

    def forward(self, h_seed: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """The query tokens' initial state from their seed embedding and features."""
        return h_seed + self.type_offset.to(h_seed.dtype) + self.embed(feat).to(h_seed.dtype)


class _SliceBlock(nn.Module):
    """One pre-LN layer of measure-weighted soft-slice attention + MLP."""

    def __init__(
        self,
        hidden: int,
        n_slices: int,
        mlp_ratio: int = 4,
        geo_checkpoint: bool = False,
        geo_kernel: GeoKernel = "eager",
        n_global_vectors: int = 1,
        n_block_vecs: int = 0,
        n_block_tens: int = 0,
        slice_prototypes: SlicePrototypes = "static",
    ) -> None:
        super().__init__()
        ### "fused" evaluates the dense geometry region (invariants -> routing bias
        ### -> point->slice mix -> pooled invariants) in one Triton kernel per
        ### direction that never materializes a (B,N,S,.) tensor (see
        ### geo_kernel.py); same arithmetic as _geo_region, recompute built in, so
        ### geo_checkpoint is moot on that path.
        self.geo_kernel = geo_kernel
        self.n_geo = _geo_width(n_global_vectors, n_block_vecs, n_block_tens)
        ### Activation recompute: the per-slice geometry tensors -- rel (B,N,S,3),
        ### rel_hat, dist and two bf16 copies of the (B,N,S,geo) invariants -- are
        ### 76% of ISLA's saved activations at 10k tokens. With geo_checkpoint the
        ### invariants are rebuilt from (r, n_hat, g_hat, z_pos, m_s) inside
        ### backward instead of stored; the forward is bitwise unchanged (same
        ### ops, same order).
        self.geo_checkpoint = bool(geo_checkpoint)
        self.norm_assign = nn.LayerNorm(hidden)
        self.assign = _RoutingLogits(hidden, n_slices, slice_prototypes)
        ### Relational geometry: per-slice equivariant anchors and point-anchor
        ### invariants -- many local, data-adaptive reference points instead of
        ### any global frame (smooth by construction).
        self.geo_width = hidden // 2
        self.geo_logit = nn.Linear(self.n_geo, 1)
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
    ) -> Float[torch.Tensor, "batch tokens hidden"]:
        ### Soft assignment of points to slices; measure weights enter as a
        ### log-space bias so slice states are quadrature-weighted means.
        logits = self.assign(self.norm_assign(h), log_w)  # (B, N, S)
        a = _softmax_over_points(logits + log_w)  # normalized over points
        ### Equivariant anchors: weighted mean position AND mean normal
        ### direction per slice -- anchors gain orientation.
        z_pos = torch.einsum("bns,bnc->bsc", a, r)  # (B, S, d)
        m_s = torch.einsum("bns,bnc->bsc", a, n_hat)
        m_s = m_s / m_s.norm(dim=-1, keepdim=True).clamp_min(eps)
        ### Geometry refines the routing and the readback. Pool the invariants
        ### over slices FIRST, then project: exactly equal to projecting then
        ### pooling (the projection is affine and point_mix sums to one over
        ### slices), but the saved activation is (B, N, geo) instead of
        ### (B, N, S, hidden/2) -- ~0.5 GB per layer at 10k tokens, 256 slices,
        ### hidden 192.
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
        a = _softmax_over_points(logits + log_w)
        z = torch.einsum("bns,bnh->bsh", a, h)  # slice states
        z = z + self.slice_mlp(z)
        back = torch.einsum("bns,bsh->bnh", point_mix, z)
        geo_pool = self.geo_feat(pooled)
        h = h + self.broadcast(torch.cat([h, back, geo_pool], dim=-1))
        return h + self.mlp(self.norm_mlp(h))


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


def nearest_boundary_geometry(
    query_points: Float[torch.Tensor, "batch queries dim"],
    points: Float[torch.Tensor, "batch tokens dim"],
    normals: Float[torch.Tensor, "batch tokens dim"],
    *,
    eps: float = 1e-12,
) -> tuple[
    Float[torch.Tensor, "batch queries"], Float[torch.Tensor, "batch queries dim"]
]:
    r"""Signed distance and direction from the nearest boundary sample to each query.

    For every query :math:`x_q` the nearest boundary point :math:`y_{i^*}` is
    found by the library's nearest-neighbour search
    (:func:`physicsnemo.nn.functional.neighbors.knn.knn` with ``k=1``, one
    call per batch element: cuML on CUDA, a scipy KD-tree on CPU, torch
    otherwise; any spatial dimension); then

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
    positions. ``normals`` are renormalized.

    Because the boundary is a *sample* (200,000 of up to 17 million cells on
    a DrivAerML car), :math:`\lvert d_q \rvert` overestimates the true wall
    distance by up to about the sample spacing near the wall and :math:`u_q`
    deviates from the true SDF gradient by the same order; far from the wall
    both agree with the full-mesh field.
    """
    b, nq, _ = query_points.shape
    with torch.autocast(query_points.device.type, enabled=False), torch.no_grad():
        ### knn is unbatched; the batch axis is one in practice.
        idx = torch.stack(
            [
                knn(points[i].contiguous(), query_points[i].contiguous(), 1)[0][:, 0]
                for i in range(b)
            ]
        ).long()  # (B, Q)
    gather = idx[..., None].expand(b, nq, points.shape[-1])
    y = torch.gather(points, 1, gather)
    n_star = torch.gather(normals, 1, gather)
    n_star = n_star / n_star.norm(dim=-1, keepdim=True).clamp_min(eps)
    diff = query_points - y
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


@dataclass(frozen=True, eq=False)
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

    spec: RankSpec
    n_true: int
    n_pseudo: int
    dim: int = 3
    n_vec: int = field(init=False)
    n_free: int = field(init=False)
    index: torch.Tensor | None = field(init=False)
    iso: list[torch.Tensor] = field(init=False)
    n_coeffs: int = field(init=False)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        spec = self.spec
        r = spec.rank
        set_(self, "dim", int(self.dim))
        n_vec = self.n_true if spec.parity == "even" else self.n_true + self.n_pseudo
        set_(self, "n_vec", n_vec)
        if r == 0:
            set_(self, "n_free", 1)
            set_(self, "index", None)
            set_(self, "iso", [])
            set_(self, "n_coeffs", 1)
            return
        free: dict[tuple[int, ...], int] = {}
        index = []
        for j in itertools.product(range(n_vec), repeat=r):
            n_odd = sum(1 for a in j if a >= self.n_true)
            valid = n_odd == (0 if spec.parity == "even" else 1)
            if not valid:
                index.append(-1)
                continue
            key = tuple(sorted(j)) if spec.symmetric else j
            index.append(free.setdefault(key, len(free)))
        n_free = len(free)
        set_(self, "n_free", n_free)
        ### -1 (no coefficient) points at the appended zero slot n_free.
        set_(
            self,
            "index",
            torch.tensor([i if i >= 0 else n_free for i in index], dtype=torch.long),
        )
        iso = _isotropic_tensors(r, spec.symmetric, spec.parity, self.dim)
        set_(self, "iso", iso)
        set_(self, "n_coeffs", n_free + len(iso))


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
    the module docstring for the contracts). Every argument is keyword-only.

    **Rank specs.** ``output_schema``, ``boundary_schema``, ``query_schema`` and
    ``global_schema`` are :data:`~physicsnemo.mesh.fields.FieldSchemaLike`
    declarations, the grammar shared by every mesh model
    (:meth:`~physicsnemo.mesh.fields.FieldSchema.parse`): a field is a
    :class:`~physicsnemo.mesh.fields.RankSpec` or a mapping of values
    ``{"rank": r, "symmetric": False, "parity": "even"}``; a mapping of
    mappings is a nested group, and a flat name may contain ``"."`` for
    the same nesting (``{"surface": {"pressure": {"rank": 0}}}`` and
    ``{"surface.pressure": {"rank": 0}}`` declare the same field). Each
    parsed declaration is the :class:`~physicsnemo.mesh.fields.FieldSchema`
    attribute of the same name. ``symmetric`` means
    fully symmetric under index permutation (rank :math:`\ge 2` only);
    ``parity="odd"`` declares a pseudotensor output (outputs only). One name
    declares one field; several scalars are several rank-0 names.

    **Mesh API.** ``forward`` takes the boundary as a mapping of named
    :class:`~physicsnemo.mesh.Mesh` objects (one per boundary condition or
    part, each a ``Mesh[dim - 1, dim]`` whose cells have normals), the
    per-case data as a TensorDict, and, for an interior task, the prediction
    points as a point-cloud ``Mesh`` (or a bare ``(N, dim)`` tensor when
    ``query_schema`` declares no fields). It returns a point-cloud ``Mesh`` whose
    ``point_data`` holds the declared outputs. Details under ``forward``.

    Args:
        output_schema: Output fields, a rank spec per name. Each field is a leaf of
            the returned Mesh's ``point_data`` (nested per the dotted name): a
            rank-0 field has shape ``(N,)``, a rank-``r`` field ``(N, d, ...,
            d)`` with ``r`` trailing axes (``d`` the spatial dimension,
            ``dim``); a symmetric spec gives a symmetric tensor. Ranks up to
            :attr:`MAX_OUTPUT_RANK`. Each field's coefficients come from the
            shared linear head; with ``m = 2K + 3`` true basis vectors and
            ``p`` pseudovectors (``(K+2)(K+1)/2`` in three dimensions, ``K +
            2`` in two) the counts per field are ``1`` at rank 0; ``m`` (even)
            or ``p`` (odd) at rank 1; ``m^2 + 1`` (even, the identity
            included) or ``m(m+1)/2 + 1`` (even symmetric), ``2pm`` or ``pm``
            (odd) at rank 2. For ``K = 1`` (``m = 5, p = 3``): a vector costs
            5, a symmetric rank-2 tensor 16, a pseudovector 3.
        boundary_schema: Per-cell boundary fields, a rank spec per name with
            ranks 0 to 2 and even parity (a prescribed traction as
            ``{"rank": 1}``, a Dirichlet trace or a condition tag as
            ``{"rank": 0}``). One declaration
            applies to every boundary mesh: each mesh's ``cell_data`` must
            carry every declared field (per-boundary declarations are a
            follow-up). Their invariants widen the boundary tokens' seed;
            prediction-point tokens carry zeros there. Rank-1 and rank-2
            inputs also contract with the token-to-anchor direction in every
            slice layer (module docstring).
        query_schema: Per-prediction-point fields of the interior task
            (``interior=True`` required), a rank spec per name with ranks 0
            to 2 and even parity, read from the prediction Mesh's
            ``point_data`` (a material label, a source term, a dataset's
            stored signed distance or its gradient). Carried by the
            prediction-point tokens through their own embedding, with the
            same invariants as boundary inputs. Rank-0 entries are lengths in
            the frame's unit: each is divided by the length unit and enters
            as ``[s, sign(s) log(|s| + eps)]``; rank-1 and rank-2 entries are
            never scaled.
        global_schema: Per-case fields, a rank spec per name with rank 0
            (appended to every token's seed) or rank 1 (a global direction,
            normalized to unit length inside the model). The ``K`` rank-1
            entries, in declaration order, are the global directions of the
            geometry region and the head basis. Read from ``global_data`` by
            name; extra leaves there are ignored.
        interior: The task structure. ``False`` (default): a surface task,
            the outputs are predicted at the boundary cells themselves;
            ``forward`` takes no ``prediction_points`` and the model has no
            prediction-point parameters. ``True``: an interior task, the
            outputs are predicted at ``prediction_points``, which join the
            encoder as interacting tokens; ``forward`` requires them. The two
            are different parameter sets (a parameter without a gradient stops
            DDP), so the choice is made at construction.
        hidden: Token width of the slice blocks. The defaults ``hidden=384``,
            ``n_layers=12`` and ``n_slices=256`` are the reference
            configuration; the reference training rate is 5e-4.
        n_layers: Number of slice-attention blocks.
        n_slices: Number of soft slices (data-adaptive anchors) per block.
        mlp_ratio: Expansion ratio of the per-token and per-slice MLPs.
        geo_checkpoint: Rebuild the per-slice geometric invariants in the
            backward pass instead of storing them (bitwise-identical forward and
            gradients, lower peak memory).
        geo_kernel: ``None`` (default) selects automatically at the first
            forward pass: ``"fused"`` (exact Triton kernel for the per-layer
            geometry region) when the inputs are on CUDA, Triton is available,
            there are one to four per-point vectors (global directions plus rank-1
            boundary or query inputs), at most 1024 slices and no rank-2 inputs,
            else ``"eager"`` (the PyTorch reference region). Pass ``"eager"`` or
            ``"fused"`` to force one; ``"fused"`` raises when its constraints
            cannot be met.
        compile_blocks: Run every slice block through :func:`torch.compile`
            (default ``False``). The compiled function is the block's forward,
            built at the first forward pass, so parameter names, checkpoints and
            the eager path are unchanged; the fused geometry kernel stays a
            custom-op leaf. Compiles once per distinct token count (the
            reporting protocol's fixed sample compiles once); with a token count
            that changes every case, leave it off. On four GB300 ranks at
            200,000 cells it cut the training step by 19 to 35% and peak memory
            by 27% at bfloat16-roundoff differences (PERF-3, 2026-10-01).
        slice_prototypes: ``"static"`` (default): each slice block routes tokens
            against a learned prototype per slice that is the same for every
            sample. ``"centered"``: centered, scale-bounded sample-conditioned
            prototypes: each seed attends over the token states with a
            measure-weighted softmax, the attended value term is centered across
            the slices and passed through a smooth saturation that holds its rms
            to at most a learned fraction ``g`` (one scalar per block, init 0.1)
            of the seeds' rms spread, and the result is added to the seeds
            (:class:`_RoutingLogits`). Token states are invariant, so the
            prototypes and the contracts are unchanged; ``2 hidden^2 + 1``
            parameters per block are added.
        dim: Spatial dimension of the domain (default 3). Every boundary mesh
            is a ``Mesh[dim - 1, dim]`` and every prediction point has ``dim``
            coordinates. Every geometric quantity is built for ``dim``
            dimensions and the head's basis follows it (pseudovectors by the
            Levi-Civita construction: cross products in three dimensions,
            perpendiculars in two; odd-parity outputs are refused for ``dim >=
            4``). The fused Triton geometry kernel is written for three
            dimensions; other dimensions run the eager region. The
            nearest-boundary search works in any dimension. A
            two-dimensional problem is passed with ``dim=2`` (polyline
            boundaries) rather than embedded with a zero third coordinate.
        eps: Numerical floor for norms and logarithms.

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

    ### 0.2.0 renamed the field declarations after the schemas they hold.
    __model_checkpoint_version__ = "0.2.0"
    __supported_model_checkpoint_version__ = {
        "0.1.0": (
            "Loading a 0.1.0 ISLA checkpoint: renaming its field declarations "
            "outputs, boundary_inputs, query_inputs and global_inputs to "
            "output_schema, boundary_schema, query_schema and global_schema."
        ),
    }
    _RENAMED_ARGS = {
        "outputs": "output_schema",
        "boundary_inputs": "boundary_schema",
        "query_inputs": "query_schema",
        "global_inputs": "global_schema",
    }

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Rename the field declarations of a 0.1.0 checkpoint's arguments."""
        args = super()._backward_compat_arg_mapper(version, args)
        if version == "0.1.0":
            for old, new in cls._RENAMED_ARGS.items():
                if old in args:
                    args[new] = args.pop(old)
        return args

    def __init__(
        self,
        *,
        output_schema: FieldSchemaLike,
        boundary_schema: FieldSchemaLike | None = None,
        query_schema: FieldSchemaLike | None = None,
        global_schema: FieldSchemaLike | None = None,
        interior: bool = False,
        hidden: int = 384,
        n_layers: int = 12,
        n_slices: int = 256,
        mlp_ratio: int = 4,
        geo_checkpoint: bool = False,
        geo_kernel: GeoKernel | None = None,
        compile_blocks: bool = False,
        slice_prototypes: SlicePrototypes = "static",
        dim: int = 3,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(meta=self.MetaData())
        ### compile_blocks: run each slice block through torch.compile (built lazily at
        ### the first forward pass, on the block's bound forward so parameter names and
        ### checkpoints are unchanged). The blocks see only the (B, n_tokens, .) tensors,
        ### so a fixed token count per case compiles once; the mesh-facing forward,
        ### whose vertex count varies per case, stays eager. The fused geometry kernel
        ### is a registered custom op and stays a leaf. Measured 2026-10-01 (PERF-3,
        ### four GB300 ranks, 200,000 cells): step 221 -> 180 ms and peak memory
        ### 44.9 -> 32.9 GiB on DrivAerML Surface, 475 -> 308 ms and 88.5 -> 64.5 GiB
        ### on DrivAerML Volume, losses at bfloat16 roundoff of the eager path.
        self.compile_blocks = bool(compile_blocks)
        self._block_forwards: list | None = None
        ### The spatial dimension of the points. Every geometric quantity is built
        ### for d dimensions; the head's basis and the coefficient count depend on
        ### d (pseudovectors exist by the Levi-Civita construction: perpendiculars
        ### in two dimensions, cross products in three), which is why d is a
        ### constructor argument and forward checks the points.
        if int(dim) < 2:
            raise ValueError(f"dim must be at least 2, got {dim!r}")
        self.dim = int(dim)
        if slice_prototypes not in ("static", "centered"):
            raise ValueError(
                f"slice_prototypes must be 'static' or 'centered', got {slice_prototypes!r}"
            )
        self.slice_prototypes = str(slice_prototypes)
        ### Every input and output is a named field of declared rank; the grammar
        ### is the mesh models' shared one (physicsnemo.mesh.fields), flattened to
        ### dotted names here.
        self.output_schema = _declare(
            "output_schema",
            output_schema,
            max_rank=self.MAX_OUTPUT_RANK,
            allow_odd=True,
        )
        if not self.output_schema:
            raise ValueError("output_schema must declare at least one field")
        self.boundary_schema = _declare(
            "boundary_schema",
            boundary_schema,
            max_rank=self.MAX_INPUT_RANK,
            allow_odd=False,
        )
        self.query_schema = _declare(
            "query_schema", query_schema, max_rank=self.MAX_INPUT_RANK, allow_odd=False
        )
        self.global_schema = _declare(
            "global_schema", global_schema, max_rank=1, allow_odd=False
        )
        ### The task structure: a surface task predicts at the boundary cells and
        ### has no prediction-point parameters; an interior task predicts at
        ### prediction points that join the encoder as tokens.
        self.interior = bool(interior)
        if self.query_schema and not self.interior:
            raise ValueError(
                f"query_schema {sorted(self.query_schema)!r} declares per-prediction-"
                "point fields of the interior task; declare them with interior=True"
            )
        ### Checkpoints store the constructor arguments as JSON. A FieldSchema
        ### is a dict and serializes as it stands; recording the parsed schemas
        ### keeps the metadata JSON-safe whatever container the caller passed
        ### (a Hydra DictConfig, for instance).
        for key in ("output_schema", "boundary_schema", "query_schema", "global_schema"):
            self._args["__args__"][key] = getattr(self, key) or None
        ### The problem's global data are zero or more global DIRECTIONS (each
        ### enters as a unit direction: one seed term n.g_k per token, one
        ### direction cosine rel.g_k per point-anchor pair, and the head basis
        ### gains g_k with the spherical complements of (u, g_k)) and zero or more
        ### global SCALARS (PDE parameters; appended to every token's seed).
        self.global_vector_names = [
            n for n, s in self.global_schema.items() if s.rank == 1
        ]
        self.global_scalar_names = [
            n for n, s in self.global_schema.items() if s.rank == 0
        ]
        self.n_global_vectors = K = len(self.global_vector_names)
        self.n_global_scalars = len(self.global_scalar_names)
        ### geo_kernel selects the implementation of the per-layer geometry region,
        ### not its arithmetic. "eager" is the PyTorch region (with geo_checkpoint
        ### deciding whether its (B,N,S,.) intermediates are stored or rebuilt in
        ### backward); "fused" is the Triton kernel of geo_kernel.py, which reads
        ### the pre-logits once and writes the bias and mix once per direction,
        ### with the recompute built in (CUDA only). geo_kernel=None resolves at
        ### the first forward pass to "fused" wherever the fused kernel can run and
        ### to "eager" otherwise; the blocks read the resolved name. An explicit
        ### "eager" or "fused" is honored.
        if geo_kernel is None:
            geo_kernel = "auto"
        if geo_kernel not in ("auto", "eager", "fused"):
            raise ValueError(
                f"geo_kernel must be None (auto), 'eager' or 'fused', got {geo_kernel!r}"
            )
        self.geo_kernel = geo_kernel
        self.n_slices = int(n_slices)
        self.eps = eps
        ### The boundary cells' fields and the interior queries' fields, each set
        ### with its own invariant layout (_InputSet). Every token carries the
        ### UNION of the two block-level vector/tensor stacks (zeros where a class
        ### has no such field), so one geometry region serves boundary and query
        ### tokens alike. Rank-0 query fields are lengths in the frame's unit.
        n_frame = 1 + K  # {n_hat, g_1..g_K}
        self.bin = _InputSet(self.boundary_schema, n_frame, scaled_scalars=False, dim=self.dim)
        self.qin = _InputSet(self.query_schema, n_frame, scaled_scalars=True, dim=self.dim)
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
                geo_checkpoint=geo_checkpoint,
                geo_kernel=geo_kernel,
                n_global_vectors=K,
                n_block_vecs=self.n_block_vecs,
                n_block_tens=self.n_block_tens,
                slice_prototypes=self.slice_prototypes,
            )
            for _ in range(n_layers)
        )
        ### Head frame: per-point soft slice anchor for the radial basis vector.
        ### Built only when an output of rank >= 1 reads the token basis: a
        ### scalar-only model never touches the radial direction, and a parameter
        ### that receives no gradient stops DDP at the second step.
        self.needs_frame = any(spec.rank >= 1 for spec in self.output_schema.values())
        self.frame_assign = (
            _RoutingLogits(hidden, n_slices) if self.needs_frame else None
        )
        self.norm_out = nn.LayerNorm(hidden)
        ### One linear head emits every output's invariant coefficients; each
        ### output is contracted with tensor products of the token's basis vectors
        ### (_OutputLayout / _assemble_output). True vectors: g_1..g_K, n, rhat,
        ### e_th(rhat, n), e_th(rhat, g_k), m = 2K+3 in every dimension.
        ### Pseudovectors by the Levi-Civita construction from the primaries
        ### {g_k, n, rhat}: their perpendiculars in two dimensions (p = K + 2),
        ### their pairwise cross products in three (p = C(K+2, 2)); a pseudovector
        ### in d dimensions needs d - 1 vectors, so odd-parity outputs are refused
        ### for d >= 4 until a dataset needs them. For K = 1 in 3-D and outputs
        ### {scalar: 0, vector: 1} the head has width 1 + 5.
        self.n_true_basis = 3 + 2 * K
        if self.dim == 3:
            self.n_pseudo_basis = (K + 2) * (K + 1) // 2
        elif self.dim == 2:
            self.n_pseudo_basis = K + 2
        else:
            self.n_pseudo_basis = 0
            odd = [n for n, sp in self.output_schema.items() if sp.parity == "odd"]
            if odd:
                raise NotImplementedError(
                    f"odd-parity outputs {odd!r} in dim={self.dim}: a pseudovector needs "
                    f"{self.dim - 1} vectors (the Levi-Civita construction); only two and "
                    "three dimensions are implemented"
                )
        self.output_layouts = {
            name: _OutputLayout(spec, self.n_true_basis, self.n_pseudo_basis, self.dim)
            for name, spec in self.output_schema.items()
        }
        self.n_head_coeffs = sum(
            layout.n_coeffs for layout in self.output_layouts.values()
        )
        self.head = nn.Linear(hidden, self.n_head_coeffs)
        ### Prediction-point tokens (interior=True): the prediction points join the
        ### encoder as INTERACTING tokens, carrying the same {n_q, g_k} invariant
        ### seeds as the boundary tokens plus the parameters of _QueryTokens.
        ### Exactly SE(d)-covariant, not query-set independent. Outputs are
        ### returned for these tokens only. The tokens' total routing weight is a
        ### learned fraction exp(qt.logw) of the TOTAL boundary measure, split
        ### equally over them: invariant to any re-representation of the same
        ### discrete measure (splitting a boundary cell in two changes nothing) and
        ### scaling like an area, so the similarity-gauge contract holds.
        ### Registered last, so the boundary-side parameter order is that of a
        ### surface model; a surface model (interior=False) has none of them, so
        ### every parameter it owns takes part in every forward pass.
        self.n_query_builtin = 2
        self.n_query_features = self.n_query_builtin + self.qin.n_features
        self.qt = _QueryTokens(hidden, self.n_query_features) if self.interior else None

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

    def _block_callables(self):
        """The slice blocks as callables: the modules themselves, or their forwards
        under :func:`torch.compile` when ``compile_blocks`` is set (built once)."""
        if not self.compile_blocks or torch.compiler.is_compiling():
            return self.blocks
        if self._block_forwards is None:
            self._block_forwards = [torch.compile(block.forward) for block in self.blocks]
        return self._block_forwards

    def _apply_geo_kernel(self, points: torch.Tensor) -> None:
        """Write the resolved kernel name into the blocks, once per resolution."""
        kernel = self.resolve_geo_kernel(points)
        if getattr(self, "_geo_kernel_applied", None) == kernel:
            return
        for block in self.blocks:
            block.geo_kernel = kernel
        self._geo_kernel_applied = kernel

    def _validate_inputs(
        self,
        points,
        normals,
        measure_weights,
        boundary_inputs,
        query_points,
        query_inputs,
    ) -> None:
        """Shape and presence checks for :meth:`_forward_tensors`."""
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
        if query_points is None:
            if self.interior:
                raise ValueError("an interior model (interior=True) needs query_points")
            if query_inputs:
                raise ValueError("query_inputs given without query_points")
            return
        if not self.interior:
            raise ValueError(
                "query_points given to a surface model (interior=False); build with interior=True"
            )
        if query_points.ndim != 3 or query_points.shape[0] != b or query_points.shape[-1] != d:
            raise ValueError(
                f"Expected query_points of shape ({b}, Q, {d}) but got tensor of shape {tuple(query_points.shape)}"
            )
        self.qin.validate("query_inputs", query_inputs, b, query_points.shape[1])

    def _global_fields(self, global_inputs, b: int, points):
        """Assemble the unit global directions (B, K, d) and the global scalars
        (B, S) from the named per-case fields (flat dotted names). A single case's
        values (one row) are shared across the batch."""
        fields = global_inputs or {}
        missing = [k for k in self.global_schema if k not in fields]
        extra = [k for k in fields if k not in self.global_schema]
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
                [_rows(k, fields[k], 1) for k in self.global_scalar_names], dim=-1
            )
        else:
            g_scalars = None
        return g_unit, g_scalars

    ### -----------------------------------------------------------------------
    ### The mesh API
    ### -----------------------------------------------------------------------

    def _check_fields(self, specs, data: TensorDict, n: int, label: str) -> None:
        """Every declared field is a leaf of ``data`` (by dotted name) with the
        shape its rank prescribes over ``n`` entities; extra leaves are ignored."""
        missing = sorted(name for name in specs if _leaf(data, name) is None)
        if missing:
            raise ValueError(f"{label} is missing the declared fields {missing!r}")
        for name, spec in specs.items():
            t = _leaf(data, name)
            expected = (n, *spec.shape(self.dim))
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"{label}[{name!r}]: expected shape {expected} (rank {spec.rank}), "
                    f"got {tuple(t.shape)}"
                )

    def _validate_meshes(self, boundary_meshes, prediction_points) -> None:
        """Mesh-level checks for :meth:`forward`: dimensions, declared fields,
        and the task structure (``interior``)."""
        d = self.dim
        if not isinstance(boundary_meshes, Mapping) or not boundary_meshes:
            raise ValueError("boundary_meshes must be a non-empty mapping of name to Mesh")
        for name, mesh in boundary_meshes.items():
            if not isinstance(mesh, Mesh):
                raise TypeError(
                    f"boundary_meshes[{name!r}] must be a physicsnemo.mesh.Mesh, got "
                    f"{type(mesh).__name__}"
                )
            if mesh.n_spatial_dims != d:
                raise ValueError(
                    f"boundary_meshes[{name!r}] has {mesh.n_spatial_dims} spatial dimensions; "
                    f"this model has dim={d}"
                )
            if mesh.n_manifold_dims != d - 1:
                raise ValueError(
                    f"boundary_meshes[{name!r}]: the boundary of a {d}-dimensional domain is a "
                    f"Mesh[{d - 1}, {d}] (its cells carry the normals); got a "
                    f"Mesh[{mesh.n_manifold_dims}, {d}]"
                )
            self._check_fields(
                self.boundary_schema,
                mesh.cell_data,
                mesh.n_cells,
                f"boundary_meshes[{name!r}].cell_data",
            )
        if prediction_points is None:
            if self.interior:
                raise ValueError(
                    "this model was built with interior=True and predicts at prediction "
                    "points; pass prediction_points (a point-cloud Mesh, or an (N, dim) "
                    "tensor when query_schema declares no fields)"
                )
            return
        if not self.interior:
            raise ValueError(
                "this model was built with interior=False (a surface task: predictions at "
                "the boundary cells), so prediction_points must be None; build with "
                "interior=True for an interior task"
            )
        if isinstance(prediction_points, Mesh):
            if prediction_points.n_spatial_dims != d:
                raise ValueError(
                    f"prediction_points has {prediction_points.n_spatial_dims} spatial "
                    f"dimensions; this model has dim={d}"
                )
            self._check_fields(
                self.query_schema,
                prediction_points.point_data,
                prediction_points.n_points,
                "prediction_points.point_data",
            )
            return
        if (
            not isinstance(prediction_points, torch.Tensor)
            or prediction_points.ndim != 2
            or prediction_points.shape[-1] != d
        ):
            raise ValueError(
                f"prediction_points must be a point-cloud Mesh or a tensor of shape (N, {d}); "
                f"got {tuple(prediction_points.shape) if isinstance(prediction_points, torch.Tensor) else type(prediction_points).__name__}"
            )
        if self.query_schema:
            raise ValueError(
                f"this model's query_schema declares {sorted(self.query_schema)!r}; pass "
                "prediction_points as a Mesh whose point_data carries them"
            )

    def _select_global_data(self, global_data, device) -> TensorDict:
        """The declared leaves of ``global_data`` as a TensorDict (extra leaves
        dropped, a missing one an error); this is also what the output Mesh carries."""
        if global_data is None:
            td = TensorDict({}, batch_size=[], device=device)
        elif isinstance(global_data, TensorDict):
            td = global_data
        elif isinstance(global_data, Mapping):
            ### A plain mapping may name its leaves with dotted strings; TensorDict
            ### takes those as nested keys only in tuple form.
            td = TensorDict({}, batch_size=[], device=device)
            for name, value in global_data.items():
                td.set(FieldSchema.key(name) if isinstance(name, str) else name, value)
        else:
            raise TypeError(
                f"global_data must be a TensorDict or a mapping of name to tensor, got "
                f"{type(global_data).__name__}"
            )
        missing = sorted(name for name in self.global_schema if _leaf(td, name) is None)
        if missing:
            raise ValueError(f"global_data is missing the declared fields {missing!r}")
        return td.select(*(FieldSchema.key(name) for name in self.global_schema))

    def forward(
        self,
        *,
        boundary_meshes: Mapping[str, Mesh],
        prediction_points: Mesh | Float[torch.Tensor, "n_points dim"] | None = None,
        global_data: TensorDict | Mapping[str, torch.Tensor] | None = None,
    ) -> Mesh:
        r"""Predict the declared output fields for one case.

        Args:
            boundary_meshes: The boundary, as named ``Mesh[dim - 1, dim]``
                objects (one per boundary condition or part; a polyline in two
                dimensions, a surface in three). Their cells are the boundary
                tokens, concatenated in sorted key order: a token's position is
                the cell centroid, its normal the cell normal, its quadrature
                measure the cell's effective measure
                (``cell_data["_effective_measure"]``, the Horvitz-Thompson
                corrected measure of a subsampled boundary) when present, else
                the cell's geometric measure. Every ``boundary_schema`` field
                is read from each mesh's ``cell_data`` by its (possibly
                dotted) name; other cell data are ignored.
            prediction_points: The interior task's prediction points
                (``interior=True``): a point-cloud ``Mesh`` whose ``points`` are
                the prediction points and whose ``point_data`` carries the
                ``query_schema`` fields, or a bare ``(N, dim)`` tensor when
                none are declared. The points join the encoder as interacting
                tokens; each one's built-in geometry, its signed distance to
                the nearest boundary token and the direction to it, is computed
                inside the model from the boundary tokens
                (:func:`nearest_boundary_geometry`) and is never an input. A
                dataset's stored signed-distance field or its gradient enter, if
                wanted, as ordinary ``query_schema`` fields (rank 0 / rank 1)
                like any other PDE data. ``None`` for a surface task
                (``interior=False``): the outputs are predicted at the boundary
                tokens.
            global_data: The case's per-case fields, a TensorDict (or mapping)
                with the ``global_schema`` fields as leaves (a rank-0 field a
                scalar, a rank-1 field a ``(dim,)`` vector); extra leaves are
                ignored, a missing one is an error. ``None`` when nothing is
                declared.

        Returns:
            A point-cloud ``Mesh`` whose ``points`` are the prediction points
            (interior) or the boundary token positions in token order
            (surface), whose ``point_data`` holds one leaf per declared output
            at its natural shape (``(N,)`` at rank 0, ``(N, dim)`` at rank 1,
            ``(N, dim, dim)`` at rank 2, nested per the dotted names), and
            whose ``global_data`` is the selected ``global_data`` passed
            through.
        """
        names = sorted(boundary_meshes) if isinstance(boundary_meshes, Mapping) else ()
        ### Validation at the API boundary (skipped under torch.compile): fail
        ### here with the offending mesh, not inside the network.
        if not torch.compiler.is_compiling():
            self._validate_meshes(boundary_meshes, prediction_points)
        meshes = [boundary_meshes[name] for name in names]
        points = torch.cat([mesh.cell_centroids for mesh in meshes])
        normals = torch.cat([mesh.cell_normals for mesh in meshes])
        measures = torch.cat([cell_measures(mesh) for mesh in meshes])
        boundary_inputs = {
            name: torch.cat([_leaf(mesh.cell_data, name) for mesh in meshes])[None]
            for name in self.boundary_schema
        } or None
        global_td = self._select_global_data(global_data, points.device)
        global_inputs = {name: _leaf(global_td, name) for name in self.global_schema}
        if prediction_points is None:
            out_points, query_points, query_inputs = points, None, None
        elif isinstance(prediction_points, Mesh):
            out_points = prediction_points.points
            query_points = out_points[None]
            query_inputs = {
                name: _leaf(prediction_points.point_data, name)[None]
                for name in self.query_schema
            } or None
        else:
            out_points, query_inputs = prediction_points, None
            query_points = out_points[None]
        ### One case: a batch axis of one for the batched machinery below.
        fields = self._forward_tensors(
            points=points[None],
            normals=normals[None],
            measure_weights=measures[None],
            global_inputs=global_inputs,
            boundary_inputs=boundary_inputs,
            query_points=query_points,
            query_inputs=query_inputs,
        )
        point_data = TensorDict({}, batch_size=[out_points.shape[0]], device=out_points.device)
        for name, value in fields.items():
            point_data.set(FieldSchema.key(name), value[0])
        return Mesh(points=out_points, point_data=point_data, global_data=global_td)

    def _forward_tensors(
        self,
        *,
        points: Float[torch.Tensor, "batch tokens dim"],
        normals: Float[torch.Tensor, "batch tokens dim"],
        measure_weights: Float[torch.Tensor, "batch tokens"] | None = None,
        global_inputs: Mapping[str, torch.Tensor] | None = None,
        boundary_inputs: Mapping[str, torch.Tensor] | None = None,
        query_points: Float[torch.Tensor, "batch queries dim"] | None = None,
        query_inputs: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """The batched network on token tensors; :meth:`forward` is its only
        caller and the release tests pin it.

        ``points``, ``normals`` ``(B, N, d)``; ``measure_weights`` ``(B, N)``
        (``None``: every token the same weight); ``global_inputs``,
        ``boundary_inputs`` and ``query_inputs`` flat ``{dotted name: tensor}``
        mappings over the declared fields (per-case values ``(B,)`` / ``(B, d)``,
        one row shared across the batch; per-token values ``(B, N, ...)``);
        ``query_points`` ``(B, Q, d)`` for an interior model. Returns
        ``{dotted name: (B, N_out, ...)}`` over the tokens the head read.
        """
        b, n, _ = points.shape
        if not torch.compiler.is_compiling():
            self._validate_inputs(
                points,
                normals,
                measure_weights,
                boundary_inputs,
                query_points,
                query_inputs,
            )
        self._apply_geo_kernel(points)
        K = self.n_global_vectors
        g_unit, g_scalars = self._global_fields(global_inputs, b, points)
        g_hat = g_unit[:, None].expand(b, n, K, self.dim)

        ### Frame: no origin; the one length unit is the measure-weighted RMS
        ### pairwise distance, computed through the variance identity
        ### sum_ij w_i w_j |x_i-x_j|^2 = 2 sum_i w_i |x_i - c|^2 with c the
        ### weighted mean (used only inside this scalar; the frame keeps no
        ### centre). Normalized weights make it converge under refinement where
        ### the total measure diverges, and make it invariant to a weight rescale.
        gauge = rms_pairwise_distance(points, measure_weights, self.eps)
        r = points / gauge
        n_hat = normals / normals.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        if measure_weights is None:
            ### No measures given: every token carries the same measure, i.e. the
            ### count-weighted quadrature.
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

        ### n_boundary counts the boundary tokens only: the boundary measure
        ### statistic that the query tokens' routing weight refers to must not
        ### see the tokens appended below.
        n_boundary = n
        interior = query_points is not None
        if interior:
            bq, nq, _ = query_points.shape
            q_r = query_points / gauge
            ### The query's geometry from the boundary sample: the signed distance
            ### to the nearest boundary token (the built-in channel) and the
            ### direction to it (the token's "normal" in every invariant).
            with torch.autocast(points.device.type, enabled=False):
                d, q_nhat = nearest_boundary_geometry(
                    query_points, points, n_hat, eps=self.eps
                )
                s =(d / gauge.squeeze(-1))[..., None]  # (B, Q, 1), in frame units
                builtin = torch.cat([s, torch.sign(s) * torch.log(s.abs() + self.eps)], -1)
            q_g = g_unit[:, None].expand(bq, nq, K, self.dim)
            if self.qin:
                q_frame = torch.cat([q_nhat[:, :, None], q_g], dim=2)
                q_feat, q_vecs, q_tens = self.qin.features(
                    query_inputs, bq, nq, q_frame, gauge=gauge, eps=self.eps
                )
                q_feat = torch.cat([builtin, q_feat], -1)
            else:
                q_feat, q_vecs, q_tens = builtin, None, None
            h_q = self.qt(self.embed(self._token_seed(q_nhat, q_g, None, g_scalars, nq)), q_feat)
            ### The query tokens' total routing weight is a learned fraction of the
            ### total boundary measure, split equally (see __init__): rescaling the
            ### geometry (areas x k^2) shifts every token's log-weight by the same
            ### 2 log k, so the similarity-gauge contract holds exactly.
            q_logw = (
                self.qt.logw.to(log_w.dtype)
                + torch.logsumexp(log_w[:, :n_boundary], dim=1, keepdim=True)
                - math.log(nq)
            )
            q_vecs, q_tens = self._block_extras(None, None, q_vecs, q_tens, bq, nq, points)
            h = torch.cat([h, h_q], dim=1)
            r = torch.cat([r, q_r], dim=1)
            n_hat = torch.cat([n_hat, q_nhat], dim=1)
            g_hat = torch.cat([g_hat, q_g], dim=1)
            log_w = torch.cat([log_w, q_logw.expand(b, nq, 1)], dim=1)
            if vecs is not None:
                vecs = torch.cat([vecs, q_vecs], dim=1)
                tens = torch.cat([tens, q_tens], dim=1)
            n = n + nq

        for block in self._block_callables():
            h = block(h, log_w, r, n_hat, g_hat, self.eps, vecs, tens)

        ### The head frame builds its slice anchors from ALL tokens (src_r, src_logw
        ### and h); in the interior mode the head reads the query tokens only, so
        ### the token-side tensors are overwritten with the query-side values.
        src_r, src_logw = r, log_w
        r_out = r  # positions of the tokens the head reads (radial basis)
        g_out = g_hat  # the global directions seen by the tokens the head reads
        if interior:
            h_out = h[:, n_boundary:]
            n_hat, g_out, b, n = q_nhat, q_g, bq, nq
            r_out = q_r
        else:
            h_out = h
        coeffs = self.head(self.norm_out(h_out))
        if not self.needs_frame:
            ### Scalar-only outputs read the coefficients directly; no token basis.
            return {name: coeffs[..., i] for i, name in enumerate(self.output_schema)}
        ### Head frame: the radial basis vector is the direction from the token's
        ### soft slice anchor (measure-weighted mean of token positions), not
        ### from a centroid; translation covariant.
        a_f = _softmax_over_points(self.frame_assign(h) + src_logw)
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
        for name, layout in self.output_layouts.items():
            c = coeffs[..., start : start + layout.n_coeffs]
            out[name] = _assemble_output(c, layout, true_vecs, pseudo_vecs)
            start += layout.n_coeffs
        return out
