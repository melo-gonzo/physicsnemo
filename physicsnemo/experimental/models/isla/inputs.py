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

"""ISLA's input side: the field declarations and the invariants they contribute.

:func:`_declare` parses and checks one of the constructor's rank-spec declarations;
:class:`_InputSet` holds one token class's per-token fields and turns them into the
seed invariants, the block-level derived vectors and the block-level rank-2 tensors.
Private to :mod:`physicsnemo.experimental.models.isla`."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from tensordict import TensorDict

from physicsnemo.mesh.fields import FieldSchema, FieldSchemaLike, RankSpec


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
                why = (
                    "see the ISLA module docstring for how higher ranks would enter"
                )
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
