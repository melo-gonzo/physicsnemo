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

"""ISLA's own scales: the field declarations' ISLA-local keys and the per-case scales.

A field of a boundary condition's schema, of ``global_schema`` or of
``output_schema`` may carry, next to its rank spec, the keys of :data:`SCALE_KEYS`:
``units`` (exponents of the base scales :data:`BASE_SCALES`: the length ``L``, the
velocity ``U`` and the density ``rho``), ``velocity_scale: true`` (a rank-1 field
that is a source of ``U`` and of the direction ``g``) and, for a rank-0 global,
``role`` (``density`` or ``viscosity``, consumed for the units and the Reynolds
number) and, for an output, a boundary field or a rank-0 global input, ``scale`` (a
positive constant ``c``, the typical size of the field made dimensionless by its units:
an input enters as ``x / (c prod scale^exp)``, and the head predicts ``y / (c prod
scale^exp)``, both of order one). :func:`split_scale_keys` pops them before
:meth:`~physicsnemo.mesh.fields.FieldSchema.parse`, so the shared
:class:`~physicsnemo.mesh.fields.RankSpec` grammar is untouched. Private to
:mod:`physicsnemo.experimental.models.isla`."""

import math
from collections.abc import Mapping
from numbers import Real

import torch

from physicsnemo.mesh.fields import RankSpec

#: The base scales a field's ``units`` may name.
BASE_SCALES = ("L", "U", "rho")
#: The roles of rank-0 globals, each with the units it implies.
ROLES = {"density": {"rho": 1}, "viscosity": {"U": 1, "L": 1}}
#: The ISLA-local keys of a field declaration.
SCALE_KEYS = ("units", "velocity_scale", "role", "scale")


def _check_local(kind: str, name: str, local: Mapping) -> dict:
    """One field's ISLA-local keys, checked and in a JSON-safe form."""
    where = f"{kind}[{name!r}]"
    out = {}
    if "units" in local:
        units = local["units"]
        if not isinstance(units, Mapping):
            raise TypeError(
                f"{where}: units must be a mapping of base scale to exponent; got {units!r}"
            )
        unknown = sorted(str(k) for k in units if k not in BASE_SCALES)
        if unknown:
            raise ValueError(
                f"{where}: units names {unknown!r}; the base scales are {list(BASE_SCALES)!r}"
            )
        for k, e in units.items():
            if isinstance(e, bool) or not isinstance(e, Real) or not math.isfinite(e):
                raise TypeError(
                    f"{where}: the exponent of {k!r} must be a finite number; got {e!r}"
                )
        out["units"] = {
            k: (int(e) if float(e).is_integer() else float(e))
            for k, e in units.items()
            if e
        }
    if "velocity_scale" in local:
        if not isinstance(local["velocity_scale"], bool):
            raise TypeError(
                f"{where}: velocity_scale must be a bool; got {local['velocity_scale']!r}"
            )
        if kind == "output_schema" and local["velocity_scale"]:
            raise ValueError(f"{where}: an output cannot be a velocity_scale source")
        out["velocity_scale"] = local["velocity_scale"]
    if "scale" in local:
        c = local["scale"]
        if (
            isinstance(c, bool)
            or not isinstance(c, Real)
            or not (math.isfinite(c) and c > 0)
        ):
            raise ValueError(
                f"{where}: scale must be a positive finite number; got {c!r}"
            )
        out["scale"] = float(c)
    if "role" in local:
        if kind != "global_schema":
            raise ValueError(
                f"{where}: a role is declared on a rank-0 global; got one in {kind}"
            )
        if local["role"] not in ROLES:
            raise ValueError(
                f"{where}: role must be one of {list(ROLES)!r}; got {local['role']!r}"
            )
        implied = ROLES[local["role"]]
        if out.get("units", implied) != implied:
            raise ValueError(
                f"{where}: the role {local['role']!r} has the units {implied!r}; got {out['units']!r}"
            )
        out["role"] = local["role"]
    return out


def split_scale_keys(kind: str, specs):
    """``specs`` without the ISLA-local keys, and those keys by dotted field name.

    A field is a :class:`~physicsnemo.mesh.fields.RankSpec` or a mapping with a
    non-mapping ``rank``; any other mapping is a group, walked with its name as the
    prefix (the grammar's nested and dotted forms give the same dotted names).
    Everything else is left for :meth:`~physicsnemo.mesh.fields.FieldSchema.parse`
    to accept or refuse."""
    extras: dict[str, dict] = {}
    if not isinstance(specs, Mapping):
        return specs, extras

    def walk(group: Mapping, prefix: str) -> dict:
        out = {}
        for name, value in group.items():
            dotted = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Mapping) and not isinstance(value, RankSpec):
                if "rank" in value and not isinstance(value["rank"], Mapping):
                    local = {k: value[k] for k in SCALE_KEYS if k in value}
                    if local:
                        extras[str(dotted)] = _check_local(kind, str(dotted), local)
                    value = {k: v for k, v in value.items() if k not in SCALE_KEYS}
                else:
                    value = walk(value, dotted)
            out[name] = value
        return out

    return walk(specs, ""), extras


def merged(schema, extras: Mapping):
    """The declaration a checkpoint stores: the parsed ``schema`` (``None`` when
    empty) with each field's ISLA-local keys merged back."""
    if not extras:
        return schema or None
    return {name: {**spec, **extras.get(name, {})} for name, spec in schema.items()}


def check_output_stats(stats, output_schema) -> dict | None:
    """``output_stats`` (``{name: {"mean", "std"}}``, the format of
    ``NormalizeMeshFields.stats``; a ``type`` key is ignored) as JSON-safe floats.
    Each name is a declared output, each std positive, and the mean of a field of
    rank 1 or more zero (a nonzero vector mean would break rotation covariance)."""
    if stats is None:
        return None
    if not isinstance(stats, Mapping):
        raise TypeError(
            f"output_stats must be a mapping of output name to {{mean, std}}; got {stats!r}"
        )
    out = {}
    for name, s in stats.items():
        if name not in output_schema:
            raise ValueError(
                f"output_stats[{name!r}]: not a declared output ({list(output_schema)!r})"
            )
        mean = torch.as_tensor(s["mean"], dtype=torch.float64)
        std = torch.as_tensor(s["std"], dtype=torch.float64)
        if std.numel() != 1 or not float(std) > 0:
            raise ValueError(
                f"output_stats[{name!r}]: std must be one positive number; got {s['std']!r}"
            )
        if output_schema[name].rank == 0:
            if mean.numel() != 1:
                raise ValueError(
                    f"output_stats[{name!r}]: a scalar's mean is one number; got {s['mean']!r}"
                )
        elif bool((mean != 0).any()):
            raise ValueError(
                f"output_stats[{name!r}]: the mean of a field of rank {output_schema[name].rank} must be 0 "
                f"(a nonzero mean breaks rotation covariance); got {s['mean']!r}"
            )
        out[name] = {
            "mean": float(mean.reshape(-1)[0]) if mean.numel() else 0.0,
            "std": float(std),
        }
    return out


def scale_factor(units: Mapping | None, scales: Mapping):
    """``prod_k scales[k] ** units[k]`` (``None`` for a dimensionless field)."""
    if not units:
        return None
    factor = None
    for k, e in units.items():
        term = scales[k] ** e
        factor = term if factor is None else factor * term
    return factor


def velocity_moments(sources):
    """``(sum a, sum a |v|^2, sum a v)`` in float64 over ``sources``, pairs of a
    per-cell vector field ``v (n, d)`` and the cells' measures ``a (n,)``."""
    s0 = s2 = s1 = 0.0
    for v, a in sources:
        v, a = v.double(), a.double()
        s0 = s0 + a.sum()
        s2 = s2 + (a * v.square().sum(-1)).sum()
        s1 = s1 + (a[:, None] * v).sum(0)
    return s0, s2, s1
