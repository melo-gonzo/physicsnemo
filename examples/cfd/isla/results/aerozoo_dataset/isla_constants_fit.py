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

"""ISLA-SCALES round 2 (2026-10-19): fit ISLA's dataset constants on the train parts with the implemented scale function
(ISLA._case_scales on the full-resolution boundaries, the model built from the recipe's model config).

Per case: L, U, log10 Re = log10(U L / nu), and per declared output the RMS of the target made dimensionless by its
units (a seeded 20,000-point interior sample, as the output_stats fit; velocity the isotropic sqrt(mean |v|^2 / 3)).
Per split: reynolds = {mean, std (population)} of log10 Re over the train cases; scale c = the median over the train
cases of the per-case RMS.
usage: isla_constants_fit.py aerozoo <data> <model yaml> <out.jsonl> <shard> <n_shards> <workers>
       isla_constants_fit.py airfrans <dataset yaml> <model yaml> <out.jsonl> <workers> [<split>...]
       isla_constants_fit.py summarize <aerozoo manifest> <airfrans manifest> <out.json> <jsonl...>
       isla_constants_fit.py airfrans_fit <airfrans manifest> <out.json> <split,...> <jsonl...>
       isla_constants_fit.py aerozoo_rms <data> <out.jsonl> <shard> <n_shards> <workers>
       isla_constants_fit.py aerozoo_rms_fit <aerozoo manifest> <model yaml> <out.json> <jsonl...>

AF-EXTRAP (2026-10-21): the airfrans scan takes the splits whose cases it reads (default full_train) and also records
each case's freestream speed, angle of attack and Reynolds number and the mean and population std of nut over every
interior point (the converter's per-case statistics). airfrans_fit fits each named split on its own cases only:
isla_constants as above, the nut NormalizeMeshFields stats as the mean over the cases of the per-case mean and of the
per-case std (how the dataset files' 7.9e-4 / 2.4e-3 were made, there over every case), and the mean and population
std of U_inf_mag (the baselines' SplitGlobalVector).

AF-WHOLE (2026-10-23): the airfrans scan reads the case files and the far field where the dataset yaml's reader does
(train_datadir, and vtk_boundaries through the recipe's own read_vtk_boundary), so L is that of every boundary the
model config declares.

AZ7 (2026-10-26): the interior task's seven fields. AZ_FIELDS adds intermittency and retheta (dimensionless), so the
scan fits all seven. A dimensionless output's scale factor is 1, so its per-case RMS needs no case scale:
aerozoo_rms reads only the seeded 20,000-point interior sample of the scan above (the same torch permutation of the
same rows) and writes each field's RMS in SI units (rms_si); aerozoo_rms_fit fits, per split, the median over the
train cases for the outputs the model config declares without units."""

import glob
import json
import math
import os
import statistics
import sys
from multiprocessing import Pool

import torch

AZ_FIELDS = {
    "velocity": ("solution", "velocity"),
    "pressure": ("solution", "gauge_pressure"),
    "nut": ("solution", "turbulent_kinematic_viscosity"),
    "tke": ("solution", "turbulent_kinetic_energy"),
    "omega": ("solution", "specific_dissipation_rate"),
    "intermittency": ("solution", "intermittency"),
    "retheta": ("solution", "momentum_thickness_reynolds_number"),
}
AF_FIELDS = {"nut": "nut"}
#: The boundary inputs by condition, under their raw names (the pipeline's RenameMeshFields).
AZ_INPUTS = {
    "inlet": {
        "bc_tke": ("prescribed", "dirichlet", "turbulent_kinetic_energy"),
        "bc_omega": ("prescribed", "dirichlet", "specific_dissipation_rate"),
        "bc_intermittency": ("prescribed", "dirichlet", "intermittency"),
        "bc_retheta": ("prescribed", "dirichlet", "momentum_thickness_reynolds_number"),
    },
    "outlet": {
        "bc_velocity_reverse": ("prescribed", "reverse_flow", "velocity"),
        "bc_tke_reverse": ("prescribed", "reverse_flow", "turbulent_kinetic_energy"),
        "bc_omega_reverse": ("prescribed", "reverse_flow", "specific_dissipation_rate"),
        "bc_intermittency_reverse": ("prescribed", "reverse_flow", "intermittency"),
        "bc_retheta_reverse": (
            "prescribed",
            "reverse_flow",
            "momentum_thickness_reynolds_number",
        ),
    },
}
AZ_SPLITS = [
    "full",
    "family_non_box",
    "reynolds",
    "bank_blobs",
    "bank_zero_to_cad",
    "profile_crossflow",
    "profile_dual_inlet",
]
AF_SPLIT = "full_train"


def _model(yaml, dim):
    from omegaconf import OmegaConf

    from physicsnemo.experimental.models.isla import ISLA

    c = OmegaConf.to_container(OmegaConf.load(yaml).model)

    def strip(fields):
        return {
            k: {kk: vv for kk, vv in v.items() if kk != "scale"}
            for k, v in (fields or {}).items()
        }

    return ISLA(
        output_schema=strip(c["output_schema"]),
        boundary_schemas={k: strip(v) for k, v in c["boundary_schemas"].items()},
        global_schema=c["global_schema"],
        reynolds={"mean": 0.0, "std": 1.0},
        interior=True,
        dim=dim,
        hidden=16,
        n_layers=1,
        n_slices=4,
    ).double()


def _scales(m, names, meshes, gd):
    """The model's own length unit (ISLA._gauge) and case scales (ISLA._case_scales)."""
    from physicsnemo.mesh.calculus.measure import cell_measures

    measures = [cell_measures(x) for x in meshes]
    like = meshes[0].points
    rows = m._global_rows(m._select_global_data(gd, like.device), 1, like)
    return m._case_scales(names, meshes, measures, rows, m._gauge(meshes, measures))


def _rms(m, s, name, y):
    from physicsnemo.experimental.models.isla.scales import scale_factor

    f = scale_factor(m._output_units.get(name), s)
    y = y.double() / (float(f) if f is not None else 1.0)
    return math.sqrt(
        float(y.square().sum(-1).mean() / y.shape[-1])
        if y.ndim == 2
        else float(y.square().mean())
    )


def aerozoo_case(arg):
    """Fit inputs for one AeroZoo case.

    Returns ISLA's scales (L, U, log10 Re), the dimensionless RMS of each target
    field and of each boundary input, or an error row.
    """
    path, yaml = arg
    torch.set_num_threads(1)
    from physicsnemo.mesh import DomainMesh

    try:
        m = _model(yaml, 3)
        dm = DomainMesh.load(path)
        names = [n for n in m.boundary_names if n in dm.boundary_names]
        meshes = [dm.boundaries[n].to(torch.float64) for n in names]
        for name, x in zip(
            names, meshes
        ):  # the pipeline's rename of the source (RenameMeshFields)
            if name == "inlet":
                x.cell_data["bc_velocity"] = x.cell_data[
                    "prescribed", "dirichlet", "velocity"
                ]
        s = _scales(m, names, meshes, dm.global_data.to(torch.float64))
        idx = torch.randperm(
            dm.interior.n_points, generator=torch.Generator().manual_seed(0)
        )[:20000]
        pd = dm.interior.point_data
        row = {
            "case": os.path.basename(os.path.dirname(path)),
            "L": float(s["L"]),
            "U": float(s["U"]),
            "log10_re": float(s["z"]),
            "n_interior": dm.interior.n_points,
        }
        row.update({f: _rms(m, s, f, pd[k][idx]) for f, k in AZ_FIELDS.items()})
        from physicsnemo.experimental.models.isla.scales import scale_factor
        from physicsnemo.mesh.calculus.measure import cell_measures

        for cond, fields in AZ_INPUTS.items():
            mesh = dict(zip(names, meshes)).get(cond)
            for f, key in fields.items():
                if mesh is None or mesh.n_cells == 0:
                    continue
                a = cell_measures(mesh)
                x = mesh.cell_data[key].double()
                factor = scale_factor(m._boundary_units[cond].get(f), s)
                x = x / (float(factor) if factor is not None else 1.0)
                sq = (
                    x.square().sum(-1) if x.ndim == 2 else x.square()
                )  # a vector's magnitude
                row[f"{cond}.{f}"] = math.sqrt(float((a * sq).sum() / a.sum()))
        return row
    except Exception:  # noqa: BLE001
        import traceback

        return {"case": path, "error": traceback.format_exc()[-1200:]}


def aerozoo_rms_case(path):
    """Per AZ_FIELDS field, the RMS in SI units of aerozoo_case's interior sample (velocity the isotropic one)."""
    import numpy as np

    torch.set_num_threads(1)
    try:
        sol = os.path.join(
            path, "_tensordict", "interior", "_tensordict", "point_data", "solution"
        )
        meta = json.load(open(os.path.join(sol, "meta.json")))
        n = int(meta["shape"][0])
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))[
            :20000
        ].numpy()
        order = idx.argsort()  # read in file order, keep the sample's order
        row = {
            "case": os.path.basename(os.path.dirname(path)),
            "n_interior": n,
            "rms_si": {},
        }
        for f, (_, key) in AZ_FIELDS.items():
            shape = tuple(meta[key]["shape"])
            if meta[key]["dtype"] != "torch.float32":
                raise ValueError(f"{key}: expected torch.float32, got {meta[key]}")
            mm = np.memmap(
                os.path.join(sol, f"{key}.memmap"),
                dtype=np.float32,
                mode="r",
                shape=shape,
            )
            x = np.empty((len(idx), *shape[1:]), np.float32)
            x[order] = mm[idx[order]]
            y = torch.from_numpy(x).double() / 1.0
            row["rms_si"][f] = math.sqrt(
                float(y.square().sum(-1).mean() / y.shape[-1])
                if y.ndim == 2
                else float(y.square().mean())
            )
        return row
    except Exception:  # noqa: BLE001
        import traceback

        return {"case": path, "error": traceback.format_exc()[-1200:]}


def airfrans_case(arg):
    """Fit inputs for one AirfRANS case.

    Returns ISLA's scales on the in-plane boundaries the model declares (the
    file's airfoil and the reader's vtk_boundaries), the dimensionless RMS of nut,
    the nut and freestream statistics, or an error row.
    """
    path, yaml, vtk = arg
    torch.set_num_threads(1)
    from physicsnemo.mesh import DomainMesh, Mesh

    try:
        from domain_transforms import read_vtk_boundary

        m = _model(yaml, 2)
        dm = DomainMesh.load(path)
        case = os.path.basename(os.path.dirname(path))
        bnd = dict(dm.boundaries)
        bnd.update(
            {
                k: read_vtk_boundary(t.format(case=case), dm.interior.points, 2)
                for k, t in vtk.items()
            }
        )
        names = list(m.boundary_names)
        meshes = [
            Mesh(points=bnd[n].points[:, :2].double(), cells=bnd[n].cells)
            for n in names
        ]  # RestrictToPlane (axis 2)
        gd = dm.global_data.to(torch.float64).clone()
        gd["U_inf"] = gd["U_inf"][:2]
        gd["nu"] = torch.tensor(
            1.56e-5, dtype=torch.float64
        )  # the config's SetGlobalField
        s = _scales(m, names, meshes, gd)
        idx = torch.randperm(
            dm.interior.n_points, generator=torch.Generator().manual_seed(0)
        )[:20000]
        nut_all = dm.interior.point_data["nut"].reshape(-1).double()
        nut = nut_all[idx]
        return {
            "case": os.path.basename(os.path.dirname(path)),
            "L": float(s["L"]),
            "U": float(s["U"]),
            "log10_re": float(s["z"]),
            "nut": _rms(m, s, "nut", nut),
            "nut_mean": float(nut_all.mean()),
            "nut_std": float(nut_all.std(correction=0)),
            "U_inf_mag": float(dm.global_data["U_inf_mag"]),
            "aoa_deg": float(dm.global_data["aoa_deg"]),
            "Re": float(dm.global_data["Re"]),
        }
    except Exception:  # noqa: BLE001
        import traceback

        return {"case": path, "error": traceback.format_exc()[-1200:]}


def _run(fn, args, out, nw):
    with Pool(nw) as pool, open(out, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(fn, args, chunksize=1)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(i, len(args), flush=True)


def _inputs(rows):
    """Per input, the median over the cases where it is nonzero of its per-case RMS (with
    the count of such cases); an input zero on every case takes no scale."""
    keys = sorted({k for r in rows for k in r if "." in k})
    out = {}
    for k in keys:
        v = [r[k] for r in rows if k in r and r[k] > 0]
        if v:
            out[k] = {
                "scale": statistics.median(v),
                "n_nonzero": len(v),
                "n": sum(k in r for r in rows),
            }
    return out


def _fit(rows, fields):
    re = [r["log10_re"] for r in rows]
    return {
        "inputs": _inputs(rows),
        "n_cases": len(rows),
        "reynolds": {"mean": statistics.fmean(re), "std": statistics.pstdev(re)},
        "scale": {f: statistics.median(r[f] for r in rows) for f in fields},
        "scale_quartiles": {
            f: statistics.quantiles([r[f] for r in rows], n=4) for f in fields
        },
    }


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "aerozoo":
        data, yaml, out, shard, nsh, nw = (
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            int(sys.argv[5]),
            int(sys.argv[6]),
            int(sys.argv[7]),
        )
        man = json.load(open(os.path.join(data, "manifest.json")))
        cases = sorted(set().union(*(man[f"{s}.train"] for s in AZ_SPLITS)))[shard::nsh]
        done = set()
        if os.path.exists(out):
            done = {
                json.loads(line)["case"] for line in open(out) if "error" not in line
            }
        paths = [
            p
            for c in cases
            if c not in done
            for p in glob.glob(os.path.join(data, "shard-*", c, "domain.pdmsh"))
        ]
        print("cases", len(cases), "to do", len(paths), flush=True)
        _run(aerozoo_case, [(p, yaml) for p in paths], out, nw)
    elif mode == "airfrans":
        from omegaconf import OmegaConf

        ds, yaml, out, nw = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
        sys.path.insert(
            0, os.path.join(os.path.dirname(os.path.abspath(yaml)), "..", "..", "src")
        )
        ds = OmegaConf.load(ds)
        data = ds.train_datadir
        vtk = OmegaConf.to_container(
            ds.pipeline.reader.get("vtk_boundaries", {}), resolve=True
        )
        man = json.load(open(os.path.join(data, "manifest.json")))
        cases = sorted(set().union(*(man[s] for s in (sys.argv[6:] or [AF_SPLIT]))))
        paths = [p for c in cases for p in glob.glob(os.path.join(data, c, "*.pdmsh"))]
        print("cases", len(cases), "paths", len(paths), "vtk", vtk, flush=True)
        _run(airfrans_case, [(p, yaml, vtk) for p in paths], out, nw)
    elif mode == "aerozoo_rms":
        data, out, shard, nsh, nw = (
            sys.argv[2],
            sys.argv[3],
            int(sys.argv[4]),
            int(sys.argv[5]),
            int(sys.argv[6]),
        )
        man = json.load(open(os.path.join(data, "manifest.json")))
        cases = sorted(set().union(*(man[f"{s}.train"] for s in AZ_SPLITS)))[shard::nsh]
        paths = [
            p
            for c in cases
            for p in glob.glob(os.path.join(data, "shard-*", c, "domain.pdmsh"))
        ]
        print("cases", len(cases), "paths", len(paths), flush=True)
        _run(aerozoo_rms_case, paths, out, nw)
    elif mode == "aerozoo_rms_fit":
        from omegaconf import OmegaConf

        az_man, yaml, out = json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
        schema = OmegaConf.to_container(OmegaConf.load(yaml).model.output_schema)
        fields = [f for f, v in schema.items() if not v.get("units")]  # scale factor 1
        rows = [json.loads(line) for f in sys.argv[5:] for line in open(f)]
        errors = [r for r in rows if "error" in r]
        good = {r["case"]: r["rms_si"] for r in rows if "error" not in r}
        fits = {}
        for s in AZ_SPLITS:
            have = [good[c] for c in az_man[f"{s}.train"] if c in good]
            fits[s] = {
                "n_cases": len(have),
                "n_train": len(az_man[f"{s}.train"]),
                "scale": {f: statistics.median(r[f] for r in have) for f in fields},
                "scale_quartiles": {
                    f: statistics.quantiles([r[f] for r in have], n=4) for f in fields
                },
            }
        json.dump(
            {
                "fields": fields,
                "fits": fits,
                "n_errors": len(errors),
                "errors": errors[:5],
            },
            open(out, "w"),
            indent=1,
        )
        print(
            json.dumps(
                {
                    k: {kk: v[kk] for kk in ("n_cases", "n_train", "scale")}
                    for k, v in fits.items()
                },
                indent=1,
            )
        )
        print("errors", len(errors), errors[:2])
    elif mode == "airfrans_fit":
        af_man, out, splits = json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
        rows = [json.loads(line) for f in sys.argv[5:] for line in open(f)]
        errors = [r for r in rows if "error" in r]
        good = {r["case"]: r for r in rows if "error" not in r}
        fits = {}
        for s in splits.split(","):
            names = sorted(
                set().union(*(af_man[x] for x in s.split("+")))
            )  # a+b: the union
            af = [good[c] for c in names if c in good]
            u = [r["U_inf_mag"] for r in af]
            fits["airfrans." + s] = {
                **_fit(af, AF_FIELDS),
                "n_train": len(names),
                "nut_norm": {
                    "mean": statistics.fmean(r["nut_mean"] for r in af),
                    "std": statistics.fmean(r["nut_std"] for r in af),
                },
                "U_inf_mag": {"mean": statistics.fmean(u), "std": statistics.pstdev(u)},
            }
        json.dump(
            {"fits": fits, "n_errors": len(errors), "errors": errors[:5]},
            open(out, "w"),
            indent=1,
        )
        print(
            json.dumps(
                {
                    k: {
                        kk: v[kk]
                        for kk in (
                            "n_cases",
                            "n_train",
                            "reynolds",
                            "scale",
                            "nut_norm",
                            "U_inf_mag",
                        )
                    }
                    for k, v in fits.items()
                },
                indent=1,
            )
        )
        print("errors", len(errors), errors[:2])
    else:
        az_man, af_man, out = (
            json.load(open(sys.argv[2])),
            json.load(open(sys.argv[3])),
            sys.argv[4],
        )
        rows = [json.loads(line) for f in sys.argv[5:] for line in open(f)]
        errors = [r for r in rows if "error" in r]
        good = {r["case"]: r for r in rows if "error" not in r}
        fits = {}
        for s in AZ_SPLITS:
            have = [
                good[c]
                for c in az_man[f"{s}.train"]
                if c in good and "omega" in good[c]
            ]
            fits[s] = {**_fit(have, AZ_FIELDS), "n_train": len(az_man[f"{s}.train"])}
        af = [good[c] for c in af_man[AF_SPLIT] if c in good and "omega" not in good[c]]
        fits["airfrans." + AF_SPLIT] = {
            **_fit(af, AF_FIELDS),
            "n_train": len(af_man[AF_SPLIT]),
        }
        json.dump(
            {"fits": fits, "n_errors": len(errors), "errors": errors[:5]},
            open(out, "w"),
            indent=1,
        )
        print(
            json.dumps(
                {
                    k: {kk: v[kk] for kk in ("n_cases", "n_train", "reynolds", "scale")}
                    for k, v in fits.items()
                },
                indent=1,
            )
        )
        print("errors", len(errors), errors[:2])
