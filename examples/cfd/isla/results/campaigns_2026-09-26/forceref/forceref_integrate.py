"""FORCEREF (2026-09-26): integrated force coefficients on the 200,000-cell DrivAerML Surface checkpoints.

Runs on the cluster under the code_ref snapshot (PYTHONPATH=code_ref:recipe_globin/src). Two force references,
integrated by the same function:

  (1) truth_full     : ground-truth fields on ALL vehicle cells of the full validation mesh (about 17M cells);
  (2) truth_sampled  : ground-truth fields on the 200,000 cells the model was evaluated on, each cell carrying
                       its Horvitz-Thompson measure weight (the recipe's `_measure_weights`, N/k for a k-of-N draw);
  (3) pred           : the checkpoint's prediction on the same 200,000 cells, same weights.

Traction coefficient t = -Cp n + Cf with Cp = (p - p_inf)/q_inf, Cf = tau/q_inf, q_inf = 0.5 rho |U_inf|^2; the
force coefficient vector is sum_c t_c A_c w_c with A_ref = 1 (the recipe's infer.yaml convention), projected on
the (drag, lift, side) triad of forces.build_axis_frame (drag along U_inf, lift = z orthogonalized against drag).
Cell normals follow the cell winding (Mesh.cell_normals), the same convention the recipe's ForceContext uses; the
subsample keeps the winding, so the sign convention is identical across (1), (2), (3).

Usage: python forceref_integrate.py <T> <run1> [<run2> ...]
  reads  <T>/forceref/<run>/<run>/predictions/*.pdmsh and <T>/forceref/<run>/<run>/metrics.jsonl
  writes <T>/forceref/forceref_cases.json
"""

import json
import re
import sys
import time
from pathlib import Path

import torch
from tensordict import TensorDict

from forces import build_axis_frame
from physicsnemo.datapipes.readers.mesh import read_only_memory_maps
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import MEASURE_WEIGHTS_KEY, cell_measures

DATA_ROOT = Path("/home/psharpe/coreai_modulus_cae/datasets/PhysicsNeMo-DrivaerML")
UP = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)


def coefficients(mesh: Mesh, p: torch.Tensor, tau: torch.Tensor, gd: TensorDict) -> dict:
    """Force coefficients (A_ref = 1) from per-cell physical pressure and wall shear stress."""
    U = gd["U_inf"].double().reshape(3)
    rho = gd["rho_inf"].double()
    p_inf = gd["p_inf"].double()
    q = 0.5 * rho * (U @ U)
    cp = (p.double().reshape(-1) - p_inf) / q
    cf = tau.double().reshape(-1, 3) / q
    n = mesh.cell_normals.double()
    mu = cell_measures(mesh).double()  # area * measure weight
    traction = -cp[:, None] * n + cf
    F = (traction * mu[:, None]).sum(0)
    F_p = (-cp[:, None] * n * mu[:, None]).sum(0)
    drag, lift, side = build_axis_frame(U, UP)
    out = {
        "CD": float(F @ drag), "CL": float(F @ lift), "CS": float(F @ side),
        "CD_pressure": float(F_p @ drag), "CL_pressure": float(F_p @ lift),
        "n_cells": int(mesh.n_cells),
        "area_represented": float(mu.sum()),
        "area_geometric": float(mesh.cell_areas.double().sum()),
    }
    w = mesh.cell_data.get(MEASURE_WEIGHTS_KEY, None)
    out["measure_weight"] = None if w is None else float(w.double().mean())
    return out


def case_of(sample_id: str) -> str:
    m = re.search(r"(run_\d+)", sample_id)
    assert m, sample_id
    return m.group(1)


def truth_full(case: str) -> dict:
    root = DATA_ROOT / case / f"domain_{case}.pdmsh" / "_tensordict"
    with read_only_memory_maps():
        vehicle = Mesh.load(root / "boundaries" / "vehicle")
        gd = TensorDict.load_memmap(root / "global_data")
    p = vehicle.cell_data["pMeanTrim"]
    tau = vehicle.cell_data["wallShearStressMeanTrim"]
    out = coefficients(vehicle, p, tau, gd)
    out["U_inf"] = [float(v) for v in gd["U_inf"].reshape(3)]
    out["L_ref"] = float(gd["L_ref"])
    return out


def main() -> None:
    T = Path(sys.argv[1])
    runs = sys.argv[2:]
    out_path = T / "forceref" / "forceref_cases.json"
    result = {"protocol": __doc__, "runs": {}, "truth_full": {}}
    torch.set_num_threads(16)

    for run in runs:
        run_dir = T / "forceref" / run / run
        recipe = {}
        for line in (run_dir / "metrics.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if rec.get("phase") == "infer_step" and "forces" in rec:
                recipe[rec["sample_id"]] = rec["forces"]
        cases = {}
        for pd_path in sorted((run_dir / "predictions").glob("*.pdmsh")):
            t0 = time.time()
            sample_id = pd_path.name[: -len(".pdmsh")]
            with read_only_memory_maps():
                domain = DomainMesh.load(pd_path)
            vehicle = domain.boundaries["vehicle"]
            pdat = domain.interior.point_data
            assert domain.interior.points.shape[0] == vehicle.n_cells
            gd = domain.global_data
            entry = {
                "sample_id": sample_id,
                "pred": coefficients(vehicle, pdat["pred_pressure"], pdat["pred_wss"], gd),
                "truth_sampled": coefficients(vehicle, pdat["true_pressure"], pdat["true_wss"], gd),
                "recipe_forces": recipe.get(sample_id),
            }
            cases[case_of(sample_id)] = entry
            print(f"[{run}] {sample_id}: CD pred {entry['pred']['CD']:.4f} true_sampled {entry['truth_sampled']['CD']:.4f}"
                  f" recipe {recipe.get(sample_id, {}).get('true', {}).get('CD')} ({time.time() - t0:.1f}s)", flush=True)
        result["runs"][run] = cases

    all_cases = sorted({c for cases in result["runs"].values() for c in cases}, key=lambda s: int(s.split("_")[1]))
    for case in all_cases:
        t0 = time.time()
        result["truth_full"][case] = truth_full(case)
        tf = result["truth_full"][case]
        print(f"[full] {case}: CD {tf['CD']:.4f} CL {tf['CL']:.4f} n_cells {tf['n_cells']} area {tf['area_geometric']:.3f} ({time.time() - t0:.1f}s)", flush=True)
        out_path.write_text(json.dumps(result, indent=1))
    out_path.write_text(json.dumps(result, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
