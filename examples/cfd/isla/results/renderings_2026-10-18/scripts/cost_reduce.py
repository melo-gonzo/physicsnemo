"""COST reduction (FIG lane, 2026-10-18): merge the five per-arm bench_step.py outputs of LHR job 271948 (cost.sbatch,
one GB300 node, each arm on its own GPU, the four GPU streams concurrent) into ../cost_sweep_ref_a2_gb300_bf16.json.

Per arm and token count: median, p10 and p90 of 25 timed training steps (forward + loss + backward + optimizer step,
bf16 autocast, after 8 untimed warm-up steps that include compilation), the peak allocated memory over the timed steps,
the resident bytes (parameters + optimizer state + the batch) before them, and the model's input shapes. "tokens" is the
DrivAerML sampling_resolution: surface cells of the vehicle (ISLA: boundary_meshes.vehicle.cells; GeoTransolver and
Transolver: geometry / embedding points); for the volume arms, N boundary cells plus N prediction points.
The learning rate does not enter the step time; the baseline arms composed the recipe default (3e-3) where their lanes ran
1e-3 (lane table), which changes no shape or operation.
Source files: scratch/isla-book-overhaul-2026-10-18/fig/cost/<arm>.json (copies of LHR
$T/book_figs_2026-10-18/cost/<arm>.json).
usage: python cost_reduce.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/psharpe/gh/physicsnemo/scratch/isla-book-overhaul-2026-10-18/fig/cost"
ARMS = {
    "isla_surface_512x12": dict(name="ISLA", task="DrivAerML Surface", code="code_a2r (7e167820a, A2 reads)", recipe="recipe_a2r"),
    "geotransolver_surface_256": dict(name="GeoTransolver 256", task="DrivAerML Surface", code="code_v20 (7ef7c34fa)", recipe="recipe_v8 (343b284eb)"),
    "transolver_surface_512x16": dict(name="Transolver 512 x 16", task="DrivAerML Surface", code="code_v20 (7ef7c34fa)", recipe="recipe_v8 (343b284eb)"),
    "isla_volume_512x12": dict(name="ISLA", task="DrivAerML Volume", code="code_a2r (7e167820a, A2 reads)", recipe="recipe_a2r"),
    "isla_dtt_volume_512x12": dict(name="ISLA-DTT", task="DrivAerML Volume", code="code_a2r (7e167820a, A2 reads)", recipe="recipe_a2r"),
}

out = {"protocol": __doc__, "job": "LHR 271948", "arms": {}}
for arm, meta in ARMS.items():
    d = json.load(open(f"{SRC}/{arm}.json"))
    out.setdefault("env", d["env"])
    rows = []
    for r in d["runs"]:
        row = {"tokens": r["tokens_requested"], "status": r["status"], "input_shapes": r.get("input_shapes")}
        if r["status"] == "ok":
            row.update({k: r[k] for k in ("step_ms_median", "step_ms_p10", "step_ms_p90", "peak_allocated_gib",
                                          "peak_allocated_bytes", "resident_bytes", "warmup_s", "step_ms")})
        else:
            row["error"] = r.get("error", "")[-300:]
        rows.append(row)
    out["arms"][arm] = {**meta, "model_yaml": d["model"], "dataset_yaml": d["dataset"], "extra": d["extra"],
                        "n_params": d.get("n_params"), "compile": d.get("compile"), "precision": d.get("precision"),
                        "warmup_steps": d["warmup"], "timed_steps": d["steps"], "gpu_env": d["env"], "runs": rows}
    print(f"{arm:28s} " + "  ".join(f"{r['tokens'] // 1000}k:" + (f"{r['step_ms_median']:.0f}ms/{r['peak_allocated_gib']:.0f}G"
                                                             if r["status"] == "ok" else r["status"]) for r in rows))
json.dump(out, open(os.path.join(os.path.dirname(HERE), "cost_sweep_ref_a2_gb300_bf16.json"), "w"), indent=1)
