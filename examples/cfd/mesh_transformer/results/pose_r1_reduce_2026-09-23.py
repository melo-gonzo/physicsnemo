"""POSE under the reference configuration (REF-BOARD pose twin, 2026-09-23): ISLA R1 (384 x 12, cosine routing, fused kernel,
5e-4) trained on the DrivAerML Surface set with every training and validation case in its own random SO(3) pose
(rb_dr_isla_pose_seed{42,43}, no augmentation), scored in float32 at 200,000 cells on the 48 posed validation cars by the
fleet scorer (campaigns_2026-09-23/rb/metrics), against the same configuration's native-frame standing
(ref_isla_r1_dr435, results/ref_surface_2026-09-23.json). The rotation-augmented GeoTransolver-256 and Transolver-256 arms
and their native-pose references are copied unchanged from results/p200k_pose_reduction_2026-09-19.json (released
baselines, not re-run). Same schema as that artifact so the chapter cell reads either.
Output: p200k_pose_r1_2026-09-23.json."""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
RES = Path(__file__).resolve().parent
old = json.load(open(RES / "p200k_pose_reduction_2026-09-19.json"))
ref = json.load(open(RES / "ref_surface_2026-09-23.json"))["summary"]["isla_r1|dr435"]["fields"]["pressure_l2"]
per_seed = {}
for s in (42, 43):
    p = RES / "campaigns_2026-09-23" / "rb" / "metrics" / f"rb_dr_isla_pose_seed{s}" / f"rb_dr_isla_pose_seed{s}" / "metrics.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    v = [r["metrics"]["pressure_l2"] for r in rows if r.get("phase") == "infer_step"]
    assert len(v) == 48, (s, len(v))
    per_seed[f"seed{s}"] = {"pressure_l2": st.fmean(v), "n": len(v)}
posed = st.fmean(a["pressure_l2"] for a in per_seed.values())
native = ref["mean"]
arms = dict(old["arms"]); arms["isla"] = {"per_seed": per_seed, "posed_mean": posed, "posed_over_native": posed / native,
                                          "lanes": ["rb_dr_isla_pose_seed42", "rb_dr_isla_pose_seed43"], "native_lanes": ["ref_isla_r1_dr435_seed42", "ref_isla_r1_dr435_seed43"]}
nfr = dict(old["native_frame_reference"]); nfr["isla"] = native
nfr["source"] = "ISLA: results/ref_surface_2026-09-23.json (reference configuration, 435 cars, 200,000 cells); baselines: " + old["native_frame_reference"]["source"]
verdict = {"isla_posed_over_native": posed / native,
           "gt_aug_posed_over_isla_posed": arms["gt_unit"]["posed_mean"] / posed,
           "t_aug_posed_over_isla_posed": arms["transolver_unit"]["posed_mean"] / posed,
           "bars": old["verdict"]["bars"]}
verdict["verdict"] = "supports" if verdict["isla_posed_over_native"] <= 1.03 and min(verdict["gt_aug_posed_over_isla_posed"], verdict["t_aug_posed_over_isla_posed"]) >= 1.05 else "not supported"
out = {k: v for k, v in old.items() if k not in ("arms", "native_frame_reference", "verdict")}
out.update({"study": "POSE-REF under the reference configuration (REF-BOARD twin)", "protocol": __doc__, "arms": arms, "native_frame_reference": nfr, "verdict": verdict})
(RES / "p200k_pose_r1_2026-09-23.json").write_text(json.dumps(out, indent=1))
print(f"ISLA posed {posed:.4f} (seeds {[round(a['pressure_l2'],4) for a in per_seed.values()]}) / native {native:.4f} = {posed/native:.3f}; "
      f"GT aug ÷ posed ISLA {verdict['gt_aug_posed_over_isla_posed']:.2f}, T {verdict['t_aug_posed_over_isla_posed']:.2f}; {verdict['verdict']}")
