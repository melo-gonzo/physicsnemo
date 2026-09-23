"""TRAIN-AREA-DR twin under the reference configuration R1 (REF-BOARD, 2026-09-23): ISLA R1 trained on 435 DrivAerML
cars under the area-proportional Poisson draw at 200,000 expected cells (rb_dr_isla_area_seed{42,43}) against the
uniform-trained R1 reference (ref_isla_r1_dr435_seed{42,43}), both scored by the consistency benchmark's independent
uniform, stratified and 10:1-biased draws at 200,000 cells, float32, 48 validation cars (mirror
campaigns_2026-09-23/rp/metrics/<run>/<draw>_200000/<run>/metrics.jsonl). Bar (TRAIN-AREA-DR): area-trained /
uniform-trained on the uniform draw within 5% = training-law independent; the 192-wide lanes at 100,000 cells gave 1.08.
Output: trainarea_dr_r1_2026-09-23.json."""
from __future__ import annotations
import json, statistics as st
from pathlib import Path
RES = Path(__file__).resolve().parent
ROOT = RES / "campaigns_2026-09-23" / "rp" / "metrics"
DRAWS = ("unif", "strat", "biased")
def mean_p(run, draw):
    p = ROOT / run / f"{draw}_200000" / run / "metrics.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    v = [r["metrics"]["pressure_l2"] for r in rows if r.get("phase") == "infer_step"]
    assert len(v) == 48, (run, draw, len(v))
    return st.fmean(v)
arms = {}
for name, runs in {"isla_r1_area": ["rb_dr_isla_area_seed42", "rb_dr_isla_area_seed43"],
                   "isla_r1_uniform": ["ref_isla_r1_dr435_seed42", "ref_isla_r1_dr435_seed43"]}.items():
    per = {d: [mean_p(r, d) for r in runs] for d in DRAWS}
    arms[name] = {"runs": runs, "seeds": {d: per[d] for d in DRAWS}, "mean": {d: st.fmean(per[d]) for d in DRAWS}}
ratio = arms["isla_r1_area"]["mean"]["unif"] / arms["isla_r1_uniform"]["mean"]["unif"]
per_seed = [a / u for a, u in zip(arms["isla_r1_area"]["seeds"]["unif"], arms["isla_r1_uniform"]["seeds"]["unif"])]
old = json.load(open(RES / "trainarea_dr_reduction_2026-09-20.json"))["arms"]
out = {"study": "TRAIN-AREA-DR under R1 (REF-BOARD area-law twin)", "protocol": __doc__, "arms": arms,
       "area_over_uniform_on_uniform_draw": {"mean": ratio, "per_seed": per_seed, "bar": 1.05,
                                             "verdict": "within the 5% bar" if ratio <= 1.05 else "at or past the 5% bar: weakly dependent"},
       "density_factor_area_trained": arms["isla_r1_area"]["mean"]["biased"] / arms["isla_r1_area"]["mean"]["unif"],
       "comparators_192_wide_100k": {"isla": old["isla_area"]["mean"]["unif"] / old["isla_uniform"]["mean"]["unif"],
                                     "gt_unit": old["gt_unit_area"]["mean"]["unif"] / old["gt_unit_uniform"]["mean"]["unif"],
                                     "transolver_unit": old["transolver_unit_area"]["mean"]["unif"] / old["transolver_unit_uniform"]["mean"]["unif"]}}
(RES / "trainarea_dr_r1_2026-09-23.json").write_text(json.dumps(out, indent=1))
print(f"area/uniform on the uniform draw: {ratio:.4f} (seeds {per_seed[0]:.4f} / {per_seed[1]:.4f}); 192-wide {out['comparators_192_wide_100k']['isla']:.4f}; density factor {out['density_factor_area_trained']:.3f}")
