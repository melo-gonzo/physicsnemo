"""AirfRANS boundary draw at a reduced budget, on the checkpoints of the AirfRANS standing (results/char/airfrans_a2.json).

The reduction of area2_2026-10-16.py applied to the normalized-read records: per run (seed) and field, the mean over cases of
the per-case relative L2 under three draws of the airfoil at an expected 400 of its about 1,000 segments, "ca2a" (inclusion in
proportion to the segment length), "ca2u" (uniform inclusion) and "ca2v" (uniform inclusion with an independent seed), each kept
segment's measure divided by its inclusion probability; "native" is the protocol record of the same checkpoint. Readouts per
field: r = ca2a / ca2u (the sampling law), d = |ca2v / ca2u - 1| (the noise floor of a redraw), count_effect = ca2u / native
(thinning the airfoil at a uniform law). The interior queries, the targets and the global inputs are the same in all draws.

Arms:
  ISLA  afu_a2s_isla_seed42/43: ISLA's reference configuration with the normalized slice reads, the ISLA arm of airfrans_a2.json.
  exact-geometry variant  afu_geo1_isla_seed42/43: the same configuration with each query's distance and direction taken from the
        exact closest point on the full-resolution airfoil instead of the nearest boundary token (branch psharpe/isla-geo-exact,
        not part of ISLA); kept for the open question on exact query geometry.
Records: campaigns_2026-10-14/afu/ca2_400/<run>/<draw>/metrics.jsonl, campaigns_2026-10-14/afu/protocols/<run>/native/metrics.jsonl
(float32, the 200 full_test cases; the last scoring pass per file counts).
usage: python airfrans_area2_a2_2026-10-18.py  ->  results/char/airfrans_area2_a2.json"""
import json
import os

R = os.path.dirname(os.path.abspath(__file__))
AFU = f"{os.path.dirname(R)}/campaigns_2026-10-14/afu"
FIELDS = ["pressure_l2", "velocity_l2", "nut_l2"]
ARMS = {"ISLA": "afu_a2s_isla", "exact-geometry variant": "afu_geo1_isla"}
DRAWS = ("ca2a", "ca2u", "ca2v")


def last_pass(path):
    """The per-case metrics of the last scoring pass in a metrics.jsonl (as area2_2026-10-16.py reads them)."""
    passes, cur = [], {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("phase") == "config":
            if cur:
                passes.append(cur)
            cur = {}
        elif r.get("phase") == "infer_step":
            cur[r["sample_id"]] = r["metrics"]
    if cur:
        passes.append(cur)
    return passes[-1] if passes else {}


def main():
    runs = {}
    for arm, stem in ARMS.items():
        for s in (42, 43):
            run = f"{stem}_seed{s}"
            p = {d: last_pass(f"{AFU}/ca2_400/{run}/{d}/metrics.jsonl") for d in DRAWS}
            p["native"] = last_pass(f"{AFU}/protocols/{run}/native/metrics.jsonl")
            keys = sorted(set(p["ca2a"]) & set(p["ca2u"]) & set(p["ca2v"]) & set(p["native"]))
            assert len(keys) == 200, (run, len(keys))
            rec = {"arm": arm, "seed": s, "n_cases": len(keys), "fields": {}}
            for f in FIELDS:
                m = {k: sum(p[k][c][f] for c in keys) / len(keys) for k in (*DRAWS, "native")}
                rec["fields"][f] = {"mean": m, "r": m["ca2a"] / m["ca2u"], "d": abs(m["ca2v"] / m["ca2u"] - 1),
                                    "count_effect": m["ca2u"] / m["native"]}
            runs[run] = rec
    out = {"protocol": __doc__, "budget_segments": 400, "fields": {"pressure_l2": "pressure", "velocity_l2": "velocity",
                                                                    "nut_l2": "eddy viscosity"}, "runs": runs}
    path = f"{R}/airfrans_area2_a2.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"== {path}")
    for run, rec in runs.items():
        for f, v in rec["fields"].items():
            m = v["mean"]
            print(f"{run:24s} {f:12s} native {m['native']:.4f}  u {m['ca2u']:.4f}  a {m['ca2a']:.4f}  v {m['ca2v']:.4f}  "
                  f"r {v['r']:.3f}  d {v['d']:.3f}  count {v['count_effect']:.2f}")


if __name__ == "__main__":
    main()
