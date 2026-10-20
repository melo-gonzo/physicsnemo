"""GEO0 unregistered diagnostic: distribution over the 200 AirfRANS test cases of the per-case area/uniform ratio
(g0a/g0u) of the relative L2, provided vs internal, to see whether r is carried by a few cases."""
import json, statistics as st
EV = "evals_lhr"
def lp(f):
    cur = {}
    for line in open(f):
        r = json.loads(line)
        if r.get("phase") == "config": cur = {}
        elif r.get("phase") == "infer_step": cur[r["sample_id"]] = r["metrics"]
    return cur
for geo in ("provided", "internal"):
    for s in (42, 43):
        run = f"sg_af_{geo}_seed{s}"
        a, u = (lp(f"{EV}/{run}/{p}/{run}/metrics.jsonl") for p in ("g0a", "g0u"))
        for f in ("pressure_l2", "velocity_l2"):
            q = sorted(a[c][f] / u[c][f] for c in a)
            print(f"{run} {f[:-3]}: per-case a/u median {st.median(q):.3f}, IQR {q[50]:.3f}-{q[150]:.3f}, "
                  f"fraction > 1: {sum(x > 1 for x in q) / len(q):.2f}, max {q[-1]:.2f}")
