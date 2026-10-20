"""E2 GEO-CONT: reduce the Tensile2d scores (scores/<run>/<protocol>/<run>/metrics.jsonl), the A2-RERUN way: the
infer_summary row, else the mean of the per-sample infer_step rows (last row per sample id)."""
import json, math, statistics as st
F = {"stress_l2": "stress", "displacement_l2": "displacement", "q_l2": "plastic strain"}
P = ("native", "area", "posed")
def summary(path):
    by = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("phase") == "infer_summary": return r["metrics"]
        if r.get("phase") == "infer_step": by[r.get("sample_id", len(by))] = r["metrics"]
    return {k: st.mean(m[k] for m in by.values()) for k in F}
v = {}
for arm in ("ctl", "cont"):
    for s in ("42", "43"):
        run = f"gcont_{arm}_t2dz_isla_seed{s}"
        for p in P:
            m = summary(f"scores/{run}/{p}/{run}/metrics.jsonl")
            for f in F: v[(arm, s, p, f)] = m[f]
print("| arm | protocol | " + " | ".join(F.values()) + " |"); print("|---|---|" + "---|" * len(F))
for arm in ("ctl", "cont"):
    for p in P:
        print(f"| {arm} | {p} | " + " | ".join(f"{st.mean(v[(arm,s,p,f)] for s in ('42','43')):.4f} ({v[(arm,'42',p,f)]:.4f}, {v[(arm,'43',p,f)]:.4f})" for f in F) + " |")
print("\nCONT / CONTROL native (seed means; per-seed CONT / CONTROL mean; CONTROL two-seed range; CONT seeds outside it?):")
for f, lab in F.items():
    c = [v[("ctl", s, "native", f)] for s in ("42", "43")]; n = [v[("cont", s, "native", f)] for s in ("42", "43")]
    r = st.mean(n) / st.mean(c); out = all(x < min(c) or x > max(c) for x in n) and (all(x < min(c) for x in n) or all(x > max(c) for x in n))
    print(f"- {lab}: {r:.3f}; seeds {n[0]/st.mean(c):.3f}, {n[1]/st.mean(c):.3f}; CONTROL range [{min(c):.4f}, {max(c):.4f}]; both CONT seeds outside on one side: {out}")
print("\nposed / native per seed (CONTROL | CONT):")
for f, lab in F.items():
    print(f"- {lab}: " + "; ".join(f"s{s} {v[('ctl',s,'posed',f)]/v[('ctl',s,'native',f)]:.4f} | {v[('cont',s,'posed',f)]/v[('cont',s,'native',f)]:.4f}" for s in ("42", "43")))
print("\narea / native per seed (CONTROL | CONT):")
for f, lab in F.items():
    print(f"- {lab}: " + "; ".join(f"s{s} {v[('ctl',s,'area',f)]/v[('ctl',s,'native',f)]:.4f} | {v[('cont',s,'area',f)]/v[('cont',s,'native',f)]:.4f}" for s in ("42", "43")))
json.dump({f"{a}|{s}|{p}|{f}": x for (a, s, p, f), x in v.items()}, open("t2d_scores.json", "w"), indent=1)
