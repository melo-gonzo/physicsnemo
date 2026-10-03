import json,sys,os,glob
for d in sorted(glob.glob(sys.argv[1]+"/runs/cs_*")):
    r=os.path.basename(d); tr=val=None
    try:
        for l in open(d+"/metrics.jsonl"):
            j=json.loads(l)
            if j.get("phase")=="train_summary": tr=j
            if j.get("phase")=="val_summary": val=j
    except FileNotFoundError: pass
    def m(j):
        if not j: return "-"
        ks=[k for k in j if k.endswith("_l2") and not any(a in k for a in ("_x_","_y_","_z_"))]
        return " ".join(f"{k}={j[k]:.4f}" for k in ks)
    peak = f"{tr['peak_memory_gib']:.1f}" if tr and 'peak_memory_gib' in tr else "-"
    es = f"{tr['epoch_seconds']:.0f}" if tr and 'epoch_seconds' in tr else "-"
    done = "yes" if os.path.exists(d+"/train.log") and "Training completed" in open(d+"/train.log").read() else "NO"
    print(f"{r:13s} completed={done:3s} peak_gib={peak:>6s} epoch_s={es:>5s} val: {m(val)}")
