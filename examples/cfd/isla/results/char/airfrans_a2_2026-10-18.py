"""AirfRANS standing with ISLA's normalized slice reads (A2), in the schema of the other results/char/<dataset>_a2.json files.

ISLA's arm is afu_a2s_isla_seed42/43: ISLA at its reference configuration (width 512, 12 layers, 512 slices, 3.75e-4,
normalized slice reads, snapshot code_a2s on recipe_v11), trained 500 epochs on the uniform-interior AirfRANS files and
scored in float32 on the 200 full_test cases under the three scorings of one checkpoint; it is the control arm of GEO1
(#sec-nb-geo1-airfrans). The replication pair afu_bn2_isla_seed42/43 (H-RESID arm A2 on snapshot code_v20a2, the registered
P5 pair of #sec-nb-airfrans-hresid-p5) is stored under "replication". The baselines are the arms of airfrans_uniform.json
(afu_r_gt, afu_r_t: same uniform-interior files, same scorer), and the raw-read ISLA arm of that file is kept under
"raw_read_isla", as the other *_a2.json files do. Records: campaigns_2026-10-14/afu/protocols/<run>/<protocol>/metrics.jsonl
(their provenance is in afu_protocol_artifacts_2026-10-14.py).
usage: python airfrans_a2_2026-10-18.py  ->  results/char/airfrans_a2.json"""
import importlib.util
import json
import os

R = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("afu", f"{R}/afu_protocol_artifacts_2026-10-14.py")
afu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(afu)

SRC = f"{os.path.dirname(R)}/campaigns_2026-10-14/afu/protocols"
ISLA_ARM = "ISLA 512 with 512 slices"


def main():
    base = json.load(open(f"{R}/airfrans_uniform.json"))
    a2, ok = afu.arm(SRC, "afu_a2s_isla")
    rep, ok_rep = afu.arm(SRC, "afu_bn2_isla")
    assert afu.full(a2) and afu.full(rep), "an A2 record is missing"
    out = {
        "protocol": __doc__,
        "protocols": base["protocols"],
        "fields": base["fields"],
        "isla_arm": ISLA_ARM,
        "arms": {ISLA_ARM: a2, "GeoTransolver 256": base["arms"]["GeoTransolver 256"],
                 "Transolver 512 x 16": base["arms"]["Transolver 512 x 16"]},
        "complete": bool(ok and base["complete"]),
        "replication": {"label": "ISLA 512 with 512 slices, normalized reads, snapshot code_v20a2 (afu_bn2_isla)", **rep},
        "raw_read_isla": {"label": "ISLA 512 with 512 slices, raw slice reads (afu_r_isla)", **base["arms"][ISLA_ARM]},
    }
    path = f"{R}/airfrans_a2.json"
    json.dump(out, open(path, "w"), indent=1)
    isla = out["arms"][ISLA_ARM]
    print(f"== {path} ({'complete' if out['complete'] else 'partial'})")
    for arm_name, d in out["arms"].items():
        print(f"{arm_name:28s}", "  ".join(f"{pk}: " + " / ".join(f"{d[pk][f]['mean']:.5f}" for f in afu.FIELDS) for pk in afu.PROTOCOLS))
        if arm_name != ISLA_ARM:
            print(f"{'  ÷ ISLA':28s}", "  ".join(f"{pk}: " + " / ".join(f"{d[pk][f]['mean'] / isla[pk][f]['mean']:.3f}" for f in afu.FIELDS)
                                               for pk in afu.PROTOCOLS))
    print("replication ÷ standing (native):", " / ".join(f"{rep['native'][f]['mean'] / isla['native'][f]['mean']:.3f}" for f in afu.FIELDS))


if __name__ == "__main__":
    main()
