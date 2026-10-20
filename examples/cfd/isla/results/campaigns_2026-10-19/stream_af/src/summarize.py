"""STREAM-AF: the book's para-isla-stream numbers (and a few more) for the AeroZoo original and the AirfRANS re-measure."""
import json, sys
def summ(path, raw, a2):
    SN = json.load(open(path))["runs"]
    fac = lambda e: e[-1]["norm_in"][-1] / e[-1]["norm_in"][0]
    g = lambda k, ep: next((x["stream_out"] for x in SN[k]["epochs"] if x["epoch"] == ep), None)
    print(f"== {path.split('/')[-1]}")
    for grp, ks in (("raw", raw), ("A2", a2)):
        for k in ks:
            e = SN[k]["epochs"]
            print(f"  {k:26s} e{e[0]['epoch']}->{e[-1]['epoch']}  stream_out {e[0]['stream_out']:9.4g} -> {e[-1]['stream_out']:9.4g} "
                  f"(x{e[-1]['stream_out']/e[0]['stream_out']:.3g}; max {max(x['stream_out'] for x in e):.4g}) | e99 {g(k,99)} "
                  f"| depth factor norm_in[11]/norm_in[0] at last {fac(e):.3g} | log10_prod_amp last {e[-1]['log10_prod_amp']:.3g}"
                  f" | max amp_bcast last {max(e[-1]['amp_bcast']):.3g} | val_rel_l2 last {', '.join(f'{a}={b:.3g}' for a,b in e[-1]['val_rel_l2'].items())}")
summ("/home/psharpe/gh/physicsnemo/scratch/isla-stream-af-2026-10-19/src/stream_norms_aerozoo_ref.json",
     ["azu_full_isla_seed42", "azu_full_isla_seed43"], ["azu_full_islabn2_seed42", "azu_full_islabn2_seed43"])
summ("/home/psharpe/gh/physicsnemo/scratch/isla-stream-af-2026-10-19/stream_norms_airfrans.json",
     ["afu_r_isla_seed42", "afu_r_isla_seed43"], ["afu_bn2_isla_seed42", "afu_bn2_isla_seed43", "afu_a2s_isla_seed42", "afu_a2s_isla_seed43"])
summ("/home/psharpe/gh/physicsnemo/scratch/isla-stream-af-2026-10-19/stream_norms_deepjebpp.json",
     ["cl_r_jeb_isla_seed42", "cl_r_jeb_isla_seed43"], ["jbn2_isla_seed42", "jbn2_isla_seed43"])
