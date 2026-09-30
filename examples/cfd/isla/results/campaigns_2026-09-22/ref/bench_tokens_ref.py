"""COST-SWEEP-REF (2026-09-22): training-step time and peak memory against token count for the reference-configuration
candidates (ISLA 384x12 and 384x24, cosine routing, fused geometry kernel, compiled), ISLA 192x12 under the same
settings, GeoTransolver at 256 and 512 wide (compiled) and Transolver at 256 wide (compiled); one GB300, bf16 autocast,
batch 1, AdamW step; random inputs, median of ten measured steps after three warm-up steps; OOM recorded as a result.
Runs under the code_ref snapshot (current ISLA class) with the kernel study's bench_common / bench_pareto helpers on
the path (they supply the instrument, the GeoTransolver wrapper and the Transolver wrapper).

usage: python bench_tokens_ref.py <out.json> [tokens,comma,separated]
"""
import sys

import torch

import bench_common as bc  # noqa: E402  (kernel study helpers)
from bench_pareto import GTWrap, GT_KW  # noqa: E402
from bench_tokens import TWrap  # noqa: E402
from physicsnemo.experimental.nn.isla import ISLA  # noqa: E402

ISLA_KW = dict(out_scalars=1, out_vectors=1, n_slices=256, mlp_ratio=4, routing_norm="cosine", geo_kernel="fused",
               scale_mode="rms_distance")


class ISLAWrap(torch.nn.Module):
    """ISLA with the benchmark's input names (drive -> global_vectors)."""

    def __init__(self, **kw):
        super().__init__()
        torch.manual_seed(0)
        self.m = ISLA(**ISLA_KW, **kw)

    def forward(self, points, normals, drive, measure_weights):
        return self.m(points=points, normals=normals, measure_weights=measure_weights, global_vectors=drive[:, None, :])


class GTWrapW(GTWrap):
    """GeoTransolver at a chosen width."""

    def __init__(self, n_hidden):
        torch.nn.Module.__init__(self)
        from physicsnemo.models.geotransolver import GeoTransolver
        torch.manual_seed(0)
        self.m = GeoTransolver(**dict(GT_KW, n_hidden=n_hidden))


ARMS = {
    "isla_ref_r1_384x12_cos_fused": lambda: torch.compile(ISLAWrap(hidden=384, n_layers=12).cuda()),
    "isla_ref_r2_384x24_cos_fused": lambda: torch.compile(ISLAWrap(hidden=384, n_layers=24).cuda()),
    "isla_192x12_cos_fused": lambda: torch.compile(ISLAWrap(hidden=192, n_layers=12).cuda()),
    "geotransolver_256": lambda: torch.compile(GTWrapW(256).cuda()),
    "geotransolver_512": lambda: torch.compile(GTWrapW(512).cuda()),
    "transolver_256": lambda: torch.compile(TWrap().cuda()),
}


def main():
    out_path = sys.argv[1]
    tokens = [int(t) for t in (sys.argv[2].split(",") if len(sys.argv) > 2 else
                               ["2500", "5000", "10000", "20000", "40000", "80000", "100000", "200000", "400000", "600000"])]
    res = dict(env=bc.env_info(), isla_kw=ISLA_KW, gt_kw=GT_KW, arms=list(ARMS), tokens=tokens, runs=[])
    for n in tokens:
        for name, build in ARMS.items():
            torch._dynamo.reset()
            rec = bc.run_config(build, n, 1, True)
            rec["arm"] = name
            res["runs"].append(rec)
            print(f"{name:30s} N={n:6d}: {rec.get('step_ms_median', float('nan')):8.1f} ms  peak {rec.get('peak_allocated_gib', float('nan')):6.2f} GiB  "
                  f"{rec['status']} {rec.get('error', '')[-160:] if rec['status'] != 'ok' else ''}", flush=True)
            bc.dump(res, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
