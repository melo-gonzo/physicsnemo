"""HILIFT-090 spatial probe (2026-10-20): run the recipe's src/infer.py unchanged, logging each cyclic block draw of the
reader (physicsnemo.datapipes.readers.mesh._cyclic_block_indices: total cells, first index, run breaks) to $BLOCKLOG.<pid>,
so the saved 200k-cell predictions can be mapped back onto the full mesh's element labels.
usage: python infer_wrap.py <infer.py> <overrides...>"""
import json
import os
import runpy
import sys

import torch

INF = sys.argv[1]
sys.argv = [INF] + sys.argv[2:]
import physicsnemo.datapipes.readers.mesh as M  # noqa: E402

_orig = M._cyclic_block_indices
LOG = os.environ["BLOCKLOG"]


def _wrap(total, k, *a, **kw):
    idx = _orig(total, k, *a, **kw)
    t = torch.as_tensor(idx).flatten()
    breaks = ((t[1:] - t[:-1]) != 1).nonzero().flatten().tolist()
    with open(f"{LOG}.{os.getpid()}", "a") as f:
        f.write(json.dumps({"total": int(total), "k": int(k), "first": int(t[0]), "last": int(t[-1]), "breaks": breaks}) + "\n")
    return idx


M._cyclic_block_indices = _wrap
runpy.run_path(INF, run_name="__main__")
