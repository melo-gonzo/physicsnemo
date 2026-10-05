# QI-KERNEL results (2026-10-05)

The GLOBE-style boundary read for read-only ISLA (`kernel_read="sum"`, R = 0.01 L,
last 4 blocks), epoch 500, seeds 42 and 43, DrivAerML Volume and AirfRANS. Report:
`examples/cfd/isla/research/qi_kernel/REPORT.md`. AGA task dir
`agents/2026-10-03-isla-qi-kernel`.

- `kq_reduction.{json,txt}`: the reduction against the QI references
  (`results/qi_2026-10-03`) and the preregistered bars (`kq_reduce.py`), with the
  probe summaries of the four final checkpoints.
- `score/<run>/metrics.jsonl`: `infer.py` scorer rows, float32 (the protocol).
- `probe/<run>.json`: `qi_probe.py` on the final checkpoints (independence,
  covariance, band anatomy).
- `score_bf16/<run>/metrics.jsonl` and `kq_reduction_bf16.json`: every arm (kernel
  and QI references) re-scored with `infer.py` at `precision=bfloat16`; a diagnostic,
  not the protocol.
- `prec_probe/<run>.json` (and `prec_probe_groups/`): `prec_probe.py`, which part of
  a trained kernel-read model depends on the arithmetic.
- `band_swap_<dataset>_vs_<ref>.json`: `band_swap.py`, the excess error by
  \|sdf\|/L band against interacting (`int`) and read-only (`pas`).
- `pilot/`: the 100-epoch pilots' scorer rows and probes.
