# Cost extract of the ISLA 0.9.0 reference lanes (read by results/char/cost_090_2026-10-22.py)

`extract.jsonl` holds one line per training run, made on LHR (2026-10-22, read only) by streaming `extract_cost.py` over ssh:

```
ssh lhr2 "nice -n 19 <LHR venv python> - <run> ..." < extract_cost.py > extract.jsonl
```

Each line keeps only what the cost needs: the md5 and size of the run's `metrics.jsonl`, its last training config row (without
`params`) and the precision and compile entries of `params`, per epoch the last `train_summary` row's `epoch_seconds` and
`peak_memory_gib`, the step-time statistics over the `train_step` rows with `step_time_s` < 10 s (the rows themselves, 10^4 to
10^5 per run, stay on LHR), the CUDA devices, nodes and code snapshots `train.log` names, and the last config row of the run's
native-mesh scorer output.

`$T` = `/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0`.

md5: `extract_cost.py` 9b87878ef3cbd533db2bf606c78ae94c (license header, formatting and one docstring added after the run); `extract.jsonl` 5579907c846df8281409acbcddb8d3fd.

| run | source on LHR | md5 of metrics.jsonl | scorer config row |
| --- | --- | --- | --- |
| r090_dr_isla_seed42 | $T/runs/r090_dr_isla_seed42/metrics.jsonl | 65f6b1f84d900cbfa542a50a6d504102 | $T/evals_cmpaf_eager/r090_dr_isla_seed42/native/r090_dr_isla_seed42/metrics.jsonl |
| r090_dr_isla_seed43 | $T/runs/r090_dr_isla_seed43/metrics.jsonl | d74267d8eb0009f9652ab2109a40dc06 | $T/evals_cmpaf_eager/r090_dr_isla_seed43/native/r090_dr_isla_seed43/metrics.jsonl |
| r090_vol_isla_seed42 | $T/runs/r090_vol_isla_seed42/metrics.jsonl | 36124ae361c3e18cfbfe7b5183dfdb71 | $T/evals_r090/r090_vol_isla_seed42/native/r090_vol_isla_seed42/metrics.jsonl |
| r090_vol_isla_seed43 | $T/runs/r090_vol_isla_seed43/metrics.jsonl | d4c43ceceea6e2cf4c80e41b8fda9988 | $T/evals_r090/r090_vol_isla_seed43/native/r090_vol_isla_seed43/metrics.jsonl |
| r090_suv_isla_seed42 | $T/runs/r090_suv_isla_seed42/metrics.jsonl | b6be9d315e0ed3b04607a8d3ddfb2524 | $T/evals_cmpaf_eager/r090_suv_isla_seed42/native/r090_suv_isla_seed42/metrics.jsonl |
| r090_suv_isla_seed43 | $T/runs/r090_suv_isla_seed43/metrics.jsonl | 4ff077949398e7a9fd465c4d9e75d75d | $T/evals_cmpaf_eager/r090_suv_isla_seed43/native/r090_suv_isla_seed43/metrics.jsonl |
| r090_jb_isla_seed42 | $T/runs/r090_jb_isla_seed42/metrics.jsonl | c0578284d090db14b921a7a312e9201a | $T/evals_r090/r090_jb_isla_seed42/native/r090_jb_isla_seed42/metrics.jsonl |
| r090_jb_isla_seed43 | $T/runs/r090_jb_isla_seed43/metrics.jsonl | 6b8ab941b515ff1c4e306006edd21333 | $T/evals_r090/r090_jb_isla_seed43/native/r090_jb_isla_seed43/metrics.jsonl |
| h090_hl_isla_seed42 | $T/runs/h090_hl_isla_seed42/metrics.jsonl | 1cb341a8e770f4cbccc1446b11545ebd | $T/evals_cmpaf_eager/h090_hl_isla_seed42/native/h090_hl_isla_seed42/metrics.jsonl |
| h090_hl_isla_seed43 | $T/runs/h090_hl_isla_seed43/metrics.jsonl | 8c39e69460d9bdcd2441cbec0036983a | $T/evals_cmpaf_eager/h090_hl_isla_seed43/native/h090_hl_isla_seed43/metrics.jsonl |
| azc_full_isla_seed42 | $T/runs/azc_full_isla_seed42/metrics.jsonl | 4b5b0da69f76fd7a630721d0e6b05ba9 | $T/p200k_fp32_test_azc/azc_full_isla_seed42/azc_full_isla_seed42/metrics.jsonl |
| azc_full_isla_seed43 | $T/runs/azc_full_isla_seed43/metrics.jsonl | 9d9cbd548f5efb2701b8ef30ac446ad4 | $T/p200k_fp32_test_azc/azc_full_isla_seed43/azc_full_isla_seed43/metrics.jsonl |
| r090fix_af_isla_seed42 | $T/runs/r090fix_af_isla_seed42/metrics.jsonl | 1a2d87fc19e6832f30ed4277477dd8e6 | $T/evals_r090fix/r090fix_af_isla_seed42/native/r090fix_af_isla_seed42/metrics.jsonl |
| r090fix_af_isla_seed43 | $T/runs/r090fix_af_isla_seed43/metrics.jsonl | 91b6881db55c0d6473708aa02d6db2aa | $T/evals_r090fix/r090fix_af_isla_seed43/native/r090fix_af_isla_seed43/metrics.jsonl |
| r090fix_lug_isla_seed42 | $T/runs/r090fix_lug_isla_seed42/metrics.jsonl | 2b7137410febf4d9129bef6bba12a2dc | $T/evals_r090fix/r090fix_lug_isla_seed42/native/r090fix_lug_isla_seed42/metrics.jsonl |
| r090fix_lug_isla_seed43 | $T/runs/r090fix_lug_isla_seed43/metrics.jsonl | dfb1e91529c9a60ef79e20e2c9fac13d | $T/evals_r090fix/r090fix_lug_isla_seed43/native/r090fix_lug_isla_seed43/metrics.jsonl |
| r090fix_af_dtt_seed42 | $T/runs/r090fix_af_dtt_seed42/metrics.jsonl | 14232e76ab840ff66be21fc5b0af22cb | $T/evals_att2_bases/r090fix_af_dtt_seed42/native/r090fix_af_dtt_seed42/metrics.jsonl |
| r090fix_af_dtt_seed43 | $T/runs/r090fix_af_dtt_seed43/metrics.jsonl | 529bc2a01a3e87b1b7a195485c7fd498 | $T/evals_att2_bases/r090fix_af_dtt_seed43/native/r090fix_af_dtt_seed43/metrics.jsonl |
| r090fix_lug_dtt_seed42 | $T/runs/r090fix_lug_dtt_seed42/metrics.jsonl | fd0a9691406f0962333dfde9246f2407 | none (not yet scored) |
| r090fix_lug_dtt_seed43 | $T/runs/r090fix_lug_dtt_seed43/metrics.jsonl | 24ee207661538f99fba7842fd385848d | none (not yet scored) |
| azc_full_dtt_seed42 | $T/runs/azc_full_dtt_seed42/metrics.jsonl | b76d6adaf9d46358691d6247ac288b17 | none (not yet scored) |
| azc_full_dtt_seed43 | $T/runs/azc_full_dtt_seed43/metrics.jsonl | 085c62bae777605c7d407a1c7ee8f3af | none (not yet scored) |
| r090_af_isla_seed42 | $T/runs/r090_af_isla_seed42/metrics.jsonl | 20449b5275415648a54f676cc933ffe6 | $T/evals_cmpaf_eager/r090_af_isla_seed42/native/r090_af_isla_seed42/metrics.jsonl |
| r090_af_isla_seed43 | $T/runs/r090_af_isla_seed43/metrics.jsonl | 949d39b3fc0a546058962df6fac216a0 | $T/evals_cmpaf_eager/r090_af_isla_seed43/native/r090_af_isla_seed43/metrics.jsonl |
| r090_af_dtt_seed42 | $T/runs/r090_af_dtt_seed42/metrics.jsonl | 8a5264a34c9c104f78823194b79cd34c | $T/evals_cmpaf_eager/r090_af_dtt_seed42/native/r090_af_dtt_seed42/metrics.jsonl |
| r090_af_dtt_seed43 | $T/runs/r090_af_dtt_seed43/metrics.jsonl | 3d734a6d9f83c46f213519cb4ff1fa41 | $T/evals_cmpaf_eager/r090_af_dtt_seed43/native/r090_af_dtt_seed43/metrics.jsonl |
| r090_lug_isla_seed42 | $T/runs/r090_lug_isla_seed42/metrics.jsonl | 931b5f7a785de3d97758eca24e7313c1 | $T/evals_cmpaf_eager/r090_lug_isla_seed42/native/r090_lug_isla_seed42/metrics.jsonl |
| r090_lug_isla_seed43 | $T/runs/r090_lug_isla_seed43/metrics.jsonl | d2deee4c958c6bfe2aeab1f7f4c673d9 | $T/evals_cmpaf_eager/r090_lug_isla_seed43/native/r090_lug_isla_seed43/metrics.jsonl |
| r090_lug_dtt_seed42 | $T/runs/r090_lug_dtt_seed42/metrics.jsonl | 3337a44dc6e162e2a4c3060319d0a32b | none (not yet scored) |
| r090_lug_dtt_seed43 | $T/runs/r090_lug_dtt_seed43/metrics.jsonl | 969336629539f91ec7dd3b1f5a2b59a7 | none (not yet scored) |
| r090_t2dz_isla_seed42 | $T/runs/r090_t2dz_isla_seed42/metrics.jsonl | cfc4acf1935dc47a37248a7d95d683af | $T/evals_cmpaf_eager/r090_t2dz_isla_seed42/native/r090_t2dz_isla_seed42/metrics.jsonl |
| r090_t2dz_isla_seed43 | $T/runs/r090_t2dz_isla_seed43/metrics.jsonl | 2aae1e648137fae6f52d632a49eaa781 | $T/evals_cmpaf_eager/r090_t2dz_isla_seed43/native/r090_t2dz_isla_seed43/metrics.jsonl |
| r80ud_dr_gt_h256_seed42 | $T/runs/r80ud_dr_gt_h256_seed42/metrics.jsonl | 588b1e0767bceab981fd1cc49dbef99a | $T/evals_ud2/r80ud_dr_gt_h256_seed42/native/r80ud_dr_gt_h256_seed42/metrics.jsonl |
| r80ud_dr_gt_h256_seed43 | $T/runs/r80ud_dr_gt_h256_seed43/metrics.jsonl | a222bed44ea0ad4a84c8f0e48608caa5 | $T/evals_ud2/r80ud_dr_gt_h256_seed43/native/r80ud_dr_gt_h256_seed43/metrics.jsonl |
| r80ud_dr_t_h512_l16_c100k_seed42 | $T/runs/r80ud_dr_t_h512_l16_c100k_seed42/metrics.jsonl | ff3aeabdf32159b99e055c5d2a208ed0 | $T/evals_r80ud/r80ud_dr_t_h512_l16_c100k_seed42/native/r80ud_dr_t_h512_l16_c100k_seed42/metrics.jsonl |
| r80ud_dr_t_h512_l16_c100k_seed43 | $T/runs/r80ud_dr_t_h512_l16_c100k_seed43/metrics.jsonl | c7912fa2d948a221cc6f17e4ee613f21 | $T/evals_r80ud/r80ud_dr_t_h512_l16_c100k_seed43/native/r80ud_dr_t_h512_l16_c100k_seed43/metrics.jsonl |
| clud_vol_gt_seed42 | $T/runs/clud_vol_gt_seed42/metrics.jsonl | 41f45b6ffc699f3c8ac589c4489563ae | $T/evals_ud2/clud_vol_gt_seed42/native/clud_vol_gt_seed42/metrics.jsonl |
| clud_vol_gt_seed43 | $T/runs/clud_vol_gt_seed43/metrics.jsonl | eedd965515364f2ec12fae35e0f621ff | $T/evals_ud2/clud_vol_gt_seed43/native/clud_vol_gt_seed43/metrics.jsonl |
| clud_vol_t_seed42 | $T/runs/clud_vol_t_seed42/metrics.jsonl | df0c1aba192dba81541bd6076e00c73c | $T/evals_ud2/clud_vol_t_seed42/native/clud_vol_t_seed42/metrics.jsonl |
| clud_vol_t_seed43 | $T/runs/clud_vol_t_seed43/metrics.jsonl | 52adcb29c011a2dab1a71e84b957a11c | $T/evals_ud2/clud_vol_t_seed43/native/clud_vol_t_seed43/metrics.jsonl |
| clud_suv_gt_seed42 | $T/runs/clud_suv_gt_seed42/metrics.jsonl | 30191579cf6c5c112f6226b7aeffd6f3 | $T/evals_ud2/clud_suv_gt_seed42/native/clud_suv_gt_seed42/metrics.jsonl |
| clud_suv_gt_seed43 | $T/runs/clud_suv_gt_seed43/metrics.jsonl | 12cb6ff81dc2ddf95419c64f65df9529 | $T/evals_ud2/clud_suv_gt_seed43/native/clud_suv_gt_seed43/metrics.jsonl |
| clud_suv_t_seed42 | $T/runs/clud_suv_t_seed42/metrics.jsonl | 2ab227403a99673fc42ffc21e94aa20f | $T/evals_ud2/clud_suv_t_seed42/native/clud_suv_t_seed42/metrics.jsonl |
| clud_suv_t_seed43 | $T/runs/clud_suv_t_seed43/metrics.jsonl | d0b11741c3a26deba4170d3d5be46d55 | $T/evals_ud2/clud_suv_t_seed43/native/clud_suv_t_seed43/metrics.jsonl |
| udrv_hl_gt_seed42 | $T/runs/udrv_hl_gt_seed42/metrics.jsonl | 97d9c950ebfb5a5d5a873897c7bd980a | $T/evals_udrv/udrv_hl_gt_seed42/native/udrv_hl_gt_seed42/metrics.jsonl |
| udrv_hl_gt_seed43 | $T/runs/udrv_hl_gt_seed43/metrics.jsonl | ca51ec6625f1bc455284291bbc9fc303 | $T/evals_udrv/udrv_hl_gt_seed43/native/udrv_hl_gt_seed43/metrics.jsonl |
| udrv_hl_t_seed42 | $T/runs/udrv_hl_t_seed42/metrics.jsonl | ab6815aaba7164aa09ba106e618189b9 | $T/evals_udrv/udrv_hl_t_seed42/native/udrv_hl_t_seed42/metrics.jsonl |
| udrv_hl_t_seed43 | $T/runs/udrv_hl_t_seed43/metrics.jsonl | 38f1847380336ec5ec374599a6f72eaf | $T/evals_udrv/udrv_hl_t_seed43/native/udrv_hl_t_seed43/metrics.jsonl |
| afu_ud_gt_seed42 | $T/runs/afu_ud_gt_seed42/metrics.jsonl | 643dd9809a810a43e59ec2967e3dbc4b | $T/evals_afplanar/afu_ud_gt_seed42/native/afu_ud_gt_seed42/metrics.jsonl |
| afu_ud_gt_seed43 | $T/runs/afu_ud_gt_seed43/metrics.jsonl | bfaafb33011ed06cb2287627dd6ba9f9 | $T/evals_afplanar/afu_ud_gt_seed43/native/afu_ud_gt_seed43/metrics.jsonl |
| afu_ud_t_seed42 | $T/runs/afu_ud_t_seed42/metrics.jsonl | daebd1cf81121ac88c2be24839b923ab | $T/evals_afplanar/afu_ud_t_seed42/native/afu_ud_t_seed42/metrics.jsonl |
| afu_ud_t_seed43 | $T/runs/afu_ud_t_seed43/metrics.jsonl | 04b8577e4826708e4be44a28c0efd4f9 | $T/evals_afplanar/afu_ud_t_seed43/native/afu_ud_t_seed43/metrics.jsonl |
| lugbc_gt_seed42 | $T/runs/lugbc_gt_seed42/metrics.jsonl | b8ba2bae02aca9f3033f2ecbfd834618 | $T/evals_lugbc/lugbc_gt_seed42/native/lugbc_gt_seed42/metrics.jsonl |
| lugbc_gt_seed43 | $T/runs/lugbc_gt_seed43/metrics.jsonl | dc40b687d12cb86901f3081da3c88518 | $T/evals_lugbc/lugbc_gt_seed43/native/lugbc_gt_seed43/metrics.jsonl |
| azc_full_gt_seed42 | $T/runs/azc_full_gt_seed42/metrics.jsonl | bd0f790ad4fcf391a65df21510373f08 | $T/p200k_fp32_test_azc/azc_full_gt_seed42/azc_full_gt_seed42/metrics.jsonl |
| azc_full_gt_seed43 | $T/runs/azc_full_gt_seed43/metrics.jsonl | f268bfed97ff67f63041d557e386c077 | $T/p200k_fp32_test_azc/azc_full_gt_seed43/azc_full_gt_seed43/metrics.jsonl |
| azc_full_t_seed42 | $T/runs/azc_full_t_seed42/metrics.jsonl | c41776a1e6670bc98751463261c5e0ed | $T/p200k_fp32_test_azc/azc_full_t_seed42/azc_full_t_seed42/metrics.jsonl |
| azc_full_t_seed43 | $T/runs/azc_full_t_seed43/metrics.jsonl | 2628b54b2d0e3b45111c9e81f3f10912 | $T/p200k_fp32_test_azc/azc_full_t_seed43/azc_full_t_seed43/metrics.jsonl |
| r090fix_t2dz_isla_seed42 | $T/runs/r090fix_t2dz_isla_seed42/metrics.jsonl | f60bbdda2d6bf9e8cd5cc6f5884e351d | $T/evals_r090fix/r090fix_t2dz_isla_seed42/native/r090fix_t2dz_isla_seed42/metrics.jsonl |
| r090fix_t2dz_isla_seed43 | $T/runs/r090fix_t2dz_isla_seed43/metrics.jsonl | e9ae56069f34853c595d02c089442a5b | $T/evals_r090fix/r090fix_t2dz_isla_seed43/native/r090fix_t2dz_isla_seed43/metrics.jsonl |
| r090fix_t2dz_isla_seed44 | $T/runs/r090fix_t2dz_isla_seed44/metrics.jsonl | 497a50f8ef0e0aad4341821af70c2b81 | $T/evals_r090fix/r090fix_t2dz_isla_seed44/native/r090fix_t2dz_isla_seed44/metrics.jsonl |
| r090fix_t2dz_isla_seed45 | $T/runs/r090fix_t2dz_isla_seed45/metrics.jsonl | c7f723d675ba65817c2e4f9260bf447c | $T/evals_r090fix/r090fix_t2dz_isla_seed45/native/r090fix_t2dz_isla_seed45/metrics.jsonl |

Five standing baseline lanes have no training log left on LHR (`$T/runs/cl_r_{t2dz_gt,t2dz_t,jeb_gt,jeb_t,lug_t}_seed{42,43}`);
the script prices them from the one-epoch smoke lanes already in the repository, `results/campaigns_2026-10-02/char/cs/cs_r_<task>_<arch>/metrics.jsonl`.
