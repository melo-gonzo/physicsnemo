## Host timeline per warm step (ms; timed phase, mean over steps and ranks)

| part | ISLA_dtd | DTT_dtd | ISLA_geo2 |
|---|---|---|---|
| step (node) | 744.8 | 1416.8 | 590.0 |
| data: next(): lookahead consume (H2D + transforms, main thread) | 48.8 | 48.4 | 41.5 |
| data: _fail_together collective (GPU drain + rank skew) | 21.9 | 15.3 | 20.1 |
| forward: host | 76.7 | 502.5 | 78.0 |
| forward: _fail_together collective (forward GPU tail + rank skew) | 155.5 | 112.1 | 85.8 |
| backward: host (zero_grad + backward()) | 230.1 | 624.1 | 206.7 |
| prefetch of the next batch (RS_EARLY_NEXT) | 0.0 | 0.0 | 0.0 |
| grad-norm isfinite (backward GPU tail + DDP all-reduce + rank waits) | 181.4 | 80.8 | 129.2 |
| optimizer: host (Muon + AdamW) | 14.6 | 22.0 | 14.2 |
| loss reduce + logging | 10.5 | 9.6 | 11.3 |

## Module attribution (sync phase: synchronized at every region boundary; ms per step)

| region | ISLA_dtd | DTT_dtd | ISLA_geo2 |
|---|---|---|---|
| data.consume(h2d+transforms) | 57.8 | 53.5 | 47.8 |
| data.wait_reader | 1.0 | 0.0 | 38.6 |
| fwd.blocks | 146.9 | 136.7 | 127.4 |
| fwd.exact_boundary_geometry | nan | nan | 20.0 |
| fwd.model_forward(total) | 228.2 | 606.0 | 177.8 |
| fwd.nearest_boundary_geometry | 59.3 | 59.5 | nan |
| fwd.read._TreeKernelRead | nan | 91.8 | nan |
| fwd.rms_pairwise_distance | 0.4 | 0.4 | 0.3 |
| fwd.sample_boundary_cells | nan | nan | 7.6 |
| fwd.tree(total) | nan | 290.9 | nan |
| fwd.tree.basis | nan | 35.1 | nan |
| fwd.tree.forest_build | nan | 148.4 | nan |
| fwd.tree.forest_build.from_points | nan | 124.1 | nan |
| fwd.tree.pair_plan | nan | 5.9 | nan |
| fwd.tree.walk | nan | 99.0 | nan |
| bwd_host_ms | 245.4 | 664.4 | 223.4 |
| bwd_gpu_drain_ms | 182.4 | 94.4 | 148.1 |
| opt_host_ms | 15.8 | 23.6 | 15.6 |

## GPU per step (torch.profiler, prof phase; mean over ranks; launches and syncs counted on the host)

| quantity | ISLA_dtd | DTT_dtd | ISLA_geo2 |
|---|---|---|---|
| step wall under the profiler | 951.5 | 1801.9 | 1067.8 |
| GPU busy (union of streams) | 765.9 | 1194.3 | 802.6 |
| GPU busy fraction % | 80.5 | 66.3 | 75.2 |
| data: kernels ms (excl. NCCL) | 14.4 | 14.3 | 3.6 |
| fwd: kernels ms (excl. NCCL) | 189.1 | 342.9 | 135.4 |
| bwd: kernels ms (excl. NCCL) | 375.8 | 650.6 | 339.1 |
| opt: kernels ms (excl. NCCL) | 3.1 | 3.8 | 3.1 |
| log: kernels ms (excl. NCCL) | 0.1 | 0.1 | 0.1 |
| bwd: NCCL kernel ms | 189.8 | 275.7 | 49.8 |
| coll_data: NCCL kernel ms | 98.4 | 49.6 | 140.2 |
| coll_fwd: NCCL kernel ms | 34.7 | 69.6 | 171.1 |
| log: NCCL kernel ms | 19.5 | 11.9 | 8.3 |
| early_next: NCCL kernel ms | 0.0 | 0.0 | 0.0 |
| data: launches / launch CPU ms | 2304 / 21.2 | 2304 / 20.3 | 1833 / 15.0 |
| fwd: launches / launch CPU ms | 1276 / 14.3 | 14899 / 169.3 | 2192 / 23.5 |
| bwd: launches / launch CPU ms | 2269 / 135.1 | 10767 / 91.7 | 2286 / 114.5 |
| opt: launches / launch CPU ms | 461 / 5.4 | 752 / 8.2 | 461 / 5.1 |
| data: host syncs / ms waited | 9 / 0.1 | 9 / 0.1 | 8 / 0.1 |
| fwd: host syncs / ms waited | 0 / 0.0 | 341 / 55.6 | 2 / 4.6 |
| bwd: host syncs / ms waited | 0 / 0.0 | 12 / 285.9 | 0 / 0.0 |
| opt: host syncs / ms waited | 0 / 0.0 | 0 / 0.0 | 0 / 0.0 |
| gradnorm: host syncs / ms waited | 1 / 172.8 | 1 / 56.7 | 1 / 120.8 |
| coll_fwd: host syncs / ms waited | 2 / 144.0 | 2 / 111.6 | 2 / 211.6 |
| GPU idle while host in data | 86.8 | 88.2 | 114.6 |
| GPU idle while host in coll_data | 1.3 | 1.1 | 1.0 |
| GPU idle while host in fwd | 11.8 | 416.6 | 95.3 |
| GPU idle while host in coll_fwd | 13.1 | 2.0 | 1.4 |
| GPU idle while host in bwd | 4.9 | 44.3 | 13.7 |
| GPU idle while host in early_next | 0.0 | 0.0 | 0.0 |
| GPU idle while host in gradnorm | 6.8 | 0.8 | 2.9 |
| GPU idle while host in opt | 26.7 | 38.6 | 22.4 |
| GPU idle while host in log | 32.8 | 15.6 | 13.6 |
| GPU idle while host in other | 1.5 | 0.4 | 0.4 |
| H2D (pinned) MB / ms per step | 50 / 0.48 | 50 / 0.48 | 50 / 0.48 |
| peak allocated / reserved GiB | 127 / 130 | 162 / 165 | 127 / 130 |
| boundary cells per rank: mean / max over ranks (mean) | 166032 / 189300 | 166032 / 189300 | 415282 / 613856 |

### Top kernels per step (rank 0)

- ISLA_dtd fwd: 55.3 _nearest_index_kernel; 20.6 _geo_fwd_kernel; 12.4 nvjet_sm103_tst_128x256_64x6_2x1_2cta_v_bz_bias_; 9.7 triton_poi_fused_gelu_view_34; 9.6 triton_red_fused__softmax__to_copy_add_bmm_mul_p; 7.7 triton_red_fused__softmax_add_prepare_softmax_on; 5.8 void cutlass::Kernel2<cutlass_80_tensorop_bf16_s; 5.4 triton_red_fused__softmax_add_prepare_softmax_on
- ISLA_dtd bwd: 128.2 ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevK; 92.8 _geo_bwd_kernel; 28.3 triton_red_fused__softmax__softmax_backward_data; 19.8 triton_red_fused__softmax__softmax_backward_data; 17.8 void cutlass::Kernel2<cutlass_80_tensorop_bf16_s; 17.6 void cutlass::Kernel2<cutlass_80_tensorop_bf16_s; 15.2 nvjet_sm103_tst_128x256_64x6_2x1_2cta_v_bz_NNT; 12.9 triton_poi_fused_gelu_gelu_backward_view_0
- DTT_dtd fwd: 55.1 _nearest_index_kernel; 30.1 _ssr_fwd_kernel; 20.6 _geo_fwd_kernel; 12.4 nvjet_sm103_tst_128x256_64x6_2x1_2cta_v_bz_bias_; 12.4 void at::native::index_elementwise_kernel<128, 4; 9.7 triton_poi_fused_gelu_view_34; 9.6 triton_red_fused__softmax__to_copy_add_bmm_mul_p; 9.4 nvjet_sm103_tst_8x384_64x4_1x2_h_bz_TNN
- DTT_dtd bwd: 221.7 ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevK; 109.6 _ssr_bwd_source_kernel; 91.9 _geo_bwd_kernel; 31.2 _ssr_bwd_point_kernel; 29.4 triton_red_fused__softmax__softmax_backward_data; 20.2 triton_red_fused__softmax__softmax_backward_data; 17.8 void cutlass::Kernel2<cutlass_80_tensorop_bf16_s; 17.5 void cutlass::Kernel2<cutlass_80_tensorop_bf16_s
- ISLA_geo2 fwd: 21.9 _geo_fwd_kernel; 10.1 _bvh_query_distance_a16068b8_cuda_kernel_forward; 10.0 triton_poi_fused_gelu_view_34; 8.7 nvjet_sm103_tst_128x256_64x6_2x2_2cta_h_bz_bias_; 7.2 nvjet_sm103_tst_128x256_64x6_2x1_2cta_v_bz_TNT; 7.1 triton_red_fused__softmax__to_copy_add_bmm_mul_p; 6.0 void cutlass::Kernel2<cutlass_75_tensorop_s1688g; 5.9 nvjet_sm103_tst_128x256_64x6_2x1_2cta_v_bz_bias_
- ISLA_geo2 bwd: 96.9 _geo_bwd_kernel; 37.9 triton_red_fused__softmax__softmax_backward_data; 32.8 triton_red_fused__softmax__softmax_backward_data; 14.9 void at::native::elementwise_kernel<128, 4, at::; 13.5 triton_poi_fused_gelu_gelu_backward_view_0; 11.3 nvjet_sm103_tst_128x256_64x6_2x1_2cta_v_bz_NNT; 10.4 void at::native::reduce_kernel<128, 4, at::nativ; 8.6 triton_red_fused__softmax__softmax_backward_data
