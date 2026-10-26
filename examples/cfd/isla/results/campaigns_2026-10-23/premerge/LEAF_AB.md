# Loader-leaf A/B: psharpe/isla 9f279664c (A, tensordict 0.15.2) vs psharpe/isla-premerge-1026 (B, tensordict 0.15.3)

Harness: lhr/abp.py --mode hash (as PERF-RECIPE's pr.py --mode hash: main's start, build_dataloaders, set_epoch(0), 3 train + 2 val batches through load()), lhr/abcmp.py; every leaf's dtype, shape and blake2b.
GB300: job 395320 (B = 8112112f2 code + recipe), fix rerun 396490 (B = 737f1c48c recipe); CPU: 394745/394746 (B = cb275c637). Configs: lhr/leafcfgs.tsv (md5 in FINAL_REPORT).

| config | overrides | GB300 leaves equal/compared (train+val) | GB300 verdict | after the fix (737f1c48c) | CPU verdict |
|---|---|---|---|---|---|
| i_drv | isla_volume / drivaer_ml_volume_reference sampling_resolution=200000 | 170/170 | BITWISE |  | BITWISE |
| i_drv_posed | isla_volume / drivaer_ml_volume_reference_posed sampling_resolution=200000 | 175/175 | BITWISE |  | BITWISE |
| i_drv_area | isla_volume / drivaer_ml_volume_reference_area sampling_resolution=200000 | 170/170 | BITWISE |  | BITWISE |
| i_az_full | isla_volume_aerozoo / aerozoo_uvolume sampling_resolution=200000 train_split=full.train val_split=full.test | 706/706 | BITWISE |  | BITWISE |
| i_az_bank_zero_to_cad | isla_volume_aerozoo / aerozoo_uvolume_bank_zero_to_cad sampling_resolution=200000 train_split=bank_zero_to_cad.train val_split=bank_zero_to_cad.test | 702/702 | BITWISE |  | BITWISE |
| i_az_bank_blobs | isla_volume_aerozoo / aerozoo_uvolume_blobs sampling_resolution=200000 train_split=bank_blobs.train val_split=bank_blobs.test | 700/700 | BITWISE |  | BITWISE |
| i_az_family_non_box | isla_volume_aerozoo / aerozoo_uvolume_nonbox sampling_resolution=200000 train_split=family_non_box.train val_split=family_non_box.test | 702/702 | BITWISE |  | BITWISE |
| i_az_profile_crossflow | isla_volume_aerozoo / aerozoo_uvolume_profile_crossflow sampling_resolution=200000 train_split=profile_crossflow.train val_split=profile_crossflow.test | 699/699 | BITWISE |  | BITWISE |
| i_az_profile_dual_inlet | isla_volume_aerozoo / aerozoo_uvolume_profile_dual_inlet sampling_resolution=200000 train_split=profile_dual_inlet.train val_split=profile_dual_inlet.test | 699/699 | BITWISE |  | BITWISE |
| i_az_reynolds | isla_volume_aerozoo / aerozoo_uvolume_reynolds sampling_resolution=200000 train_split=reynolds.train val_split=reynolds.test | 702/702 | BITWISE |  | BITWISE |
| i_af | isla_volume_airfrans / airfrans_uvolume_2d sampling_resolution=100000 train_split=full_train val_split=full_test | 165/165 | BITWISE |  | BITWISE |
| i_af_posed | isla_volume_airfrans / airfrans_uvolume_2d_posed sampling_resolution=100000 train_split=full_train val_split=full_test | 170/170 | BITWISE |  | BITWISE |
| i_af_area | isla_volume_airfrans / airfrans_uvolume_2d_area sampling_resolution=100000 train_split=full_train val_split=full_test | 165/165 | BITWISE |  | BITWISE |
| i_lug | isla_lug / ginot_lug_volume sampling_resolution=20000 train_split=train val_split=test | 210/210 | BITWISE |  | BITWISE |
| i_lug_posed | isla_lug / ginot_lug_volume_posed sampling_resolution=20000 train_split=train val_split=test | 215/215 | BITWISE |  | BITWISE |
| i_lug_area | isla_lug / ginot_lug_volume_area sampling_resolution=20000 train_split=train val_split=test | 210/210 | BITWISE |  | BITWISE |
| i_ten | isla_tensile2d / plaid_tensile2d_2d sampling_resolution=20000 train_split=train_400 val_split=val_100 | 350/350 | BITWISE |  | BITWISE |
| i_ten_posed | isla_tensile2d / plaid_tensile2d_2d_posed sampling_resolution=20000 train_split=train_400 val_split=val_100 | 355/355 | BITWISE |  | BITWISE |
| i_ten_area | isla_tensile2d / plaid_tensile2d_2d_area sampling_resolution=20000 train_split=train_400 val_split=val_100 | 350/350 | BITWISE |  | BITWISE |
| i_drs | isla_surface / drivaer_ml_surface sampling_resolution=200000 | 145/145 | BITWISE |  | BITWISE |
| i_drs_posed | isla_surface / drivaer_ml_surface_posed sampling_resolution=200000 | 150/150 | BITWISE |  | BITWISE |
| i_drs_area | isla_surface / drivaer_ml_surface_area sampling_resolution=200000 | 0/0 | STRUCT (fails to compose on both sides) |  | STRUCT |
| i_suv | isla_surface_shift_suv / shift_suv_surface sampling_resolution=200000 | 145/145 | BITWISE |  | BITWISE |
| i_suv_posed | isla_surface_shift_suv / shift_suv_surface_posed sampling_resolution=200000 | 150/150 | BITWISE |  | BITWISE |
| i_suv_area | isla_surface_shift_suv / shift_suv_surface_area sampling_resolution=200000 | 110/145 | STRUCT (moved: leaves.train batch 0) | BITWISE 145/145 | BITWISE |
| i_hl_val | isla_surface_highlift / highlift_surface sampling_resolution=200000 train_split=medium_train val_split=medium_val | 140/140 | BITWISE |  | BITWISE |
| i_djb | isla_surface_deepjebpp / deepjebpp_surface sampling_resolution=200000 train_split=train val_split=test | 415/415 | BITWISE |  | BITWISE |
| i_djb_posed | isla_surface_deepjebpp / deepjebpp_surface_posed sampling_resolution=200000 train_split=train val_split=test | 435/435 | BITWISE |  | BITWISE |
| i_djb_area | isla_surface_deepjebpp / deepjebpp_surface_area sampling_resolution=200000 train_split=train val_split=test | 415/415 | BITWISE |  | BITWISE |
| b_gt_drvref | geotransolver_volume / drivaer_ml_volume_reference sampling_resolution=200000 | 0/0 | STRUCT (fails to compose on both sides) |  | STRUCT |
| b_t_drvref | transolver_volume / drivaer_ml_volume_reference sampling_resolution=100000 | 0/0 | STRUCT (fails to compose on both sides) |  | STRUCT |
| b_gt_drv | geotransolver_volume / drivaer_ml_volume sampling_resolution=200000 | 35/35 | BITWISE |  | BITWISE |
| b_t_drv | transolver_volume / drivaer_ml_volume sampling_resolution=100000 | 25/25 | BITWISE |  | BITWISE |
| b_gt_az_full | geotransolver_volume_aerozoo / aerozoo_uvolume sampling_resolution=200000 train_split=full.train val_split=full.test | 40/40 | BITWISE |  | BITWISE |
| b_gt_az_bank_zero_to_cad | geotransolver_volume_aerozoo / aerozoo_uvolume_bank_zero_to_cad sampling_resolution=200000 train_split=bank_zero_to_cad.train val_split=bank_zero_to_cad.test | 40/40 | BITWISE |  | BITWISE |
| b_gt_az_bank_blobs | geotransolver_volume_aerozoo / aerozoo_uvolume_blobs sampling_resolution=200000 train_split=bank_blobs.train val_split=bank_blobs.test | 40/40 | BITWISE |  | BITWISE |
| b_gt_az_family_non_box | geotransolver_volume_aerozoo / aerozoo_uvolume_nonbox sampling_resolution=200000 train_split=family_non_box.train val_split=family_non_box.test | 40/40 | BITWISE |  | BITWISE |
| b_gt_az_profile_crossflow | geotransolver_volume_aerozoo / aerozoo_uvolume_profile_crossflow sampling_resolution=200000 train_split=profile_crossflow.train val_split=profile_crossflow.test | 40/40 | BITWISE |  | BITWISE |
| b_gt_az_profile_dual_inlet | geotransolver_volume_aerozoo / aerozoo_uvolume_profile_dual_inlet sampling_resolution=200000 train_split=profile_dual_inlet.train val_split=profile_dual_inlet.test | 40/40 | BITWISE |  | BITWISE |
| b_gt_az_reynolds | geotransolver_volume_aerozoo / aerozoo_uvolume_reynolds sampling_resolution=200000 train_split=reynolds.train val_split=reynolds.test | 40/40 | BITWISE |  | BITWISE |
| b_t_az_full | transolver_volume_aerozoo / aerozoo_uvolume sampling_resolution=200000 train_split=full.train val_split=full.test | 35/35 | BITWISE |  | BITWISE |
| b_gt_af | geotransolver_volume_airfrans / airfrans_uvolume_ud sampling_resolution=100000 train_split=full_train val_split=full_test | 35/35 | BITWISE |  | BITWISE |
| b_t_af | transolver_volume / airfrans_uvolume_ud sampling_resolution=100000 train_split=full_train val_split=full_test | 25/25 | BITWISE |  | BITWISE |
| b_gt_af_posed | geotransolver_volume_airfrans / airfrans_uvolume_ud_posed sampling_resolution=100000 train_split=full_train val_split=full_test | 35/35 | BITWISE |  | BITWISE |
| b_t_af_posed | transolver_volume / airfrans_uvolume_ud_posed sampling_resolution=100000 train_split=full_train val_split=full_test | 25/25 | BITWISE |  | BITWISE |
| b_gt_af_area | geotransolver_volume_airfrans / airfrans_uvolume_ud_area sampling_resolution=100000 train_split=full_train val_split=full_test | 35/35 | BITWISE |  | BITWISE |
| b_t_af_area | transolver_volume / airfrans_uvolume_ud_area sampling_resolution=100000 train_split=full_train val_split=full_test | 25/25 | BITWISE |  | BITWISE |
| b_globe_af | globe_volume_airfrans / airfrans_uvolume_2d sampling_resolution=4096 train_split=full_train val_split=full_test | 85/85 | BITWISE |  | BITWISE |
| b_gt_drs | geotransolver_surface / drivaer_ml_surface sampling_resolution=200000 | -/- | not compared (A port clash; see bisect 395965: BITWISE) |  | BITWISE |
| b_t_drs | transolver_surface / drivaer_ml_surface sampling_resolution=100000 | 25/25 | BITWISE |  | BITWISE |
| b_gt_drs_posed | geotransolver_surface / drivaer_ml_surface_posed sampling_resolution=200000 | 35/35 | BITWISE |  | BITWISE |
| b_t_drs_posed | transolver_surface / drivaer_ml_surface_posed sampling_resolution=100000 | 25/25 | BITWISE |  | BITWISE |
| b_gt_drs_area | geotransolver_surface / drivaer_ml_surface_area sampling_resolution=200000 | 5/35 | STRUCT (moved: leaves.train batch 0) | BITWISE 35/35 | BITWISE |
| b_t_drs_area | transolver_surface / drivaer_ml_surface_area sampling_resolution=100000 | 0/25 | STRUCT (moved: leaves.train batch 0) | BITWISE 25/25 | BITWISE |
| b_globe_drs | globe_surface / drivaer_ml_surface sampling_resolution=200000 | 60/60 | BITWISE |  | BITWISE |
| b_globedrv_drs | globe_surface_drivaerml / drivaer_ml_surface sampling_resolution=80000 | 95/95 | BITWISE |  | BITWISE |
| b_gt_suv | geotransolver_surface / shift_suv_surface sampling_resolution=200000 | 35/35 | BITWISE |  | BITWISE |
| b_t_suv | transolver_surface / shift_suv_surface sampling_resolution=100000 | 25/25 | BITWISE |  | BITWISE |
| b_gt_suv_posed | geotransolver_surface / shift_suv_surface_posed sampling_resolution=200000 | 35/35 | BITWISE |  | BITWISE |
| b_t_suv_posed | transolver_surface / shift_suv_surface_posed sampling_resolution=100000 | 25/25 | BITWISE |  | BITWISE |
| b_gt_suv_area | geotransolver_surface / shift_suv_surface_area sampling_resolution=200000 | 5/35 | STRUCT (moved: leaves.train batch 0) | BITWISE 35/35 | BITWISE |
| b_t_suv_area | transolver_surface / shift_suv_surface_area sampling_resolution=100000 | 0/25 | STRUCT (moved: leaves.train batch 0) | BITWISE 25/25 | BITWISE |
| b_gt_hl | geotransolver_surface / highlift_surface_medium_dedup sampling_resolution=200000 | 50/50 | BITWISE |  | BITWISE |
| b_t_hl | transolver_surface / highlift_surface_medium_dedup sampling_resolution=100000 | 40/40 | BITWISE |  | BITWISE |
| b_gt_hl_posed | geotransolver_surface / highlift_surface_medium_dedup_posed sampling_resolution=200000 | 50/50 | BITWISE |  | BITWISE |
| b_t_hl_posed | transolver_surface / highlift_surface_medium_dedup_posed sampling_resolution=100000 | 40/40 | BITWISE |  | BITWISE |
| b_gt_lug | geotransolver_lug / ginot_lug_volume sampling_resolution=20000 train_split=train val_split=test ++dataset_reader_overrides.drop_in_file_boundaries=false | 25/25 | BITWISE |  | BITWISE |
| b_t_lug | transolver_lug / ginot_lug_volume sampling_resolution=20000 train_split=train val_split=test | 15/15 | BITWISE |  | BITWISE |
| b_gt_ten | geotransolver_tensile2d / plaid_tensile2d_2d sampling_resolution=20000 train_split=train_400 val_split=val_100 | 35/35 | BITWISE |  | BITWISE |
| b_t_ten | transolver_tensile2d / plaid_tensile2d_2d sampling_resolution=20000 train_split=train_400 val_split=val_100 | 25/25 | BITWISE |  | BITWISE |
| b_gt_djb | geotransolver_surface_deepjebpp / deepjebpp_surface sampling_resolution=200000 train_split=train val_split=test | 35/35 | BITWISE |  | BITWISE |
| b_t_djb | transolver_surface_deepjebpp / deepjebpp_surface sampling_resolution=200000 train_split=train val_split=test | 25/25 | BITWISE |  | BITWISE |
| i_suvxb | isla_surface_shift_suv / shift_suv_surface_fastback sampling_resolution=200000 | 0/0 | STRUCT (fails to compose on both sides) |  | STRUCT |
| b_gt_suvxb | geotransolver_surface / shift_suv_surface_fastback sampling_resolution=200000 | 35/35 | BITWISE |  | BITWISE |
| b_t_suvxb | transolver_surface / shift_suv_surface_fastback sampling_resolution=100000 | 25/25 | BITWISE |  | BITWISE |
| b_gt_hl_area | geotransolver_surface / highlift_surface_medium_area sampling_resolution=200000 | 5/50 | STRUCT (moved: leaves.train batch 0) | BITWISE 50/50 | BITWISE |
| b_t_hl_area | transolver_surface / highlift_surface_medium_area sampling_resolution=100000 | 0/40 | STRUCT (moved: leaves.train batch 0) | BITWISE 40/40 | BITWISE |

Final tree (737f1c48c recipe), GB300: 11365 of 11365 leaves bitwise over the configs that compose (b_gt_drs from bisect 395965 not counted).
