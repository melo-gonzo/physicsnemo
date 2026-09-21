"""Generate the lane tables for the 2026-09-21 large-scale campaigns (Peter: ~1,400 idle AGA nodes).

sl  SCALE-LAW: error vs (training cases, parameters) for cosine ISLA and GeoTransolver on two datasets at 100,000 cells.
sd  SEEDS:     third and fourth seeds (44, 45) of every headline reporting-protocol lane on both datasets.
ct  COSINE-TWINS: cosine ISLA-192 at 1e-3 twins of the linear ISLA evidence lanes at the reporting protocol.

Every lane: 500 epochs, bf16, no augmentation, one node (4 GPUs), code snapshot code_routenorm (the class with routing_norm;
GeoTransolver and Transolver are unaffected by the snapshot choice). Writes <out_dir>/<campaign>/lanes.tsv.
"""

import csv, sys
from pathlib import Path

out = Path(sys.argv[1])
COS = "+model.geo_kernel=fused +model.routing_norm=cosine"
GT = "forward_kwargs.global_embedding=global_data.U_inf_dir"
TR = "forward_kwargs.fx.source=global_data.U_inf_dir"
HL = "model.out_scalars=3 model.out_vectors=2 +training.val_interval=25"
HLB = "+training.val_interval=25"  # baselines on the wings
ISLA_W = {
    192: "1e-3",
    384: "5e-4",
    768: "5e-4",
}  # cosine, best clean rate per width (ROUTE-NORM cars)
GT_W = {
    256: "3e-3",
    512: "3e-3",
    1024: "1e-3",
}  # GeoTransolver, best rate per width (RATE-GT, CAP-435)
SEEDS2 = (42, 43)
SEEDS_EXTRA = (44, 45)


def lane(group, run, dataset, model, extra, lr, seed, cells, code="code_routenorm"):
    return dict(
        group=group,
        run_id=run,
        dataset=dataset,
        model=model,
        extra=extra or "-",
        lr=lr,
        seed=seed,
        cells=cells,
        code=code,
    )


# ---------------- sl: scaling law ----------------
sl = []
for n in (27, 54, 109, 218):
    for w, lr in ISLA_W.items():
        for s in SEEDS2:
            sl.append(
                lane(
                    "sl",
                    f"sl_dr_isla_cos_w{w}_n{n}_c100k_seed{s}",
                    "drivaer_scaling",
                    "isla_surface_reference",
                    f"{COS} model.hidden={w} train_split=train_n{n}",
                    lr,
                    s,
                    100000,
                )
            )
    for w, lr in GT_W.items():
        for s in SEEDS2:
            sl.append(
                lane(
                    "sl",
                    f"sl_dr_gt_w{w}_n{n}_c100k_seed{s}",
                    "drivaer_scaling",
                    "geotransolver_surface",
                    f"{GT} model.n_hidden={w} train_split=train_n{n}",
                    lr,
                    s,
                    100000,
                )
            )
HL_RUNGS = {"super_scarce": 35, "scarce": 105, "medium": 510}
for rung, n in HL_RUNGS.items():
    for w, lr in ISLA_W.items():
        for s in SEEDS2:
            sl.append(
                lane(
                    "sl",
                    f"sl_hl_isla_cos_w{w}_{rung}_c100k_seed{s}",
                    f"highlift_surface_{rung}",
                    "isla_surface_reference",
                    f"{COS} {HL} model.hidden={w}",
                    lr,
                    s,
                    100000,
                )
            )
    for w, lr in GT_W.items():
        if rung == "medium" and w in (256, 512):
            continue  # SCALE-HL already trains GeoTransolver 256 and 512 on the medium rung at 100,000 cells
        for s in SEEDS2:
            sl.append(
                lane(
                    "sl",
                    f"sl_hl_gt_w{w}_{rung}_c100k_seed{s}",
                    f"highlift_surface_{rung}",
                    "geotransolver_surface",
                    f"{GT} {HLB} model.n_hidden={w}",
                    lr,
                    s,
                    100000,
                )
            )

# ---------------- sd: seeds 44 and 45 of the headline lanes ----------------
sd = []


def trio(
    prefix, dataset, isla_extra, gt_extra, t_extra, cells, isla_lr="1e-3", gt_lr="3e-3"
):
    for s in SEEDS_EXTRA:
        sd.append(
            lane(
                "sd",
                f"sd_{prefix}_isla_seed{s}",
                dataset,
                "isla_surface_reference",
                isla_extra,
                isla_lr,
                s,
                cells,
            )
        )
        sd.append(
            lane(
                "sd",
                f"sd_{prefix}_gt_unit_lr{gt_lr.replace('-', '')}_seed{s}",
                dataset,
                "geotransolver_surface",
                gt_extra,
                gt_lr,
                s,
                cells,
            )
        )
        sd.append(
            lane(
                "sd",
                f"sd_{prefix}_transolver_unit_seed{s}",
                dataset,
                "transolver_surface",
                t_extra,
                "3e-3",
                s,
                cells,
            )
        )


trio("dr435", "drivaer_ml_surface", "", GT, TR, 200000)  # parity trio
for n in (27, 54, 109, 218):  # DrivAerML ladder: ISLA and GeoTransolver 3e-3
    for s in SEEDS_EXTRA:
        sd.append(
            lane(
                "sd",
                f"sd_dr_isla_n{n}_seed{s}",
                "drivaer_scaling",
                "isla_surface_reference",
                f"train_split=train_n{n}",
                "1e-3",
                s,
                200000,
            )
        )
        sd.append(
            lane(
                "sd",
                f"sd_dr_gt_unit_lr3e3_n{n}_seed{s}",
                "drivaer_scaling",
                "geotransolver_surface",
                f"{GT} train_split=train_n{n}",
                "3e-3",
                s,
                200000,
            )
        )
trio(
    "dr_pose", "drivaer_ml_surface_pose", "", GT, TR, 200000, gt_lr="1e-3"
)  # pose trio (GeoTransolver at 1e-3 as in the fleet)
trio(
    "hl_super_scarce",
    "highlift_surface_super_scarce",
    HL,
    f"{GT} {HLB}",
    f"{TR} {HLB}",
    200000,
)
trio(
    "hl_single_aoa_12",
    "highlift_surface_single_aoa_12",
    HL,
    f"{GT} {HLB}",
    f"{TR} {HLB}",
    200000,
)
trio(
    "hl_super_scarce_area",
    "highlift_surface_super_scarce_area",
    HL,
    f"{GT} {HLB}",
    f"{TR} {HLB}",
    200000,
)  # TRAIN-AREA wings
trio(
    "dr_area", "drivaer_ml_surface_area", "+model.geo_kernel=fused", GT, TR, 100000
)  # TRAIN-AREA-DR (100k)
trio(
    "dr_c100k", "drivaer_ml_surface", "+model.geo_kernel=fused", GT, TR, 100000
)  # its uniform twins
for s in SEEDS_EXTRA:  # interior
    sd.append(
        lane(
            "sd",
            f"sd_dr_isla_vol_lr5e4_seed{s}",
            "drivaer_ml_volume_reference",
            "isla_volume_reference",
            "++model.query_mass=geometric_mean +training.val_interval=10",
            "5e-4",
            s,
            200000,
        )
    )
    sd.append(
        lane(
            "sd",
            f"sd_dr_gt_unit_vol_seed{s}",
            "drivaer_ml_volume_mt",
            "geotransolver_volume",
            f"{GT} +training.val_interval=10",
            "1e-3",
            s,
            200000,
        )
    )

# ---------------- ct: cosine twins of the ISLA evidence lanes at 200k ----------------
ct = []
for s in SEEDS2:
    ct.append(
        lane(
            "ct",
            f"ct_dr_isla_cos_seed{s}",
            "drivaer_ml_surface",
            "isla_surface_reference",
            COS,
            "1e-3",
            s,
            200000,
        )
    )
    for n in (27, 54, 109, 218):
        ct.append(
            lane(
                "ct",
                f"ct_dr_isla_cos_n{n}_seed{s}",
                "drivaer_scaling",
                "isla_surface_reference",
                f"{COS} train_split=train_n{n}",
                "1e-3",
                s,
                200000,
            )
        )
    ct.append(
        lane(
            "ct",
            f"ct_dr_isla_cos_pose_seed{s}",
            "drivaer_ml_surface_pose",
            "isla_surface_reference",
            COS,
            "1e-3",
            s,
            200000,
        )
    )
    ct.append(
        lane(
            "ct",
            f"ct_hl_isla_cos_single_aoa_12_seed{s}",
            "highlift_surface_single_aoa_12",
            "isla_surface_reference",
            f"{COS} {HL}",
            "1e-3",
            s,
            200000,
        )
    )
    ct.append(
        lane(
            "ct",
            f"ct_hl_isla_cos_super_scarce_area_seed{s}",
            "highlift_surface_super_scarce_area",
            "isla_surface_reference",
            f"{COS} {HL}",
            "1e-3",
            s,
            200000,
        )
    )
    ct.append(
        lane(
            "ct",
            f"ct_dr_isla_cos_area_c100k_seed{s}",
            "drivaer_ml_surface_area",
            "isla_surface_reference",
            COS,
            "1e-3",
            s,
            100000,
        )
    )

for name, rows in (("sl", sl), ("sd", sd), ("ct", ct)):
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "lanes.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "idx",
                "group",
                "run_id",
                "dataset",
                "model",
                "extra",
                "lr",
                "seed",
                "cells",
                "code",
            ]
        )
        for i, r in enumerate(rows):
            w.writerow(
                [
                    i,
                    r["group"],
                    r["run_id"],
                    r["dataset"],
                    r["model"],
                    r["extra"],
                    r["lr"],
                    r["seed"],
                    r["cells"],
                    r["code"],
                ]
            )
    assert len({r["run_id"] for r in rows}) == len(rows), f"duplicate run ids in {name}"
    print(f"{name}: {len(rows)} lanes -> {d / 'lanes.tsv'}")
print("total nodes:", len(sl) + len(sd) + len(ct))
