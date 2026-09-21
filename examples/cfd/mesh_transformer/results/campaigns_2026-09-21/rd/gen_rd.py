"""PROTOCOL-REDO (rd): 10,000-cell accuracy claims re-measured at the reporting protocol (200,000 cells), two seeds.
Study 2: ISLA design-price ablations (weights-off, invariants-off) on the 35-case rung, the fixed-angle rung and 435 cars.
Study 3: the passive decoder's price, surface (435 cars) and interior (queries decoded passively).
Study 4: GeoTransolver and Transolver with ISLA's measure-weighted pooling (ablation) on the 35-case rung and 435 cars.
Study 5: interior sub-standings: parameter-matched ISLA (width 344), no-signed-distance arm, source-total pair, 54- and 109-car rungs of ISLA and GeoTransolver-volume."""
import csv, sys
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
HL = "model.out_scalars=3 model.out_vectors=2 +training.val_interval=25"
VOL = "+training.val_interval=10"
GTD = "forward_kwargs.global_embedding=global_data.U_inf_dir"
TD = "forward_kwargs.fx.source=global_data.U_inf_dir"
rows = []
def add(rid, ds, model, extra, lr, s, code="code_routenorm", cells=200000):
    rows.append(["rd", rid, ds, model, extra, lr, s, cells, code])
for s in (42, 43):
    # study 2
    for rung, ds, ex in (("super_scarce", "highlift_surface_super_scarce", HL), ("single_aoa_12", "highlift_surface_single_aoa_12", HL), ("dr435", "drivaer_ml_surface", "-")):
        base = "" if ex == "-" else ex + " "
        add(f"rd_isla_wtoff_{rung}_seed{s}", ds, "isla_surface_reference", base + "++model.use_measure_weights=false", "1e-3", s)
        add(f"rd_isla_geooff_{rung}_seed{s}", ds, "isla_surface_reference", base + "++model.use_relational_geo=false", "1e-3", s)
    # study 3
    add(f"rd_isla_passive_surface_dr435_seed{s}", "drivaer_ml_surface", "isla_surface_reference", "++model.query_independent=true", "1e-3", s)
    add(f"rd_isla_passive_vol_dr435_seed{s}", "drivaer_ml_volume_reference", "isla_volume_reference", f"++model.query_tokens=false ++model.query_independent=true ++model.query_mass=source_total {VOL}", "5e-4", s)
    # study 4 (measure-weighted baselines live in the code_mw snapshot)
    add(f"rd_gt_mw_super_scarce_seed{s}", "highlift_surface_super_scarce", "geotransolver_surface_mw", f"{GTD} +training.val_interval=25", "3e-3", s, "code_mw")
    add(f"rd_t_mw_super_scarce_seed{s}", "highlift_surface_super_scarce", "transolver_surface_mw", f"{TD} +training.val_interval=25", "3e-3", s, "code_mw")
    add(f"rd_gt_mw_dr435_seed{s}", "drivaer_ml_surface", "geotransolver_surface_mw", GTD, "3e-3", s, "code_mw")
    add(f"rd_t_mw_dr435_seed{s}", "drivaer_ml_surface", "transolver_surface_mw", TD, "1e-3", s, "code_mw")
    # study 5
    add(f"rd_isla_vol_w344_dr435_seed{s}", "drivaer_ml_volume_reference", "isla_volume_reference", f"model.hidden=344 ++model.query_mass=source_total {VOL}", "2.8e-4", s)
    add(f"rd_isla_vol_nosdf_dr435_seed{s}", "drivaer_ml_volume_reference", "isla_volume_reference", f"model.n_query_scalars=0 ~forward_kwargs.query_scalars ++model.query_mass=source_total {VOL}", "5e-4", s)
    add(f"rd_isla_vol_st_dr435_seed{s}", "drivaer_ml_volume_reference", "isla_volume_reference", f"++model.query_mass=source_total {VOL}", "5e-4", s)
    for n in (54, 109):
        add(f"rd_isla_vol_n{n}_seed{s}", "drivaer_ml_volume_reference", "isla_volume_reference", f"+dataset.manifest={T}/scaling_manifest.json train_split=train_n{n} ++model.query_mass=source_total {VOL}", "5e-4", s)
        add(f"rd_gt_vol_n{n}_seed{s}", "drivaer_ml_volume_mt", "geotransolver_volume", f"{GTD} +dataset.manifest={T}/scaling_manifest.json train_split=train_n{n} {VOL}", "1e-3", s)
with open(out / "lanes.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["idx", "group", "run_id", "dataset", "model", "extra", "lr", "seed", "cells", "code"])
    for i, r in enumerate(rows):
        w.writerow([i] + r)
print(len(rows), "lanes")
