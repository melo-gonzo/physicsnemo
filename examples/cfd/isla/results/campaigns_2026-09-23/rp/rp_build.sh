#!/bin/bash
# REF-BOARD inference half (2026-09-23): build the inference recipe copy under $HOME (scratch is at its
# file-count quota). The copy is recipe_globin plus (a) the six consistency-bench dataset yamls from
# recipe_bench and (b) the StratifiedSubsampleMesh transform from recipe_bench/src/domain_transforms.py.
# Nothing under $T is modified.
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
RP=$HOME/rp_probes
mkdir -p "$RP/logs" "$RP/bench_evals" "$RP/territory"
rsync -a --delete --exclude outputs --exclude '__pycache__' --exclude '*.bak_*' "$T/recipe_globin/" "$RP/recipe_rp/"
for f in drivaer_bench_unif drivaer_bench_strat drivaer_bench_biased \
         highlift_bench_super_scarce_unif highlift_bench_super_scarce_strat highlift_bench_super_scarce_biased; do
  cp "$T/recipe_bench/datasets/$f.yaml" "$RP/recipe_rp/datasets/$f.yaml"
done
# The wing strat yaml never received store_center_as (only unif/biased did, for the territory probe); add it so
# the three wing draws share one pipeline definition. Inert for the model; the center lands in global_data only.
python3 - "$RP/recipe_rp/datasets/highlift_bench_super_scarce_strat.yaml" <<'EOF'
import sys, re
p = sys.argv[1]; s = open(p).read()
old = "    - _target_: ${dp:CenterMesh}\n      use_area_weighting: false\n    - _target_: ${dp:ComputeSurfaceNormals}"
new = "    - _target_: ${dp:CenterMesh}\n      use_area_weighting: false\n      store_center_as: center\n    - _target_: ${dp:ComputeSurfaceNormals}"
assert s.count(old) == 1, "CenterMesh block not found once"
open(p, "w").write(s.replace(old, new))
EOF
# Append StratifiedSubsampleMesh (bench file: from its @register() line to EOF; it is the last class there).
start=$(grep -n "^class StratifiedSubsampleMesh" "$T/recipe_bench/src/domain_transforms.py" | cut -d: -f1)
start=$((start - 1))
sed -n "${start},\$p" "$T/recipe_bench/src/domain_transforms.py" > /tmp/rp_strat_class.py
grep -q "^@register()" /tmp/rp_strat_class.py
grep -q "^class StratifiedSubsampleMesh" "$RP/recipe_rp/src/domain_transforms.py" || {
  printf "\n\n### REF-BOARD inference half (2026-09-23): the QUAD-VAR stratified draw, copied verbatim from recipe_bench.\n" >> "$RP/recipe_rp/src/domain_transforms.py"
  cat /tmp/rp_strat_class.py >> "$RP/recipe_rp/src/domain_transforms.py"
}
# Probe manifest (territory_probe.py reads run_id / dataset / model / extra_overrides).
printf 'run_id\tdataset\tmodel\textra_overrides\n' > "$RP/manifest.tsv"
printf 'ref_isla_r1_dr435_seed42\tdrivaer_ml_surface\tisla_surface_ref\t-\n' >> "$RP/manifest.tsv"
printf 'ref_isla_r1_dr435_seed43\tdrivaer_ml_surface\tisla_surface_ref\t-\n' >> "$RP/manifest.tsv"
printf 'ref_isla_r1_hl35_seed42\thighlift_surface_super_scarce\tisla_surface_ref\tmodel.out_scalars=3 model.out_vectors=2 +training.val_interval=25\n' >> "$RP/manifest.tsv"
printf 'ref_isla_r1_hl35_seed43\thighlift_surface_super_scarce\tisla_surface_ref\tmodel.out_scalars=3 model.out_vectors=2 +training.val_interval=25\n' >> "$RP/manifest.tsv"
echo "--- diff recipe_rp vs recipe_globin (expected: 6 new yamls, 1 changed src file)"
diff -rq --exclude outputs --exclude '__pycache__' --exclude '*.bak_*' "$T/recipe_globin" "$RP/recipe_rp" || true
echo "--- smoke: import + hydra compose under code_ref"
VENV=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe
cd "$RP/recipe_rp"
PYTHONPATH="$T/code_ref:$RP/recipe_rp/src" "$VENV/bin/python" - <<'EOF'
import sys, os
import hydra
from omegaconf import OmegaConf
import domain_transforms as dt
assert hasattr(dt, "StratifiedSubsampleMesh") and hasattr(dt, "PoissonBiasedSubsampleMesh")
from datasets import load_dataset_config
from pathlib import Path
root = Path(os.getcwd())
for name in ["drivaer_bench_unif", "drivaer_bench_strat", "drivaer_bench_biased",
             "highlift_bench_super_scarce_unif", "highlift_bench_super_scarce_strat", "highlift_bench_super_scarce_biased"]:
    y = load_dataset_config(root / "datasets" / f"{name}.yaml")
    y = OmegaConf.merge(y, {"sampling_resolution": 200000})
    tr = OmegaConf.to_container(y.pipeline.transforms, resolve=True)
    samplers = [t["_target_"].split(".")[-1] for t in tr if "Subsample" in t["_target_"]]
    centers = [t.get("store_center_as") for t in tr if t["_target_"].endswith("CenterMesh")]
    print(f"{name:40s} sampler={samplers} center={centers}")
with hydra.initialize_config_dir(config_dir=str(root / "conf"), version_base=None):
    cfg = hydra.compose(config_name="infer", overrides=["model=isla_surface_ref", "dataset=drivaer_bench_biased", "run_id=x",
                                                       "infer_split=val", "sampling_resolution=200000", "precision=float32", "+save_predictions=false"])
print("model:", OmegaConf.to_yaml(cfg.model).strip().replace("\n", "; "))
print("seed:", cfg.training.seed, "save_predictions:", cfg.save_predictions)
EOF
echo BUILD-OK
