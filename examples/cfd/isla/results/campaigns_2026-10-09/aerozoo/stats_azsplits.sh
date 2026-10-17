#!/bin/bash
# AZ-SPLITS (2026-10-14): reduce the training-side statistics of the remaining AeroZoo splits exactly as
# kernel/aerozoo/aerozoo_stats.sbatch did for the first four (filter the per-case sample lines of stats/full.*.jsonl to
# the split part's cases, in sorted-glob order, then `aerozoo_stats.py reduce`), and fill the dataset yaml with
# fill_stats.py. Re-derives family_non_box and full first and checks them byte for byte against the originals.
# Writes only under kernel/aerozoo/stats_azsplits/ (new); the original stats/ is read only.
set -euo pipefail
T=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0
K=${T}/kernel/aerozoo
MAN=${K}/manifest.json
SRC=${K}/stats
OUT=${K}/stats_azsplits
TEMPLATE=${OUT}/aerozoo_volume.template.yaml   # the 73fd89e5f template, "???" placeholders
PY=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/physicsnemo-mesh-transformer/.venv-recipe/bin/python
cd "${K}"
SPLITS="family_non_box full bank_airframes bank_blender_bluff bank_blender_streamlined bank_cars bank_streamlined bank_zero_to_cad family_duct family_opposed_hemisphere family_opposed_sphere layout profile_confined_opposed profile_crossflow profile_dual_inlet profile_ground_effect profile_open_opposed reynolds_interior scarce super_scarce"
for split in ${SPLITS}; do
  "${PY}" -I - "${MAN}" "${split}.train" "${SRC}" "${OUT}" <<'EOF'
import json, sys, glob
man, key, src, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cases = set(json.load(open(man))[key])
seen, n = set(), 0
with open(f"{out}/filtered_{key}.jsonl", "w") as f:
    for p in sorted(glob.glob(f"{src}/full.*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if r["case"] in cases and r["case"] not in seen:
                seen.add(r["case"]); f.write(line); n += 1
print(f"{key}: {n} of {len(cases)} cases sampled", file=sys.stderr)
EOF
  "${PY}" -I aerozoo_stats.py reduce "${OUT}/filtered_${split}.train.jsonl" --out "${OUT}/stats_${split}.train.json" > "${OUT}/yaml_${split}.train.txt" 2>> "${OUT}/reduce.log"
done
echo "== reproduction against the originals"
for split in family_non_box full; do
  for f in filtered_${split}.train.jsonl stats_${split}.train.json yaml_${split}.train.txt; do
    cmp "${SRC}/${f}" "${OUT}/${f}" && echo "identical: ${f}"
  done
done
"${PY}" -I fill_stats.py "${TEMPLATE}" "${OUT}/stats_family_non_box.train.json" family_non_box.train "${OUT}/check_nonbox.yaml"
"${PY}" -I fill_stats.py "${TEMPLATE}" "${OUT}/stats_full.train.json" full.train "${OUT}/check_full.yaml"
cmp "${OUT}/check_nonbox.yaml" "${T}/recipe_v9/datasets/aerozoo_volume_nonbox.yaml" && echo "identical: aerozoo_volume_nonbox.yaml"
cmp "${OUT}/check_full.yaml" "${T}/recipe_v9/datasets/aerozoo_volume.yaml" && echo "identical: aerozoo_volume.yaml"
# The filled full yaml as template gives the same file (fill_stats.py rewrites earlier numbers and the header).
"${PY}" -I fill_stats.py "${T}/recipe_v9/datasets/aerozoo_volume.yaml" "${OUT}/stats_family_non_box.train.json" family_non_box.train "${OUT}/check_nonbox_from_filled.yaml"
cmp "${OUT}/check_nonbox_from_filled.yaml" "${T}/recipe_v9/datasets/aerozoo_volume_nonbox.yaml" && echo "identical: nonbox from the filled template"
echo "== new dataset files"
mkdir -p "${OUT}/datasets"
for split in bank_airframes bank_blender_bluff bank_blender_streamlined bank_cars bank_streamlined bank_zero_to_cad family_duct family_opposed_hemisphere family_opposed_sphere layout profile_confined_opposed profile_crossflow profile_dual_inlet profile_ground_effect profile_open_opposed reynolds_interior scarce super_scarce; do
  "${PY}" -I fill_stats.py "${TEMPLATE}" "${OUT}/stats_${split}.train.json" "${split}.train" "${OUT}/datasets/aerozoo_volume_${split}.yaml"
done
cat "${OUT}/reduce.log" | grep -v '^wrote' || true
