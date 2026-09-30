#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J jeb.download
#SBATCH --time=12:00:00
#SBATCH -p cpu_datamover
#SBATCH -q cpu-datamover
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -o sbatch_logs/%x-%j.log
#SBATCH -e sbatch_logs/%x-%j.log
# DeepJEB++ raw download (2026-09-22): Hugging Face dataset KAIST-SmartDesignLab/DeepJEB-PP (ODC-By 1.0), revision main
# (= v1.2: v1.1 fields + reproducibility/ split files). snapshot_download is resumable (incomplete files are resumed, finished
# ones skipped), so a requeue just continues. The four archives are then extracted in parallel (pigz when present), the file
# counts are checked against the card (15,360 per modality) and the torsion fields of one case are compared between
# 3_fea_fields.tar.gz and the v1.1 torsion patch archive to establish whether the main archive already carries the Z-axis torsion.
set -uo pipefail
U=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe
RAW=$U/datasets/deepjebpp_pdmsh/deepjebpp_raw
VENV=$U/.venv-jeb
export HF_HOME=$U/.hfcache HF_HUB_ENABLE_HF_TRANSFER=$("$VENV/bin/python" -c 'import importlib.util as u; print(int(u.find_spec("hf_transfer") is not None))')
export PYTHONUNBUFFERED=1
mkdir -p "$RAW/hf" "$HF_HOME"
echo "=== start $(date -Is) job=$SLURM_JOB_ID node=$(hostname) raw=$RAW hf_transfer=$HF_HUB_ENABLE_HF_TRANSFER"
"$VENV/bin/python" - <<EOF
import time
from huggingface_hub import snapshot_download, HfApi
t0 = time.time()
api = HfApi()
info = api.dataset_info("KAIST-SmartDesignLab/DeepJEB-PP", files_metadata=True)
tot = sum((s.size or 0) for s in info.siblings)
print("revision", info.sha, "files", len(info.siblings), "bytes", tot, flush=True)
for s in info.siblings: print(f"  {s.rfilename}  {s.size}", flush=True)
p = snapshot_download("KAIST-SmartDesignLab/DeepJEB-PP", repo_type="dataset", local_dir="$RAW/hf", max_workers=8)
print("SNAPSHOT-DONE", p, f"{time.time()-t0:.0f}s", flush=True)
EOF
rc=$?; echo "snapshot rc=$rc"; [ $rc -ne 0 ] && exit $rc
cd "$RAW/hf" && ls -la && du -sh . && git -C . log -1 2>/dev/null; cat .cache/huggingface/download/*.metadata 2>/dev/null | head -20
ZIP=$(command -v pigz >/dev/null && echo "-I pigz" || echo "-z"); echo "tar decompressor: ${ZIP}"
mkdir -p "$RAW/1_surface_meshes" "$RAW/2_boundary_conditions" "$RAW/3_fea_fields" "$RAW/3_fea_fields_torsion_z_v1.1"
ex() { local a=$1 d=$2; [ -f "$RAW/$d/.extracted" ] && { echo "skip $a (extracted)"; return; }
       echo "extract $a -> $d $(date -Is)"; tar $ZIP -xf "$RAW/hf/$a" -C "$RAW/$d" && touch "$RAW/$d/.extracted"; echo "extract $a rc=$? $(date -Is)"; }
ex 1_surface_meshes.tar.gz 1_surface_meshes & ex 2_boundary_conditions.tar.gz 2_boundary_conditions &
ex 3_fea_fields.tar.gz 3_fea_fields & ex 3_fea_fields_torsion_z_v1.1.tar.gz 3_fea_fields_torsion_z_v1.1 &
wait
for d in 1_surface_meshes 2_boundary_conditions 3_fea_fields 3_fea_fields_torsion_z_v1.1; do
  echo "--- $d: top-level $(ls "$RAW/$d" | head -3 | tr '\n' ' '); files obj=$(find "$RAW/$d" -name '*.obj' | wc -l) npz=$(find "$RAW/$d" -name '*.npz' | wc -l); du=$(du -sh "$RAW/$d" | cut -f1)"
done
cp "$RAW/hf/deepjebpp_labels.csv" "$RAW/hf/metadata.json" "$RAW/hf/README.txt" "$RAW/hf/README.md" "$RAW/hf/LICENSE" "$RAW/" 2>/dev/null
cp -r "$RAW/hf/reproducibility" "$RAW/" 2>/dev/null
"$VENV/bin/python" - <<EOF
import numpy as np, glob, os
R = "$RAW"
f = sorted(glob.glob(f"{R}/3_fea_fields/**/*.npz", recursive=True)); p = sorted(glob.glob(f"{R}/3_fea_fields_torsion_z_v1.1/**/*.npz", recursive=True))
print("n fields", len(f), "n patch", len(p))
a = np.load(f[0]); print("fields keys", f[0], {k: (a[k].shape, str(a[k].dtype)) for k in a.files})
pf = {os.path.basename(x): x for x in p}; b = np.load(pf[os.path.basename(f[0])]); print("patch keys", {k: (b[k].shape, str(b[k].dtype)) for k in b.files})
print("surface_points equal", np.array_equal(a["surface_points"], b["surface_points"]))
for k in ("tor_U", "tor_vm", "tor_maxu", "tor_p95vm"):
    print(k, "main==patch:", np.array_equal(np.asarray(a[k]), np.asarray(b[k])) if k in a.files else "missing in main", "| main maxabs", float(np.abs(a[k]).max()) if k in a.files else None, "patch maxabs", float(np.abs(b[k]).max()))
print("tor_maxvm in main:", "tor_maxvm" in a.files)
bc = np.load(glob.glob(f"{R}/2_boundary_conditions/**/{os.path.basename(f[0])}", recursive=True)[0]); print("bc keys", {k: (bc[k].shape, str(bc[k].dtype)) for k in bc.files}); print("bc n_surface", bc["n_surface"], "N", a["surface_points"].shape[0], "M", a["surface_faces"].shape[0])
EOF
echo "=== DOWNLOAD-DONE rc=0 $(date -Is)"
