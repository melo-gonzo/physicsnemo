#!/bin/bash
#SBATCH -A coreai_modulus_cae
#SBATCH -J olbc.download.ranged
#SBATCH --time=1-00:00:00
#SBATCH -p cpu_datamover
#SBATCH -q cpu-datamover
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-1
#SBATCH -o sbatch_logs/%x-%A_%a.log
#SBATCH -e sbatch_logs/%x-%A_%a.log
# OLBC raw download, ranged (2026-09-22): Zenodo throttles each connection to ~1 MB/s, so each train.nc is fetched as
# 16 byte-range streams into parts/ and assembled; md5 checked against the Zenodo file checksum. Array 0 = poisson-square-bc5
# (record 18377514, 15,821,508,752 B), 1 = elasticity-circlehollow-m1 (record 18377539, 21,478,490,184 B). CC BY 4.0.
set -uo pipefail
U=/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/datasets
RAW=$U/olbc_raw
NSTREAMS=16
case "${SLURM_ARRAY_TASK_ID}" in
  0) D=poisson-square-bc5;          R=18377514; SIZE=15821508752; MD5=109c152d506a3d7d98baca863f58ddd3 ;;
  1) D=elasticity-circlehollow-m1;  R=18377539; SIZE=21478490184; MD5=d3344c49f799a5b32a63a12eec194dcc ;;
  *) echo bad index; exit 2 ;;
esac
URL="https://zenodo.org/api/records/$R/files/train.nc/content"
P=$RAW/$D/parts; mkdir -p "$P"
echo "=== start $(date -Is) job=$SLURM_JOB_ID node=$(hostname) set=$D size=$SIZE streams=$NSTREAMS"
CH=$(( (SIZE + NSTREAMS - 1) / NSTREAMS ))
fetch_part() {  # i
  local i=$1 s=$(( i * CH )) e=$(( (i + 1) * CH - 1 )); [ $e -ge $SIZE ] && e=$(( SIZE - 1 ))
  local want=$(( e - s + 1 )) f="$P/part_$(printf %02d $i)"
  for attempt in $(seq 1 40); do
    local have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] && break
    curl -sS -L --fail -r $(( s + have ))-$e --speed-limit 20000 --speed-time 120 -o - "$URL" >> "$f"
    rc=$?; echo "part $i attempt $attempt rc=$rc have=$(stat -c %s "$f") want=$want $(date -Is)"
    [ $rc -eq 0 ] && [ "$(stat -c %s "$f")" -ge "$want" ] && break
    sleep 15
  done
  have=$(stat -c %s "$f"); [ "$have" -gt "$want" ] && truncate -s $want "$f"
  echo "part $i DONE size=$(stat -c %s "$f") want=$want"
}
for i in $(seq 0 $((NSTREAMS-1))); do fetch_part $i & done
( while true; do sleep 60; t=0; for f in "$P"/part_*; do [ -f "$f" ] && t=$(( t + $(stat -c %s "$f") )); done; echo "progress $(date -Is) bytes=$t of $SIZE ($(( 100 * t / SIZE ))%)"; done ) &
mon=$!
wait $(jobs -p | grep -v $mon) 2>/dev/null
kill $mon 2>/dev/null
# assemble only once the single-stream job 828201 no longer exists (it wrote the same train.nc path)
while squeue -h -j 828201 2>/dev/null | grep -q .; do echo "waiting for job 828201 to finish/cancel $(date -Is)"; sleep 60; done
total=0; for i in $(seq 0 $((NSTREAMS-1))); do total=$(( total + $(stat -c %s "$P/part_$(printf %02d $i)") )); done
if [ "$total" -ne "$SIZE" ]; then echo "SIZE MISMATCH total=$total want=$SIZE; not assembling"; exit 3; fi
cat "$P"/part_?? > "$RAW/$D/train.nc"
got=$(md5sum "$RAW/$D/train.nc" | cut -d' ' -f1)
echo "md5 $D/train.nc got=$got want=$MD5 $([ "$got" = "$MD5" ] && echo OK || echo MISMATCH)"
[ "$got" = "$MD5" ] && rm -rf "$P"
ls -la "$RAW/$D"
echo "=== RANGED-DOWNLOAD-DONE $(date -Is)"
