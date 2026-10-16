#!/bin/bash
cd "$(dirname "$0")"
source ../venv/bin/activate
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONUNBUFFERED=1
mkdir -p logs results
DS=msmarco
IDXF=indices/disk_$DS/disk.index
LS="50 100 150 200"
W=4
RAM_LIMIT_KB=$((8*1024*1024))
FAILED=()
run() { echo; echo "=================== $(date +%T)  $*"; "$@" || { echo "!!! FAILED: $*"; FAILED+=("$*"); }; }

[ -f data/processed/${DS}_meta.json ] || run python prep_beir.py --dataset $DS --sample 200000 --seed 42 --solver randomized
[ -f data/processed/${DS}_gt.npy ] || run python exact_gt.py --dataset $DS --no-timing
[ -f indices/vamana_${DS}_main/nbrs.npy ] || run python vamana_build.py --dataset $DS --R 48 --L 80 --alpha 1.2 --tag ${DS}_main
[ -f $IDXF ] || run python disk_build.py --dataset $DS --tag ${DS}_main --pq-m 32

DEV=$(df --output=source "$IDXF" | tail -1)
{ echo "index file: $(realpath $IDXF)"; echo "filesystem device: $DEV"; lsblk -o NAME,ROTA,SIZE,TYPE,MODEL,TRAN "$DEV" 2>/dev/null;
  lsblk -d -o NAME,ROTA,SIZE,MODEL,TRAN; } > results/ec_device.txt
cat results/ec_device.txt

for L in $LS; do
  run python cache_tools.py evict $IDXF --json results/ec_fincore_cold_L$L.json
  run bash -c "ulimit -v $RAM_LIMIT_KB; OMP_NUM_THREADS=1 python disk_search.py --dataset $DS --mode cold --Ls $L --W $W --fincore-json results/ec_fincore_cold_L$L.json"
done
run python cache_tools.py status $IDXF --json results/ec_fincore_before_warmup.json
run bash -c "ulimit -v $RAM_LIMIT_KB; OMP_NUM_THREADS=1 python disk_search.py --dataset $DS --mode warm --Ls ${LS// /,} --W $W"

run python make_ec_report.py
echo; echo "=================== EC DONE $(date +%T)"
if [ ${#FAILED[@]} -gt 0 ]; then printf '!!! FAILED STEP: %s\n' "${FAILED[@]}"; else echo "no failures"; fi
