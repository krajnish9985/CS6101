#!/bin/bash
cd "$(dirname "$0")"
source ../venv/bin/activate
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONUNBUFFERED=1
FAILED=()
run() { echo; echo "=================== $(date +%T)  $*"; "$@" || { echo "!!! FAILED: $*"; FAILED+=("$*"); }; }
LS_DEV=10,20,30,50,75,100,150,200
LS_FULL=10,15,20,30,40,50,75,100,150,200

run python prep_sift.py
run python exact_gt.py --dataset sift100k
run python exact_gt.py --dataset sift1m
run python vamana_build.py --dataset sift100k --R 64 --L 100 --alpha 1.2 --tag sift100k_ref
run python vamana_search.py --dataset sift100k --tag sift100k_ref --Ls $LS_DEV
run python sanity_query.py --tag sift100k_ref --qid 0 --L 100
run python vamana_build.py --dataset sift100k --R 32 --L 100 --alpha 1.2 --tag sift100k_R32
run python vamana_build.py --dataset sift100k --R 64 --L 100 --alpha 1.0 --tag sift100k_alpha1.0
run python vamana_search.py --dataset sift100k --tag sift100k_R32 --Ls $LS_DEV
run python vamana_search.py --dataset sift100k --tag sift100k_alpha1.0 --Ls $LS_DEV

run python prep_beir.py --dataset quora --sample 200000 --seed 42 --solver randomized
run python prep_beir.py --dataset nq    --sample 200000 --seed 42 --solver randomized
run python exact_gt.py --dataset quora
run python exact_gt.py --dataset nq

for ds in sift100k sift1m quora nq; do
  run python hnsw.py --dataset $ds --M 32 --efc 200 --efs 16,32,64,100,150,200,256,512
done

for n in 100000 250000 500000; do
  run python vamana_build.py --dataset sift1m --n-sub $n --no-save --tag scale_sift1m_$n
done
for n in 250000 500000 1000000; do
  run python vamana_build.py --dataset nq --n-sub $n --no-save --tag scale_nq_$n
done

for ds in sift1m quora nq; do
  run python vamana_build.py --dataset $ds --R 64 --L 100 --alpha 1.2 --tag ${ds}_main
  run python vamana_search.py --dataset $ds --tag ${ds}_main --Ls $LS_FULL
done

run python make_report.py
echo; echo "=================== ALL DONE $(date +%T)"
if [ ${#FAILED[@]} -gt 0 ]; then printf '!!! FAILED STEP: %s\n' "${FAILED[@]}"; else echo "no failures"; fi
