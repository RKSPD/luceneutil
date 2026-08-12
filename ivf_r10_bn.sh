#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=udot6 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 KNN_SPILL_BITS=3 KNN_TOPK=10 \
  KNN_NDOC=1000000 KNN_NLIST=8000 IVFASTER_SPILL_MARGIN=1.40 \
  IVFASTER_VERIFY_MULT=2 IVFASTER_NPROBE_MARGIN=0.75 \
  IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 KNN_GCUT_AXES=1 IVFASTER_REPORT=1 \
  IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
run() { echo "=== bn$1 np$2 (recall@10) ==="; for R in 1 2; do IVFASTER_BRUTE_N=$1 KNN_NPROBE=$2 python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1; done }
run 200 10
run 150 8
run 150 5
run 100 5
run 100 3
echo R10BNDONE
