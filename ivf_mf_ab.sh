#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=int8 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 \
  KNN_NDOC=1000000 KNN_NLIST=8000 IVFASTER_SPILL_MARGIN=1.60 IVFASTER_BRUTE_N=600 \
  KNN_NPROBE=60 IVFASTER_VERIFY_MULT=2 IVFASTER_NPROBE_MARGIN=0.75 \
  IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 KNN_GCUT_AXES=1 IVFASTER_REPORT=1 \
  IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
for R in 1 2 3; do
  echo "=== int8 np60 rep$R (MF no-not) ==="
  python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1
done
echo MFABDONE
