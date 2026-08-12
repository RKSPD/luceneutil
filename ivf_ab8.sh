#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=afterburner8 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 \
  KNN_NDOC=1000000 KNN_NLIST=8000 IVFASTER_SPILL_MARGIN=1.60 IVFASTER_BRUTE_N=600 \
  IVFASTER_VERIFY_MULT=2 IVFASTER_NPROBE_MARGIN=0.75 \
  IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 KNN_GCUT_AXES=1 IVFASTER_REPORT=1 \
  IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
# Does 9-effective-bit ranking clear 0.95 at LOWER nprobe than int8's np60?
for NP in 40 50 60; do
  echo "=== nprobe=$NP ==="
  KNN_NPROBE=$NP python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:|udot=" | tail -2
done
echo AB8DONE
