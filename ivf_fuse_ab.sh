#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=afterburner5 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 \
  KNN_NDOC=1000000 KNN_NLIST=8000 IVFASTER_SPILL_MARGIN=1.60 IVFASTER_BRUTE_N=600 \
  KNN_NPROBE=100 IVFASTER_VERIFY_MULT=2 IVFASTER_NPROBE_MARGIN=0.75 \
  IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 KNN_GCUT_AXES=1 IVFASTER_REPORT=1 \
  IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
echo "=== baseline (exact, fusion OFF) np100 ==="
python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1
echo "=== fusion ON np100 (default taper) ==="
IVFASTER_FUSE_CELLS=1 python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1
echo FUSEABDONE
