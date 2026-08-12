#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
# spillBits=3 (was 2): each doc in up to 4 cells -> fewer probed cells for coverage; pool = bruteN*4.
# Write-time (index key sp3) -> one rebuild. Then sweep the low-nprobe knee at bn600/700.
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=udot6 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 KNN_SPILL_BITS=3 \
  KNN_NDOC=1000000 KNN_NLIST=8000 IVFASTER_SPILL_MARGIN=1.60 \
  IVFASTER_VERIFY_MULT=2 IVFASTER_NPROBE_MARGIN=0.75 \
  IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 KNN_GCUT_AXES=1 IVFASTER_REPORT=1 \
  IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
run() { echo "=== sp3 $1 ==="; for R in 1 2 3; do IVFASTER_BRUTE_N=$2 KNN_NPROBE=$3 python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1; done }
run "bn600 np45" 600 45
run "bn600 np50" 600 50
run "bn700 np45" 700 45
run "bn700 np50" 700 50
echo U6SP3DONE
