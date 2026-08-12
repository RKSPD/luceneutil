#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
# spillBits=3 with TIGHTER spillMargin (was overspilling at 1.60). Each is a rebuild (write-time).
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=udot6 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 KNN_SPILL_BITS=3 \
  KNN_NDOC=1000000 KNN_NLIST=8000 \
  IVFASTER_VERIFY_MULT=2 IVFASTER_NPROBE_MARGIN=0.75 \
  IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 KNN_GCUT_AXES=1 IVFASTER_REPORT=1 \
  IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
run() { echo "=== sm$1 bn$2 np$3 ==="; for R in 1 2 3; do IVFASTER_SPILL_MARGIN=$1 IVFASTER_BRUTE_N=$2 KNN_NPROBE=$3 python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1; done }
run 1.20 700 45
run 1.20 700 50
run 1.40 700 45
run 1.40 700 50
echo U6SP3MARGINDONE
