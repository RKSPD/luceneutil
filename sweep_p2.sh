#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
# PHASE 2: refine the 0.90 recall@100 knee. sp2 won; check lower spillMargin (needs builds) + bruteN,
# and nail the sp2/sm1.60 crossing precisely.
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=udot6 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 KNN_TOPK=100 \
  KNN_NDOC=1000000 KNN_NLIST=8000 \
  IVFASTER_VERIFY_MULT=2 IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 \
  KNN_GCUT_AXES=1 IVFASTER_REPORT=1 IVFASTER_MF_QUERY_CLIP_Q=0.95 \
  IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
run() { # sp sm bn np margin
  echo "=== sp$1 sm$2 bn$3 np$4 m$5 ==="
  for R in 1 2; do KNN_SPILL_BITS=$1 IVFASTER_SPILL_MARGIN=$2 IVFASTER_BRUTE_N=$3 KNN_NPROBE=$4 IVFASTER_NPROBE_MARGIN=$5 \
    python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1; done
}
# cached sp2/sm1.60 winner: nail crossing + try smaller bruteN (0.9 may saturate lower)
run 2 1.60 400 24 0.75
run 2 1.60 400 26 0.75
run 2 1.60 300 26 0.75
run 2 1.60 250 28 0.75
# lower spill: sp1 and sp0 (fewer copies -> less scan, but needs more nprobe for coverage). BUILDS.
run 1 1.60 400 30 0.75
run 1 1.60 400 40 0.75
run 0 0.0 400 40 0.75
run 0 0.0 400 60 0.75
echo SWEEP_P2_DONE
