#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
# PHASE 3: nlist sweep at the sp2 winner (0.90 recall@100). nl2000 and nl16000 are BUILDS.
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=udot6 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 KNN_TOPK=100 \
  KNN_NDOC=1000000 KNN_SPILL_BITS=2 IVFASTER_SPILL_MARGIN=1.60 IVFASTER_BRUTE_N=400 \
  IVFASTER_VERIFY_MULT=2 IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 \
  KNN_GCUT_AXES=1 IVFASTER_REPORT=1 IVFASTER_MF_QUERY_CLIP_Q=0.95 \
  IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
run() { # nlist np
  echo "=== nl$1 np$2 ==="
  for R in 1 2; do KNN_NLIST=$1 KNN_NPROBE=$2 python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1; done
}
# nl2000: bigger cells, fewer probed for same coverage. scale nprobe down ~4x.
run 2000 6
run 2000 8
run 2000 10
# nl16000: smaller cells, more probed but fewer docs each. scale nprobe up ~2x.
run 16000 45
run 16000 55
echo SWEEP_P3_DONE
