#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
# PHASE 1: 0.9 recall@100 knee over CACHED nl8000 indices (search-only, free).
# Sweep spillBits x spillMargin (pick cached) x nprobe x nprobeMargin. bruteN fixed 400.
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=udot6 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 KNN_TOPK=100 \
  KNN_NDOC=1000000 KNN_NLIST=8000 IVFASTER_BRUTE_N=400 \
  IVFASTER_VERIFY_MULT=2 IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 \
  KNN_GCUT_AXES=1 IVFASTER_REPORT=1 IVFASTER_MF_QUERY_CLIP_Q=0.95 \
  IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
run() { # sp sm np margin
  echo "=== sp$1 sm$2 np$3 m$4 ==="
  KNN_SPILL_BITS=$1 IVFASTER_SPILL_MARGIN=$2 KNN_NPROBE=$3 IVFASTER_NPROBE_MARGIN=$4 \
    python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1
}
# sp3/sm1.40 (the recall@100 winner index) -- find 0.9 knee, low nprobe, two margins
for NP in 15 20 25 30; do run 3 1.40 $NP 0.75; done
for NP in 15 20 25 30; do run 3 1.40 $NP 1.0; done
# sp2/sm1.60 (the older frontier index)
for NP in 20 25 30 40; do run 2 1.60 $NP 0.75; done
# sp2/sm1.40
for NP in 20 25 30; do run 2 1.40 $NP 0.75; done
echo SWEEP_P1_DONE
