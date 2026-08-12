#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
OUT=/tmp/ivf_10m_p1.tsv
: > $OUT
base() {
  export KNN_HEAP=160g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=int8 IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 \
    KNN_NDOC=10000000 KNN_NLIST=65000 IVFASTER_SPILL_MARGIN=1.40 IVFASTER_BRUTE_N=3000 \
    IVFASTER_VERIFY_MULT=2 IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 \
    KNN_GCUT_AXES=1 IVFASTER_REPORT=1 IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_NPROBE_MARGIN=0.75
}
# Phase 1: bruteN=3000 fixed, find the LOWEST nprobe that still clears 0.95.
for np in 60 80 100 130 160; do
  base
  export KNN_NPROBE=$np
  L=/tmp/10mp1_${np}.log
  if python -u src/python/knnPerfTest.py > $L 2>&1; then
    S=$(grep -m1 '^SUMMARY:' $L | sed 's/^SUMMARY:[[:space:]]*//')
    echo -e "bn=3000\tnp=$np\trecall=$(echo "$S"|cut -f1)\tlatency=$(echo "$S"|cut -f2)" | tee -a $OUT
  else
    echo -e "bn=3000\tnp=$np\tFAIL" | tee -a $OUT
  fi
done
echo DONE >> $OUT
