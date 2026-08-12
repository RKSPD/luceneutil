#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
OUT=/tmp/ivf_10m_p2.tsv
: > $OUT
base() {
  export KNN_HEAP=160g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=int8 IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 \
    KNN_NDOC=10000000 KNN_NLIST=65000 IVFASTER_SPILL_MARGIN=1.40 \
    IVFASTER_VERIFY_MULT=2 IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 \
    KNN_GCUT_AXES=1 IVFASTER_REPORT=1 IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_NPROBE_MARGIN=0.75
}
# Phase 2: hold nprobe at the coverage floor (80, 100); walk bruteN DOWN to find min shortlist >= 0.95.
for np in 80 100; do
  for bn in 3000 2500 2000 1500; do
    base
    export KNN_NPROBE=$np IVFASTER_BRUTE_N=$bn
    L=/tmp/10mp2_${np}_${bn}.log
    if python -u src/python/knnPerfTest.py > $L 2>&1; then
      S=$(grep -m1 '^SUMMARY:' $L | sed 's/^SUMMARY:[[:space:]]*//')
      echo -e "np=$np\tbn=$bn\trecall=$(echo "$S"|cut -f1)\tlatency=$(echo "$S"|cut -f2)" | tee -a $OUT
    else
      echo -e "np=$np\tbn=$bn\tFAIL" | tee -a $OUT
    fi
  done
done
echo DONE >> $OUT
