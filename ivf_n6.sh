#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
OUT=/tmp/ivf_n6.tsv
: > $OUT
for clip in 3.0 3.5 4.0; do
  export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=nibble6 \
    IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 \
    KNN_NDOC=1000000 KNN_NLIST=8000 IVFASTER_SPILL_MARGIN=1.40 IVFASTER_BRUTE_N=3000 \
    KNN_NPROBE=80 IVFASTER_VERIFY_MULT=2 IVFASTER_NPROBE_MARGIN=1.0 \
    IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 KNN_GCUT_AXES=1 IVFASTER_REPORT=1 \
    IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_NIBBLE6_CLIP_STD=$clip
  L=/tmp/n6_${clip}.log
  if python -u src/python/knnPerfTest.py > $L 2>&1; then
    S=$(grep -m1 '^SUMMARY:' $L | sed 's/^SUMMARY:[[:space:]]*//')
    E=$(grep -m1 'graphDescents=2000' $L | grep -oE 'rows/query=[0-9]+')
    echo -e "clip=$clip\trecall=$(echo "$S"|cut -f1)\tlatency=$(echo "$S"|cut -f2)\t$E" | tee -a $OUT
  else echo -e "clip=$clip\tFAIL $L" | tee -a $OUT; fi
done
echo DONE >> $OUT
