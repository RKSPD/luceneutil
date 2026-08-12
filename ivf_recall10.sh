#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
# Frontier config (sp3/sm1.40/bn700/np45), compare recall@10 vs recall@100 on the SAME index.
export KNN_HEAP=16g KNN_INDEX_TYPE=ivfaster IVFASTER_FINE_TIER=udot6 \
  IVFASTER_COARSE_BITS=1 IVFASTER_COARSE_MF=1 KNN_SPILL_BITS=3 \
  KNN_NDOC=1000000 KNN_NLIST=8000 IVFASTER_SPILL_MARGIN=1.40 IVFASTER_BRUTE_N=700 \
  KNN_NPROBE=45 IVFASTER_VERIFY_MULT=2 IVFASTER_NPROBE_MARGIN=0.75 \
  IVFASTER_GRAPH_M=16 IVFASTER_EF_CONSTRUCTION=64 KNN_GCUT_AXES=1 IVFASTER_REPORT=1 \
  IVFASTER_MF_QUERY_CLIP_Q=0.95 IVFASTER_UDOT_LIB=/tmp/libivfasterudot.so
for TK in 100 10; do
  echo "=== topK=$TK (frontier np45/bn700) ==="
  KNN_TOPK=$TK python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1
done
echo R10DONE
