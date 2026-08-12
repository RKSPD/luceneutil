#!/bin/zsh
set -e
cd /local/home/rikhil/vectordb/luceneutil
source .venv/bin/activate
# Standard Lucene HNSW, int8 (7-bit scalar quant), M=16, beamWidth=100. Same 1M Cohere, topK=100, warm.
# Sweep fanout (efSearch = topK + fanout) to find the 0.95 knee.
export KNN_HEAP=16g KNN_INDEX_TYPE=hnsw KNN_NDOC=1000000 KNN_NQUERY=1000 KNN_TOPK=100 \
  KNN_MAXCONN=16 KNN_BEAM_WIDTH=100 KNN_GCUT_AXES=1
for F in 60 100 150 200; do
  echo "=== hnsw int8 fanout=$F (ef=$((100+F))) ==="
  KNN_FANOUT=$F python -u src/python/knnPerfTest.py 2>&1 | grep -iE "^SUMMARY:" | tail -1
done
echo HNSWDONE
