#!/usr/bin/env bash
# Standard Lucene HNSW, int8-quantized, 1M docs -- apples-to-apples vs the lloyd_ivf golden (both int8,
# same corpus/dim/topK). maxConn=16, beamWidth=100 (Lucene's good-recall defaults, line 627). fanout is
# search-time (efSearch = topK + fanout), so the sweep reuses ONE graph -- the cheap recall/latency curve.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_hnsw_compare_${STAMP}.log"
export KNN_INDEX_TYPE=hnsw
export KNN_NDOC=1000000
export KNN_MAXCONN=16 KNN_BEAM_WIDTH=100
# fanout sweep: efSearch = topK(100) + fanout. Trace recall/latency like the ivf nprobe sweep.
export KNN_FANOUT="${KNN_FANOUT:-0,20,50,100,200,400}"
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export KNN_DROP_CACHE_AFTER_WARMUP=0
echo "=== standard HNSW int8, 1M, fanout sweep $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep -E "reindex takes|force merge" "$LOG" | sed 's/^/  /'
echo "fanout order: $KNN_FANOUT  (efSearch = 100 + fanout)"
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s  lat=%s ms  idxMB=%s\n",$1,$2,$21}'
echo "=== done $(date). log: $LOG ==="
