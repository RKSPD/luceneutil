#!/usr/bin/env bash
# Standard Lucene HNSW at 4-BIT quantization, 1M docs -- the tougher comparison (smaller codes, faster
# traversal) vs the int8 lloyd_ivf golden. -quantizeBits 4. fanout sweep (search-time, one graph).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_hnsw_q4_${STAMP}.log"
export KNN_INDEX_TYPE=hnsw KNN_NDOC=1000000 KNN_MAXCONN=16 KNN_BEAM_WIDTH=100
export KNN_QUANTIZE_BITS=4
export KNN_FANOUT="${KNN_FANOUT:-0,50,100,200,300,500}"
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}" KNN_DROP_CACHE_AFTER_WARMUP=0
echo "=== HNSW 4-bit, 1M, fanout sweep $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep -E "reindex takes" "$LOG" | sed 's/^/  /'
grep -oE "cmd: \[.*\]" "$LOG" | head -1 | tr ',' '\n' | grep -A1 quantizeBits | head | sed 's/^/  /'
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms idxMB=%s bits=%s\n",$1,$2,$21,$14}' | sort -u
echo "=== done $(date) ==="
