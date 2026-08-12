#!/usr/bin/env bash
# spillBits=3 + spillMargin=1.15 combined, nprobe=50, bruteN=2000, counting-select on. Both coverage levers
# together. Compare vs sb2/m1.15 (0.957/1.148) and sb3/m1.10 (0.950/1.123). Reindexes.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=3 IVF_SPILL_MARGIN=1.15
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=50 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g LLOYD_BRUTE_N=2000
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
LOG="$OUT/run_1m_sb3m115_${STAMP}.log"
echo "=== [spillBits=3, m1.15, np50, bn2000] $(date) ==="
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 3 > "$LOG" 2>&1
idx=$(grep -iE "reindex takes" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+ sec" | head -1)
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -v ix="$idx" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "    sb3+m1.15 recall=%s MIN lat=%s ms  (reindex %s)\n",r,min,ix}'
echo "=== done $(date) ==="
