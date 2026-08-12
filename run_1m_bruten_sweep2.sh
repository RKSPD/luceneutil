#!/usr/bin/env bash
# Sweep bruteN (rerank pool) DOWN from 2000 to shrink the ~33% rerank leg, WITH counting-select default-on
# and the cleaner shortlist it produces. Does 1500/1750 still hold ~0.95 recall? Reuses the cached nlist=2000
# golden index (bruteN is search-time -Dlloyd.bruteN, no reindex). nprobe=60, clean box, 1000 q x3, MIN lat.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run_bn() {
  local bn="$1"; local log="$OUT/run_1m_bn_${bn}_${STAMP}.log"
  echo "=== [bruteN=$bn] $(date) ==="
  KNN_CLEAR_CACHE=0 LLOYD_BRUTE_N="$bn" ./run_knn_bench.sh 3 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v b="$bn" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "    bruteN=%s recall=%s MIN lat=%s ms\n",b,r,min}'
}
run_bn 2000
run_bn 1750
run_bn 1500
echo "=== done $(date) ==="
