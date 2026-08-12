#!/usr/bin/env bash
# nprobe x bruteN grid around the 0.95 frontier, counting-select default-on, cached nlist=2000 index (both
# knobs are search-time -> no reindex). Goal: find the MIN-latency point that still holds >=0.95 recall.
# Brackets nprobe 40/50/60/80 x bruteN 1750/2000/2500. 1000 q x3, MIN lat.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run_pt() {
  local np="$1" bn="$2"; local log="$OUT/run_1m_grid_np${np}_bn${bn}_${STAMP}.log"
  KNN_NPROBE="$np" KNN_CLEAR_CACHE=0 LLOYD_BRUTE_N="$bn" ./run_knn_bench.sh 3 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v np="$np" -v b="$bn" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "  nprobe=%-3s bruteN=%-4s recall=%s MIN lat=%s ms\n",np,b,r,min}'
}
for np in 40 50 60 80; do
  for bn in 1750 2000 2500; do
    run_pt "$np" "$bn"
  done
done
echo "=== done $(date) ==="
