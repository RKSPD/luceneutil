#!/usr/bin/env bash
# A/B tiled rerank (bulkDotAt shares query load across 4 candidate records) vs per-candidate dot4. Cached
# m1.25 index, np40/bn1750, counting-select on. Recall must be identical (bit-identical dot + same combine).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.25
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g LLOYD_BRUTE_N=1750
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run_arm() {
  local name="$1"; local opts="$2"; local log="$OUT/run_1m_tile_${name}_${STAMP}.log"
  echo "=== [$name] $(date) ==="
  KNN_CLEAR_CACHE=0 JAVA_TOOL_OPTIONS="$opts" ./run_knn_bench.sh 3 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v nm="$name" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "    %s recall=%s MIN lat=%s ms\n",nm,r,min}'
}
run_arm TILED    ""
run_arm PERCAND  "-Dlloyd.noTileRerank=true"
echo "=== done $(date) ==="
