#!/usr/bin/env bash
# A/B: comparison-heap coarse shortlist vs counting-bucket shortlist (-Dlloyd.bucketShortlist=true).
# Same cached golden m1.10 index, nprobe=60, bulk dedup, bruteN=2000. Reader flag via JAVA_TOOL_OPTIONS.
# EQUIVALENCE: recall must match to ~3 decimals (bucket selects the same top-BRUTE_N modulo cutoff ties).
# WIN: bucket drops TernaryLongHeap.downHeap (~12% of query CPU per JFR) for O(1) appends + one histogram.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_bucket_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v nm="$name" -F'\t' '{printf "    %s: recall=%s  lat=%s ms\n",nm,$1,$2}'
}
run_arm HEAP   JAVA_TOOL_OPTIONS=
run_arm BUCKET JAVA_TOOL_OPTIONS=-Dlloyd.bucketShortlist=true
echo "=== done $(date) ==="
