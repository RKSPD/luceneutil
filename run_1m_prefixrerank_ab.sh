#!/usr/bin/env bash
# A/B the middle-tier prefix rerank on the cached 1M index (search-time, no rebuild). Baseline (no prefix)
# vs 512-dim and 256-dim int8 prefix prefilter, keep top 50%. Tests: does recall hold (prefix ranks well?)
# and does latency drop (fewer full 1024-dim scores)? 3 passes, MIN latency (40M build co-running).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
run_arm() {
  local name="$1"; shift; local log="$OUT/run_1m_pfxrr_${name}_${STAMP}.log"
  echo "=== [$name] $(date) ==="
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh 3 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v nm="$name" -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "    %s recall=%s MIN lat=%s ms\n",nm,r,min}'
}
run_arm BASELINE  JAVA_TOOL_OPTIONS=
run_arm PFX512    JAVA_TOOL_OPTIONS="-Dlloyd.rerankPrefixDims=512 -Dlloyd.rerankPrefixKeep=0.5"
run_arm PFX256    JAVA_TOOL_OPTIONS="-Dlloyd.rerankPrefixDims=256 -Dlloyd.rerankPrefixKeep=0.5"
echo "=== done $(date) ==="
