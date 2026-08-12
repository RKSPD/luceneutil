#!/usr/bin/env bash
# A/B the topology-only refine (skip dead code re-quantization in localized refine) at nlist=40000, by BUILD
# TIME and RECALL. NEW = default (routerForTopology skips quantizeCentroids in the refine loop);
# CONTROL = -Divf.refreshCodesAlways=true (old always-re-quantize). Recall MUST be identical (the skipped
# codes were unused in the localized path); build time should drop. Both rebuild the 1M index.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=40000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0 KNN_JFR=0
run_arm() {
  local name="$1"; local opts="$2"; local log="$OUT/run_1m_topo_${name}_${STAMP}.log"
  echo "=== [$name] $(date) ==="
  KNN_CLEAR_CACHE=1 JAVA_TOOL_OPTIONS="$opts" ./run_knn_bench.sh 1 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v nm="$name" -F'\t' '{printf "    %s recall=%s indexSec=%s\n",nm,$1,$23}'
}
run_arm NEW_TOPOSKIP  ""
run_arm CTRL_REQUANT  "-Divf.refreshCodesAlways=true"
echo "=== done $(date) ==="
