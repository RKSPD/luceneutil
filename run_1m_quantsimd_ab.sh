#!/usr/bin/env bash
# Clean A/B of the SIMD QuantizeKernel at nlist=40000, BUILD TIME. Both arms have topology-skip on (default),
# so this isolates the kernel's effect on the LIVE quantization that remains (initial-assign fallback, spill
# pass, reader load). SIMD (default) vs -Dlloyd.noSimdQuantize=true (forced scalar), same code/index config.
# Recall must be identical (kernel is bit-identical). reindex time is the metric (SUMMARY field 16).
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
  local name="$1"; local opts="$2"; local log="$OUT/run_1m_qsimd2_${name}_${STAMP}.log"
  echo "=== [$name] $(date) ==="
  KNN_CLEAR_CACHE=1 JAVA_TOOL_OPTIONS="$opts" ./run_knn_bench.sh 1 > "$log" 2>&1
  local sec=$(grep -iE "reindex takes" "$log" | tail -1 | grep -oE "[0-9]+\.[0-9]+ sec" | head -1)
  local rec=$(grep "^SUMMARY" "$log" | head -1 | awk -F'\t' '{print $1}')
  echo "    $name reindex=$sec recall=$rec"
}
run_arm SIMD   ""
run_arm SCALAR "-Dlloyd.noSimdQuantize=true"
echo "=== done $(date) ==="
