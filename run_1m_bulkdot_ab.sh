#!/usr/bin/env bash
#
# A/B of the multi-target int8 dot kernel (BulkDotKernel) on the INDEX path, at nlist=100k where JFR put
# ~85% of index CPU in the per-hop uint8 dot. Both arms build the SAME config from scratch; the only
# difference is -Dlloyd.noSimdCentroidDot=true (forces the scalar per-target loop) on the control.
#
# 1M docs, nlist=100000, spillBits=2, osq int8. Reindex time is the metric (printed as "reindex takes").
# Recall must be IDENTICAL between arms (the kernel is bit-exact) -- if it moves, the kernel changed
# clustering and is wrong.
#
# READ:
#   reindex(SIMD) < reindex(SCALAR)      => the batched kernel sped up assignment
#   recall(SIMD)  == recall(SCALAR)      => bit-exact, clustering unchanged (required)
#   [lloyd bulkDot] simdBatches > 0      => the vectorized kernel actually engaged (proof)
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000
export KNN_NLIST=100000
export KNN_SPILL_BITS=2
export IVF_QUANTIZER=osq
export IVF_QUANT_BITS=8
export IVF_BEAM_SPILL=1
export IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32
export IVF_CENTROID_HNSW_BEAM_WIDTH=64
export IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000
export IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40
export KNN_NQUERY="${KNN_NQUERY:-1000}"
export KNN_SKIP_SMELL=1
export KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1
export KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_bulkdot_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  # Both arms clear the cache: this is an INDEX-path A/B, each must build fresh.
  KNN_CLEAR_CACHE=1 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "    exit=$?"
  grep -E "reindex takes|force.merge" "$log" 2>/dev/null | sed 's/^/    /'
  grep "bulkDot" "$log" 2>/dev/null | tail -1 | sed 's/^/    /'
  grep "^SUMMARY" "$log" 2>/dev/null | sed 's/SUMMARY: //' | awk -F'\t' '{printf "    recall=%s  lat=%s ms\n",$1,$2}'
}

# SIMD arm (default kernel).
run_arm SIMD   JAVA_TOOL_OPTIONS=
# SCALAR control: forces BulkDotKernel.Scalar (per-target uint8DotProduct loop = pre-change behaviour).
run_arm SCALAR JAVA_TOOL_OPTIONS=-Dlloyd.noSimdCentroidDot=true

echo "=== done $(date). logs: $OUT/run_1m_bulkdot_*_${STAMP}.log ==="
