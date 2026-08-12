#!/usr/bin/env bash
#
# The §16.4 GOLDEN 1M config (0.957 @ 2.63 ms, measured WITHOUT SIMD), rerun with the Panama Hamming
# kernel to see how far below 2.63 ms it goes. Faithful reproduction of §16.1/§16.4:
#   nlist=2000, spillBits=2, BEAM SPILL on, spillMargin=codec default 1.10 (§16.4 pinned no margin),
#   osq int8 rerank, soarLambda=1.0, flushIters=5, bruteN=2000, nprobe=40, nquery=1000, force-merged.
#
# A/B: SIMD (default) vs -Dlloyd.noSimdPopcount=true, SAME cached index (kernel is read-time, not in the
# index key). The SCALAR arm should reproduce ~2.63 ms / 0.957 -- that is the validation that this really
# is the golden setup; the SIMD arm is the new number.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000
export KNN_NLIST=2000
export KNN_SPILL_BITS=2
# spillMargin: DELIBERATELY UNSET -- §16.4 pinned no margin, so it used the codec default (1.10). Setting
# it would change clustering and this would no longer be the golden index.
export IVF_QUANTIZER=osq
export IVF_QUANT_BITS=8
# Beam-adaptive spill WAS on in §16.1 ("spillBits=2 (beam spill)").
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
  local clear="$1"; shift
  local log="$OUT/run_1m_golden_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE="$clear" env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "    exit=$?"
  grep -E "NOTE: (index\(s\)|force_merge)" "$log" 2>/dev/null | sed 's/^/    /'
  grep "hammingKernel" "$log" 2>/dev/null | grep -v "cellsScanned=0" | tail -1 | sed 's/^/    /'
  grep "^SUMMARY" "$log" 2>/dev/null | sed 's/SUMMARY: //' | awk -F'\t' '{printf "    recall=%s  lat=%s ms\n",$1,$2}'
}

# SIMD arm builds the golden index (spillBits=2 is a NEW key -> rebuild).
run_arm SIMD   1 JAVA_TOOL_OPTIONS=
# SCALAR control reuses it; should land ~2.63 ms / 0.957 (the golden baseline validation).
run_arm SCALAR 0 JAVA_TOOL_OPTIONS=-Dlloyd.noSimdPopcount=true

echo "=== done $(date). logs: $OUT/run_1m_golden_*_${STAMP}.log ==="
