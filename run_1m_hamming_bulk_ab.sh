#!/usr/bin/env bash
#
# Follow-up to run_1m_hamming_simd_ab.sh. The first A/B ran with spill dedup on the per-slot scan
# (seen != null), which gates OFF the bulk Hamming entry point, so it measured the PER-ROW SIMD kernel.
# This variant adds -Dlloyd.shortlistDedup=true (seen == null), which is exactly the condition under
# which bulkHamming() fires -- the flagship "query vectors hoisted out of the row loop" path, and the
# only one that increments HammingKernel.simdEngaged. It reuses the SAME cached 1M index (all flags here
# are read-time and not in the index key), so no rebuild.
#
# READ THE RESULT AS:
#   cellsScanned > 0 (SIMD arm)  => the BULK vectorized kernel actually engaged (proof, not assumed)
#   recall SIMD == recall SCALAR => bit-identical, AND == the per-row A/B's recall => dedup is neutral
#   lat SIMD < lat SCALAR        => bulk transplant paid off end-to-end
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000
export KNN_NLIST=2000
export KNN_SPILL_BITS=5
export IVF_SPILL_MARGIN=1.15
export IVF_QUANTIZER=blocksphere
export IVF_BLOCK_P=2
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
# The bulk path fires only with shortlist dedup (seen == null on the per-slot scan).
export LLOYD_SHORTLIST_DEDUP=1
export KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_hammingbulk_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  # KNN_CLEAR_CACHE=0: reuse the index built by run_1m_hamming_simd_ab.sh.
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "    exit=$?"
  grep "hammingKernel" "$log" 2>/dev/null | grep -v "cellsScanned=0" | tail -2 | sed 's/^/    /'
  grep "^SUMMARY" "$log" 2>/dev/null | sed 's/^/    /'
}

run_arm SIMD   JAVA_TOOL_OPTIONS=
run_arm SCALAR JAVA_TOOL_OPTIONS=-Dlloyd.noSimdPopcount=true

echo "=== done $(date). logs: $OUT/run_1m_hammingbulk_*_${STAMP}.log ==="
