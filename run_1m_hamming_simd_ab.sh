#!/usr/bin/env bash
#
# 1M A/B of the coarse-scan Hamming kernel: Panama SIMD (default) vs the scalar four-accumulator loop
# (-Dlloyd.noSimdPopcount=true). Both arms run the SAME cached index -- the kernel is a READ-time choice
# that is not in the index key -- so this isolates the coarse popcount and nothing else.
#
# Config (as requested): spill=5, margin=1.15, nlist=2000, 1M docs. Warm search (this A/B is about the
# coarse-scan CPU, not I/O), so all async/uring stages are off.
#
# READ THE RESULT AS:
#   [lloyd hammingKernel] vectorized=true ...   => the SIMD path actually engaged (SIMD arm)
#   [lloyd hammingKernel] vectorized=false ...  => scalar arm (control) confirmed on the scalar loop
#   recall SIMD == recall SCALAR                => kernel is bit-identical (a wrong Hamming reorders the
#                                                  shortlist and only shows as slightly-off recall)
#   lat   SIMD  <  lat   SCALAR                 => the transplant paid off end-to-end (Amdahl-capped:
#                                                  coarse popcount was 26.5% of warm CPU, so ~<=23% max)
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb

# 1M reference config.
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
# Coarse Hamming sketch scan is the path under test -- it is the harness default, made explicit here.
export LLOYD_SKETCH_SCAN=1
# WARM search: async I/O stages off; this A/B is coarse-scan CPU only.
export KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

./ram_balloon.sh --release >/dev/null 2>&1 || true

run_arm() {
  local name="$1"; shift
  local clear="$1"; shift
  local log="$OUT/run_1m_hamming_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE="$clear" env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  echo "    exit=$?"
  grep -E "NOTE: (index\(s\)|force_merge)" "$log" 2>/dev/null | sed 's/^/    /'
  grep "hammingKernel" "$log" 2>/dev/null | sed 's/^/    /'
  grep "^SUMMARY" "$log" 2>/dev/null | sed 's/^/    /'
}

# SIMD arm first, clearing the cache (codec/read-path change => rebuild the 1M index once).
# JAVA_TOOL_OPTIONS is the only knob that reaches the search JVM without a harness change (it is read by
# every JVM launch automatically); empty here so the default SIMD kernel is used.
run_arm SIMD   1 JAVA_TOOL_OPTIONS=
# SCALAR control: -Dlloyd.noSimdPopcount=true forces HammingKernel.Scalar. Reuses the SIMD arm's cached
# index (KNN_CLEAR_CACHE=0): the kernel is not in the index key, so this is one index scored two ways.
run_arm SCALAR 0 JAVA_TOOL_OPTIONS=-Dlloyd.noSimdPopcount=true

echo "=== done $(date). logs: $OUT/run_1m_hamming_*_${STAMP}.log ==="
