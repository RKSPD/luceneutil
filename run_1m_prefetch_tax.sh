#!/usr/bin/env bash
# Does the ZERO-COPY prefetch path (madvise WILLNEED + normal mmap scan) have a warm tax? Unlike io_uring,
# it makes no blocking syscall and no buffer copy -- warm, pages are resident so madvise is ~free and the
# scan is zero-copy. If warm lat == default, prefetch is the always-on >RAM mechanism that satisfies
# "behavior same, only caching differs" (cold: madvise async-faults so the sequential scan doesn't stall).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_CO_RESIDENT_CODES=0
run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_prefetchtax_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  grep -iE "prefetchAudit" "$log" | tail -1 | sed 's/^/    /'
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v nm="$name" -F'\t' '{printf "    %s: recall=%s lat=%s ms\n",nm,$1,$2}'
}
run_arm DEFAULT  LLOYD_PREFETCH_CELLS=0
run_arm PREFETCH LLOYD_PREFETCH_CELLS=1
echo "=== done $(date) ==="
