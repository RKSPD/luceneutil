#!/usr/bin/env bash
# COLD-cache >RAM comparison under a 233GiB balloon (only ~4GiB available, index 2.3GiB can't stay cached).
# dropCacheAfterWarmup evicts after warmup; the tight cap prevents refill -> every query faults cold from
# disk. 2g heap so the JVM doesn't hold the index. Compares read paths: DEFAULT (serial mmap faults = worst
# case), PREFETCH (madvise WILLNEED, zero-copy), STAGE_D (io_uring code reads), STAGE_G (io_uring sketch+code).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY="${KNN_NQUERY:-200}" KNN_SKIP_SMELL=1
export KNN_HEAP=2g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1
export KNN_DROP_CACHE_AFTER_WARMUP=1   # <-- cold: evict index after warmup
run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_cold_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v nm="$name" -F'\t' '{printf "    %s: recall=%s COLD lat=%s ms\n",nm,$1,$2}'
}
run_arm DEFAULT  LLOYD_PREFETCH_CELLS=0 LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_CO_RESIDENT_CODES=0
run_arm PREFETCH LLOYD_PREFETCH_CELLS=1 LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_CO_RESIDENT_CODES=0
run_arm STAGE_D  LLOYD_URING_RERANK=1 LLOYD_PREFETCH_CELLS=0 LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_CO_RESIDENT_CODES=0
run_arm STAGE_G  LLOYD_CO_RESIDENT_CODES=1 LLOYD_PREFETCH_CELLS=0 LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0
echo "=== done $(date) ==="
