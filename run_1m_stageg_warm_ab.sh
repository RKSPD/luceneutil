#!/usr/bin/env bash
# WARM A/B: default mmap coarse+rerank vs Stage G (coResidentCodes) -- reassess the Stage-C-era warm
# penalty for the OPTIMAL >RAM setting, NOW that the warm path is faster (4*STEP kernel + bulk dedup +
# postings-skip + int-guard). Both arms warm (page cache hot, cached golden index), nprobe=60, bulk dedup.
# LLOYD_URING_DEBUG=1 proves Stage G actually engaged (else the number is a silent-fallback no-op).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_PREFETCH_CELLS=0
run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_stagegwarm_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  grep -iE "Stage G|stageG|coResident" "$log" | grep -iE "engaged|uring|available|fell back|debug" | head -2 | sed 's/^/    /'
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v nm="$name" -F'\t' '{printf "    %s: recall=%s lat=%s ms\n",nm,$1,$2}'
}
# DEFAULT: all async off -- the plain warm mmap path.
run_arm DEFAULT LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_CO_RESIDENT_CODES=0
# STAGE_G: co-resident codes (optimal >RAM setting), warm.
run_arm STAGE_G LLOYD_CO_RESIDENT_CODES=1 LLOYD_URING_DEBUG=1 LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0
echo "=== done $(date) ==="
