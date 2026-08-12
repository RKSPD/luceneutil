#!/usr/bin/env bash
# WARM tax comparison of the >RAM variants at the golden point (nprobe=60, bulk dedup, cached index).
# Hypothesis: the warm io_uring tax is dominated by COPYING the sequential sketch (~12MB) into scan.buf.
# The code reads (~1MB, scattered) are the real COLD value. So Stage D (io_uring the CODE reads only,
# sketch stays zero-copy mmap) should have a far smaller warm tax than Stage G (sketch+code batched).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_PREFETCH_CELLS=0
run_arm() {
  local name="$1"; shift
  local log="$OUT/run_1m_uringtax_${name}_${STAMP}.log"
  echo "=== [arm $name] $(date) ==="
  KNN_CLEAR_CACHE=0 env "$@" ./run_knn_bench.sh 1 > "$log" 2>&1
  grep "^SUMMARY" "$log" | sed 's/SUMMARY: //' | awk -v nm="$name" -F'\t' '{printf "    %s: recall=%s lat=%s ms\n",nm,$1,$2}'
}
# DEFAULT: all mmap, zero-copy warm.
run_arm DEFAULT   LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_CO_RESIDENT_CODES=0
# STAGE_D: io_uring the CODE rerank reads only; sketch stays zero-copy mmap.
run_arm STAGE_D   LLOYD_URING_RERANK=1 LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_CO_RESIDENT_CODES=0
# STAGE_C: io_uring the SKETCH reads only (the 12MB copy) -- isolates the sketch-copy tax.
run_arm STAGE_C   LLOYD_URING_SKETCH_SCAN=1 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_CO_RESIDENT_CODES=0
echo "=== done $(date) ==="
