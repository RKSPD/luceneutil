#!/usr/bin/env bash
# Measure the cell-order int-guard change at the golden point (nprobe=60, bulk dedup, spill=2, m1.10).
# Reuses the cached index (read-path change only). Compare recall/lat to the pre-change 0.954 @ 1.77 ms.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_intguard_${STAMP}.log"
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE="${KNN_NPROBE:-40,60,80}" KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
echo "=== int-guard, golden nprobe sweep $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
grep "hammingKernel" "$LOG" | grep -v cellsScanned=0 | tail -1 | sed 's/^/  /'
echo "nprobe order: $KNN_NPROBE"
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}'
echo "=== done $(date) ==="
