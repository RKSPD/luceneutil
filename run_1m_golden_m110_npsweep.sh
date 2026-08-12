#!/usr/bin/env bash
# Golden margin=1.10 index (nlist=2000, spill=2, osq int8), then sweep nprobe to find the smallest that
# clears 0.95 -- now with the faster Hamming 4*STEP kernel. nprobe is search-time (not in the index key),
# so all points reuse ONE cached index: it builds on the first point, reuses after. SIMD kernels on.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_golden_m110_npsweep_${STAMP}.log"
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE="${KNN_NPROBE:-16,20,25,32,40,48,60}"
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
echo "=== golden m1.10 nprobe sweep $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep "ivfNprobe = " "$LOG" | head -1
grep "hammingKernel" "$LOG" | tail -1 | sed 's/^/  /'
echo "nprobe list: $KNN_NPROBE  (SUMMARY lines below in nprobe order)"
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s  lat=%s ms\n",$1,$2}'
echo "=== done $(date). log: $LOG ==="
