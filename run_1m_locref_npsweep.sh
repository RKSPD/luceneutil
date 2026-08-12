#!/usr/bin/env bash
# Sweep nprobe DOWN on the cached LOCALIZED-refine golden index (higher recall 0.958@60 -> headroom to cut
# nprobe for latency). nprobe is search-time -> reuses the cached index, NO rebuild. Find the lowest nprobe
# that still clears 0.95, and its latency. Bulk dedup + int-guard + postings-skip all in the current binary.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_locref_npsweep_${STAMP}.log"
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE="${KNN_NPROBE:-30,35,40,45,50,55,60}"
export KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
echo "=== localized-refine index, nprobe sweep $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
grep "ivfNprobe = " "$LOG" | head -1
echo "nprobe order: $KNN_NPROBE  (SUMMARY in order)"
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s  lat=%s ms\n",$1,$2}'
echo "=== done $(date) ==="
