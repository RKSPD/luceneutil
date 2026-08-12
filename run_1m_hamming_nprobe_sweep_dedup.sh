#!/usr/bin/env bash
#
# nprobe sweep on the osq/int8 1M index built by run_1m_hamming_osq_ab.sh, SIMD Hamming kernel on.
# spill=5 covers each doc with more cells than the §16.4 spill=2 reference, so the SAME recall should be
# reachable at a LOWER nprobe -- trading the extra spill coverage back for latency. nprobe is SEARCH-time
# (not in the index key), so every point reuses the cached index: NO rebuild, KNN_CLEAR_CACHE=0.
#
# READ: find the smallest nprobe that still clears the recall target; that is the spill=5 operating point.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_hamming_npsweep_dedup_${STAMP}.log"

export KNN_NDOC=1000000
export KNN_NLIST=2000
export KNN_SPILL_BITS=5
export IVF_SPILL_MARGIN=1.15
export IVF_QUANTIZER=osq
export IVF_QUANT_BITS=8
export IVF_BEAM_SPILL=1
export IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32
export IVF_CENTROID_HNSW_BEAM_WIDTH=64
export IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000
export IVF_TRAIN_SAMPLE_CAP=100000
# Lower nprobe sweep -- more spilling means less nprobe should hold recall. Baseline was 40.
export KNN_NPROBE="${KNN_NPROBE:-8,12,16,20,25,32,40}"
export KNN_NQUERY="${KNN_NQUERY:-1000}"
export KNN_SKIP_SMELL=1
export KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1
export LLOYD_SHORTLIST_DEDUP=1
export KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

echo "=== nprobe sweep (SIMD on, osq int8, spill=5) $(date) ===" | tee "$LOG"
# KNN_CLEAR_CACHE=0: reuse the osq index from run_1m_hamming_osq_ab.sh.
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "    exit=$?"
grep "hammingKernel" "$LOG" 2>/dev/null | grep -v "cellsScanned=0" | tail -2 | sed 's/^/    /'
echo "nprobe list: $KNN_NPROBE"
echo "SUMMARY lines are in nprobe order (recall  lat_ms  netCPU_ms ...):"
grep "^SUMMARY" "$LOG" 2>/dev/null | awk -F'\t' '{printf "  recall=%s  lat=%s ms\n",$1,$2}'
echo "=== done $(date). log: $LOG ==="
