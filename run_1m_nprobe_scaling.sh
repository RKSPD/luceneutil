#!/usr/bin/env bash
#
# nprobe SCALING sweep on the golden 1M index (nlist=2000, spillBits=2, beam spill, osq int8), SIMD Hamming
# + shortlist dedup (bulk kernel). bruteN fixed at 2000. nprobe is search-time (not in the index key), so
# every point reuses the cached index -- NO rebuild.
#
# QUESTION: how does latency scale with nprobe now that the coarse scan is SIMD-fast? If the coarse tier is
# cheap, doubling nprobe (≈2x the docs Hamming-scanned) should add much less than 2x latency, because the
# rerank pool stays fixed at bruteN and only the coarse scan grows. Sublinear => coarse tier is no longer
# the bottleneck.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_nprobe_scaling_${STAMP}.log"

export KNN_NDOC=1000000
export KNN_NLIST=2000
export KNN_SPILL_BITS=2
export IVF_QUANTIZER=osq
export IVF_QUANT_BITS=8
export IVF_BEAM_SPILL=1
export IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32
export IVF_CENTROID_HNSW_BEAM_WIDTH=64
export IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000
export IVF_TRAIN_SAMPLE_CAP=100000
# Full nprobe scaling curve: doubling steps so the sublinearity is easy to read off.
export KNN_NPROBE="${KNN_NPROBE:-10,20,40,80,160,320}"
export KNN_NQUERY="${KNN_NQUERY:-1000}"
export KNN_SKIP_SMELL=1
export KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1
export LLOYD_SHORTLIST_DEDUP=1
export LLOYD_BRUTE_N=2000
export KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

echo "=== nprobe scaling (SIMD bulk + shortlist dedup, bruteN=2000, golden spill=2) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "    exit=$?"
grep "ivfNprobe = " "$LOG" 2>/dev/null | head -1
grep "hammingKernel" "$LOG" 2>/dev/null | grep -v "cellsScanned=0" | tail -1
echo "SUMMARY lines in nprobe order (recall  lat_ms):"
grep "^SUMMARY" "$LOG" 2>/dev/null | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s  lat=%s ms\n",$1,$2}'
echo "=== done $(date). log: $LOG ==="
