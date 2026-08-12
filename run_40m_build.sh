#!/usr/bin/env bash
# Build + SAVE the 40M index for the genuine cold >RAM read-path test (1M was too small to force off-cache
# without OOM). Config matches the 1M golden: osq int8, spill=2, margin=1.10; nlist=40000 (~1000 docs/cell
# at 40M). Max indexing threads (64-core box). KNN_CLEAR_CACHE=1 builds fresh; it persists under knn-reuse.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_40m_build_${STAMP}.log"
export KNN_NDOC=39767748 KNN_NLIST=40000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60
# Tiny query set: we only want the index built + persisted here; the cold sweep comes after.
export KNN_NQUERY=50 KNN_SKIP_SMELL=1
export KNN_HEAP=48g
export KNN_INDEX_THREADS=64
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
echo "=== 40M build (nlist=40000 sp2 osq, 64 index threads) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep -E "reindex takes|force merge|index disk usage|^SUMMARY" "$LOG" | sed 's/^/  /'
echo "=== done $(date). log: $LOG ==="
