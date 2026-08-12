#!/usr/bin/env bash
# JFR the SEARCH path at the sub-2ms 0.95 operating point: golden margin=1.10 index, nprobe=60, bulk
# dedup (bulk Hamming kernel engaged). Reuses the cached index (KNN_CLEAR_CACHE=0) so the JVM does NO
# build -- the CPU-time profile is dominated by search. High nquery so search accumulates enough samples.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_golden_search_jfr_${STAMP}.log"
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.25
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40
export LLOYD_BRUTE_N=1750
# Many queries so the search phase (not the tiny reader-open) dominates the whole-JVM JFR.
export KNN_NQUERY="${KNN_NQUERY:-10000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
export KNN_JFR=1
echo "=== golden search JFR (nprobe=60, bulk dedup, cached index) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
grep -E "will now reindex|reused|^SUMMARY" "$LOG" | head -3
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}'
echo "=== done $(date). log: $LOG ; jfr in $OUT/logs/*.jfr ==="
