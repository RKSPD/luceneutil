#!/usr/bin/env bash
#
# JFR profile of the INDEX path at 1M (osq int8 golden config). KNN_JFR=1 records the whole JVM
# (jdk.CPUTimeSample, dumponexit); with the cache cleared the JVM does a fresh ~150s reindex that dwarfs
# the search phase, so the CPU-time profile is overwhelmingly the writer/merge path. nquery is cut low to
# minimise the search contribution further.
#
# NOTE: this CLEARS the cached 1M osq index (a fresh build is the point); it repopulates under the same key.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_jfr_index_nl100k_${STAMP}.log"

export KNN_NDOC=1000000
export KNN_NLIST=100000
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
export KNN_NPROBE=40
# Minimise the search phase so the whole-JVM JFR is dominated by the reindex.
export KNN_NQUERY=50
export KNN_SKIP_SMELL=1
export KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1
export KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_URING_SKETCH_SCAN=0
export LLOYD_URING_RERANK=0
export LLOYD_URING_PIPELINE=0
export LLOYD_URING_RERANK_PIPELINE=0
export LLOYD_PREFETCH_CELLS=0

# JFR on; cache cleared to force a fresh build (the thing being profiled).
export KNN_JFR=1
echo "=== JFR index profile (1M osq, fresh build) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "    exit=$?"
grep -E "reindex takes|will now reindex|PROFILE SUMMARY" "$LOG" 2>/dev/null | head
echo "=== done $(date). log: $LOG ; jfr in $OUT/logs/*.jfr ==="
