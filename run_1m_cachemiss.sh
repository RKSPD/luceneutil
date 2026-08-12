#!/usr/bin/env bash
# CACHE-MISS profile of the search path via async-profiler PMU (event=cache-misses), instead of the JFR CPU
# sampler. Attributes L2/L3 misses to Java frames -- shows WHERE the memory stalls are (the coarse scan's
# sketch loads were 50.8% of kernel CPU; this says whether that is cache-miss-bound). Reuses the cached 1M
# index (currently sketchDims=1024). counting-select + histogram rewrite are committed, so this profiles HEAD.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
AP=/usr/lib/jvm/amazon-corretto-25.0.4.7.1-linux-aarch64/lib/libasyncProfiler.so
EVENT="${AP_EVENT:-cache-misses}"
APOUT="$OUT/cachemiss_${EVENT//[^a-zA-Z0-9]/_}_${STAMP}.txt"
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY="${KNN_NQUERY:-10000}" KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
export KNN_JFR=0
# async-profiler: only the long-lived search JVM accumulates meaningful samples; short helper JVMs write and
# overwrite the same file harmlessly. flat=40 -> top 40 frames; also dump a collapsed stack for tree analysis.
export JAVA_TOOL_OPTIONS="-agentpath:${AP}=start,event=${EVENT},file=${APOUT},flat=60"
LOG="$OUT/run_1m_cachemiss_${STAMP}.log"
echo "=== cache-miss profile event=$EVENT nquery=$KNN_NQUERY $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "=== done $(date). ap out: $APOUT ; log: $LOG ===" | tee -a "$LOG"
grep -E "^SUMMARY" "$LOG" | tail -1
