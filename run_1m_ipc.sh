#!/usr/bin/env bash
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
AP=/usr/lib/jvm/amazon-corretto-25.0.4.7.1-linux-aarch64/lib/libasyncProfiler.so
EV="${AP_EVENT:-cycles}"
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 KNN_NQUERY=2000 KNN_SKIP_SMELL=1 KNN_HEAP=24g LLOYD_BRUTE_N=1750
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0 KNN_JFR=0
# %p -> per-JVM file so the search JVM's dump is NOT clobbered by helper JVMs sharing JAVA_TOOL_OPTIONS.
export JAVA_TOOL_OPTIONS="-agentpath:${AP}=start,event=${EV},file=${OUT}/ipc_${EV}_p%p_${STAMP}.txt,flat=40"
echo "=== IPC probe event=$EV nq=2000 $(date) ==="
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 > "$OUT/ipc_${EV}_driver_${STAMP}.log" 2>&1
echo "done; files: $OUT/ipc_${EV}_p*_${STAMP}.txt"
