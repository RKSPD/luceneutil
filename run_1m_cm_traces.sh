#!/usr/bin/env bash
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY=3000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0 KNN_JFR=0
export JAVA_TOOL_OPTIONS="-agentpath:/usr/lib/jvm/amazon-corretto-25.0.4.7.1-linux-aarch64/lib/libasyncProfiler.so=start,event=cache-misses,file=/local/home/rikhil/vectordb/cachemiss_traces_040647.txt,traces=15"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 1 > /local/home/rikhil/vectordb/cm_traces_driver.log 2>&1
echo "TRACES: /local/home/rikhil/vectordb/cachemiss_traces_040647.txt"
