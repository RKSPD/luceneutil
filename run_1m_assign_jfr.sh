#!/usr/bin/env bash
# Profile the ASSIGNMENT/build path at nlist=40000 via JFR (per-JVM file, no async-profiler shared-file
# clobber). Forces rebuild. High-fidelity CPU-time sample of index+merge to settle whether graph routing
# (nearest/quantizeInto/signedDot) dominates at high nlist -- the memory says 2% at nlist=2000.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
export KNN_NDOC=1000000 KNN_NLIST=40000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY=50 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
export KNN_JFR=1
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 > /local/home/rikhil/vectordb/assign_jfr_driver.log 2>&1
echo "done"
