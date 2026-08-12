#!/usr/bin/env bash
# Current build (dot4 + postingStart-stash) at the operating point, cached m1.25 index. Compare to dot4-only
# 1.034 ms to isolate the postingStart win. 3 passes MIN.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.25
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g LLOYD_BRUTE_N=1750
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 > "$OUT/run_1m_curop_${STAMP}.log" 2>&1
grep "^SUMMARY" "$OUT/run_1m_curop_${STAMP}.log" | sed 's/SUMMARY: //' | awk -F'\t' '{r=$1; if(min==""||$2<min)min=$2} END{printf "  CURRENT(dot4+postingStash) recall=%s MIN lat=%s ms  (dot4-only was 1.034)\n",r,min}'
