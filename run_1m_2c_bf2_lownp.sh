#!/usr/bin/env bash
# Lower nprobe with beamFactor=2: bf2 makes each probed cell more accurate, so fewer cells may hold 0.948
# while scanning FEWER docs -- directly cutting the ~28% coarse-plane pole (the path to 0.8ms per the JFR).
# 2/8 golden (fused coarse + osq8 + bf2), sp2/m1.20, bn500. Sweep np {20,25,30,35,40}. Reuses cached sp2
# index (search-time). Baseline: np40=0.948@1.16ms. runs=3.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_2c_bf2lownp_${STAMP}.log"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=20,25,30,35,40 LLOYD_BRUTE_N=500 LLOYD_BEAM_FACTOR=2
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== 2/8 bf2 low-nprobe @ bn500 (fused coarse) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOG" 2>&1
echo "exit=$?"
echo "=== recall/lat vs nprobe (bn500, bf2) -- baseline np40=0.948/1.16 ==="
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms visited=%s\n",$1,$2,$16}' | sort -u
echo "=== log: $LOG ==="
