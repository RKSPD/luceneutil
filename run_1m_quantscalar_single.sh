#!/usr/bin/env bash
# Single run: topology-skip ON (default) + SIMD quantize OFF (-Dlloyd.noSimdQuantize=true), nlist=40000.
# Compare its reindex time against the already-measured topology-skip + SIMD-ON build (385.2 s / 0.884) to
# isolate the SIMD QuantizeKernel's build-time delta on the LIVE quant (assign fallback / spill / reader).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
export KNN_NDOC=1000000 KNN_NLIST=40000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.10
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=60 KNN_NQUERY=1000 KNN_SKIP_SMELL=1 KNN_HEAP=24g
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 LLOYD_SKETCH_DIMS=1024
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0 KNN_JFR=0
LOG="$OUT/run_1m_qscalar_${STAMP}.log"
echo "=== [SCALAR quantize, topo-skip on] $(date) ==="
KNN_CLEAR_CACHE=1 JAVA_TOOL_OPTIONS="-Dlloyd.noSimdQuantize=true" ./run_knn_bench.sh 1 > "$LOG" 2>&1
sec=$(grep -iE "reindex takes" "$LOG" | tail -1 | grep -oE "[0-9]+\.[0-9]+ sec" | head -1)
rec=$(grep "^SUMMARY" "$LOG" | head -1 | awk -F'\t' '{print $1}')
echo "    SCALAR reindex=$sec recall=$rec   (vs SIMD 385.2 sec / 0.884)"
echo "=== done $(date) ==="
