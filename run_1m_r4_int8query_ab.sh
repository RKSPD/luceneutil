#!/usr/bin/env bash
# A/B: int8-query rerank vs exact-float rerank, at the 0.95 operating point (bn750, np40), 2-bit Hamming
# coarse, sp2/m1.20. Same cached baseline index (int8Query is a read-path flag). Two arms in one script:
# arm A = float (baseline), arm B = LLOYD_INT8_QUERY=1. Confirms (a) recall is ~unchanged (query int8 error
# only) and (b) the end-to-end latency delta -- expected small since rerank is DRAM-bound, not compute-bound.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=residual4 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=750
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1 LLOYD_SKETCH_LO_CLIP=3.5
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

# arm A: exact float (baseline). Reuse the cached baseline index. First arm builds only if cache is stale.
unset LLOYD_INT8_QUERY
LOGA="$OUT/run_1m_r4_int8ab_FLOAT_${STAMP}.log"
echo "=== arm A: FLOAT query (bn750 np40) $(date) ===" | tee "$LOGA"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOGA" 2>&1
echo "float exit=$?"

# arm B: int8 query. Same index (read-path flag).
export LLOYD_INT8_QUERY=1
LOGB="$OUT/run_1m_r4_int8ab_INT8_${STAMP}.log"
echo "=== arm B: INT8 query (bn750 np40) $(date) ===" | tee "$LOGB"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOGB" 2>&1
echo "int8 exit=$?"

echo; echo "=== A/B RESULTS (bn750 np40) ==="
for L in "$LOGA" "$LOGB"; do
  echo "--- $(basename "$L") ---"
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms sel=%s\n",$1,$2,$4}' | sort -u
done
echo "=== logs: $LOGA  $LOGB ==="
