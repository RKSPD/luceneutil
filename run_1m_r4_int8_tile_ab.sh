#!/usr/bin/env bash
# A/B the G=4 INTERLEAVE (R4_TILE) under int8 query, at the 0.95 point (bn750/np40), 2-bit Hamming coarse,
# sp2/m1.20. Same cached baseline index (read-path flags). arm A = R4_TILE=1 (per-doc, current best 1.77ms),
# arm B = R4_TILE=4 (4 independent load+MAC streams -> latency-bound to throughput-bound on the warm DRAM
# reads). Tests whether the interleave moves the warm number. runs=3 medians.
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
export LLOYD_INT8_QUERY=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

# LLOYD_R4_TILE maps to -Dlloyd.r4Tile in knnPerfTest.py.
LOGA="$OUT/run_1m_r4_i8tile_T1_${STAMP}.log"
export LLOYD_R4_TILE=1
echo "=== arm A: int8 R4_TILE=1 (bn750 np40) $(date) ===" | tee "$LOGA"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOGA" 2>&1
echo "T1 exit=$?"

LOGB="$OUT/run_1m_r4_i8tile_T4_${STAMP}.log"
export LLOYD_R4_TILE=4
echo "=== arm B: int8 R4_TILE=4 (bn750 np40) $(date) ===" | tee "$LOGB"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOGB" 2>&1
echo "T4 exit=$?"

echo; echo "=== TILE A/B (int8, bn750 np40) ==="
for L in "$LOGA" "$LOGB"; do
  echo "--- $(basename "$L") ---"
  grep -oE "r4Tile=[0-9]+|int8Query=true" "$L" | sort -u | head -2
  grep "^SUMMARY" "$L" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}' | sort -u
done
echo "=== logs: $LOGA  $LOGB ==="
