#!/usr/bin/env bash
# DIAGNOSTIC: is the bn<500 recall ceiling ROUTING-bound or COARSE-RANK-bound? Hold the rerank set FIXED at
# bn500 (osq8, 2-bit coarse, fixed shared threshold) and sweep nprobe {40,55,70,90}. If recall climbs toward
# 0.95 as nprobe rises, true neighbors were being dropped at ROUTING (not in probed cells) -> the bn<500
# lever is nprobe/spill. If recall is FLAT in nprobe, the neighbors are in the cells but rank >500 in the
# coarse order -> COARSE ranking is the ceiling and bn<500 needs a sharper sketch. Reuses the cached osq8
# 2-coarse index (nprobe is search-time, not in the key). runs=3.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_2c_npbn500_${STAMP}.log"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40,55,70,90
export LLOYD_BRUTE_N=500
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== np sweep @ fixed bn500 (osq8 2-coarse, routing-vs-coarse gate) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOG" 2>&1
echo "exit=$?"

echo; echo "=== recall/lat vs nprobe (bn fixed 500) ==="
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms visited=%s sel=%s\n",$1,$2,$16,$4}' | sort -u
echo "=== log: $LOG ==="
