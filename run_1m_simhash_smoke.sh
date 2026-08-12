#!/usr/bin/env bash
# SMOKE TEST the anchor-SimHash end-to-end path at small scale before a full 1M build: encode (B sign-bits) ->
# 2-bit Gray Hamming coarse shortlist -> SimHash Hamming rerank. 200k docs (fast build), np40, bn750, B=4096
# (512 B/doc, half osq8). Confirms the WIRING works + sane recall; NOT a perf number. Compare to a 2/8 osq8
# run at the same scale to see if SimHash rerank is in the same ballpark.
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_simhash_smoke_${STAMP}.log"

export KNN_NDOC="${KNN_NDOC:-200000}" KNN_NLIST=1000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=simhash IVF_SIMHASH_BITS=4096
export IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=750
export KNN_NQUERY="${KNN_NQUERY:-500}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0 KNN_SKIP_MODEL_CHECK=1
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== simhash SMOKE (ndoc=$KNN_NDOC, B=4096, np40 bn750) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=1 ./run_knn_bench.sh 1 >> "$LOG" 2>&1
echo "exit=$?"
echo "=== engaged? (quantizer=simhash in cmd) + recall/lat ==="
grep -oE "quantizer=simhash|simhashBits=[0-9]+|qzsimhash" "$LOG" | sort -u | head
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}'
echo "=== log: $LOG ==="
