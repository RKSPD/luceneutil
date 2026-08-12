#!/usr/bin/env bash
# A/B the FUSED two-plane Hamming coarse scan (bulkDistances2, one pass) vs the two-pass form, at the 2/8
# golden (2-bit coarse + osq8 + bf2, sp2/m1.20, np40, bn500). The fuse is compiled in and auto-engages when
# COARSE_SIGN_W==1 (default) -- so arm A (fused, default) vs arm B (coarseSignW=2 forces the two-pass path,
# which ALSO changes ranking, so B is only a latency-shape reference, not a recall match). Cleaner A/B: arm A
# = default (fused). Reuses cached sp2 index. Must be bit-recall-identical to the pre-fuse 0.948 @ 1.45ms
# baseline -- if recall moved, the fuse is wrong. GOAL: lower latency at identical recall (JFR said the
# two-pass fromMemorySegment was ~24% loading each plane separately).
set -uo pipefail
cd /local/home/rikhil/vectordb/luceneutil
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT=/local/home/rikhil/vectordb
LOG="$OUT/run_1m_2c_fused_${STAMP}.log"

export KNN_NDOC=1000000 KNN_NLIST=2000 KNN_SPILL_BITS=2 IVF_SPILL_MARGIN=1.20
export IVF_QUANTIZER=osq IVF_QUANT_BITS=8 IVF_BEAM_SPILL=1 IVF_SPILL_EF_SEARCH=128
export IVF_CENTROID_HNSW_M=32 IVF_CENTROID_HNSW_BEAM_WIDTH=64 IVF_STREAM_REFINE_ITERS=3
export IVF_STREAM_FLUSH_MIN_DOCS=100000 IVF_TRAIN_SAMPLE_CAP=100000
export KNN_NPROBE=40 LLOYD_BRUTE_N=500 LLOYD_BEAM_FACTOR=2
export KNN_NQUERY="${KNN_NQUERY:-1000}" KNN_SKIP_SMELL=1 KNN_HEAP="${KNN_HEAP:-24g}"
export LLOYD_SKETCH_SCAN=1 LLOYD_SHORTLIST_DEDUP=1 KNN_DROP_CACHE_AFTER_WARMUP=0
export LLOYD_SKETCH_DIMS=1024 LLOYD_SKETCH_LO_BITS=1
export LLOYD_URING_SKETCH_SCAN=0 LLOYD_URING_RERANK=0 LLOYD_URING_PIPELINE=0 LLOYD_URING_RERANK_PIPELINE=0 LLOYD_PREFETCH_CELLS=0

echo "=== 2/8 FUSED two-plane coarse (default COARSE_SIGN_W=1 -> auto-fuse) $(date) ===" | tee "$LOG"
KNN_CLEAR_CACHE=0 ./run_knn_bench.sh 3 >> "$LOG" 2>&1
echo "exit=$?"
echo "=== recall/lat (vs pre-fuse baseline 0.948 @ 1.44-1.45ms) ==="
grep "^SUMMARY" "$LOG" | sed 's/SUMMARY: //' | awk -F'\t' '{printf "  recall=%s lat=%s ms\n",$1,$2}' | sort -u
echo "=== proof the fused kernel engaged (fused2Rows / hammingKernel) ==="
grep -iE "fused2|hammingKernel|bulkDist" "$LOG" | grep -iv cmd: | head -3
echo "=== log: $LOG ==="
